"""API de inferência de churn (Etapa 3).

Os endpoints do Contrato 3 (`docs/decisions.md`): o `GET /health` é um
liveness simples, sem dependência externa, que responde `{"status": "ok"}`
mesmo sem modelo carregado, porque estar de pé e conseguir prever são
condições diferentes. O `POST /predict` recebe um cliente no formato
pós-ETL (`ChurnRequest`) e devolve `ChurnResponse` com classe,
probabilidade e versão. O `GET /sample` devolve uma amostra de clientes de
exemplo já pontuados, para quem quiser ver a API funcionando sem montar
payload na mão.

Os dois caminhos de predição passam pelo mesmo `ChurnRequest.to_dataframe`
e pela mesma função `prever`, então o resultado que aparece no `/sample` é
o mesmo que sai ao mandar aquele payload no `/predict`. Existe teste
garantindo isso.

Tratamento de erro: payload inválido (tipo errado, campo faltando, número
fora de faixa, campo extra como `services_offer`) vira 422, gerado pelo
próprio Pydantic/FastAPI. Modelo indisponível (sem
`models/champion_model.joblib` e sem MLflow configurado) vira 503 com
instrução de como obter o artefato. Falha inesperada durante a predição
vira 500 com mensagem limpa, e o detalhe fica só no log.

O modelo é carregado uma única vez no startup (lifespan) e reutilizado em
todas as requisições. Carregar um RandomForest do disco a cada request
inviabilizaria a latência.

Rodar localmente: `uv run uvicorn src.api.main:app --reload` (ver README).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src import __version__
from src.api.samples import CLIENTES_EXEMPLO
from src.api.schemas import ChurnRequest, ChurnResponse, ClienteExemplo, SampleResponse
from src.config import THRESHOLD_DECISAO
from src.logger import get_logger
from src.models.predict import ModeloIndisponivelError, carregar_campeao, prever

logger = get_logger(__name__)

_DETALHE_SEM_MODELO = (
    "Modelo indisponível. Gere models/champion_model.joblib executando "
    "notebooks/05_modelagem.ipynb, ou configure o MLflow/DagsHub no .env "
    "(ver .env.example), e reinicie a API."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o campeão no startup; a ausência dele não derruba a API."""
    try:
        app.state.modelo = carregar_campeao()
    except ModeloIndisponivelError as erro:
        logger.warning("API subiu sem modelo: %s", erro)
        app.state.modelo = None
    yield


app = FastAPI(
    title="churn-prediction API",
    description=(
        "Predição de churn (Telco/IBM). Grupo 57, Pós Tech ML Engineering (FIAP).\n\n"
        "Comece pelo `GET /sample`: ele devolve clientes de exemplo já pontuados, "
        "e o campo `payload` de cada um pode ser colado direto no `POST /predict` "
        "para reproduzir o mesmo resultado."
    ),
    version=__version__,
    lifespan=lifespan,
)


def _modelo_ou_503(request: Request):
    modelo = request.app.state.modelo
    if modelo is None:
        raise HTTPException(status_code=503, detail=_DETALHE_SEM_MODELO)
    return modelo


def _prever_ou_500(modelo, dados) -> dict:
    try:
        return prever(modelo, dados)[0]
    except Exception as erro:
        logger.exception("Falha inesperada ao prever: %s", erro)
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao gerar a predição. Consulte os logs da API.",
        ) from erro


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: confirma apenas que a API está no ar (Contrato 3)."""
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(payload: ChurnRequest, request: Request) -> ChurnResponse:
    """Propensão de churn de um cliente (probabilidade + classe).

    Todos os campos são opcionais. O pipeline imputa o que faltar, então
    ficha incompleta continua sendo pontuada, com a ressalva de que quanto
    menos informação chega, mais a predição se apoia no perfil mediano do
    treino (ADR-006).
    """
    modelo = _modelo_ou_503(request)
    resultado = _prever_ou_500(modelo, payload.to_dataframe())

    return ChurnResponse(
        churn=resultado["churn"],
        probability=resultado["probability"],
        model_version=__version__,
    )


@app.get("/sample", response_model=SampleResponse)
def sample(request: Request) -> SampleResponse:
    """Clientes de exemplo já pontuados, para ver a API funcionando.

    Cada item traz o `payload` completo e o `resultado` do modelo. Mandar
    aquele payload no `POST /predict` devolve exatamente os mesmos números,
    porque os dois caminhos usam a mesma conversão e a mesma função de
    predição.

    Os perfis cobrem os dois extremos de risco e as decisões de validação
    do ADR-006: cliente sem internet, ficha incompleta e categoria que o
    modelo nunca viu.
    """
    modelo = _modelo_ou_503(request)

    clientes = []
    for exemplo in CLIENTES_EXEMPLO:
        payload = ChurnRequest(**exemplo["payload"])
        resultado = _prever_ou_500(modelo, payload.to_dataframe())
        clientes.append(
            ClienteExemplo(
                nome=exemplo["nome"],
                descricao=exemplo["descricao"],
                payload=payload,
                resultado=ChurnResponse(
                    churn=resultado["churn"],
                    probability=resultado["probability"],
                    model_version=__version__,
                ),
            )
        )

    return SampleResponse(
        total=len(clientes),
        threshold=THRESHOLD_DECISAO,
        model_version=__version__,
        clientes=clientes,
    )
