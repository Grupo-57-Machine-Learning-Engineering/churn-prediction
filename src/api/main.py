"""API de inferência de churn (Etapa 3).

Os dois endpoints do Contrato 3 (`docs/decisions.md`): o `GET /health` é um
liveness simples, sem dependência externa, que responde `{"status": "ok"}`
mesmo sem modelo carregado, porque estar de pé e conseguir prever são
condições diferentes. O `POST /predict` recebe um cliente no formato
pós-ETL (`ChurnRequest`) e devolve `ChurnResponse` com classe,
probabilidade e versão.

Tratamento de erro: payload inválido (tipo ou domínio errado, campo
faltando, campo extra como `services_offer`) vira 422, gerado pelo próprio
Pydantic/FastAPI. Modelo indisponível (sem `models/champion_model.joblib`
e sem MLflow configurado) vira 503 com instrução de como obter o artefato.
Falha inesperada durante a predição vira 500 com mensagem limpa, e o
detalhe fica só no log.

O modelo é carregado uma única vez no startup (lifespan) e reutilizado em
todas as requisições. Carregar um RandomForest do disco a cada request
inviabilizaria a latência.

Rodar localmente: `uv run uvicorn src.api.main:app --reload` (ver README).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src import __version__
from src.api.schemas import ChurnRequest, ChurnResponse
from src.logger import get_logger
from src.models.predict import ModeloIndisponivelError, carregar_campeao, prever

logger = get_logger(__name__)


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
    description="Predição de churn (Telco/IBM). Grupo 57, Pós Tech ML Engineering (FIAP).",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: confirma apenas que a API está no ar (Contrato 3)."""
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(payload: ChurnRequest, request: Request) -> ChurnResponse:
    """Propensão de churn de um cliente (probabilidade + classe)."""
    modelo = request.app.state.modelo
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo indisponível. Gere models/champion_model.joblib executando "
                "notebooks/05_modelagem.ipynb, ou configure o MLflow/DagsHub no .env "
                "(ver .env.example), e reinicie a API."
            ),
        )

    dados = pd.DataFrame([payload.model_dump()])
    try:
        resultado = prever(modelo, dados)[0]
    except Exception as erro:
        logger.exception("Falha inesperada ao prever: %s", erro)
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao gerar a predição. Consulte os logs da API.",
        ) from erro

    return ChurnResponse(
        churn=resultado["churn"],
        probability=resultado["probability"],
        model_version=__version__,
    )
