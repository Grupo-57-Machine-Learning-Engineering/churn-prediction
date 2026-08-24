"""Carregamento e uso do modelo campeão (Etapa 3).

O modelo pode vir de duas fontes. A prioridade é o Model Registry do
MLflow/DagsHub, quando o tracking está configurado no `.env` (ver ADR-004 e
o `.env.example`): é a fonte de verdade de qual versão está marcada como
`@champion`, e é o que garante que reiniciar a API depois de promover um
campeão novo já serve o modelo certo, sem precisar sincronizar nenhum
arquivo manualmente. O artefato local `models/champion_model.joblib`, que o
notebook 05 grava sempre, é o fallback: cobre o caso de rodar sem rede ou
sem credencial (ex: dev local sem `.env`), mas pode ficar desatualizado em
relação ao registry, então só é usado quando o MLflow não está disponível.

O artefato é um Pipeline completo do sklearn (EngenhariaEstrutural,
DescartadorDeColunas, ColumnTransformer e o estimador no fim), já fitado no
treino. Por isso a entrada da predição são as colunas pós-ETL do Contrato 3
em `docs/decisions.md`: não tem pré-processamento manual antes de chamar
`prever`, o próprio pipeline imputa, escala e codifica.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import joblib
import pandas as pd

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

__all__ = ["Campeao", "ModeloIndisponivelError", "carregar_campeao", "prever"]

ORIGEM_LOCAL = "joblib-local"
"""Origem reportada quando o modelo veio do artefato em disco.

Sem número de versão de propósito: o `.joblib` não carrega essa informação,
e inventar uma (data do arquivo, hash) daria falsa sensação de rastreio.
"""


class Campeao(NamedTuple):
    """Modelo carregado junto com a origem de onde ele veio.

    A origem vai para o `model_source` da resposta (Contrato 3). Ela existe
    porque, com o registry na frente do joblib, o modelo que responde pode
    mudar sem o pacote mudar de versão: sem esse campo a resposta não diria
    qual dos dois respondeu (ADR-006).
    """

    modelo: Any
    origem: str


class ModeloIndisponivelError(RuntimeError):
    """Nenhuma fonte conseguiu fornecer o modelo campeão."""


def carregar_campeao(caminho: Path | str | None = None) -> Campeao:
    """Carrega o pipeline campeão: MLflow primeiro, joblib local como fallback.

    Parameters
    ----------
    caminho
        Caminho alternativo para o artefato `.joblib`, usado só se o MLflow
        não estiver disponível. `None` usa o padrão do projeto
        (`config.CHAMPION_MODEL_PATH`).

    Returns
    -------
    Campeao
        O modelo e a origem de onde ele veio.

    Raises
    ------
    ModeloIndisponivelError
        Se o MLflow não está configurado (ou falhou) e o arquivo local
        também não existe.
    """
    campeao = _carregar_do_mlflow()
    if campeao is not None:
        return campeao

    caminho = Path(caminho) if caminho is not None else config.CHAMPION_MODEL_PATH

    if caminho.exists():
        logger.info("Carregando modelo campeão do artefato local: %s", caminho)
        return Campeao(joblib.load(caminho), ORIGEM_LOCAL)

    raise ModeloIndisponivelError(
        "Modelo campeão não encontrado no MLflow (tracking não configurado ou "
        f"indisponível) nem em '{caminho}'. Gere o artefato executando o notebook "
        "notebooks/05_modelagem.ipynb, ou configure o tracking no .env "
        "(ver .env.example) para baixar do Model Registry do DagsHub."
    )


def _carregar_do_mlflow() -> Campeao | None:
    """Tenta baixar o campeão do Model Registry. Devolve `None` se não der.

    É best-effort de propósito: a API precisa subir e responder o `/health`
    mesmo sem DagsHub configurado. Quando o modelo falta, quem lida com isso
    é o `/predict`, respondendo 503 (ver `src/api/main.py`).

    O tempo limite das chamadas HTTP vem do `MLFLOW_HTTP_REQUEST_TIMEOUT`
    (ver `.env.example`). Sem ele, um DagsHub lento não levanta exceção nem
    cai no fallback, ele só não termina, e segura o startup da API.
    """
    tem_token = all(
        (
            config.MLFLOW_TRACKING_URI,
            config.MLFLOW_TRACKING_USERNAME,
            config.MLFLOW_TRACKING_PASSWORD,
        )
    )
    tem_dagshub = all((config.DAGSHUB_REPO_OWNER, config.DAGSHUB_REPO_NAME))
    if not (tem_token or tem_dagshub):
        return None

    try:
        import mlflow.sklearn

        config.configurar_mlflow_tracking()
        logger.info("Baixando modelo campeão do registry: %s", config.CHAMPION_MODEL_URI)
        modelo = mlflow.sklearn.load_model(config.CHAMPION_MODEL_URI)
        return Campeao(modelo, f"mlflow:{config.CHAMPION_MODEL_NAME}/{_versao_no_registry()}")
    except Exception as erro:  # pragma: no cover - depende de rede/credencial
        logger.warning("Carregamento via MLflow falhou, tentando o joblib local: %s", erro)
        return None


def _versao_no_registry() -> str:
    """Número da versão que está com o alias `@champion` agora.

    Best-effort e chamada só depois do modelo já ter carregado: se resolver o
    alias falhar, o modelo continua servindo e a origem sai com a versão
    marcada como desconhecida, em vez de derrubar o carregamento por causa de
    um rótulo.
    """
    try:
        from mlflow import MlflowClient

        versao = MlflowClient().get_model_version_by_alias(
            config.CHAMPION_MODEL_NAME, config.CHAMPION_MODEL_ALIAS
        )
        return str(versao.version)
    except Exception as erro:  # pragma: no cover - depende de rede/credencial
        logger.warning("Não consegui resolver a versão do alias @champion: %s", erro)
        return "desconhecida"


def prever(
    modelo,
    dados: pd.DataFrame,
    threshold: float = config.THRESHOLD_DECISAO,
) -> list[dict]:
    """Calcula a propensão de churn para cada linha de `dados`.

    Parameters
    ----------
    modelo
        Pipeline campeão (saída de `carregar_campeao`).
    dados
        DataFrame com as colunas pós-ETL do Contrato 3, uma linha por
        cliente. O pipeline interno cuida de imputação, escala e one-hot.
    threshold
        Corte para converter probabilidade em classe (padrão
        `config.THRESHOLD_DECISAO`).

    Returns
    -------
    list[dict]
        Um dict por linha: `{"probability": float, "churn": bool}`.
    """
    probabilidades = modelo.predict_proba(dados)[:, 1]
    return [{"probability": float(p), "churn": bool(p >= threshold)} for p in probabilidades]
