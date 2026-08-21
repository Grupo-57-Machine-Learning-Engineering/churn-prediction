"""Carregamento e uso do modelo campeão (Etapa 3).

O modelo pode vir de duas fontes. A primeira é o artefato local
`models/champion_model.joblib`, que o notebook 05 grava sempre, com ou sem
MLflow no ar. É o caminho padrão porque funciona sem rede e sem credencial.
A segunda é o Model Registry do MLflow/DagsHub, usado como fallback quando
o arquivo local não existe e o tracking está configurado no `.env` (ver
ADR-004 e o `.env.example`).

O artefato é um Pipeline completo do sklearn (EngenhariaEstrutural,
DescartadorDeColunas, ColumnTransformer e o estimador no fim), já fitado no
treino. Por isso a entrada da predição são as colunas pós-ETL do Contrato 3
em `docs/decisions.md`: não tem pré-processamento manual antes de chamar
`prever`, o próprio pipeline imputa, escala e codifica.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

__all__ = ["ModeloIndisponivelError", "carregar_campeao", "prever"]


class ModeloIndisponivelError(RuntimeError):
    """Nenhuma fonte conseguiu fornecer o modelo campeão."""


def carregar_campeao(caminho: Path | str | None = None):
    """Carrega o pipeline campeão: joblib local primeiro, MLflow como fallback.

    Parameters
    ----------
    caminho
        Caminho alternativo para o artefato `.joblib`. `None` usa o padrão
        do projeto (`config.CHAMPION_MODEL_PATH`).

    Raises
    ------
    ModeloIndisponivelError
        Se o arquivo local não existe e o fallback via MLflow não está
        configurado ou falhou.
    """
    caminho = Path(caminho) if caminho is not None else config.CHAMPION_MODEL_PATH

    if caminho.exists():
        logger.info("Carregando modelo campeão do artefato local: %s", caminho)
        return joblib.load(caminho)

    modelo = _carregar_do_mlflow()
    if modelo is not None:
        return modelo

    raise ModeloIndisponivelError(
        f"Modelo campeão não encontrado em '{caminho}' e o fallback via MLflow "
        "não está disponível. Gere o artefato executando o notebook "
        "notebooks/05_modelagem.ipynb, ou configure o tracking no .env "
        "(ver .env.example) para baixar do Model Registry do DagsHub."
    )


def _carregar_do_mlflow():
    """Tenta baixar o campeão do Model Registry. Devolve `None` se não der.

    É best-effort de propósito: a API precisa subir e responder o `/health`
    mesmo sem DagsHub configurado. Quando o modelo falta, quem lida com isso
    é o `/predict`, respondendo 503 (ver `src/api/main.py`).
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
        return mlflow.sklearn.load_model(config.CHAMPION_MODEL_URI)
    except Exception as erro:  # pragma: no cover - depende de rede/credencial
        logger.warning("Fallback via MLflow falhou: %s", erro)
        return None


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
