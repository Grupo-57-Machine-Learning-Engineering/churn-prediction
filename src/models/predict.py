"""Carregamento e uso do modelo campeão para inferência.

Espelha o que a célula 31 do `notebooks/05_modelagem.ipynb` produz. Lá o
campeão sai por duas portas: `joblib.dump` em `models/champion_model.joblib`
e `mlflow.sklearn.log_model` com `registered_model_name="churn_champion"`,
seguido de `set_registered_model_alias(..., "champion", versao)`. Este módulo
lê pelas mesmas duas portas, nessa ordem.

O artefato é o Pipeline inteiro (`EngenhariaEstrutural`,
`DescartadorDeColunas`, `ColumnTransformer` e o estimador), já fitado. Então
a entrada de `prever` são as colunas pós-ETL, sem nenhum pré-processamento
manual antes: imputação, escala e one-hot acontecem dentro do pipeline. É a
mesma chamada que o notebook faz no teste, `pipeline.predict_proba(X)[:, 1]`.

O `DescartadorDeColunas` ignora coluna ausente de propósito, então o mesmo
pipeline aceita o parquet completo do treino e um payload reduzido, sem as
colunas `status_*` que não existem no momento da predição.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

NOME_ARQUIVO_CAMPEAO = "champion_model.joblib"
"""Artefato local gravado pelo notebook 05. Não é versionado no git."""

NOME_REGISTRO = "churn_champion"
ALIAS_CAMPEAO = "champion"

URI_CAMPEAO = f"models:/{NOME_REGISTRO}@{ALIAS_CAMPEAO}"
"""Endereço do campeão no Model Registry, o mesmo que o notebook 05 alimenta."""

THRESHOLD_PADRAO = 0.5
"""Corte de probabilidade para converter em classe.

0,5 é o threshold implícito do notebook, que avalia o campeão com
`pipeline.predict(X_test)`. Num classificador binário do sklearn, `predict`
é o argmax das duas probabilidades, ou seja, exatamente `proba >= 0.5`. As
métricas de negócio do ADR-004 foram medidas nesse corte, então mudar aqui
invalida a comparação com elas.
"""

__all__ = [
    "ALIAS_CAMPEAO",
    "NOME_ARQUIVO_CAMPEAO",
    "NOME_REGISTRO",
    "THRESHOLD_PADRAO",
    "URI_CAMPEAO",
    "ModeloIndisponivelError",
    "caminho_padrao_do_campeao",
    "carregar_campeao",
    "prever",
    "prever_um",
]


class ModeloIndisponivelError(RuntimeError):
    """Nenhuma das duas fontes conseguiu fornecer o campeão."""


def caminho_padrao_do_campeao() -> Path:
    """Caminho do joblib local, resolvido a partir de `config.MODELS_DIR`."""
    return config.MODELS_DIR / NOME_ARQUIVO_CAMPEAO


def carregar_campeao(caminho: Path | str | None = None) -> Pipeline:
    """Carrega o pipeline campeão: joblib local primeiro, Registry depois.

    A ordem não é arbitrária. O joblib é o entregável explícito do enunciado,
    funciona sem rede e sem credencial, e o notebook 05 o grava sempre, mesmo
    quando o MLflow está fora. O Registry é conveniência para quem não rodou
    o notebook.

    Parameters
    ----------
    caminho
        Artefato alternativo. `None` usa `caminho_padrao_do_campeao()`.

    Raises
    ------
    ModeloIndisponivelError
        Quando o arquivo não existe e o fallback via MLflow não está
        configurado ou falhou.
    """
    caminho = Path(caminho) if caminho is not None else caminho_padrao_do_campeao()

    if caminho.exists():
        logger.info("Carregando o campeão do artefato local: %s", caminho)
        return joblib.load(caminho)

    modelo = _carregar_do_registry()
    if modelo is not None:
        return modelo

    raise ModeloIndisponivelError(
        f"Campeão não encontrado em '{caminho}' e o fallback via MLflow não está "
        "disponível. Gere o artefato rodando notebooks/05_modelagem.ipynb, ou "
        "configure o tracking no .env (ver .env.example) para baixar do Model "
        "Registry do DagsHub."
    )


def _carregar_do_registry() -> Pipeline | None:
    """Baixa o campeão do Model Registry. Devolve `None` se não der.

    Best-effort de propósito: quem chama decide o que fazer sem modelo. Uma
    API, por exemplo, precisa subir e responder o health check mesmo assim.
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
        logger.warning("Tracking do MLflow não configurado: fallback indisponível.")
        return None

    try:
        import mlflow.sklearn

        config.configurar_mlflow_tracking()
        logger.info("Baixando o campeão do Model Registry: %s", URI_CAMPEAO)
        return mlflow.sklearn.load_model(URI_CAMPEAO)
    except Exception as erro:  # pragma: no cover - depende de rede/credencial
        logger.warning("Fallback via MLflow falhou: %s", erro)
        return None


def prever(
    modelo: Pipeline,
    dados: pd.DataFrame,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Propensão a churn de cada linha de `dados`.

    Parameters
    ----------
    modelo
        Pipeline campeão, saída de `carregar_campeao`.
    dados
        DataFrame com as colunas pós-ETL, uma linha por cliente. O pipeline
        cuida da preparação.
    threshold
        Corte para converter probabilidade em classe. `None` usa
        `THRESHOLD_PADRAO`.

    Returns
    -------
    list[dict]
        Um dict por linha, com `probability` (float) e `churn` (bool), na
        mesma ordem das linhas de entrada.

    Raises
    ------
    ValueError
        Se `dados` vier vazio. Um DataFrame sem linhas quase sempre é erro de
        montagem do payload, e devolver lista vazia em silêncio esconde isso.
    """
    if dados.empty:
        raise ValueError("DataFrame de entrada está vazio: nada a prever.")

    threshold = THRESHOLD_PADRAO if threshold is None else threshold
    probabilidades = modelo.predict_proba(dados)[:, 1]
    return [{"probability": float(p), "churn": bool(p >= threshold)} for p in probabilidades]


def prever_um(
    modelo: Pipeline,
    cliente: dict[str, Any],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Atalho de uma linha, para quem recebe um cliente como dict."""
    return prever(modelo, pd.DataFrame([cliente]), threshold)[0]
