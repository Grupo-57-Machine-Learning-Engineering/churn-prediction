"""Baseline da Etapa 1: Regressão Logística sem tuning.

Fluxo separado do campeão de propósito. O baseline é a referência contra a
qual a Etapa 2 se compara, então ele não pode ser retunado nem substituído
quando um candidato novo vence: as métricas dele têm que continuar sendo as
mesmas que foram medidas lá atrás.

Duas diferenças concretas em relação a `champion.registrar_campeao`:

* não entra no Model Registry nem move o alias `@champion`, porque baseline
  não é modelo servido;
* se o MLflow falhar, grava `models/baseline_logistic_regression.joblib` como
  fallback, para a Etapa 2 conseguir recuperar o artefato depois (ver
  `champion.resolver_pipeline_campeao`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.logger import get_logger
from src.training.comparison import NOME_RUN_BASELINE

logger = get_logger(__name__)

FONTE_PADRAO = "notebooks/04_baseline.ipynb"
"""Valor de `mlflow.source.name` das runs (ver `config.iniciar_run`)."""

__all__ = ["FONTE_PADRAO", "criar_baseline", "params_do_baseline", "registrar_baseline"]


def criar_baseline(**overrides) -> LogisticRegression:
    """Regressão Logística do notebook 04, em configuração default.

    Não reaproveita `estimators.criar_logistic_regression` porque aquela
    fábrica força `solver="liblinear"`, necessário para o grid alternar entre
    penalidade `l1` e `l2`. O baseline usa o solver padrão do sklearn
    (`lbfgs`), e trocar isso mudaria o número de referência da Etapa 1.
    """
    params: dict[str, Any] = dict(max_iter=1000, class_weight="balanced", random_state=config.SEED)
    params.update(overrides)
    return LogisticRegression(**params)


def params_do_baseline(modelo: LogisticRegression) -> dict[str, Any]:
    """Params que descrevem o baseline na run do MLflow."""
    return {
        "modelo": type(modelo).__name__,
        "class_weight": modelo.class_weight,
        "random_state": modelo.random_state,
    }


def registrar_baseline(
    pipeline: Pipeline,
    metricas_cv: dict[str, float],
    metricas_teste: dict[str, float],
    *,
    params: dict[str, Any] | None = None,
    nome: str = NOME_RUN_BASELINE,
    fonte: str = FONTE_PADRAO,
    caminho_fallback: Path | str | None = None,
) -> Path | None:
    """Registra a run do baseline no MLflow, com joblib local como fallback.

    Parameters
    ----------
    metricas_cv
        Saída de `metrics.avaliar_por_cv`, já nas chaves `cv_*` que
        `comparison.buscar_baseline` espera encontrar de volta.
    metricas_teste
        Saída de `metrics.calcular_metricas` no conjunto de teste. Vai para o
        MLflow com o prefixo `teste_`.
    params
        Params extras da run. `None` usa `params_do_baseline` sobre o último
        passo do pipeline.

    Returns
    -------
    Path | None
        Caminho do joblib quando o fallback foi acionado, `None` quando a run
        subiu normalmente.
    """
    if params is None:
        params = params_do_baseline(pipeline.steps[-1][1])

    try:
        import mlflow
        import mlflow.sklearn

        config.configurar_mlflow_tracking()
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        config.limpar_runs_anteriores([nome])

        with config.iniciar_run(fonte, run_name=nome):
            mlflow.log_params(params)
            mlflow.log_metrics(metricas_cv)
            mlflow.log_metrics({f"teste_{k}": v for k, v in metricas_teste.items()})
            mlflow.sklearn.log_model(pipeline, name="modelo", serialization_format="cloudpickle")

        logger.info("Run '%s' registrada no MLflow, com o modelo como artefato.", nome)
        return None
    except Exception as erro:
        logger.warning("Não foi possível registrar o baseline no MLflow: %s", erro)
        return _salvar_fallback(pipeline, nome, caminho_fallback)


def _salvar_fallback(pipeline: Pipeline, nome: str, caminho: Path | str | None) -> Path | None:
    """Grava o baseline em disco quando o MLflow não está disponível."""
    caminho = Path(caminho) if caminho is not None else config.MODELS_DIR / f"{nome}.joblib"
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, caminho)
        logger.info("Baseline salvo localmente em: %s", caminho)
        return caminho
    except Exception as erro:  # pragma: no cover - falha de disco
        logger.error("Também não foi possível salvar o baseline localmente: %s", erro)
        return None
