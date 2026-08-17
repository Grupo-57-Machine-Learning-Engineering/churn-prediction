"""Baseline — Regressão Logística (docs/decisions.md, Contrato 2 + ADR-004).

Treina e avalia o modelo mais simples das 3 famílias comparadas no Tech
Challenge (Regressão Logística, Random Forest, MLPClassifier). Serve como
piso de referência: qualquer modelo mais complexo precisa justificar a
complexidade extra superando estas métricas.

Uso:
    from src.training.baseline import treinar_baseline
    resultado = treinar_baseline()
    print(resultado["metricas_teste"])
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import MLFLOW_EXPERIMENT_NAME, MODELS_DIR, PROCESSED_DATA_DIR
from src.features.preparation import build_pipeline, filtrar_censura, separar_alvo
from src.training.metrics import calcular_metricas, formatar_metricas

logger = logging.getLogger(__name__)

NOME_MODELO = "baseline_logistic_regression"


def _registrar_no_mlflow(params: dict, metricas_cv: dict, metricas_teste: dict) -> None:
    """Registra a run no MLflow. Funciona tanto local (sem DagsHub
    configurado, grava em ./mlruns) quanto remoto (se MLFLOW_TRACKING_URI
    estiver setado no .env) — ver ADR-004 em docs/decisions.md.

    Falha silenciosamente (só loga um aviso) se o MLflow não estiver
    disponível, para não quebrar o treino por causa de tracking opcional.
    """
    try:
        import mlflow

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=NOME_MODELO):
            mlflow.log_params(params)
            mlflow.log_metrics({f"cv_{k}": v for k, v in metricas_cv.items()})
            mlflow.log_metrics({f"teste_{k}": v for k, v in metricas_teste.items()})
    except Exception as erro:  # pragma: no cover - tracking é best-effort
        logger.warning("Não foi possível registrar no MLflow: %s", erro)


def treinar_baseline(
    random_state: int = 42,
    test_size: float = 0.2,
    n_splits: int = 5,
    salvar_modelo: bool = True,
) -> dict:
    """Treina e avalia o baseline de Regressão Logística.

    Fluxo: carrega o parquet processado -> filtra censura (mantém "Joined",
    src/features/preparation.py) -> separa X/y ->
    split estratificado -> validação cruzada estratificada no treino ->
    fit final -> avaliação única no teste guardado.

    Returns
    -------
    dict com: pipeline (treinado), metricas_cv, metricas_teste,
    X_test, y_test (para inspeção posterior, ex.: matriz de confusão).
    """
    df = pd.read_parquet(PROCESSED_DATA_DIR / "telco_churn_processed.parquet")
    df_modelagem, _ = filtrar_censura(df, remover_joined=False)
    X, y = separar_alvo(df_modelagem)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    modelo = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    pipeline: Pipeline = build_pipeline(modelo=modelo)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores_f1 = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1")
    scores_auc = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="roc_auc")
    scores_prauc = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="average_precision")
    metricas_cv = {
        "f1": scores_f1.mean(),
        "f1_std": scores_f1.std(),
        "auc_roc": scores_auc.mean(),
        "auc_roc_std": scores_auc.std(),
        "pr_auc": scores_prauc.mean(),
        "pr_auc_std": scores_prauc.std(),
    }

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metricas_teste = calcular_metricas(y_test, y_pred, y_proba)

    logger.info(
        "Validação cruzada (%d folds) — F1=%.3f AUC-ROC=%.3f PR-AUC=%.3f",
        n_splits,
        metricas_cv["f1"],
        metricas_cv["auc_roc"],
        metricas_cv["pr_auc"],
    )
    logger.info(formatar_metricas("Teste", metricas_teste))

    if salvar_modelo:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        caminho = Path(MODELS_DIR) / f"{NOME_MODELO}.joblib"
        joblib.dump(pipeline, caminho)
        logger.info("Modelo salvo em %s", caminho)

    _registrar_no_mlflow(
        params={
            "modelo": "LogisticRegression",
            "class_weight": "balanced",
            "random_state": random_state,
            "test_size": test_size,
            "n_splits": n_splits,
        },
        metricas_cv=metricas_cv,
        metricas_teste=metricas_teste.as_dict(),
    )

    return {
        "pipeline": pipeline,
        "metricas_cv": metricas_cv,
        "metricas_teste": metricas_teste,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    resultado = treinar_baseline()
    linha_final = formatar_metricas(
        "Baseline (Regressão Logística) — teste", resultado["metricas_teste"]
    )
    print("\n" + linha_final)
