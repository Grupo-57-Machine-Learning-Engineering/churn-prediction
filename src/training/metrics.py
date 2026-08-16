from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


@dataclass
class Metricas:
    """Métricas de classificação para a classe positiva (churn=1).

    Acurácia é omitida de propósito: sozinha, é enganosa nesta base
    desbalanceada. F1/AUC-ROC/PR-AUC são as métricas de referência.
    """

    f1: float
    auc_roc: float
    pr_auc: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Metricas:
    """Calcula F1, AUC-ROC e PR-AUC para uma previsão binária.

    Parameters
    ----------
    y_true: rótulos reais (0/1).
    y_pred: rótulos previstos pelo modelo (0/1, após o threshold padrão 0.5).
    y_proba: probabilidade prevista da classe positiva (churn=1).
    """
    return Metricas(
        f1=f1_score(y_true, y_pred),
        auc_roc=roc_auc_score(y_true, y_proba),
        pr_auc=average_precision_score(y_true, y_proba),
    )


def formatar_metricas(nome_modelo: str, metricas: Metricas) -> str:
    """Formata as métricas para impressão/log legível."""
    return (
        f"{nome_modelo}: "
        f"F1={metricas.f1:.3f}  "
        f"AUC-ROC={metricas.auc_roc:.3f}  "
        f"PR-AUC={metricas.pr_auc:.3f}"
    )
