"""Métricas de avaliação da Etapa 2.

Duas famílias, com propósitos diferentes.

As métricas *técnicas* (`calcular_metricas`) são as usadas para ranquear
candidatos: F1, AUC-ROC e PR-AUC. PR-AUC é o critério de desempate do
projeto, por ser a mais sensível à classe minoritária numa base desbalanceada.

As métricas *de negócio* (`calcular_metricas_negocio`) traduzem a matriz de
confusão em termos de decisão de retenção: quantos cancelamentos o modelo
pega (sensibilidade) e quanto da lista de risco é ruído (precisão). Elas
dependem do threshold, ao contrário das técnicas baseadas em probabilidade.

O protocolo de validação cruzada (`N_SPLITS`, `SCORING`) mora aqui, e não em
`tuning`, para que `avaliar_por_cv` funcione sem arrastar `mlflow` e
`optuna`. Avaliar um pipeline não deveria exigir servidor de tracking.
"""

from __future__ import annotations

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import SEED

N_SPLITS = 5
"""Folds da validação cruzada estratificada, iguais em toda a etapa."""

SCORING = {"f1": "f1", "auc_roc": "roc_auc", "pr_auc": "average_precision"}
"""Métricas coletadas na CV. As buscas de `tuning` fazem refit em `pr_auc`."""

__all__ = [
    "N_SPLITS",
    "SCORING",
    "avaliar_por_cv",
    "calcular_metricas",
    "calcular_metricas_negocio",
    "formatar_metricas",
    "formatar_metricas_cv",
    "formatar_metricas_negocio",
]


def calcular_metricas(y_true, y_pred, y_proba) -> dict[str, float]:
    """F1, AUC-ROC e PR-AUC de um conjunto qualquer.

    `y_pred` já vem binarizado; `y_proba` é a probabilidade da classe positiva.
    """
    return {
        "f1": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def formatar_metricas(nome_modelo: str, metricas: dict[str, float]) -> str:
    """Linha única para log ou print, no formato usado na comparação da etapa."""
    return (
        f"{nome_modelo}: "
        f"F1={metricas['f1']:.3f}  "
        f"AUC-ROC={metricas['auc_roc']:.3f}  "
        f"PR-AUC={metricas['pr_auc']:.3f}"
    )


def _divisao_segura(numerador: float, denominador: float) -> float:
    """Razão que devolve 0.0 em vez de estourar quando a célula está vazia.

    Acontece em recorte degenerado (por exemplo, nenhum positivo previsto),
    situação em que a métrica não é definida e interromper o treino inteiro
    seria pior do que reportar zero.
    """
    return float(numerador / denominador) if denominador else 0.0


def calcular_metricas_negocio(y_true, y_pred) -> dict[str, float]:
    """Matriz de confusão traduzida para as quatro taxas de decisão.

    * **sensibilidade** (recall da classe churn): dos cancelamentos reais,
      quantos o modelo pegou.
    * **especificidade** (recall da classe não-churn): de quem ia ficar,
      quantos o modelo deixou em paz.
    * **precisao** (VPP): de quem o modelo apontou como risco, quantos
      realmente cancelam. Define o desperdício da ação de retenção.
    * **vpn**: de quem o modelo liberou como sem risco, quantos ficam mesmo.

    As contagens `vn`/`fp`/`fn`/`vp` vêm junto porque o Model Card reporta a
    matriz bruta, não só as taxas.
    """
    # labels=[0, 1] força a matriz 2x2 mesmo quando o recorte tem uma classe
    # só. Sem isso o sklearn devolve 1x1, o unpack estoura, e a proteção de
    # `_divisao_segura` logo abaixo nunca chega a rodar.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "vn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "vp": int(tp),
        "sensibilidade": _divisao_segura(tp, tp + fn),
        "especificidade": _divisao_segura(tn, tn + fp),
        "precisao": _divisao_segura(tp, tp + fp),
        "vpn": _divisao_segura(tn, tn + fn),
    }


def formatar_metricas_negocio(metricas: dict[str, float], threshold: float = 0.5) -> str:
    """Bloco de duas linhas com a matriz e as taxas derivadas."""
    return (
        f"Matriz de confusão (threshold={threshold}): "
        f"VN={metricas['vn']} FP={metricas['fp']} "
        f"FN={metricas['fn']} VP={metricas['vp']}\n"
        f"Sensibilidade={metricas['sensibilidade']:.3f}  "
        f"Especificidade={metricas['especificidade']:.3f}  "
        f"Precisão={metricas['precisao']:.3f}  "
        f"VPN={metricas['vpn']:.3f}"
    )


def avaliar_por_cv(
    pipeline,
    X,
    y,
    *,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    scoring: dict[str, str] | None = None,
) -> dict[str, float]:
    """Validação cruzada estratificada de um pipeline, sem busca.

    É a avaliação do baseline: mede um pipeline já configurado, ao contrário
    de `tuning.rodar_tuning`, que procura hiperparâmetro. O `fit` acontece
    dentro de cada fold, então a preparação de features não vaza entre treino
    e validação.

    Returns
    -------
    dict[str, float]
        Média e desvio de cada métrica, nas chaves `cv_<metrica>` e
        `cv_<metrica>_std`. Esses nomes são os mesmos que
        `comparison.buscar_baseline` lê de volta do MLflow, então mudar um
        exige mudar o outro.
    """
    scoring = scoring or SCORING
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    resultados = cross_validate(pipeline, X, y, cv=skf, scoring=scoring)

    metricas: dict[str, float] = {}
    for nome in scoring:
        scores = resultados[f"test_{nome}"]
        metricas[f"cv_{nome}"] = float(scores.mean())
        metricas[f"cv_{nome}_std"] = float(scores.std())
    return metricas


def formatar_metricas_cv(metricas: dict[str, float]) -> str:
    """Uma linha por métrica, no formato `media (+/- desvio)`."""
    nomes = [chave[3:] for chave in metricas if not chave.endswith("_std")]
    return "\n".join(
        f"{nome.upper():8s} {metricas[f'cv_{nome}']:.3f} (+/- {metricas[f'cv_{nome}_std']:.3f})"
        for nome in nomes
    )
