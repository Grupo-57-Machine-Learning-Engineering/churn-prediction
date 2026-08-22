"""Candidatos da Etapa 2 e seus espaços de busca.

Cada candidato tem uma *fábrica* (`criar_*`) e um `EspacoDeBusca`. A fábrica
fixa o que não é hiperparâmetro de busca (seed, balanceamento de classe,
critério de parada) e aceita `**overrides` para a configuração vencedora do
tuning. O espaço reúne as quatro peças que os três métodos de busca consomem,
que antes andavam soltas como argumentos posicionais.

Por que `class_weight="balanced"` no Random Forest e na Regressão Logística
mas não no MLP: o `MLPClassifier` do sklearn não aceita esse parâmetro, e o
balanceamento tem que entrar por `sample_weight` no `fit`. É por isso que o
notebook avalia `mlp` e `mlp_balanceado` como candidatos separados.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.config import SEED

__all__ = [
    "ESPACO_LOGISTIC_REGRESSION",
    "ESPACO_MLP",
    "ESPACO_RANDOM_FOREST",
    "EspacoDeBusca",
    "criar_logistic_regression",
    "criar_mlp",
    "criar_random_forest",
]


# ==========================================================================
# Fábricas de estimadores
# ==========================================================================


def criar_random_forest(*, n_jobs: int = -1, **overrides) -> RandomForestClassifier:
    """Random Forest com peso de classe balanceado e seed do projeto."""
    params: dict[str, Any] = dict(class_weight="balanced", random_state=SEED, n_jobs=n_jobs)
    params.update(overrides)
    return RandomForestClassifier(**params)


def criar_logistic_regression(**overrides) -> LogisticRegression:
    """Regressão Logística tunável.

    `solver="liblinear"` de propósito: é o único dos solvers padrão que
    suporta as duas penalidades do grid (`l1` e `l2`).
    """
    params: dict[str, Any] = dict(
        class_weight="balanced", random_state=SEED, max_iter=1000, solver="liblinear"
    )
    params.update(overrides)
    return LogisticRegression(**params)


def criar_mlp(**overrides) -> MLPClassifier:
    """MLP com early stopping ligado, para o custo do tuning não explodir."""
    params: dict[str, Any] = dict(
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        max_iter=500,
        random_state=SEED,
    )
    params.update(overrides)
    return MLPClassifier(**params)


# ==========================================================================
# Espaços de busca
# ==========================================================================


@dataclass(frozen=True)
class EspacoDeBusca:
    """As quatro visões do mesmo espaço, uma por método de busca.

    Attributes
    ----------
    grid
        Grade discreta do `GridSearchCV`. Chaves com o prefixo `modelo__`,
        porque o estimador é o último passo do Pipeline.
    distribuicoes
        Espaço amostrado pelo `RandomizedSearchCV`, mais largo que o grid.
    sugerir
        `Callable[[optuna.Trial], dict]` que devolve kwargs da fábrica.
    reconstruir
        Converte `study.best_params` de volta em kwargs da fábrica. Existe
        separado de `sugerir` porque o Optuna achata parâmetros compostos:
        `hidden_layer_sizes` vira dois inteiros no trial e precisa ser
        remontado como tupla.
    """

    grid: dict[str, list]
    distribuicoes: dict[str, list]
    sugerir: Callable[[Any], dict[str, Any]]
    reconstruir: Callable[[dict[str, Any]], dict[str, Any]]


def _sugerir_random_forest(trial) -> dict[str, Any]:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        n_jobs=-1,
    )


def _reconstruir_random_forest(params: dict[str, Any]) -> dict[str, Any]:
    return dict(params, n_jobs=-1)


ESPACO_RANDOM_FOREST = EspacoDeBusca(
    grid={
        "modelo__n_estimators": [200, 400],
        "modelo__max_depth": [None, 15],
        "modelo__min_samples_leaf": [1, 5],
    },
    distribuicoes={
        "modelo__n_estimators": list(range(100, 550, 50)),
        "modelo__max_depth": [None, 10, 15, 20, 30],
        "modelo__min_samples_leaf": list(range(1, 11)),
    },
    sugerir=_sugerir_random_forest,
    reconstruir=_reconstruir_random_forest,
)


def _montar_camadas(n1: int, n2: int) -> tuple[int, ...]:
    """`n2 == 0` significa rede de uma camada só."""
    return (n1,) if n2 == 0 else (n1, n2)


def _sugerir_mlp(trial) -> dict[str, Any]:
    n1 = trial.suggest_categorical("n_units_1", [32, 64, 128])
    n2 = trial.suggest_categorical("n_units_2", [0, 16, 32, 64])
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    return dict(hidden_layer_sizes=_montar_camadas(n1, n2), alpha=alpha)


def _reconstruir_mlp(params: dict[str, Any]) -> dict[str, Any]:
    camadas = _montar_camadas(params["n_units_1"], params["n_units_2"])
    return dict(hidden_layer_sizes=camadas, alpha=params["alpha"])


ESPACO_MLP = EspacoDeBusca(
    grid={
        "modelo__hidden_layer_sizes": [(32,), (64, 32)],
        "modelo__alpha": [1e-4, 1e-2],
    },
    distribuicoes={
        "modelo__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
        "modelo__alpha": np.logspace(-5, -1, 20).tolist(),
    },
    sugerir=_sugerir_mlp,
    reconstruir=_reconstruir_mlp,
)


def _sugerir_logistic_regression(trial) -> dict[str, Any]:
    return dict(
        C=trial.suggest_float("C", 1e-3, 1e3, log=True),
        penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
    )


def _reconstruir_logistic_regression(params: dict[str, Any]) -> dict[str, Any]:
    return dict(params)


ESPACO_LOGISTIC_REGRESSION = EspacoDeBusca(
    grid={
        "modelo__C": [0.01, 0.1, 1, 10, 100],
        "modelo__penalty": ["l1", "l2"],
    },
    distribuicoes={
        "modelo__C": np.logspace(-3, 3, 30).tolist(),
        "modelo__penalty": ["l1", "l2"],
    },
    sugerir=_sugerir_logistic_regression,
    reconstruir=_reconstruir_logistic_regression,
)
