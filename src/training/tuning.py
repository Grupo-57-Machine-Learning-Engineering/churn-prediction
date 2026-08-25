"""Tuning de hiperparâmetros: GridSearchCV, RandomizedSearchCV e Optuna.

Os três métodos rodam sobre o mesmo candidato e o mesmo `EspacoDeBusca`, e o
vencedor é o de maior PR-AUC médio de validação cruzada. Comparar os métodos
entre si, e não só os modelos, é exigência da Etapa 2.

Estrutura das runs no MLflow: uma run pai por candidato e três runs aninhadas,
uma por método. A run pai recebe `melhor_metodo` e `melhor_pr_auc_mean`, o que
permite ler o resultado sem abrir as filhas.

O protocolo de CV (`N_SPLITS`, `SCORING`) vem de `metrics`, para ser o mesmo
que `metrics.avaliar_por_cv` usa no baseline. Comparar candidato tunado com
baseline exige que os dois tenham sido medidos do mesmo jeito.

Atenção ao `pr_auc_std`: nos três métodos ele reproduz o notebook 05, mas não
quer dizer a mesma coisa. No grid e no random é a dispersão da configuração
vencedora entre os folds; no Optuna é a dispersão entre os trials do estudo.
Para comparar os três, use o `pr_auc_std_entre_folds`, logado junto na run do
Optuna. O motivo de não sobrescrever a coluna está no ADR-008 de
`docs/decisions.md`.

Todo estado entra por parâmetro. As funções não leem `X_train`/`y_train` de
escopo global, que era como o notebook operava.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from src.config import SEED, iniciar_run
from src.features.preparation import build_pipeline
from src.logger import get_logger
from src.training.estimators import EspacoDeBusca
from src.training.metrics import N_SPLITS, SCORING

logger = get_logger(__name__)

FONTE_PADRAO = "notebooks/05_modelagem.ipynb"
"""Valor de `mlflow.source.name` das runs (ver `config.iniciar_run`)."""

__all__ = [
    "FONTE_PADRAO",
    "ResultadoTuning",
    "rodar_tuning",
    "silenciar_optuna",
]


def silenciar_optuna() -> None:
    """Reduz o log do Optuna a warnings.

    Fica como função em vez de efeito de import: um módulo importado não
    deveria reconfigurar o logging do processo inteiro por conta própria.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ResultadoTuning:
    """Resultado de um método de busca para um candidato.

    `pipeline` é o Pipeline completo já fitado no treino inteiro, pronto para
    virar campeão sem refit.
    """

    metodo: str
    pr_auc_mean: float
    pr_auc_std: float
    f1_mean: float
    auc_roc_mean: float
    tempo_segundos: float
    params: dict[str, Any] = field(default_factory=dict)
    pipeline: Pipeline | None = None


def _fit_kwargs(sample_weight, indices=None) -> dict[str, Any]:
    """Repassa `sample_weight` para o passo `modelo` do Pipeline, se houver."""
    if sample_weight is None:
        return {}
    pesos = sample_weight if indices is None else sample_weight[indices]
    return {"modelo__sample_weight": pesos}


def _logar_resultado(resultado: ResultadoTuning, extras: dict[str, Any] | None = None) -> None:
    """Escreve params e métricas do método na run aninhada ativa."""
    mlflow.log_param("metodo_tuning", resultado.metodo)
    mlflow.log_params({f"param_{k}": v for k, v in resultado.params.items()})
    if extras:
        mlflow.log_params(extras)
    mlflow.log_metrics(
        {
            "pr_auc_mean": resultado.pr_auc_mean,
            "pr_auc_std": resultado.pr_auc_std,
            "f1_mean": resultado.f1_mean,
            "auc_roc_mean": resultado.auc_roc_mean,
            "tempo_segundos": resultado.tempo_segundos,
        }
    )


def _resultado_de_busca(busca, metodo: str, tempo: float) -> ResultadoTuning:
    """Converte o `cv_results_` do sklearn na configuração vencedora."""
    idx = busca.best_index_
    cv = busca.cv_results_
    return ResultadoTuning(
        metodo=metodo,
        pr_auc_mean=float(cv["mean_test_pr_auc"][idx]),
        pr_auc_std=float(cv["std_test_pr_auc"][idx]),
        f1_mean=float(cv["mean_test_f1"][idx]),
        auc_roc_mean=float(cv["mean_test_auc_roc"][idx]),
        tempo_segundos=tempo,
        params=busca.best_params_,
        pipeline=busca.best_estimator_,
    )


def _rodar_grid_search(
    criar_estimador, espaco, X_train, y_train, *, cv, sample_weight
) -> ResultadoTuning:
    t0 = time.time()
    busca = GridSearchCV(
        build_pipeline(modelo=criar_estimador()),
        param_grid=espaco.grid,
        scoring=SCORING,
        refit="pr_auc",
        cv=cv,
        n_jobs=-1,
    )
    busca.fit(X_train, y_train, **_fit_kwargs(sample_weight))
    return _resultado_de_busca(busca, "grid_search", time.time() - t0)


def _rodar_random_search(
    criar_estimador, espaco, X_train, y_train, *, cv, sample_weight, n_iter, seed
) -> ResultadoTuning:
    t0 = time.time()
    # Sem n_iter explícito, sorteia tantas combinações quanto a grade completa
    # tem, para o orçamento das duas buscas do sklearn ficar comparável.
    n_iter = n_iter or len(list(ParameterGrid(espaco.grid)))
    busca = RandomizedSearchCV(
        build_pipeline(modelo=criar_estimador()),
        param_distributions=espaco.distribuicoes,
        n_iter=n_iter,
        scoring=SCORING,
        refit="pr_auc",
        cv=cv,
        n_jobs=-1,
        random_state=seed,
    )
    busca.fit(X_train, y_train, **_fit_kwargs(sample_weight))
    return _resultado_de_busca(busca, "random_search", time.time() - t0)


def _objetivo_optuna(
    trial, criar_estimador, espaco, X_train, y_train, sample_weight, n_splits, seed
) -> float:
    """PR-AUC média dos folds, com pruning por fold.

    O `trial.report` a cada fold permite ao `MedianPruner` abortar uma
    configuração ruim antes dos 5 folds, que é de onde vem a economia de tempo
    do Optuna em relação às buscas exaustivas.
    """
    estimador = criar_estimador(**espaco.sugerir(trial))
    pipeline = build_pipeline(modelo=estimador)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = []
    for fold, (idx_tr, idx_val) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[idx_tr], X_train.iloc[idx_val]
        y_tr, y_val = y_train.iloc[idx_tr], y_train.iloc[idx_val]

        pipeline.fit(X_tr, y_tr, **_fit_kwargs(sample_weight, idx_tr))
        score = average_precision_score(y_val, pipeline.predict_proba(X_val)[:, 1])
        scores.append(score)

        trial.report(score, step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def _metricas_cv_complementares(
    criar_estimador, melhores_kwargs, X_train, y_train, *, sample_weight, n_splits, seed
) -> tuple[float, float, float]:
    """Métricas de CV da configuração vencedora do Optuna, fold a fold.

    O objetivo do estudo só devolve a PR-AUC média. Sem esta recomputação, a
    linha do Optuna na tabela comparativa ficaria com colunas vazias em
    relação às buscas do sklearn, que entregam as três métricas de graça no
    `cv_results_`.

    O terceiro valor devolvido é o desvio da PR-AUC **entre os folds**, a
    mesma grandeza que o `std_test_pr_auc` do GridSearchCV e do
    RandomizedSearchCV. Ele não substitui o `pr_auc_std` do resultado, que
    reproduz o notebook: vai para a run como `pr_auc_std_entre_folds`, para
    quem precisar comparar os três métodos (ver ADR-008). Sai de graça, porque
    o laço já refita a configuração vencedora em cada fold.

    Returns
    -------
    (f1_mean, auc_roc_mean, pr_auc_std_entre_folds)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    f1s, aucs, pr_aucs = [], [], []
    for idx_tr, idx_val in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[idx_tr], X_train.iloc[idx_val]
        y_tr, y_val = y_train.iloc[idx_tr], y_train.iloc[idx_val]

        pipeline = build_pipeline(modelo=criar_estimador(**melhores_kwargs))
        pipeline.fit(X_tr, y_tr, **_fit_kwargs(sample_weight, idx_tr))
        proba = pipeline.predict_proba(X_val)[:, 1]

        f1s.append(f1_score(y_val, (proba >= 0.5).astype(int)))
        aucs.append(roc_auc_score(y_val, proba))
        pr_aucs.append(average_precision_score(y_val, proba))

    return float(np.mean(f1s)), float(np.mean(aucs)), float(np.std(pr_aucs))


def _rodar_optuna(
    nome, criar_estimador, espaco, X_train, y_train, *, sample_weight, n_trials, n_splits, seed
) -> tuple[ResultadoTuning, dict[str, Any]]:
    estudo = optuna.create_study(
        study_name=nome,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    t0 = time.time()
    estudo.optimize(
        lambda trial: _objetivo_optuna(
            trial, criar_estimador, espaco, X_train, y_train, sample_weight, n_splits, seed
        ),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )
    tempo = time.time() - t0

    melhores_kwargs = espaco.reconstruir(estudo.best_params)
    pipeline_final = build_pipeline(modelo=criar_estimador(**melhores_kwargs))
    pipeline_final.fit(X_train, y_train, **_fit_kwargs(sample_weight))

    f1_mean, auc_roc_mean, pr_auc_std_entre_folds = _metricas_cv_complementares(
        criar_estimador,
        melhores_kwargs,
        X_train,
        y_train,
        sample_weight=sample_weight,
        n_splits=n_splits,
        seed=seed,
    )

    # `pr_auc_std` é calculado exatamente como no notebook 05, dispersão entre
    # os trials do estudo, para o número bater com as runs já registradas na
    # Etapa 2. Ele não é comparável com o `std_test_pr_auc` que o grid e o
    # random logam sob o mesmo nome, que é dispersão entre folds. O desvio
    # entre folds vai junto, como métrica separada, para quem precisar
    # comparar os três métodos (ADR-008 em docs/decisions.md).
    valores = [t.value for t in estudo.trials if t.value is not None]
    podados = sum(1 for t in estudo.trials if t.state == optuna.trial.TrialState.PRUNED)

    resultado = ResultadoTuning(
        metodo="optuna",
        pr_auc_mean=estudo.best_value,
        pr_auc_std=float(np.std(valores)),
        f1_mean=f1_mean,
        auc_roc_mean=auc_roc_mean,
        tempo_segundos=tempo,
        params=melhores_kwargs,
        pipeline=pipeline_final,
    )
    extras = {
        "n_trials": n_trials,
        "n_trials_podados": podados,
        "pr_auc_std_entre_folds": pr_auc_std_entre_folds,
    }
    return resultado, extras


def rodar_tuning(
    nome: str,
    criar_estimador,
    espaco: EspacoDeBusca,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    sample_weight=None,
    n_iter_random: int | None = None,
    n_trials_optuna: int = 25,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    fonte: str = FONTE_PADRAO,
) -> dict[str, ResultadoTuning]:
    """Roda os três métodos de busca para um candidato e loga tudo no MLflow.

    Parameters
    ----------
    nome
        Nome do candidato, usado como nome da run pai (`random_forest`, `mlp`,
        `mlp_balanceado`, `logistic_regression_tunada`).
    criar_estimador
        Fábrica de `src.training.estimators`.
    espaco
        `EspacoDeBusca` do mesmo candidato.
    sample_weight
        Pesos por linha do treino, para o candidato balanceado por
        `sample_weight` em vez de `class_weight`. Fatiado por fold nas CVs.
    n_iter_random
        Combinações sorteadas pelo `RandomizedSearchCV`. `None` iguala ao
        tamanho da grade completa.
    fonte
        Caminho registrado como `mlflow.source.name`, para a run não expor o
        caminho local de quem executou (ver ADR-004).

    Returns
    -------
    dict[str, ResultadoTuning]
        Chaveado por `grid_search`, `random_search` e `optuna`.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    resultados: dict[str, ResultadoTuning] = {}

    with iniciar_run(fonte, run_name=nome):
        with iniciar_run(fonte, run_name=f"{nome}__grid_search", nested=True):
            resultados["grid_search"] = _rodar_grid_search(
                criar_estimador, espaco, X_train, y_train, cv=cv, sample_weight=sample_weight
            )
            _logar_resultado(resultados["grid_search"])

        with iniciar_run(fonte, run_name=f"{nome}__random_search", nested=True):
            resultados["random_search"] = _rodar_random_search(
                criar_estimador,
                espaco,
                X_train,
                y_train,
                cv=cv,
                sample_weight=sample_weight,
                n_iter=n_iter_random,
                seed=seed,
            )
            _logar_resultado(resultados["random_search"])

        with iniciar_run(fonte, run_name=f"{nome}__optuna", nested=True):
            resultados["optuna"], extras = _rodar_optuna(
                nome,
                criar_estimador,
                espaco,
                X_train,
                y_train,
                sample_weight=sample_weight,
                n_trials=n_trials_optuna,
                n_splits=n_splits,
                seed=seed,
            )
            _logar_resultado(resultados["optuna"], extras)

        melhor_metodo = max(resultados, key=lambda m: resultados[m].pr_auc_mean)
        mlflow.log_param("melhor_metodo", melhor_metodo)
        mlflow.log_metric("melhor_pr_auc_mean", resultados[melhor_metodo].pr_auc_mean)

    for metodo, resultado in resultados.items():
        logger.info(
            "[%s] %-14s PR-AUC=%.4f (%.1fs)",
            nome,
            metodo,
            resultado.pr_auc_mean,
            resultado.tempo_segundos,
        )

    return resultados
