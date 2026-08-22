"""Comparação dos candidatos a partir das runs registradas no MLflow.

A tabela comparativa é montada lendo o próprio MLflow, não as variáveis em
memória. Assim o ranking reflete o que ficou registrado, que é a fonte de
verdade auditável da etapa, e continua funcionando numa sessão nova sem
reexecutar o tuning.

O baseline da Etapa 1 entra na tabela como uma linha a mais, com as métricas
de CV recuperadas da run `baseline_logistic_regression` do notebook 04. Ele
não é retunado aqui: a comparação é justamente candidato tunado contra
baseline default.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from src import config
from src.logger import get_logger

logger = get_logger(__name__)

NOME_RUN_BASELINE = "baseline_logistic_regression"

CANDIDATOS_PADRAO = (
    "random_forest",
    "mlp",
    "mlp_balanceado",
    "logistic_regression_tunada",
)
"""Candidatos da Etapa 2, na ordem em que o notebook 05 os executa."""

METODOS = ("grid_search", "random_search", "optuna")

RUNS_DE_FECHAMENTO = ("etapa2_comparacao_final", "etapa2_campeao_final")
"""Runs de resumo, criadas depois que todos os candidatos rodaram."""

COLUNAS_TABELA = (
    "candidato",
    "metodo",
    "pr_auc_mean",
    "f1_mean",
    "auc_roc_mean",
    "tempo_segundos",
)
"""Colunas da tabela comparativa, na ordem. Vale também para a tabela vazia."""

METRICAS_BASELINE = (
    "cv_f1",
    "cv_auc_roc",
    "cv_pr_auc",
    "teste_f1",
    "teste_auc_roc",
    "teste_pr_auc",
)
"""Métricas que a run do baseline precisa ter para entrar na comparação."""

__all__ = [
    "CANDIDATOS_PADRAO",
    "COLUNAS_TABELA",
    "METODOS",
    "NOME_RUN_BASELINE",
    "RUNS_DE_FECHAMENTO",
    "buscar_baseline",
    "montar_tabela_comparativa",
    "nomes_de_runs",
    "registrar_comparacao",
]


def nomes_de_runs(candidatos: tuple[str, ...] = CANDIDATOS_PADRAO) -> list[str]:
    """Todos os nomes de run que uma execução completa da Etapa 2 cria.

    Serve de entrada para `config.limpar_runs_anteriores`, que apaga as runs
    antigas de mesmo nome antes de recriar. Sem isso, cada pessoa do grupo que
    roda o notebook duplica o histórico do experimento.
    """
    nomes = []
    for candidato in candidatos:
        nomes.append(candidato)
        nomes.extend(f"{candidato}__{metodo}" for metodo in METODOS)
    nomes.extend(RUNS_DE_FECHAMENTO)
    return nomes


def buscar_baseline(
    *,
    experimento: str | None = None,
    nome_run: str = NOME_RUN_BASELINE,
) -> dict[str, float] | None:
    """Recupera as métricas do baseline registrado pelo notebook 04.

    Returns
    -------
    dict | None
        `None` quando a run não existe **ou** existe sem as métricas
        esperadas, caso em que a comparação segue sem a linha do baseline.
        Rode `notebooks/04_baseline.ipynb` para criá-la.

        A run existir incompleta é situação real: `registrar_baseline` apaga a
        run antiga antes de criar a nova, então um notebook interrompido no
        meio da CV deixa uma run com o nome certo e sem métrica nenhuma.
    """
    experimento = experimento or config.MLFLOW_EXPERIMENT_NAME
    runs = mlflow.search_runs(
        experiment_names=[experimento],
        filter_string=f"tags.mlflow.runName = '{nome_run}'",
        order_by=["start_time DESC"],
    )

    if runs.empty:
        logger.warning(
            "Run '%s' não encontrada no experimento '%s'. A comparação vai sair sem "
            "o baseline: rode notebooks/04_baseline.ipynb antes.",
            nome_run,
            experimento,
        )
        return None

    linha = runs.iloc[0]

    faltando = [m for m in METRICAS_BASELINE if f"metrics.{m}" not in runs.columns]
    if faltando:
        logger.warning(
            "Run '%s' existe mas está sem as métricas %s. Provavelmente foi "
            "interrompida no meio. A comparação vai sair sem o baseline.",
            nome_run,
            faltando,
        )
        return None

    return {
        "cv_f1": linha["metrics.cv_f1"],
        "cv_auc_roc": linha["metrics.cv_auc_roc"],
        "cv_pr_auc": linha["metrics.cv_pr_auc"],
        "teste_f1": linha["metrics.teste_f1"],
        "teste_auc_roc": linha["metrics.teste_auc_roc"],
        "teste_pr_auc": linha["metrics.teste_pr_auc"],
        "run_id": linha["run_id"],
        "tempo_segundos": (linha["end_time"] - linha["start_time"]).total_seconds(),
    }


def montar_tabela_comparativa(
    *,
    candidatos: tuple[str, ...] = CANDIDATOS_PADRAO,
    baseline: dict[str, float] | None = None,
    experimento: str | None = None,
    max_results: int = 200,
) -> pd.DataFrame:
    """Uma linha por par (candidato, método), ordenada por PR-AUC de CV.

    Lê apenas runs aninhadas (`parentRunId` preenchido) cujo nome siga a
    convenção `candidato__metodo`, o que exclui as runs pai e as de
    fechamento sem precisar listá-las.

    Devolve uma tabela vazia, com as colunas certas, quando não há run que
    case. Isso acontece na primeira execução num experimento novo e depois de
    `config.limpar_runs_anteriores` apagar as runs antigas, e uma tabela vazia
    é mais fácil de diagnosticar do que um `KeyError` de coluna ausente.
    """
    experimento = experimento or config.MLFLOW_EXPERIMENT_NAME
    runs = mlflow.search_runs(
        experiment_names=[experimento],
        filter_string="tags.mlflow.parentRunId != ''",
        order_by=["start_time DESC"],
        max_results=max_results,
    )

    linhas = []
    if "tags.mlflow.runName" in runs.columns:
        prefixos = tuple(f"{candidato}__" for candidato in candidatos)
        runs = runs[
            runs["tags.mlflow.runName"].str.contains("__", na=False)
            & runs["tags.mlflow.runName"].str.startswith(prefixos)
        ]
    else:
        runs = runs.iloc[0:0]

    for _, linha in runs.iterrows():
        candidato, metodo = linha["tags.mlflow.runName"].split("__", 1)
        linhas.append(
            {
                "candidato": candidato,
                "metodo": metodo,
                "pr_auc_mean": linha.get("metrics.pr_auc_mean"),
                "f1_mean": linha.get("metrics.f1_mean"),
                "auc_roc_mean": linha.get("metrics.auc_roc_mean"),
                "tempo_segundos": linha.get("metrics.tempo_segundos"),
            }
        )

    if baseline is not None:
        linhas.append(
            {
                "candidato": NOME_RUN_BASELINE,
                "metodo": "n/a (nao tunado nesta etapa)",
                "pr_auc_mean": baseline["cv_pr_auc"],
                "f1_mean": baseline["cv_f1"],
                "auc_roc_mean": baseline["cv_auc_roc"],
                "tempo_segundos": baseline["tempo_segundos"],
            }
        )

    if not linhas:
        logger.warning(
            "Nenhuma run aninhada com o padrão 'candidato__metodo' no experimento "
            "'%s'. A tabela comparativa saiu vazia: rode o tuning antes.",
            experimento,
        )
        return pd.DataFrame(columns=list(COLUNAS_TABELA))

    return pd.DataFrame(linhas).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


def registrar_comparacao(
    tabela: pd.DataFrame,
    *,
    caminho_csv: Path | str | None = None,
    fonte: str = "notebooks/05_modelagem.ipynb",
    run_name: str = "etapa2_comparacao_final",
) -> pd.Series:
    """Salva a tabela em CSV, anexa como artefato e devolve a linha vencedora.

    O CSV vai junto da run porque a tabela é o insumo do Model Card, e um
    artefato versionado no MLflow é mais fácil de citar que um print.

    Raises
    ------
    ValueError
        Se a tabela está vazia, ou se a linha do topo não tem `pr_auc_mean`.
        Sem esse número não existe ranking, e anunciar um "vencedor por PR-AUC
        médio" a partir de uma ordenação arbitrária contaminaria o Model Card
        e a escolha do campeão logo em seguida.
    """
    if tabela.empty:
        raise ValueError(
            "Tabela comparativa vazia: nada a registrar. Rode o tuning dos "
            "candidatos antes (ver tuning.rodar_tuning)."
        )

    melhor_linha = tabela.iloc[0]
    if pd.isna(melhor_linha["pr_auc_mean"]):
        raise ValueError(
            "A linha do topo da tabela não tem 'pr_auc_mean', então a ordenação "
            "não significa nada. Verifique se as runs do tuning logaram a "
            "métrica pr_auc_mean."
        )

    caminho_csv = (
        Path(caminho_csv)
        if caminho_csv is not None
        else config.MODELS_DIR / "etapa2_tabela_comparativa.csv"
    )
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(caminho_csv, index=False)

    with config.iniciar_run(fonte, run_name=run_name):
        mlflow.log_artifact(str(caminho_csv))
        mlflow.log_params(
            {
                "candidato_vencedor": melhor_linha["candidato"],
                "metodo_vencedor": melhor_linha["metodo"],
            }
        )
        mlflow.log_metric("cv_pr_auc_mean_vencedor", melhor_linha["pr_auc_mean"])

    logger.info(
        "Vencedor por PR-AUC médio de CV: %s (%s).",
        melhor_linha["candidato"],
        melhor_linha["metodo"],
    )
    return melhor_linha
