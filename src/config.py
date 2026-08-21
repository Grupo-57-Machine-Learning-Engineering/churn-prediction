"""Configuração central: seeds, paths e constantes.

Fonte única de verdade para caminhos e parâmetros compartilhados entre as
trilhas (dados, modelo, API). Sem lógica de negócio — apenas constantes.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # carrega variáveis de .env (não versionado) se existir

logger = logging.getLogger(__name__)

# --- Reprodutibilidade -------------------------------------------------------
SEED: int = 42

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# --- Dataset (Telco Customer Churn / IBM) ------------------------------------
# Nomes pós-ETL (Contrato 1, docs/decisions.md), nao os do CSV Kaggle de 21
# colunas que o ADR-003 decidiu nao usar como fonte de dados.
TARGET: str = "status_churn_value"
ID_COLUMN: str = "customer_id"

# --- Split -------------------------------------------------------------------
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.2  # fração do conjunto de treino reservada para validação

# --- Modelo campeão / API (Etapa 3) ------------------------------------------
CHAMPION_MODEL_PATH: Path = MODELS_DIR / "champion_model.joblib"
"""Artefato local do campeão. O notebook 05 grava esse arquivo sempre, com ou
sem MLflow no ar, porque ele é entregável explícito do enunciado (ADR-004)."""

CHAMPION_MODEL_URI: str = "models:/churn_champion@champion"
"""Endereço do campeão no Model Registry do MLflow/DagsHub (fallback da API)."""

THRESHOLD_DECISAO: float = 0.5
"""Threshold padrão para converter probabilidade em classe no `/predict`.

Mantido em 0,5, o mesmo valor com que o campeão foi avaliado (matriz de
confusão e métricas de negócio do ADR-004: sensibilidade 0,813, precisão
0,624). O trade-off entre sensibilidade e precisão para outros thresholds
está documentado no notebook 05 (seções 9 e 10). Ajustar aqui é decisão de
grupo e deve ser registrada em docs/decisions.md."""

# --- MLflow / DagsHub (config por variável de ambiente) ----------------------
MLFLOW_TRACKING_URI: str | None = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME: str | None = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD: str | None = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")

# Repositório DagsHub do grupo, usado só pela Opção B do dagshub.init().
# Sem default aqui de propósito -- configure em .env (ver .env.example e
# README). Nome diferente do repo no GitHub.
DAGSHUB_REPO_OWNER: str | None = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME: str | None = os.getenv("DAGSHUB_REPO_NAME")


def configurar_mlflow_tracking() -> None:
    """Aponta o MLflow para o DagsHub antes do primeiro `mlflow.start_run()`.

    Duas formas de autenticar (ver ADR-004 em docs/decisions.md):

    * **Opção A -- token**: se `MLFLOW_TRACKING_URI` + `MLFLOW_TRACKING_USERNAME`
      + `MLFLOW_TRACKING_PASSWORD` estiverem todos preenchidos no `.env`, não faz
      nada -- o cliente MLflow já lê essas 3 variáveis de ambiente sozinho.
    * **Opção B -- login interativo**: se qualquer uma estiver faltando, chama
      `dagshub.init(...)`, que autentica via navegador na primeira vez (token
      cacheado localmente depois) e configura o MLflow sozinho.

    Best-effort: se o DagsHub não estiver acessível (rede, import ausente),
    loga um aviso e segue sem tracking -- não é requisito do enunciado atual.
    """
    if MLFLOW_TRACKING_URI and MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD:
        return

    if not (DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME):
        logger.warning(
            "Tracking no DagsHub não configurado: preencha MLFLOW_TRACKING_URI/"
            "USERNAME/PASSWORD ou DAGSHUB_REPO_OWNER/DAGSHUB_REPO_NAME no .env "
            "(ver .env.example)."
        )
        return

    _forcar_utf8_no_console()

    try:
        import dagshub

        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    except Exception as erro:  # pragma: no cover - tracking é best-effort
        logger.warning("Não foi possível configurar o tracking no DagsHub: %s", erro)


def limpar_runs_anteriores(nomes: list[str]) -> None:
    """Remove (soft-delete) runs do MLflow com esses nomes no experimento atual.

    Torna reexecucoes de notebook idempotentes: sem isso, cada pessoa do grupo
    que roda o mesmo notebook gera runs duplicadas (ja aconteceu com o baseline
    da Etapa 1, ver ADR-004 em docs/decisions.md).

    Busca um nome por vez em vez de um filtro `" or "` unico: o backend do MLflow
    no DagsHub rejeita (`INVALID_PARAMETER_VALUE`) um filter_string com 2+
    clausulas OR, mesmo sendo filtro valido pro MLflow em geral.
    """
    if not nomes:
        return

    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    total_removidas = 0
    for nome in nomes:
        existentes = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            filter_string=f"tags.mlflow.runName = '{nome}'",
            max_results=200,
        )
        for run_id in existentes.get("run_id", []):
            client.delete_run(run_id)
        total_removidas += len(existentes)
    if total_removidas:
        logger.info("Idempotencia: removida(s) %d run(s) anterior(es).", total_removidas)


def iniciar_run(nome_arquivo_fonte: str, **kwargs):
    """`mlflow.start_run()` fixando `mlflow.source.name` explicitamente.

    Em execucao headless (ex.: `jupyter nbconvert --execute`, sem servidor
    Jupyter interativo por tras), o MLflow cai para o caminho local do
    `ipykernel_launcher.py`, expondo o caminho absoluto de quem rodou. Ver
    ADR-004 em docs/decisions.md.
    """
    import mlflow

    ctx = mlflow.start_run(**kwargs)
    mlflow.set_tag("mlflow.source.name", nome_arquivo_fonte)
    return ctx


def _forcar_utf8_no_console() -> None:
    """Evita `UnicodeEncodeError` do `dagshub.init()` no console do Windows.

    O terminal do Windows usa por padrão uma codepage legada (cp1252) que não
    representa os caracteres/cores que a lib `dagshub` (via `rich`) usa para
    imprimir o link e o código do login OAuth -- o processo quebra silenciosamente
    no meio do print, antes de mostrar o link. Reconfigurar stdout/stderr para
    UTF-8 resolve; é um no-op fora do Windows.
    """
    import sys

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(Exception):  # pragma: no cover - best-effort
                stream.reconfigure(encoding="utf-8")
