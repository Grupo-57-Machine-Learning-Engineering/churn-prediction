"""Configuração central: seeds, paths e constantes.

Fonte única de verdade para caminhos e parâmetros compartilhados entre as
trilhas (dados, modelo, API). Sem lógica de negócio — apenas constantes.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # carrega variáveis de .env (não versionado) se existir

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

# --- MLflow / DagsHub (config por variável de ambiente) ----------------------
MLFLOW_TRACKING_URI: str | None = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")
