"""Configuração central: seeds, paths e constantes.

Fonte única de verdade para caminhos e parâmetros compartilhados entre as
trilhas (dados, modelo, API). Sem lógica de negócio — apenas constantes.
"""

from __future__ import annotations

from pathlib import Path

# --- Reprodutibilidade -------------------------------------------------------
SEED: int = 42

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# --- Dataset (Telco Customer Churn / IBM) ------------------------------------
TARGET: str = "Churn"
ID_COLUMN: str = "customerID"

# --- Split -------------------------------------------------------------------
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.2  # fração do conjunto de treino reservada para validação
