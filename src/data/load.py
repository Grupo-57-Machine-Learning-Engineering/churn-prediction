"""Carga: persistência do DataFrame final em formato parquet."""

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.data.etl_config import OUTPUT_FILENAME
from src.logger import get_logger

logger = get_logger(__name__)


def salvar_parquet(df: pd.DataFrame, destino: Path | None = None) -> Path:
    """Salva o DataFrame processado em parquet.

    Args:
        df: DataFrame final a persistir.
        destino: caminho completo do arquivo de saída.

    Returns:
        Path do arquivo salvo.
    """
    if destino is None:
        destino = PROCESSED_DATA_DIR / OUTPUT_FILENAME
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)
    logger.info("Base unificada salva em: %s", destino)
    logger.info("Shape final dos dados: %s", df.shape)
    return destino
