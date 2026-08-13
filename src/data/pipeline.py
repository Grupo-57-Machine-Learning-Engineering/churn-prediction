"""Pipeline de ETL Telco: orquestra extract -> transform -> load."""

import pandas as pd

from src.config import RAW_DATA_DIR
from src.data.extract import baixar_arquivos_ibm, criar_diretorios
from src.data.load import salvar_parquet
from src.data.transform import unir_bases
from src.logger import get_logger

logger = get_logger(__name__)


def _carregar_bases_brutas() -> dict[str, pd.DataFrame]:
    """Lê os cinco arquivos xlsx brutos do diretório RAW_DATA_DIR em DataFrames."""
    return {
        "demographics": pd.read_excel(RAW_DATA_DIR / "Telco_customer_churn_demographics.xlsx"),
        "locations": pd.read_excel(RAW_DATA_DIR / "Telco_customer_churn_location.xlsx"),
        "populations": pd.read_excel(RAW_DATA_DIR / "Telco_customer_churn_population.xlsx"),
        "services": pd.read_excel(RAW_DATA_DIR / "Telco_customer_churn_services.xlsx"),
        "status": pd.read_excel(RAW_DATA_DIR / "Telco_customer_churn_status.xlsx"),
    }


def executar_pipeline() -> None:
    """Executa o pipeline completo: cria diretórios, baixa, transforma e salva."""
    criar_diretorios()
    baixar_arquivos_ibm()

    bases = _carregar_bases_brutas()
    df_final = unir_bases(**bases)

    salvar_parquet(df_final)
    logger.info("Pipeline de ETL concluído com sucesso.")


if __name__ == "__main__":
    executar_pipeline()
