"""Pipeline de ETL Telco: orquestra extract -> transform -> load."""

import pandas as pd

from src.config import RAW_DATA_DIR
from src.data.etl_config import ARQUIVOS_ALVO
from src.data.extract import baixar_arquivos_ibm, criar_diretorios
from src.data.load import salvar_parquet
from src.data.transform import unir_bases
from src.logger import get_logger

logger = get_logger(__name__)

# Mapeia cada arquivo de ARQUIVOS_ALVO (fonte única em etl_config) para a
# chave usada nos kwargs de `unir_bases`.
_CHAVE_POR_ARQUIVO: dict[str, str] = {
    "Telco_customer_churn_demographics.xlsx": "demographics",
    "Telco_customer_churn_location.xlsx": "locations",
    "Telco_customer_churn_population.xlsx": "populations",
    "Telco_customer_churn_services.xlsx": "services",
    "Telco_customer_churn_status.xlsx": "status",
}


def _carregar_bases_brutas() -> dict[str, pd.DataFrame]:
    """Lê os cinco arquivos xlsx brutos do diretório RAW_DATA_DIR em DataFrames."""
    return {
        _CHAVE_POR_ARQUIVO[nome_arquivo]: pd.read_excel(RAW_DATA_DIR / nome_arquivo)
        for nome_arquivo in ARQUIVOS_ALVO
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
