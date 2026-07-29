"""Constantes específicas do pipeline de ETL Telco (fonte, arquivos alvo, saída)."""

BASE_URL: str = "https://public.dhe.ibm.com/software/data/sw-library/cognos/mobile/C11/data/"

ARQUIVOS_ALVO: list[str] = [
    "Telco_customer_churn_demographics.xlsx",
    "Telco_customer_churn_location.xlsx",
    "Telco_customer_churn_population.xlsx",
    "Telco_customer_churn_services.xlsx",
    "Telco_customer_churn_status.xlsx",
]

OUTPUT_FILENAME: str = "telco_churn_processed.parquet"
