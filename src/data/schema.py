"""Contrato de dados do parquet processado da Telco.

Promovido do schema diagnóstico validado em notebooks/02_eda.ipynb (seção 14)
contra a base real (data/processed/telco_churn_processed.parquet). Ver
docs/eda-findings.md.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column

SCHEMA = pa.DataFrameSchema(
    {
        "customer_id": Column(str, Check.str_length(min_value=1), unique=True),
        "demographics_age": Column(int, Check.in_range(0, 120)),
        "services_tenure_in_months": Column(int, Check.in_range(0, 100)),
        "services_monthly_charge": Column(float, Check.greater_than(0)),
        "services_total_charges": Column(float, Check.greater_than_or_equal_to(0)),
        "services_total_refunds": Column(float, Check.greater_than_or_equal_to(0)),
        "services_total_revenue": Column(float, Check.greater_than(0)),
        "services_total_extra_data_charges": Column(int, Check.greater_than_or_equal_to(0)),
        "services_avg_monthly_long_distance_charges": Column(
            float, Check.greater_than_or_equal_to(0)
        ),
        "services_total_long_distance_charges": Column(float, Check.greater_than_or_equal_to(0)),
        "services_avg_monthly_gb_download": Column(int, Check.greater_than_or_equal_to(0)),
        "services_number_of_referrals": Column(int, Check.greater_than_or_equal_to(0)),
        "demographics_number_of_dependents": Column(int, Check.greater_than_or_equal_to(0)),
        "populations_population": Column(int, Check.greater_than(0)),
        "status_satisfaction_score": Column(int, Check.in_range(1, 5)),
        "status_churn_score": Column(int, Check.in_range(0, 100)),
        "status_churn_label": Column(str, Check.isin(["Yes", "No"])),
        "services_contract": Column(str, Check.isin(["Month-to-Month", "One Year", "Two Year"])),
        # Categóricas de domínio fechado confirmado na EDA — seção 4
        # (demographics_gender é Female/Male; as demais são Yes/No):
        "demographics_gender": Column(str, Check.isin(["Female", "Male"])),
        "demographics_senior_citizen": Column(str, Check.isin(["Yes", "No"])),
        "demographics_dependents": Column(str, Check.isin(["Yes", "No"])),
        "demographics_married": Column(str, Check.isin(["Yes", "No"])),
        # Seção 10 (flags de serviço Yes/No + demographics_under_30):
        "demographics_under_30": Column(str, Check.isin(["Yes", "No"])),
        "services_phone_service": Column(str, Check.isin(["Yes", "No"])),
        "services_internet_service": Column(str, Check.isin(["Yes", "No"])),
        "services_multiple_lines": Column(str, Check.isin(["Yes", "No"])),
        "services_referred_a_friend": Column(str, Check.isin(["Yes", "No"])),
        "services_online_security": Column(str, Check.isin(["Yes", "No"])),
        "services_online_backup": Column(str, Check.isin(["Yes", "No"])),
        "services_device_protection_plan": Column(str, Check.isin(["Yes", "No"])),
        "services_premium_tech_support": Column(str, Check.isin(["Yes", "No"])),
        "services_streaming_tv": Column(str, Check.isin(["Yes", "No"])),
        "services_streaming_movies": Column(str, Check.isin(["Yes", "No"])),
        "services_streaming_music": Column(str, Check.isin(["Yes", "No"])),
        "services_unlimited_data": Column(str, Check.isin(["Yes", "No"])),
        "services_paperless_billing": Column(str, Check.isin(["Yes", "No"])),
        "services_payment_method": Column(  # seção 5
            str, Check.isin(["Bank Withdrawal", "Credit Card", "Mailed Check"])
        ),
        "status_customer_status": Column(  # seção 3
            str, Check.isin(["Stayed", "Churned", "Joined"])
        ),
        # Domínio confirmado na EDA, mas nulas legitimamente (ver
        # docs/eda-findings.md): precisam de nullable=True para não rejeitar
        # as linhas onde o serviço/evento não se aplica.
        "services_internet_type": Column(  # seção 5
            str, Check.isin(["Cable", "DSL", "Fiber Optic"]), nullable=True
        ),
        "services_offer": Column(  # seção 10
            str,
            Check.isin(["Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]),
            nullable=True,
        ),
        "status_churn_category": Column(  # seção 7
            str,
            Check.isin(["Attitude", "Competitor", "Dissatisfaction", "Other", "Price"]),
            nullable=True,
        ),
    },
    strict=False,  # colunas de texto livre, geolocalização e identificadores ficam de fora
)


def validar(df: pd.DataFrame) -> pd.DataFrame:
    """Valida df contra o contrato de dados.

    Levanta pandera.errors.SchemaErrors (com todas as falhas coletadas via
    lazy=True) se algum registro violar uma regra de domínio.
    """
    return SCHEMA.validate(df, lazy=True)
