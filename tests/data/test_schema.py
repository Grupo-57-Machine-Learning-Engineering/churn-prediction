"""Testes do contrato de dados em src.data.schema."""

import pandas as pd
import pandera.pandas as pa
import pytest
from src.data.etl_config import OUTPUT_FILENAME

from src.config import PROCESSED_DATA_DIR
from src.data.schema import validar


@pytest.fixture
def df_valido() -> pd.DataFrame:
    """Duas linhas com perfis internamente consistentes (verificado contra a base real):

    - churn_label="Yes" <=> customer_status="Churned" (correspondência 1:1 na base real).
    - senior_citizen="Yes" <=> age >= 65 (limiar exato observado na base real).
    - internet_service="No" <=> internet_type nulo (equivalência 100% na base real);
      sem internet, todos os add-ons de internet são "No" e gb_download é 0
      (zero estrutural, nas 1.526 linhas sem internet da base real).
    - referred_a_friend="Yes" <=> number_of_referrals > 0 (equivalência 100%).
    - total_revenue == total_charges - refunds + extra_data + long_distance e
      total_long_distance == avg_monthly_long_distance * tenure
      (identidades exatas na base real).
    - churn_category só é preenchido para customer_status="Churned".
    """
    return pd.DataFrame(
        {
            "customer_id": ["0001-ABC", "0002-DEF"],
            "demographics_age": [34, 68],
            "services_tenure_in_months": [12, 48],
            "services_monthly_charge": [65.5, 89.9],
            "services_total_charges": [786.0, 4315.2],
            "services_total_refunds": [0.0, 0.0],
            "services_total_extra_data_charges": [0, 0],
            "services_avg_monthly_long_distance_charges": [10.0, 12.5],
            "services_total_long_distance_charges": [120.0, 600.0],
            "services_total_revenue": [906.0, 4915.2],
            "services_avg_monthly_gb_download": [20, 0],
            "services_number_of_referrals": [2, 0],
            "demographics_number_of_dependents": [1, 0],
            "populations_population": [68701, 55668],
            "status_satisfaction_score": [3, 2],
            "status_churn_score": [20, 85],
            "status_churn_label": ["No", "Yes"],
            "services_contract": ["Month-to-Month", "Month-to-Month"],
            "demographics_gender": ["Female", "Male"],
            "demographics_senior_citizen": ["No", "Yes"],
            "demographics_dependents": ["Yes", "No"],
            "demographics_married": ["Yes", "No"],
            "demographics_under_30": ["No", "No"],
            "services_phone_service": ["Yes", "Yes"],
            "services_multiple_lines": ["No", "Yes"],
            "services_internet_service": ["Yes", "No"],
            "services_referred_a_friend": ["Yes", "No"],
            "services_online_security": ["Yes", "No"],
            "services_online_backup": ["Yes", "No"],
            "services_device_protection_plan": ["No", "No"],
            "services_premium_tech_support": ["Yes", "No"],
            "services_streaming_tv": ["No", "No"],
            "services_streaming_movies": ["No", "No"],
            "services_streaming_music": ["No", "No"],
            "services_unlimited_data": ["No", "No"],
            "services_paperless_billing": ["Yes", "No"],
            "services_payment_method": ["Credit Card", "Mailed Check"],
            "status_customer_status": ["Stayed", "Churned"],
            "services_internet_type": ["Fiber Optic", None],
            "services_offer": ["Offer A", None],
            "status_churn_category": [None, "Competitor"],
        }
    )


class TestValidar:
    def test_dataframe_valido_passa(self, df_valido: pd.DataFrame) -> None:
        validar(df_valido)

    def test_idade_fora_de_faixa_levanta_erro(self, df_valido: pd.DataFrame) -> None:
        df_valido.loc[0, "demographics_age"] = -5
        with pytest.raises(pa.errors.SchemaErrors):
            validar(df_valido)

    def test_churn_label_invalido_levanta_erro(self, df_valido: pd.DataFrame) -> None:
        df_valido.loc[0, "status_churn_label"] = "Maybe"
        with pytest.raises(pa.errors.SchemaErrors):
            validar(df_valido)

    def test_customer_id_duplicado_levanta_erro(self, df_valido: pd.DataFrame) -> None:
        df_valido.loc[1, "customer_id"] = df_valido.loc[0, "customer_id"]
        with pytest.raises(pa.errors.SchemaErrors):
            validar(df_valido)

    def test_categoria_fechada_invalida_levanta_erro(self, df_valido: pd.DataFrame) -> None:
        df_valido.loc[0, "services_internet_type"] = "Satellite"
        with pytest.raises(pa.errors.SchemaErrors):
            validar(df_valido)

    def test_cobranca_negativa_levanta_erro(self, df_valido: pd.DataFrame) -> None:
        df_valido.loc[0, "services_total_long_distance_charges"] = -10.0
        with pytest.raises(pa.errors.SchemaErrors):
            validar(df_valido)

    def test_nulo_legitimo_em_categoria_nullable_passa(self, df_valido: pd.DataFrame) -> None:
        assert df_valido["services_offer"].isna().any()
        validar(df_valido)


class TestValidarBaseReal:
    def test_parquet_processado_passa_no_contrato(self) -> None:
        caminho = PROCESSED_DATA_DIR / OUTPUT_FILENAME
        if not caminho.exists():
            pytest.skip(f"{caminho} não existe — rode o pipeline de ETL primeiro.")
        df = pd.read_parquet(caminho)
        validar(df)
