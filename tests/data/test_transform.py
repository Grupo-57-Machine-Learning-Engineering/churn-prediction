"""Testes do módulo src.data.transform."""

import pandas as pd
import pytest

from src.data.transform import limpar_nome_coluna, padronizar_colunas, unir_bases


class TestLimparNomeColuna:
    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            ("Zip Code", "zip_code"),
            ("Customer ID", "customer_id"),
            ("Under 30 (Yes/No)", "under_30_yesno"),
            ("Churn-Score", "churn_score"),
            ("  Multiple   Spaces  ", "multiple_spaces"),
            ("Contract[Type]", "contracttype"),
            ("ALLCAPS", "allcaps"),
        ],
    )
    def test_normaliza_variados_padroes(self, entrada: str, esperado: str) -> None:
        assert limpar_nome_coluna(entrada) == esperado


@pytest.fixture
def location_key() -> str:
    return "zip_code"


@pytest.fixture
def df_amostra() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Customer ID": ["0001-ABC", "0002-DEF"],
            "Zip Code": [90001, 90002],
            "Some Column (X)": ["a", "b"],
        }
    )


class TestPadronizarColunas:
    def test_chaves_nao_recebem_prefixo(self, df_amostra: pd.DataFrame, location_key: str) -> None:
        resultado = padronizar_colunas(df_amostra, "demographics", location_key)
        assert "customer_id" in resultado.columns
        assert "zip_code" in resultado.columns

    def test_demais_colunas_recebem_prefixo(
        self, df_amostra: pd.DataFrame, location_key: str
    ) -> None:
        resultado = padronizar_colunas(df_amostra, "demographics", location_key)
        assert "demographics_some_column_x" in resultado.columns

    def test_nao_muta_dataframe_original(self, df_amostra: pd.DataFrame, location_key: str) -> None:
        colunas_originais = list(df_amostra.columns)
        padronizar_colunas(df_amostra, "demographics", location_key)
        assert list(df_amostra.columns) == colunas_originais


@pytest.fixture
def bases_sinteticas() -> dict[str, pd.DataFrame]:
    customer_ids = ["0001-ABC", "0002-DEF", "0003-GHI"]
    zip_codes = [90001, 90002, 90001]

    locations = pd.DataFrame({"Customer ID": customer_ids, "Zip Code": zip_codes})
    demographics = pd.DataFrame(
        {"Customer ID": customer_ids, "Gender": ["Male", "Female", "Female"]}
    )
    services = pd.DataFrame({"Customer ID": customer_ids, "Phone Service": ["Yes", "No", "Yes"]})
    status = pd.DataFrame({"Customer ID": customer_ids, "Churn Label": ["Yes", "No", "No"]})
    populations = pd.DataFrame({"Zip Code": [90001, 90002], "Population": [12000, 8000]})

    return {
        "demographics": demographics,
        "locations": locations,
        "populations": populations,
        "services": services,
        "status": status,
    }


class TestUnirBases:
    def test_shape_final_sem_explosao(self, bases_sinteticas: dict[str, pd.DataFrame]) -> None:
        resultado = unir_bases(**bases_sinteticas)
        assert len(resultado) == 3

    def test_colunas_de_chave_unicas_sem_sufixo_merge(
        self, bases_sinteticas: dict[str, pd.DataFrame]
    ) -> None:
        resultado = unir_bases(**bases_sinteticas)
        assert "customer_id" in resultado.columns
        assert "zip_code" in resultado.columns
        assert "zip_code_x" not in resultado.columns
        assert "zip_code_y" not in resultado.columns

    def test_dados_de_population_juntados_corretamente(
        self, bases_sinteticas: dict[str, pd.DataFrame]
    ) -> None:
        resultado = unir_bases(**bases_sinteticas)
        linha_zip_90001 = resultado[resultado["zip_code"] == 90001].iloc[0]
        assert linha_zip_90001["populations_population"] == 12000

    def test_prefixos_aplicados_nas_colunas_nao_chave(
        self, bases_sinteticas: dict[str, pd.DataFrame]
    ) -> None:
        resultado = unir_bases(**bases_sinteticas)
        assert "demographics_gender" in resultado.columns
        assert "services_phone_service" in resultado.columns
        assert "status_churn_label" in resultado.columns
