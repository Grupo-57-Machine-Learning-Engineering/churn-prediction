"""Fixtures compartilhadas dos testes da Etapa 3 (API + módulo de predição).

O `models/champion_model.joblib` não é versionado, só existe em quem rodou
o notebook 05, então nenhum teste pode depender dele porque o CI não o tem.
No lugar, `campeao_sintetico` treina em segundos um pipeline com a mesma
factory de produção (`build_pipeline`) sobre uma base sintética com as 28
colunas do Contrato 3, e os testes injetam esse substituto na API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import SEED
from src.features.preparation import build_pipeline

N = 240
SEED_FIXTURE = 57


def _base_contrato_3(n: int = N, seed: int = SEED_FIXTURE) -> tuple[pd.DataFrame, pd.Series]:
    """Base sintética com exatamente as 28 colunas de entrada do Contrato 3."""
    rng = np.random.default_rng(seed)

    tem_internet = rng.random(n) > 0.25
    tem_telefone = rng.random(n) > 0.10
    tenure = rng.integers(1, 72, n)
    mensal = rng.uniform(20, 120, n).round(2)
    contrato = rng.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.55, 0.22, 0.23])

    X = pd.DataFrame(
        {
            "demographics_gender": rng.choice(["Male", "Female"], n),
            "demographics_age": rng.integers(19, 80, n),
            "demographics_senior_citizen": rng.choice(["Yes", "No"], n),
            "demographics_married": rng.choice(["Yes", "No"], n),
            "demographics_dependents": rng.choice(["Yes", "No"], n),
            "demographics_number_of_dependents": rng.integers(0, 4, n),
            "services_number_of_referrals": rng.integers(0, 6, n),
            "services_tenure_in_months": tenure,
            "services_phone_service": np.where(tem_telefone, "Yes", "No"),
            "services_avg_monthly_long_distance_charges": np.where(
                tem_telefone, rng.uniform(1, 50, n).round(2), 0.0
            ),
            "services_multiple_lines": rng.choice(["Yes", "No"], n),
            "services_internet_type": np.where(
                tem_internet, rng.choice(["Fiber Optic", "Cable", "DSL"], n), None
            ),
            "services_avg_monthly_gb_download": np.where(tem_internet, rng.integers(5, 90, n), 0),
            "services_online_security": rng.choice(["Yes", "No"], n),
            "services_online_backup": rng.choice(["Yes", "No"], n),
            "services_device_protection_plan": rng.choice(["Yes", "No"], n),
            "services_premium_tech_support": rng.choice(["Yes", "No"], n),
            "services_streaming_tv": rng.choice(["Yes", "No"], n),
            "services_streaming_movies": rng.choice(["Yes", "No"], n),
            "services_streaming_music": rng.choice(["Yes", "No"], n),
            "services_unlimited_data": rng.choice(["Yes", "No"], n),
            "services_contract": contrato,
            "services_paperless_billing": rng.choice(["Yes", "No"], n),
            "services_payment_method": rng.choice(
                ["Bank Withdrawal", "Credit Card", "Mailed Check"], n
            ),
            "services_monthly_charge": mensal,
            "services_total_charges": (mensal * tenure).round(2),
            "services_total_refunds": 0.0,
            "services_total_extra_data_charges": rng.integers(0, 30, n),
        }
    )

    # Alvo correlacionado com o tipo de contrato (mês a mês cancela mais),
    # só para o modelo aprender algo minimamente não aleatório.
    p_churn = np.where(contrato == "Month-to-Month", 0.55, 0.12)
    y = pd.Series((rng.random(n) < p_churn).astype(int), name="status_churn_value")
    return X, y


@pytest.fixture(scope="session")
def campeao_sintetico():
    """Pipeline completo fitado, no mesmo formato do champion_model.joblib real."""
    X, y = _base_contrato_3()
    pipeline = build_pipeline(modelo=LogisticRegression(max_iter=1000, random_state=SEED))
    return pipeline.fit(X, y)


@pytest.fixture
def payload_valido() -> dict:
    """Exemplo de request do Contrato 3 (docs/decisions.md)."""
    return {
        "demographics_gender": "Female",
        "demographics_age": 34,
        "demographics_senior_citizen": "No",
        "demographics_married": "Yes",
        "demographics_dependents": "No",
        "demographics_number_of_dependents": 0,
        "services_number_of_referrals": 0,
        "services_tenure_in_months": 12,
        "services_phone_service": "Yes",
        "services_avg_monthly_long_distance_charges": 15.3,
        "services_multiple_lines": "No",
        "services_internet_type": "Fiber Optic",
        "services_avg_monthly_gb_download": 24,
        "services_online_security": "No",
        "services_online_backup": "Yes",
        "services_device_protection_plan": "No",
        "services_premium_tech_support": "No",
        "services_streaming_tv": "Yes",
        "services_streaming_movies": "No",
        "services_streaming_music": "No",
        "services_unlimited_data": "Yes",
        "services_contract": "Month-to-Month",
        "services_paperless_billing": "Yes",
        "services_payment_method": "Credit Card",
        "services_monthly_charge": 79.85,
        "services_total_charges": 958.2,
        "services_total_refunds": 0.0,
        "services_total_extra_data_charges": 0,
    }
