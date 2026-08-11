"""Testes da preparacao de features.

A fixture reproduz o formato do parquet processado (nomes de coluna,
tipos, nulos legitimos, zeros estruturais, colunas constantes) sem depender
do arquivo real, que nao e versionado. Se o ETL mudar a convencao de nomes,
estes testes quebram -- que e o comportamento desejado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.features import config as cfg
from src.features.preparation import (
    DescartadorDeColunas,
    EngenhariaEstrutural,
    build_pipeline,
    colunas_descartadas,
    filtrar_censura,
    separar_alvo,
)

N = 300
SEED = 57


@pytest.fixture
def base() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    tenure = rng.integers(1, 72, N)
    mensal = rng.uniform(20, 120, N).round(2)
    tem_internet = rng.random(N) > 0.22
    tem_telefone = rng.random(N) > 0.10
    status = rng.choice(["Stayed", "Churned", "Joined"], N, p=[0.66, 0.265, 0.075])

    df = pd.DataFrame(
        {
            # chaves e identificadores
            "customer_id": [f"{i:04d}-ABCD" for i in range(N)],
            "locations_location_id": [f"L{i:05d}" for i in range(N)],
            "services_service_id": [f"S{i:05d}" for i in range(N)],
            "status_status_id": [f"T{i:05d}" for i in range(N)],
            # constantes
            "services_quarter": "Q3",
            "status_quarter": "Q3",
            "locations_state": "California",
            "locations_country": "United States",
            "locations_count": 1,
            "demographics_count": 1,
            "services_count": 1,
            "status_count": 1,
            # geografia
            "locations_city": rng.choice(["Los Angeles", "San Diego", "Fresno"], N),
            "zip_code": rng.integers(90001, 96162, N),
            "locations_latitude": rng.uniform(32.5, 42.0, N),
            "locations_longitude": rng.uniform(-124.4, -114.1, N),
            "populations_population": rng.integers(100, 100_000, N),
            # demografia
            "demographics_gender": rng.choice(["Male", "Female"], N),
            "demographics_age": rng.integers(19, 80, N),
            "demographics_under_30": rng.choice(["Yes", "No"], N),
            "demographics_senior_citizen": rng.choice(["Yes", "No"], N),
            "demographics_married": rng.choice(["Yes", "No"], N),
            "demographics_dependents": rng.choice(["Yes", "No"], N),
            "demographics_number_of_dependents": rng.integers(0, 4, N),
            # servicos
            "services_referred_a_friend": rng.choice(["Yes", "No"], N),
            "services_number_of_referrals": rng.integers(0, 6, N),
            "services_tenure_in_months": tenure,
            "services_offer": rng.choice(
                ["Offer A", "Offer B", "Offer E", None], N, p=[0.15, 0.15, 0.15, 0.55]
            ),
            "services_phone_service": np.where(tem_telefone, "Yes", "No"),
            "services_avg_monthly_long_distance_charges": np.where(
                tem_telefone, rng.uniform(1, 50, N).round(2), 0.0
            ),
            "services_internet_service": np.where(tem_internet, "Yes", "No"),
            "services_internet_type": np.where(
                tem_internet, rng.choice(["Fiber Optic", "Cable", "DSL"], N), None
            ),
            "services_avg_monthly_gb_download": np.where(tem_internet, rng.integers(5, 90, N), 0),
            "services_online_security": rng.choice(["Yes", "No"], N),
            "services_premium_tech_support": rng.choice(["Yes", "No"], N),
            "services_unlimited_data": rng.choice(["Yes", "No"], N),
            "services_contract": rng.choice(
                ["Month-to-Month", "One Year", "Two Year"], N, p=[0.55, 0.22, 0.23]
            ),
            "services_paperless_billing": rng.choice(["Yes", "No"], N),
            "services_payment_method": rng.choice(
                ["Bank Withdrawal", "Credit Card", "Mailed Check"], N
            ),
            "services_monthly_charge": mensal,
            "services_total_refunds": 0.0,
            "services_total_extra_data_charges": rng.integers(0, 30, N),
            # alvo e vazamento
            "status_customer_status": status,
            "status_churn_label": np.where(status == "Churned", "Yes", "No"),
            "status_churn_value": (status == "Churned").astype(int),
            "status_satisfaction_score": np.where(
                status == "Churned", rng.integers(1, 3, N), rng.integers(3, 6, N)
            ),
            "status_churn_score": np.where(
                status == "Churned", rng.integers(60, 101, N), rng.integers(0, 60, N)
            ),
            "status_cltv": rng.integers(2000, 7000, N),
            "status_churn_category": np.where(
                status == "Churned", rng.choice(["Competitor", "Price"], N), None
            ),
            "status_churn_reason": np.where(status == "Churned", "Motivo X", None),
        }
    )

    # identidades contabeis exatas -- reproduzem a colinearidade da base real
    df["services_total_charges"] = (df["services_monthly_charge"] * tenure).round(2)
    df["services_total_long_distance_charges"] = (
        df["services_avg_monthly_long_distance_charges"] * tenure
    ).round(2)
    df["services_total_revenue"] = (
        df["services_total_charges"]
        - df["services_total_refunds"]
        + df["services_total_extra_data_charges"]
        + df["services_total_long_distance_charges"]
    ).round(2)

    return df


# --------------------------------------------------------------------------
# Nivel de linha
# --------------------------------------------------------------------------


def test_filtrar_censura_mantem_joined_por_padrao(base):
    """Padrao mudou para False depois da medicao na base real."""
    modelagem, censurados = filtrar_censura(base)

    assert len(modelagem) == len(base)
    assert censurados.empty


def test_filtrar_censura_separa_joined(base):
    modelagem, censurados = filtrar_censura(base, remover_joined=True)

    assert len(modelagem) + len(censurados) == len(base)
    assert (censurados[cfg.COLUNA_STATUS] == "Joined").all()
    assert "Joined" not in set(modelagem[cfg.COLUNA_STATUS])
    assert len(censurados) > 0, "fixture deveria conter clientes Joined"


def test_separar_alvo(base):
    X, y = separar_alvo(base)

    assert cfg.ALVO not in X.columns
    assert y.dtype.kind == "i"
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)


def test_separar_alvo_sem_coluna(base):
    with pytest.raises(KeyError):
        separar_alvo(base.drop(columns=[cfg.ALVO]))


# --------------------------------------------------------------------------
# Engenharia estrutural
# --------------------------------------------------------------------------


def test_nulos_legitimos_viram_categoria(base):
    saida = EngenhariaEstrutural().fit_transform(base)

    assert saida["services_internet_type"].isna().sum() == 0
    assert "Sem Internet" in set(saida["services_internet_type"])
    assert "Sem Oferta" in set(saida["services_offer"])


def test_flags_de_zero_estrutural(base):
    saida = EngenhariaEstrutural().fit_transform(base)

    sem_internet = saida["services_avg_monthly_gb_download"] == 0
    assert (saida.loc[sem_internet, "flag_sem_internet"] == 1).all()
    assert (saida.loc[~sem_internet, "flag_sem_internet"] == 0).all()
    assert "flag_sem_telefone" in saida.columns


def test_delta_cobranca_zero_quando_nao_ha_reajuste(base):
    """Na fixture total_charges == monthly * tenure por construcao."""
    saida = EngenhariaEstrutural().fit_transform(base)

    assert np.allclose(saida["delta_cobranca"], 0, atol=0.01)


def test_delta_cobranca_captura_reajuste(base):
    base = base.copy()
    base.loc[0, "services_total_charges"] += 500.0

    saida = EngenhariaEstrutural().fit_transform(base)

    assert saida.loc[0, "delta_cobranca"] == pytest.approx(500.0, abs=0.01)
    assert saida.loc[0, "delta_cobranca_pct"] > 0


def test_delta_cobranca_nao_estoura_com_tenure_zero(base):
    base = base.copy()
    base.loc[0, "services_tenure_in_months"] = 0

    saida = EngenhariaEstrutural().fit_transform(base)

    assert np.isfinite(saida["delta_cobranca_pct"].dropna()).all()


def test_transformador_nao_muta_entrada(base):
    antes = base.copy()
    EngenhariaEstrutural().fit_transform(base)

    pd.testing.assert_frame_equal(base, antes)


# --------------------------------------------------------------------------
# Descarte
# --------------------------------------------------------------------------


def test_descarte_remove_todas_as_categorias(base):
    alvo_removido = colunas_descartadas()
    saida = DescartadorDeColunas(alvo_removido).fit_transform(base)

    assert not set(saida.columns) & set(alvo_removido)


def test_nenhuma_coluna_de_vazamento_sobrevive(base):
    saida = DescartadorDeColunas(colunas_descartadas()).fit_transform(base)

    proibidas = set(cfg.COLS_VAZAMENTO) | set(cfg.COLS_DERIVADAS_DO_ALVO)
    assert not set(saida.columns) & proibidas


def test_descarte_tolera_colunas_ausentes(base):
    """O payload do POST /predict nao carrega colunas status_*."""
    reduzida = base.drop(columns=[c for c in base.columns if c.startswith("status_")])
    descartador = DescartadorDeColunas(colunas_descartadas()).fit(reduzida)

    assert descartador.transform(reduzida) is not None
    assert set(cfg.COLS_VAZAMENTO) <= set(descartador.colunas_ausentes_)


def test_flags_de_configuracao(base):
    assert "services_offer" in colunas_descartadas()
    assert "services_offer" not in colunas_descartadas(incluir_offer=True)
    assert "locations_city" in colunas_descartadas()
    assert "locations_city" not in colunas_descartadas(incluir_geografia=True)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def test_pipeline_produz_matriz_numerica(base):
    modelagem, _ = filtrar_censura(base)
    X, y = separar_alvo(modelagem)

    matriz = build_pipeline().fit_transform(X, y)

    assert matriz.shape[0] == len(X)
    assert matriz.shape[1] > 0
    assert np.isfinite(matriz).all(), "restaram NaN ou inf apos a preparacao"


def test_pipeline_nao_vaza_entre_treino_e_teste(base):
    """O scaler so pode ver estatisticas do treino (bug fix/scaler-leak)."""
    modelagem, _ = filtrar_censura(base)
    X, y = separar_alvo(modelagem)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    pipe = build_pipeline().fit(X_tr, y_tr)
    escala = pipe.named_steps["preprocessamento"].named_transformers_["num"]
    medias_treino = escala.named_steps["escala"].mean_

    pipe_completo = build_pipeline().fit(X, y)
    medias_completas = (
        pipe_completo.named_steps["preprocessamento"]
        .named_transformers_["num"]
        .named_steps["escala"]
        .mean_
    )

    assert not np.allclose(
        medias_treino, medias_completas
    ), "as estatisticas do treino coincidem com as da base inteira -- indicio de fit fora do fold"
    assert pipe.transform(X_te).shape[0] == len(X_te)


def test_pipeline_com_estimador_treina(base):
    modelagem, _ = filtrar_censura(base)
    X, y = separar_alvo(modelagem)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED))
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    assert proba.shape == (len(X_te),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert len(y_te) == len(proba)


def test_pipeline_aceita_linha_unica_apos_fit(base):
    """Formato do POST /predict: um cliente por vez."""
    modelagem, _ = filtrar_censura(base)
    X, y = separar_alvo(modelagem)

    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)).fit(X, y)
    proba = pipe.predict_proba(X.iloc[[0]])[:, 1]

    assert proba.shape == (1,)


def test_pipeline_lida_com_categoria_nova(base):
    """handle_unknown deve absorver valores nao vistos no treino."""
    modelagem, _ = filtrar_censura(base)
    X, y = separar_alvo(modelagem)
    pipe = build_pipeline().fit(X, y)

    novo = X.iloc[[0]].copy()
    novo.loc[:, "services_payment_method"] = "Pix"

    assert pipe.transform(novo).shape[0] == 1
