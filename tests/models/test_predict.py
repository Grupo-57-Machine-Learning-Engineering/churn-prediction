"""Testes do módulo de predição.

O `models/champion_model.joblib` não é versionado, então nenhum teste pode
depender dele: o CI não o tem. No lugar, `campeao_sintetico` treina em
segundos um pipeline com a mesma `build_pipeline` de produção sobre uma base
sintética, e é ele que os testes carregam e usam.
"""

from __future__ import annotations

import types

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config
from src.features.preparation import build_pipeline
from src.models.predict import (
    THRESHOLD_PADRAO,
    ModeloIndisponivelError,
    caminho_padrao_do_campeao,
    carregar_campeao,
    prever,
    prever_um,
)

N = 200


@pytest.fixture(scope="module")
def campeao_sintetico():
    """Pipeline fitado, no mesmo formato do champion_model.joblib real."""
    rng = np.random.default_rng(config.SEED)
    contrato = rng.choice(["Month-to-Month", "One Year", "Two Year"], N, p=[0.6, 0.2, 0.2])
    tenure = rng.integers(1, 72, N)
    mensal = rng.uniform(20, 120, N).round(2)

    X = pd.DataFrame(
        {
            "services_contract": contrato,
            "services_tenure_in_months": tenure,
            "services_monthly_charge": mensal,
            "services_total_charges": (mensal * tenure).round(2),
            "services_internet_type": rng.choice(["Cable", "DSL", None], N),
            "demographics_gender": rng.choice(["Male", "Female"], N),
            "demographics_age": rng.integers(19, 80, N),
        }
    )
    y = pd.Series((rng.random(N) < np.where(contrato == "Month-to-Month", 0.55, 0.12)).astype(int))

    pipeline = build_pipeline(modelo=LogisticRegression(max_iter=1000, random_state=config.SEED))
    return pipeline.fit(X, y), X


@pytest.fixture
def modelo(campeao_sintetico):
    return campeao_sintetico[0]


@pytest.fixture
def entrada(campeao_sintetico):
    return campeao_sintetico[1].head(5)


# --------------------------------------------------------------------------
# carregar_campeao
# --------------------------------------------------------------------------


def test_carrega_do_joblib_local(tmp_path, modelo, entrada):
    caminho = tmp_path / "champion_model.joblib"
    joblib.dump(modelo, caminho)

    carregado = carregar_campeao(caminho)

    assert np.allclose(carregado.predict_proba(entrada), modelo.predict_proba(entrada))


def test_sem_fonte_disponivel_levanta_erro_com_instrucao(tmp_path, monkeypatch):
    """Sem joblib e sem MLflow configurado, o erro tem que dizer o que fazer."""
    for atributo in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "DAGSHUB_REPO_OWNER",
        "DAGSHUB_REPO_NAME",
    ):
        monkeypatch.setattr(config, atributo, None)

    with pytest.raises(ModeloIndisponivelError, match="05_modelagem"):
        carregar_campeao(tmp_path / "nao_existe.joblib")


def test_caminho_padrao_sai_de_config():
    caminho = caminho_padrao_do_campeao()

    assert caminho.parent == config.MODELS_DIR
    assert caminho.name == "champion_model.joblib"


# --------------------------------------------------------------------------
# prever
# --------------------------------------------------------------------------


def test_prever_devolve_uma_linha_por_cliente(modelo, entrada):
    resultado = prever(modelo, entrada)

    assert len(resultado) == len(entrada)
    assert all(set(r) == {"probability", "churn"} for r in resultado)
    assert all(0.0 <= r["probability"] <= 1.0 for r in resultado)
    assert all(isinstance(r["churn"], bool) for r in resultado)


def test_classe_bate_com_o_predict_do_pipeline(modelo, entrada):
    """0,5 é o corte implícito do notebook, que avalia com pipeline.predict()."""
    resultado = prever(modelo, entrada)

    esperado = modelo.predict(entrada).astype(bool)
    assert [r["churn"] for r in resultado] == list(esperado)


def test_probabilidade_bate_com_o_predict_proba(modelo, entrada):
    resultado = prever(modelo, entrada)

    esperado = modelo.predict_proba(entrada)[:, 1]
    assert np.allclose([r["probability"] for r in resultado], esperado)


def test_threshold_customizado_muda_so_a_classe(modelo, entrada):
    sempre = prever(modelo, entrada, threshold=0.0)
    nunca = prever(modelo, entrada, threshold=1.01)

    assert all(r["churn"] for r in sempre)
    assert not any(r["churn"] for r in nunca)
    assert [r["probability"] for r in sempre] == [r["probability"] for r in nunca]


def test_threshold_padrao_e_o_do_projeto(modelo, entrada):
    assert prever(modelo, entrada) == prever(modelo, entrada, threshold=THRESHOLD_PADRAO)


def test_dataframe_vazio_levanta_erro(modelo, entrada):
    with pytest.raises(ValueError, match="vazio"):
        prever(modelo, entrada.iloc[0:0])


def test_prever_um_aceita_dict(modelo, entrada):
    cliente = entrada.iloc[0].to_dict()

    resultado = prever_um(modelo, cliente)

    assert set(resultado) == {"probability", "churn"}
    assert resultado["probability"] == pytest.approx(prever(modelo, entrada)[0]["probability"])


def test_ordem_das_linhas_e_preservada(modelo, entrada):
    invertida = entrada.iloc[::-1]

    resultado = prever(modelo, invertida)

    esperado = modelo.predict_proba(invertida)[:, 1]
    assert np.allclose([r["probability"] for r in resultado], esperado)


# --------------------------------------------------------------------------
# Fallback via Model Registry
# --------------------------------------------------------------------------


def _desligar_tracking(monkeypatch):
    for atributo in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "DAGSHUB_REPO_OWNER",
        "DAGSHUB_REPO_NAME",
    ):
        monkeypatch.setattr(config, atributo, None)


def test_fallback_nao_e_tentado_sem_tracking_configurado(tmp_path, monkeypatch):
    """Sem .env preenchido nao adianta tentar o registry: evita espera inutil."""
    from src.models import predict

    _desligar_tracking(monkeypatch)
    chamou = []
    monkeypatch.setattr(predict.config, "configurar_mlflow_tracking", lambda: chamou.append(1))

    with pytest.raises(ModeloIndisponivelError):
        carregar_campeao(tmp_path / "nao_existe.joblib")

    assert chamou == [], "nem deve configurar o tracking"


def test_fallback_baixa_do_registry_quando_configurado(tmp_path, monkeypatch, modelo):
    """Caminho de quem nao rodou o notebook 05 mas tem o .env preenchido."""
    from src.models import predict

    _desligar_tracking(monkeypatch)
    monkeypatch.setattr(config, "DAGSHUB_REPO_OWNER", "grupo")
    monkeypatch.setattr(config, "DAGSHUB_REPO_NAME", "repo")
    monkeypatch.setattr(predict.config, "configurar_mlflow_tracking", lambda: None)

    uris = []

    def _load_model(uri):
        uris.append(uri)
        return modelo

    # Troca o submodulo inteiro no pacote `mlflow`. Substituir so o atributo
    # `load_model` nao pega: dentro de `_carregar_do_registry` o
    # `import mlflow.sklearn` resolve o submodulo pelo pacote.
    import mlflow
    import mlflow.sklearn  # noqa: F401  garante que ja esta carregado

    monkeypatch.setattr(mlflow, "sklearn", types.SimpleNamespace(load_model=_load_model))

    carregado = carregar_campeao(tmp_path / "nao_existe.joblib")

    assert carregado is modelo
    assert uris == [predict.URI_CAMPEAO]
    assert uris[0] == "models:/churn_champion@champion"


def test_falha_no_registry_vira_erro_com_instrucao(tmp_path, monkeypatch):
    """A API precisa subir mesmo assim, entao o erro tem que ser tratavel."""
    from src.models import predict

    _desligar_tracking(monkeypatch)
    monkeypatch.setattr(config, "DAGSHUB_REPO_OWNER", "grupo")
    monkeypatch.setattr(config, "DAGSHUB_REPO_NAME", "repo")

    def _explodir():
        raise RuntimeError("dagshub fora do ar")

    monkeypatch.setattr(predict.config, "configurar_mlflow_tracking", _explodir)

    with pytest.raises(ModeloIndisponivelError, match=r"\.env"):
        carregar_campeao(tmp_path / "nao_existe.joblib")


def test_joblib_local_tem_prioridade_sobre_o_registry(tmp_path, monkeypatch, modelo):
    """Ordem do ADR-006: local primeiro, porque funciona sem rede."""
    from src.models import predict

    caminho = tmp_path / "champion_model.joblib"
    joblib.dump(modelo, caminho)

    def _nao_deveria_ser_chamado():
        raise AssertionError("o registry nao pode ser consultado com joblib presente")

    monkeypatch.setattr(predict, "_carregar_do_registry", _nao_deveria_ser_chamado)

    assert carregar_campeao(caminho) is not None
