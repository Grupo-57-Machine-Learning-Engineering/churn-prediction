"""Testes do módulo de predição (`src/models/predict.py`)."""

from __future__ import annotations

import joblib
import pandas as pd
import pytest

from src import config
from src.models.predict import (
    ORIGEM_LOCAL,
    Campeao,
    ModeloIndisponivelError,
    carregar_campeao,
    prever,
)


def _sem_mlflow(monkeypatch):
    """Desliga a fonte MLflow, mesmo que o .env local esteja preenchido.

    Sem isso, quem tem `.env` configurado testa contra o registry de verdade:
    o MLflow é a primeira fonte, então ele responderia antes do joblib.
    """
    for atributo in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "DAGSHUB_REPO_OWNER",
        "DAGSHUB_REPO_NAME",
    ):
        monkeypatch.setattr(config, atributo, None)


def test_carregar_campeao_do_joblib(tmp_path, campeao_sintetico, payload_valido, monkeypatch):
    """Fluxo completo pelo fallback: salva o joblib, carrega de volta e prevê."""
    _sem_mlflow(monkeypatch)
    caminho = tmp_path / "champion_model.joblib"
    joblib.dump(campeao_sintetico, caminho)

    campeao = carregar_campeao(caminho)
    resultado = prever(campeao.modelo, pd.DataFrame([payload_valido]))

    assert campeao.origem == ORIGEM_LOCAL
    assert len(resultado) == 1
    assert 0.0 <= resultado[0]["probability"] <= 1.0
    assert isinstance(resultado[0]["churn"], bool)


def test_carregar_campeao_sem_fonte_disponivel(tmp_path, monkeypatch):
    _sem_mlflow(monkeypatch)

    with pytest.raises(ModeloIndisponivelError):
        carregar_campeao(tmp_path / "nao_existe.joblib")


def test_carregar_campeao_prioriza_mlflow_sobre_joblib_local(
    tmp_path, campeao_sintetico, monkeypatch
):
    """Com as duas fontes disponíveis, o MLflow ganha: é a fonte de verdade do
    campeão atual, e o joblib local pode estar desatualizado. Isso é o que
    garante que reiniciar a API depois de promover um campeão novo já serve
    o modelo certo, sem precisar sincronizar nenhum arquivo manualmente.
    """
    caminho = tmp_path / "champion_model.joblib"
    joblib.dump("modelo_local_desatualizado", caminho)

    do_registry = Campeao(campeao_sintetico, "mlflow:churn_champion/7")
    monkeypatch.setattr("src.models.predict._carregar_do_mlflow", lambda: do_registry)

    campeao = carregar_campeao(caminho)

    assert campeao.modelo is campeao_sintetico
    assert campeao.origem == "mlflow:churn_champion/7"


def test_prever_respeita_threshold(campeao_sintetico, payload_valido):
    dados = pd.DataFrame([payload_valido])

    sempre_churn = prever(campeao_sintetico, dados, threshold=0.0)
    nunca_churn = prever(campeao_sintetico, dados, threshold=1.01)

    assert sempre_churn[0]["churn"] is True
    assert nunca_churn[0]["churn"] is False
    assert sempre_churn[0]["probability"] == nunca_churn[0]["probability"]


def test_prever_varias_linhas(campeao_sintetico, payload_valido):
    dados = pd.DataFrame([payload_valido] * 3)

    resultado = prever(campeao_sintetico, dados)

    assert len(resultado) == 3
    probabilidades = {r["probability"] for r in resultado}
    assert len(probabilidades) == 1, "linhas idênticas devem ter a mesma probabilidade"
