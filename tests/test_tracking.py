"""Testes de src.config.configurar_mlflow_tracking (setup do MLflow/DagsHub)."""

from unittest.mock import MagicMock

import pytest

from src import config


def test_nao_chama_dagshub_quando_opcao_a_completa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com URI + usuário + senha preenchidos (Opção A), não deve tentar dagshub.init."""
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", "https://dagshub.com/x/y.mlflow")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_USERNAME", "usuario")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_PASSWORD", "token")

    dagshub_mock = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "dagshub", dagshub_mock)

    config.configurar_mlflow_tracking()

    dagshub_mock.init.assert_not_called()


def test_chama_dagshub_init_quando_opcao_a_incompleta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Faltando qualquer um dos 3 campos, cai para dagshub.init (Opção B)."""
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", None)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_USERNAME", None)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_PASSWORD", None)
    monkeypatch.setattr(config, "DAGSHUB_REPO_OWNER", "dono")
    monkeypatch.setattr(config, "DAGSHUB_REPO_NAME", "repo")

    dagshub_mock = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "dagshub", dagshub_mock)

    config.configurar_mlflow_tracking()

    dagshub_mock.init.assert_called_once_with(repo_owner="dono", repo_name="repo", mlflow=True)


def test_falha_do_dagshub_nao_propaga(monkeypatch: pytest.MonkeyPatch) -> None:
    """Se dagshub.init lançar (rede indisponível, etc.), é best-effort: não deve propagar."""
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", None)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_USERNAME", None)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_PASSWORD", None)
    monkeypatch.setattr(config, "DAGSHUB_REPO_OWNER", "dono")
    monkeypatch.setattr(config, "DAGSHUB_REPO_NAME", "repo")

    dagshub_mock = MagicMock()
    dagshub_mock.init.side_effect = RuntimeError("sem rede")
    monkeypatch.setitem(__import__("sys").modules, "dagshub", dagshub_mock)

    config.configurar_mlflow_tracking()  # não deve lançar


def test_nao_chama_dagshub_sem_repo_owner_e_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem nenhuma das duas opções configuradas, só avisa -- sem default silencioso."""
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", None)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_USERNAME", None)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_PASSWORD", None)
    monkeypatch.setattr(config, "DAGSHUB_REPO_OWNER", None)
    monkeypatch.setattr(config, "DAGSHUB_REPO_NAME", None)

    dagshub_mock = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "dagshub", dagshub_mock)

    config.configurar_mlflow_tracking()

    dagshub_mock.init.assert_not_called()
