"""Testes de src.config: setup do MLflow/DagsHub e utilitarios de run."""

from unittest.mock import MagicMock

import pandas as pd
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


def test_limpar_runs_anteriores_deleta_runs_encontradas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs existentes com os nomes passados devem ser soft-deletadas uma a uma."""
    mlflow_mock = MagicMock()
    mlflow_mock.search_runs.return_value = pd.DataFrame({"run_id": ["r1", "r2"]})
    client_mock = MagicMock()
    tracking_mock = MagicMock(MlflowClient=MagicMock(return_value=client_mock))
    monkeypatch.setitem(__import__("sys").modules, "mlflow", mlflow_mock)
    monkeypatch.setitem(__import__("sys").modules, "mlflow.tracking", tracking_mock)

    config.limpar_runs_anteriores(["baseline_logistic_regression"])

    assert client_mock.delete_run.call_count == 2
    client_mock.delete_run.assert_any_call("r1")
    client_mock.delete_run.assert_any_call("r2")


def test_limpar_runs_anteriores_sem_runs_nao_deleta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem runs anteriores com esses nomes, não deve chamar delete_run."""
    mlflow_mock = MagicMock()
    mlflow_mock.search_runs.return_value = pd.DataFrame({"run_id": []})
    client_mock = MagicMock()
    tracking_mock = MagicMock(MlflowClient=MagicMock(return_value=client_mock))
    monkeypatch.setitem(__import__("sys").modules, "mlflow", mlflow_mock)
    monkeypatch.setitem(__import__("sys").modules, "mlflow.tracking", tracking_mock)

    config.limpar_runs_anteriores(["algum_run_inexistente"])

    client_mock.delete_run.assert_not_called()


def test_limpar_runs_anteriores_lista_vazia_nao_busca(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lista vazia deve retornar cedo, sem chamar search_runs (filtro vazio == sem filtro)."""
    mlflow_mock = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", mlflow_mock)

    config.limpar_runs_anteriores([])

    mlflow_mock.search_runs.assert_not_called()


def test_iniciar_run_fixa_source_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """iniciar_run deve fixar mlflow.source.name e devolver o contexto do start_run."""
    mlflow_mock = MagicMock()
    contexto_esperado = MagicMock()
    mlflow_mock.start_run.return_value = contexto_esperado
    monkeypatch.setitem(__import__("sys").modules, "mlflow", mlflow_mock)

    resultado = config.iniciar_run("notebooks/05_modelagem.ipynb", run_name="minha_run")

    mlflow_mock.start_run.assert_called_once_with(run_name="minha_run")
    mlflow_mock.set_tag.assert_called_once_with(
        "mlflow.source.name", "notebooks/05_modelagem.ipynb"
    )
    assert resultado is contexto_esperado


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
