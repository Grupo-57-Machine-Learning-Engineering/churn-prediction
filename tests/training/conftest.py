"""Fixtures dos testes de treino.

O ponto central é o `mlflow_local`: sobe um tracking store em `tmp_path` e
neutraliza a configuração de DagsHub que o `.env` de quem roda pode ter. Sem
isso, `configurar_mlflow_tracking()` tentaria `dagshub.init()` e o teste
dependeria de rede e credencial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config

VARIAVEIS_DE_TRACKING = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "DAGSHUB_REPO_OWNER",
    "DAGSHUB_REPO_NAME",
)


@pytest.fixture
def sem_dagshub(monkeypatch):
    """Zera a config de tracking remoto lida do `.env`."""
    for atributo in VARIAVEIS_DE_TRACKING:
        monkeypatch.setattr(config, atributo, None)
        monkeypatch.delenv(atributo, raising=False)


@pytest.fixture
def mlflow_local(tmp_path, monkeypatch, sem_dagshub):
    """MLflow apontando para um store de arquivo descartável.

    Devolve o nome do experimento criado, para os testes consultarem as runs
    depois via `mlflow.search_runs`.
    """
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    import mlflow

    # Guarda a URI anterior para devolvê-la no teardown. `set_tracking_uri(None)`
    # não desfaz nada: reseta o MLflow para o padrão, que nesta versão cria um
    # `mlflow.db` na raiz do repositório assim que alguém logar de novo.
    uri_anterior = mlflow.get_tracking_uri()

    store = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file:///{store.as_posix()}")
    experimento = "teste"
    mlflow.set_experiment(experimento)
    monkeypatch.setattr(config, "MLFLOW_EXPERIMENT_NAME", experimento)

    yield experimento

    mlflow.set_tracking_uri(uri_anterior)


@pytest.fixture
def base_pequena() -> tuple[pd.DataFrame, pd.Series]:
    """Base mínima no formato pós-ETL, suficiente para o pipeline rodar.

    Inclui uma coluna descartável (`customer_id`) para o
    `DescartadorDeColunas` ter o que fazer, e um nulo legítimo em
    `services_internet_type`.
    """
    n = 90
    rng = np.random.default_rng(config.SEED)
    contrato = rng.choice(["Month-to-Month", "One Year"], n, p=[0.6, 0.4])
    tenure = rng.integers(1, 72, n)
    mensal = rng.uniform(20, 120, n).round(2)

    X = pd.DataFrame(
        {
            "services_contract": contrato,
            "services_tenure_in_months": tenure,
            "services_monthly_charge": mensal,
            "services_total_charges": (mensal * tenure).round(2),
            "services_internet_type": rng.choice(["Cable", "DSL", None], n),
            "demographics_gender": rng.choice(["Male", "Female"], n),
            "customer_id": [f"c{i}" for i in range(n)],
        }
    )
    y = pd.Series(
        (rng.random(n) < np.where(contrato == "Month-to-Month", 0.6, 0.15)).astype(int),
        name=config.TARGET,
    )
    return X, y
