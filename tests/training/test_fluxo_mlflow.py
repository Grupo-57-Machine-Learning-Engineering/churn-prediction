"""Testes das funções que escrevem ou leem MLflow.

Todas rodam contra um store de arquivo em `tmp_path` (fixture `mlflow_local`),
sem rede e sem credencial. O objetivo é garantir que o fluxo principal da
Etapa 2 roda de ponta a ponta e produz as runs, os params e as métricas que o
resto do projeto espera encontrar, não reavaliar os números: paridade com o
notebook é verificada à parte.

Os orçamentos são mínimos de propósito (grade de 2 combinações, 2 trials de
Optuna, 2 folds), porque o que está sob teste é o encanamento.
"""

from __future__ import annotations

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config
from src.features.preparation import build_pipeline
from src.training.baseline import criar_baseline, registrar_baseline
from src.training.champion import (
    Campeao,
    registrar_campeao,
    resolver_pipeline_campeao,
)
from src.training.comparison import (
    COLUNAS_TABELA,
    NOME_RUN_BASELINE,
    buscar_baseline,
    montar_tabela_comparativa,
    registrar_comparacao,
)
from src.training.estimators import EspacoDeBusca, criar_logistic_regression
from src.training.metrics import avaliar_por_cv, calcular_metricas
from src.training.tuning import ResultadoTuning, rodar_tuning, silenciar_optuna

CANDIDATO = "regressao_teste"

ESPACO_MINIMO = EspacoDeBusca(
    grid={"modelo__C": [0.1, 1.0]},
    distribuicoes={"modelo__C": [0.1, 1.0, 10.0]},
    sugerir=lambda trial: {"C": trial.suggest_float("C", 0.1, 10.0, log=True)},
    reconstruir=dict,
)


# --------------------------------------------------------------------------
# tuning.rodar_tuning
# --------------------------------------------------------------------------


@pytest.fixture
def resultados_tuning(mlflow_local, base_pequena) -> dict[str, ResultadoTuning]:
    """Roda o fluxo de tuning uma vez e reaproveita nos testes do módulo."""
    silenciar_optuna()
    X, y = base_pequena
    return rodar_tuning(
        CANDIDATO,
        criar_logistic_regression,
        ESPACO_MINIMO,
        X,
        y,
        n_trials_optuna=2,
        n_splits=2,
        fonte="tests/training/test_fluxo_mlflow.py",
    )


def test_rodar_tuning_devolve_os_tres_metodos(resultados_tuning):
    assert set(resultados_tuning) == {"grid_search", "random_search", "optuna"}


def test_rodar_tuning_preenche_o_resultado_de_cada_metodo(resultados_tuning):
    for metodo, resultado in resultados_tuning.items():
        assert resultado.metodo == metodo
        assert 0.0 <= resultado.pr_auc_mean <= 1.0, metodo
        assert resultado.pr_auc_std >= 0.0, metodo
        assert 0.0 <= resultado.f1_mean <= 1.0, metodo
        assert 0.0 <= resultado.auc_roc_mean <= 1.0, metodo
        assert resultado.tempo_segundos > 0.0, metodo
        assert resultado.params, metodo


def test_pipeline_de_cada_metodo_sai_fitado_e_pronto_para_prever(resultados_tuning, base_pequena):
    """O campeão é escolhido sem refit, então o pipeline tem que vir treinado."""
    X, _ = base_pequena

    for metodo, resultado in resultados_tuning.items():
        proba = resultado.pipeline.predict_proba(X.head(3))[:, 1]
        assert proba.shape == (3,), metodo
        assert ((proba >= 0.0) & (proba <= 1.0)).all(), metodo


def test_rodar_tuning_cria_uma_run_pai_e_tres_filhas(resultados_tuning, mlflow_local):
    runs = mlflow.search_runs(experiment_names=[mlflow_local], max_results=100)
    nomes = set(runs["tags.mlflow.runName"])

    assert CANDIDATO in nomes
    for metodo in ("grid_search", "random_search", "optuna"):
        assert f"{CANDIDATO}__{metodo}" in nomes

    filhas = runs[runs["tags.mlflow.runName"] != CANDIDATO]
    assert filhas["tags.mlflow.parentRunId"].notna().all()


def test_run_pai_registra_o_metodo_vencedor(resultados_tuning, mlflow_local):
    runs = mlflow.search_runs(experiment_names=[mlflow_local], max_results=100)
    pai = runs[runs["tags.mlflow.runName"] == CANDIDATO].iloc[0]

    assert pai["params.melhor_metodo"] in resultados_tuning
    esperado = max(resultados_tuning.values(), key=lambda r: r.pr_auc_mean).pr_auc_mean
    assert pai["metrics.melhor_pr_auc_mean"] == pytest.approx(esperado)


def test_run_do_optuna_registra_trials_e_o_desvio_entre_folds(resultados_tuning, mlflow_local):
    runs = mlflow.search_runs(experiment_names=[mlflow_local], max_results=100)
    optuna_run = runs[runs["tags.mlflow.runName"] == f"{CANDIDATO}__optuna"].iloc[0]

    assert optuna_run["params.n_trials"] == "2"
    assert int(optuna_run["params.n_trials_podados"]) >= 0
    # Métrica extra do ADR-008, comparável com o std_test_pr_auc das outras buscas.
    assert float(optuna_run["params.pr_auc_std_entre_folds"]) >= 0.0


def test_rodar_tuning_com_sample_weight(mlflow_local, base_pequena):
    """Caminho do candidato balanceado por peso, fatiado por fold."""
    from sklearn.utils.class_weight import compute_sample_weight

    silenciar_optuna()
    X, y = base_pequena
    pesos = compute_sample_weight("balanced", y)

    resultados = rodar_tuning(
        "com_peso",
        criar_logistic_regression,
        ESPACO_MINIMO,
        X,
        y,
        sample_weight=pesos,
        n_trials_optuna=2,
        n_splits=2,
        fonte="teste.py",
    )

    assert set(resultados) == {"grid_search", "random_search", "optuna"}
    assert all(r.pipeline is not None for r in resultados.values())


def test_n_iter_random_limita_o_sorteio(mlflow_local, base_pequena):
    silenciar_optuna()
    X, y = base_pequena

    resultados = rodar_tuning(
        "n_iter",
        criar_logistic_regression,
        ESPACO_MINIMO,
        X,
        y,
        n_iter_random=1,
        n_trials_optuna=2,
        n_splits=2,
        fonte="teste.py",
    )

    assert resultados["random_search"].pipeline is not None


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def test_montar_tabela_le_as_runs_do_tuning(resultados_tuning, mlflow_local):
    tabela = montar_tabela_comparativa(candidatos=(CANDIDATO,), experimento=mlflow_local)

    assert list(tabela.columns) == list(COLUNAS_TABELA)
    assert len(tabela) == 3
    assert set(tabela["metodo"]) == {"grid_search", "random_search", "optuna"}
    assert tabela["pr_auc_mean"].is_monotonic_decreasing, "tem que sair ordenada"


def test_montar_tabela_inclui_a_linha_do_baseline(resultados_tuning, mlflow_local):
    baseline = {
        "cv_pr_auc": 0.99,
        "cv_f1": 0.9,
        "cv_auc_roc": 0.95,
        "tempo_segundos": 1.0,
    }

    tabela = montar_tabela_comparativa(
        candidatos=(CANDIDATO,), baseline=baseline, experimento=mlflow_local
    )

    assert len(tabela) == 4
    assert tabela.iloc[0]["candidato"] == NOME_RUN_BASELINE, "0.99 tem que liderar"


def test_registrar_comparacao_salva_csv_e_devolve_o_vencedor(
    resultados_tuning, mlflow_local, tmp_path
):
    tabela = montar_tabela_comparativa(candidatos=(CANDIDATO,), experimento=mlflow_local)
    destino = tmp_path / "sub" / "tabela.csv"

    melhor = registrar_comparacao(tabela, caminho_csv=destino, fonte="teste.py")

    assert destino.exists()
    assert pd.read_csv(destino).shape == tabela.shape
    assert melhor["candidato"] == CANDIDATO

    runs = mlflow.search_runs(experiment_names=[mlflow_local], max_results=100)
    fechamento = runs[runs["tags.mlflow.runName"] == "etapa2_comparacao_final"].iloc[0]
    assert fechamento["params.metodo_vencedor"] == melhor["metodo"]


# --------------------------------------------------------------------------
# baseline
# --------------------------------------------------------------------------


def test_registrar_baseline_cria_a_run_que_buscar_baseline_encontra(mlflow_local, base_pequena):
    """Contrato de ida e volta entre os dois notebooks, 04 e 05."""
    X, y = base_pequena
    pipeline = build_pipeline(modelo=criar_baseline()).fit(X, y)
    metricas_cv = avaliar_por_cv(pipeline, X, y, n_splits=2)
    metricas_teste = calcular_metricas(y, pipeline.predict(X), pipeline.predict_proba(X)[:, 1])

    fallback = registrar_baseline(pipeline, metricas_cv, metricas_teste, fonte="teste.py")

    assert fallback is None, "com MLflow no ar não deve cair para o joblib"

    recuperado = buscar_baseline(experimento=mlflow_local)
    assert recuperado is not None
    assert recuperado["cv_f1"] == pytest.approx(metricas_cv["cv_f1"])
    assert recuperado["teste_pr_auc"] == pytest.approx(metricas_teste["pr_auc"])
    assert recuperado["run_id"]
    assert recuperado["tempo_segundos"] >= 0


def test_registrar_baseline_cai_para_joblib_quando_o_mlflow_falha(
    mlflow_local, base_pequena, tmp_path, monkeypatch
):
    X, y = base_pequena
    pipeline = build_pipeline(modelo=criar_baseline()).fit(X, y)

    def explodir(*args, **kwargs):
        raise RuntimeError("tracking server fora do ar")

    monkeypatch.setattr(mlflow, "set_experiment", explodir)
    destino = tmp_path / "baseline.joblib"

    caminho = registrar_baseline(
        pipeline, {"cv_f1": 0.5}, {"f1": 0.5}, caminho_fallback=destino, fonte="teste.py"
    )

    assert caminho == destino
    assert joblib.load(destino).predict(X.head(2)).shape == (2,)


def test_params_do_baseline_saem_do_pipeline_quando_nao_informados(mlflow_local, base_pequena):
    X, y = base_pequena
    pipeline = build_pipeline(modelo=criar_baseline()).fit(X, y)

    registrar_baseline(pipeline, {"cv_f1": 0.5}, {"f1": 0.5}, fonte="teste.py")

    runs = mlflow.search_runs(experiment_names=[mlflow_local], max_results=10)
    run = runs[runs["tags.mlflow.runName"] == NOME_RUN_BASELINE].iloc[0]
    assert run["params.modelo"] == "LogisticRegression"
    assert run["params.class_weight"] == "balanced"


# --------------------------------------------------------------------------
# champion
# --------------------------------------------------------------------------


def test_resolver_pipeline_devolve_o_do_campeao_quando_existe(base_pequena):
    X, y = base_pequena
    pipeline = build_pipeline(modelo=criar_baseline()).fit(X, y)
    campeao = Campeao(candidato="rf", metodo="optuna", pipeline=pipeline)

    assert resolver_pipeline_campeao(campeao) is pipeline


def test_resolver_pipeline_carrega_o_joblib_do_baseline(base_pequena, tmp_path):
    """Caminho de quando o baseline vence: não há retreino, só recuperação."""
    X, y = base_pequena
    pipeline = build_pipeline(modelo=criar_baseline()).fit(X, y)
    origem = tmp_path / "baseline_logistic_regression.joblib"
    joblib.dump(pipeline, origem)
    campeao = Campeao(candidato=NOME_RUN_BASELINE, metodo="n/a", pipeline=None)

    recuperado = resolver_pipeline_campeao(campeao, caminho_baseline=origem)

    assert np.allclose(recuperado.predict_proba(X.head(3)), pipeline.predict_proba(X.head(3)))


def test_resolver_pipeline_sem_nenhuma_fonte_levanta_erro(tmp_path):
    campeao = Campeao(candidato=NOME_RUN_BASELINE, metodo="n/a", pipeline=None)

    with pytest.raises(FileNotFoundError, match="04_baseline"):
        resolver_pipeline_campeao(campeao, caminho_baseline=tmp_path / "nao_existe.joblib")


def test_registrar_campeao_loga_metricas_e_devolve_a_versao(mlflow_local, base_pequena):
    X, y = base_pequena
    pipeline = build_pipeline(modelo=criar_baseline()).fit(X, y)
    campeao = Campeao(
        candidato=CANDIDATO,
        metodo="optuna",
        pipeline=pipeline,
        metricas_teste={"f1": 0.7, "auc_roc": 0.8, "pr_auc": 0.75},
    )

    versao = registrar_campeao(
        pipeline,
        campeao,
        metricas_negocio={"sensibilidade": 0.81, "precisao": 0.62},
        metricas_cv={"pr_auc_mean": 0.74, "f1_mean": 0.68, "auc_roc_mean": 0.89},
        fonte="teste.py",
    )

    assert versao is not None, "o file store local suporta Model Registry"

    runs = mlflow.search_runs(experiment_names=[mlflow_local], max_results=10)
    run = runs[runs["tags.mlflow.runName"] == "etapa2_campeao_final"].iloc[0]
    assert run["params.candidato_vencedor"] == CANDIDATO
    assert run["metrics.teste_f1"] == pytest.approx(0.7)
    assert run["metrics.teste_sensibilidade"] == pytest.approx(0.81)
    assert run["metrics.cv_pr_auc_mean"] == pytest.approx(0.74)


def test_registrar_campeao_nao_derruba_o_treino_se_o_mlflow_falhar(
    mlflow_local, base_pequena, monkeypatch
):
    """Best-effort: treino que já terminou não pode ser perdido por erro de rede."""
    X, y = base_pequena
    pipeline = build_pipeline(modelo=criar_baseline()).fit(X, y)
    campeao = Campeao(candidato=CANDIDATO, metodo="optuna", metricas_teste={"f1": 0.7})

    def explodir(*args, **kwargs):
        raise RuntimeError("registry indisponível")

    monkeypatch.setattr("mlflow.sklearn.log_model", explodir)

    assert registrar_campeao(pipeline, campeao, fonte="teste.py") is None


def test_registrar_campeao_sem_metricas_de_cv(mlflow_local, base_pequena):
    """Acontece quando o baseline vence sem métricas recuperadas."""
    X, y = base_pequena
    pipeline = build_pipeline(modelo=criar_baseline()).fit(X, y)
    campeao = Campeao(candidato=NOME_RUN_BASELINE, metodo="n/a", metricas_teste={"f1": 0.5})

    versao = registrar_campeao(pipeline, campeao, metricas_cv={}, fonte="teste.py")

    assert versao is not None


# --------------------------------------------------------------------------
# Fluxo completo, na ordem do notebook 05
# --------------------------------------------------------------------------


def test_etapa2_de_ponta_a_ponta(mlflow_local, base_pequena, tmp_path):
    """Tuning, comparação, seleção, avaliação e registro, sem erro."""
    from src.training.champion import (
        metricas_cv_do_campeao,
        salvar_campeao,
        selecionar_campeao,
    )
    from src.training.metrics import calcular_metricas_negocio

    silenciar_optuna()
    X, y = base_pequena
    X_train, X_test = X.iloc[:60], X.iloc[60:]
    y_train, y_test = y.iloc[:60], y.iloc[60:]

    resultados = rodar_tuning(
        CANDIDATO,
        criar_logistic_regression,
        ESPACO_MINIMO,
        X_train,
        y_train,
        n_trials_optuna=2,
        n_splits=2,
        fonte="teste.py",
    )

    tabela = montar_tabela_comparativa(candidatos=(CANDIDATO,), experimento=mlflow_local)
    melhor = registrar_comparacao(tabela, caminho_csv=tmp_path / "tabela.csv", fonte="teste.py")

    campeao = selecionar_campeao(melhor, {CANDIDATO: resultados}, X_test, y_test)
    negocio = calcular_metricas_negocio(y_test, campeao.pipeline.predict(X_test))
    cv = metricas_cv_do_campeao(campeao, {CANDIDATO: resultados})
    caminho = salvar_campeao(campeao.pipeline, tmp_path / "champion_model.joblib")
    versao = registrar_campeao(
        campeao.pipeline,
        campeao,
        metricas_negocio=negocio,
        metricas_cv=cv,
        fonte="teste.py",
    )

    assert caminho.exists()
    assert versao is not None
    assert set(campeao.metricas_teste) == {"f1", "auc_roc", "pr_auc"}
    assert negocio["vn"] + negocio["fp"] + negocio["fn"] + negocio["vp"] == len(y_test)
    assert cv["pr_auc_mean"] == pytest.approx(resultados[campeao.metodo].pr_auc_mean)
    assert joblib.load(caminho).predict(X_test).shape == (len(X_test),)


# --------------------------------------------------------------------------
# Utilitários
# --------------------------------------------------------------------------


def test_silenciar_optuna_nao_explode():
    import optuna

    silenciar_optuna()

    assert optuna.logging.get_verbosity() == optuna.logging.WARNING


def test_fit_kwargs_repassa_o_peso_para_o_passo_do_modelo():
    from src.training.tuning import _fit_kwargs

    pesos = np.array([1.0, 2.0, 3.0, 4.0])

    assert _fit_kwargs(None) == {}
    assert np.array_equal(_fit_kwargs(pesos)["modelo__sample_weight"], pesos)
    fatiado = _fit_kwargs(pesos, np.array([0, 2]))["modelo__sample_weight"]
    assert np.array_equal(fatiado, np.array([1.0, 3.0]))


def test_espacos_de_busca_sugerem_parametros_validos():
    """Exercita as funções `sugerir` com um trial de verdade."""
    import optuna

    from src.training.estimators import (
        ESPACO_LOGISTIC_REGRESSION,
        ESPACO_MLP,
        ESPACO_RANDOM_FOREST,
        criar_mlp,
        criar_random_forest,
    )

    estudo = optuna.create_study()
    trial = estudo.ask()

    rf = ESPACO_RANDOM_FOREST.sugerir(trial)
    mlp = ESPACO_MLP.sugerir(trial)
    lr = ESPACO_LOGISTIC_REGRESSION.sugerir(trial)

    # Os kwargs sugeridos têm que ser aceitos pelas fábricas correspondentes.
    assert criar_random_forest(**rf).n_estimators == rf["n_estimators"]
    assert criar_mlp(**mlp).hidden_layer_sizes == mlp["hidden_layer_sizes"]
    assert isinstance(criar_logistic_regression(**lr), LogisticRegression)


def test_avaliar_por_cv_aceita_scoring_customizado(base_pequena):
    X, y = base_pequena
    pipeline = build_pipeline(modelo=criar_baseline())

    metricas = avaliar_por_cv(pipeline, X, y, n_splits=2, scoring={"f1": "f1"})

    assert set(metricas) == {"cv_f1", "cv_f1_std"}


def test_experimento_isolado_por_teste(mlflow_local):
    """Garante que a fixture não vaza runs de um teste para outro."""
    runs = mlflow.search_runs(experiment_names=[mlflow_local], max_results=10)

    assert runs.empty
    assert config.MLFLOW_TRACKING_URI is None, "config remota tem que estar neutralizada"


# --------------------------------------------------------------------------
# dataset.carregar_base_modelagem
# --------------------------------------------------------------------------


def test_carregar_base_le_o_parquet_e_separa_o_alvo(tmp_path, base_pequena):
    from src.training.dataset import carregar_base_modelagem

    X_original, y_original = base_pequena
    parquet = tmp_path / "processado.parquet"
    X_original.assign(**{config.TARGET: y_original}).to_parquet(parquet, index=False)

    X, y = carregar_base_modelagem(parquet)

    assert config.TARGET not in X.columns, "o alvo nao pode sobrar como feature"
    assert list(X.columns) == list(X_original.columns)
    assert len(X) == len(y) == len(X_original)
    assert y.tolist() == y_original.tolist()


def test_carregar_base_aplica_o_filtro_de_censura(tmp_path, base_pequena):
    """remover_joined=True tira os clientes recentes do conjunto de modelagem."""
    from src.features import config as cfg
    from src.training.dataset import carregar_base_modelagem

    X_original, y_original = base_pequena
    status = ["Joined"] * 10 + ["Stayed"] * (len(X_original) - 10)
    parquet = tmp_path / "com_status.parquet"
    X_original.assign(**{config.TARGET: y_original, cfg.COLUNA_STATUS: status}).to_parquet(
        parquet, index=False
    )

    X_com, _ = carregar_base_modelagem(parquet, remover_joined=False)
    X_sem, _ = carregar_base_modelagem(parquet, remover_joined=True)

    assert len(X_com) == len(X_original)
    assert len(X_sem) == len(X_original) - 10


def test_carregar_base_sem_parquet_diz_como_gerar(tmp_path):
    from src.training.dataset import carregar_base_modelagem

    with pytest.raises(FileNotFoundError, match="make etl"):
        carregar_base_modelagem(tmp_path / "nao_existe.parquet")


def test_carregar_base_usa_o_caminho_padrao_do_projeto(tmp_path, monkeypatch):
    """Sem argumento, monta o caminho a partir de config.PROCESSED_DATA_DIR."""
    from src.training import dataset

    monkeypatch.setattr(dataset.config, "PROCESSED_DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match=dataset.NOME_PARQUET):
        dataset.carregar_base_modelagem()


# --------------------------------------------------------------------------
# Ramos restantes
# --------------------------------------------------------------------------


def test_buscar_baseline_sem_run_nenhuma(monkeypatch):
    """Notebook 04 nunca rodou: a comparacao segue sem a linha do baseline."""
    monkeypatch.setattr("mlflow.search_runs", lambda **kwargs: pd.DataFrame())

    assert buscar_baseline(experimento="vazio") is None


def test_resolver_pipeline_baixa_o_baseline_do_mlflow(tmp_path, base_pequena, monkeypatch):
    """Ultima cartada quando o joblib da Etapa 1 sumiu: a run tem o modelo."""
    import types

    import mlflow as mlflow_pkg

    X, y = base_pequena
    pipeline = build_pipeline(modelo=criar_baseline()).fit(X, y)
    uris = []

    def _load_model(uri):
        uris.append(uri)
        return pipeline

    monkeypatch.setattr(mlflow_pkg, "sklearn", types.SimpleNamespace(load_model=_load_model))
    campeao = Campeao(candidato=NOME_RUN_BASELINE, metodo="n/a", pipeline=None)

    recuperado = resolver_pipeline_campeao(
        campeao,
        baseline={"run_id": "abc123"},
        caminho_baseline=tmp_path / "nao_existe.joblib",
    )

    assert recuperado is pipeline
    assert uris == ["runs:/abc123/modelo"]


def test_metricas_cv_do_campeao_sem_baseline_informado():
    """Baseline venceu mas ninguem passou as metricas: devolve vazio em vez de estourar."""
    from src.training.champion import metricas_cv_do_campeao

    campeao = Campeao(candidato=NOME_RUN_BASELINE, metodo="n/a")

    assert metricas_cv_do_campeao(campeao, {}) == {}


def test_metricas_cv_do_campeao_com_candidato_ausente():
    from src.training.champion import metricas_cv_do_campeao

    campeao = Campeao(candidato="fantasma", metodo="optuna")

    assert metricas_cv_do_campeao(campeao, {}) == {}


def test_reconstruir_da_logistica_copia_os_params():
    """Sem parametro composto, reconstruir e so uma copia defensiva."""
    from src.training.estimators import ESPACO_LOGISTIC_REGRESSION

    originais = {"C": 1.5, "penalty": "l1"}
    reconstruidos = ESPACO_LOGISTIC_REGRESSION.reconstruir(originais)

    assert reconstruidos == originais
    assert reconstruidos is not originais


def test_objetivo_do_optuna_aborta_quando_o_pruner_manda(base_pequena):
    """Ramo de pruning: interrompe a configuracao ruim antes de rodar todo fold."""
    import optuna

    from src.training.tuning import _objetivo_optuna

    X, y = base_pequena
    reportados = []

    class TrialQuePoda:
        """Duplo de trial que sempre manda podar, logo apos o primeiro fold."""

        def suggest_float(self, nome, baixo, alto, log=False):
            return baixo

        def report(self, valor, step):
            reportados.append((step, valor))

        def should_prune(self):
            return True

    with pytest.raises(optuna.TrialPruned):
        _objetivo_optuna(
            TrialQuePoda(),
            criar_logistic_regression,
            ESPACO_MINIMO,
            X,
            y,
            None,
            2,
            config.SEED,
        )

    assert len(reportados) == 1, "tem que parar no primeiro fold, nao rodar os dois"
    assert reportados[0][0] == 0
