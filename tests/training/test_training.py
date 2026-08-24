"""Testes dos módulos de treino da Etapa 2.

Cobrem só o que roda sem rede: métricas, fábricas de estimadores, espaços de
busca, split e a seleção do campeão. Tuning e registro no MLflow ficam de
fora de propósito, porque dependem de servidor de tracking e de minutos de
CPU, o que não cabe no CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.config import SEED
from src.training.baseline import criar_baseline, params_do_baseline
from src.training.champion import (
    Campeao,
    metricas_cv_do_campeao,
    salvar_campeao,
    selecionar_campeao,
)
from src.training.comparison import (
    CANDIDATOS_PADRAO,
    COLUNAS_TABELA,
    NOME_RUN_BASELINE,
    buscar_baseline,
    montar_tabela_comparativa,
    nomes_de_runs,
    registrar_comparacao,
)
from src.training.dataset import dividir_treino_teste
from src.training.estimators import (
    ESPACO_LOGISTIC_REGRESSION,
    ESPACO_MLP,
    ESPACO_RANDOM_FOREST,
    criar_logistic_regression,
    criar_mlp,
    criar_random_forest,
)
from src.training.metrics import (
    N_SPLITS,
    SCORING,
    avaliar_por_cv,
    calcular_metricas,
    calcular_metricas_negocio,
    formatar_metricas,
    formatar_metricas_cv,
    formatar_metricas_negocio,
)
from src.training.tuning import ResultadoTuning

N = 200


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------


def test_calcular_metricas_do_classificador_perfeito():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.9, 0.95])

    metricas = calcular_metricas(y_true, (y_proba >= 0.5).astype(int), y_proba)

    assert metricas["f1"] == pytest.approx(1.0)
    assert metricas["auc_roc"] == pytest.approx(1.0)
    assert metricas["pr_auc"] == pytest.approx(1.0)


def test_formatar_metricas_traz_as_tres():
    texto = formatar_metricas("modelo", {"f1": 0.5, "auc_roc": 0.6, "pr_auc": 0.7})

    assert "modelo" in texto
    assert "F1=0.500" in texto
    assert "AUC-ROC=0.600" in texto
    assert "PR-AUC=0.700" in texto


def test_metricas_negocio_batem_com_a_matriz_de_confusao():
    #                VN            FP       FN          VP
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0, 1, 1])

    m = calcular_metricas_negocio(y_true, y_pred)

    assert (m["vn"], m["fp"], m["fn"], m["vp"]) == (4, 1, 1, 2)
    assert m["sensibilidade"] == pytest.approx(2 / 3)
    assert m["especificidade"] == pytest.approx(4 / 5)
    assert m["precisao"] == pytest.approx(2 / 3)
    assert m["vpn"] == pytest.approx(4 / 5)


def test_metricas_negocio_sem_positivo_previsto_nao_estoura():
    """Recorte degenerado: precisão indefinida vira 0.0 em vez de ZeroDivisionError."""
    m = calcular_metricas_negocio(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]))

    assert m["precisao"] == 0.0
    assert m["sensibilidade"] == 0.0


def test_formatar_metricas_negocio_mostra_threshold():
    m = calcular_metricas_negocio(np.array([0, 1]), np.array([0, 1]))

    texto = formatar_metricas_negocio(m, threshold=0.4)

    assert "threshold=0.4" in texto
    assert "Sensibilidade=1.000" in texto


# --------------------------------------------------------------------------
# Avaliação por CV (baseline, sem busca)
# --------------------------------------------------------------------------


def test_avaliar_por_cv_devolve_media_e_desvio_de_cada_metrica(dados_teste):
    X, y = dados_teste
    pipeline = Pipeline([("modelo", DummyClassifier(strategy="stratified", random_state=SEED))])

    metricas = avaliar_por_cv(pipeline, X, y, n_splits=3)

    esperadas = {f"cv_{nome}" for nome in SCORING} | {f"cv_{nome}_std" for nome in SCORING}
    assert set(metricas) == esperadas
    assert all(isinstance(v, float) for v in metricas.values())
    assert all(metricas[f"cv_{nome}_std"] >= 0 for nome in SCORING)


def test_avaliar_por_cv_usa_as_chaves_que_o_buscar_baseline_le():
    """Contrato com comparison.buscar_baseline: cv_f1, cv_auc_roc, cv_pr_auc."""
    assert {f"cv_{nome}" for nome in SCORING} == {"cv_f1", "cv_auc_roc", "cv_pr_auc"}


def test_protocolo_de_cv_e_unico_para_baseline_e_tuning():
    """tuning importa N_SPLITS/SCORING de metrics, então não pode divergir."""
    from src.training import tuning

    assert tuning.N_SPLITS is N_SPLITS
    assert tuning.SCORING is SCORING


def test_formatar_metricas_cv_traz_media_e_desvio():
    texto = formatar_metricas_cv({"cv_f1": 0.5, "cv_f1_std": 0.02})

    assert "F1" in texto
    assert "0.500 (+/- 0.020)" in texto


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def test_baseline_usa_o_solver_default_do_sklearn():
    """Trocar para liblinear mudaria o numero de referencia da Etapa 1."""
    modelo = criar_baseline()

    assert modelo.solver == "lbfgs"
    assert modelo.class_weight == "balanced"
    assert modelo.random_state == SEED
    assert modelo.max_iter == 1000


def test_baseline_difere_do_candidato_tunado_da_etapa_2():
    """A fabrica do tuning forca liblinear para suportar penalidade l1."""
    assert criar_baseline().solver != criar_logistic_regression().solver


def test_params_do_baseline_descrevem_a_run():
    params = params_do_baseline(criar_baseline())

    assert params["modelo"] == "LogisticRegression"
    assert params["class_weight"] == "balanced"
    assert params["random_state"] == SEED


# --------------------------------------------------------------------------
# Estimadores e espaços de busca
# --------------------------------------------------------------------------


def test_fabricas_fixam_seed_do_projeto():
    assert criar_random_forest().random_state == SEED
    assert criar_logistic_regression().random_state == SEED
    assert criar_mlp().random_state == SEED


def test_fabricas_aceitam_override():
    assert criar_random_forest(n_estimators=7).n_estimators == 7
    assert criar_logistic_regression(C=0.25).C == 0.25
    assert criar_mlp(alpha=0.5).alpha == 0.5


def test_balanceamento_por_class_weight_onde_o_sklearn_suporta():
    """O MLP não aceita class_weight; o balanceamento dele entra por sample_weight."""
    assert criar_random_forest().class_weight == "balanced"
    assert criar_logistic_regression().class_weight == "balanced"
    assert not hasattr(criar_mlp(), "class_weight")


def test_logistic_regression_usa_solver_compativel_com_l1():
    """`liblinear` é o que permite o grid alternar entre l1 e l2."""
    assert criar_logistic_regression().solver == "liblinear"
    assert set(ESPACO_LOGISTIC_REGRESSION.grid["modelo__penalty"]) == {"l1", "l2"}


@pytest.mark.parametrize("espaco", [ESPACO_RANDOM_FOREST, ESPACO_MLP, ESPACO_LOGISTIC_REGRESSION])
def test_espacos_usam_o_prefixo_do_passo_do_pipeline(espaco):
    """As chaves precisam do prefixo `modelo__` para o sklearn achar o estimador."""
    assert all(chave.startswith("modelo__") for chave in espaco.grid)
    assert all(chave.startswith("modelo__") for chave in espaco.distribuicoes)


def test_reconstruir_mlp_remonta_a_tupla_de_camadas():
    uma_camada = ESPACO_MLP.reconstruir({"n_units_1": 64, "n_units_2": 0, "alpha": 0.01})
    duas_camadas = ESPACO_MLP.reconstruir({"n_units_1": 64, "n_units_2": 32, "alpha": 0.01})

    assert uma_camada["hidden_layer_sizes"] == (64,)
    assert duas_camadas["hidden_layer_sizes"] == (64, 32)


def test_reconstruir_random_forest_preserva_n_jobs():
    assert ESPACO_RANDOM_FOREST.reconstruir({"n_estimators": 100})["n_jobs"] == -1


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def test_split_e_estratificado_e_reprodutivel():
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame({"a": rng.normal(size=N)})
    y = pd.Series((rng.random(N) < 0.3).astype(int))

    X_tr, X_te, y_tr, y_te = dividir_treino_teste(X, y)
    X_tr2, _, _, _ = dividir_treino_teste(X, y)

    assert len(X_te) == pytest.approx(N * 0.2, abs=1)
    assert len(X_tr) + len(X_te) == N
    assert y_tr.mean() == pytest.approx(y_te.mean(), abs=0.05)
    assert X_tr.index.equals(X_tr2.index)


# --------------------------------------------------------------------------
# Comparação
# --------------------------------------------------------------------------


def test_nomes_de_runs_cobre_pai_filhas_e_fechamento():
    nomes = nomes_de_runs()

    for candidato in CANDIDATOS_PADRAO:
        assert candidato in nomes
        assert f"{candidato}__grid_search" in nomes
        assert f"{candidato}__random_search" in nomes
        assert f"{candidato}__optuna" in nomes
    assert "etapa2_comparacao_final" in nomes
    assert "etapa2_campeao_final" in nomes
    assert len(nomes) == len(set(nomes))


# --------------------------------------------------------------------------
# Campeão
# --------------------------------------------------------------------------


@pytest.fixture
def dados_teste() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame({"a": rng.normal(size=40)})
    y = pd.Series((rng.random(40) < 0.4).astype(int))
    return X, y


@pytest.fixture
def resultados(dados_teste) -> dict[str, dict[str, ResultadoTuning]]:
    X, y = dados_teste
    pipeline = Pipeline([("modelo", DummyClassifier(strategy="prior"))]).fit(X, y)
    return {
        "random_forest": {
            "optuna": ResultadoTuning(
                metodo="optuna",
                pr_auc_mean=0.61,
                pr_auc_std=0.01,
                f1_mean=0.55,
                auc_roc_mean=0.80,
                tempo_segundos=1.0,
                params={"n_estimators": 100},
                pipeline=pipeline,
            )
        }
    }


def test_selecionar_campeao_avalia_o_vencedor_no_teste(resultados, dados_teste):
    X, y = dados_teste
    melhor_linha = pd.Series({"candidato": "random_forest", "metodo": "optuna"})

    campeao = selecionar_campeao(melhor_linha, resultados, X, y)

    assert campeao.candidato == "random_forest"
    assert campeao.pipeline is resultados["random_forest"]["optuna"].pipeline
    assert set(campeao.metricas_teste) == {"f1", "auc_roc", "pr_auc"}
    assert not campeao.baseline_venceu


def test_selecionar_campeao_quando_o_baseline_vence(resultados, dados_teste):
    X, y = dados_teste
    melhor_linha = pd.Series({"candidato": NOME_RUN_BASELINE, "metodo": "n/a"})
    baseline = {"teste_f1": 0.5, "teste_auc_roc": 0.8, "teste_pr_auc": 0.6}

    campeao = selecionar_campeao(melhor_linha, resultados, X, y, baseline=baseline)

    assert campeao.baseline_venceu
    assert campeao.pipeline is None, "baseline não é retreinado nesta etapa"
    assert campeao.metricas_teste["f1"] == 0.5


def test_selecionar_campeao_exige_metricas_quando_o_baseline_vence(resultados, dados_teste):
    X, y = dados_teste
    melhor_linha = pd.Series({"candidato": NOME_RUN_BASELINE, "metodo": "n/a"})

    with pytest.raises(ValueError, match="métricas não foram passadas"):
        selecionar_campeao(melhor_linha, resultados, X, y)


def test_selecionar_campeao_rejeita_candidato_desconhecido(resultados, dados_teste):
    """A tabela vem do MLflow e pode citar uma run de execução anterior."""
    X, y = dados_teste
    melhor_linha = pd.Series({"candidato": "modelo_fantasma", "metodo": "optuna"})

    with pytest.raises(ValueError, match="modelo_fantasma"):
        selecionar_campeao(melhor_linha, resultados, X, y)


def test_metricas_cv_do_campeao_vem_do_resultado_do_tuning(resultados):
    campeao = Campeao(candidato="random_forest", metodo="optuna")

    metricas = metricas_cv_do_campeao(campeao, resultados)

    assert metricas["pr_auc_mean"] == 0.61
    assert metricas["f1_mean"] == 0.55


def test_metricas_cv_do_campeao_usa_o_baseline_quando_ele_vence(resultados):
    campeao = Campeao(candidato=NOME_RUN_BASELINE, metodo="n/a")
    baseline = {"cv_pr_auc": 0.58, "cv_f1": 0.52, "cv_auc_roc": 0.79}

    metricas = metricas_cv_do_campeao(campeao, resultados, baseline=baseline)

    assert metricas["pr_auc_mean"] == 0.58


def test_salvar_campeao_grava_e_recarrega(tmp_path, dados_teste):
    import joblib

    X, y = dados_teste
    modelo = Pipeline([("modelo", DummyClassifier(strategy="prior"))]).fit(X, y)

    caminho = salvar_campeao(modelo, tmp_path / "sub" / "champion_model.joblib")

    assert caminho.exists(), "o diretório de destino deve ser criado"
    assert joblib.load(caminho).predict(X).shape == (len(X),)


# --------------------------------------------------------------------------
# Guardas apontadas pelo code review
# --------------------------------------------------------------------------


def test_metricas_negocio_com_uma_classe_so_nao_estoura():
    """confusion_matrix sem labels devolveria 1x1 e o unpack de 4 quebraria."""
    m = calcular_metricas_negocio(np.zeros(5, dtype=int), np.zeros(5, dtype=int))

    assert (m["vn"], m["fp"], m["fn"], m["vp"]) == (5, 0, 0, 0)
    assert m["sensibilidade"] == 0.0
    assert m["especificidade"] == 1.0


def test_tabela_comparativa_vazia_quando_o_mlflow_nao_devolve_run(monkeypatch):
    """search_runs devolve DataFrame sem colunas quando nada casa."""
    monkeypatch.setattr("mlflow.search_runs", lambda **kwargs: pd.DataFrame())

    tabela = montar_tabela_comparativa(experimento="inexistente")

    assert tabela.empty
    assert list(tabela.columns) == list(COLUNAS_TABELA)


def test_tabela_comparativa_vazia_quando_nenhum_nome_casa(monkeypatch):
    """Runs aninhadas existem, mas com outro esquema de nome."""
    runs = pd.DataFrame(
        {
            "tags.mlflow.runName": ["outro_modelo__grid_search"],
            "metrics.pr_auc_mean": [0.5],
        }
    )
    monkeypatch.setattr("mlflow.search_runs", lambda **kwargs: runs)

    tabela = montar_tabela_comparativa(experimento="qualquer")

    assert tabela.empty
    assert list(tabela.columns) == list(COLUNAS_TABELA)


def test_buscar_baseline_devolve_none_para_run_sem_metricas(monkeypatch):
    """Run interrompida no meio da CV sobrevive com o nome certo e sem metrica."""
    runs = pd.DataFrame({"tags.mlflow.runName": [NOME_RUN_BASELINE], "run_id": ["abc"]})
    monkeypatch.setattr("mlflow.search_runs", lambda **kwargs: runs)

    assert buscar_baseline(experimento="qualquer") is None


def test_registrar_comparacao_recusa_tabela_vazia():
    with pytest.raises(ValueError, match="vazia"):
        registrar_comparacao(pd.DataFrame(columns=list(COLUNAS_TABELA)))


def test_registrar_comparacao_recusa_topo_sem_pr_auc():
    """Sem pr_auc_mean a ordenacao e arbitraria: nao ha vencedor a anunciar."""
    tabela = pd.DataFrame(
        [{"candidato": "random_forest", "metodo": "optuna", "pr_auc_mean": np.nan}]
    )

    with pytest.raises(ValueError, match="pr_auc_mean"):
        registrar_comparacao(tabela)


def test_selecionar_campeao_rejeita_metodo_nao_executado(resultados, dados_teste):
    X, y = dados_teste
    melhor_linha = pd.Series({"candidato": "random_forest", "metodo": "grid_search"})

    with pytest.raises(ValueError, match="grid_search"):
        selecionar_campeao(melhor_linha, resultados, X, y)


def test_selecionar_campeao_rejeita_resultado_sem_pipeline(dados_teste):
    X, y = dados_teste
    sem_pipeline = {
        "random_forest": {
            "optuna": ResultadoTuning(
                metodo="optuna",
                pr_auc_mean=0.61,
                pr_auc_std=0.01,
                f1_mean=0.55,
                auc_roc_mean=0.80,
                tempo_segundos=1.0,
            )
        }
    }
    melhor_linha = pd.Series({"candidato": "random_forest", "metodo": "optuna"})

    with pytest.raises(ValueError, match="pipeline"):
        selecionar_campeao(melhor_linha, sem_pipeline, X, y)


def test_constantes_do_campeao_tem_fonte_unica():
    """champion escreve, predict le: os enderecos precisam vir do mesmo lugar.

    As constantes moraram em `src.models.predict` antes de `src.config`
    centralizar (ver ADR-006); este teste garante que `champion.salvar_campeao`
    e `champion.registrar_campeao` usam os defaults do `config`, e não uma
    cópia própria que poderia divergir do que `predict.carregar_campeao` lê.
    """
    from src import config
    from src.models import predict

    assert predict.config.CHAMPION_MODEL_PATH is config.CHAMPION_MODEL_PATH
    assert predict.config.CHAMPION_MODEL_URI == config.CHAMPION_MODEL_URI


# --------------------------------------------------------------------------
# ADR-008: pr_auc_std comparavel entre os tres metodos
# --------------------------------------------------------------------------


def test_desvio_entre_folds_casa_com_o_std_do_sklearn():
    """Metrica extra da run do Optuna: mesma grandeza que o std_test_pr_auc.

    Nao substitui o pr_auc_std, que reproduz o notebook (ver ADR-008).
    """
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold

    from src.features.preparation import build_pipeline
    from src.training.tuning import _metricas_cv_complementares

    rng = np.random.default_rng(SEED)
    n, n_splits = 150, 3
    contrato = rng.choice(["Month-to-Month", "One Year"], n)
    X = pd.DataFrame(
        {
            "services_contract": contrato,
            "services_tenure_in_months": rng.integers(1, 72, n),
            "services_monthly_charge": rng.uniform(20, 120, n).round(2),
        }
    )
    y = pd.Series((rng.random(n) < np.where(contrato == "Month-to-Month", 0.6, 0.15)).astype(int))
    kwargs = {"C": 1.0}

    _, _, pr_auc_std_entre_folds = _metricas_cv_complementares(
        criar_logistic_regression,
        kwargs,
        X,
        y,
        sample_weight=None,
        n_splits=n_splits,
        seed=SEED,
    )

    # Mesmo calculo na mao: PR-AUC de cada fold, desvio entre eles.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    por_fold = []
    for idx_tr, idx_val in skf.split(X, y):
        p = build_pipeline(modelo=criar_logistic_regression(**kwargs)).fit(
            X.iloc[idx_tr], y.iloc[idx_tr]
        )
        proba = p.predict_proba(X.iloc[idx_val])[:, 1]
        por_fold.append(average_precision_score(y.iloc[idx_val], proba))

    assert pr_auc_std_entre_folds == pytest.approx(float(np.std(por_fold)))
    assert len(por_fold) == n_splits, "o desvio vem de n_splits pontos, nao de n_trials"
