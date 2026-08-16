"""Testes de src.training.metrics e src.training.baseline."""

import numpy as np
import pytest

from src.config import PROCESSED_DATA_DIR
from src.training.metrics import Metricas, calcular_metricas, formatar_metricas


class TestCalcularMetricas:
    def test_retorna_metricas_esperadas_em_previsao_perfeita(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.05, 0.1, 0.9, 0.95])

        m = calcular_metricas(y_true, y_pred, y_proba)

        assert isinstance(m, Metricas)
        assert m.f1 == 1.0
        assert m.auc_roc == 1.0
        assert m.pr_auc == 1.0

    def test_penaliza_previsao_pior_que_aleatoria(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.95, 0.05, 0.1])

        m = calcular_metricas(y_true, y_pred, y_proba)

        assert m.f1 == 0.0
        assert m.auc_roc == 0.0

    def test_as_dict_contem_as_tres_chaves(self) -> None:
        m = Metricas(f1=0.7, auc_roc=0.9, pr_auc=0.75)
        d = m.as_dict()
        assert set(d.keys()) == {"f1", "auc_roc", "pr_auc"}


class TestFormatarMetricas:
    def test_formata_com_nome_do_modelo(self) -> None:
        m = Metricas(f1=0.694, auc_roc=0.908, pr_auc=0.766)
        texto = formatar_metricas("Baseline", m)
        assert "Baseline" in texto
        assert "0.694" in texto
        assert "0.908" in texto
        assert "0.766" in texto


class TestTreinarBaseline:
    def test_treinar_baseline_sobre_a_base_real(self) -> None:
        from src.training.baseline import treinar_baseline

        caminho = PROCESSED_DATA_DIR / "telco_churn_processed.parquet"
        if not caminho.exists():
            pytest.skip(f"{caminho} não existe — rode o pipeline de ETL primeiro.")

        resultado = treinar_baseline(salvar_modelo=False)

        assert "pipeline" in resultado
        assert "metricas_cv" in resultado
        assert "metricas_teste" in resultado

        metricas_teste = resultado["metricas_teste"]
        # Sanidade: um baseline minimamente decente deve superar bastante
        # o "chute aleatório" (AUC-ROC=0.5) nesta base.
        assert metricas_teste.auc_roc > 0.7
        assert 0.0 <= metricas_teste.f1 <= 1.0
        assert 0.0 <= metricas_teste.pr_auc <= 1.0

        # Proporção de churn preservada pelo split estratificado.
        y_test = resultado["y_test"]
        assert abs(y_test.mean() - 0.265) < 0.03
