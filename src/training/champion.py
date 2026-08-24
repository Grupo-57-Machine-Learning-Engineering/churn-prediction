"""Seleção, persistência e registro do modelo campeão da Etapa 2.

O campeão é o par (candidato, método) com maior PR-AUC médio de CV. Ele é
avaliado **uma única vez** no conjunto de teste, depois da escolha, para que
o teste não vire critério de seleção.

Dois destinos, de propósito. O `models/champion_model.joblib` é o entregável
local e funciona sem rede nem credencial. O Model Registry do MLflow recebe o
mesmo pipeline com o alias `@champion`, que é o endereço estável que a API
consulta quando o arquivo local não existe. O registro é best-effort: se o
DagsHub estiver fora, o joblib continua sendo a fonte de verdade.

O baseline da Etapa 1 pode vencer a comparação. Nesse caso não há retreino,
e o campeão é o artefato que o notebook 04 já produziu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from src import config
from src.logger import get_logger
from src.training.comparison import NOME_RUN_BASELINE
from src.training.metrics import calcular_metricas
from src.training.tuning import FONTE_PADRAO, ResultadoTuning

logger = get_logger(__name__)

NOME_ARQUIVO_BASELINE = "baseline_logistic_regression.joblib"

# O nome do arquivo, o do registro e o do alias vêm de `src.config`, e não
# são redeclarados aqui. Este módulo escreve o artefato e `src.models.predict`
# lê: com duas cópias das constantes, renomear o registro de um lado deixaria
# o outro procurando um endereço que não existe mais.

__all__ = [
    "Campeao",
    "metricas_cv_do_campeao",
    "registrar_campeao",
    "resolver_pipeline_campeao",
    "salvar_campeao",
    "selecionar_campeao",
]


@dataclass
class Campeao:
    """Candidato vencedor e sua avaliação única no teste.

    `pipeline` vem `None` quando o baseline vence, porque nesse caso o
    artefato não foi treinado nesta etapa. Use `resolver_pipeline_campeao`
    para materializá-lo antes de salvar.
    """

    candidato: str
    metodo: str
    pipeline: Pipeline | None = None
    metricas_teste: dict[str, float] = field(default_factory=dict)

    @property
    def baseline_venceu(self) -> bool:
        return self.candidato == NOME_RUN_BASELINE


def selecionar_campeao(
    melhor_linha: pd.Series,
    resultados_por_candidato: dict[str, dict[str, ResultadoTuning]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    baseline: dict[str, float] | None = None,
) -> Campeao:
    """Escolhe o vencedor da tabela e mede o teste uma única vez.

    Parameters
    ----------
    melhor_linha
        Primeira linha da tabela comparativa (saída de `registrar_comparacao`).
    resultados_por_candidato
        `{nome_candidato: saída de rodar_tuning}`. Fornece o pipeline já
        fitado, evitando um retreino do vencedor.
    baseline
        Métricas do baseline. Obrigatório quando ele é o vencedor, já que as
        métricas de teste dele vêm da run da Etapa 1.

    Raises
    ------
    ValueError
        Se o baseline vence sem que suas métricas tenham sido passadas, ou se
        o candidato vencedor não está em `resultados_por_candidato`.
    """
    candidato = melhor_linha["candidato"]
    metodo = melhor_linha["metodo"]

    if candidato == NOME_RUN_BASELINE:
        if baseline is None:
            raise ValueError(
                "O baseline venceu a comparação, mas suas métricas não foram "
                "passadas. Chame buscar_baseline() e repasse o resultado."
            )
        logger.info("O baseline venceu a comparação de CV. Nenhum retreino necessário.")
        return Campeao(
            candidato=candidato,
            metodo=metodo,
            pipeline=None,
            metricas_teste={
                "f1": baseline["teste_f1"],
                "auc_roc": baseline["teste_auc_roc"],
                "pr_auc": baseline["teste_pr_auc"],
            },
        )

    if candidato not in resultados_por_candidato:
        raise ValueError(
            f"Candidato vencedor '{candidato}' não está em resultados_por_candidato "
            f"(tem: {sorted(resultados_por_candidato)}). A tabela veio do MLflow e "
            "pode conter runs de execuções anteriores."
        )

    metodos = resultados_por_candidato[candidato]
    if metodo not in metodos:
        raise ValueError(
            f"Método vencedor '{metodo}' não foi executado para '{candidato}' nesta "
            f"sessão (tem: {sorted(metodos)}). Mesma causa do caso acima: a tabela "
            "vem do MLflow e pode citar uma run antiga."
        )

    pipeline = metodos[metodo].pipeline
    if pipeline is None:
        raise ValueError(
            f"O resultado de '{candidato}__{metodo}' não tem pipeline fitado. "
            "rodar_tuning sempre preenche esse campo, então um None aqui indica "
            "um ResultadoTuning montado à mão."
        )

    metricas = calcular_metricas(
        y_test,
        pipeline.predict(X_test),
        pipeline.predict_proba(X_test)[:, 1],
    )
    return Campeao(candidato=candidato, metodo=metodo, pipeline=pipeline, metricas_teste=metricas)


def resolver_pipeline_campeao(
    campeao: Campeao,
    *,
    baseline: dict[str, Any] | None = None,
    caminho_baseline: Path | str | None = None,
) -> Pipeline:
    """Devolve o pipeline do campeão, materializando o do baseline se preciso.

    Quando o baseline vence, tenta primeiro o joblib local da Etapa 1 e cai
    para o modelo registrado na run do baseline no MLflow.

    Raises
    ------
    FileNotFoundError
        Se o baseline venceu e nenhuma das duas fontes está disponível.
    """
    if campeao.pipeline is not None:
        return campeao.pipeline

    caminho_baseline = (
        Path(caminho_baseline)
        if caminho_baseline is not None
        else config.MODELS_DIR / NOME_ARQUIVO_BASELINE
    )
    if caminho_baseline.exists():
        logger.info("Reaproveitando o artefato do baseline: %s", caminho_baseline)
        return joblib.load(caminho_baseline)

    if baseline is not None and baseline.get("run_id"):
        uri = f"runs:/{baseline['run_id']}/modelo"
        logger.info("Baixando o baseline do MLflow: %s", uri)
        return mlflow.sklearn.load_model(uri)

    raise FileNotFoundError(
        f"O baseline venceu, mas não há artefato em '{caminho_baseline}' nem "
        "run_id do baseline para baixar do MLflow. Rode notebooks/04_baseline.ipynb."
    )


def salvar_campeao(pipeline: Pipeline, caminho: Path | str | None = None) -> Path:
    """Grava o pipeline campeão em disco. Roda com ou sem MLflow no ar."""
    caminho = Path(caminho) if caminho is not None else config.CHAMPION_MODEL_PATH
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, caminho)
    logger.info("Modelo campeão salvo em: %s", caminho)
    return caminho


def metricas_cv_do_campeao(
    campeao: Campeao,
    resultados_por_candidato: dict[str, dict[str, ResultadoTuning]],
    *,
    baseline: dict[str, float] | None = None,
) -> dict[str, float]:
    """Métricas de CV do vencedor, para a run final mostrar CV e teste juntos.

    Sem isso seria preciso cruzar duas runs para explicar a diferença entre o
    desempenho de validação cruzada e o de teste.
    """
    if campeao.baseline_venceu:
        if baseline is None:
            return {}
        return {
            "pr_auc_mean": baseline["cv_pr_auc"],
            "f1_mean": baseline["cv_f1"],
            "auc_roc_mean": baseline["cv_auc_roc"],
        }

    resultado = resultados_por_candidato.get(campeao.candidato, {}).get(campeao.metodo)
    if resultado is None:
        return {}
    return {
        "pr_auc_mean": resultado.pr_auc_mean,
        "f1_mean": resultado.f1_mean,
        "auc_roc_mean": resultado.auc_roc_mean,
    }


def registrar_campeao(
    pipeline: Pipeline,
    campeao: Campeao,
    *,
    metricas_negocio: dict[str, float] | None = None,
    metricas_cv: dict[str, float] | None = None,
    fonte: str = FONTE_PADRAO,
    run_name: str = "etapa2_campeao_final",
) -> str | int | None:
    """Loga o campeão no MLflow, registra no Registry e move o alias.

    Best-effort de propósito: uma falha de rede ou credencial não pode
    invalidar um treino que já terminou e já foi salvo em disco.

    Returns
    -------
    str | int | None
        Versão criada no Model Registry, ou `None` se o registro falhou. O
        tipo depende da versão do MLflow: `ModelInfo.registered_model_version`
        vem como inteiro, e `set_registered_model_alias` aceita os dois.
    """
    metricas_negocio = metricas_negocio or {}
    metricas_cv = metricas_cv or {}

    try:
        with config.iniciar_run(fonte, run_name=run_name):
            mlflow.log_param("candidato_vencedor", campeao.candidato)
            mlflow.log_metrics({f"teste_{k}": v for k, v in campeao.metricas_teste.items()})
            mlflow.log_metrics({f"teste_{k}": v for k, v in metricas_negocio.items()})
            if metricas_cv.get("pr_auc_mean") is not None:
                mlflow.log_metrics(
                    {
                        "cv_pr_auc_mean": metricas_cv["pr_auc_mean"],
                        "cv_f1_mean": metricas_cv["f1_mean"],
                        "cv_auc_roc_mean": metricas_cv["auc_roc_mean"],
                    }
                )

            info = mlflow.sklearn.log_model(
                pipeline,
                name="modelo",
                serialization_format="cloudpickle",
                registered_model_name=config.CHAMPION_MODEL_NAME,
            )
            versao = info.registered_model_version
            if versao is not None:
                MlflowClient().set_registered_model_alias(
                    config.CHAMPION_MODEL_NAME, config.CHAMPION_MODEL_ALIAS, versao
                )

        logger.info(
            "Campeão registrado (Model Registry: %s, alias @%s -> versão %s).",
            config.CHAMPION_MODEL_NAME,
            config.CHAMPION_MODEL_ALIAS,
            versao,
        )
        return versao
    except Exception as erro:  # pragma: no cover - depende de rede/credencial
        logger.warning("Não foi possível registrar no Model Registry: %s", erro)
        logger.warning(
            "O artefato local em '%s' continua sendo a fonte de verdade.",
            config.CHAMPION_MODEL_PATH,
        )
        return None
