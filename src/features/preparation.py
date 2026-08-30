"""Preparacao de features da base Telco.

Divisao de responsabilidades:

* **Funcoes de nivel de linha** (`filtrar_censura`, `separar_alvo`) rodam
  ANTES do split. Alteram a quantidade de linhas, o que um `Pipeline` do
  sklearn nao sabe fazer.
* **Transformadores** (`EngenhariaEstrutural`, `DescartadorDeColunas`) rodam
  DENTRO do `Pipeline`, logo sao aplicados no `fit` do treino e replicados
  no teste sem reajuste.

Tudo que aprende parametro (imputacao, escala, one-hot) esta dentro do
`ColumnTransformer`, portanto o `fit` acontece so no treino. Esse e o bug
citado como `fix/scaler-leak` no README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import config as cfg

__all__ = [
    "EngenhariaEstrutural",
    "DescartadorDeColunas",
    "build_pipeline",
    "filtrar_censura",
    "separar_alvo",
    "colunas_descartadas",
]


# ==========================================================================
# Nivel de linha -- roda antes do split
# ==========================================================================


def filtrar_censura(
    df: pd.DataFrame,
    remover_joined: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa (opcionalmente) os clientes `Joined` do conjunto de modelagem.

    Os 454 clientes com status `Joined` entraram ha 1 a 3 meses e estao
    rotulados como nao-churn. A leitura inicial foi de censura a direita:
    eles nao teriam tido tempo de cancelar, e mante-los atenuaria o efeito
    de tenure.

    A base contradiz essa leitura. Na faixa tenure <= 3 existem 597
    `Churned` e 454 `Joined`, e nenhum `Stayed`. As duas categorias cobrem
    a faixa inteira, ou seja, `Joined` e o rotulo que a IBM da ao cliente
    recente que ainda nao cancelou. Isso e a mesma condicao de `Stayed`,
    so que com menos tempo de casa, e como negativo dentro da janela
    observada e legitimo.

    Remove-los tem custo alem do beneficio: os 597 restantes viram 100% de
    churn na faixa, uma separacao perfeita que e artefato de rotulagem. O
    ganho medido foi de cerca de 2 pontos de ROC-AUC (0,913 -> 0,927),
    performance que o modelo nao teria em producao.

    Por isso o padrao e `remover_joined=False`. A objecao da censura volta
    a valer se o grupo definir o alvo como uma janela futura ("cancela nos
    proximos N meses") em vez do snapshot atual; ver a pendencia sobre
    janela temporal na secao 10 do notebook 03.

    Returns
    -------
    (modelagem, censurados)
        O segundo DataFrame vem vazio quando `remover_joined=False`.
    """
    if cfg.COLUNA_STATUS not in df.columns or not remover_joined:
        return df.copy(), df.iloc[0:0].copy()

    eh_joined = df[cfg.COLUNA_STATUS].astype(str).str.strip().str.casefold() == "joined"
    return df.loc[~eh_joined].copy(), df.loc[eh_joined].copy()


def separar_alvo(df: pd.DataFrame, alvo: str = cfg.ALVO) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X e y. `y` sai como inteiro 0/1."""
    if alvo not in df.columns:
        raise KeyError(f"Coluna alvo '{alvo}' ausente. Colunas: {list(df.columns)[:10]}...")

    y = pd.to_numeric(df[alvo], errors="coerce")
    if y.isna().any():
        raise ValueError(f"'{alvo}' tem {int(y.isna().sum())} valores nao numericos.")

    return df.drop(columns=[alvo]), y.astype(int)


# ==========================================================================
# Transformadores -- rodam dentro do Pipeline
# ==========================================================================


class EngenhariaEstrutural(BaseEstimator, TransformerMixin):
    """Trata zeros estruturais, nulos legitimos e cria features derivadas.

    Sem estado: o `fit` nao aprende nada, entao nao ha risco de vazamento
    mesmo que fosse chamado na base inteira. Ainda assim vive no Pipeline
    para garantir que treino e producao passem pela mesma transformacao.

    O que faz
    ---------
    1. Nulos legitimos -> categoria explicita ("No Internet Service", "No Offer").
    2. Zeros estruturais -> flag binaria separada, preservando o valor
       numerico para quem tem o servico.
    3. `delta_cobranca` = total_charges - (monthly_charge * tenure).
       A EDA mediu essa diferenca (secao 12) mas nao interpretou (revisao
       9.5). Ela e residuo de reajuste ou mudanca de plano ao longo da
       relacao: positivo indica que o cliente pagou mais do que a
       mensalidade atual sugere (plano barateou ou houve cobranca extra),
       negativo indica reajuste recente para cima -- gatilho classico de
       cancelamento.
    """

    def __init__(self, criar_delta_cobranca: bool = True, criar_flags: bool = True):
        self.criar_delta_cobranca = criar_delta_cobranca
        self.criar_flags = criar_flags

    def fit(self, X: pd.DataFrame, y=None):  # noqa: ARG002
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for coluna, rotulo in cfg.NULOS_LEGITIMOS.items():
            if coluna in X.columns:
                X[coluna] = X[coluna].astype("object").fillna(rotulo)

        if self.criar_flags:
            for coluna, nome_flag in cfg.ZEROS_ESTRUTURAIS.items():
                if coluna in X.columns:
                    valores = pd.to_numeric(X[coluna], errors="coerce")
                    # Ausente nao e zero. Zero aqui significa "nao tem o servico",
                    # entao tratar nulo como zero afirmaria que o cliente nao tem
                    # internet ou telefone so porque o dado nao veio, e "sem
                    # internet" e o fator de protecao mais forte da base (7,4% de
                    # churn contra 31,8%). O nulo segue nulo e o SimpleImputer do
                    # ColumnTransformer decide o que fazer, com add_indicator
                    # avisando o modelo de que aquilo veio faltando.
                    # Na base de treino essas colunas nunca sao nulas (o zero e
                    # real), entao esta linha nao muda nada do que o modelo
                    # aprendeu, so o comportamento na inferencia com dado parcial.
                    flag = (valores == 0).astype("float64")
                    X[nome_flag] = flag.where(valores.notna())

        if self.criar_delta_cobranca:
            X = self._delta_cobranca(X)

        return X

    @staticmethod
    def _delta_cobranca(X: pd.DataFrame) -> pd.DataFrame:
        necessarias = (
            "services_total_charges",
            "services_monthly_charge",
            "services_tenure_in_months",
        )
        if not all(c in X.columns for c in necessarias):
            return X

        total = pd.to_numeric(X["services_total_charges"], errors="coerce")
        mensal = pd.to_numeric(X["services_monthly_charge"], errors="coerce")
        tenure = pd.to_numeric(X["services_tenure_in_months"], errors="coerce")

        esperado = mensal * tenure
        X["delta_cobranca"] = total - esperado

        # Versao relativa: comparavel entre clientes de mensalidade diferente.
        # Denominador protegido -- tenure 0 existe na base.
        denominador = esperado.replace(0, np.nan)
        X["delta_cobranca_pct"] = (X["delta_cobranca"] / denominador).replace(
            [np.inf, -np.inf], np.nan
        )
        return X


class DescartadorDeColunas(BaseEstimator, TransformerMixin):
    """Remove as colunas decididas na EDA.

    Silencioso quanto a colunas ausentes de proposito: o mesmo Pipeline
    precisa rodar sobre o parquet completo (61 colunas) e sobre o payload
    reduzido do `POST /predict`, que nunca vai carregar `status_*`.
    """

    def __init__(self, colunas: list[str] | None = None):
        self.colunas = colunas or []

    def fit(self, X: pd.DataFrame, y=None):  # noqa: ARG002
        self.colunas_removidas_ = [c for c in self.colunas if c in X.columns]
        self.colunas_ausentes_ = [c for c in self.colunas if c not in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.colunas if c in X.columns])


# ==========================================================================
# Factory
# ==========================================================================


def colunas_descartadas(
    incluir_geografia: bool = False,
    incluir_offer: bool = False,
) -> list[str]:
    """Lista consolidada de colunas a remover, na ordem das categorias."""
    colunas = [
        *cfg.COLS_CONSTANTES,
        *cfg.COLS_IDENTIFICADORAS,
        *cfg.COLS_VAZAMENTO,
        *cfg.COLS_DERIVADAS_DO_ALVO,
        *cfg.COLS_REDUNDANTES,
    ]
    if not incluir_geografia:
        colunas += cfg.COLS_GEOGRAFIA
    if not incluir_offer:
        colunas += cfg.COLS_QUARENTENA

    vistas: set[str] = set()
    return [c for c in colunas if not (c in vistas or vistas.add(c))]


def build_pipeline(
    modelo=None,
    *,
    incluir_geografia: bool = False,
    incluir_offer: bool = False,
    criar_delta_cobranca: bool = False,
    min_frequency: float | None = 0.01,
) -> Pipeline:
    """Monta o Pipeline de preparacao, opcionalmente com um estimador no fim.

    Contrato 2 do `docs/decisions.md`. Chamado sem argumento devolve so a
    transformacao, util para inspecionar a matriz resultante.

    Parameters
    ----------
    modelo
        Estimador sklearn anexado como ultimo passo. `None` devolve apenas
        a transformacao.
    incluir_geografia
        Reativa cidade/CEP/lat-long/populacao. Padrao `False`.
    incluir_offer
        Reativa `services_offer`, em quarentena por suspeita de vazamento.
    criar_delta_cobranca
        Liga as features `delta_cobranca` e `delta_cobranca_pct`. Padrao
        `False`: na base real a correlacao com o alvo ficou em 0,00 e -0,01.
        A mediana da diferenca e exatamente zero, ou seja, `total_charges`
        bate com `monthly_charge x tenure` para a maioria dos clientes --
        nao ha reajuste registrado nesta base. Hipotese descartada.
    min_frequency
        Categorias abaixo dessa frequencia sao agrupadas em `infrequent`
        pelo one-hot. Protege contra explosao de colunas se a geografia for
        reativada. `None` desliga o agrupamento.

    Notes
    -----
    O `StandardScaler` e obrigatorio para o `MLPClassifier` -- redes neurais
    nao convergem de forma estavel com features em escalas diferentes. Ele
    fica dentro do Pipeline justamente para que o `fit` ocorra apenas sobre
    o fold de treino.
    """
    numericas = Pipeline(
        [
            ("imputacao", SimpleImputer(strategy="median", add_indicator=True)),
            ("escala", StandardScaler()),
        ]
    )

    categoricas = Pipeline(
        [
            ("imputacao", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessamento = ColumnTransformer(
        [
            ("num", numericas, make_column_selector(dtype_include=np.number)),
            (
                "cat",
                categoricas,
                make_column_selector(dtype_include=["object", "category", "bool"]),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    passos = [
        (
            "estrutural",
            EngenhariaEstrutural(criar_delta_cobranca=criar_delta_cobranca),
        ),
        (
            "descarte",
            DescartadorDeColunas(
                colunas_descartadas(
                    incluir_geografia=incluir_geografia,
                    incluir_offer=incluir_offer,
                )
            ),
        ),
        ("preprocessamento", preprocessamento),
    ]

    if modelo is not None:
        passos.append(("modelo", modelo))

    return Pipeline(passos)
