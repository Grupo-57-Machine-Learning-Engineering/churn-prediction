"""Inventario canonico de colunas para a preparacao de features.

Cada lista abaixo corresponde a uma decisao documentada na EDA
(`docs/eda-findings.md`, branch `feat/eda-baselines`). Alterar qualquer
uma delas exige atualizar tambem o Contrato 1 em `docs/decisions.md`.

Convencao de nomes (ETL): `<base>_<coluna>` em snake_case, exceto as
chaves de juncao (`customer_id`, `zip_code`), que ficam sem prefixo.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Alvo
# --------------------------------------------------------------------------

ALVO = "status_churn_value"

COLUNA_STATUS = "status_customer_status"
"""Rotulo textual Joined / Stayed / Churned. Usado para o filtro de censura
antes do split; nunca entra como feature."""


# --------------------------------------------------------------------------
# Descartes
# --------------------------------------------------------------------------

COLS_CONSTANTES = [
    "services_quarter",
    "status_quarter",
    "locations_state",
    "locations_country",
    "locations_count",
    "demographics_count",
    "services_count",
    "status_count",
]
"""Variancia zero (EDA secao 13). Nao carregam informacao."""

COLS_IDENTIFICADORAS = [
    "customer_id",
    "locations_location_id",
    "services_service_id",
    "status_status_id",
    "populations_id",
]
"""Chaves surrogadas. `populations_id` e por CEP (1625 valores), nao por
cliente -- entrou na lista depois de aparecer como feature numerica
escalonada na primeira execucao sobre a base real."""

COLS_VAZAMENTO = [
    "status_satisfaction_score",
    "status_churn_score",
    "status_churn_category",
    "status_churn_reason",
    "status_cltv",
]
"""Informacao indisponivel no momento da previsao (EDA secao de vazamento).

`status_churn_score` fica reservado como *benchmark* a superar, nunca como
input -- ver o contra-argumento ao stacking em `docs/eda-findings.md`.
"""

COLS_DERIVADAS_DO_ALVO = [
    "status_churn_label",
    COLUNA_STATUS,
]
"""Reescritas diretas do alvo. NAO estavam marcadas no Contrato 1 nem no
`eda-findings.md`; incluidas aqui como achado adicional. Manter qualquer
uma delas produz AUC = 1,0 e mascara o vazamento como sucesso."""

COLS_REDUNDANTES = [
    "demographics_under_30",  # == demographics_age < 30
    "services_internet_service",  # == services_internet_type nao nulo
    "services_referred_a_friend",  # == services_number_of_referrals > 0
    "services_total_revenue",  # identidade contabil exata
    "services_total_long_distance_charges",  # == avg_monthly_ldc * tenure
]
"""Colinearidade perfeita ou quase (EDA secao 12). Mantida so uma de cada par."""

COLS_GEOGRAFIA = [
    "locations_city",
    "locations_lat_long",
    "zip_code",
    "locations_latitude",
    "locations_longitude",
    "populations_population",
]
"""Base inteira e da California. Cidade tem cardinalidade alta com N pequeno
por celula (San Diego: 64,9% de churn em 285 clientes). Descartada por padrao;
`incluir_geografia=True` no `build_pipeline` reativa para experimento."""

COLS_QUARENTENA = [
    "services_offer",
]
"""Suspeita de vazamento nao resolvida (revisao 9.1).

Offer E tem 52,9% de churn e Offer A 6,7%. Se as ofertas forem acoes de
RETENCAO aplicadas a clientes ja sinalizados como em risco, a causalidade
esta invertida e a coluna carrega informacao posterior ao sinal.
Pendente de checagem no dicionario da IBM. Descartada por padrao.
"""


# --------------------------------------------------------------------------
# Zeros estruturais
# --------------------------------------------------------------------------

ZEROS_ESTRUTURAIS = {
    "services_avg_monthly_gb_download": "flag_sem_internet",
    "services_avg_monthly_long_distance_charges": "flag_sem_telefone",
}
"""O zero codifica AUSENCIA DO SERVICO, nao consumo baixo (EDA secao 8 do
handoff). Escalonar sem separar mistura duas populacoes na mesma distribuicao.
"""


# --------------------------------------------------------------------------
# Nulos legitimos
# --------------------------------------------------------------------------

NULOS_LEGITIMOS = {
    "services_internet_type": "Sem Internet",
    "services_offer": "Sem Oferta",
}
"""Ausencia com significado de negocio, nao dado faltante. Um `dropna`
generico aqui derruba 22% e 55% da base respectivamente."""


# --------------------------------------------------------------------------
# Nucleo de features recomendado pela EDA
# --------------------------------------------------------------------------

NUCLEO_EDA = [
    "services_contract",
    "services_tenure_in_months",
    "services_internet_type",
    "services_payment_method",
    "services_monthly_charge",
    "demographics_dependents",
    "demographics_married",
    "demographics_senior_citizen",
    "services_online_security",
    "services_premium_tech_support",
]
"""Preditores de maior forca. Usado apenas pelo modo `subconjunto="nucleo"`,
para o baseline enxuto -- nao e o conjunto padrao."""
