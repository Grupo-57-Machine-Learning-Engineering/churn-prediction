"""Schemas Pydantic da API (Contrato 3, `docs/decisions.md`).

O request espelha as 28 features de entrada do Contrato 2, ou seja, as
colunas pós-ETL do Contrato 1 que sobrevivem ao descarte padrão de
`src/features/config.py`. Vale lembrar que esses nomes vêm do ETL do
projeto; o CSV do Kaggle, descartado pelo ADR-003, usa outros.

Algumas colunas ficam fora do schema de propósito. A principal é
`services_offer`, em quarentena por suspeita de vazamento (notebook 03,
seção 5.2): com `extra="forbid"`, um consumidor que a envie recebe 422 em
vez de ter o campo ignorado em silêncio. Identificadores, constantes,
geografia, redundantes e as colunas `status_*` também nunca são input de
predição (ver Contrato 2).

Sobre os domínios categóricos: as colunas de texto aceitam qualquer
string, e não uma lista fechada de valores. A decisão é do review da
Etapa 3 (ver ADR-006) e o motivo é operacional: o score do modelo tende a
ser gatilho de outros sistemas, então travar o request porque a operadora
lançou um plano novo transforma um problema de dado num problema de
disponibilidade. Categoria desconhecida cai no
`OneHotEncoder(handle_unknown="infrequent_if_exist")` do pipeline e a
predição sai normalmente, com a ressalva de que ela ignora o que aquele
valor novo significa. Detectar esse caso é assunto de monitoramento de
data drift, planejado para a etapa seguinte.

O que continua barrado com 422 é dado inválido de fato: idade negativa,
cobrança negativa, campo faltando, tipo errado. Esse tipo de payload não
descreve cliente nenhum, então não faz sentido pontuar.

Os valores conhecidos da base seguem documentados nas descrições dos
campos, para o Swagger continuar servindo de guia de quem consome.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_EXEMPLO_REQUEST = {
    "demographics_gender": "Female",
    "demographics_age": 34,
    "demographics_senior_citizen": "No",
    "demographics_married": "Yes",
    "demographics_dependents": "No",
    "demographics_number_of_dependents": 0,
    "services_number_of_referrals": 0,
    "services_tenure_in_months": 12,
    "services_phone_service": "Yes",
    "services_avg_monthly_long_distance_charges": 15.3,
    "services_multiple_lines": "No",
    "services_internet_type": "Fiber Optic",
    "services_avg_monthly_gb_download": 24,
    "services_online_security": "No",
    "services_online_backup": "Yes",
    "services_device_protection_plan": "No",
    "services_premium_tech_support": "No",
    "services_streaming_tv": "Yes",
    "services_streaming_movies": "No",
    "services_streaming_music": "No",
    "services_unlimited_data": "Yes",
    "services_contract": "Month-to-Month",
    "services_paperless_billing": "Yes",
    "services_payment_method": "Credit Card",
    "services_monthly_charge": 79.85,
    "services_total_charges": 958.2,
    "services_total_refunds": 0.0,
    "services_total_extra_data_charges": 0,
}

_SIM_NAO = "Valores da base: 'Yes' ou 'No'."


class ChurnRequest(BaseModel):
    """Um cliente no formato pós-ETL (Contrato 1), pronto para o pipeline."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [_EXEMPLO_REQUEST]},
    )

    demographics_gender: str = Field(description="Valores da base: 'Male' ou 'Female'.")
    demographics_age: int = Field(ge=0, le=120)
    demographics_senior_citizen: str = Field(description=_SIM_NAO)
    demographics_married: str = Field(description=_SIM_NAO)
    demographics_dependents: str = Field(description=_SIM_NAO)
    demographics_number_of_dependents: int = Field(ge=0)
    services_number_of_referrals: int = Field(ge=0)
    services_tenure_in_months: int = Field(ge=0)
    services_phone_service: str = Field(description=_SIM_NAO)
    services_avg_monthly_long_distance_charges: float = Field(ge=0)
    services_multiple_lines: str = Field(description=_SIM_NAO)
    services_internet_type: str | None = Field(
        description=(
            "Valores da base: 'Cable', 'DSL' ou 'Fiber Optic'. Use null para "
            "cliente sem internet, que vira a categoria 'No Internet Service' "
            "dentro do pipeline (EngenhariaEstrutural)."
        )
    )
    services_avg_monthly_gb_download: float = Field(ge=0)
    services_online_security: str = Field(description=_SIM_NAO)
    services_online_backup: str = Field(description=_SIM_NAO)
    services_device_protection_plan: str = Field(description=_SIM_NAO)
    services_premium_tech_support: str = Field(description=_SIM_NAO)
    services_streaming_tv: str = Field(description=_SIM_NAO)
    services_streaming_movies: str = Field(description=_SIM_NAO)
    services_streaming_music: str = Field(description=_SIM_NAO)
    services_unlimited_data: str = Field(description=_SIM_NAO)
    services_contract: str = Field(
        description="Valores da base: 'Month-to-Month', 'One Year' ou 'Two Year'."
    )
    services_paperless_billing: str = Field(description=_SIM_NAO)
    services_payment_method: str = Field(
        description="Valores da base: 'Bank Withdrawal', 'Credit Card' ou 'Mailed Check'."
    )
    services_monthly_charge: float = Field(ge=0)
    services_total_charges: float = Field(ge=0)
    services_total_refunds: float = Field(ge=0)
    services_total_extra_data_charges: float = Field(ge=0)


class ChurnResponse(BaseModel):
    """Resposta do `POST /predict` (Contrato 3)."""

    churn: bool = Field(description="Classe prevista com o threshold padrão do projeto.")
    probability: float = Field(
        ge=0,
        le=1,
        description=(
            "Propensão a um comportamento de churn observável agora, na mesma "
            "definição do alvo: snapshot, sem horizonte temporal. Interpretar "
            "como 'chance de cancelar nos próximos N meses' seria errado, "
            "porque a base tem trimestre único (ver ADR-005)."
        ),
    )
    model_version: str = Field(description="Versão do modelo/pacote usada na predição.")
