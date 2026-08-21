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

Sobre os domínios categóricos: todos os campos validam domínio fechado
via `Literal`, espelhando o dicionário da IBM (Contrato 1). Valor fora da
lista devolve 422, sem exceção. A regra é uniforme de propósito, pedido do
review da Etapa 3: o rascunho inicial deixava `services_payment_method`
como texto livre e as demais fechadas, e a mistura confundia. O
`handle_unknown` do pipeline continua existindo como segunda linha de
defesa, mas o caminho esperado é a validação barrar antes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SimNao = Literal["Yes", "No"]

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


class ChurnRequest(BaseModel):
    """Um cliente no formato pós-ETL (Contrato 1), pronto para o pipeline."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [_EXEMPLO_REQUEST]},
    )

    demographics_gender: Literal["Male", "Female"]
    demographics_age: int = Field(ge=0, le=120)
    demographics_senior_citizen: SimNao
    demographics_married: SimNao
    demographics_dependents: SimNao
    demographics_number_of_dependents: int = Field(ge=0)
    services_number_of_referrals: int = Field(ge=0)
    services_tenure_in_months: int = Field(ge=0)
    services_phone_service: SimNao
    services_avg_monthly_long_distance_charges: float = Field(ge=0)
    services_multiple_lines: SimNao
    services_internet_type: Literal["Cable", "DSL", "Fiber Optic"] | None = Field(
        description=(
            "null = cliente sem internet; vira a categoria 'No Internet Service' "
            "dentro do pipeline (EngenhariaEstrutural)."
        )
    )
    services_avg_monthly_gb_download: float = Field(ge=0)
    services_online_security: SimNao
    services_online_backup: SimNao
    services_device_protection_plan: SimNao
    services_premium_tech_support: SimNao
    services_streaming_tv: SimNao
    services_streaming_movies: SimNao
    services_streaming_music: SimNao
    services_unlimited_data: SimNao
    services_contract: Literal["Month-to-Month", "One Year", "Two Year"]
    services_paperless_billing: SimNao
    services_payment_method: Literal["Bank Withdrawal", "Credit Card", "Mailed Check"]
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
