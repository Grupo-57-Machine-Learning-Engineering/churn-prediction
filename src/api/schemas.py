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

Duas decisões do review da Etapa 3 moldam a validação (detalhe no ADR-006).

A primeira: as colunas de texto aceitam qualquer string, e não uma lista
fechada de valores. O motivo é operacional, o score costuma ser gatilho de
outros sistemas, então travar o request porque a operadora lançou um plano
novo transforma um problema de dado num problema de disponibilidade.
Categoria desconhecida cai no
`OneHotEncoder(handle_unknown="infrequent_if_exist")` do pipeline.

A segunda: todo campo é opcional. Quem consome nem sempre tem a ficha
completa do cliente, e o pipeline já foi construído para isso, com
`SimpleImputer(add_indicator=True)` nos numéricos e imputação por constante
nos categóricos. Campo ausente vale o mesmo que string vazia, que a API já
aceitava. Quanto menos informação chega, menos a predição se apoia no
cliente e mais ela se apoia no perfil mediano do treino, o que não aparece
na resposta. Sinalizar isso é trabalho de detecção de data drift, planejado
para a etapa de monitoramento.

O que continua barrado com 422 é dado inválido de fato: idade negativa,
cobrança negativa, tipo errado, campo extra. Esse tipo de payload não
descreve cliente nenhum, então não faz sentido pontuar.

Os valores conhecidos da base seguem documentados nas descrições dos
campos, para o Swagger continuar servindo de guia de quem consome.
"""

from __future__ import annotations

import pandas as pd
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

_SIM_NAO = "Valores da base: 'Yes' ou 'No'. Opcional."


class ChurnRequest(BaseModel):
    """Um cliente no formato pós-ETL (Contrato 1), pronto para o pipeline."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [_EXEMPLO_REQUEST]},
    )

    demographics_gender: str | None = Field(
        default=None, description="Valores da base: 'Male' ou 'Female'. Opcional."
    )
    demographics_age: int | None = Field(default=None, ge=0, le=120)
    demographics_senior_citizen: str | None = Field(default=None, description=_SIM_NAO)
    demographics_married: str | None = Field(default=None, description=_SIM_NAO)
    demographics_dependents: str | None = Field(default=None, description=_SIM_NAO)
    demographics_number_of_dependents: int | None = Field(default=None, ge=0)
    services_number_of_referrals: int | None = Field(default=None, ge=0)
    services_tenure_in_months: int | None = Field(default=None, ge=0)
    services_phone_service: str | None = Field(default=None, description=_SIM_NAO)
    services_avg_monthly_long_distance_charges: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Cobrança média mensal de longa distância. Zero significa que o cliente "
            "não tem telefonia, e é diferente de omitir o campo, que significa que "
            "o dado não veio."
        ),
    )
    services_multiple_lines: str | None = Field(default=None, description=_SIM_NAO)
    services_internet_type: str | None = Field(
        default=None,
        description=(
            "Valores da base: 'Cable', 'DSL' ou 'Fiber Optic'. Use null para "
            "cliente sem internet, que vira a categoria 'No Internet Service' "
            "dentro do pipeline (EngenhariaEstrutural)."
        ),
    )
    services_avg_monthly_gb_download: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Volume médio mensal baixado. Zero significa que o cliente não tem "
            "internet, e é diferente de omitir o campo, que significa que o dado "
            "não veio."
        ),
    )
    services_online_security: str | None = Field(default=None, description=_SIM_NAO)
    services_online_backup: str | None = Field(default=None, description=_SIM_NAO)
    services_device_protection_plan: str | None = Field(default=None, description=_SIM_NAO)
    services_premium_tech_support: str | None = Field(default=None, description=_SIM_NAO)
    services_streaming_tv: str | None = Field(default=None, description=_SIM_NAO)
    services_streaming_movies: str | None = Field(default=None, description=_SIM_NAO)
    services_streaming_music: str | None = Field(default=None, description=_SIM_NAO)
    services_unlimited_data: str | None = Field(default=None, description=_SIM_NAO)
    services_contract: str | None = Field(
        default=None,
        description="Valores da base: 'Month-to-Month', 'One Year' ou 'Two Year'. Opcional.",
    )
    services_paperless_billing: str | None = Field(default=None, description=_SIM_NAO)
    services_payment_method: str | None = Field(
        default=None,
        description=(
            "Valores da base: 'Bank Withdrawal', 'Credit Card' ou 'Mailed Check'. Opcional."
        ),
    )
    services_monthly_charge: float | None = Field(default=None, ge=0)
    services_total_charges: float | None = Field(default=None, ge=0)
    services_total_refunds: float | None = Field(default=None, ge=0)
    services_total_extra_data_charges: float | None = Field(default=None, ge=0)

    def to_dataframe(self) -> pd.DataFrame:
        """Converte o payload na linha única que o pipeline espera.

        A coerção de tipo é explícita porque uma coluna numérica que chega
        nula vira `object` na inferência do pandas, e aí o comportamento
        passaria a depender de quanto o sklearn tolera em vez do que a gente
        decidiu. Passar pelo mesmo método garante também que `GET /sample` e
        `POST /predict` entreguem exatamente a mesma matriz ao modelo.
        """
        dados = pd.DataFrame([self.model_dump()])
        for coluna in _COLUNAS_NUMERICAS:
            dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
        return dados


_COLUNAS_NUMERICAS = tuple(
    nome
    for nome, campo in ChurnRequest.model_fields.items()
    if campo.annotation in (int | None, float | None)
)


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


class ClienteExemplo(BaseModel):
    """Um cliente da amostra do `GET /sample`, com payload e resultado juntos."""

    nome: str = Field(description="Identificador curto do perfil.")
    descricao: str = Field(description="O que esse perfil demonstra.")
    payload: ChurnRequest = Field(
        description="Corpo exato para reproduzir o resultado em POST /predict."
    )
    resultado: ChurnResponse = Field(description="Saída do modelo para esse payload.")


class SampleResponse(BaseModel):
    """Resposta do `GET /sample`."""

    total: int = Field(description="Quantidade de clientes na amostra.")
    threshold: float = Field(description="Threshold usado para converter probabilidade em classe.")
    model_version: str = Field(description="Versão do modelo/pacote usada nas predições.")
    clientes: list[ClienteExemplo]
