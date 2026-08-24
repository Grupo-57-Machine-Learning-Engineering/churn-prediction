"""Amostra de clientes de exemplo servida pelo `GET /sample`.

São perfis fixos escritos à mão, e não linhas sorteadas do parquet, por
dois motivos. O parquet não é versionado, então um exemplo tirado dele
quebraria em qualquer máquina que ainda não rodou o ETL. E cliente real,
mesmo de base pública, não é o tipo de coisa que se devolve num endpoint
aberto de demonstração.

Cada perfil existe para mostrar um comportamento específico da API, os
dois extremos do risco e as três decisões de validação do ADR-006. Os
valores seguem o domínio do Contrato 1 e as faixas observadas na EDA
(`docs/eda-findings.md`), então a leitura de risco é plausível, mas eles
não reproduzem nenhum cliente específico da base.
"""

from __future__ import annotations

CLIENTES_EXEMPLO: list[dict] = [
    {
        "nome": "mensal_fibra_recente",
        "descricao": (
            "Contrato mês a mês, fibra, pouco tempo de casa e sem serviços de "
            "proteção. Combinação de maior risco segundo a EDA."
        ),
        "payload": {
            "demographics_gender": "Male",
            "demographics_age": 27,
            "demographics_senior_citizen": "No",
            "demographics_married": "No",
            "demographics_dependents": "No",
            "demographics_number_of_dependents": 0,
            "services_number_of_referrals": 0,
            "services_tenure_in_months": 3,
            "services_phone_service": "Yes",
            "services_avg_monthly_long_distance_charges": 22.5,
            "services_multiple_lines": "No",
            "services_internet_type": "Fiber Optic",
            "services_avg_monthly_gb_download": 68,
            "services_online_security": "No",
            "services_online_backup": "No",
            "services_device_protection_plan": "No",
            "services_premium_tech_support": "No",
            "services_streaming_tv": "Yes",
            "services_streaming_movies": "Yes",
            "services_streaming_music": "No",
            "services_unlimited_data": "Yes",
            "services_contract": "Month-to-Month",
            "services_paperless_billing": "Yes",
            "services_payment_method": "Bank Withdrawal",
            "services_monthly_charge": 94.4,
            "services_total_charges": 283.2,
            "services_total_refunds": 0.0,
            "services_total_extra_data_charges": 0,
        },
    },
    {
        "nome": "fidelizado_dois_anos",
        "descricao": (
            "Contrato de dois anos, cinco anos de casa, pacote completo de "
            "serviços. Perfil de menor risco."
        ),
        "payload": {
            "demographics_gender": "Female",
            "demographics_age": 52,
            "demographics_senior_citizen": "No",
            "demographics_married": "Yes",
            "demographics_dependents": "Yes",
            "demographics_number_of_dependents": 2,
            "services_number_of_referrals": 4,
            "services_tenure_in_months": 62,
            "services_phone_service": "Yes",
            "services_avg_monthly_long_distance_charges": 11.2,
            "services_multiple_lines": "Yes",
            "services_internet_type": "DSL",
            "services_avg_monthly_gb_download": 18,
            "services_online_security": "Yes",
            "services_online_backup": "Yes",
            "services_device_protection_plan": "Yes",
            "services_premium_tech_support": "Yes",
            "services_streaming_tv": "No",
            "services_streaming_movies": "No",
            "services_streaming_music": "No",
            "services_unlimited_data": "No",
            "services_contract": "Two Year",
            "services_paperless_billing": "No",
            "services_payment_method": "Credit Card",
            "services_monthly_charge": 65.3,
            "services_total_charges": 4048.6,
            "services_total_refunds": 0.0,
            "services_total_extra_data_charges": 0,
        },
    },
    {
        "nome": "sem_internet",
        "descricao": (
            "Só telefonia. `services_internet_type` nulo vira a categoria "
            "'No Internet Service' e o zero em GB baixado marca a ausência do "
            "serviço, que é o fator de proteção mais forte da base."
        ),
        "payload": {
            "demographics_gender": "Female",
            "demographics_age": 71,
            "demographics_senior_citizen": "Yes",
            "demographics_married": "Yes",
            "demographics_dependents": "No",
            "demographics_number_of_dependents": 0,
            "services_number_of_referrals": 1,
            "services_tenure_in_months": 40,
            "services_phone_service": "Yes",
            "services_avg_monthly_long_distance_charges": 8.7,
            "services_multiple_lines": "No",
            "services_internet_type": None,
            "services_avg_monthly_gb_download": 0,
            "services_online_security": "No",
            "services_online_backup": "No",
            "services_device_protection_plan": "No",
            "services_premium_tech_support": "No",
            "services_streaming_tv": "No",
            "services_streaming_movies": "No",
            "services_streaming_music": "No",
            "services_unlimited_data": "No",
            "services_contract": "One Year",
            "services_paperless_billing": "No",
            "services_payment_method": "Mailed Check",
            "services_monthly_charge": 20.1,
            "services_total_charges": 804.0,
            "services_total_refunds": 0.0,
            "services_total_extra_data_charges": 0,
        },
    },
    {
        "nome": "ficha_incompleta",
        "descricao": (
            "Só o que o time comercial tinha em mãos. Os campos ausentes são "
            "imputados pelo pipeline, então a predição sai, mas apoiada no "
            "perfil mediano do treino no lugar do que falta."
        ),
        "payload": {
            "services_contract": "Month-to-Month",
            "services_tenure_in_months": 5,
            "services_monthly_charge": 88.0,
            "services_internet_type": "Fiber Optic",
        },
    },
    {
        "nome": "categoria_nova",
        "descricao": (
            "Plano semanal e meio de pagamento que não existiam no treino. "
            "A API pontua em vez de recusar, e o encoder trata os dois valores "
            "como desconhecidos (ADR-006)."
        ),
        "payload": {
            "demographics_gender": "Male",
            "demographics_age": 34,
            "demographics_senior_citizen": "No",
            "demographics_married": "No",
            "demographics_dependents": "No",
            "demographics_number_of_dependents": 0,
            "services_number_of_referrals": 0,
            "services_tenure_in_months": 8,
            "services_phone_service": "Yes",
            "services_avg_monthly_long_distance_charges": 14.0,
            "services_multiple_lines": "No",
            "services_internet_type": "Fiber Optic",
            "services_avg_monthly_gb_download": 45,
            "services_online_security": "No",
            "services_online_backup": "No",
            "services_device_protection_plan": "No",
            "services_premium_tech_support": "No",
            "services_streaming_tv": "Yes",
            "services_streaming_movies": "No",
            "services_streaming_music": "No",
            "services_unlimited_data": "Yes",
            "services_contract": "Weekly",
            "services_paperless_billing": "Yes",
            "services_payment_method": "Pix",
            "services_monthly_charge": 79.9,
            "services_total_charges": 639.2,
            "services_total_refunds": 0.0,
            "services_total_extra_data_charges": 0,
        },
    },
]
