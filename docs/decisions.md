# Log de Decisões Técnicas

Registro vivo das decisões do projeto. Alimenta o Model Card. Toda decisão
relevante (arquitetura, dados, contrato) ganha uma entrada aqui.

## Decisões da equipe

| Tema                     | Decisão                                | Observação                                                                                                            |
| ------------------------ | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Linguagem                | Python 3.11                             | Estável e bem suportada por scikit-learn e FastAPI                                                                     |
| Gerenciador de pacotes   | uv                                      | Lê o `pyproject.toml` (single source of truth)                                                                        |
| Dataset                  | Telco Customer Churn (IBM)              | Caminho seguro, atende os requisitos                                                                                   |
| Modelo principal         | `sklearn.neural_network.MLPClassifier`  | Enunciado atual exige comparar 3 famílias sklearn (ver ADR-004); nada de rede neural em PyTorch                       |
| Tracking de experimentos | MLflow no DagsHub                       | Mantido por decisão do grupo mesmo não sendo mais exigido pelo enunciado atual (ver ADR-004)                          |
| Gestão de tarefas        | GitHub Projects + Issues                | Mantém tudo junto do código                                                                                            |
| Lint / formatação        | ruff                                    | Sem erros é critério de qualidade de código                                                                           |
| Testes                   | pytest                                  | Mínimo 2: pré-processamento e status da API                                                                           |
| Fluxo de branches / gate | `develop` integração; `main` só via PR | Org free privada → sem rulesets. Gate por hooks locais + CI. Ver [workflow.md](workflow.md)                           |

---

## Contratos entre módulos

Combinados para que as trilhas (dados, modelo, API) trabalhem em paralelo sem
travar uma na outra. **Mudou um contrato? Atualize aqui e avise no canal antes
de implementar.**

### Contrato 1 — Schema dos dados (após ETL)

Dataset Telco Customer Churn (IBM) — 7043 linhas × 61 colunas após merge das 5 bases.
Processamento: limpeza de nomes de coluna (snake_case), prefixo de base (`locations_`,
`demographics_`, `services_`, `status_`, `populations_`), chaves de união sem prefixo
(`customer_id`, `zip_code`). Validação implementada em `src/data` (schema pandera,
ver `src/data/schema.py`).

| Coluna                                       | Tipo         | Observação |
| --------------------------------------------- | ------------ | ---------- |
| `customer_id`                                 | object (str) |            |
| `locations_location_id`                       | object       |            |
| `locations_count`                             | int64        |            |
| `locations_country`                           | object       |            |
| `locations_state`                             | object       |            |
| `locations_city`                              | object       |            |
| `zip_code`                                    | int64        |            |
| `locations_lat_long`                          | object       |            |
| `locations_latitude`                          | float64      |            |
| `locations_longitude`                         | float64      |            |
| `demographics_count`                          | int64        |            |
| `demographics_gender`                         | object       |            |
| `demographics_age`                            | int64        |            |
| `demographics_under_30`                       | object       |            |
| `demographics_senior_citizen`                 | object       |            |
| `demographics_married`                        | object       |            |
| `demographics_dependents`                     | object       |            |
| `demographics_number_of_dependents`           | int64        |            |
| `services_service_id`                         | object       |            |
| `services_count`                              | int64        |            |
| `services_quarter`                            | object       |            |
| `services_referred_a_friend`                  | object       |            |
| `services_number_of_referrals`                | int64        |            |
| `services_tenure_in_months`                   | int64        |            |
| `services_offer`                              | object       |            |
| `services_phone_service`                      | object       |            |
| `services_avg_monthly_long_distance_charges`  | float64      |            |
| `services_multiple_lines`                     | object       |            |
| `services_internet_service`                   | object       |            |
| `services_internet_type`                      | object       |            |
| `services_avg_monthly_gb_download`            | int64        |            |
| `services_online_security`                    | object       |            |
| `services_online_backup`                      | object       |            |
| `services_device_protection_plan`             | object       |            |
| `services_premium_tech_support`               | object       |            |
| `services_streaming_tv`                       | object       |            |
| `services_streaming_movies`                   | object       |            |
| `services_streaming_music`                    | object       |            |
| `services_unlimited_data`                     | object       |            |
| `services_contract`                           | object       |            |
| `services_paperless_billing`                  | object       |            |
| `services_payment_method`                     | object       |            |
| `services_monthly_charge`                     | float64      |            |
| `services_total_charges`                      | float64      |            |
| `services_total_refunds`                      | float64      |            |
| `services_total_extra_data_charges`           | int64        |            |
| `services_total_long_distance_charges`        | float64      |            |
| `services_total_revenue`                      | float64      |            |
| `status_status_id`                            | object       |            |
| `status_count`                                | int64        |            |
| `status_quarter`                              | object       |            |
| `status_satisfaction_score`                   | int64        | **Excluída das features de treino — vazamento de dados, ver docs/eda-findings.md** |
| `status_customer_status`                      | object       |            |
| `status_churn_label`                          | object       |            |
| `status_churn_value`                          | int64        |            |
| `status_churn_score`                          | int64        | **Excluída das features de treino — vazamento de dados, ver docs/eda-findings.md** |
| `status_cltv`                                 | int64        |            |
| `status_churn_category`                       | object       |            |
| `status_churn_reason`                         | object       |            |
| `populations_id`                              | int64        |            |
| `populations_population`                      | int64        |            |

### Contrato 2 — Interface do Pipeline (sklearn)

- **Entrada:** `pandas.DataFrame` com as colunas de feature do Contrato 1 (sem o target, sem `customer_id`, sem `status_satisfaction_score`/`status_churn_score`).
- **Saída de `transform`:** `numpy.ndarray` (ou DataFrame) com features numéricas prontas para o modelo.
- **Forma:** `sklearn.pipeline.Pipeline` exposto por uma factory em `src/features` (ex.: `build_pipeline() -> Pipeline`).
- **Regra anti-leak:** `fit` só no treino; o scaler/encoder é serializado junto do modelo e reaplicado na inferência.

### Contrato 3 — API (Pydantic)

Endpoints em `src/api`, ambos exigidos explicitamente pelo enunciado atual:

- `GET /health` — liveness simples, sem dependências externas.

  ```jsonc
  // Response
  { "status": "ok" }
  ```

- `POST /predict` — request espelha as features do Contrato 1 (sem as colunas com vazamento de dados).

```jsonc
// Request  (ChurnRequest)
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 79.85,
  "TotalCharges": 958.2,
}
```

```jsonc
// Response (ChurnResponse)
{
  "churn": true, // bool: previsão de churn
  "probability": 0.78, // float em [0, 1]
  "model_version": "0.1.0", // versão do modelo usada
}
```

- Validação de tipos e domínios via Pydantic; categorias inválidas → HTTP 422.
- Enquanto o modelo final não existe, a trilha de API mocka a resposta seguindo este formato.

---

## ADRs

### ADR-001 — Layout `src/` como pacote único

- **Contexto:** o enunciado exige `src/{data,features,models,training,api}` direto sob `src/`.
- **Decisão:** `src/` é o pacote; imports na forma `src.<módulo>`. Build via hatchling (`packages = ["src"]`), pytest com `pythonpath = ["."]`.
- **Consequências:** simples e fiel ao enunciado; evita renomear pastas.

### ADR-002 — Conventional Commits + commitizen

- **Contexto:** padronizar histórico para revisão e geração de changelog.
- **Decisão:** validação no `commit-msg` via pre-commit (commitizen). Tipos: `feat, fix, docs, refactor, test, exp, chore`.
- **Consequências:** commits rejeitados se fora do padrão; histórico legível.

### ADR-003 — Fonte do dataset Telco

- **Contexto:** o dataset Telco Customer Churn está disponível tanto como um CSV único (Kaggle/IBM Community) quanto como cinco planilhas xlsx oficiais publicadas pela IBM (demographics, location, population, services, status), que juntas contêm mais colunas (ex.: dados geográficos e de população por CEP) que o CSV consolidado.
- **Decisão:** baixar e unir as cinco planilhas xlsx diretamente do domínio oficial da IBM (`public.dhe.ibm.com`) via `src/data/extract.py`, em vez de usar o CSV único. A união é feita por `customer_id` (chave primária de cliente) e, para a base de população, por `zip_code` (chave geográfica).
- **Consequências:** pipeline de ingestão mais rico em features (dados de localização e população), porém mais complexo (5 downloads, lógica de merge, tratamento de erro de rede por arquivo) do que um único `read_csv`. Exige testes dedicados de merge (`tests/data/test_transform.py`) para evitar explosão de linhas e bugs de chave, e validação exploratória em notebook (`notebooks/01_validacao_merge.ipynb`).

### ADR-004 — Enunciado vigente e permanência do MLflow/DagsHub

- **Contexto:** este repositório foi iniciado sob um enunciado do Tech Challenge Fase 1 que exigia PyTorch e MLflow/DagsHub. O enunciado vigente (o que vale para avaliação) trocou o modelo principal para `MLPClassifier` do scikit-learn e não exige mais MLflow (tracking pode ser "planilha, log em texto, ou tracking opcional").
- **Decisão:** seguir o enunciado vigente como padrão geral (sem PyTorch, comparação entre Regressão Logística, Random Forest e `MLPClassifier`), mas **manter o MLflow com backend no DagsHub** para tracking de experimentos — não é mais um requisito, mas o grupo optou por preservá-lo por já ser prática validada.
- **Consequências:** `pyproject.toml` mantém as dependências `mlflow` e `dagshub` (ver `chore/mlflow-dagshub-tracking`); `.env.example` e o bloco `MLFLOW_TRACKING_URI`/`MLFLOW_EXPERIMENT_NAME` em `src/config.py` continuam existindo. Documentos que citam MLflow/DagsHub (`monitoring_plan.md`, este arquivo) não precisam de ajuste nesse ponto.
