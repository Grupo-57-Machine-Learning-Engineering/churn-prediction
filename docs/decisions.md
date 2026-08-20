# Log de Decisões Técnicas

Registro vivo das decisões do projeto. Alimenta o Model Card. Toda decisão
relevante (arquitetura, dados, contrato) ganha uma entrada aqui.

## Decisões da equipe

| Tema                     | Decisão                                      | Observação                                                                                     |
| ------------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Linguagem                | Python 3.11                                   | Estável e bem suportada por scikit-learn e FastAPI                                              |
| Gerenciador de pacotes   | uv                                            | Lê o `pyproject.toml` (single source of truth)                                                |
| Dataset                  | Telco Customer Churn (IBM)                    | Caminho seguro, atende os requisitos                                                             |
| Modelo principal         | `sklearn.neural_network.MLPClassifier`      | Enunciado atual exige comparar 3 famílias sklearn (ver ADR-004); nada de rede neural em PyTorch |
| Tracking de experimentos | MLflow no DagsHub                             | Mantido por decisão do grupo mesmo não sendo mais exigido pelo enunciado atual (ver ADR-004)   |
| Gestão de tarefas       | GitHub Projects + Issues                      | Mantém tudo junto do código                                                                    |
| Lint / formatação      | ruff                                          | Sem erros é critério de qualidade de código                                                   |
| Testes                   | pytest                                        | Mínimo 2: pré-processamento e status da API                                                    |
| Fluxo de branches / gate | `develop` integração; `main` só via PR | Org free privada → sem rulesets. Gate por hooks locais + CI. Ver [workflow.md](workflow.md)      |

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

**Fonte da coluna "Significado":** dicionário de dados oficial da IBM para o dataset Telco
Customer Churn (as 5 planilhas citadas no ADR-003), publicado em
[community.ibm.com — "Telco Customer Churn"](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).
As observações de vazamento/derivação do alvo vêm da análise empírica do próprio grupo —
`docs/eda-findings.md` e `notebooks/03_preparacao.ipynb` — e não da documentação da IBM.

| Coluna                                         | Tipo         | Significado                                                                              | Observação                                                                                                                                                                                                                                               |
| ---------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `customer_id`                                | object (str) | Identificador único do cliente, chave usada para unir todas as bases                    |                                                                                                                                                                                                                                                            |
| `locations_location_id`                      | object       | Identificador único do registro de localização                                        |                                                                                                                                                                                                                                                            |
| `locations_count`                            | int64        | Contador auxiliar da IBM, sempre igual a 1 por linha                                     |                                                                                                                                                                                                                                                            |
| `locations_country`                          | object       | País do cliente (sempre "United States")                                                |                                                                                                                                                                                                                                                            |
| `locations_state`                            | object       | Estado do cliente (sempre "California")                                                  |                                                                                                                                                                                                                                                            |
| `locations_city`                             | object       | Cidade de residência do cliente                                                         |                                                                                                                                                                                                                                                            |
| `zip_code`                                   | int64        | CEP do cliente, chave de união com a base de população                                |                                                                                                                                                                                                                                                            |
| `locations_lat_long`                         | object       | Latitude e longitude combinadas em uma única string                                     |                                                                                                                                                                                                                                                            |
| `locations_latitude`                         | float64      | Latitude da localização do cliente                                                     |                                                                                                                                                                                                                                                            |
| `locations_longitude`                        | float64      | Longitude da localização do cliente                                                    |                                                                                                                                                                                                                                                            |
| `demographics_count`                         | int64        | Contador auxiliar da IBM, sempre igual a 1 por linha                                     |                                                                                                                                                                                                                                                            |
| `demographics_gender`                        | object       | Gênero do cliente (masculino/feminino)                                                  |                                                                                                                                                                                                                                                            |
| `demographics_age`                           | int64        | Idade do cliente em anos                                                                 |                                                                                                                                                                                                                                                            |
| `demographics_under_30`                      | object       | Indica se o cliente tem menos de 30 anos                                                 |                                                                                                                                                                                                                                                            |
| `demographics_senior_citizen`                | object       | Indica se o cliente tem 65 anos ou mais                                                  |                                                                                                                                                                                                                                                            |
| `demographics_married`                       | object       | Indica se o cliente é casado                                                            |                                                                                                                                                                                                                                                            |
| `demographics_dependents`                    | object       | Indica se o cliente possui dependentes (filhos, pais, avós) morando com ele             |                                                                                                                                                                                                                                                            |
| `demographics_number_of_dependents`          | int64        | Quantidade de dependentes do cliente                                                     |                                                                                                                                                                                                                                                            |
| `services_service_id`                        | object       | Identificador único do registro de serviços                                            |                                                                                                                                                                                                                                                            |
| `services_count`                             | int64        | Contador auxiliar da IBM, sempre igual a 1 por linha                                     |                                                                                                                                                                                                                                                            |
| `services_quarter`                           | object       | Trimestre de referência dos dados de serviço (ex.: "Q3")                               |                                                                                                                                                                                                                                                            |
| `services_referred_a_friend`                 | object       | Indica se o cliente já indicou a operadora para algum amigo                             |                                                                                                                                                                                                                                                            |
| `services_number_of_referrals`               | int64        | Quantidade de indicações feitas pelo cliente                                           |                                                                                                                                                                                                                                                            |
| `services_tenure_in_months`                  | int64        | Tempo de permanência do cliente na operadora, em meses                                  |                                                                                                                                                                                                                                                            |
| `services_offer`                             | object       | Última oferta promocional aceita pelo cliente (ou nenhuma)                              |                                                                                                                                                                                                                                                            |
| `services_phone_service`                     | object       | Indica se o cliente assina serviço de telefonia                                         |                                                                                                                                                                                                                                                            |
| `services_avg_monthly_long_distance_charges` | float64      | Cobrança média mensal com ligações de longa distância                               |                                                                                                                                                                                                                                                            |
| `services_multiple_lines`                    | object       | Indica se o cliente assina múltiplas linhas telefônicas                                |                                                                                                                                                                                                                                                            |
| `services_internet_service`                  | object       | Indica se o cliente assina serviço de internet                                          |                                                                                                                                                                                                                                                            |
| `services_internet_type`                     | object       | Tipo de conexão de internet (DSL, fibra ótica, cabo, etc.)                             |                                                                                                                                                                                                                                                            |
| `services_avg_monthly_gb_download`           | int64        | Volume médio mensal de dados baixados, em GB                                            |                                                                                                                                                                                                                                                            |
| `services_online_security`                   | object       | Indica se o cliente assina o serviço adicional de segurança online                     |                                                                                                                                                                                                                                                            |
| `services_online_backup`                     | object       | Indica se o cliente assina o serviço adicional de backup online                         |                                                                                                                                                                                                                                                            |
| `services_device_protection_plan`            | object       | Indica se o cliente assina o plano de proteção de dispositivos                         |                                                                                                                                                                                                                                                            |
| `services_premium_tech_support`              | object       | Indica se o cliente assina suporte técnico premium (menor tempo de espera)              |                                                                                                                                                                                                                                                            |
| `services_streaming_tv`                      | object       | Indica se o cliente usa o serviço de streaming de TV                                    |                                                                                                                                                                                                                                                            |
| `services_streaming_movies`                  | object       | Indica se o cliente usa o serviço de streaming de filmes                                |                                                                                                                                                                                                                                                            |
| `services_streaming_music`                   | object       | Indica se o cliente usa o serviço de streaming de música                               |                                                                                                                                                                                                                                                            |
| `services_unlimited_data`                    | object       | Indica se o cliente assina o plano de dados ilimitados                                   |                                                                                                                                                                                                                                                            |
| `services_contract`                          | object       | Tipo de contrato do cliente (mensal, um ano ou dois anos)                                |                                                                                                                                                                                                                                                            |
| `services_paperless_billing`                 | object       | Indica se o cliente optou por fatura sem papel (eletrônica)                             |                                                                                                                                                                                                                                                            |
| `services_payment_method`                    | object       | Forma de pagamento utilizada pelo cliente                                                |                                                                                                                                                                                                                                                            |
| `services_monthly_charge`                    | float64      | Valor total cobrado do cliente no mês de referência                                    |                                                                                                                                                                                                                                                            |
| `services_total_charges`                     | float64      | Valor total cobrado do cliente durante todo o tempo como cliente                         |                                                                                                                                                                                                                                                            |
| `services_total_refunds`                     | float64      | Valor total de reembolsos concedidos ao cliente                                          |                                                                                                                                                                                                                                                            |
| `services_total_extra_data_charges`          | int64        | Valor total cobrado por consumo de dados acima do limite do plano                        |                                                                                                                                                                                                                                                            |
| `services_total_long_distance_charges`       | float64      | Valor total cobrado com ligações de longa distância                                   |                                                                                                                                                                                                                                                            |
| `services_total_revenue`                     | float64      | Receita total gerada pelo cliente (cobranças menos reembolsos)                          |                                                                                                                                                                                                                                                            |
| `status_status_id`                           | object       | Identificador único do registro de status                                               |                                                                                                                                                                                                                                                            |
| `status_count`                               | int64        | Contador auxiliar da IBM, sempre igual a 1 por linha                                     |                                                                                                                                                                                                                                                            |
| `status_quarter`                             | object       | Trimestre de referência do status do cliente (ex.: "Q3")                                |                                                                                                                                                                                                                                                            |
| `status_satisfaction_score`                  | int64        | Nota de satisfação do cliente (1 a 5). **Coletada após o desfecho de churn**    | **Excluída das features de treino — vazamento de dados, ver docs/eda-findings.md**                                                                                                                                                                 |
| `status_customer_status`                     | object       | Status atual do cliente: permaneceu, saiu (churn) ou é novo cliente                     | **Excluída das features de treino — reescrita literal do alvo (AUC=1,0 sozinha), ver `notebooks/03_preparacao.ipynb` §5.1 e `src/features/config.py:COLS_DERIVADAS_DO_ALVO`. Usada só pré-split para filtrar censura, nunca como feature.** |
| `status_churn_label`                         | object       | Indica em texto (Yes/No) se o cliente cancelou o serviço no trimestre                   | **Excluída das features de treino — reescrita literal do alvo (AUC=1,0 sozinha), ver `notebooks/03_preparacao.ipynb` §5.1 e `src/features/config.py:COLS_DERIVADAS_DO_ALVO`**                                                                 |
| `status_churn_value`                         | int64        | Versão numérica (0/1) do rótulo de churn, variável alvo do modelo                    |                                                                                                                                                                                                                                                            |
| `status_churn_score`                         | int64        | Score de propensão ao churn (0 a 100) calculado pela IBM a partir de um modelo próprio | **Excluída das features de treino — vazamento de dados, ver docs/eda-findings.md**                                                                                                                                                                 |
| `status_cltv`                                | int64        | Customer Lifetime Value: valor estimado que o cliente gera ao longo do relacionamento    | **Excluída das features de treino — vazamento de dados (estimativa pré-calculada de proveniência desconhecida), ver docs/eda-findings.md**                                                                                                       |
| `status_churn_category`                      | object       | Categoria geral do motivo de cancelamento (ex.: atendimento, concorrência, preço)      | **Excluída das features de treino — vazamento de dados (só existe para quem já cancelou), ver docs/eda-findings.md**                                                                                                                             |
| `status_churn_reason`                        | object       | Motivo específico informado pelo cliente para o cancelamento                            | **Excluída das features de treino — vazamento de dados (só existe para quem já cancelou), ver docs/eda-findings.md**                                                                                                                             |
| `populations_id`                             | int64        | Identificador único do registro de população (por CEP)                                |                                                                                                                                                                                                                                                            |
| `populations_population`                     | int64        | População estimada residente na área do CEP do cliente                                |                                                                                                                                                                                                                                                            |

### Contrato 2 — Interface do Pipeline (sklearn)

- **Entrada:** `pandas.DataFrame` com as colunas de feature do Contrato 1, exceto o target
  (`status_churn_value`) e as colunas descartadas por padrão em
  `src/features/config.py`/`colunas_descartadas()`: constantes, identificadoras, vazamento
  (`status_satisfaction_score`, `status_churn_score`, `status_cltv`, `status_churn_category`,
  `status_churn_reason`), derivadas do alvo (`status_churn_label`, `status_customer_status`),
  redundantes, geografia e `services_offer` (as três últimas reativáveis via flags do
  `build_pipeline`). Na prática, são as 28 colunas usadas no Contrato 3.
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
- `POST /predict` — request espelha as 28 features de entrada do Contrato 2 (colunas do
  Contrato 1 que sobrevivem ao descarte padrão).

```jsonc
// Request  (ChurnRequest)
{
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
  "services_payment_method": "Electronic Check",
  "services_monthly_charge": 79.85,
  "services_total_charges": 958.2,
  "services_total_refunds": 0.0,
  "services_total_extra_data_charges": 0
}
```

`services_internet_type` aceita `null` (cliente sem internet — vira categoria
`"No Internet Service"` dentro do pipeline, ver `EngenhariaEstrutural`). Rótulo em inglês
de propósito: o resto do domínio categórico dessa coluna (`"Cable"`/`"DSL"`/`"Fiber Optic"`)
já vem em inglês da IBM (ver `fix/categoria-nulos-internet-oferta-em-ingles`).

```jsonc
// Response (ChurnResponse)
{
  "churn": true, // bool: previsão de churn
  "probability": 0.78, // float em [0, 1]: propensão a um comportamento de churn observável AGORA
  // (mesma definição do alvo, status_churn_value — snapshot, sem horizonte
  // temporal; não é "probabilidade de cancelar nos próximos N meses", ver ADR-005)
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

**Etapa 2 — Modelagem e Avaliação (`notebooks/05_modelagem.ipynb`):**

- **Escopo mantido em notebook.** `src/models/` e `src/training/` continuam vazios de
  propósito nesta etapa — a população desses módulos fica para a Etapa 3, quando a
  configuração de modelo estiver definitivamente escolhida. Toda a experimentação
  (candidatos, tuning, comparação, registro do campeão) roda em
  `notebooks/05_modelagem.ipynb`.
- **Sem retreino do baseline.** A Regressão Logística da Etapa 1
  (`notebooks/04_baseline.ipynb`, run `baseline_logistic_regression`) não é retreinada —
  o notebook 05 busca essa run existente no MLflow (`mlflow.search_runs`) e a usa como
  referência na comparação, evitando run duplicada.
- **Candidatos novos: 2 famílias, não 3.** `RandomForestClassifier` e `MLPClassifier`
  (o `MLPClassifier` em duas *variantes de treino*: sem peso de classe, e com
  `sample_weight` balanceado no `.fit()`, já que não aceita `class_weight`
  nativamente — decisão do grupo para comparar as duas formas de tratar o
  desbalanceamento, não uma exigência do enunciado). Junto com o baseline de Regressão
  Logística da Etapa 1, isso fecha as 3 famílias exigidas pelo enunciado (Linear,
  Árvore/Ensemble, MLP) — a segunda variante do MLP é uma exploração interna dentro da
  família MLP, não uma quarta família. Sem reamostragem (SMOTE/over/undersampling) —
  `class_weight`/`sample_weight` já cobrem o desbalanceamento (26,5% churn), reamostrar
  por cima duplicaria a correção. Sem feature engineering nova — usa as features que já
  saem de `build_pipeline()`.
- **Tuning com 3 estratégias comparadas** (`GridSearchCV`, `RandomizedSearchCV`,
  `Optuna` — nova dependência de dev em `pyproject.toml`), aplicadas ao Random Forest e
  às 2 variantes de MLP, com o mesmo `StratifiedKFold(5, shuffle=True, random_state=42)`
  em todas. O
  Optuna usa *pruning* fold a fold (`trial.report`/`should_prune` com `MedianPruner`),
  distinto do `early_stopping` interno do `MLPClassifier` (que só controla uma única
  rede).
- **Critério de escolha do campeão:** maior PR-AUC médio de validação cruzada entre o
  baseline e as 9 combinações candidato×método (decidido *a priori*, antes de qualquer
  avaliação no teste). O teste é avaliado uma única vez, só para o candidato vencedor.
- **Resultado desta rodada:** campeão = Random Forest tunado via Optuna (PR-AUC de CV =
  0,7751 ± 0,0123, muito próximo do Grid/RandomizedSearchCV para o mesmo modelo — as 3
  estratégias convergem quando o espaço de busca é pequeno). No teste, o campeão supera o
  baseline em F1 (0,706 vs 0,694) mas fica marginalmente abaixo em PR-AUC (0,761 vs
  0,766) e AUC-ROC (0,905 vs 0,908) — diferença pequena o suficiente para caber dentro do
  ruído entre folds, registrada como achado honesto, não escondida. Nenhuma variante do
  MLP superou o Random Forest ou o baseline. Números completos e todas as runs (10 de
  tuning + baseline + comparação final + campeão) estão no MLflow/DagsHub, experimento
  `churn-prediction` — não duplicados aqui para não desatualizar.
- **Registro do campeão:** `mlflow.sklearn.log_model(..., registered_model_name=
  "churn_champion")` no Model Registry do MLflow/DagsHub, com o alias `champion` apontando
  para a versão vigente (`models:/churn_champion@champion` — API de *stages* do MLflow
  está deprecada desde a 2.9, alias é o substituto atual). Pensado para a Etapa 3, onde a
  API deve poder puxar o modelo de produção direto de lá, sem hardcodar número de versão.
  Com *fallback* gracioso se o Registry não estiver disponível. O `.joblib` do campeão é
  **sempre** gravado em `models/champion_model.joblib` via `joblib.dump`, independente do
  MLflow — entregável explícito do enunciado, não pode depender só do MLflow/DagsHub
  estarem no ar.
- **Métricas de negócio (threshold=0,5, teste):** sensibilidade 0,813, especificidade
  0,823, precisão (VPP) 0,624, VPN 0,924 — ou seja, o campeão captura ~81% dos churners
  reais e, de cada 10 alertas de risco, ~6 realmente cancelariam. Calculadas via matriz de
  confusão contra `models/champion_model.joblib`, também logadas no MLflow em cada run de
  tuning (`cv_sensibilidade`/`cv_especificidade`/`cv_precisao`/`cv_vpn`) para comparação
  entre candidatos — o MLP sem peso de classe é o mais conservador (sensibilidade ~0,65,
  precisão ~0,72); balancear via `sample_weight` desloca seu comportamento para perto do
  Random Forest (sensibilidade ~0,85, precisão ~0,58), relevante dado o custo assimétrico
  de perder um churner (ver `docs/eda-findings.md`).
- **PR-AUC de 0,77 não é baixo:** o piso de um classificador sem informação é a
  prevalência da classe (26,5%) — o campeão está a ~3x esse piso, e o ROC-AUC (0,90) é
  forte para o domínio. O único número "mais alto" citado no projeto
  (`status_churn_score` da IBM, ROC-AUC≈0,94) é vazamento, não uma meta legítima.
- **Melhorias futuras não implementadas nesta rodada** (ver `notebooks/05_modelagem.ipynb`
  §10 para detalhe): busca de hiperparâmetros mais ampla, feature engineering (interações
  contrato×tenure, bucketização), candidatos de gradient boosting
  (HistGradientBoosting/LightGBM/XGBoost/CatBoost), calibração de probabilidade,
  investigar `services_offer`, e ajuste de threshold para o trade-off sensibilidade×precisão.

### ADR-005 — Alvo sem janela temporal definida

- **Contexto:** a base é um snapshot (Q3) do dataset Telco Customer Churn disponibilizado
  pela pós para fins de portfólio — não uma série temporal por cliente. `status_churn_value`
  responde "esse cliente já está em situação de churn no snapshot?", não "vai cancelar nos
  próximos N meses?". `notebooks/03_preparacao.ipynb` §10 deixou isso como pendência, porque
  a definição muda a decisão sobre manter os clientes `Joined` no treino (ver
  `filtrar_censura` em `src/features/preparation.py`) e o significado do campo
  `probability` no Contrato 3.
- **Decisão:** manter o alvo como está — classificar o **comportamento** de churn observável
  no momento do snapshot, sem horizonte temporal. Não redefinir como janela futura.
  Motivos: (1) a base não tem estrutura de série temporal por cliente para sustentar essa
  redefinição sem inventar suposições; (2) o enunciado (`NOVO Tech Challenge Fase 1.pdf`) pede um modelo que "classifique
  clientes com risco de cancelamento", sem exigir horizonte de tempo.
- **Consequências:** `filtrar_censura(remover_joined=False)` continua sendo o padrão,
  como testado empiricamente no notebook 03. O Contrato 3 documenta explicitamente que
  `probability` não tem unidade de tempo. O Model Card deve registrar essa limitação
  (previsão é sobre estado observado, não sobre risco em um horizonte futuro).
