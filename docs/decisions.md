# Log de Decisões Técnicas

Registro vivo das decisões do projeto. Alimenta o Model Card. Toda decisão
relevante (arquitetura, dados, contrato) ganha uma entrada aqui.

## Decisões da equipe

| Tema                     | Decisão                                      | Observação                                                                                     |
| ------------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Linguagem                | Python 3.11                                   | Estável e bem suportada por scikit-learn e FastAPI                                              |
| Gerenciador de pacotes   | uv                                            | Lê o `pyproject.toml` (single source of truth)                                                |
| Dataset                  | Telco Customer Churn (IBM)                    | Caminho seguro, atende os requisitos                                                             |
| Famílias comparadas      | Regressão Logística (baseline), Random Forest, `MLPClassifier` | Enunciado atual exige comparar as 3 famílias sklearn (ver ADR-004); nada de rede neural em PyTorch. Campeão real (Etapa 2): Random Forest |
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

Endpoints em `src/api`. Os dois primeiros são exigidos explicitamente pelo enunciado
atual; o terceiro é conveniência de demonstração pedida no review da Etapa 3.

- `GET /health` — liveness simples, sem dependências externas.

  ```jsonc
  // Response
  { "status": "ok" }
  ```
- `POST /predict` — request espelha as 28 features de entrada do Contrato 2 (colunas do
  Contrato 1 que sobrevivem ao descarte padrão). **Todos os campos são opcionais**, ver
  ADR-006.
- `GET /sample` — amostra de clientes de exemplo já pontuados, cada um com o `payload`
  completo e o `resultado` do modelo. Serve para ver a API funcionando sem montar payload
  na mão e para a demonstração da entrega. O `payload` de qualquer item pode ser colado no
  `POST /predict` e devolve exatamente os mesmos números, garantido por teste
  (`test_sample_bate_com_predict`), porque os dois caminhos usam a mesma conversão
  (`ChurnRequest.to_dataframe`) e a mesma função `prever`. Os perfis são fixos, escritos à
  mão dentro do domínio do Contrato 1, e não linhas do parquet: o parquet não é versionado
  e cliente real não é coisa para devolver em endpoint de demonstração. A amostra cobre os
  dois extremos de risco, cliente sem internet, ficha incompleta e categoria desconhecida.

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
  "services_payment_method": "Credit Card",
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

`services_payment_method`: o exemplo anterior usava `"Electronic Check"`, que é domínio do
CSV do Kaggle de 21 colunas (fonte descartada pelo ADR-003). Na base real da IBM os
valores são `"Bank Withdrawal"`/`"Credit Card"`/`"Mailed Check"`.

**Colunas categóricas aceitam qualquer string.** Os valores acima são os que existem na
base e ficam documentados na descrição de cada campo (visíveis no Swagger), mas não são
uma lista fechada: categoria nova passa pela validação e é absorvida pelo
`OneHotEncoder(handle_unknown="infrequent_if_exist")` do pipeline. Decisão do review da
Etapa 3, detalhada no ADR-006.

**Todo campo é opcional.** O exemplo acima é um payload completo, mas nenhum campo é
obrigatório: o que faltar é imputado pelo pipeline. Duas colunas têm ressalva de
significado, `services_avg_monthly_gb_download` e
`services_avg_monthly_long_distance_charges`, onde **zero e ausente querem dizer coisas
diferentes**. Zero significa que o cliente não tem o serviço (é o que alimenta
`flag_sem_internet` e `flag_sem_telefone`, ver `ZEROS_ESTRUTURAIS`), enquanto ausente
significa apenas que o dado não veio. Mandar zero por não saber o valor empurra a
predição para baixo, porque não ter internet é o fator de proteção mais forte da base.

```jsonc
// Response (ChurnResponse)
{
  "churn": true, // bool: previsão de churn
  "probability": 0.78, // float em [0, 1]: propensão a um comportamento de churn observável AGORA
  // (mesma definição do alvo, status_churn_value — snapshot, sem horizonte
  // temporal; não é "probabilidade de cancelar nos próximos N meses", ver ADR-005)
  "model_version": "0.1.0", // versão do PACOTE que respondeu (src.__version__, commitizen).
  // Identifica o código, incluindo o pré-processamento do pipeline.
  "model_source": "mlflow:churn_champion/3", // de onde o campeão foi carregado no startup:
  // "mlflow:churn_champion/<versao>" (Model Registry) ou "joblib-local" (artefato em disco).
  // Existe porque o modelo servido pode mudar sem o pacote mudar de versão (ADR-006).
}
```

O `GET /sample` devolve `model_version` e `model_source` no topo, com os mesmos valores que
aparecem no `resultado` de cada cliente. Os dois são lidos uma vez, no startup: a API não
troca de modelo enquanto está no ar.

- Validação de tipos e faixas via Pydantic: tipo errado, idade ou valor monetário negativo
  devolvem HTTP 422. Categoria de texto desconhecida e campo ausente **não** devolvem 422,
  pontuam normalmente (ADR-006).
- Campos extras também dão 422 (`extra="forbid"`), decisão mantida no ADR-006: em
  particular `services_offer`, em
  quarentena por suspeita de vazamento (notebook 03 §5.2), é rejeitada explicitamente em
  vez de silenciosamente ignorada.
- Implementado na Etapa 3 (`src/api/schemas.py` + `src/api/main.py`, ver ADR-006). O mock
  previsto originalmente para esta trilha deixou de ser necessário.

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
  **Superado pelo ADR-007**: os dois módulos foram populados depois, extraindo esse
  mesmo código do notebook.
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
  baseline e as 12 combinações candidato×método (decidido *a priori*, antes de qualquer
  avaliação no teste). O teste é avaliado uma única vez, só para o candidato vencedor.
- **Resultado desta rodada:** campeão = Random Forest tunado via Optuna (PR-AUC de CV =
  0,7751 ± 0,0123, muito próximo do Grid/RandomizedSearchCV para o mesmo modelo — as 3
  estratégias convergem quando o espaço de busca é pequeno). No teste, o campeão supera o
  baseline em F1 (0,706 vs 0,694) mas fica marginalmente abaixo em PR-AUC (0,761 vs
  0,766) e AUC-ROC (0,905 vs 0,908) — diferença pequena o suficiente para caber dentro do
  ruído entre folds, registrada como achado honesto, não escondida. Nenhuma variante do
  MLP superou o Random Forest ou o baseline. Números completos e todas as runs de tuning
  dos 4 candidatos + baseline + comparação final + campeão estão no MLflow/DagsHub,
  experimento `churn-prediction` — não duplicados aqui para não desatualizar.
- **Regressão Logística também tunada, por checagem de assimetria:** o baseline nunca
  tinha sido tunado (nem na Etapa 1, nem aqui), diferente de RF/MLP que passaram por 9
  configurações tunadas no total. Como o teste mostrou baseline e campeão muito próximos
  (ver acima), essa assimetria virou uma dúvida legítima — tunamos uma versão nova da
  Regressão Logística (`logistic_regression_tunada`, mesma metodologia dos outros
  candidatos, grade de `C` e `penalty`) pra checar. Resultado: PR-AUC de CV de
  0,7574-0,7575, essencialmente igual ao baseline default e bem abaixo do Random Forest
  — o campeão não muda, a assimetria era real mas não escondia resultado diferente.
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

### ADR-006 — Etapa 3: módulo de predição e API de inferência

- **Contexto:** com o campeão definido e registrado (ADR-004), a Etapa 3 popula
  `src/models/` e `src/api/` com o serviço de inferência exigido pelo enunciado
  (`GET /health` + `POST /predict`, mínimo 2 testes automatizados).
- **Decisão (fonte do modelo, MLflow primeiro):** `carregar_campeao()` em
  `src/models/predict.py` tenta `models:/churn_champion@champion` no MLflow/DagsHub
  primeiro, quando o `.env` está configurado, e usa `models/champion_model.joblib` como
  fallback (funciona sem rede e sem credencial). O Registry é a fonte de verdade de qual
  versão está marcada como campeã; priorizá-lo garante que reiniciar a API depois de
  promover um campeão novo já sirva o modelo certo, sem depender de sincronizar o
  `.joblib` manualmente em todo ambiente de deploy. Sem nenhuma das duas fontes a API
  sobe mesmo assim: `/health` responde `ok` e `/predict` devolve 503 com instrução de
  como obter o artefato.
  Custo assumido: o startup passou a depender de rede. O `MLFLOW_HTTP_REQUEST_TIMEOUT`
  (ver `.env.example`) limita a espera, porque um DagsHub lento não levanta exceção nem
  cai no fallback, ele só não termina e segura a subida da API. O startup medido em
  desenvolvimento fica por volta de 7 segundos, número que também importa para o período
  de carência de healthcheck em qualquer deploy conteinerizado. Recarregar o campeão sem
  reiniciar o processo (endpoint administrativo ou checagem periódica do registry) ficou
  de fora: com o modelo sendo treinado à mão, reiniciar é procedimento aceitável.
- **Decisão (threshold de decisão):** mantido em 0,5 (`src/config.py:THRESHOLD_DECISAO`),
  o mesmo com que o campeão foi avaliado (métricas de negócio do ADR-004: sensibilidade
  0,813, precisão 0,624). O ajuste do trade-off entre sensibilidade e precisão segue como
  melhoria futura (notebook 05, seção 10). Se acontecer, muda numa constante única e
  ganha entrada aqui.
- **Decisão (formato de entrada):** a API recebe o cliente já no formato pós-ETL
  (Contrato 3), não no formato cru das 5 planilhas. O artefato salvo é o Pipeline
  completo (`EngenhariaEstrutural`, `DescartadorDeColunas`, `ColumnTransformer`,
  `RandomForest`), então imputação, escala e one-hot acontecem dentro dele, e o
  `DescartadorDeColunas` já foi desenhado para tolerar o payload reduzido de 28 colunas.
  Rodar o ETL de junção dentro da API não faz sentido para predição de cliente único.
- **Decisão (validação):** categórica é `str` livre, numérica é validada por faixa. O
  score de um modelo costuma ser gatilho de sistemas a jusante, e recusar o request
  porque um valor de categoria é novo (ex: operadora lançou plano novo) transforma uma
  questão de dado numa indisponibilidade em cascata, sem necessidade: o
  `OneHotEncoder(handle_unknown="infrequent_if_exist")` do pipeline já absorve categoria
  desconhecida, é o mesmo comportamento do treino. Dado que não descreve cliente nenhum
  (tipo errado, campo faltando, idade ou cobrança negativa) continua devolvendo 422. Os
  valores conhecidos da base seguem documentados na descrição de cada campo, para o
  Swagger continuar guiando quem consome. Custo assumido: a predição com categoria nova
  ignora o significado daquele valor, sem sinalizar isso na resposta — sinalizar é
  trabalho de detecção de data drift, que fica para a etapa de monitoramento
  (`monitoring_plan.md`).
- **Decisão (campos opcionais):** nenhum campo do `POST /predict` é obrigatório. Quem
  consome nem sempre tem a ficha completa do cliente, e o pipeline foi construído para
  isso desde a Etapa 1, com `SimpleImputer(strategy="median", add_indicator=True)` nos
  numéricos e imputação por constante nos categóricos.
  Consequência em `EngenhariaEstrutural` (`src/features/preparation.py`): as flags de
  zero estrutural tratavam nulo como zero, e zero ali significa "não tem o serviço".
  Omitir `services_avg_monthly_gb_download` marcava `flag_sem_internet=1` — o cliente
  entrava no modelo como se não tivesse internet, o fator de proteção mais forte da base
  (7,4% de churn contra 31,8%), enviesando a predição para baixo em silêncio. Corrigido
  para nulo permanecer nulo, com o imputer decidindo. Não altera nada do que o modelo
  aprendeu: nessas colunas a base de treino nunca tem nulo (o zero é real), então só muda
  o comportamento na inferência com dado parcial.
  Custo assumido: quanto menos campo chega, mais a predição se apoia no perfil mediano do
  treino, sem sinalizar isso na resposta — mesmo trabalho de data drift adiado para o
  monitoramento.
- **Decisão (testes sem o artefato real):** `models/champion_model.joblib` não é
  versionado, logo o CI não o tem. Os testes de API e de predição
  (`tests/api/test_api.py`, `tests/models/test_predict.py`) usam um "campeão sintético":
  a mesma factory `build_pipeline()` de produção, fitada em segundos sobre uma base
  sintética com as 28 colunas do Contrato 3 (`tests/conftest.py`), injetada via
  monkeypatch no lugar do carregamento real.
- **Decisão (identificação do modelo servido):** a resposta ganhou `model_source`, campo
  novo, ao lado do `model_version`. O `model_version` continua sendo `src.__version__`
  (commitizen), que identifica o código: o pré-processamento vive em
  `src/features/preparation.py` e é o pacote que determina como o dado chega no
  estimador. O `model_source` diz de qual fonte o campeão veio, com a versão do Registry
  quando aplicável (`mlflow:churn_champion/3` ou `joblib-local`). Motivo: com o registry
  na frente do joblib, o modelo que responde passou a poder mudar sem o pacote mudar de
  versão, então `model_version` sozinho deixou de identificar quem respondeu. Campo novo
  em vez de redefinir o antigo porque as duas informações são reais e diferentes, e
  porque adicionar não quebra quem já lê `model_version`. A versão do alias é resolvida
  em chamada separada ao Registry, best-effort: se ela falhar o modelo continua servindo
  e a origem sai como `mlflow:churn_champion/desconhecida`, já que rótulo não é motivo
  para derrubar predição.
- **Decisão (`extra="forbid"` fica):** campo desconhecido no `POST /predict` continua
  devolvendo 422, mesmo depois de tudo o mais ter afrouxado. A assimetria é proposital.
  Soltar as categóricas protege disponibilidade: valor novo numa coluna conhecida ainda
  descreve o cliente, e o pipeline sabe absorver. Campo desconhecido é outra coisa, é
  alguém mandando dado que a API não modela, e o caso concreto é o `services_offer`, em
  quarentena por suspeita de vazamento (notebook 03 §5.2). Aceitar e ignorar em silêncio
  seria o pior dos dois mundos: quem consome acharia que a informação entrou na predição.
  Custo assumido: cliente que evoluir o schema antes da API toma 422 numa chamada que
  teria funcionado ignorando o campo extra. É o custo que o grupo aceita para o
  `services_offer` nunca entrar por descuido.
- **Consequências:** o Contrato 3 mudou (campo novo na resposta), avisado no canal.
  Como rodar a API está no README, na seção "Rodando a API localmente".

### ADR-007 — Extração do treino e da predição do notebook para `src/`

- **Contexto:** o pré-processamento já vivia em `src/features/preparation.py` desde a Etapa 1,
  e `notebooks/03_preparacao.ipynb` o consome em vez de manter cópia. O treino não: todo o
  código de métricas, fábricas de estimadores, espaços de busca, os três métodos de tuning,
  a comparação via MLflow e o registro do campeão estava dentro de
  `notebooks/05_modelagem.ipynb`, dependendo de variáveis globais (`X_train`, `y_train`,
  `N_SPLITS`, `SCORING`). Nada disso era importável nem testável. `src/training/` e
  `src/models/` estavam vazios, e o README já os marcava como "a implementar".
- **Decisão:** extrair para módulos, um por responsabilidade: `training/dataset.py`,
  `training/metrics.py`, `training/estimators.py`, `training/tuning.py`,
  `training/comparison.py`, `training/champion.py`, `training/baseline.py` e
  `models/predict.py`. Todo estado entra por parâmetro. Optuna e MLflow permanecem exatamente
  como no notebook, incluindo `TPESampler(seed)`, `MedianPruner(n_warmup_steps=1)`, pruning por
  fold e a estrutura de uma run pai com três filhas por candidato.
- **Decisão (paridade como critério de aceite):** a refatoração foi validada rodando o código
  original do notebook e o dos módulos sobre os mesmos dados e a mesma seed, comparando com
  tolerância absoluta de 1e-12. Os três métodos de busca, as quatro métricas de cada um e os
  `best_params` deram idênticos para dois candidatos (Regressão Logística sem peso e MLP com
  `sample_weight`), assim como a tabela comparativa, a seleção do campeão e as métricas de
  negócio. A predição foi comparada contra o artefato real baixado do DagsHub.
- **Decisão (o `__init__.py` não reexporta):** importa-se sempre do módulo
  (`from src.training.metrics import calcular_metricas`). Uma fachada no `__init__` obrigaria
  a importar `tuning` e `champion` para reexportá-los, e aí qualquer import do pacote
  arrastaria `mlflow` e `optuna` junto. Medido: 2,68s contra 1,49s para chegar na mesma função.
- **Decisão (baseline com fábrica própria):** `baseline.criar_baseline` não reaproveita
  `estimators.criar_logistic_regression`, porque aquela força `solver="liblinear"` para o grid
  poder alternar entre penalidade `l1` e `l2`. O baseline do notebook 04 usa o solver padrão
  (`lbfgs`), e unificar mudaria o número de referência da Etapa 1.
- **Divergências deliberadas em relação ao notebook:** (1) `calcular_metricas_negocio` devolve
  `0.0` onde o notebook estouraria com `ZeroDivisionError`, e usa `labels=[0, 1]` para a matriz
  de confusão não degenerar em 1x1; (2) `montar_tabela_comparativa`, `buscar_baseline`,
  `registrar_comparacao` e `selecionar_campeao` ganharam guardas que trocam `KeyError` cru por
  erro explicando a causa; (3) os erros de arquivo ausente dizem o que rodar para gerar o
  artefato. Nenhuma delas altera número no caminho feliz.
- **Consequências:** os notebooks 01, 04 e 05 não foram alterados, então continuam com as
  cópias antigas do mesmo código. Enquanto ninguém trocar as células por imports, existe
  duplicação entre notebook e módulo, e as duas podem divergir. Migrar exige reexecutar os
  notebooks, o que cria runs novas no DagsHub e pode trocar a versão do campeão no Registry.

### ADR-008 — `pr_auc_std` do Optuna não é comparável com o das outras buscas

- **Contexto:** achado de code review durante a extração do ADR-007, herdado do notebook 05.
  Em `_rodar_optuna`, o `pr_auc_std` vem da dispersão **entre os trials** do estudo, enquanto o
  `GridSearchCV` e o `RandomizedSearchCV` logam `std_test_pr_auc`, que é a dispersão da
  configuração vencedora **entre os folds**. As três linhas aparecem lado a lado na comparação
  do MLflow com o mesmo nome de métrica querendo dizer coisas diferentes. Soma-se a isso um
  filtro que não filtra: `[t.value for t in estudo.trials if t.value is not None]` foi escrito
  para pular os trials podados, mas o Optuna preenche o `value` de um trial podado com a última
  pontuação intermediária, então nenhum é removido, e médias de 5 folds ficam misturadas com
  scores de um fold só. Verificado num estudo de 15 trials com `MedianPruner`: 3 podados, nenhum
  com `value` nulo.
- **Magnitude na base real** (Regressão Logística, 5 folds, 5 trials, `SEED=42`):

  | medida | valor |
  |---|---|
  | `pr_auc_std` do grid_search, entre folds | 0,0125431083 |
  | `pr_auc_std` do optuna, entre trials | 0,0622188279 |
  | o mesmo optuna, se medido entre folds | 0,0124789143 |

  O 0,0622 não é instabilidade do modelo: vem de um único trial ruim (0,601) inflando a
  dispersão da busca. Medida entre folds, a configuração vencedora do Optuna varia tanto quanto
  a do grid, que é a leitura que a tabela comparativa sugere ao pôr as duas na mesma coluna.
- **Decisão:** manter o cálculo do notebook. Corrigir mudaria um número já registrado nas runs
  da Etapa 2 e quebraria a paridade bit a bit que é o critério de aceite do ADR-007. Em
  compensação, o desvio entre folds passou a ser logado junto, na run do Optuna, como
  `pr_auc_std_entre_folds`. Ele sai de graça: `_metricas_cv_complementares` já refita a
  configuração vencedora nos 5 folds para calcular F1 e AUC-ROC. Assim nenhum número muda e
  quem precisar comparar os três métodos tem a grandeza certa disponível.
- **Consequências:** a coluna `pr_auc_std` da tabela comparativa continua não sendo comparável
  entre linhas, e quem ler a tabela precisa saber disso. O `pr_auc_mean`, que é o critério de
  escolha do campeão, não é afetado em nenhum cenário. Se o grupo decidir corrigir de vez, o
  caminho é trocar `pr_auc_std` por `pr_auc_std_entre_folds` no `ResultadoTuning` e filtrar os
  trials por `TrialState.COMPLETE`, o que exige reexecutar o notebook 05 e anotar a quebra de
  comparabilidade com as runs antigas do DagsHub.
### ADR-009 — Deploy da API em container

- **Contexto:** deploy em nuvem é entrega opcional (`monitoring_plan.md`, linha 6), então
  a régua aqui é ponto extra e não requisito. O que a Etapa 4 precisa é de um lugar com
  URL permanente para demonstrar a API funcionando.
- **Decisão (o modelo não entra na imagem):** o `.dockerignore` exclui `models/` e a API
  busca o campeão no Registry durante o startup, coerente com o ADR-006. Promover campeão
  novo passa a ser reiniciar o container, não reconstruir a imagem, e um container
  reiniciado nunca serve joblib desatualizado porque ele não existe lá dentro.
  Custo assumido: sem rede ou sem credencial o container não tem fallback nenhum, a API
  sobe e as rotas de predição devolvem 503. É o comportamento desejado: melhor recusar
  explicitamente do que servir um modelo que ninguém sabe qual é.
- **Decisão (plataforma: Render free):** o Hugging Face Spaces era a escolha original e
  caiu. A documentação do Hub passou a exigir plano pago para Space com SDK Docker ou
  Gradio em conta pessoal, e só Static continua gratuito. O sintoma é enganoso: o Space
  fica `Paused` e o restart devolve 403 com "You've reached your cpu-basic quota limit",
  mesmo numa conta com um único Space e nenhum minuto consumido. Não é cota temporária e
  não adianta hospedar na conta de outro integrante, porque a barreira é do plano e não
  da conta. Descartados junto: Koyeb (exige cartão para criar conta, mesmo no free),
  Railway (sem free real), Fly (cartão). O Render free ficou por não pedir cartão e por
  caber na medição de memória abaixo.
  Custo assumido: a instância free dorme depois de 15 minutos sem tráfego e leva perto de
  um minuto para acordar, com 0,1 vCPU. Para uma demonstração gravada isso se resolve
  aquecendo a URL antes; como serviço de verdade não serviria, e o plano de monitoramento
  parte do princípio de que isso é vitrine, não produção.
- **Decisão (porta lida de `PORT`):** o `CMD` passou a ler `${PORT:-7860}` em shell form
  com `exec`. O 7860 continua sendo o default porque é o que o HF Spaces espera, mas
  plataforma que sorteia a porta injeta `PORT` no ambiente. O `exec` está ali para o
  uvicorn substituir o `/bin/sh` e receber o `SIGTERM` da plataforma direto: sem ele todo
  restart vira kill por timeout.
- **Medições (25/08/2026):** feitas rodando os passos do Dockerfile em Python 3.11 no
  Linux, com o campeão vindo do joblib local.
  - `uv sync --frozen --no-dev` instala em 22s e nenhuma dependência é compilada de
    fonte, tudo vem de wheel. Confirma que a `python:3.11-slim`, sem toolchain de build,
    dá conta.
  - O único build de fonte é o próprio `churn-prediction` pelo hatchling, o que confirma
    que o `COPY` do `README.md` na imagem é obrigatório e não zelo.
  - `.venv` de 898 MB, imagem estimada em torno de 1,05 GB.
  - A API sobe em 2,6s, `/health` responde 200 e `/sample` devolve os cinco clientes.
  - Pico de memória residente de 260 MB, contra os 512 MB da instância free.
  - O startup custa 2,4s de CPU (0,68 `sklearn`, 0,67 `mlflow`, 0,36 `pandas`, 0,22 para
    desserializar o campeão). Num core inteiro isso é rápido; em 0,1 vCPU vira dezenas de
    segundos, e é daí que sai o `--start-period=90s` do healthcheck.
- **Limitação da verificação:** o `docker build` não chegou a rodar em lugar nenhum. O
  ambiente onde as medições foram feitas não alcança registry de container, então as
  camadas de base (`useradd` uid 1000, `chown`, `USER app`, `COPY --from` do uv) seguem
  sem prova. O que está verificado é a instalação a partir do `uv.lock` e a API servindo.
- **Fora de escopo:** trocar `mlflow` por `mlflow-skinny` derruba o `.venv` de 898 MB para
  439 MB, porque o `mlflow` cheio arrasta pyarrow, matplotlib, fontTools e sqlalchemy, que
  a API não usa. Ficou de fora porque mexer em dependência de núcleo perto da entrega
  troca risco real por peso de imagem que ninguém está pagando.
- **Consequências:** `DAGSHUB_REPO_OWNER` e `DAGSHUB_REPO_NAME` ficam vazios no ambiente
  de deploy, de propósito. Preenchidos, o `configurar_mlflow_tracking()` chama
  `dagshub.init()`, que é login interativo por navegador, e container não tem navegador:
  ou falha ou fica pendurado segurando o startup. Só os três do token entram como
  variável de ambiente da plataforma. Como rodar em container está no README, na seção
  "Rodando com Docker".
