# Churn Prediction

Previsão de churn de clientes (dataset Telco Customer Churn / IBM) com Scikit-Learn — comparando modelos lineares, baseados em árvores e rede neural (`MLPClassifier`) — e API de inferência FastAPI.

## Estrutura

```
churn-prediction/
├── src/
│   ├── data/            # ETL: extração das 5 planilhas IBM, merge, schema pandera
│   ├── features/        # transformadores custom + build_pipeline() (Contrato 2)
│   ├── models/          # carregamento do campeão e predição (Etapa 3)
│   ├── training/        # (a implementar) treino como script — hoje vive no notebook 05
│   ├── api/             # FastAPI: rotas e schemas Pydantic (Etapa 3)
│   └── config.py        # seeds, paths, constantes
├── data/                # NÃO versionado (só local)
├── models/              # artefatos treinados (não versionado)
├── tests/               # pytest
├── notebooks/           # EDA, preparação de features e modelagem (Etapas 1-2)
├── docs/                # Model Card e documentação do modelo
├── scripts/             # hooks/scripts de suporte (ex.: guard de push)
├── .github/             # workflows (CI + guards) e template de PR
├── pyproject.toml       # single source of truth (deps, ruff, pytest, commitizen)
├── uv.lock              # versões travadas (reprodutibilidade)
└── Makefile
```

## Setup

Pré-requisito: [uv](https://docs.astral.sh/uv/). Python 3.11 é provisionado automaticamente (ver `.python-version`).

```bash
make install      # uv sync + instala hooks de pre-commit (pre-commit/commit-msg/pre-push)
```

### Tracking de experimentos (MLflow / DagsHub)

Opcional (ver ADR-004 em `docs/decisions.md`), mas se quiser registrar métricas de
treino no DagsHub do grupo (`https://dagshub.com/ThiagoZulian/Grupo-57-Machine-Learning-Engineering`
— nome diferente do repo no GitHub), `mlflow` e `dagshub` já vêm com o `make install`.
Copie `.env.example` para `.env` e escolha uma das duas formas de autenticar:

- **Opção A — token**: preencha `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` e
  `MLFLOW_TRACKING_PASSWORD` (token em DagsHub → Settings → Tokens). Sem interação.
- **Opção B — login interativo**: deixe os 3 campos acima em branco. Na primeira run,
  `dagshub.init()` abre um link de autorização no terminal — autorize **assim que o link
  aparecer** (o código expira rápido); o token fica cacheado localmente depois disso.

Sem nenhuma das duas, o treino não quebra: o tracking cai para um backend local
best-effort e só loga um aviso.

**Nota Windows**: se o terminal usar a codepage legada (cp1252), o `dagshub.init()`
pode falhar com `UnicodeEncodeError` ao tentar imprimir o link colorido — já corrigido
em `src/config.py:configurar_mlflow_tracking()`, que força UTF-8 no console antes de
chamar o DagsHub.

## Rodando a API localmente

A API de inferência (Etapa 3) serve o modelo campeão via FastAPI.

Pré-requisito: o modelo campeão. O artefato `models/champion_model.joblib` não é
versionado, e há duas formas de obtê-lo:

1. Rodar o notebook de modelagem (`notebooks/05_modelagem.ipynb`), que grava o joblib
   sempre, com ou sem MLflow no ar; ou
2. Deixar o fallback trabalhar: com o `.env` configurado (ver seção de tracking acima),
   a API baixa `models:/churn_champion@champion` do Model Registry do DagsHub sozinha
   quando não encontra o joblib local.

Sem nenhuma das duas, a API sobe mesmo assim: `GET /health` responde `ok` e
`POST /predict` devolve 503 explicando o que falta.

**Subir o servidor:**

```bash
uv run uvicorn src.api.main:app --reload
```

- Documentação interativa (Swagger): <http://127.0.0.1:8000/docs>. Dá pra testar o
  `/predict` pelo navegador, com o payload de exemplo já preenchido.
- Healthcheck: `curl http://127.0.0.1:8000/health` responde `{"status":"ok"}`

Exemplo de predição (payload completo no Contrato 3 de `docs/decisions.md`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
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
EOF
```

Resposta: `{"churn": true|false, "probability": 0.0 a 1.0, "model_version": "0.1.0"}`.
A probabilidade é a propensão de churn no snapshot atual, sem horizonte temporal
(ADR-005); a classe usa o threshold padrão 0,5 (`src/config.py:THRESHOLD_DECISAO`,
ADR-006). Payload inválido devolve 422: tipo ou domínio errado, campo faltando ou campo
extra, incluindo `services_offer`, que está em quarentena por suspeita de vazamento.

No PowerShell o heredoc acima não existe. Use o Swagger em `/docs`, ou salve o JSON num
arquivo e rode `curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@payload.json"`.

## Comandos (Makefile)

| Comando | O que faz |
|---|---|
| `make install` | Instala dependências e hooks de pre-commit |
| `make lint` | `ruff check` + checagem de formatação |
| `make format` | Formata e auto-corrige com ruff |
| `make test` | Roda os testes (pytest) |
| `make cov` | Testes com cobertura |
| `make pre-commit` | Roda todos os hooks em todos os arquivos |
| `make clean` | Limpa caches (`.pytest_cache`, `.ruff_cache`, `__pycache__`) |

## Fluxo de branch

Duas branches longas: `develop` (integração) e `main` (estável).

- `main` nunca recebe commit/push direto. Só entra código via Pull Request vindo de `develop`, com o CI verde.
- `develop` é a branch do dia a dia: crie uma branch (`feat/*`, `fix/*`, ...) a partir dela e integre de volta.

```
feat/eda-baselines ─┐
fix/scaler-leak ────┼──► develop ──(PR + CI verde)──► main
docs/model-card ────┘
```

Rode `make install` antes de começar: instala os hooks locais que bloqueiam commit e push direto na `main`.

### O gate não é 100% à prova de bala

Repo privado numa org free → GitHub não oferece branch protection/rulesets. Duas camadas de proteção, nenhuma delas é servidor:

- **Hooks locais** (`.pre-commit-config.yaml`): `no-commit-to-branch` bloqueia commit na `main`; `scripts/reject-push-to-main.sh` bloqueia push na `main`. Ambos puláveis com `--no-verify`.
- **CI** (`.github/workflows/`): `ci.yml` roda lint+test; `guard-no-direct-push-main.yml` denuncia (run vermelho, aponta o autor) push direto que escapou dos hooks; `guard-pr-base-develop.yml` recusa PR pra `main` que não venha de `develop`. Nenhum dos dois **impede** — só sinaliza.

Se alguém furar a fila, reverta e reabra via PR da `develop`.

## Nome de branch

`<tipo>/<descrição-curta>` — ex.: `feat/eda-baselines`, `fix/scaler-leak`, `test/api-smoke`.

## Commits — Conventional Commits

`<tipo>(<escopo>): descrição no imperativo`. Validado no `commit-msg` via hook `commitizen`.

| Tipo | Quando usar |
|---|---|
| `feat` | nova funcionalidade |
| `fix` | correção de bug |
| `docs` | documentação |
| `refactor` | mudança de código sem mudar comportamento |
| `test` | adição ou ajuste de testes |
| `build` | build, empacotamento, automação (Makefile) |
| `ci` | workflows do GitHub Actions |
| `chore` | infra, config, dependências |

Exemplos:

```
feat(api): adiciona endpoint /predict com validacao pydantic
fix(features): corrige data leakage no StandardScaler
docs(readme): documenta fluxo de branch
```

## Pull Request

PR de `develop` para `main`: pequeno, `make lint` e `make test` verdes, descrição do que mudou e como testar. O checklist está em `.github/pull_request_template.md` e é preenchido automaticamente ao abrir o PR.
