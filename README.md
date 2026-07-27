# Churn Prediction

Previsão de churn de clientes (dataset Telco Customer Churn / IBM) com Scikit-Learn — comparando modelos lineares, baseados em árvores e rede neural (`MLPClassifier`) — e API de inferência FastAPI.

## Estrutura

Os módulos de `src/` estão criados mas ainda vazios — o comentário indica a
responsabilidade planejada de cada um.

```
churn-prediction/
├── src/
│   ├── data/            # (a implementar) carregamento e split
│   ├── features/        # (a implementar) transformadores custom do pipeline
│   ├── models/          # (a implementar) baseline, ensemble e MLPClassifier
│   ├── training/        # (a implementar) treino, validação cruzada e comparação
│   ├── api/             # (a implementar) FastAPI: rotas, schemas Pydantic
│   └── config.py        # seeds, paths, constantes
├── data/                # NÃO versionado (só local)
├── models/              # artefatos treinados (não versionado)
├── tests/               # pytest
├── notebooks/           # EDA e exploração
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
