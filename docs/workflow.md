# Workflow de Desenvolvimento

Guia de convenções de branches, commits e pull requests do projeto. Complementa
o [`README.md`](../README.md) (fluxo de branch resumido) com o detalhe do
processo de revisão e os exemplos de commit.

---

## Fluxo de Branches

```
feat/eda-baselines ─┐
fix/scaler-leak ────┼──► develop (integração) ──(PR + CI verde)──► main (estável)
docs/model-card ────┘
```

| Branch | Propósito | Proteção |
| --- | --- | --- |
| `develop` | Integração diária de features, fixes, etc. | Sem proteção formal; gate por CI |
| `main` | Referência estável. Só via PR vindo de `develop`. | Hooks locais + CI guards |

**Regra de ouro:** nunca faça commit ou push direto na `main`. Toda mudança
entra via PR: primeiro `<tipo>/<descrição>` → `develop`, depois `develop` →
`main`.

---

## Naming de Branches

Formato: **`<tipo>/<descrição-curta>`**, kebab-case, sem número de issue no
nome (isso vai na PR).

| Tipo | Quando usar | Exemplo |
| --- | --- | --- |
| `feat/` | Nova funcionalidade | `feat/eda-baselines` |
| `fix/` | Correção de bug ou data leakage | `fix/scaler-leak` |
| `docs/` | Documentação (README, Model Card, specs) | `docs/model-card` |
| `refactor/` | Reorganização sem mudança de comportamento | `refactor/pipeline-modules` |
| `test/` | Testes novos ou ajustes de suite existente | `test/api-smoke` |
| `exp/` | Experimento de modelo (treino, tuning, PoC) | `exp/mlp-architecture` |
| `chore/` | Infra, config, deps, CI/CD | `chore/upgrade-deps` |

✅ **Bom:** `feat/fastapi-predict`, `fix/training-data-schema`, `exp/xgboost-tuning`
❌ **Ruim:** `my-changes`, `feat/123-new-feature` (número de issue no nome), `feat/adiciona-eda-e-refatora-pipeline` (dois assuntos — separe em branches)

---

## Conventional Commits

```
<tipo>(<escopo opcional>): descrição no imperativo

<corpo opcional, se necessário>

<footer opcional (ex: Closes #123)>
```

| Tipo | Quando usar |
| --- | --- |
| `feat` | Nova funcionalidade |
| `fix` | Correção de bug |
| `docs` | Documentação, README, Model Card, specs |
| `refactor` | Mudança de código sem alterar comportamento |
| `test` | Adição ou ajuste de testes |
| `exp` | Experimento de modelo (treino, tuning, PoC) |
| `chore` | Infra, config, dependências, build, CI/CD |
| `perf` | Otimização de performance (raro) |

Escopo indica onde a mudança aconteceu (ex.: `feat(api)`, `fix(features)`,
`test(data)`). Se não há um escopo óbvio, omita.

### Exemplos bons

```
feat(api): adiciona endpoint /predict com validacao Pydantic
```

```
fix(features): corrige data leakage no StandardScaler

O scaler estava sendo fit no dataset inteiro (train + test).
Agora fica fit só no treino e reaplicado na inferência.
```

```
exp(mlp): testa arquitetura 64-32 com dropout 0.3

Baseline com MLP simples: 78% acurácia, loss = 0.45.
Será comparado com árvores no próximo experimento.
```

### Exemplos ruins

❌ `feat: adiciona features, refatora pipeline, corrige scaler` — três coisas em um commit
❌ `fix stuff` — nenhuma descrição
❌ `test: update` — muito vago

### Regra de ouro: commit pequeno e frequente

Se a descrição precisa de **"e"**, provavelmente são **dois commits**.

Benefícios: histórico legível (`git log --oneline` fica claro), reverter é
seguro (um commit = uma coisa), `git bisect` funciona melhor, review fica
focado.

---

## Pull Requests

1. **PR de `<tipo>/<descrição>` para `develop`.** Só depois, quando `develop`
   estiver estável, PR de `develop` para `main`.
2. **Título:** espelha o primeiro commit ou resume a mudança em 50-70
   caracteres.
3. **Descrição:** usa o template em `.github/pull_request_template.md`
   (o que mudou, como testar, checklist).
4. **Tamanho:** mire PRs de até ~400 linhas de mudança. Se ficou grande,
   considere splitar.

### Revisão e aprovação

- Quem revisa? Todo mundo revisa PR de todo mundo, mesmo fora da trilha —
  visão cruzada apanha erros e distribui conhecimento do projeto.
- Mínimo 1 aprovador antes do merge.
- CI precisa estar verde: `lint-and-test` (ruff + pytest) e, para PRs contra
  `main`, o guard que confirma origem em `develop`.

### Depois do merge

- Delete a branch de feature.
- Só o merge de `develop` → `main` é considerado "pronto para deploy/release".

---

## Checklist rápido antes de abrir a PR

- [ ] Branch nomeada `<tipo>/<descrição>`?
- [ ] Cada commit é pequeno, com mensagem Conventional Commits?
- [ ] `make lint` passou?
- [ ] `make test` passou?
- [ ] `make pre-commit` passou?
- [ ] Descrição da PR é clara e tem contexto?
- [ ] PR tem até ~400 linhas?
