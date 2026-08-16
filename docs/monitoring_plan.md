# Plano de Monitoramento

> Como saberemos que o modelo continua saudável em produção.

Citado no ML Canvas e no ADR-004 de `docs/decisions.md`. Escopo do Tech Challenge Fase 1:
plano documentado, não implementação (deploy em nuvem é entrega opcional).

## O que monitorar

- **Data drift:** distribuição das features de entrada vs. treino (ex.: `services_tenure_in_months`,
  `services_monthly_charge`, `services_contract`).
- **Target/concept drift:** taxa de churn observada vs. prevista ao longo do tempo.
- **Performance:** PR-AUC (métrica primária, ver `notebooks/03_preparacao.ipynb` §9) e
  ROC-AUC quando os rótulos reais chegarem (label delay).
- **Qualidade dos dados:** schema `pandera` (`src/data/schema.py`) falhando, NaN inesperado
  fora dos nulos legítimos conhecidos (`services_offer`, `services_internet_type`),
  categorias novas nas colunas categóricas do Contrato 1.
- **Operacional:** latência do `/predict`, taxa de erro (4xx/5xx), throughput.

## Como monitorar

- Logging estruturado na API (entrada validada, probabilidade, versão do modelo — ver
  `model_version` no Contrato 3).
- Métricas de experimento e runs no **MLflow (DagsHub)**, mantido por decisão do grupo
  (ADR-004).
- Checagem periódica de schema com `pandera` (`src/data/schema.py`) nos dados que entram.

## Gatilhos e ações

| Sinal                               | Ação                                           |
| ------------------------------------ | ------------------------------------------------ |
| Drift relevante numa feature-chave  | Investigar; considerar re-treino                |
| Queda de PR-AUC/recall               | Re-treinar com dados recentes                   |
| Schema/validação `pandera` falhando | Bloquear inferência; alertar a trilha de dados  |
| Latência/erros acima do limite      | Investigar API/infra                            |

## Limitação conhecida: sem horizonte temporal

Ver ADR-005. `probability` no Contrato 3 mede propensão a um comportamento de churn
observável no snapshot atual, não risco em uma janela futura (ex.: "próximos 3 meses").
Monitorar drift/performance não substitui essa limitação — um modelo estável ainda não
diz "quando" o cliente sinalizado tende a cancelar, só que o perfil dele se parece com o
de quem já cancelou.

## Re-treino

<!-- Frequência, dados usados, critério de promoção do novo modelo. -->
