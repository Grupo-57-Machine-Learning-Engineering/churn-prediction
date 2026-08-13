# ML Canvas — Churn Prediction

> Template a preencher pela equipe. Cada bloco em 1–3 frases objetivas.

## 1. Proposta de valor
<!-- Que decisão de negócio o modelo apoia? Por que prever churn importa? -->

## 2. Fonte dos dados
- Dataset: **Telco Customer Churn (IBM)**.
- Granularidade: 1 linha por cliente.
- Target: `Churn` (`Yes`/`No`). Ver schema em [decisions.md](decisions.md).

## 3. Predição
- Tipo: classificação binária (churn vs. não-churn).
- Saída: probabilidade em `[0,1]` + rótulo.

## 4. Coleta de features
<!-- Como as features chegam em produção? Mesma origem do treino? -->

## 5. Construção do modelo
- Baselines: Regressão Logística e Random Forest (ou outro ensemble de árvores).
- Modelo principal: **`MLPClassifier` (scikit-learn)**.
- Comparação entre as três famílias é critério de avaliação.

## 6. Métricas de avaliação (offline)
<!-- Ex.: ROC-AUC, F1, recall na classe churn. Justifique a escolha. -->

## 7. Métricas de negócio / monitoramento
- Ver [monitoring_plan.md](monitoring_plan.md).

## 8. Decisões e fazer a predição acontecer
<!-- Como a previsão vira ação? Threshold? Custo de falso positivo/negativo? -->

## 9. Reprodutibilidade
- Seed fixa (`src/config.py`), `pyproject.toml`, instala do zero com `uv sync`.
- Runs de experimento registrados no MLflow (DagsHub) — ver ADR-004 em [decisions.md](decisions.md).
