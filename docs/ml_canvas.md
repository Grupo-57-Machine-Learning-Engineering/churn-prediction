# ML Canvas — Churn Prediction

> Template a preencher pela equipe. Cada bloco em 1–3 frases objetivas.

## 1. Proposta de valor

O modelo apoia a decisão de **retenção proativa**: sinalizar, entre os clientes ativos,
quais têm perfil de comportamento parecido com o de quem já cancelou, para priorizar
ações da equipe de retenção (contato, oferta, upgrade de suporte) antes do cancelamento
acontecer — hoje essa priorização não existe de forma sistemática. Prever churn importa
porque adquirir cliente novo custa mais do que reter um existente, e a EDA
(`docs/eda-findings.md`) mostra que o risco não é uniforme: concentra-se em contrato
mensal, início da relação (poucos meses de casa) e ausência de add-ons de suporte —
exatamente o segmento onde uma ação de retenção bem direcionada tem mais chance de
funcionar. Cerca de um terço dos cancelamentos declara motivo de atendimento/insatisfação
(não preço), o que reforça que a alavanca não é só desconto.

## 2. Fonte dos dados
- Dataset: **Telco Customer Churn (IBM)**, as 5 planilhas oficiais (ver ADR-003).
- Granularidade: 1 linha por cliente.
- Target: `status_churn_value` (0/1). Ver schema completo em [decisions.md](decisions.md)
  (Contrato 1) e a decisão sobre horizonte temporal no ADR-005.

## 3. Predição
- Tipo: classificação binária (churn vs. não-churn).
- Saída: probabilidade em `[0,1]` + rótulo. Sem horizonte temporal — ver ADR-005.

## 4. Coleta de features

Mesma origem do treino: as 28 features de entrada do `/predict` (Contrato 3) são um
subconjunto direto das mesmas 5 planilhas/schema do treino (Contrato 1), validado pelo
mesmo contrato `pandera` (`src/data/schema.py`) tanto no ETL quanto na API. Não há
feature store nem cálculo derivado em produção: quem chama o `/predict` informa os
atributos atuais do cliente (perfil, contrato, serviços, cobrança) — presumivelmente
extraídos do CRM/sistema de billing da operadora — e o pipeline de features
(`src/features/preparation.py`) aplica exatamente a mesma transformação usada no treino
(engenharia estrutural + descarte de colunas + scaler/encoder serializados junto do
modelo). Isso elimina o risco de train/serve skew por transformação divergente; o risco
que resta é de origem (o CRM real ter uma taxonomia de categorias diferente da do
dataset IBM), não coberto pelos dados disponíveis hoje.

## 5. Construção do modelo
- Baselines: Regressão Logística e Random Forest (ou outro ensemble de árvores).
- Modelo principal: **`MLPClassifier` (scikit-learn)**.
- Comparação entre as três famílias é critério de avaliação.

## 6. Métricas de avaliação (offline)

Definidas em `notebooks/03_preparacao.ipynb` §9, a partir da armadilha da acurácia: com
26,5% de churn na base, um modelo que prevê "ninguém cancela" já acerta 73,5% — qualquer
acurácia reportada sem esse piso ao lado não diz nada.

1. **PR-AUC como métrica primária**: insensível ao desbalanceamento, focada na classe
   positiva (churn), que é a que importa para a decisão de negócio.
2. **ROC-AUC como secundária**: usada para comparar com o benchmark do `status_churn_score`
   pré-calculado pela IBM (ROC-AUC ≈ 0,94 usando só essa coluna) — se o modelo do grupo,
   treinado sem vazamento, não superar esse número, ele não agrega nada ao que a IBM já
   entrega pronto.
3. **Recall na classe positiva como restrição operacional**: churner não identificado é
   cliente perdido; falso positivo é um contato de retenção desnecessário. O custo é
   assimétrico (ver bloco 8).
4. **Acurácia só entra em relatório acompanhada do baseline de 73,5%**, nunca isolada.

Nenhum modelo foi tunado ainda nem o `MLPClassifier` (principal, ver ADR-004) treinado —
os números do notebook 03 são piso, não resultado final.

## 7. Métricas de negócio / monitoramento
- Ver [monitoring_plan.md](monitoring_plan.md).

## 8. Decisões e fazer a predição acontecer

**Pendente de decisão do grupo — não preenchido por julgamento individual.**

O que já está estabelecido (`notebooks/03_preparacao.ipynb` §9): o custo é assimétrico
(perder um cliente pesa mais que um contato de retenção desnecessário), e um exemplo com
`class_weight="balanced"` mostrou o efeito de trocar precisão por recall na classe churn
(recall de 85,0%, precisão de 59,8% — contra recall de 79,3% e precisão de 93,6% na
classe "ficou").

O que falta o grupo decidir antes de ir para produção:
- Threshold de classificação (0,5 é o padrão do sklearn, mas não necessariamente o
  correto aqui dado o custo assimétrico).
- Custo relativo real de falso positivo vs. falso negativo (em termos de negócio, não só
  de métrica) — sem isso, qualquer threshold escolhido é arbitrário.
- Se o threshold é único ou varia por segmento (ex.: CLTV do cliente).

## 9. Reprodutibilidade
- Seed fixa (`src/config.py`), `pyproject.toml`, instala do zero com `uv sync`.
- Runs de experimento registrados no MLflow (DagsHub) — ver ADR-004 em [decisions.md](decisions.md).
