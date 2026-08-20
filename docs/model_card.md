# Model Card — Churn Prediction

> Documentação do modelo em produção: o que ele faz, como foi avaliado, onde ele erra e
> o que ainda precisa de decisão humana antes de operar. Ver também [`ml_canvas.md`](ml_canvas.md)
> (visão de produto) e [`decisions.md`](decisions.md) (ADR-004, histórico completo de
> experimentos).

## 1. Resumo

| | |
|---|---|
| Tarefa | Classificação binária: propensão de um cliente ativo cancelar o serviço (churn) |
| Algoritmo | `RandomForestClassifier` (scikit-learn), tunado via Optuna |
| Métrica de decisão do campeão | PR-AUC médio de validação cruzada (5-fold) |
| Data do treino final | 2026-08-20 |
| Artefato | `models/champion_model.joblib` e `models:/churn_champion@champion` (MLflow/DagsHub) |
| Dataset | Telco Customer Churn (IBM), 7.043 clientes, 26,5% churn |

## 2. Uso pretendido

**Para quê:** apoiar a equipe de retenção a priorizar contato proativo (oferta, upgrade
de suporte) entre clientes ativos com perfil parecido com quem já cancelou. Ver proposta
de valor completa em `ml_canvas.md` §1.

**Para quem:** consumido via API (`POST /predict`, Etapa 3) por sistemas internos da
operadora, não por clientes finais diretamente.

**Fora do escopo:**
- Não estima **quando** o cliente vai cancelar. A saída é uma propensão observável no
  snapshot atual dos dados, sem horizonte temporal definido (ver ADR-005 em
  `decisions.md`) — não é "risco nos próximos 3 meses".
- Não foi validado para justificar decisões automatizadas sem revisão humana (ex.:
  suspender benefício, negar renovação). Uso pretendido é **priorização de contato**, não
  decisão automática sobre o cliente.
- Não foi testado fora do perfil do dataset de treino (ver §6, Limitações).

## 3. Dados de treino

- Fonte: Telco Customer Churn (IBM), 5 planilhas oficiais unidas por `customer_id`/`zip_code`
  (ADR-003 em `decisions.md`). Achados completos da EDA em `eda-findings.md`.
- 7.043 clientes, 1 linha por cliente. Split treino/teste 80/20 estratificado,
  `random_state=42` (`src/config.py`), idêntico entre baseline e candidatos.
- Taxa de churn: 26,5% — desbalanceamento tratado via `class_weight="balanced"`
  (Random Forest) ou `sample_weight` (MLP), sem reamostragem.
- 454 clientes `Joined` (1-3 meses de casa, sem tempo de cancelar) mantidos no treino,
  rotulados como não-churn — limitação de censura documentada no ADR-005.

### Features usadas (28, ver Contrato 3 em `decisions.md`)

Perfil demográfico (idade, dependentes, casado, idoso), contrato e cobrança (tipo de
contrato, forma de pagamento, mensalidade, cobrança acumulada), serviços contratados
(internet, telefonia, add-ons de segurança/suporte, streaming) e tempo de casa
(`tenure`). As mais associadas ao churn, por força de efeito observado na EDA: tipo de
contrato, tenure, tipo de internet e ausência de add-ons de suporte
(`eda-findings.md` §"Variáveis mais associadas ao churn").

### Features explicitamente excluídas (vazamento de dados)

| Coluna | Motivo |
|---|---|
| `status_satisfaction_score` | Separação quase perfeita com o alvo (100% churn nas notas 1-2); nota atribuída junto/depois do cancelamento |
| `status_churn_score` | Saída de um modelo/score anterior da IBM sobre o mesmo alvo (correlação 0,661) |
| `status_churn_category`, `status_churn_reason` | Só existem para quem já cancelou |
| `status_cltv` | Proveniência de cálculo desconhecida, sem garantia de não usar informação pós-evento |
| `services_offer` | Fora do pipeline por suspeita de vazamento (a oferta pode ter sido atribuída em resposta a um sinal de risco já observado) — não confirmado nem descartado, ver `eda-findings.md` |

Detalhamento completo em `eda-findings.md` §"Vazamento de dados".

## 4. Avaliação e performance

Métrica primária: **PR-AUC** (insensível ao desbalanceamento, foca na classe churn — uma
acurácia de 73,5% já é o piso de "prever ninguém cancela", ver `ml_canvas.md` §6).

### Comparação de candidatos (CV, 5-fold estratificado)

| Modelo | Método | PR-AUC (CV) | F1 (CV) | AUC-ROC (CV) |
|---|---|---|---|---|
| **Random Forest (campeão)** | Optuna | **0,7751** | 0,7023 | 0,9013 |
| Random Forest | GridSearchCV | 0,7750 | 0,7050 | 0,9011 |
| Random Forest | RandomizedSearchCV | 0,7748 | 0,7008 | 0,9008 |
| MLP (sem peso de classe) | Optuna | 0,7709 | 0,6902 | 0,9018 |
| MLP (balanceado) | Optuna | 0,7682 | 0,6913 | 0,9000 |
| Regressão Logística (baseline) | — | 0,757 | 0,679 | 0,896 |

Critério de escolha decidido *a priori*: maior PR-AUC médio de CV. O teste foi avaliado
uma única vez, só para o vencedor.

### Resultado no teste (avaliado uma vez)

| Modelo | F1 | AUC-ROC | PR-AUC |
|---|---|---|---|
| Baseline (Regressão Logística) | 0,694 | 0,908 | **0,766** |
| **Campeão (Random Forest / Optuna)** | **0,706** | 0,905 | 0,761 |

**Achado a não esconder:** no teste, o baseline tem PR-AUC e AUC-ROC ligeiramente
*maiores* que o campeão — diferença pequena o suficiente para caber dentro do ruído entre
folds (desvio-padrão de PR-AUC de CV do Random Forest = 0,0123). Os dois modelos têm
desempenho estatisticamente comparável neste split; o Random Forest venceu por decisão
tomada antes de olhar o teste, não por um resultado inequívoco. Detalhes em
`notebooks/05_modelagem.ipynb` §9.

### Benchmark de vazamento (referência, não comparável diretamente)

O `status_churn_score` pré-calculado pela IBM (não usado como feature, ver §3) tem
correlação de 0,661 com o alvo — equivalente a um ROC-AUC muito alto se usado diretamente.
O campeão, treinado sem esse vazamento, chega a AUC-ROC de teste = 0,905 usando só
features legítimas e disponíveis no momento real da previsão.

## 5. Métricas de negócio (threshold = 0,5, conjunto de teste)

Matriz de confusão do campeão: VN=852, FP=183, FN=70, VP=304.

| Métrica | Valor | O que significa |
|---|---|---|
| Sensibilidade (recall churn) | 0,813 | De cada 10 clientes que realmente cancelam, o modelo identifica ~8 |
| Especificidade (recall não-churn) | 0,823 | De cada 10 clientes que ficam, o modelo classifica ~8 corretamente como sem risco |
| Precisão (VPP) | 0,624 | De cada 10 clientes sinalizados como risco, ~6 realmente cancelam |
| VPN | 0,924 | De cada 10 clientes sinalizados como "sem risco", ~9 realmente ficam |

**Trade-off entre variantes do MLP** (referência para decisão de threshold/operação):
o MLP sem peso de classe é o mais conservador (sensibilidade ~0,65, precisão ~0,72,
poucos falsos alarmes, perde mais churner de verdade); balancear via `sample_weight`
desloca o comportamento para perto do Random Forest (sensibilidade ~0,85, precisão
~0,58). Como perder um churner tende a custar mais que um contato de retenção
desnecessário, a operação mais "agressiva" (alta sensibilidade) tende a ser preferível —
mas essa é uma decisão de negócio, não uma conclusão puramente técnica (ver §7).

## 6. Limitações conhecidas

- **Sem horizonte temporal.** A saída é uma propensão no snapshot atual, não uma
  probabilidade de cancelar "nos próximos N meses" (ADR-005). Monitorar performance ao
  longo do tempo não resolve essa limitação estrutural.
- **Geografia única.** O dataset cobre um único estado (Califórnia) e país (EUA); o
  modelo não foi validado para outras regiões/mercados.
- **`services_offer` fora do pipeline.** Se a suspeita de vazamento (§3) for descartada
  numa investigação futura, essa coluna pode conter sinal forte hoje não aproveitado.
- **Escala de dados moderada** (~5,6 mil linhas de treino). Pode não favorecer redes
  neurais mais profundas; o resultado do MLP aqui não deve ser extrapolado para datasets
  maiores.
- **Sem calibração de probabilidade.** `probability` na resposta da API é a saída bruta
  do `RandomForestClassifier.predict_proba`, não uma probabilidade calibrada — não
  deveria ser interpretada como "X% de chance real de cancelar" sem calibração adicional
  (`CalibratedClassifierCV`, no backlog de melhorias futuras).
- **Diferença campeão vs. baseline é pequena no teste** (§4) — ambos são candidatos
  defensáveis; a escolha do Random Forest reflete o critério de CV definido a priori, não
  uma vitória inequívoca.

## 7. Considerações éticas e de viés

O modelo usa atributos demográficos como entrada (`demographics_gender`,
`demographics_age`, `demographics_senior_citizen`, `demographics_married`,
`demographics_dependents`) porque a EDA mostrou sinal preditivo real neles (ex.: cliente
idoso quase dobra o risco de churn; ter dependentes reduz o risco de 32,6% para 6,5% —
`eda-findings.md`). Usar esse sinal para *priorizar contato de retenção* é razoável; usar
para **diferenciar o tratamento** que um cliente recebe (ex.: oferta pior para idosos, por
"terem mais propensão a sair de qualquer forma") seria um uso indevido do modelo e uma
fonte de discriminação indireta.

**Recomendação:** antes de qualquer uso que vá além de "priorizar quem a equipe de
retenção contata primeiro" — por exemplo, personalizar o valor de uma oferta com base na
predição — o grupo deveria auditar se o modelo produz disparidade de tratamento entre
grupos demográficos protegidos, o que não foi feito nesta rodada.

## 8. Como carregar e usar

```python
import joblib

pipeline = joblib.load("models/champion_model.joblib")
pipeline.predict_proba(dados_do_cliente)  # dados no formato do Contrato 1 (pós-ETL)
```

Ou via MLflow Model Registry (não depende de saber o número da versão):

```python
import mlflow

modelo = mlflow.pyfunc.load_model("models:/churn_champion@champion")
```

Contrato de entrada/saída da API (`/predict`), ver `decisions.md` §"Contrato 3":

```jsonc
// Response
{
  "churn": true,
  "probability": 0.78,
  "model_version": "0.1.0"
}
```

## 9. Decisão pendente: threshold de classificação

`0,5` é o padrão do scikit-learn, não uma escolha de negócio. O custo de um falso
negativo (perder um churner) tende a ser maior que o de um falso positivo (contato de
retenção desnecessário), o que sugere um threshold mais baixo que 0,5 — mas o custo
relativo real não foi quantificado pelo grupo (ver `ml_canvas.md` §8, ainda pendente de
decisão). Enquanto isso não é decidido, `0,5` é o threshold em uso.

## 10. Reprodutibilidade

- Seed fixa (`SEED=42`, `src/config.py`), split idêntico entre baseline e candidatos.
- Ambiente travado via `uv.lock` (`pyproject.toml` é a fonte única de dependências).
- Todos os experimentos (baseline + 10 runs de tuning + comparação final + campeão)
  registrados no MLflow/DagsHub, experimento `churn-prediction`.
- Notebook de treino: `notebooks/05_modelagem.ipynb` (reexecução headless reproduz os
  mesmos números, já que split e seeds são determinísticos).
- Hiperparâmetros finais do campeão (`RandomForestClassifier`, escolhidos pelo Optuna):
  `n_estimators=450`, `max_depth=30`, `min_samples_leaf=7`, `class_weight="balanced"`,
  `random_state=42`.
