# EDA — perfil e motivos do churn

> Achados da análise exploratória (`notebooks/02_eda.ipynb`) sobre quem
> cancela, por que cancela, e quais colunas são vazamento de dados. Base:
> `data/processed/telco_churn_processed.parquet` (7.043 clientes).

## Taxa de churn geral

26,5% (1.869 de 7.043 clientes). 454 clientes (6,4%) são `Joined` (entraram recentemente).

## Variáveis mais associadas ao churn

Ordenadas por força do efeito observado:

1. **`services_contract`**: Month-to-Month 45,8% vs. One Year 10,7% vs. Two Year 2,5%. A interação com tenure intensifica: Month-to-Month nos primeiros 12 meses chega a **53,5%**, e Two Year tem 0% nos primeiros 24 meses.
2. **`services_tenure_in_months`**: mediana de 10 meses para quem cancela vs. 38 meses para quem fica.
3. **`services_internet_service`/`services_internet_type`**: ter internet já eleva o churn de 7,4% para 31,8%; dentro de quem tem, Fiber Optic 40,7% vs. Cable 25,7% vs. DSL 18,6% — o produto de internet concentra quase todo o churn.
4. **`services_payment_method`**: Mailed Check 36,9% e Bank Withdrawal 34,0% vs. Credit Card 14,5%.
5. **Cobrança — o sinal está na mensalidade, não no acumulado**: `services_monthly_charge` correlaciona +0,193 com churn (mediana de US$ 79,65 para quem sai vs. 64,43); os acumulados (`total_charges` −0,199, `total_revenue` −0,223, `total_long_distance_charges` −0,224) são negativos por serem proxies de tenure, não proteção real. `total_refunds`, `total_extra_data_charges` e `avg_monthly_long_distance_charges` não têm sinal (|r| < 0,04).
6. **`demographics_dependents`/`demographics_married`/`demographics_senior_citizen`**: ter dependentes derruba churn de 32,6% para 6,5%; casado de 33,0% para 19,7%; idoso quase dobra o risco (41,7% vs. 23,6%).
7. **Add-ons de suporte/segurança protegem**: `services_online_security` (31,3% sem vs. 14,6% com), `services_premium_tech_support` (31,2% vs. 15,2%) e `services_referred_a_friend` (32,6% vs. 19,4%) reduzem o churn à metade.
8. **`services_offer`**: entre os 45% de clientes com oferta promocional aplicada, "Offer E" tem 52,9% de churn vs. só 6,7% da "Offer A" — sinal forte, mas correlacional (a oferta pode atrair perfil que já cancela mais).
9. **Achado contra-intuitivo**: `services_paperless_billing` (33,6% vs. 16,3%) e `services_unlimited_data` (31,7% vs. 16,0%) têm churn *maior* entre quem tem o serviço — em boa parte porque quem tem essas opções é quem tem internet (grupo de 31,8%), não causal direto.
10. **`demographics_gender`**: sem efeito (26,2% vs. 26,9%) — não é útil como feature.
11. **Geografia**: base cobre um único estado (Califórnia) e um único país (`locations_country` constante); sem variação de estado a explorar. Variação por cidade existe mas com N pequeno (ex.: San Diego 64,9% de churn em 285 clientes) — não confiável sem mais dados.

> `status_satisfaction_score` tinha correlação de -0,755 (o sinal numérico mais forte da base), mas foi **reclassificado como vazamento** — ver seção abaixo.

## Motivo declarado do cancelamento (`status_churn_category`/`status_churn_reason`)

Só preenchido para os 1.869 clientes com `status_customer_status == "Churned"`.

| Categoria | % dos churns |
|---|---|
| Competitor | 45% |
| Attitude | 17% |
| Dissatisfaction | 16% |
| Price | 11% |
| Other | 11% |

Quase metade dos cancelamentos vai para a concorrência ("Competitor had better devices", "Competitor made better offer"). Preço isolado ("Price too high") é só 4% do total — a narrativa de "cancela porque é caro" não é o motivo dominante; falta de fidelidade contratual + experiência (atendimento/satisfação) pesam mais.

## Vazamento de dados — não usar como feature

- **`status_satisfaction_score`** (1-5): separação **perfeita nos extremos** — 100% de churn nas notas 1-2 (1.440 clientes), 0% nas notas 4-5 (2.938 clientes), e nota máxima 3 entre quem cancelou. Nenhuma pesquisa de satisfação medida *antes* do evento separa assim; a nota foi atribuída junto/depois do desfecho. Parecia o melhor preditor da base (correlação -0,755), mas é vazamento.
- **`status_churn_score`** (0-100, pré-calculado pela IBM): correlação de 0,661 com o alvo, médias de 81,8 (churn) vs. 50,1 (não-churn). É a saída de um modelo/score anterior sobre o mesmo alvo — usá-lo como feature é vazamento.
- **`status_churn_category`**, **`status_churn_reason`**: só existem para quem já cancelou; indisponíveis no momento de prever um cliente ativo.
- **`status_cltv`**: preenchido para *todos* os clientes (diferente dos motivos acima), mas é uma estimativa pré-calculada pela IBM com proveniência desconhecida — sem garantia de que o cálculo não usa informação do desfecho, o conservador é deixá-lo fora das features.

### Por que trocar de técnica de modelagem (boosting sequencial, stacking) não resolve

Vazamento de dados é um problema de **disponibilidade do dado no momento da previsão**, não um problema de arquitetura do modelo. Trocar a técnica — por exemplo, usar um boosting sequencial (XGBoost/LightGBM) ou um stacking (usar a saída de um modelo como feature de outro, o que é exatamente o que `status_churn_score` seria se usado como feature) — não muda o fato de que:

1. **Não sabemos como `status_churn_score` foi calculado.** Se ele foi construído com acesso ao rótulo real (ou a informação pós-evento), qualquer modelo que o use — não importa a técnica — está aprendendo a partir do próprio alvo, disfarçado de feature.
2. **Não sabemos se ele estaria disponível para um cliente novo/ativo antes da nossa própria previsão.** É um score de demonstração embutido no dataset de exemplo da IBM, não um serviço de terceiros documentado que recalcula esse valor de forma contínua e independente.

**Stacking/boosting seriam legítimos, sim, mas só sob uma condição**: se `status_churn_score` fosse comprovadamente um score de terceiros, recalculado de forma confiável e independente para *todo* cliente ativo, disponível *antes* do momento em que nosso modelo precisa prever — análogo a um score de crédito externo. Sem essa garantia de proveniência documentada, tratar essa coluna como feature (em qualquer técnica) é vazamento; a diferença entre "feature legítima de terceiros" e "vazamento" está inteiramente na origem e disponibilidade do dado, não no algoritmo que o consome.

**Caminho recomendado**: usar `status_churn_score` como **benchmark de comparação** (o modelo novo deve superar a correlação de 0,661/AUC implícito desse score), não como input. Isso deve virar um critério de aceitação explícito na spec de modelagem futura, não é uma decisão para tomar aqui na EDA.

## Qualidade e data readiness

Checagem de domínio (idade, cobranças, tenure, satisfação, churn score) não encontrou nenhum valor fora de faixa nem linha 100% duplicada — validado com um contrato de dados `pandera` formal em [`src/data/schema.py`](../src/data/schema.py) (testado em `tests/data/test_schema.py`), reutilizado pelo notebook (seção 14). Os pontos de atenção são estruturais, não sujeira:

- **Colunas constantes** (zero variância, mesmo valor em todas as linhas): `services_quarter`, `status_quarter`, `locations_state`, `locations_country`, `locations_count`, `demographics_count`, `services_count`, `status_count`.
- **Colunas identificadoras** (1 valor por cliente, sem sinal): `customer_id`, `locations_location_id`, `services_service_id`, `status_status_id`.
- **Redundâncias** (manter só uma de cada par): `demographics_under_30` ⇔ `demographics_age < 30`; `services_internet_service == "No"` ⇔ `services_internet_type` nulo; `services_referred_a_friend` ⇔ `services_number_of_referrals > 0`. Além disso, duas identidades exatas (diferença máxima 0,0): `services_total_revenue` = `total_charges − refunds + extra_data + long_distance` e `total_long_distance_charges` = `avg_monthly_long_distance_charges × tenure` — colinearidade perfeita se as parcelas também virarem features.
- **Zeros estruturais**: `avg_monthly_gb_download == 0` ⇔ sem internet; `avg_monthly_long_distance_charges == 0` ⇔ sem telefone. O zero codifica ausência do serviço, não consumo baixo — cuidado em escalonamento/imputação.
- **Nulos legítimos** (não são problema de qualidade): `services_offer` (55% nulo = sem oferta aplicada), `services_internet_type` (22% nulo = cliente sem internet), `status_churn_reason`/`status_churn_category` (73% nulo = cliente não cancelou).
- **Censura dos `Joined`**: os 454 clientes `Joined` (tenure 1-3 meses) estão rotulados como "não churn", mas não tiveram tempo de churnar. A spec de modelagem precisa decidir explicitamente se entram no treino, saem, ou viram recorte separado.

## Como aplicar

Ao montar o pipeline de features, usar como núcleo: `services_contract`, `services_tenure_in_months`, `services_internet_type`, `services_payment_method`, `services_monthly_charge`, `demographics_dependents`, `demographics_married`, `demographics_senior_citizen`, `services_online_security`, `services_premium_tech_support`. Excluir explicitamente as colunas de vazamento (incluindo `status_satisfaction_score`), constantes e identificadoras listadas acima. Sem o `satisfaction_score`, o teto de performance realista é mais baixo do que a correlação de -0,755 sugeria — o benchmark honesto passa a ser superar o `churn_score` da IBM (0,661) usando só features legítimas. A modelagem também deve decidir o tratamento dos 454 `Joined` (censura). O contrato de dados que garante essas regras já está formalizado em `src/data/schema.py`; o treino deve chamar `validar(df)` desse módulo antes de treinar.
