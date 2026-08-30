"""Treino da Etapa 2: dados, candidatos, tuning, comparação e campeão.

Ordem de uso típica:

1. `dataset.carregar_base_modelagem` + `dataset.dividir_treino_teste`
2. `tuning.rodar_tuning` por candidato, com a fábrica e o espaço de
   `estimators`. Para avaliar um pipeline sem busca de hiperparâmetro,
   `metrics.avaliar_por_cv`.
3. `comparison.buscar_baseline` + `comparison.montar_tabela_comparativa` +
   `comparison.registrar_comparacao`
4. `champion.selecionar_campeao` + `champion.salvar_campeao` +
   `champion.registrar_campeao`

O baseline da Etapa 1 tem fluxo próprio em `baseline`.

Importe sempre do módulo, não do pacote: `from src.training.metrics import
calcular_metricas`. Não há reexportação aqui de propósito. Uma fachada faria
`import src.training.metrics` arrastar `mlflow` e `optuna` junto, porque o
`__init__` precisaria importar `tuning` e `champion` para reexportá-los.
Medido: 2,68s contra 1,49s para chegar na mesma função.
"""
