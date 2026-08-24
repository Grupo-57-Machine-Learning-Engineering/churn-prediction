"""Carregamento e split da base de modelagem (Etapa 2).

Extraído da seção 2 do `notebooks/05_modelagem.ipynb`. As três operações que
antecedem qualquer treino ficam aqui: ler o parquet processado, aplicar o
filtro de censura e separar o alvo, seguidas do split estratificado.

O split usa `config.SEED` e `config.TEST_SIZE`, os mesmos valores do notebook
04, o que mantém o conjunto de teste idêntico entre as etapas. Trocar
qualquer um dos dois invalida a comparação com o baseline já registrado no
MLflow.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.features.preparation import filtrar_censura, separar_alvo
from src.logger import get_logger

logger = get_logger(__name__)

NOME_PARQUET = "telco_churn_processed.parquet"
"""Arquivo gerado por `src/data/pipeline.py` (alvo do `make etl`)."""

__all__ = ["NOME_PARQUET", "carregar_base_modelagem", "dividir_treino_teste"]


def carregar_base_modelagem(
    caminho: Path | str | None = None,
    *,
    remover_joined: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Lê o parquet processado e devolve `(X, y)` prontos para o split.

    Parameters
    ----------
    caminho
        Parquet alternativo. `None` usa
        `config.PROCESSED_DATA_DIR / NOME_PARQUET`.
    remover_joined
        Repassado para `filtrar_censura`. O padrão `False` mantém os clientes
        `Joined` na modelagem, decisão justificada no docstring daquela função.

    Raises
    ------
    FileNotFoundError
        Se o parquet não existe. O arquivo não é versionado: rode `make etl`.
    """
    caminho = Path(caminho) if caminho is not None else config.PROCESSED_DATA_DIR / NOME_PARQUET

    if not caminho.exists():
        raise FileNotFoundError(
            f"Parquet processado não encontrado em '{caminho}'. O arquivo não é "
            "versionado: gere com `make etl` (ou `uv run python -m src.data.pipeline`)."
        )

    df = pd.read_parquet(caminho)
    df_modelagem, censurados = filtrar_censura(df, remover_joined=remover_joined)
    logger.info(
        "Base de modelagem: %d linhas (%d censuradas removidas).",
        len(df_modelagem),
        len(censurados),
    )
    return separar_alvo(df_modelagem)


def dividir_treino_teste(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split estratificado pelo alvo, na mesma configuração do notebook 04.

    `test_size` e `seed` em `None` resolvem `config.TEST_SIZE` e `config.SEED`
    na hora da chamada. Deixá-los como valor padrão na assinatura congelaria a
    configuração no import do módulo.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    test_size = config.TEST_SIZE if test_size is None else test_size
    seed = config.SEED if seed is None else seed
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
