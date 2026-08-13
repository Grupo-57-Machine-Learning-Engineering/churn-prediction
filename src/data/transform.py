"""Transformação: limpeza de nomes de colunas e união das bases Telco."""

import re

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

_RE_CARACTERES_ESPECIAIS = re.compile(r"[()[\]/]")
_RE_ESPACO_HIFEN = re.compile(r"[\s\-]+")


def limpar_nome_coluna(coluna: str) -> str:
    """Normaliza um nome de coluna: remove parênteses/colchetes/barras, troca
    espaços/hífens por underscore e converte para minúsculas.

    Args:
        coluna: nome original da coluna.

    Returns:
        Nome normalizado (snake_case simplificado).
    """
    nome = str(coluna).strip()
    nome = _RE_CARACTERES_ESPECIAIS.sub("", nome)
    nome = _RE_ESPACO_HIFEN.sub("_", nome)
    return nome.lower()


def padronizar_colunas(df: pd.DataFrame, prefixo: str, location_key: str) -> pd.DataFrame:
    """Limpa e prefixa as colunas de um DataFrame, preservando as chaves de união
    (customer_id e location_key) sem prefixo.

    Args:
        df: DataFrame de origem (uma das bases Telco).
        prefixo: prefixo a aplicar nas colunas que não são chave (ex.: nome da base).
        location_key: nome já limpo da coluna de chave geográfica (ex.: "zip_code").

    Returns:
        Novo DataFrame com colunas renomeadas (não muta o df de entrada).
    """
    novas_colunas: dict[str, str] = {}
    for col in df.columns:
        col_limpa = limpar_nome_coluna(col)
        if col_limpa in ("customer_id", location_key):
            novas_colunas[col] = col_limpa
        else:
            novas_colunas[col] = f"{prefixo}_{col_limpa}"
    return df.rename(columns=novas_colunas)


def unir_bases(
    demographics: pd.DataFrame,
    locations: pd.DataFrame,
    populations: pd.DataFrame,
    services: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    """Padroniza e une as cinco bases Telco em um único DataFrame.

    Estratégia de união (todas left join a partir de locations, por customer_id):
    locations -> demographics -> services -> status -> populations (por zip_code).

    Args:
        demographics: base de demografia.
        locations: base de localização (contém a chave geográfica).
        populations: base de população por CEP.
        services: base de serviços contratados.
        status: base de status/churn.

    Returns:
        DataFrame unificado, uma linha por customer_id.
    """
    location_key = limpar_nome_coluna("Zip Code")

    dfs = {
        "demographics": padronizar_colunas(demographics, "demographics", location_key),
        "locations": padronizar_colunas(locations, "locations", location_key),
        "services": padronizar_colunas(services, "services", location_key),
        "status": padronizar_colunas(status, "status", location_key),
        "populations": padronizar_colunas(populations, "populations", location_key),
    }

    linhas_antes = len(dfs["locations"])

    merged_df = dfs["locations"].merge(dfs["demographics"], on="customer_id", how="left")
    merged_df = merged_df.merge(dfs["services"], on="customer_id", how="left")
    merged_df = merged_df.merge(dfs["status"], on="customer_id", how="left")
    merged_df = merged_df.merge(dfs["populations"], on=location_key, how="left")

    if len(merged_df) != linhas_antes:
        logger.warning(
            "Shape mudou após os merges: %d linhas antes, %d depois. "
            "Possível explosão many-to-many.",
            linhas_antes,
            len(merged_df),
        )

    logger.info("Bases unidas com sucesso. Shape final: %s", merged_df.shape)
    return merged_df
