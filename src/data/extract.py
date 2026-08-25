"""Extração: criação de diretórios e download dos arquivos brutos da IBM."""

import urllib.error
import urllib.request
from pathlib import Path

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.etl_config import ARQUIVOS_ALVO, BASE_URL
from src.logger import get_logger

logger = get_logger(__name__)


def criar_diretorios() -> None:
    """Garante que os diretórios de dados raw/processed existem."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Diretórios verificados/criados: %s, %s", RAW_DATA_DIR, PROCESSED_DATA_DIR)


def baixar_arquivo(nome_arquivo: str, destino_dir: Path | None = None) -> Path:
    """Baixa um único arquivo da IBM, pulando se já existir localmente.

    Args:
        nome_arquivo: nome do arquivo alvo (deve estar em ARQUIVOS_ALVO).
        destino_dir: diretório de destino do download. `None` resolve
            `RAW_DATA_DIR` no momento da chamada, e não no import do módulo,
            para que apontar o projeto para outro diretório de dados tenha
            efeito de fato.

    Returns:
        Path do arquivo local (baixado ou já existente).

    Raises:
        urllib.error.URLError: se o download falhar (rede, DNS, timeout).
        urllib.error.HTTPError: se o servidor retornar erro HTTP (subclasse de URLError).
    """
    destino_dir = Path(destino_dir) if destino_dir is not None else RAW_DATA_DIR
    url_completa = f"{BASE_URL}{nome_arquivo}"
    caminho_destino = destino_dir / nome_arquivo

    if caminho_destino.exists():
        logger.info("Arquivo %s já existe localmente. Pulando download.", nome_arquivo)
        return caminho_destino

    logger.info("Baixando: %s", nome_arquivo)
    req = urllib.request.Request(url_completa, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as response:
            caminho_destino.write_bytes(response.read())
    except urllib.error.URLError:
        logger.exception("Falha ao baixar %s de %s", nome_arquivo, url_completa)
        raise
    logger.info("%s salvo com sucesso em %s", nome_arquivo, caminho_destino)
    return caminho_destino


def baixar_arquivos_ibm(
    arquivos: list[str] | None = None,
    destino_dir: Path | None = None,
) -> list[Path]:
    """Baixa todos os arquivos alvo da IBM.

    Args:
        arquivos: lista de nomes de arquivo a baixar. `None` usa `ARQUIVOS_ALVO`.
        destino_dir: diretório de destino, repassado a `baixar_arquivo`.

    Returns:
        Lista de Paths locais dos arquivos (baixados ou pré-existentes).
    """
    arquivos = list(ARQUIVOS_ALVO) if arquivos is None else arquivos
    logger.info("Iniciando o download dos arquivos diretamente da IBM.")
    return [baixar_arquivo(arquivo, destino_dir) for arquivo in arquivos]
