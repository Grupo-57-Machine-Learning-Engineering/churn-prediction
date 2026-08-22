"""Testes do pipeline de ETL: extração, carga e orquestração.

Nenhum teste toca a rede. O `urlopen` é substituído por um duplo que devolve
bytes conhecidos, o que permite exercitar o download, o pulo de arquivo já
existente e a propagação de erro sem depender do servidor da IBM estar no ar.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from src.data import extract, load, pipeline
from src.data.etl_config import ARQUIVOS_ALVO, BASE_URL, OUTPUT_FILENAME

CONTEUDO = b"conteudo-falso-de-xlsx"


class _RespostaFalsa:
    """Duplo do objeto devolvido por `urlopen`, usado como context manager."""

    def __init__(self, conteudo: bytes = CONTEUDO):
        self._conteudo = conteudo

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._conteudo


@pytest.fixture
def urlopen_falso(monkeypatch):
    """Intercepta o download e registra as URLs pedidas."""
    chamadas: list[str] = []

    def _urlopen(req, *args, **kwargs):
        chamadas.append(req.full_url)
        return _RespostaFalsa()

    monkeypatch.setattr(extract.urllib.request, "urlopen", _urlopen)
    return chamadas


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------


def test_criar_diretorios_e_idempotente(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    monkeypatch.setattr(extract, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(extract, "PROCESSED_DATA_DIR", processed)

    extract.criar_diretorios()
    extract.criar_diretorios()  # segunda vez não pode explodir

    assert raw.is_dir()
    assert processed.is_dir()


def test_baixar_arquivo_grava_o_conteudo(tmp_path, urlopen_falso):
    nome = ARQUIVOS_ALVO[0]

    caminho = extract.baixar_arquivo(nome, destino_dir=tmp_path)

    assert caminho == tmp_path / nome
    assert caminho.read_bytes() == CONTEUDO
    assert urlopen_falso == [f"{BASE_URL}{nome}"]


def test_baixar_arquivo_pula_o_que_ja_existe(tmp_path, urlopen_falso):
    """Reexecutar o ETL não deve rebaixar o que já está em disco."""
    nome = ARQUIVOS_ALVO[0]
    destino = tmp_path / nome
    destino.write_bytes(b"ja-estava-aqui")

    caminho = extract.baixar_arquivo(nome, destino_dir=tmp_path)

    assert caminho.read_bytes() == b"ja-estava-aqui"
    assert urlopen_falso == [], "não pode ter chamado a rede"


def test_baixar_arquivo_propaga_erro_de_rede(tmp_path, monkeypatch):
    """Falha de download não pode passar silenciosa: o parquet sairia incompleto."""

    def _explodir(req, *args, **kwargs):
        raise urllib.error.URLError("sem rede")

    monkeypatch.setattr(extract.urllib.request, "urlopen", _explodir)

    with pytest.raises(urllib.error.URLError):
        extract.baixar_arquivo(ARQUIVOS_ALVO[0], destino_dir=tmp_path)

    assert not (tmp_path / ARQUIVOS_ALVO[0]).exists(), "não deve deixar arquivo pela metade"


def test_baixar_arquivos_ibm_cobre_as_cinco_planilhas(tmp_path, urlopen_falso):
    caminhos = extract.baixar_arquivos_ibm(destino_dir=tmp_path)

    assert len(caminhos) == len(ARQUIVOS_ALVO) == 5
    assert {c.name for c in caminhos} == set(ARQUIVOS_ALVO)
    assert len(urlopen_falso) == 5
    assert all(c.read_bytes() == CONTEUDO for c in caminhos)


def test_baixar_arquivos_ibm_aceita_uma_lista_menor(tmp_path, urlopen_falso):
    caminhos = extract.baixar_arquivos_ibm([ARQUIVOS_ALVO[0]], destino_dir=tmp_path)

    assert len(caminhos) == 1
    assert len(urlopen_falso) == 1


def test_lista_padrao_de_arquivos_nao_e_compartilhada(tmp_path, urlopen_falso):
    """Default mutável: mexer na lista de uma chamada não pode afetar a próxima."""
    primeira = extract.baixar_arquivos_ibm(destino_dir=tmp_path)
    primeira.clear()

    assert len(extract.baixar_arquivos_ibm(destino_dir=tmp_path)) == 5
    assert len(ARQUIVOS_ALVO) == 5


def test_destino_do_download_resolve_na_chamada(tmp_path, urlopen_falso, monkeypatch):
    """Apontar o projeto para outro diretório de dados tem que ter efeito.

    O padrão do parâmetro é `None`, resolvido no corpo. Se voltasse a ser
    `destino_dir=RAW_DATA_DIR` na assinatura, o caminho ficaria congelado no
    import e este teste escreveria no `data/raw` de verdade.
    """
    outro = tmp_path / "outro_lugar"
    outro.mkdir()
    monkeypatch.setattr(extract, "RAW_DATA_DIR", outro)

    caminho = extract.baixar_arquivo(ARQUIVOS_ALVO[0])

    assert caminho.parent == outro


def test_arquivos_alvo_sao_as_cinco_bases_da_ibm():
    """O ADR-003 descartou o CSV do Kaggle: a fonte são as 5 planilhas."""
    assert len(ARQUIVOS_ALVO) == 5
    assert all(nome.endswith(".xlsx") for nome in ARQUIVOS_ALVO)
    assert BASE_URL.startswith("https://")


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------


def test_salvar_parquet_ida_e_volta(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    destino = tmp_path / "sub" / "saida.parquet"

    caminho = load.salvar_parquet(df, destino)

    assert caminho == destino
    assert pd.read_parquet(caminho).equals(df)


def test_salvar_parquet_cria_o_diretorio_de_destino(tmp_path):
    destino = tmp_path / "nao" / "existe" / "ainda" / "saida.parquet"

    load.salvar_parquet(pd.DataFrame({"a": [1]}), destino)

    assert destino.exists()


def test_salvar_parquet_nao_grava_o_indice(tmp_path):
    """Índice do pandas não é dado: gravá-lo criaria uma coluna fantasma."""
    df = pd.DataFrame({"a": [1, 2]}, index=["primeira", "segunda"])

    caminho = load.salvar_parquet(df, tmp_path / "saida.parquet")

    assert list(pd.read_parquet(caminho).columns) == ["a"]


def test_salvar_parquet_usa_o_destino_padrao_do_projeto(tmp_path, monkeypatch):
    monkeypatch.setattr(load, "PROCESSED_DATA_DIR", tmp_path)

    caminho = load.salvar_parquet(pd.DataFrame({"a": [1]}))

    assert caminho == tmp_path / OUTPUT_FILENAME


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


def test_executar_pipeline_encadeia_as_quatro_etapas(tmp_path, monkeypatch):
    """Ordem importa: criar diretórios, baixar, unir e só então salvar."""
    ordem: list[str] = []

    bases = {
        nome: pd.DataFrame({"Customer ID": ["1"], "Zip Code": [90001]})
        for nome in ("demographics", "locations", "populations", "services", "status")
    }
    unido = pd.DataFrame({"customer_id": ["1"]})

    monkeypatch.setattr(pipeline, "criar_diretorios", lambda: ordem.append("diretorios"))
    monkeypatch.setattr(pipeline, "baixar_arquivos_ibm", lambda: ordem.append("download"))
    monkeypatch.setattr(
        pipeline, "_carregar_bases_brutas", lambda: (ordem.append("leitura"), bases)[1]
    )

    def _unir(**kwargs):
        ordem.append("uniao")
        assert set(kwargs) == set(bases), "as cinco bases têm que chegar nomeadas"
        return unido

    def _salvar(df):
        ordem.append("salvamento")
        assert df is unido
        return tmp_path / OUTPUT_FILENAME

    monkeypatch.setattr(pipeline, "unir_bases", _unir)
    monkeypatch.setattr(pipeline, "salvar_parquet", _salvar)

    pipeline.executar_pipeline()

    assert ordem == ["diretorios", "download", "leitura", "uniao", "salvamento"]


def test_carregar_bases_brutas_le_as_cinco_planilhas(tmp_path, monkeypatch):
    lidos: list[Path] = []

    def _read_excel(caminho, *args, **kwargs):
        lidos.append(Path(caminho))
        return pd.DataFrame({"Customer ID": ["1"]})

    monkeypatch.setattr(pipeline, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(pipeline.pd, "read_excel", _read_excel)

    bases = pipeline._carregar_bases_brutas()

    assert set(bases) == {"demographics", "locations", "populations", "services", "status"}
    assert len(lidos) == 5
    assert {c.name for c in lidos} == set(ARQUIVOS_ALVO)


def test_unir_bases_avisa_quando_o_merge_explode(caplog):
    """CEP duplicado em populations multiplica linhas: tem que aparecer no log."""
    import logging

    from src.data.transform import unir_bases

    def _base(**colunas):
        return pd.DataFrame(colunas)

    locations = _base(**{"Customer ID": ["1", "2"], "Zip Code": [90001, 90002]})
    populations = _base(**{"Zip Code": [90001, 90001, 90002], "Population": [10, 20, 30]})
    vazia = _base(**{"Customer ID": ["1", "2"]})

    with caplog.at_level(logging.WARNING):
        unido = unir_bases(
            demographics=vazia.copy(),
            locations=locations,
            populations=populations,
            services=vazia.copy(),
            status=vazia.copy(),
        )

    assert len(unido) == 3, "o CEP repetido duplica a linha do cliente 1"
    assert "explosão many-to-many" in caplog.text
