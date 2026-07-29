import logging
from unittest.mock import patch

import pytest

import src.logger as logger_module
from src.logger import get_logger


@pytest.fixture(autouse=True)
def reset_logger_configurado():
    """Reseta o estado global _CONFIGURED antes de cada teste.

    Evita vazamento de estado entre testes (um teste anterior configurando
    o logger não pode influenciar o resultado do próximo).
    """
    logger_module._CONFIGURED = False
    yield
    logger_module._CONFIGURED = False


def test_get_logger_retorna_logger_com_nome_correto():
    logger = get_logger("meu_modulo")
    assert logger.name == "meu_modulo"


@patch("src.logger.logging.basicConfig")
def test_get_logger_configura_nivel_info(mock_basic_config):
    get_logger("outro_modulo")

    mock_basic_config.assert_called_once()
    _, kwargs = mock_basic_config.call_args
    assert kwargs["level"] == logging.INFO


def test_get_logger_nao_duplica_configuracao():
    with patch("src.logger.logging.basicConfig") as mock_basic_config:
        get_logger("modulo_a")
        get_logger("modulo_b")
        mock_basic_config.assert_called_once()
