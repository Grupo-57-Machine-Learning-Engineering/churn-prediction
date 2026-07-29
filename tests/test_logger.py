import logging
from unittest.mock import patch

from src.logger import get_logger


def test_get_logger_retorna_logger_com_nome_correto():
    logger = get_logger("meu_modulo")
    assert logger.name == "meu_modulo"


@patch("src.logger.logging.basicConfig")
def test_get_logger_configura_nivel_info(mock_basic_config):
    import src.logger as logger_module

    logger_module._CONFIGURED = False

    get_logger("outro_modulo")

    mock_basic_config.assert_called_once()
    _, kwargs = mock_basic_config.call_args
    assert kwargs["level"] == logging.INFO


def test_get_logger_nao_duplica_configuracao():
    import src.logger as logger_module

    logger_module._CONFIGURED = False

    with patch("src.logger.logging.basicConfig") as mock_basic_config:
        get_logger("modulo_a")
        get_logger("modulo_b")
        mock_basic_config.assert_called_once()
