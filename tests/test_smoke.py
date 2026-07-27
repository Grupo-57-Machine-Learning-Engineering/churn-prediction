"""Smoke tests — garantem que o pacote importa e as constantes existem.

Mantêm `make test` verde desde o dia 1, mesmo com os módulos ainda vazios.
"""

import src
from src import config


def test_package_importa():
    assert src.__version__


def test_config_tem_seed_e_target():
    assert config.SEED == 42
    assert config.TARGET == "Churn"


def test_project_root_existe():
    assert config.PROJECT_ROOT.is_dir()
