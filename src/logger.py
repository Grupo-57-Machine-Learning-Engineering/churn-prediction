"""Logger utilitário centralizado — configuração única, reutilizável por todos os módulos."""

import logging

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger configurado uma única vez (formato padrão do projeto).

    Args:
        name: nome do módulo chamador (tipicamente __name__).

    Returns:
        Instância de logging.Logger configurada.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
