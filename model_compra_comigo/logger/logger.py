"""
Logger implementation .
"""

from logging import Logger, getLogger, StreamHandler, INFO
from os import environ
from pythonjsonlogger.jsonlogger import JsonFormatter


def create_logger(
    logger_level=INFO,
) -> Logger:
    """
    Creates a logger .

    Parameters
    ----------
    logger_level : str
        Logger level .

    Returns
    -------
    Logger
        Logger .

    """
    logger = getLogger()
    handler = StreamHandler()
    formatter = JsonFormatter(
        "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] "
        + "%(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    level = environ.get("LOG_LEVEL", logger_level)
    logger.setLevel(level)
    return logger


logger = create_logger()
