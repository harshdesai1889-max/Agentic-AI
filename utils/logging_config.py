import logging
import os
import sys

_LOGGER_NAME = "day223_agents"


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root logger for the day223_agents application."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Retrieve the application logger."""
    return logging.getLogger(_LOGGER_NAME)

