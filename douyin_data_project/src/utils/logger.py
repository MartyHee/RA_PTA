"""
Logging utilities for Douyin data project.

Configures logging using YAML configuration file.
Provides consistent logging across all modules.
"""
import logging
import logging.config
from typing import Optional
from pathlib import Path
import yaml

from .config_loader import load_config


def setup_logging(config_path: Optional[Path] = None, log_level: Optional[str] = None):
    """Setup logging configuration.

    Args:
        config_path: Path to config directory.
        log_level: Override log level.
    """
    config = load_config(config_path)
    logging_config = config.get('logging', {})

    # If logging.yaml exists, use it
    logging_yaml = Path(__file__).parent.parent.parent / 'configs' / 'logging.yaml'
    if logging_yaml.exists():
        try:
            with open(logging_yaml, 'r', encoding='utf-8') as f:
                logging_config_yaml = yaml.safe_load(f)
            logging.config.dictConfig(logging_config_yaml)
            return
        except Exception as e:
            print(f"Failed to load logging.yaml: {e}. Using basic config.")

    # Basic logging configuration
    log_format = logging_config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_format = logging_config.get('date_format', '%Y-%m-%d %H:%M:%S')

    # Determine log level
    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = getattr(logging, logging_config.get('level', 'INFO').upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    if logging_config.get('console_log', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(log_format, date_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if logging_config.get('file_log', False):
        log_dir = Path(logging_config.get('log_dir', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / 'douyin_data_project.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get logger with given name.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    # Ensure logging is setup
    if not logging.getLogger().handlers:
        setup_logging()

    return logging.getLogger(name)


class LoggingMixin:
    """Mixin class to add logging to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__ + '.' + self.__class__.__name__)
        return self._logger


# Context manager for timing
import time
from contextlib import contextmanager


@contextmanager
def log_time(operation: str, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """Context manager to log operation time.

    Args:
        operation: Description of operation.
        logger: Logger instance. If None, uses default.
        level: Log level.
    """
    if logger is None:
        logger = get_logger('timing')

    start_time = time.time()
    logger.log(level, f"Starting: {operation}")

    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.log(level, f"Completed: {operation} (took {duration:.2f}s)")


# Convenience functions
def log_error(message: str, exc_info: bool = True, logger_name: str = 'default'):
    """Log error with exception info.

    Args:
        message: Error message.
        exc_info: Whether to include exception info.
        logger_name: Logger name.
    """
    logger = get_logger(logger_name)
    logger.error(message, exc_info=exc_info)


def log_warning(message: str, logger_name: str = 'default'):
    """Log warning.

    Args:
        message: Warning message.
        logger_name: Logger name.
    """
    logger = get_logger(logger_name)
    logger.warning(message)


def log_info(message: str, logger_name: str = 'default'):
    """Log info.

    Args:
        message: Info message.
        logger_name: Logger name.
    """
    logger = get_logger(logger_name)
    logger.info(message)


def log_debug(message: str, logger_name: str = 'default'):
    """Log debug.

    Args:
        message: Debug message.
        logger_name: Logger name.
    """
    logger = get_logger(logger_name)
    logger.debug(message)