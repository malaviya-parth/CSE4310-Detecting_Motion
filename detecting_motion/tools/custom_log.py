"""This module provides custom utilities for logging and argument handling."""

import argparse
import logging
import sys

from logging import Logger, LogRecord
from types import TracebackType
from typing import ClassVar


class LoggingArgumentParser(argparse.ArgumentParser):
    """Extended argparse.ArgumentParser that logs errors using a specified logger."""

    def __init__(self, logger: Logger, *args, **kwargs) -> None:
        """Initialize the LoggingArgumentParser class.

        Args:
            logger: The logger instance used for logging errors.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.logger = logger

    def error(self, message: str) -> None:
        """Override the default error method to log parsing errors using the provided logger."""
        formatted_message = f"{self.prog}: error: {message}"
        self.logger.error(formatted_message)  # Log the argparse error message
        self.print_help(sys.stderr)
        self.exit(2, formatted_message + "\n")


class ColoredLogFormatter(logging.Formatter):
    """Custom log formatter that applies colors to different log levels.

    Attributes:
        COLORS (dict): Dictionary mapping log levels to ANSI color codes.
    """

    COLORS: ClassVar[dict] = {
        logging.DEBUG: "\033[0;36m",  # Cyan for DEBUG
        logging.INFO: "\033[0;32m",  # Green for INFO
        logging.WARNING: "\033[0;33m",  # Yellow for WARNING
        logging.ERROR: "\033[0;31m",  # Red for ERROR
        logging.CRITICAL: "\033[1;31m",  # Bold Red for CRITICAL
    }

    def format(self, record: LogRecord) -> str:
        """Format the log record with color.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            str: A color-formatted string based on the log level.
        """
        formatted_record = super().format(record)
        levelno = record.levelno
        return f"{self.COLORS.get(levelno, '')}{formatted_record}\033[0m"  # Reset to default


def configure_logger(name: str | None = None) -> Logger:
    """Setup a logger with custom formatting and a global exception handler.

    Args:
        name (str | None): The name of the logger. If None, the root logger is configured.

    Returns:
        Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = ColoredLogFormatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Global exception handler
    def handle_global_exception(
        exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(f"Unhandled exception: {exc_type}", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_global_exception
    return logger
