"""
core/logging_config.py
----------------------
Structured JSON logging pipeline for VectraCore RAG.

Configures ``structlog`` for machine-readable output in production and
pretty console output in development.  Import ``get_logger`` in every
module instead of using ``print`` statements.
"""

import logging
import sys

import structlog
from structlog.types import EventDict, Processor

from core.config import settings


def _add_app_context(
    logger: logging.Logger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Inject constant app-level fields into every log record."""
    event_dict["app"] = "VectraCore RAG"
    event_dict["environment"] = settings.environment
    return event_dict


def configure_logging() -> None:
    """
    Configure structlog once at application startup.

    - Production: JSON renderer → stdout (ingested by log aggregators).
    - Development: colourised console renderer for readability.
    """
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_app_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_production:
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(settings.log_level.upper()),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a named bound logger.  Usage: ``log = get_logger(__name__)``."""
    return structlog.get_logger(name)
