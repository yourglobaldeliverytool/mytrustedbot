"""
APEX SIGNAL™ — Structured Logging Utility
Uses structlog for production-grade structured logging.
"""
import structlog
import logging
import sys
from apex_signal.config.settings import get_settings


def setup_logging() -> None:
    """Configure structured logging for the entire application."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a named structured logger."""
    return structlog.get_logger(name or "apex_signal")