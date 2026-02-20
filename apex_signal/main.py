"""
APEX SIGNAL™ — Main Entry Point
Orchestrates the full signal generation pipeline.
"""
import asyncio
import uvicorn
from apex_signal.config.settings import get_settings
from apex_signal.utils.logger import setup_logging, get_logger

logger = get_logger("main")


def run_api():
    """Run the FastAPI application."""
    settings = get_settings()
    setup_logging()
    logger.info("starting_apex_signal", version=settings.version, port=settings.port)
    uvicorn.run(
        "apex_signal.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run_api()