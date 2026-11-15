import logging

from langfuse import Langfuse

from src.config import settings

logger = logging.getLogger(__name__)


def init_tracing() -> Langfuse:
    """Create the Langfuse client once. Tracing stays off if the keys are not set.

    The @observe decorators in the pipeline use this client automatically.
    """
    enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.app_environment,
        tracing_enabled=enabled,
    )
    logger.info(
        "Langfuse tracing %s", "enabled" if enabled else "disabled (no keys set)"
    )
    return client
