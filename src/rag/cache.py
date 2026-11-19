import hashlib
import logging

import redis

from src.config import settings
from src.rag.schemas import JobSearchResponse

logger = logging.getLogger(__name__)

# Keys include the collection name, so results from a different index never mix
KEY_PREFIX = f"search:{settings.qdrant_collection}:"


def normalize_query(query: str) -> str:
    """Lowercase, trim and collapse spaces."""
    return " ".join(query.lower().split())


class ResultCache:
    """Redis cache of returned search result.

    Any Redis problem is logged and treated as a cache miss.
    """

    def __init__(self, url: str | None = None, ttl_seconds: int | None = None):
        url = url or settings.redis_url
        self.ttl_seconds = ttl_seconds or settings.cache_ttl_seconds
        self.client = (
            redis.Redis.from_url(url, socket_timeout=1, socket_connect_timeout=1)
            if url
            else None
        )

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def key(self, query: str) -> str:
        digest = hashlib.sha256(normalize_query(query).encode()).hexdigest()
        return KEY_PREFIX + digest

    def get(self, query: str) -> JobSearchResponse | None:
        """Saved response for this query; None on a miss or if Redis is down."""
        if not self.enabled:
            return None
        try:
            data = self.client.get(self.key(query))
        except redis.RedisError as e:
            logger.warning("Cache read failed, running search instead: %s", e)
            return None
        return JobSearchResponse.model_validate_json(data) if data else None

    def set(self, query: str, response: JobSearchResponse) -> None:
        """Save a response; it expires after ttl_seconds."""
        if not self.enabled:
            return
        try:
            self.client.set(
                self.key(query), response.model_dump_json(), ex=self.ttl_seconds
            )
        except redis.RedisError as e:
            logger.warning("Cache write failed: %s", e)

    def clear(self) -> int:
        """Delete all cached searches. Returns how many were removed."""
        if not self.enabled:
            return 0
        removed = 0
        for key in self.client.scan_iter(match=KEY_PREFIX + "*", count=500):
            removed += self.client.delete(key)
        return removed

    def ping(self) -> bool:
        """True if Redis answers."""
        if not self.enabled:
            return False
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False
