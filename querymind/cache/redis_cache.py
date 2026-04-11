import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisQueryCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            logger.warning("redis package not installed. Falling back to in-memory cache.")
            self._fallback: dict = {}
            self._redis = None
            return
        try:
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            self._fallback = None
            logger.info(f"RedisQueryCache connected to {redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to in-memory cache.")
            self._redis = None
            self._fallback = {}
        self.ttl = 3600

    async def get(self, key: str) -> Optional[dict]:
        if self._redis is None:
            return self._fallback.get(key)
        try:
            value = await self._redis.get(f"qcache:{key}")
            return json.loads(value) if value else None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    async def set(self, key: str, value: dict) -> None:
        if self._redis is None:
            self._fallback[key] = value
            return
        try:
            await self._redis.setex(f"qcache:{key}", self.ttl, json.dumps(value))
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    async def clear(self) -> None:
        if self._redis is None:
            self._fallback.clear()
            return
        try:
            keys = await self._redis.keys("qcache:*")
            if keys:
                await self._redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")

    async def stats(self) -> dict:
        if self._redis is None:
            return {"backend": "in-memory", "total_cached_queries": len(self._fallback)}
        try:
            keys = await self._redis.keys("qcache:*")
            return {"backend": "redis", "total_cached_queries": len(keys)}
        except Exception as e:
            logger.warning(f"Redis stats failed: {e}")
            return {"backend": "redis-error", "total_cached_queries": 0}
