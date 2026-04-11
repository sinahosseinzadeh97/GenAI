import json
import logging
from typing import List

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisSessionMemory:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            logger.warning("redis package not installed. Falling back to in-memory memory.")
            self._fallback: dict = {}
            self._redis = None
            return
        try:
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            self._fallback = None
            logger.info(f"RedisSessionMemory connected to {redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to in-memory.")
            self._redis = None
            self._fallback = {}
        self.max_messages = 20
        self.ttl = 86400  # 24 hours

    async def get_history(self, session_id: str) -> List[dict]:
        if self._redis is None:
            return self._fallback.get(session_id, [])
        try:
            raw = await self._redis.get(f"session:{session_id}")
            return json.loads(raw) if raw else []
        except Exception as e:
            logger.warning(f"Redis get_history failed: {e}")
            return []

    async def add_message(self, session_id: str, role: str, content: str, **kwargs) -> None:
        history = await self.get_history(session_id)
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        history.append(msg)
        history = history[-self.max_messages:]
        if self._redis is None:
            self._fallback[session_id] = history
            return
        try:
            await self._redis.setex(
                f"session:{session_id}",
                self.ttl,
                json.dumps(history)
            )
        except Exception as e:
            logger.warning(f"Redis add_message failed: {e}")

    async def clear(self, session_id: str) -> None:
        if self._redis is None:
            self._fallback.pop(session_id, None)
            return
        try:
            await self._redis.delete(f"session:{session_id}")
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")
