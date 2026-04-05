import hashlib
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field

from querymind.schemas.models import QueryResult

class CacheEntry(BaseModel):
    result: QueryResult
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hit_count: int = 0
    
    model_config = ConfigDict(strict=True)

class CacheStats(BaseModel):
    total_entries: int
    total_hits: int
    oldest_entry_at: datetime | None
    cache_ttl_seconds: int
    
    model_config = ConfigDict(strict=True)

class QueryCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, CacheEntry] = {}

    def _make_key(self, nl_query: str, schema_ddl: str) -> str:
        combined = nl_query.strip().lower() + "|" + schema_ddl
        return hashlib.sha256(combined.encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        age = (datetime.now(timezone.utc) - entry.created_at).total_seconds()
        return age > self.ttl_seconds

    def get(self, nl_query: str, schema_ddl: str) -> QueryResult | None:
        key = self._make_key(nl_query, schema_ddl)
        entry = self._store.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            del self._store[key]
            return None
        entry.hit_count += 1
        return entry.result

    def set(self, nl_query: str, schema_ddl: str, result: QueryResult) -> None:
        key = self._make_key(nl_query, schema_ddl)
        self._store[key] = CacheEntry(result=result)

    def invalidate_all(self) -> int:
        count = len(self._store)
        self._store.clear()
        return count

    def stats(self) -> CacheStats:
        total_hits = sum(e.hit_count for e in self._store.values())
        oldest = min(
            (e.created_at for e in self._store.values()),
            default=None
        )
        return CacheStats(
            total_entries=len(self._store),
            total_hits=total_hits,
            oldest_entry_at=oldest,
            cache_ttl_seconds=self.ttl_seconds
        )
