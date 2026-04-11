import os
import logging
import asyncpg

logger = logging.getLogger(__name__)


class ReadOnlyEngine:
    """
    Executes SELECT queries using a dedicated read-only database user.
    Falls back to warning log if read-only DSN is not configured.
    """

    def __init__(self):
        self.dsn = os.getenv("POSTGRES_READ_ONLY_DSN")
        self._pool = None

        if not self.dsn:
            logger.warning(
                "POSTGRES_READ_ONLY_DSN not set. "
                "SQL queries will use the main connection. "
                "Set this in production for security."
            )

    async def _get_pool(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)
        return self._pool

    async def execute(self, sql: str, params: list = None) -> list[dict]:
        """Execute a SELECT query and return rows as list of dicts."""
        if not self.dsn:
            raise RuntimeError(
                "ReadOnlyEngine requires POSTGRES_READ_ONLY_DSN to be set."
            )
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                if params:
                    rows = await conn.fetch(sql, *params)
                else:
                    rows = await conn.fetch(sql)
                return [dict(row) for row in rows]
            except asyncpg.exceptions.InsufficientPrivilegeError:
                logger.error("Read-only user attempted a write operation. Blocked at DB level.")
                raise ValueError("SQL Mode only supports SELECT queries.")
            except Exception as e:
                logger.error(f"ReadOnlyEngine query failed: {e}")
                raise

    async def close(self):
        if self._pool:
            await self._pool.close()
