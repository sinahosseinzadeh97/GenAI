import hashlib
from datetime import datetime
from querymind.tools.schema_tool import get_db_schema
from querymind.schemas.models import SchemaChangeEvent, SchemaStatus

class SchemaWatcher:
    """Watches the database for schema changes and tracks its state."""
    
    def __init__(self) -> None:
        self._current_hash: str = ""
        self._last_changed_at: datetime | None = None
        self._change_count: int = 0
        self._last_event: SchemaChangeEvent | None = None

    async def initialize(self) -> None:
        """Call once at startup to set the baseline hash."""
        ddl = await get_db_schema()
        self._current_hash = self._compute_hash(ddl)

    def _compute_hash(self, ddl: str) -> str:
        return hashlib.sha256(ddl.encode()).hexdigest()

    async def check_for_changes(self) -> SchemaChangeEvent | None:
        """
        Recompute schema hash and compare to stored.
        Returns SchemaChangeEvent if changed, None if unchanged.
        """
        ddl = await get_db_schema()
        new_hash = self._compute_hash(ddl)

        if new_hash == self._current_hash:
            return None

        event = SchemaChangeEvent(
            previous_hash=self._current_hash,
            current_hash=new_hash
        )
        self._current_hash = new_hash
        self._last_changed_at = event.detected_at
        self._change_count += 1
        self._last_event = event
        return event

    def get_status(self) -> SchemaStatus:
        """Get the current status of the schema watcher."""
        return SchemaStatus(
            current_hash=self._current_hash,
            last_changed_at=self._last_changed_at,
            change_count=self._change_count,
            is_stale=self._last_event is not None
        )

    def acknowledge_change(self) -> None:
        """Call after handling a change to reset is_stale."""
        self._last_event = None
