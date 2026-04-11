from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, Integer, JSON, create_engine
from sqlalchemy.orm import DeclarativeBase
import uuid
from querymind.config import settings

class Base(DeclarativeBase):
    pass

class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False, index=True)
    user_question = Column(Text, nullable=False)
    sql_generated = Column(Text, nullable=True)
    result_data = Column(JSON, nullable=True)      # stores rows as JSON
    result_row_count = Column(Integer, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    status = Column(String, default="success")     # success | error
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Create synchronous engine for table creations
if settings.database_type == "postgresql" and settings.postgres_dsn:
    # Ensure it's using the synchronous driver for create_all if an async DSN is provided
    dsn = settings.postgres_dsn.replace("postgresql+asyncpg", "postgresql")
    engine = create_engine(dsn)
else:
    engine = create_engine(f"sqlite:///{settings.db_path}")
