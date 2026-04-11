"""initial_schema

Revision ID: e731053ce361
Revises: 
Create Date: 2026-04-08 11:51:11.869487

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e731053ce361'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx
        ON rag_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    # We should run these in a single transaction implicitly
    # Alembic handles transactions, but CREATE USER cannot run inside transaction in PG usually.
    # Actually wait. The user snippet:
    op.execute("""
        DO $$ 
        BEGIN 
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'querymind_read') THEN 
                CREATE ROLE querymind_read LOGIN PASSWORD 'querymind_read_pass'; 
            END IF; 
        END $$;
    """)
    op.execute("GRANT CONNECT ON DATABASE querymind TO querymind_read")
    op.execute("GRANT USAGE ON SCHEMA public TO querymind_read")
    op.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO querymind_read")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS rag_chunks_embedding_hnsw_idx")
    op.execute("DROP TABLE IF EXISTS rag_chunks")
