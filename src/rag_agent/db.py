# src/rag_agent/db.py

import duckdb
from .config import DATA_DIR, DB_PATH


def get_db_connection():
    """Return a DuckDB connection (single-file, portable)."""
    return duckdb.connect(str(DB_PATH))


def initialize_db():
    """Initialize persistent tables required by the agent."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            timestamp TIMESTAMP,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            sources JSON
        );
    """)
    conn.close()
