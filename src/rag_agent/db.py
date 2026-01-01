# src/rag_agent/db.py

import duckdb
import json
from .config import DATA_DIR, DB_PATH



# ---------- DB SETUP ----------

def get_db_connection():
    """Return a DuckDB connection (single-file, portable)."""
    return duckdb.connect(str(DB_PATH))

#this should really only be called once when builidng from scratch
def initialize_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    # Enable vector extension for chunking
    conn.execute("INSTALL vss;")
    conn.execute("LOAD vss;")
    conn.execute("SET hnsw_enable_experimental_persistence = TRUE;")

    """Initialize tables and seed dummy knowledge base if empty."""

    #this is going to be the logged location of queries
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
        /*had an id here but duckdb doesn't auto generate, no primary key necessary at this time or can be added later*/         
        timestamp TIMESTAMP,
        query TEXT,
        response TEXT,
        sources JSON
        );
    """)

    #this is going to go away really soon in this project
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dummy_knowledge (
            key TEXT PRIMARY KEY,
            value TEXT,
            /* we want to know the provenance of each knowledge item */
            sources JSON
        );
    """)

    # This is also going to go away/change soon
    if conn.execute("SELECT COUNT(*) FROM dummy_knowledge").fetchone()[0] == 0:
        seed_data = [
            (
                "what is this project",
                "This is a Step 0 dummy RAG agent used to validate the full agent architecture for a data engineering portfolio project.",
                json.dumps(["README.md"]),
            ),
            (
                "what database are you using",
                "This agent uses DuckDB for local persistence, logging, and future vector storage.",
                json.dumps(["architecture_decisions.md"]),
            ),
            (
                "what is rag",
                "RAG stands for Retrieval-Augmented Generation – combining retrieval of relevant context with generative LLMs.",
                json.dumps(["rag_concepts.md"]),
            ),
        ]
        conn.executemany(
            """
            INSERT INTO dummy_knowledge (key, value, sources)
            VALUES (?, ?, ?)
            """,
            seed_data,
        )


