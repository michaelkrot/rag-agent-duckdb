#!/usr/bin/env python3
"""
Dummy RAG Agent – Step 0

A lightweight, provenance-aware prototype demonstrating the full agent flow:
query → retrieval → response → logging with sources.

Built with data engineering best practices:
- DuckDB for structured, queryable persistence from day one
- Clean separation of concerns
- CLI with single-shot and interactive modes

Future evolution:
- Step 1–3: Integrate local LLM
- Step 4–6: Replace dummy retrieval with vector similarity search over embedded chunks
- Step 7+: Quantitative reasoning and incremental updates
"""

import click
import duckdb
import json
from pathlib import Path
from datetime import datetime, timezone 

# ---------- CONFIG ----------

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "agent.duckdb"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------- DB SETUP ----------
def get_db_connection():
    """Return a DuckDB connection (single-file, portable)."""
    return duckdb.connect(str(DB_PATH))

def init_db(conn):
    """Initialize tables and seed dummy knowledge base if empty."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
        /*had an id here but duckdb doesn't auto generate, no primary key necessary at this time or can be added later*/         
        timestamp TIMESTAMP,
        query TEXT,
        response TEXT,
        sources JSON
        );
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS dummy_knowledge (
            key TEXT PRIMARY KEY,
            value TEXT,
            /* we want to know the provenance of each knowledge item */
            sources JSON
        );
    """)

    # Seed only once
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
# ---------- AGENT CORE ----------

def get_response(conn, query: str):
    """
    Simulate retrieval → response.
    Currently exact/partial match on dummy_knowledge.
    Future: vector similarity search over embedded text chunks.
    """
    normalized = query.lower().strip()

    # Exact match
    row = conn.execute(
        "SELECT value, sources FROM dummy_knowledge WHERE key = ?",
        [normalized],
    ).fetchone()

    # Partial match fallback
    if not row:
        row = conn.execute(
            """
            SELECT value, sources
            FROM dummy_knowledge
            WHERE LOWER(key) LIKE ?
            ORDER BY LENGTH(key) DESC
            LIMIT 1
            """,
            [f"%{normalized}%"],
        ).fetchone()

    if row:
        response, sources = row
        return response, sources

    # Final fallback
    return (
        f"This is a placeholder response to your query: '{query}'",
        json.dumps(["dummy_source_1.md", "dummy_source_2.md"]),
    )



# Then in log_interaction():
def log_interaction(conn, query: str, response: str, sources_json: str):
    """Persist the full interaction with provenance."""
    conn.execute(
        """
        INSERT INTO query_log (timestamp, query, response, sources)
        VALUES (?, ?, ?, ?)
        """,
        [datetime.now(timezone.utc), query, response, sources_json],  
    )
# ---------- CLI ----------

@click.group()
@click.version_option(version="0.1.1", prog_name="Dummy RAG Agent")
def cli():
    """Standalone Dummy RAG Agent (Step 0)"""
    pass


@cli.command()
@click.argument("query", type=str)
def query(query: str):
    """Ask the agent a single question."""
    conn = get_db_connection()
    init_db(conn)

    response, sources_json = get_response(conn, query)
    log_interaction(conn, query, response, sources_json)

    sources = json.loads(sources_json)

    click.echo("\n=== AGENT RESPONSE ===")
    click.echo(response)
    click.echo("\nSources:")
    for src in sources:
        click.echo(f"  • {src}")
    click.echo("\n======================\n")


@cli.command()
def repl():
    """Start an interactive agent session."""
    conn = get_db_connection()
    init_db(conn)

    click.echo("🤖 Dummy RAG Agent – Interactive Mode")
    click.echo("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = click.prompt("Just ask:", type=str)
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye 👋")
            break

        if user_input.lower().strip() in {"exit", "quit"}:
            click.echo("Goodbye 👋")
            break

        try:
            response, sources_json = get_response(conn, user_input)
            log_interaction(conn, user_input, response, sources_json)
            sources = json.loads(sources_json)

            click.echo(f"\nAgent: {response}")
            click.echo("Sources:")
            for src in sources:
                click.echo(f"  • {src}")
            click.echo()
        except Exception as e:
            click.echo(f"\n⚠️  Error processing query: {e}")
            click.echo("Continuing session...\n")


@cli.command()
def stats():
    """Show basic usage statistics."""
    conn = get_db_connection()
    init_db(conn)

    total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    last_ts = conn.execute(
        "SELECT MAX(timestamp) FROM query_log"
    ).fetchone()[0]

    top_queries = conn.execute(
        """
        SELECT query, COUNT(*) AS cnt
        FROM query_log
        GROUP BY query
        ORDER BY cnt DESC
        LIMIT 5
        """
    ).fetchall()

    click.echo("\n📊 Agent Stats")
    click.echo(f"Total queries: {total}")
    click.echo(f"Last query at: {last_ts}\n")

    click.echo("Top queries:")
    for q, cnt in top_queries:
        click.echo(f"  • {q} ({cnt})")

    click.echo()


if __name__ == "__main__":
    cli()
