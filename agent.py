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
import json

from datetime import datetime, timezone 

from src.rag_agent.db import get_db_connection

conn = get_db_connection()



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

    click.echo("🤖 Dummy RAG Agent – Interactive Mode")
    click.echo("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            user_input = click.prompt("Ask me about movies (2015-2024)", type=str)
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye")
            break

        if user_input.lower().strip() in {"exit", "quit"}:
            click.echo("Goodbye")
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
