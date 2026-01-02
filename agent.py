#!/usr/bin/env python3
"""
Movie RAG Agent – v0.3 (Text Retrieval Complete)

A provenance-aware RAG agent over TMDB movie overviews (2015–2025).
Uses DuckDB + HNSW for fast vector search.
"""

import click
import json
from datetime import datetime, timezone

from src.rag_agent.db import get_db_connection
from src.rag_agent.agent_core import get_grounded_response


# ---------- LOGGING ----------
def log_interaction(conn, query: str, response: str, sources: list):
    conn.execute(
        """
        INSERT INTO query_log (timestamp, query, response, sources)
        VALUES (?, ?, ?, ?)
        """,
        [datetime.now(timezone.utc), query, response, json.dumps(sources)],
    )


# ---------- CLI ----------
@click.group()
@click.version_option(version="0.3.0", prog_name="Movie RAG Agent")
def cli():
    """Movie RAG Agent — Ask about films from 2015–2025!"""
    pass


@cli.command()
@click.argument("query", type=str)
def query(query: str):
    """Ask a single question about movies."""
    conn = get_db_connection()
    response, sources = get_grounded_response(query)
    log_interaction(conn, query, response, sources)

    click.echo("\n=== AGENT RESPONSE ===\n")
    click.echo(response)
    if sources:
        click.echo("\nSources:")
        for src in sources:
            click.echo(f"  • {src}")
    click.echo("\n======================\n")


@cli.command()
def repl():
    """Start an interactive session."""
    conn = get_db_connection()

    click.echo("🎬 Movie RAG Agent – v0.3 (Text Retrieval)")
    click.echo("This agent searches movie plot overviews from TMDB (2015–2025).")
    click.echo("Best for questions like:")
    click.echo("  • 'What is Dune about?'")
    click.echo("  • 'Movies with time travel'")
    click.echo("  • 'Superhero films with mind-bending plots'")
    click.echo("It does NOT yet answer questions about box office, ratings, popularity, or awards.")
    click.echo("Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = click.prompt("Ask me", type=str)
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye 👋")
            break

        if user_input.lower().strip() in {"exit", "quit"}:
            click.echo("Goodbye 👋")
            break

        try:
            response, sources = get_grounded_response(user_input)
            log_interaction(conn, user_input, response, sources)
            click.echo(f"Agent: {response}")
            if sources:
                click.echo("Sources:")
                for src in sources:
                    click.echo(f"  • {src}")
            click.echo()
        except Exception as e:
            click.echo(f"\n⚠️ Error: {e}")
            click.echo("Continuing...\n")


@cli.command()
def stats():
    """Show usage statistics."""
    conn = get_db_connection()
    total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    last_ts = conn.execute("SELECT MAX(timestamp) FROM query_log").fetchone()[0]
    top_queries = conn.execute("""
        SELECT query, COUNT(*) AS cnt
        FROM query_log
        GROUP BY query
        ORDER BY cnt DESC
        LIMIT 5
    """).fetchall()

    click.echo("\n📊 Agent Stats")
    click.echo(f"Total queries: {total}")
    click.echo(f"Last query: {last_ts or 'None'}\n")
    click.echo("Top queries:")
    for q, cnt in top_queries or []:
        click.echo(f"  • {q} ({cnt})")
    if not top_queries:
        click.echo("  • No queries yet")
    click.echo()


if __name__ == "__main__":
    cli()