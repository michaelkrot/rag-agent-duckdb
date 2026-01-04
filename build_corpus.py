#!/usr/bin/env python3
"""
Build movie text corpus for RAG retrieval (v0.3)

One-time script to:
- Load raw TMDB data
- Filter recent/popular movies
- Embed overviews (one chunk per movie)
- Create HNSW index for fast similarity search

Run with: python build_corpus.py
"""
import pandas as pd
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.rag_agent.db import get_db_connection, initialize_db
from src.rag_agent.config import (
    MODEL_NAME,
    BATCH_SIZE,
    MIN_OVERVIEW_LEN,
    MIN_POPULARITY,
    MIN_VOTE_COUNT,
    RAW_TMDB_DATA,
    EMBEDDING_DIM
)


def seed_raw_tmdb_data():
    """Load raw TMDB CSV into DuckDB."""
    conn = get_db_connection()
    df = pd.read_csv(RAW_TMDB_DATA)
    conn.execute("CREATE or REPLACE TABLE raw_tmdb_data AS SELECT * FROM df")


def embed_movie_batch(batch: list[tuple], model: SentenceTransformer) -> list[list[float]]:
    """
    Given a batch of movie rows, return embeddings with duplicated title & cast.
    Each row is expected to be:
    (movie_id, title, release_year, genres, overview, movie_cast, popularity, vote_count)
    """
    texts = []
    for row in batch:
        movie_title = row[1] or ""
        movie_overview = row[4] or ""
        movie_cast_str = " ".join([a.strip() for a in (row[5] or "").split(",")])
        release_year = row[2] or ""
        genres = row[3] or ""
        text_to_embed = f"{movie_overview} {movie_title} {movie_title} {movie_cast_str} {movie_cast_str} {release_year} {genres}"
        texts.append(text_to_embed)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings


def insert_batch_embeddings(conn, batch: list[tuple], embeddings: list[list[float]], chunk_id_start: int) -> int:
    """
    Insert a batch of embeddings into text_chunks table.
    Returns the next available chunk_id.
    """
    chunk_id = chunk_id_start
    for row, emb in zip(batch, embeddings):
        conn.execute(
            """
            INSERT INTO text_chunks
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                row[0],  # movie_id
                row[1],  # title
                row[2],  # release_year
                row[3],  # genres
                row[4],  # overview
                row[5],  # movie_cast
                row[6],  # popularity
                row[7],  # vote_count
                emb.tolist(),
                datetime.now(timezone.utc),
            ),
        )
        chunk_id += 1
    return chunk_id


def build_movie_corpus():
    conn = get_db_connection()

    # Create table
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS text_chunks (
            chunk_id INTEGER PRIMARY KEY,
            movie_id INTEGER,
            title VARCHAR,
            release_year INTEGER,
            genres VARCHAR,
            overview VARCHAR,
            movie_cast VARCHAR,   
            popularity DOUBLE,
            vote_count INTEGER,
            embedding FLOAT[{EMBEDDING_DIM}],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Load and filter source data
    print("Loading and filtering TMDB movies...")
    rows = conn.execute("""
        SELECT
            id AS movie_id,
            title,
            EXTRACT(YEAR FROM CAST(release_date AS DATE)) AS release_year,
            genres,
            overview,
            "cast" AS movie_cast,
            popularity,
            vote_count
        FROM raw_tmdb_data
        WHERE overview IS NOT NULL
          AND LENGTH(overview) >= ?
          AND popularity >= ?
          AND vote_count >= ?
          AND release_date IS NOT NULL
          AND CAST(release_date AS DATE) BETWEEN DATE '2015-01-01' AND DATE '2025-12-31'
        ORDER BY popularity DESC
    """, [MIN_OVERVIEW_LEN, MIN_POPULARITY, MIN_VOTE_COUNT]).fetchall()

    print(f"Found {len(rows)} movies to embed.")
    if not rows:
        print("No movies to process. Check raw_tmdb_data table and filters.")
        return

    # Load model
    model = SentenceTransformer(MODEL_NAME)

    # Get next chunk_id
    max_id = conn.execute("SELECT COALESCE(MAX(chunk_id), 0) FROM text_chunks").fetchone()[0]
    chunk_id = max_id + 1

    batch = []
    total_inserted = 0

    print("Embedding and inserting...")
    for row in tqdm(rows, desc="Processing movies"):
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            embeddings = embed_movie_batch(batch, model)
            chunk_id = insert_batch_embeddings(conn, batch, embeddings, chunk_id)
            total_inserted += len(batch)
            batch.clear()

    # Process any remaining rows in batch
    if batch:
        embeddings = embed_movie_batch(batch, model)
        chunk_id = insert_batch_embeddings(conn, batch, embeddings, chunk_id)
        total_inserted += len(batch)

    print(f"Embedded and inserted {total_inserted} movie chunks.")

    # Create HNSW index for fast similarity search
    print("Building vector index...")
    conn.execute("INSTALL vss;")
    conn.execute("LOAD vss;")
    conn.execute("SET hnsw_enable_experimental_persistence = TRUE;")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS text_chunks_embedding_idx
        ON text_chunks USING HNSW (embedding)
        WITH (metric = 'cosine')
    """)
    print("Index built. Corpus ready for retrieval!")


if __name__ == "__main__":
    initialize_db()
    seed_raw_tmdb_data()
    build_movie_corpus()
