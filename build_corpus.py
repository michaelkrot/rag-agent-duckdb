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
    conn = get_db_connection()
    df = pd.read_csv(RAW_TMDB_DATA)
    conn.execute("CREATE or REPLACE TABLE raw_tmdb_data AS SELECT * FROM df")

def build_movie_corpus():
    conn = get_db_connection()

    # Create table
    #The EMBEDDING_DIM is ugly but DuckDB isn't handling it as a parameter and I don't have time to debug
    conn.execute("""
        CREATE TABLE IF NOT EXISTS text_chunks (
            chunk_id INTEGER PRIMARY KEY,
            movie_id INTEGER,
            title VARCHAR,
            release_year INTEGER,
            genres VARCHAR,
            chunk_text VARCHAR,
            embedding FLOAT[""" + str(EMBEDDING_DIM) + """],
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
            overview
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

    if len(rows) == 0:
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
        movie_id, title, year, genres, overview = row
        batch.append((movie_id, title, year, genres, overview))

        if len(batch) >= BATCH_SIZE:
            texts = [item[4] for item in batch]
            embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

            for i, emb in enumerate(embeddings):
                conn.execute(
                    """
                    INSERT INTO text_chunks
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        batch[i][0],
                        batch[i][1],
                        batch[i][2],
                        batch[i][3],
                        batch[i][4],
                        emb.tolist(),
                        datetime.now(timezone.utc),
                    ),
                )
                chunk_id += 1
                total_inserted += 1

            batch.clear()

    # Final batch
    if batch:
        texts = [item[4] for item in batch]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        for i, emb in enumerate(embeddings):
            conn.execute(
                """
                INSERT INTO text_chunks
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    batch[i][0],
                    batch[i][1],
                    batch[i][2],
                    batch[i][3],
                    batch[i][4],
                    emb.tolist(),
                    datetime.now(timezone.utc),
                ),
            )
            chunk_id += 1
            total_inserted += 1

    print(f"Embedded and inserted {total_inserted} movie chunks.")

    # Create HNSW index for fast similarity search
    print("Building vector index...")
        # Enable vector extension for chunking
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