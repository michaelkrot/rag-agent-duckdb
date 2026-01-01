#!/usr/bin/env python3
"""
Embed TMDB movie overviews into DuckDB vector table.
v0.3 – Text RAG foundation
"""

import duckdb
from sentence_transformers import SentenceTransformer
from datetime import datetime

DB_PATH = "data/agent.duckdb"

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256  # safe for CPU
MIN_OVERVIEW_LEN = 40
MIN_POPULARITY = 1.5
MIN_VOTE_COUNT = 15


# -------------------------
# SETUP
# -------------------------
conn = duckdb.connect(DB_PATH)

# Enable vector extension
conn.execute("INSTALL vss;")
conn.execute("LOAD vss;")
conn.execute("SET hnsw_enable_experimental_persistence = TRUE;")

# Create table
conn.execute("""
CREATE TABLE IF NOT EXISTS text_chunks (
    chunk_id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    title VARCHAR,
    release_year INTEGER,
    genres VARCHAR,
    chunk_text VARCHAR,
    embedding FLOAT[384],   -- fixed-length vector
    created_at TIMESTAMP DEFAULT now()
);
""")

model = SentenceTransformer(MODEL_NAME)

# -------------------------
# LOAD SOURCE DATA
# -------------------------
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
      AND CAST(release_date AS DATE)
          BETWEEN DATE '2015-01-01' AND DATE '2025-12-31'
""", [MIN_OVERVIEW_LEN, MIN_POPULARITY, MIN_VOTE_COUNT]).fetchall()


print(f"Embedding {len(rows)} movie overviews...")

# -------------------------
# EMBED + INSERT
# -------------------------
chunk_id = conn.execute(
    "SELECT COALESCE(MAX(chunk_id), 0) FROM text_chunks"
).fetchone()[0]

batch = []

for movie_id, title, year, genres, overview in rows:
    batch.append((movie_id, title, year, genres, overview))

    if len(batch) >= BATCH_SIZE:
        texts = [b[4] for b in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        for i, emb in enumerate(embeddings):
            chunk_id += 1
            conn.execute(
                """
                INSERT INTO text_chunks
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    chunk_id,
                    batch[i][0],
                    batch[i][1],
                    batch[i][2],
                    batch[i][3],
                    batch[i][4],
                    emb.tolist(),
                    datetime.utcnow(),
                ],
            )
            if chunk_id % 500 == 0:
                print(f"Embedded {chunk_id} chunks...")


        batch.clear()

# Flush remainder
if batch:
    texts = [b[4] for b in batch]
    embeddings = model.encode(texts, show_progress_bar=False)

    for i, emb in enumerate(embeddings):
        chunk_id += 1
        conn.execute(
            """
            INSERT INTO text_chunks
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                chunk_id,
                batch[i][0],
                batch[i][1],
                batch[i][2],
                batch[i][3],
                batch[i][4],
                emb.tolist(),
                datetime.utcnow(),
            ],
        )

print("Embedding complete.")
#create vector index
print("Indexing.")
conn.execute("""CREATE INDEX IF NOT EXISTS text_chunks_embedding_idx
ON text_chunks
USING HNSW (embedding);""")

print("Indexing complete.")


