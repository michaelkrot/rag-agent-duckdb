from pathlib import Path

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "agent.duckdb"

RAW_TMDB_DATA = DATA_DIR / "raw" / "TMDB_all_movies.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256  # safe for CPU; adjust for memory
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

# --- RAG corpus filtering ---
MIN_OVERVIEW_LEN = 40       # discard ultra-short plots
MIN_POPULARITY = 1          # remove obscure / low-signal entries
MIN_VOTE_COUNT = 10         # ensure minimal audience validation
TOP_K_RETRIEVAL = 50         # number of chunks to retrieve per query
TOP_K_RETURNED = 5            # number of chunks to return per query