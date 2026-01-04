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
TOP_K_RETRIEVAL = 150         # number of chunks to retrieve per query
TOP_K_RETURNED = 10            # number of results to return by default
# Current: fixed hybrid ranking
# Future: adaptive adjustment of retrieval depth or ranking weights
# informed by query-level relevance signals and historical performance

USE_LLM = False  # v0.4 synthesis flag
