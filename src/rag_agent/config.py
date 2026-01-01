from pyparsing import Path

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "agent.duckdb"

RAW_TMDB_DATA = DATA_DIR / "raw" / "TMDB_all_movies.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256  # safe for my CPU...good to check on yours
MIN_OVERVIEW_LEN = 40
MIN_POPULARITY = 1
MIN_VOTE_COUNT = 10