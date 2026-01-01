import pandas as pd
from pathlib import Path
import duckdb

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "agent.duckdb"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------- DB SETUP ----------
def get_db_connection():
    """Return a DuckDB connection (single-file, portable)."""
    return duckdb.connect(str(DB_PATH))


conn = duckdb.connect("data/agent.duckdb")



df = pd.read_csv("data/raw/TMDB_all_movies.csv")

conn.execute("CREATE or REPLACE TABLE raw_tmdb_data AS SELECT * FROM df")


