# tests/conftest.py
import duckdb
import pytest
from src.rag_agent.config import DB_PATH


@pytest.fixture(scope="session")
def conn():
    if not DB_PATH.exists():
        pytest.skip("Database not built — run build_corpus.py first")
    return duckdb.connect(str(DB_PATH))

