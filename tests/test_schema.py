from src.rag_agent.config import EMBEDDING_DIM
def test_text_chunks_exists(conn):
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    print(tables)
    assert "text_chunks" in tables


def test_embedding_not_null(conn):
    count = conn.execute("""
        SELECT COUNT(*) FROM text_chunks
        WHERE embedding IS NULL
    """).fetchone()[0]

    assert count == 0


def test_embedding_dimension(conn):
    embedding = conn.execute("""
        SELECT embedding FROM text_chunks LIMIT 1
    """).fetchone()[0]

    assert len(embedding) == EMBEDDING_DIM
