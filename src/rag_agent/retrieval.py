from sentence_transformers import SentenceTransformer
from .db import get_db_connection
from .config import MODEL_NAME, TOP_K_RETRIEVAL, TOP_K_RETURNED



_model_instance = None

def get_embedding_model() -> SentenceTransformer:
    """Singleton for the embedding model — loaded once."""
    global _model_instance
    if _model_instance is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _model_instance = SentenceTransformer(MODEL_NAME)
    return _model_instance

def retrieve_top_k(query: str, k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Retrieve top-k most similar movie chunks for a query.
    Returns list of dicts with chunk details.
    """
    conn = get_db_connection()
    model = get_embedding_model()
    # Embed the query
    query_embedding = model.encode([query], normalize_embeddings=True)[0]



    # Cosine similarity search using HNSW index,
    # Two-stage retrieval:
    # 1) Vector recall by semantic similarity
    # 2) Lightweight reranking using popularity + vote_count as tie breakers

    results = conn.execute("""
        WITH candidates AS (
            SELECT 
                movie_id,
                title,
                release_year,
                genres,
                overview,
                popularity,
                vote_count,
                embedding <=> ? AS distance
            FROM text_chunks
            ORDER BY distance
            LIMIT ?
        )
        SELECT 
            movie_id,
            title,
            release_year,
            genres,
            overview,
            distance
        FROM candidates
        ORDER BY 
            distance ASC,           -- still respect similarity first
            popularity DESC,       -- then boost popular ones
            vote_count DESC
        LIMIT ?
    """, [query_embedding.tolist(), TOP_K_RETRIEVAL, TOP_K_RETURNED]).fetchall()
    # Format results
    chunks = []
    for row in results:
        movie_id, title, year, genres, text, distance = row
        chunks.append({
            "movie_id": movie_id,
            "title": title,
            "release_year": year,
            "genres": genres,
            "overview": text,
            "distance": distance,
        })

    return chunks