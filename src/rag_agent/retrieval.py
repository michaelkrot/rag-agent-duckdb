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

def retrieve_top_k(query: str, top_k_returned: int = TOP_K_RETURNED) -> list[dict]:
    conn = get_db_connection()
    model = get_embedding_model()

    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    query_lower = query.lower()

    # Retrieve extra candidates
    results = conn.execute("""
        SELECT
            movie_id,
            title,
            release_year,
            genres,
            overview,
            movie_cast,
            popularity,
            vote_count,
            embedding <=> ? AS distance
        FROM text_chunks
        ORDER BY distance ASC
        LIMIT ?
    """, [query_embedding.tolist(), TOP_K_RETRIEVAL]).fetchall()

    scored = []
    for row in results:
        movie_id, title, year, genres, overview, movie_cast, popularity, vote_count, distance = row

        # --- Title boost ---
        title_boost = 0.0
        if title:
            title_tokens = set(title.lower().split())
            query_tokens = set(query_lower.split())
            overlap = title_tokens & query_tokens
            if overlap:
                # Increase max title boost from 0.2 → 0.5
                title_boost = min(0.5, 0.1 * len(overlap))  # 0.1 per overlapping token

        # --- Cast boost ---
        cast_boost = 0.0
        if movie_cast:
            query_tokens = set(query_lower.split())
            for actor in movie_cast.split(","):
                actor = actor.strip().lower()
                if not actor:
                    continue
                actor_tokens = set(actor.split())
                if actor_tokens & query_tokens:
                    cast_boost = 0.15  # slightly higher base for token match
                    break
                if actor in query_lower or query_lower in actor:
                    cast_boost = 0.2  # stronger boost for full match
                    break

        # --- Popularity boost ---
        # Increase from 0.00001 → 0.00005 to make popularity more noticeable
        popularity_boost = 0.00005 * (popularity or 0)

        # --- Combined score ---
        score = -distance + title_boost + cast_boost + popularity_boost

        scored.append({
            "movie_id": movie_id,
            "title": title,
            "release_year": year,
            "genres": genres,
            "overview": overview,
            "movie_cast": movie_cast,
            "distance": distance,
            "score": score,
            "title_boost": title_boost,
            "cast_boost": cast_boost,
            "popularity_boost": popularity_boost
        })

    # Sort by final score
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k_returned]




