from .retrieval import retrieve_top_k

def get_grounded_response(query: str, k: int = 5) -> tuple[str, list[str]]:
    """
    Simple v0.3 response: return concatenated retrieved texts as context.
    Future: feed to LLM.
    """
    chunks = retrieve_top_k(query, k=k)

    if not chunks:
        response = "No relevant movies found for your query."
        sources = []
    else:
        # Concatenate chunk texts (simple for v0.3)
        context = "\n\n".join([chunk["chunk_text"] for chunk in chunks])
        response = f"Relevant movie overviews:\n\n{context}"

        # Provenance sources
        sources = [
            f"{chunk['title']} ({chunk['release_year']})"
            for chunk in chunks
        ]

    return response, sources