from .retrieval import retrieve_top_k

from .retrieval import retrieve_top_k
from .synthesis import synthesize_answer
from .config import TOP_K_RETURNED

def get_grounded_response(query: str, k: int = TOP_K_RETURNED) -> tuple[str, list[str]]:
    """
    v0.4 response: feed retrieved chunks to synthesis stub.
    """
    chunks = retrieve_top_k(query, top_k_returned=k)

    if not chunks:
        return "No relevant movies found for your query.", []

    result = synthesize_answer(query, chunks)

    # Format answer + sources for CLI (v0.4 stub)
    answer = result["answer"]
    sources = [f"{c['title']} ({c['year']})" for c in result["citations"]]

    return answer, sources
