# src/rag_agent/synthesis.py

from typing import List, Dict

def build_prompt(query: str, contexts: List[Dict]) -> str:
    """
    Deterministically assemble a prompt with explicit citation markers.

    Each context is numbered [S1], [S2], etc.
    """
    lines = [
        "You are answering a question using only the sources below.",
        "Cite sources using [S1], [S2], etc.\n",
        f"Question:\n{query}\n",
        "Sources:"
    ]

    for i, ctx in enumerate(contexts, start=1):
        lines.append(
            f"[S{i}] {ctx['title']} ({ctx.get('release_year', 'Unknown')})\n"
            f"{ctx['overview']}"
        )

    lines.append("\nAnswer:")
    return "\n\n".join(lines)


def synthesize_answer(
    query: str,
    contexts: List[Dict],
    *,
    max_tokens: int = 512
) -> Dict:
    """
    Stub v0.4 synthesis: generates a structured answer using retrieved contexts.

    Returns:
        {
            "answer": str,
            "citations": List[Dict]
        }

    Notes:
    - Deterministic placeholder until local LLM is added.
    - Citations are numbered and include score/year for provenance.
    """
    # Build prompt (not sent to any model yet)
    prompt = build_prompt(query, contexts)

    # Stub answer text
    answer = (
        "This answer will be synthesized by a local LLM in v0.4.\n\n"
        "Relevant movies include:\n"
        + ", ".join(ctx["title"] for ctx in contexts)
    )

    # Build citations list
    citations = [
        {
            "index": i,
            "title": ctx["title"],
            "year": ctx.get("release_year"),
            "score": ctx.get("score"),
        }
        for i, ctx in enumerate(contexts, start=1)
    ]

    return {
        "answer": answer,
        "citations": citations,
    }




