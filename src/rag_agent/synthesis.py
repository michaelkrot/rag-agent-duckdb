from typing import List, Dict
from transformers import pipeline
import atexit

LLM_MODEL_NAME = "google/flan-t5-large"

# Initialize once (still OK for v0.4)
llm = pipeline(
    "text2text-generation",
    model=LLM_MODEL_NAME,
    device=0,        # use -1 for CPU
)


@atexit.register
def cleanup_pipeline():
    global llm
    if llm is not None:
        del llm

def build_prompt(query: str, contexts: List[Dict]) -> str:
    """
    Assemble a concise prompt emphasizing cast and year accuracy.

    Truncates overviews to 300 chars, includes cast and year explicitly,
    and keeps instructions lightweight for faster LLM response.
    """
    lines = [
        "Answer the question using ONLY the sources below.",
        "Be precise with actor names and release years.",
        "Cite sources using [S1], [S2], etc.\n",
        f"Question:\n{query}\n",
        "Sources:"
    ]

    for i, ctx in enumerate(contexts, start=1):
        # Truncate overview to first 300 characters
        overview_snippet = ctx.get('overview', '')
        if len(overview_snippet) > 300:
            overview_snippet = overview_snippet[:300].rstrip() + "..."

        lines.append(
            f"[S{i}] {ctx.get('title', 'Unknown')} ({ctx.get('release_year', 'Unknown')}): "
            f"{overview_snippet} Cast: {ctx.get('movie_cast', 'Unknown')}"
        )

    lines.append("\nAnswer:")
    return "\n".join(lines)



def synthesize_answer(
    query: str,
    contexts: List[Dict],
    *,
    max_tokens: int = 256
) -> Dict:
    if not contexts:
        return {"answer": "No relevant sources found.", "citations": []}

    prompt = build_prompt(query, contexts)

    response = llm(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    answer_text = response[0]["generated_text"].strip()

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
        "answer": answer_text,
        "citations": citations,
    }






