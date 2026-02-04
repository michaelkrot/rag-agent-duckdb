import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict

LLM_MODEL_NAME = "google/flan-t5-large"

# Device selection
if torch.cuda.is_available():
    device = 0
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon MPS acceleration")
else:
    device = -1
    print("Using CPU (slower – consider smaller model if possible)")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
model.config.tie_word_embeddings = False  # Silence warning

if device != -1:
    model = model.to(device).half()
else:
    model = model.to(device)

print(f"Flan-T5-large loaded on {device}")

# ──────────────────────────────────────────────────────────────
# Raw generation function (no pipeline → no KeyError)
# ──────────────────────────────────────────────────────────────

def generate_text(prompt: str, max_new_tokens: int = 150, num_beams: int = 2) -> str:
    """
    Raw model.generate() inference — avoids pipeline task registry problems.
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    # Critical fix: Keep input_ids as long (integer type) — don't let them become fp16
    inputs['input_ids'] = inputs['input_ids'].long()           # Force long
    inputs['attention_mask'] = inputs['attention_mask'].long() # Also long

    # Only move tensors to device after type fix
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # If on GPU/MPS, ensure model is in fp16 but inputs are not
    if device != -1:
        # Model is already .half() from load time
        pass  # No need to half inputs — we fixed types above

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        do_sample=False,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────
# Prompt builder (unchanged — looks good)
# ──────────────────────────────────────────────────────────────

def build_prompt(query: str, contexts: List[Dict]) -> str:
    lines = [
        "Answer the question using ONLY the sources below.",
        "Be precise with actor names and release years.",
        "Cite sources using [S1], [S2], etc.\n",
        f"Question:\n{query}\n",
        "Sources:"
    ]

    for i, ctx in enumerate(contexts, start=1):
        overview_snippet = ctx.get('overview', '')
        if len(overview_snippet) > 300:
            overview_snippet = overview_snippet[:300].rstrip() + "..."

        lines.append(
            f"[S{i}] {ctx.get('title', 'Unknown')} ({ctx.get('release_year', 'Unknown')}): "
            f"{overview_snippet} Cast: {ctx.get('movie_cast', 'Unknown')}"
        )

    lines.append("\nAnswer:")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main synthesis function — now calls generate_text()
# ──────────────────────────────────────────────────────────────

def synthesize_answer(
    query: str,
    contexts: List[Dict],
    *,
    max_tokens: int = 256
) -> Dict:
    if not contexts:
        return {"answer": "No relevant sources found.", "citations": []}

    prompt = build_prompt(query, contexts)

    # Use the raw generate function
    answer_text = generate_text(
        prompt=prompt,
        max_new_tokens=max_tokens,
        num_beams=2
    )

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



if __name__ == "__main__":
    # Quick smoke test
    test_prompt = "Translate to French: Hello world"
    result = generate_text(test_prompt, max_new_tokens=50)
    print("Test generation:", result)