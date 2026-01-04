from src.rag_agent.synthesis import build_prompt, synthesize_answer

def sample_contexts():
    return [
        {
            "title": "Dune",
            "release_year": 2021,
            "overview": "A young nobleman leads a rebellion on a desert planet.",
            "score": 0.95,
        },
        {
            "title": "Interstellar",
            "release_year": 2014,
            "overview": "Explorers travel through a wormhole to find a new home for humanity.",
            "score": 0.89,
        },
    ]

def test_build_prompt_includes_all_contexts():
    query = "Recommend mind-bending sci-fi movies."
    contexts = sample_contexts()

    prompt = build_prompt(query, contexts)
    
    # Check that each movie title appears
    for ctx in contexts:
        assert ctx["title"] in prompt

    # Check that the question is included
    assert query in prompt

    # Check that source markers exist
    assert "[S1]" in prompt
    assert "[S2]" in prompt

def test_synthesize_answer_structure():
    query = "Suggest sci-fi films."
    contexts = sample_contexts()

    result = synthesize_answer(query, contexts)

    # Must return a dict
    assert isinstance(result, dict)

    # Must contain answer and citations
    assert "answer" in result
    assert "citations" in result

    # Citations list length == number of contexts
    assert len(result["citations"]) == len(contexts)

    # Check citation fields
    for i, cit in enumerate(result["citations"], start=1):
        assert cit["index"] == i
        assert "title" in cit
        assert "year" in cit
        assert "score" in cit

    # Answer text should mention movie titles (stub logic)
    for ctx in contexts:
        assert ctx["title"] in result["answer"]
