from src.rag_agent.retrieval import retrieve_top_k

def test_retrieve_top_k_returns_results():
    results = retrieve_top_k("science fiction space travel", k=3)
    assert len(results) > 0


def test_retrieval_result_shape():
    result = retrieve_top_k("romantic comedy", k=1)[0]

    for key in ["movie_id", "title", "release_year", "overview", "distance"]:
        assert key in result
