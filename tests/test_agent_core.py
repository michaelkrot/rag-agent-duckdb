from src.rag_agent.agent_core import get_grounded_response

def test_grounded_response_has_sources():
    response, sources = get_grounded_response("Batman")

    assert isinstance(response, str)
    assert isinstance(sources, list)

    if sources:
        assert "(" in sources[0]  # "Title (Year)"
