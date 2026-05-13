"""Tests for coarse.agents.literature — literature search agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from coarse.agents.literature import (
    OpenAlexWork,
    _compile_context,
    _extract_venue,
    _parse_openalex_work,
    _RankedResult,
    _RankedResults,
    _reconstruct_abstract,
    _search_perplexity,
    _SearchQueries,
    search_literature,
)


SAMPLE_OPENALEX_WORK = {
    "id": "https://openalex.org/W2741809807",
    "title": "A Great Paper on Causal Inference",
    "publication_year": 2023,
    "authorships": [
        {"author": {"display_name": "Alice Smith"}},
        {"author": {"display_name": "Bob Jones"}},
    ],
    "abstract_inverted_index": {
        "We": [0],
        "propose": [1],
        "a": [2, 6],
        "new": [3],
        "method": [4],
        "for": [5],
        "distributional": [7],
        "approach.": [8],
    },
    "primary_location": {"source": {"display_name": "AER"}},
    "cited_by_count": 142,
    "doi": "https://doi.org/10.1257/aer.123.45",
}


def test_reconstruct_abstract_basic():
    text = _reconstruct_abstract(SAMPLE_OPENALEX_WORK["abstract_inverted_index"])
    assert text == "We propose a new method for a distributional approach."


def test_reconstruct_abstract_none():
    assert _reconstruct_abstract(None) == ""
    assert _reconstruct_abstract({}) == ""


def test_extract_venue_primary_location():
    assert _extract_venue({"primary_location": {"source": {"display_name": "AER"}}}) == "AER"


def test_extract_venue_falls_back_to_host_venue():
    work = {"primary_location": {}, "host_venue": {"display_name": "NBER WP"}}
    assert _extract_venue(work) == "NBER WP"


def test_extract_venue_missing_returns_empty():
    assert _extract_venue({}) == ""
    assert _extract_venue({"primary_location": None}) == ""


def test_parse_openalex_work_full():
    w = _parse_openalex_work(SAMPLE_OPENALEX_WORK)
    assert w is not None
    assert w.work_id == "W2741809807"
    assert w.title == "A Great Paper on Causal Inference"
    assert w.authors == ["Alice Smith", "Bob Jones"]
    assert "distributional" in w.abstract
    assert w.year == "2023"
    assert w.venue == "AER"
    assert w.cited_by_count == 142
    assert w.doi == "10.1257/aer.123.45"


def test_parse_openalex_work_no_title_returns_none():
    assert _parse_openalex_work({"id": "https://openalex.org/W1"}) is None


def test_parse_openalex_work_handles_missing_optional_fields():
    minimal = {
        "id": "https://openalex.org/W999",
        "title": "Minimal Paper",
        "authorships": [],
    }
    w = _parse_openalex_work(minimal)
    assert w is not None
    assert w.year == ""
    assert w.venue == ""
    assert w.cited_by_count == 0
    assert w.doi == ""
    assert w.abstract == ""


def test_compile_context_formats_works():
    works = {
        "W1": OpenAlexWork(
            work_id="W1",
            title="Test Paper",
            authors=["Alice", "Bob", "Carol", "Dave"],
            abstract="Abstract text",
            year="2020",
            venue="JPE",
            cited_by_count=89,
            doi="10.1086/example",
        ),
    }
    ranked = [_RankedResult(work_id="W1", relevance_score=0.9, reason="Directly related")]
    result = _compile_context(ranked, works)
    assert "Test Paper" in result
    assert "Alice, Bob, Carol et al." in result
    assert "JPE" in result
    assert "2020" in result
    assert "89 citations" in result
    assert "DOI:10.1086/example" in result
    assert "Directly related" in result


def test_compile_context_falls_back_to_openalex_id_without_doi():
    works = {
        "W2": OpenAlexWork(
            work_id="W2",
            title="No DOI Paper",
            authors=["Alice"],
            abstract="",
            year="",
            venue="",
        ),
    }
    ranked = [_RankedResult(work_id="W2", relevance_score=0.5, reason="rel")]
    result = _compile_context(ranked, works)
    assert "OpenAlex:W2" in result
    assert "venue/year unknown" in result


def test_compile_context_empty():
    assert _compile_context([], {}) == ""


def test_search_literature_openalex_end_to_end():
    """Full OpenAlex pipeline with mocked HTTP and LLM calls (no OPENROUTER_API_KEY)."""
    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        _SearchQueries(queries=["causal inference distributional"]),
        _RankedResults(
            ranked=[
                _RankedResult(work_id="W1", relevance_score=0.9, reason="Core method"),
                _RankedResult(work_id="W2", relevance_score=0.7, reason="Related review"),
            ],
            refinement_queries=[],
        ),
    ]

    sample_works = [
        OpenAlexWork(
            work_id="W1",
            title="Causal Paper",
            authors=["Alice"],
            abstract="Abstract",
            year="2023",
            venue="AER",
            cited_by_count=50,
        ),
        OpenAlexWork(
            work_id="W2",
            title="IV Review",
            authors=["Bob"],
            abstract="Abstract",
            year="2022",
            venue="QJE",
            cited_by_count=20,
        ),
    ]

    with (
        patch("coarse.agents.literature._search_openalex") as mock_search,
        patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False),
    ):
        mock_search.return_value = sample_works
        result = search_literature("My Paper", "My abstract", mock_client)

    assert "Causal Paper" in result
    assert "IV Review" in result
    assert "AER" in result
    assert mock_client.complete.call_count == 2


def test_search_literature_query_generation_fails():
    """If query generation fails, fall back to title-based search."""
    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        Exception("LLM failed"),
        _RankedResults(
            ranked=[_RankedResult(work_id="W1", relevance_score=0.8, reason="Related")],
            refinement_queries=[],
        ),
    ]

    with (
        patch("coarse.agents.literature._search_openalex") as mock_search,
        patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False),
    ):
        mock_search.return_value = [
            OpenAlexWork(
                work_id="W1",
                title="Fallback Paper",
                authors=["Alice"],
                abstract="Abstract",
                year="2021",
                venue="JOLE",
            )
        ]
        result = search_literature("My Title", "My abstract", mock_client)

    assert "Fallback Paper" in result


def test_search_literature_no_results():
    mock_client = MagicMock()
    mock_client.complete.return_value = _SearchQueries(queries=["nonexistent topic xyz"])

    with (
        patch("coarse.agents.literature._search_openalex", return_value=[]),
        patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False),
    ):
        result = search_literature("Nonexistent", "Nothing", mock_client)

    assert result == ""


def test_search_perplexity_happy_path():
    caller_client = MagicMock()
    perplexity_client = MagicMock()
    perplexity_client.complete_text.return_value = "## Related Work\n1. Paper A by Author (2020)"
    perplexity_client.cost_usd = 0.025

    with patch("coarse.agents.literature.LLMClient", return_value=perplexity_client):
        result = _search_perplexity("Test Title", "Test abstract", caller_client)

    assert "Paper A" in result
    perplexity_client.complete_text.assert_called_once()
    caller_client.add_cost.assert_called_once_with(0.025)


def test_search_perplexity_sends_system_and_user_messages():
    caller_client = MagicMock()
    perplexity_client = MagicMock()
    perplexity_client.complete_text.return_value = "Literature results"
    perplexity_client.cost_usd = 0.01

    with patch("coarse.agents.literature.LLMClient", return_value=perplexity_client):
        _search_perplexity("Test Title", "Untrusted abstract.", caller_client)

    messages = perplexity_client.complete_text.call_args.args[0]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "<paper_abstract>" in messages[1]["content"]
    assert "</paper_abstract>" in messages[1]["content"]
    assert "Untrusted abstract." in messages[1]["content"]
    assert "paper_content" in messages[0]["content"]


def test_search_perplexity_empty_response_raises():
    caller_client = MagicMock()
    perplexity_client = MagicMock()
    perplexity_client.complete_text.side_effect = ValueError("empty response")

    with patch("coarse.agents.literature.LLMClient", return_value=perplexity_client):
        try:
            _search_perplexity("Test", "Abstract", caller_client)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


def test_dispatcher_uses_perplexity_when_key_set():
    mock_client = MagicMock()

    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test-key"}, clear=False),
        patch(
            "coarse.agents.literature._search_perplexity",
            return_value="Perplexity results",
        ) as mock_perp,
        patch("coarse.agents.literature._search_openalex_pipeline") as mock_openalex,
    ):
        result = search_literature("Title", "Abstract", mock_client)

    assert result == "Perplexity results"
    mock_perp.assert_called_once()
    mock_openalex.assert_not_called()


def test_dispatcher_falls_back_on_perplexity_failure():
    mock_client = MagicMock()

    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test-key"}, clear=False),
        patch("coarse.agents.literature._search_perplexity", side_effect=Exception("API error")),
        patch(
            "coarse.agents.literature._search_openalex_pipeline",
            return_value="OpenAlex results",
        ) as mock_openalex,
    ):
        result = search_literature("Title", "Abstract", mock_client)

    assert result == "OpenAlex results"
    mock_openalex.assert_called_once()


def test_dispatcher_uses_openalex_when_no_key():
    mock_client = MagicMock()

    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False),
        patch("coarse.agents.literature._search_perplexity") as mock_perp,
        patch(
            "coarse.agents.literature._search_openalex_pipeline",
            return_value="OpenAlex results",
        ) as mock_openalex,
    ):
        result = search_literature("Title", "Abstract", mock_client)

    assert result == "OpenAlex results"
    mock_perp.assert_not_called()
    mock_openalex.assert_called_once()
