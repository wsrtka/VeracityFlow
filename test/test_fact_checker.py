from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.fact_checker import (
    AgentState,
    build_graph,
    research_planner,
    search_engine,
    sufficiency_checker,
)

# --- Helpers ---


def make_state(**kwargs) -> AgentState:
    """Build a minimal valid AgentState with sensible defaults."""
    defaults = {
        "claim": "The moon is made of cheese.",
        "queries": ["moon composition facts"],
        "search_results": [],
        "revision_number": 0,
        "final_report": "",
        "veracity_score": None,
    }
    defaults.update(kwargs)
    return defaults


# --- sufficiency_checker tests ---
# These are pure logic tests — no mocking needed at all


class TestSufficiencyChecker:
    def test_insufficient_when_below_cap_and_no_score(self):
        state = make_state(revision_number=1, veracity_score=None)
        assert sufficiency_checker(state) == "insufficient"

    def test_sufficient_when_revision_cap_reached(self):
        state = make_state(revision_number=3, veracity_score=None)
        assert sufficiency_checker(state) == "insufficient"

    def test_sufficient_when_score_very_high(self):
        state = make_state(revision_number=1, veracity_score=85)
        assert sufficiency_checker(state) == "sufficient"

    def test_sufficient_when_score_very_low(self):
        state = make_state(revision_number=1, veracity_score=15)
        assert sufficiency_checker(state) == "sufficient"

    def test_insufficient_when_score_uncertain(self):
        state = make_state(revision_number=1, veracity_score=50)
        assert sufficiency_checker(state) == "insufficient"

    def test_sufficient_at_boundary_score_high(self):
        # Boundary: exactly 80 should be sufficient
        state = make_state(revision_number=1, veracity_score=80)
        assert sufficiency_checker(state) == "sufficient"

    def test_sufficient_at_boundary_score_low(self):
        # Boundary: exactly 20 should be sufficient
        state = make_state(revision_number=1, veracity_score=20)
        assert sufficiency_checker(state) == "sufficient"


# --- search_engine tests ---


class TestSearchEngine:
    def test_accumulates_results(self):
        """search_engine should append new results to existing ones."""
        state = make_state(
            queries=["moon composition"],
            search_results=["existing result"],
            revision_number=1,
        )
        mock_response = {
            "results": [
                {"content": "new result 1"},
                {"content": "new result 2"},
            ]
        }
        # Mock the tavily client at the module level
        with patch("app.fact_checker.tavily_client") as mock_tavily:
            mock_tavily.search.return_value = mock_response
            result = search_engine(state)

        assert "existing result" in result["search_results"]
        assert "new result 1" in result["search_results"]
        assert "new result 2" in result["search_results"]
        assert len(result["search_results"]) == 3

    def test_increments_revision_number(self):
        state = make_state(revision_number=2)
        mock_response = {"results": [{"content": "some result"}]}
        with patch("app.fact_checker.tavily_client") as mock_tavily:
            mock_tavily.search.return_value = mock_response
            result = search_engine(state)

        assert result["revision_number"] == 3

    def test_uses_first_query(self):
        """search_engine should always use queries[0]."""
        state = make_state(queries=["first query", "second query"])
        mock_response = {"results": [{"content": "result"}]}
        with patch("app.fact_checker.tavily_client") as mock_tavily:
            mock_tavily.search.return_value = mock_response
            search_engine(state)
            call_args = mock_tavily.search.call_args
            assert call_args[0][0] == "first query"


# --- research_planner tests ---


class TestResearchPlanner:
    def test_returns_query_list(self):
        """research_planner should return a list of queries from DSPy output."""
        state = make_state(claim="The moon is made of cheese.", search_results=[])
        mock_result = MagicMock()
        mock_result.queries = "moon composition; moon cheese myth; lunar geology facts"

        with patch("app.fact_checker.QueryGenerator") as MockQueryGen:
            instance = MockQueryGen.return_value
            instance.return_value = mock_result
            result = research_planner(state)

        assert result["queries"] == [
            "moon composition",
            "moon cheese myth",
            "lunar geology facts",
        ]

    def test_fallback_on_empty_queries(self):
        """If DSPy returns empty output, fall back to a default query."""
        state = make_state(claim="The moon is made of cheese.")
        mock_result = MagicMock()
        mock_result.queries = ""

        with patch("app.fact_checker.QueryGenerator") as MockQueryGen:
            instance = MockQueryGen.return_value
            instance.return_value = mock_result
            result = research_planner(state)

        assert result["queries"] == ["Fact check The moon is made of cheese."]


# --- Graph routing integration test ---


class TestGraphRouting:
    def test_graph_reaches_end_on_confident_score(self):
        """Full graph should complete when the reporter returns a confident score."""
        checkpointer = MemorySaver()
        app = build_graph(checkpointer)
        config = {"configurable": {"thread_id": "test-thread-1"}}

        with (
            patch("app.fact_checker.tavily_client") as mock_tavily,
            patch("app.fact_checker.asyncio.run") as mock_asyncio_run,
        ):
            mock_tavily.search.return_value = {
                "results": [{"content": "The moon is made of rock."}]
            }
            # Mock the debate to return a confident score immediately
            mock_asyncio_run.return_value = (
                "VERACITY_SCORE: 95\nThe claim is false.",
                95,
            )

            with patch("app.fact_checker.QueryGenerator") as MockQueryGen:
                instance = MockQueryGen.return_value
                mock_result = MagicMock()
                mock_result.queries = "moon composition; moon facts; lunar geology"
                instance.return_value = mock_result

                result = app.invoke(
                    {"claim": "The moon is made of cheese."},
                    config=config,
                )

        assert result["final_report"] is not None
        assert result["veracity_score"] == 95
