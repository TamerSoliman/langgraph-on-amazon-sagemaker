"""
Integration Tests for Multi-Agent Workflow

Tests the full workflow from start to finish with mocked LLM and tools.

For AI/ML Scientists:
Integration tests verify that multiple components work together correctly.
Think of it like testing your full training pipeline end-to-end, not just
individual modules.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multi_agent_graph import create_multi_agent_graph, MultiAgentState
from agents.reviewer import should_revise


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm():
    """
    Creates a mock LLM that returns predefined responses.

    For AI/ML Scientists:
    Mocking lets us test logic without calling expensive real services.
    Like using synthetic data instead of downloading real datasets.
    """
    mock = MagicMock()

    # Define different responses for different agents
    # We'll use side_effect to return different values for each call
    mock.invoke.side_effect = [
        # Researcher generates search query (if using LLM-enhanced mode)
        # "2008 financial crisis causes",

        # Writer creates initial draft
        """The 2008 financial crisis was caused by several factors.

        First, subprime mortgages were issued to unqualified borrowers,
        then packaged into mortgage-backed securities.

        Second, regulatory failures allowed excessive risk-taking by
        financial institutions.""",

        # Reviewer approves
        "APPROVED"
    ]

    return mock


@pytest.fixture
def mock_tavily_tool():
    """
    Mocks the Tavily search tool.

    Returns fake search results instead of calling the real API.
    """
    mock = MagicMock()

    mock.invoke.return_value = [
        {
            'title': 'Financial Crisis Causes',
            'content': 'The 2008 crisis was caused by subprime mortgages, lack of regulation, and risky securities.',
            'url': 'https://example.com/crisis'
        },
        {
            'title': 'Regulatory Failures',
            'content': 'Lack of oversight allowed banks to take excessive risks.',
            'url': 'https://example.com/regulation'
        }
    ]

    return mock


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_full_workflow_approval(mock_llm, mock_tavily_tool):
    """
    Test complete workflow where draft is approved on first try.

    Flow:
    1. Researcher finds information
    2. Writer creates draft
    3. Reviewer approves
    4. Graph completes with final answer

    For AI/ML Scientists:
    This is like testing the "happy path" through your model - everything
    works perfectly, no errors or edge cases.
    """

    # Patch both LLM and Tavily tool
    with patch('agents.researcher.create_tavily_tool', return_value=mock_tavily_tool):
        with patch('agents.writer.create_sagemaker_llm', return_value=mock_llm):
            with patch('agents.reviewer.create_sagemaker_llm', return_value=mock_llm):

                # Create graph
                app = create_multi_agent_graph()

                # Initial state
                initial_state: MultiAgentState = {
                    "question": "What caused the 2008 financial crisis?",
                    "research": "",
                    "draft": "",
                    "review_feedback": "",
                    "final_answer": "",
                    "revision_count": 0
                }

                # Execute
                result = app.invoke(initial_state)

                # Assertions
                assert result["research"] != "", "Research should be populated"
                assert result["draft"] != "", "Draft should be created"
                assert result["final_answer"] != "", "Final answer should be set"
                assert result["revision_count"] == 0, "Should be approved without revisions"

                # Verify tools were called
                mock_tavily_tool.invoke.assert_called_once()
                # Writer and reviewer should each be called once
                assert mock_llm.invoke.call_count >= 2


def test_workflow_with_revision(mock_tavily_tool):
    """
    Test workflow where reviewer requests revision.

    Flow:
    1. Researcher finds information
    2. Writer creates draft
    3. Reviewer requests revision
    4. Writer creates revision
    5. Reviewer approves
    6. Graph completes

    For AI/ML Scientists:
    This tests a more complex path through the system - like testing how
    your model handles data it initially gets wrong.
    """

    # Create mock LLM with revision cycle
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        # Writer: Initial draft
        "The crisis was bad.",  # Too short, will be rejected

        # Reviewer: Reject
        "NEEDS REVISION: Too brief, add more detail",

        # Writer: Revised draft
        """The 2008 financial crisis was caused by multiple factors.

        Subprime mortgages, regulatory failures, and risky securities
        all contributed to the collapse.""",

        # Reviewer: Approve
        "APPROVED"
    ]

    with patch('agents.researcher.create_tavily_tool', return_value=mock_tavily_tool):
        with patch('agents.writer.create_sagemaker_llm', return_value=mock_llm):
            with patch('agents.reviewer.create_sagemaker_llm', return_value=mock_llm):

                app = create_multi_agent_graph()

                initial_state: MultiAgentState = {
                    "question": "What caused the 2008 financial crisis?",
                    "research": "",
                    "draft": "",
                    "review_feedback": "",
                    "final_answer": "",
                    "revision_count": 0
                }

                result = app.invoke(initial_state)

                # Assertions
                assert result["final_answer"] != "", "Should have final answer"
                assert result["revision_count"] == 1, "Should have exactly 1 revision"
                assert "NEEDS REVISION" in result.get("review_feedback", ""), \
                    "Review feedback should be recorded"


def test_max_revisions_force_approval(mock_tavily_tool):
    """
    Test that graph doesn't loop forever if reviewer keeps rejecting.

    After MAX_REVISIONS, should force approval.

    For AI/ML Scientists:
    This is like early stopping in training - prevents overfitting/infinite loops.
    """

    # Create mock LLM that always rejects (to test safety mechanism)
    mock_llm = MagicMock()

    # Generate enough rejections to hit max revisions
    rejections = ["NEEDS REVISION: Still not good"] * 10

    mock_llm.invoke.side_effect = [
        "Draft attempt 1",  # Writer
        rejections[0],      # Reviewer: reject
        "Draft attempt 2",  # Writer: revision 1
        rejections[1],      # Reviewer: reject again
        "Draft attempt 3",  # Writer: revision 2
        # After this, should force approve
    ]

    with patch('agents.researcher.create_tavily_tool', return_value=mock_tavily_tool):
        with patch('agents.writer.create_sagemaker_llm', return_value=mock_llm):
            with patch('agents.reviewer.create_sagemaker_llm', return_value=mock_llm):

                app = create_multi_agent_graph()

                initial_state: MultiAgentState = {
                    "question": "Test question",
                    "research": "",
                    "draft": "",
                    "review_feedback": "",
                    "final_answer": "",
                    "revision_count": 0
                }

                result = app.invoke(initial_state, config={"recursion_limit": 20})

                # Should force approve after max revisions
                assert result["final_answer"] != "", "Should have final answer (forced)"
                assert result["revision_count"] >= 2, "Should hit max revisions"


def test_error_handling_in_researcher(mock_llm):
    """
    Test that workflow handles researcher errors gracefully.

    If Tavily search fails, should still proceed with error message.

    For AI/ML Scientists:
    Testing failure modes is crucial - like testing how your model handles
    corrupted inputs or missing data.
    """

    # Create mock that raises exception
    mock_tavily_error = MagicMock()
    mock_tavily_error.invoke.side_effect = Exception("API rate limit exceeded")

    mock_llm.invoke.side_effect = [
        # Writer gets error message in research, still produces draft
        "Unable to find specific information, answering from general knowledge.",

        # Reviewer approves anyway
        "APPROVED"
    ]

    with patch('agents.researcher.create_tavily_tool', return_value=mock_tavily_error):
        with patch('agents.writer.create_sagemaker_llm', return_value=mock_llm):
            with patch('agents.reviewer.create_sagemaker_llm', return_value=mock_llm):

                app = create_multi_agent_graph()

                initial_state: MultiAgentState = {
                    "question": "Test question",
                    "research": "",
                    "draft": "",
                    "review_feedback": "",
                    "final_answer": "",
                    "revision_count": 0
                }

                result = app.invoke(initial_state)

                # Should complete despite error
                assert result["final_answer"] != "", "Should still produce answer"
                assert "[ERROR]" in result["research"], "Research should contain error message"


# =============================================================================
# ROUTING LOGIC TESTS
# =============================================================================

def test_should_revise_logic():
    """
    Test the conditional routing logic.

    For AI/ML Scientists:
    This tests the decision boundary - like testing if your classifier
    correctly separates classes.
    """

    # Case 1: Has final_answer → approve
    approved_state: MultiAgentState = {
        "question": "test",
        "research": "test",
        "draft": "test",
        "review_feedback": "good",
        "final_answer": "This is the final answer",
        "revision_count": 0
    }

    assert should_revise(approved_state) == "approve"

    # Case 2: No final_answer → revise
    needs_revision_state: MultiAgentState = {
        "question": "test",
        "research": "test",
        "draft": "test",
        "review_feedback": "needs work",
        "final_answer": "",  # Empty
        "revision_count": 1
    }

    assert should_revise(needs_revision_state) == "revise"


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.slow
def test_workflow_performance(mock_llm, mock_tavily_tool):
    """
    Test that workflow completes within reasonable time.

    For AI/ML Scientists:
    Like measuring inference latency for your model. Important for
    production deployments with SLA requirements.
    """
    import time

    with patch('agents.researcher.create_tavily_tool', return_value=mock_tavily_tool):
        with patch('agents.writer.create_sagemaker_llm', return_value=mock_llm):
            with patch('agents.reviewer.create_sagemaker_llm', return_value=mock_llm):

                app = create_multi_agent_graph()

                initial_state: MultiAgentState = {
                    "question": "Test question",
                    "research": "",
                    "draft": "",
                    "review_feedback": "",
                    "final_answer": "",
                    "revision_count": 0
                }

                start_time = time.time()
                result = app.invoke(initial_state)
                elapsed_time = time.time() - start_time

                # With mocked LLM, should be very fast (<1 second)
                assert elapsed_time < 1.0, f"Workflow took {elapsed_time}s (should be <1s with mocks)"

                assert result["final_answer"] != ""


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    """
    Run tests directly (without pytest).

    Usage:
        python test_graph.py
    """
    print("Running integration tests...")
    pytest.main([__file__, "-v"])
