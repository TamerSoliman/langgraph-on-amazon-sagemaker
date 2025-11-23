"""
Unit Tests for Individual Agents

Tests each agent (researcher, writer, reviewer) in isolation.

For AI/ML Scientists:
Unit tests verify that individual components work correctly before testing
them together. Like testing each layer of your neural network separately
before training the full model.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from multi_agent_graph import MultiAgentState
from agents.researcher import researcher_node, format_search_results
from agents.writer import writer_node, clean_draft, check_draft_quality
from agents.reviewer import (
    reviewer_node,
    parse_review_decision,
    compute_quality_score,
    MAX_REVISIONS
)


# =============================================================================
# RESEARCHER TESTS
# =============================================================================

def test_researcher_node_success():
    """
    Test researcher agent with successful search.

    For AI/ML Scientists:
    This is like testing a data loader - verify it correctly fetches and
    formats data.
    """

    # Create mock Tavily tool
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = [
        {
            'title': 'Test Article',
            'content': 'Test content about the topic.',
            'url': 'https://example.com/test'
        }
    ]

    # Mock state
    initial_state: MultiAgentState = {
        "question": "What is machine learning?",
        "research": "",
        "draft": "",
        "review_feedback": "",
        "final_answer": "",
        "revision_count": 0
    }

    # Patch Tavily tool creation
    with patch('agents.researcher.create_tavily_tool', return_value=mock_tool):
        result = researcher_node(initial_state)

    # Assertions
    assert result["research"] != "", "Research should be populated"
    assert "Test Article" in result["research"], "Should include article title"
    assert "Test content" in result["research"], "Should include article content"
    assert mock_tool.invoke.call_count == 1, "Should call search once"


def test_researcher_node_error_handling():
    """
    Test researcher agent handles search failures gracefully.

    For AI/ML Scientists:
    Testing failure modes - like testing how your data loader handles
    corrupted files or network errors.
    """

    # Create mock that raises exception
    mock_tool = MagicMock()
    mock_tool.invoke.side_effect = Exception("API Error")

    initial_state: MultiAgentState = {
        "question": "Test question",
        "research": "",
        "draft": "",
        "review_feedback": "",
        "final_answer": "",
        "revision_count": 0
    }

    with patch('agents.researcher.create_tavily_tool', return_value=mock_tool):
        result = researcher_node(initial_state)

    # Should handle error gracefully
    assert "[ERROR]" in result["research"], "Should contain error message"
    assert "Search failed" in result["research"], "Should explain what failed"


def test_format_search_results():
    """
    Test search result formatting utility.

    For AI/ML Scientists:
    Testing preprocessing logic - like testing normalization or
    feature extraction functions.
    """

    mock_results = [
        {
            'title': 'Article 1',
            'content': 'Content 1',
            'url': 'https://example.com/1'
        },
        {
            'title': 'Article 2',
            'content': 'Content 2',
            'url': 'https://example.com/2'
        }
    ]

    formatted = format_search_results(mock_results)

    # Assertions
    assert "RESEARCH FINDINGS" in formatted
    assert "Article 1" in formatted
    assert "Article 2" in formatted
    assert "Content 1" in formatted
    assert "Content 2" in formatted
    assert "https://example.com/1" in formatted


def test_format_search_results_empty():
    """Test formatting with no results."""

    formatted = format_search_results([])

    assert "No search results found" in formatted


# =============================================================================
# WRITER TESTS
# =============================================================================

def test_writer_node_initial_draft():
    """
    Test writer agent creates initial draft.

    For AI/ML Scientists:
    Testing the generative component - like testing if your model
    produces valid output given input features.
    """

    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = """The 2008 crisis was caused by subprime mortgages.

    Banks issued risky loans that eventually defaulted."""

    initial_state: MultiAgentState = {
        "question": "What caused the 2008 financial crisis?",
        "research": "Research about subprime mortgages...",
        "draft": "",
        "review_feedback": "",
        "final_answer": "",
        "revision_count": 0
    }

    with patch('agents.writer.create_sagemaker_llm', return_value=mock_llm):
        result = writer_node(initial_state)

    # Assertions
    assert result["draft"] != "", "Draft should be created"
    assert "2008 crisis" in result["draft"], "Draft should contain relevant content"
    assert result["revision_count"] == 0, "Initial draft shouldn't increment revision count"


def test_writer_node_revision():
    """
    Test writer agent handles revision.

    For AI/ML Scientists:
    Testing if the model can adapt its output based on feedback -
    like testing if fine-tuning improves specific aspects.
    """

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Revised draft with more detail..."

    # State after first draft was rejected
    revision_state: MultiAgentState = {
        "question": "What caused the 2008 financial crisis?",
        "research": "Research about crisis...",
        "draft": "Original draft that was too short.",
        "review_feedback": "Add more detail about regulatory failures",
        "final_answer": "",
        "revision_count": 0
    }

    with patch('agents.writer.create_sagemaker_llm', return_value=mock_llm):
        result = writer_node(revision_state)

    # Should use revision prompt (indicated by revision_count increment)
    assert result["revision_count"] == 1, "Revision count should increment"
    assert result["draft"] == "Revised draft with more detail...", "Should update draft"


def test_clean_draft():
    """
    Test draft cleaning utility.

    For AI/ML Scientists:
    Testing post-processing - like cleaning up model predictions
    by removing artifacts or normalizing outputs.
    """

    # Test removing XML tags
    dirty_draft = "<tag>The answer is: This is the answer</tag>"
    clean = clean_draft(dirty_draft)
    assert "<tag>" not in clean
    assert "</tag>" not in clean

    # Test removing prefix
    prefixed_draft = "Answer: Paris is the capital."
    clean = clean_draft(prefixed_draft)
    assert not clean.startswith("Answer:")
    assert "Paris" in clean

    # Test normalizing whitespace
    spaced_draft = "Too   many    spaces"
    clean = clean_draft(spaced_draft)
    assert "Too many spaces" in clean


def test_check_draft_quality():
    """
    Test draft quality assessment utility.

    For AI/ML Scientists:
    Testing evaluation metrics - like computing precision/recall
    for model outputs.
    """

    good_draft = """The 2008 financial crisis was caused by multiple factors.

    First, according to research, subprime mortgages were issued to
    unqualified borrowers. This led to widespread defaults.

    Second, regulatory failures allowed banks to take excessive risks.
    The repeal of Glass-Steagall contributed to this problem."""

    quality = check_draft_quality(good_draft)

    assert quality["word_count"] > 0
    assert quality["paragraph_count"] >= 2
    assert quality["has_citations"] == True  # Contains "according to"
    assert quality["completeness_score"] > 0.5


# =============================================================================
# REVIEWER TESTS
# =============================================================================

def test_reviewer_node_approval():
    """
    Test reviewer agent approves good draft.

    For AI/ML Scientists:
    Testing the validation/quality control layer - like testing if
    your evaluation metric correctly identifies good predictions.
    """

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "APPROVED"

    state_with_draft: MultiAgentState = {
        "question": "Test question",
        "research": "Test research",
        "draft": "Test draft that is well-written.",
        "review_feedback": "",
        "final_answer": "",
        "revision_count": 0
    }

    with patch('agents.reviewer.create_sagemaker_llm', return_value=mock_llm):
        result = reviewer_node(state_with_draft)

    # Should approve
    assert result["final_answer"] != "", "Should set final answer"
    assert result["final_answer"] == state_with_draft["draft"], "Final answer should be the draft"


def test_reviewer_node_rejection():
    """
    Test reviewer agent requests revision.

    For AI/ML Scientists:
    Testing if quality control correctly identifies poor outputs.
    """

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "NEEDS REVISION: Too brief, add more detail"

    poor_draft_state: MultiAgentState = {
        "question": "Test question",
        "research": "Test research with lots of detail",
        "draft": "Short answer.",
        "review_feedback": "",
        "final_answer": "",
        "revision_count": 0
    }

    with patch('agents.reviewer.create_sagemaker_llm', return_value=mock_llm):
        result = reviewer_node(poor_draft_state)

    # Should request revision
    assert result["final_answer"] == "", "Should not set final answer"
    assert "NEEDS REVISION" in result["review_feedback"], "Should provide feedback"
    assert "detail" in result["review_feedback"], "Should include specific feedback"


def test_reviewer_node_max_revisions():
    """
    Test reviewer force-approves after max revisions.

    For AI/ML Scientists:
    Testing early stopping mechanism - prevents infinite loops.
    """

    # State at max revisions
    max_revision_state: MultiAgentState = {
        "question": "Test",
        "research": "Test",
        "draft": "Still not perfect draft",
        "review_feedback": "",
        "final_answer": "",
        "revision_count": MAX_REVISIONS  # At limit
    }

    # LLM would normally reject, but should force approve
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "NEEDS REVISION: Still problems"

    with patch('agents.reviewer.create_sagemaker_llm', return_value=mock_llm):
        result = reviewer_node(max_revision_state)

    # Should force approve
    assert result["final_answer"] != "", "Should force approve at max revisions"
    assert "max limit" in result["review_feedback"].lower(), "Should explain why approved"


def test_parse_review_decision_approved():
    """
    Test parsing approved review.

    For AI/ML Scientists:
    Testing output parsing - converting unstructured model output
    to structured data.
    """

    decision = parse_review_decision("APPROVED")

    assert decision["approved"] == True
    assert "quality standards" in decision["feedback"].lower()


def test_parse_review_decision_revision():
    """Test parsing revision request."""

    decision = parse_review_decision("NEEDS REVISION: Add more citations")

    assert decision["approved"] == False
    assert "citations" in decision["feedback"]


def test_parse_review_decision_ambiguous():
    """
    Test parsing ambiguous review (fallback logic).

    For AI/ML Scientists:
    Testing edge case handling - what happens when model output
    doesn't match expected format.
    """

    # Positive-leaning ambiguous review
    decision = parse_review_decision("This is good, well-written and clear.")

    assert decision["approved"] == True  # Should default to approval

    # Negative-leaning ambiguous review
    decision = parse_review_decision("This needs improvement and is unclear.")

    assert decision["approved"] == False  # Should default to rejection


def test_compute_quality_score():
    """
    Test heuristic quality scoring.

    For AI/ML Scientists:
    Testing hand-crafted evaluation metric - like testing a
    rule-based baseline before using learned metrics.
    """

    # High quality draft
    good_draft = """The 2008 financial crisis resulted from multiple causes.

    Subprime mortgages were issued to borrowers with poor credit.
    These were packaged into mortgage-backed securities.

    Regulatory failures allowed excessive risk-taking by banks."""

    research = "2008 crisis subprime mortgages regulatory failures securities banks"

    score = compute_quality_score(good_draft, research)

    assert 0 <= score <= 1, "Score should be in [0, 1]"
    assert score > 0.5, "Good draft should score >0.5"

    # Poor quality draft
    bad_draft = "Bad."

    score_bad = compute_quality_score(bad_draft, research)

    assert score_bad < score, "Poor draft should score lower"
    assert score_bad < 0.5, "Poor draft should score <0.5"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    """
    Run tests directly (without pytest).

    Usage:
        python test_agents.py
    """
    print("Running unit tests...")
    pytest.main([__file__, "-v"])
