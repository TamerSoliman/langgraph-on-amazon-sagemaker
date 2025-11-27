"""
Researcher Agent

This agent's job is to find factual information using web search.
It takes the user's question and uses Tavily search tool to gather relevant facts.

For AI/ML Scientists:
Think of this as a retrieval component in a RAG (Retrieval-Augmented Generation)
system. It's fetching external knowledge that the LLM doesn't have in its
training data.
"""

import os
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../agent'))

from langchain_community.tools.tavily_search import TavilySearchResults
from multi_agent_graph import MultiAgentState


# =============================================================================
# TOOL CONFIGURATION
# =============================================================================

def create_tavily_tool() -> TavilySearchResults:
    """
    Creates Tavily search tool for finding current information on the web.

    For AI/ML Scientists:
    Tavily is a search API optimized for LLMs - it returns clean, relevant
    snippets instead of raw HTML. Think of it like a specialized feature
    extractor for web data.

    Requires:
        TAVILY_API_KEY environment variable

    Returns:
        TavilySearchResults tool configured for 3 results per query
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Get a free key from https://app.tavily.com/"
        )

    # Create tool with limited results to reduce costs and noise
    # max_results=3 balances coverage vs. information overload
    tool = TavilySearchResults(
        max_results=3,
        search_depth="basic",  # Options: "basic" or "advanced" (advanced costs more)
        include_answer=True,   # Includes Tavily's AI-generated answer
        include_raw_content=False,  # Don't need full HTML
        include_images=False   # Don't need images for text-based QA
    )

    return tool


# =============================================================================
# RESEARCHER AGENT NODE
# =============================================================================

def researcher_node(state: MultiAgentState) -> MultiAgentState:
    """
    Researcher agent: Finds factual information using web search.

    Process:
    1. Read the question from state
    2. Use Tavily search to find relevant information
    3. Format search results into readable text
    4. Update state with research findings

    Args:
        state: Current multi-agent state containing the question

    Returns:
        Updated state with 'research' field populated

    For AI/ML Scientists:
    This is like a data loader in your training pipeline - it fetches the
    "training examples" (search results) that the downstream agents will use.
    """

    print("\n[RESEARCHER] Starting research...")

    question = state["question"]

    try:
        # Create and invoke the search tool
        tavily_tool = create_tavily_tool()

        # Run the search
        # For AI/ML Scientists: This is an API call, not a local computation.
        # It takes ~500ms-1s depending on query complexity.
        search_results = tavily_tool.invoke(question)

        # Format results for readability
        research_text = format_search_results(search_results)

        print(f"[RESEARCHER] ✓ Found {len(search_results)} sources ({len(research_text)} chars)")

        # Return updated state
        # For AI/ML Scientists: We use the spread operator (**state) to copy
        # existing state, then override only the 'research' field. This is
        # similar to updating a feature dict while preserving other features.
        return {
            **state,
            "research": research_text
        }

    except Exception as e:
        # Handle search failures gracefully
        error_msg = f"Search failed: {str(e)}"
        print(f"[RESEARCHER] ⚠️  {error_msg}")

        # Return state with error message so downstream agents know what happened
        return {
            **state,
            "research": f"[ERROR] {error_msg}\n\nPlease answer based on general knowledge."
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_search_results(results: list) -> str:
    """
    Formats Tavily search results into readable text for the writer agent.

    Args:
        results: List of search result dicts from Tavily

    Returns:
        Formatted string with all search findings

    For AI/ML Scientists:
    This is like preprocessing/feature engineering - we're transforming raw
    API output into a format optimized for downstream consumption (the LLM).
    """

    if not results:
        return "No search results found."

    formatted = "RESEARCH FINDINGS\n" + "="*50 + "\n\n"

    for i, result in enumerate(results, 1):
        # Extract fields from result dict
        # Typical structure: {'url': '...', 'content': '...', 'title': '...'}
        title = result.get('title', 'Untitled')
        content = result.get('content', 'No content available')
        url = result.get('url', '')

        formatted += f"Source {i}: {title}\n"
        formatted += f"{content}\n"
        if url:
            formatted += f"URL: {url}\n"
        formatted += "\n" + "-"*50 + "\n\n"

    return formatted


def extract_key_facts(research: str, max_facts: int = 5) -> list:
    """
    Extracts key facts from research text using simple heuristics.

    This is a helper function for more advanced use cases where you want to
    filter or prioritize information before passing to the writer.

    Args:
        research: Raw research text
        max_facts: Maximum number of facts to extract

    Returns:
        List of key facts (strings)

    For AI/ML Scientists:
    This is like applying feature selection - we're identifying the most
    important pieces of information to reduce noise for downstream models.

    Note: In production, you might use an LLM call here to do more sophisticated
    extraction, but that adds latency and cost.
    """

    # Simple heuristic: Look for sentences with key indicator words
    indicators = ['because', 'caused by', 'resulted in', 'due to', 'led to', 'reason']

    facts = []
    sentences = research.split('.')

    for sentence in sentences:
        sentence = sentence.strip()
        if any(indicator in sentence.lower() for indicator in indicators):
            facts.append(sentence + '.')

        if len(facts) >= max_facts:
            break

    return facts if facts else ["No key facts extracted."]


# =============================================================================
# ALTERNATIVE IMPLEMENTATION: LLM-Enhanced Research
# =============================================================================

def researcher_node_with_llm(state: MultiAgentState) -> MultiAgentState:
    """
    Alternative researcher implementation that uses LLM to improve search query.

    Process:
    1. Use LLM to generate better search query from question
    2. Run Tavily search with improved query
    3. Use LLM again to summarize/extract key findings
    4. Return summarized research

    Trade-offs:
    - Better quality search results
    - More expensive (+2 LLM calls)
    - Higher latency (+4-10 seconds)

    For AI/ML Scientists:
    This is like adding a pre-processing model before your main model.
    Helps when input quality is variable, but adds computational overhead.
    """

    from sagemaker_llm import create_sagemaker_llm

    print("\n[RESEARCHER] Starting LLM-enhanced research...")

    question = state["question"]
    llm = create_sagemaker_llm()

    try:
        # Step 1: Generate better search query
        query_prompt = f"""Generate a specific, focused search query for this question:

Question: {question}

Return only the search query, no explanation."""

        search_query = llm.invoke(query_prompt)
        print(f"[RESEARCHER] Generated search query: {search_query}")

        # Step 2: Run search with improved query
        tavily_tool = create_tavily_tool()
        search_results = tavily_tool.invoke(search_query)

        # Step 3: Summarize findings using LLM
        raw_results = format_search_results(search_results)

        summary_prompt = f"""Summarize the key facts from these search results:

{raw_results}

Focus on information relevant to: {question}

Format as bullet points."""

        research_summary = llm.invoke(summary_prompt)

        print(f"[RESEARCHER] ✓ Research complete (enhanced)")

        return {
            **state,
            "research": research_summary
        }

    except Exception as e:
        print(f"[RESEARCHER] ⚠️  Error: {e}")
        return {
            **state,
            "research": f"[ERROR] Research failed: {str(e)}"
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the researcher agent in isolation.

    Usage:
        export TAVILY_API_KEY="your-key"
        python agents/researcher.py
    """

    # Create test state
    test_state: MultiAgentState = {
        "question": "What caused the 2008 financial crisis?",
        "research": "",
        "draft": "",
        "review_feedback": "",
        "final_answer": "",
        "revision_count": 0
    }

    print("Testing Researcher Agent")
    print("="*70)

    # Run researcher
    result = researcher_node(test_state)

    # Display results
    print("\nInput:")
    print(f"  Question: {test_state['question']}")

    print("\nOutput:")
    print(f"  Research: {result['research'][:500]}...")
    print(f"  Total length: {len(result['research'])} characters")
