"""
Writer Agent

This agent synthesizes research findings into a coherent, well-written answer.
It takes facts from the researcher and crafts them into clear prose.

For AI/ML Scientists:
Think of this as the "generation" component in a RAG system. The researcher
retrieved the facts, now the writer generates fluent text based on those facts.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../agent'))

from sagemaker_llm import create_sagemaker_llm
from multi_agent_graph import MultiAgentState


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

WRITER_PROMPT = """You are a skilled writer who creates clear, accurate answers from research findings.

RESEARCH FINDINGS:
{research}

QUESTION:
{question}

YOUR TASK:
Write a comprehensive answer to the question using ONLY the information from the research findings above.

REQUIREMENTS:
1. 2-4 paragraphs maximum
2. Use simple, clear language (explain like I'm smart but unfamiliar with the topic)
3. Cite specific facts from the research
4. Do NOT add information not present in the research
5. If research is insufficient, acknowledge this clearly

Write your answer now:"""

# For AI/ML Scientists:
# This prompt is carefully designed to:
# 1. Constrain the model to use only provided facts (reduces hallucination)
# 2. Set clear length/style requirements (consistency)
# 3. Define quality criteria (accuracy, clarity, citations)
# Think of it like a loss function that specifies what "good output" means.

REVISION_PROMPT = """You are revising a draft answer based on feedback.

ORIGINAL QUESTION:
{question}

RESEARCH FINDINGS:
{research}

YOUR PREVIOUS DRAFT:
{draft}

REVIEWER FEEDBACK:
{feedback}

YOUR TASK:
Revise the draft to address the feedback while maintaining accuracy.

REQUIREMENTS:
1. Address all points in the feedback
2. Keep the same structure (2-4 paragraphs)
3. Still use only information from research findings
4. Improve clarity and completeness

Write your revised answer now:"""


# =============================================================================
# WRITER AGENT NODE
# =============================================================================

def writer_node(state: MultiAgentState) -> MultiAgentState:
    """
    Writer agent: Synthesizes research into a coherent answer.

    Process:
    1. Check if this is first draft or revision
    2. Select appropriate prompt (initial or revision)
    3. Call LLM to generate answer
    4. Update state with draft

    Args:
        state: Current multi-agent state with question and research

    Returns:
        Updated state with 'draft' field populated

    For AI/ML Scientists:
    This is the generative model in your pipeline. It takes structured input
    (research facts) and produces natural language output (the answer).
    """

    # Determine if this is initial draft or revision
    is_revision = state.get("review_feedback", "") != ""

    if is_revision:
        print(f"\n[WRITER] Revising draft (attempt {state['revision_count'] + 1})...")
    else:
        print("\n[WRITER] Writing initial draft...")

    try:
        # Create LLM client
        # For AI/ML Scientists: This connects to SageMaker endpoint running
        # Mistral 7B. Each invoke() call is an inference request (~2-5 seconds).
        llm = create_sagemaker_llm()

        # Select and format prompt
        if is_revision:
            prompt = REVISION_PROMPT.format(
                question=state["question"],
                research=state["research"],
                draft=state["draft"],
                feedback=state["review_feedback"]
            )
        else:
            prompt = WRITER_PROMPT.format(
                question=state["question"],
                research=state["research"]
            )

        # Generate answer
        # For AI/ML Scientists: This is the forward pass through the LLM.
        # Input: text prompt (tokenized internally)
        # Output: generated text (detokenized)
        draft = llm.invoke(prompt)

        # Clean up the output
        draft = clean_draft(draft)

        print(f"[WRITER] ✓ Draft complete ({len(draft)} chars, {count_words(draft)} words)")

        # Update state
        # Note: We increment revision_count only if this was a revision
        return {
            **state,
            "draft": draft,
            "revision_count": state["revision_count"] + 1 if is_revision else 0
        }

    except Exception as e:
        print(f"[WRITER] ⚠️  Error: {e}")

        # Return error state
        return {
            **state,
            "draft": f"[ERROR] Failed to generate answer: {str(e)}",
            "revision_count": state["revision_count"]
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_draft(draft: str) -> str:
    """
    Cleans up LLM output by removing artifacts and normalizing formatting.

    Common issues with LLM output:
    - Extra whitespace
    - Repeated phrases
    - XML tags if model didn't follow instructions
    - Leading/trailing markers

    Args:
        draft: Raw LLM output

    Returns:
        Cleaned draft text

    For AI/ML Scientists:
    Think of this as post-processing - like removing noise or artifacts
    from model predictions before using them.
    """

    # Remove common LLM artifacts
    draft = draft.strip()

    # Remove XML tags if present (shouldn't be, but sometimes models leak them)
    import re
    draft = re.sub(r'<[^>]+>', '', draft)

    # Normalize whitespace (multiple spaces → single space)
    draft = re.sub(r'\s+', ' ', draft)

    # Remove common prefix patterns like "Answer:" or "Here is the answer:"
    prefixes = [
        "answer:",
        "here is the answer:",
        "here's the answer:",
        "the answer is:",
    ]
    for prefix in prefixes:
        if draft.lower().startswith(prefix):
            draft = draft[len(prefix):].strip()

    # Normalize paragraph breaks (2+ newlines → exactly 2)
    draft = re.sub(r'\n{3,}', '\n\n', draft)

    return draft


def count_words(text: str) -> int:
    """Count words in text (simple whitespace split)."""
    return len(text.split())


def count_paragraphs(text: str) -> int:
    """Count paragraphs in text (separated by blank lines)."""
    return len([p for p in text.split('\n\n') if p.strip()])


def check_draft_quality(draft: str) -> dict:
    """
    Performs basic quality checks on draft.

    Returns dict with metrics:
    - word_count
    - paragraph_count
    - has_citations (simple heuristic)
    - completeness_score (0-1)

    For AI/ML Scientists:
    This is like computing evaluation metrics on model output.
    In production, you might use these to decide if the draft is good
    enough to skip the reviewer (optimization).
    """

    word_count = count_words(draft)
    paragraph_count = count_paragraphs(draft)

    # Simple heuristic: Look for citation indicators
    citation_indicators = ['according to', 'research shows', 'studies indicate', 'Source']
    has_citations = any(indicator in draft for indicator in citation_indicators)

    # Completeness: Based on length and structure
    # Ideal: 150-500 words, 2-4 paragraphs
    length_score = min(word_count / 150, 1.0) if word_count < 500 else 0.8
    structure_score = 1.0 if 2 <= paragraph_count <= 4 else 0.7

    completeness_score = (length_score + structure_score) / 2

    return {
        'word_count': word_count,
        'paragraph_count': paragraph_count,
        'has_citations': has_citations,
        'completeness_score': completeness_score
    }


# =============================================================================
# ALTERNATIVE IMPLEMENTATION: Streaming Writer
# =============================================================================

def writer_node_streaming(state: MultiAgentState) -> MultiAgentState:
    """
    Alternative writer implementation that streams output.

    This provides better user experience for long answers by showing
    incremental progress, but adds complexity.

    For AI/ML Scientists:
    Streaming is like online learning - you get partial results as they're
    generated rather than waiting for the full batch.

    Note: Requires streaming-capable LLM setup. Not all SageMaker endpoints
    support streaming.
    """

    print("\n[WRITER] Writing draft (streaming)...")

    llm = create_sagemaker_llm()

    prompt = WRITER_PROMPT.format(
        question=state["question"],
        research=state["research"]
    )

    try:
        draft = ""
        # Stream chunks as they're generated
        for chunk in llm.stream(prompt):
            draft += chunk
            # Could emit progress updates here
            # print(chunk, end='', flush=True)

        draft = clean_draft(draft)

        print(f"\n[WRITER] ✓ Draft complete ({len(draft)} chars)")

        return {
            **state,
            "draft": draft
        }

    except Exception as e:
        print(f"[WRITER] ⚠️  Streaming not supported or failed: {e}")
        # Fall back to regular invoke
        draft = llm.invoke(prompt)
        return {
            **state,
            "draft": clean_draft(draft)
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the writer agent in isolation.

    Usage:
        export SAGEMAKER_ENDPOINT_NAME="your-endpoint"
        python agents/writer.py
    """

    # Create test state with mock research
    test_state: MultiAgentState = {
        "question": "What caused the 2008 financial crisis?",
        "research": """RESEARCH FINDINGS
==================================================

Source 1: 2008 Financial Crisis Overview
The 2008 financial crisis was caused by a combination of factors including
the housing bubble, subprime mortgages, and risky financial instruments.
Banks had issued mortgages to borrowers with poor credit (subprime), then
packaged these into mortgage-backed securities (MBS) and sold them to
investors worldwide.
URL: https://example.com/crisis

--------------------------------------------------

Source 2: Regulatory Failures
Lack of regulatory oversight allowed banks to take excessive risks.
The repeal of Glass-Steagall in 1999 enabled commercial banks to engage
in risky investment banking activities. Credit rating agencies failed to
properly assess the risk of mortgage-backed securities.
URL: https://example.com/regulation

--------------------------------------------------
""",
        "draft": "",
        "review_feedback": "",
        "final_answer": "",
        "revision_count": 0
    }

    print("Testing Writer Agent")
    print("="*70)

    # Test initial draft
    print("\n--- Test 1: Initial Draft ---")
    result = writer_node(test_state)

    print("\nInput:")
    print(f"  Question: {test_state['question']}")
    print(f"  Research: {len(test_state['research'])} chars")

    print("\nOutput:")
    print(f"  Draft:\n{result['draft']}")

    # Check quality
    quality = check_draft_quality(result['draft'])
    print(f"\nQuality Metrics:")
    print(f"  Words: {quality['word_count']}")
    print(f"  Paragraphs: {quality['paragraph_count']}")
    print(f"  Has citations: {quality['has_citations']}")
    print(f"  Completeness: {quality['completeness_score']:.2f}")

    # Test revision
    print("\n\n--- Test 2: Revision ---")
    revision_state = {
        **result,
        "review_feedback": "Add more detail about the role of credit rating agencies."
    }

    revised_result = writer_node(revision_state)

    print("\nRevised Draft:")
    print(revised_result['draft'])
    print(f"\nRevision count: {revised_result['revision_count']}")
