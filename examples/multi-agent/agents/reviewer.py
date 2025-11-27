"""
Reviewer Agent

This agent checks the quality of the writer's draft and decides whether to
approve it or request revisions.

For AI/ML Scientists:
Think of this as a quality control / validation layer. It's like having a
classifier that determines if the model's output meets your quality threshold.
"""

import os
import sys
from typing import Literal

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../agent'))

from sagemaker_llm import create_sagemaker_llm
from multi_agent_graph import MultiAgentState


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_REVISIONS = 2
# Prevent infinite revision loops. After 2 revisions, force approval.
# For AI/ML Scientists: This is like early stopping - prevents overfitting
# to reviewer's criteria at the cost of infinite iterations.


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

REVIEWER_PROMPT = """You are a quality reviewer checking an answer for accuracy, clarity, and completeness.

QUESTION:
{question}

RESEARCH FINDINGS:
{research}

DRAFT ANSWER:
{draft}

YOUR TASK:
Review the draft answer and determine if it meets quality standards.

QUALITY CRITERIA:
1. ACCURACY: Does it only use facts from the research? No hallucinations?
2. COMPLETENESS: Does it fully answer the question?
3. CLARITY: Is it well-written and easy to understand?
4. STRUCTURE: Is it well-organized with clear paragraphs?

INSTRUCTIONS:
- If the draft meets ALL criteria, respond with exactly: APPROVED
- If the draft needs improvement, respond with: NEEDS REVISION: [specific feedback]

Your review:"""

# For AI/ML Scientists:
# This prompt creates a binary classifier with feedback.
# Output space: {APPROVED, NEEDS REVISION: <reason>}
# The structured output format makes it easy to parse and route.


# =============================================================================
# REVIEWER AGENT NODE
# =============================================================================

def reviewer_node(state: MultiAgentState) -> MultiAgentState:
    """
    Reviewer agent: Checks draft quality and decides next step.

    Process:
    1. Check if max revisions reached → force approval
    2. Otherwise, use LLM to review draft quality
    3. Parse LLM response to determine approval/revision
    4. Update state with decision

    Args:
        state: Current multi-agent state with question, research, and draft

    Returns:
        Updated state with either:
        - final_answer set (if approved)
        - review_feedback set (if needs revision)

    For AI/ML Scientists:
    This is a gating mechanism / quality filter. It's like evaluating a model's
    prediction and deciding if it passes your acceptance threshold.
    """

    print("\n[REVIEWER] Reviewing draft...")

    # Safety check: Force approval after max revisions
    if state["revision_count"] >= MAX_REVISIONS:
        print(f"[REVIEWER] ⚠️  Max revisions ({MAX_REVISIONS}) reached. Force approving.")

        return {
            **state,
            "final_answer": state["draft"],
            "review_feedback": f"Approved after {MAX_REVISIONS} revisions (max limit reached)"
        }

    try:
        # Create LLM client
        llm = create_sagemaker_llm()

        # Format review prompt
        prompt = REVIEWER_PROMPT.format(
            question=state["question"],
            research=state["research"],
            draft=state["draft"]
        )

        # Get review from LLM
        # For AI/ML Scientists: This is running inference on a classifier model.
        # Input: question + research + draft
        # Output: binary decision + optional feedback
        review = llm.invoke(prompt)

        # Parse review decision
        decision = parse_review_decision(review)

        if decision["approved"]:
            print("[REVIEWER] ✓ Draft approved!")

            return {
                **state,
                "final_answer": state["draft"],
                "review_feedback": decision["feedback"]
            }
        else:
            print(f"[REVIEWER] ⚠️  Needs revision: {decision['feedback'][:100]}...")

            return {
                **state,
                "review_feedback": decision["feedback"]
            }

    except Exception as e:
        print(f"[REVIEWER] ⚠️  Error during review: {e}")
        print("[REVIEWER] Defaulting to approval due to error")

        # On error, approve to prevent blocking the pipeline
        # For AI/ML Scientists: This is a fail-safe - prefer false positives
        # (approving mediocre drafts) over false negatives (blocking good drafts)
        return {
            **state,
            "final_answer": state["draft"],
            "review_feedback": f"Auto-approved due to review error: {str(e)}"
        }


# =============================================================================
# DECISION ROUTING
# =============================================================================

def should_revise(state: MultiAgentState) -> Literal["approve", "revise"]:
    """
    Decides whether to approve the draft or send back for revision.

    This function is used by LangGraph for conditional routing.

    Args:
        state: Current multi-agent state

    Returns:
        "approve" if final_answer is set (reviewer approved)
        "revise" if review_feedback is set but no final_answer (needs work)

    For AI/ML Scientists:
    This is the routing logic for the conditional edge in the graph.
    It's like an if/else statement based on the state:
        if final_answer exists: route to END
        else: route back to writer
    """

    # If final_answer is set and non-empty, we're approved
    if state.get("final_answer", "").strip():
        return "approve"
    else:
        return "revise"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_review_decision(review: str) -> dict:
    """
    Parses LLM review output into structured decision.

    Expected formats:
    - "APPROVED" → approved=True, feedback="Draft meets quality standards"
    - "NEEDS REVISION: <reason>" → approved=False, feedback=<reason>

    Args:
        review: Raw LLM output from reviewer

    Returns:
        Dict with 'approved' (bool) and 'feedback' (str)

    For AI/ML Scientists:
    This is output parsing - converting unstructured text into structured
    data for programmatic decision making.
    """

    review = review.strip()

    # Check for approval
    if "APPROVED" in review.upper():
        return {
            "approved": True,
            "feedback": "Draft meets quality standards."
        }

    # Check for revision request
    if "NEEDS REVISION:" in review.upper():
        # Extract feedback after "NEEDS REVISION:"
        import re
        match = re.search(r'NEEDS REVISION:\s*(.+)', review, re.IGNORECASE | re.DOTALL)
        if match:
            feedback = match.group(1).strip()
        else:
            feedback = "Needs improvement (no specific feedback provided)."

        return {
            "approved": False,
            "feedback": feedback
        }

    # Fallback: If format is unclear, look for positive/negative keywords
    positive_keywords = ['good', 'excellent', 'well-written', 'clear', 'accurate']
    negative_keywords = ['improve', 'missing', 'unclear', 'inaccurate', 'needs']

    positive_count = sum(1 for kw in positive_keywords if kw in review.lower())
    negative_count = sum(1 for kw in negative_keywords if kw in review.lower())

    if positive_count > negative_count:
        return {
            "approved": True,
            "feedback": review
        }
    else:
        return {
            "approved": False,
            "feedback": review
        }


def compute_quality_score(draft: str, research: str) -> float:
    """
    Computes a simple heuristic quality score (0-1) for the draft.

    This is a rule-based alternative to LLM review - faster and cheaper
    but less sophisticated.

    Metrics:
    - Length appropriateness (150-500 words)
    - Contains facts from research (keyword overlap)
    - Has good structure (2-4 paragraphs)

    Args:
        draft: The draft answer
        research: The research findings

    Returns:
        Quality score from 0 (terrible) to 1 (perfect)

    For AI/ML Scientists:
    This is like a handcrafted feature-based model. Simpler and faster than
    deep learning (LLM review) but less accurate. Use when speed/cost matter
    more than quality.
    """

    # 1. Length score
    word_count = len(draft.split())
    if 150 <= word_count <= 500:
        length_score = 1.0
    elif word_count < 150:
        length_score = word_count / 150  # Penalty for too short
    else:
        length_score = 0.8  # Mild penalty for too long

    # 2. Fact coverage score (what % of research keywords appear in draft?)
    research_words = set(research.lower().split())
    draft_words = set(draft.lower().split())

    # Filter out common words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    research_keywords = research_words - stopwords
    draft_keywords = draft_words - stopwords

    if len(research_keywords) > 0:
        overlap = len(research_keywords & draft_keywords)
        coverage_score = min(overlap / len(research_keywords), 1.0)
    else:
        coverage_score = 0.5  # Default if no keywords

    # 3. Structure score
    paragraph_count = len([p for p in draft.split('\n\n') if p.strip()])
    structure_score = 1.0 if 2 <= paragraph_count <= 4 else 0.7

    # Weighted average
    quality_score = (
        0.3 * length_score +
        0.5 * coverage_score +
        0.2 * structure_score
    )

    return quality_score


# =============================================================================
# ALTERNATIVE IMPLEMENTATION: Rule-Based Reviewer
# =============================================================================

def reviewer_node_heuristic(state: MultiAgentState) -> MultiAgentState:
    """
    Alternative reviewer using heuristics instead of LLM.

    Pros:
    - Faster (~10ms vs ~2 seconds)
    - Cheaper (no LLM call)
    - Deterministic (same input → same output)

    Cons:
    - Less sophisticated (can't catch subtle issues)
    - Requires manual tuning of rules
    - May miss context-specific problems

    For AI/ML Scientists:
    This is like using a rule-based baseline model instead of a neural network.
    Good for simple cases, but deep learning (LLM) handles edge cases better.
    """

    print("\n[REVIEWER] Reviewing draft (heuristic mode)...")

    # Compute quality score
    quality_score = compute_quality_score(state["draft"], state["research"])

    print(f"[REVIEWER] Quality score: {quality_score:.2f}")

    # Threshold for approval
    QUALITY_THRESHOLD = 0.7

    if quality_score >= QUALITY_THRESHOLD:
        print("[REVIEWER] ✓ Draft approved (heuristic)")

        return {
            **state,
            "final_answer": state["draft"],
            "review_feedback": f"Approved (quality score: {quality_score:.2f})"
        }
    else:
        # Generate specific feedback based on what's lacking
        feedback_items = []

        word_count = len(state["draft"].split())
        if word_count < 150:
            feedback_items.append(f"Too short ({word_count} words, need 150+)")

        research_words = set(state["research"].lower().split())
        draft_words = set(state["draft"].lower().split())
        overlap = len(research_words & draft_words)
        if overlap < 10:
            feedback_items.append("Not enough facts from research")

        paragraph_count = len([p for p in state["draft"].split('\n\n') if p.strip()])
        if paragraph_count < 2:
            feedback_items.append("Needs better paragraph structure (2+ paragraphs)")

        feedback = "NEEDS REVISION: " + "; ".join(feedback_items)

        print(f"[REVIEWER] ⚠️  {feedback}")

        return {
            **state,
            "review_feedback": feedback
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the reviewer agent in isolation.

    Usage:
        export SAGEMAKER_ENDPOINT_NAME="your-endpoint"
        python agents/reviewer.py
    """

    # Test case 1: Good draft (should approve)
    good_draft_state: MultiAgentState = {
        "question": "What caused the 2008 financial crisis?",
        "research": """The 2008 financial crisis was caused by subprime mortgages,
                      lack of regulation, and risky mortgage-backed securities.""",
        "draft": """The 2008 financial crisis resulted from multiple interconnected factors.

                   First, banks issued subprime mortgages to borrowers with poor credit,
                   then packaged these risky loans into mortgage-backed securities. When
                   housing prices declined, these securities collapsed in value.

                   Second, regulatory failures allowed excessive risk-taking. The repeal
                   of Glass-Steagall enabled banks to engage in risky investment activities
                   without proper oversight.

                   These factors combined to create a systemic financial collapse that
                   affected the global economy.""",
        "review_feedback": "",
        "final_answer": "",
        "revision_count": 0
    }

    print("Testing Reviewer Agent")
    print("="*70)

    print("\n--- Test 1: Good Draft (Should Approve) ---")
    result1 = reviewer_node(good_draft_state)

    print(f"Decision: {'APPROVED' if result1.get('final_answer') else 'NEEDS REVISION'}")
    print(f"Feedback: {result1.get('review_feedback', '')[:200]}")

    # Test case 2: Poor draft (should reject)
    poor_draft_state: MultiAgentState = {
        "question": "What caused the 2008 financial crisis?",
        "research": """The 2008 financial crisis was caused by subprime mortgages,
                      lack of regulation, and risky mortgage-backed securities.""",
        "draft": "It was caused by bad stuff.",  # Too short, vague
        "review_feedback": "",
        "final_answer": "",
        "revision_count": 0
    }

    print("\n\n--- Test 2: Poor Draft (Should Reject) ---")
    result2 = reviewer_node_heuristic(poor_draft_state)  # Use heuristic for testing

    print(f"Decision: {'APPROVED' if result2.get('final_answer') else 'NEEDS REVISION'}")
    print(f"Feedback: {result2.get('review_feedback', '')}")

    # Test case 3: Max revisions (should force approve)
    max_revisions_state: MultiAgentState = {
        **poor_draft_state,
        "revision_count": MAX_REVISIONS
    }

    print("\n\n--- Test 3: Max Revisions (Should Force Approve) ---")
    result3 = reviewer_node(max_revisions_state)

    print(f"Decision: {'APPROVED' if result3.get('final_answer') else 'NEEDS REVISION'}")
    print(f"Feedback: {result3.get('review_feedback', '')}")

    # Test routing function
    print("\n\n--- Test 4: Routing Logic ---")
    print(f"Approved state routes to: {should_revise(result1)}")
    print(f"Revision state routes to: {should_revise(result2)}")
