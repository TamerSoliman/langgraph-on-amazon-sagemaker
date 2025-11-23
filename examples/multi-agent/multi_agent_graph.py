"""
Multi-Agent LangGraph Example

This module defines a multi-agent workflow where specialized agents collaborate:
- Researcher: Finds factual information using web search
- Writer: Synthesizes research into a coherent answer
- Reviewer: Checks quality and may request revisions

For AI/ML Scientists:
Think of this as an ensemble pipeline where each "model" (agent) specializes
in a different task. The state flows through the pipeline, getting enriched
at each stage.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from agents.researcher import researcher_node
from agents.writer import writer_node
from agents.reviewer import reviewer_node, should_revise

# =============================================================================
# STATE DEFINITION
# =============================================================================

class MultiAgentState(TypedDict):
    """
    Shared state that flows through all agents.

    For AI/ML Scientists:
    This is like a feature dictionary that gets passed through your pipeline.
    Each agent reads from it and adds new fields, building up the final output.

    Fields:
    - question: Original user question (input)
    - research: Facts found by researcher (intermediate)
    - draft: Answer written by writer (intermediate)
    - review_feedback: Reviewer's comments (intermediate)
    - final_answer: Approved answer (output)
    - revision_count: How many times we've revised (for loop protection)
    """
    question: str                    # User's original question
    research: str                    # Researcher's findings
    draft: str                       # Writer's draft answer
    review_feedback: str             # Reviewer's feedback
    final_answer: str                # Final approved answer
    revision_count: int              # Number of revisions (prevents infinite loops)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_multi_agent_graph() -> StateGraph:
    """
    Builds the multi-agent workflow graph.

    Graph Structure:
        START → Researcher → Writer → Reviewer → [Decision]
                                ↑                    ↓
                                └─── (revise) ───────┤
                                                     ↓
                                                    END

    The reviewer makes a conditional decision:
    - If draft is good → route to END
    - If needs work → route back to Writer for revision
    - After max revisions → force approval to prevent infinite loops

    Returns:
        Compiled StateGraph ready to execute
    """

    # Initialize the graph with our state schema
    # For AI/ML Scientists: This is like defining the input/output signature
    # of your pipeline
    graph = StateGraph(MultiAgentState)

    # -------------------------------------------------------------------------
    # Add nodes (agents)
    # -------------------------------------------------------------------------
    # Each node is a function that takes state and returns updated state

    graph.add_node("researcher", researcher_node)
    # Researcher: Uses Tavily search to find facts
    # Input: question
    # Output: research (search results)

    graph.add_node("writer", writer_node)
    # Writer: Synthesizes research into coherent answer
    # Input: question, research
    # Output: draft

    graph.add_node("reviewer", reviewer_node)
    # Reviewer: Checks quality, may approve or request revision
    # Input: question, research, draft
    # Output: final_answer OR review_feedback

    # -------------------------------------------------------------------------
    # Define edges (flow)
    # -------------------------------------------------------------------------

    # Set entry point
    graph.set_entry_point("researcher")
    # Graph starts by running the researcher

    # Researcher always goes to Writer
    graph.add_edge("researcher", "writer")

    # Writer always goes to Reviewer
    graph.add_edge("writer", "reviewer")

    # Reviewer makes a conditional decision
    graph.add_conditional_edges(
        "reviewer",
        should_revise,  # Decision function
        {
            "revise": "writer",    # Send back for revision
            "approve": END         # We're done!
        }
    )
    # For AI/ML Scientists:
    # This is like having a gating mechanism that decides whether to
    # run another iteration or exit the loop

    # -------------------------------------------------------------------------
    # Compile and return
    # -------------------------------------------------------------------------

    return graph.compile()


# =============================================================================
# MAIN EXECUTION (for local testing)
# =============================================================================

if __name__ == "__main__":
    """
    Run the multi-agent workflow locally.

    Usage:
        python multi_agent_graph.py

    Requirements:
        - TAVILY_API_KEY environment variable
        - SAGEMAKER_ENDPOINT_NAME (or SAGEMAKER_ENDPOINT_URL for mock)
    """
    import os

    # Check required environment variables
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  Warning: TAVILY_API_KEY not set. Researcher will fail.")
        print("   Get a free key from https://app.tavily.com/")

    if not os.getenv("SAGEMAKER_ENDPOINT_NAME") and not os.getenv("SAGEMAKER_ENDPOINT_URL"):
        print("⚠️  Warning: No SageMaker endpoint configured.")
        print("   Set SAGEMAKER_ENDPOINT_NAME (real) or SAGEMAKER_ENDPOINT_URL (mock)")

    # Create the graph
    print("\n" + "="*70)
    print("Multi-Agent Workflow Demo")
    print("="*70 + "\n")

    app = create_multi_agent_graph()

    # Initial state
    initial_state: MultiAgentState = {
        "question": "What were the main causes of the 2008 financial crisis?",
        "research": "",
        "draft": "",
        "review_feedback": "",
        "final_answer": "",
        "revision_count": 0
    }

    print(f"Question: {initial_state['question']}\n")

    # Execute the graph
    # For AI/ML Scientists: This is like calling model.predict() on your pipeline
    try:
        final_state = app.invoke(
            initial_state,
            config={"recursion_limit": 20}  # Max 20 steps (protects against infinite loops)
        )

        # Display results
        print("\n" + "="*70)
        print("Results")
        print("="*70 + "\n")

        print(f"📊 Research Findings ({len(final_state['research'])} chars):")
        print(f"{final_state['research'][:200]}...\n")

        print(f"📝 Draft ({len(final_state['draft'])} chars):")
        print(f"{final_state['draft'][:200]}...\n")

        if final_state['review_feedback']:
            print(f"💬 Review Feedback:")
            print(f"{final_state['review_feedback']}\n")

        print(f"✅ Final Answer ({len(final_state['final_answer'])} chars):")
        print(final_state['final_answer'])

        print(f"\n🔄 Revisions: {final_state['revision_count']}")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("\nTroubleshooting:")
        print("1. Check TAVILY_API_KEY is set")
        print("2. Check SageMaker endpoint is running")
        print("3. Review error message above")

    print("\n" + "="*70 + "\n")
