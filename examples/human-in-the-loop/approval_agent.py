"""
Approval Pattern: Human-in-the-Loop Agent

This agent generates a plan, waits for human approval, then executes.

For AI/ML Scientists:
Think of this as adding a validation gate in your ML pipeline. The model
generates predictions, a human reviews them, and only approved predictions
are used.
"""

import os
import sys
from typing import TypedDict, Optional, Literal

# Add parent directories for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../agent'))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sagemaker_llm import create_sagemaker_llm


# =============================================================================
# STATE DEFINITION
# =============================================================================

class ApprovalState(TypedDict):
    """
    State for approval workflow.

    For AI/ML Scientists:
    This is like the data structure that flows through your pipeline.
    Each node reads from it and adds new fields.
    """
    input: str                          # User's original request
    plan: str                           # Agent's proposed plan
    approved: Optional[bool]            # Human approval decision
    rejection_reason: Optional[str]     # If rejected, why?
    execution_result: str               # Result of execution
    request_id: str                     # Unique ID for tracking


# =============================================================================
# AGENT NODES
# =============================================================================

def planner_node(state: ApprovalState) -> ApprovalState:
    """
    Plans what action to take based on user input.

    For AI/ML Scientists:
    This is like your model's forward pass - it generates a proposal
    based on input features.

    Args:
        state: Current workflow state

    Returns:
        Updated state with plan
    """

    print(f"\n[PLANNER] Analyzing request: {state['input']}")

    # Use LLM to generate plan
    try:
        llm = create_sagemaker_llm()

        prompt = f"""You are an AI assistant that creates action plans.

User request: {state['input']}

Create a clear, specific action plan. Be concise (2-3 sentences).
Format: "I will [action 1], then [action 2], and finally [action 3]."

Action plan:"""

        plan = llm.invoke(prompt)

        # Clean up response
        plan = plan.strip()

        print(f"[PLANNER] ✓ Plan created: {plan[:100]}...")

        return {
            **state,
            "plan": plan
        }

    except Exception as e:
        print(f"[PLANNER] ⚠️  Error: {e}")

        # Fallback: Simple plan
        return {
            **state,
            "plan": f"Process the request: {state['input']}"
        }


def approval_node(state: ApprovalState) -> ApprovalState:
    """
    Waits for human approval (this is the interrupt point).

    For AI/ML Scientists:
    This is the validation step where humans review model outputs.
    In production, this would be async (web UI, message queue, etc.).

    Args:
        state: Current state with plan

    Returns:
        State unchanged (waiting for human input via update_state)
    """

    print(f"\n{'='*70}")
    print("⏸️  HUMAN APPROVAL REQUIRED")
    print(f"{'='*70}")
    print(f"\nRequest ID: {state.get('request_id', 'N/A')}")
    print(f"\nOriginal Request:")
    print(f"  {state['input']}")
    print(f"\nProposed Plan:")
    print(f"  {state['plan']}")
    print(f"\n{'-'*70}")

    # In production, this would publish to approval queue
    # For now, this is just a marker - execution pauses here

    return state


def executor_node(state: ApprovalState) -> ApprovalState:
    """
    Executes the approved plan.

    For AI/ML Scientists:
    This is like deploying approved model predictions to production.
    Only runs if human approved the plan.

    Args:
        state: State with approval decision

    Returns:
        State with execution result
    """

    if not state.get("approved"):
        print(f"\n[EXECUTOR] ❌ Plan rejected: {state.get('rejection_reason', 'No reason given')}")

        return {
            **state,
            "execution_result": "Execution cancelled - plan was not approved"
        }

    print(f"\n[EXECUTOR] ✓ Plan approved. Executing...")

    # In real implementation, this would execute the actual plan
    # For demo, we'll just simulate execution

    try:
        llm = create_sagemaker_llm()

        prompt = f"""Execute this plan and report the result:

Plan: {state['plan']}

Simulate execution and provide a brief result summary.

Result:"""

        result = llm.invoke(prompt)
        result = result.strip()

        print(f"[EXECUTOR] ✓ Execution complete")

        return {
            **state,
            "execution_result": result
        }

    except Exception as e:
        print(f"[EXECUTOR] ⚠️  Error during execution: {e}")

        return {
            **state,
            "execution_result": f"Execution failed: {str(e)}"
        }


# =============================================================================
# CONDITIONAL ROUTING
# =============================================================================

def check_approval(state: ApprovalState) -> Literal["execute", "cancel"]:
    """
    Determines next step based on approval status.

    For AI/ML Scientists:
    This is like a conditional branch in your pipeline:
    if approved: deploy()
    else: discard()

    Args:
        state: Current state

    Returns:
        "execute" if approved, "cancel" if rejected
    """

    if state.get("approved") == True:
        return "execute"
    else:
        return "cancel"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_approval_agent():
    """
    Creates the approval workflow graph.

    Graph structure:
        START → Planner → Approval → [Decision]
                            ↓           ↓
                          [WAIT]    Execute / Cancel → END

    For AI/ML Scientists:
    The interrupt_before parameter is key - it pauses execution before
    the approval node, allowing external input (human decision).

    Returns:
        Compiled graph with checkpoint support
    """

    # Initialize graph
    graph = StateGraph(ApprovalState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("approval", approval_node)
    graph.add_node("executor", executor_node)

    # Set entry point
    graph.set_entry_point("planner")

    # Add edges
    graph.add_edge("planner", "approval")

    # Conditional edge after approval
    graph.add_conditional_edges(
        "approval",
        check_approval,
        {
            "execute": "executor",
            "cancel": END
        }
    )

    graph.add_edge("executor", END)

    # Compile with checkpointer and interrupt
    # For AI/ML Scientists: MemorySaver persists state between runs
    # In production, use PostgresSaver for persistence across restarts
    checkpointer = MemorySaver()

    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["approval"]  # ⚠️ CRITICAL: Pauses here
    )

    return app


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_human_approval_cli(app, config) -> bool:
    """
    Gets human approval via command line (for testing).

    In production, this would be replaced by:
    - Web UI with approve/reject buttons
    - Slack notification with reaction emojis
    - API endpoint that receives approval via POST request

    Args:
        app: Compiled graph application
        config: Configuration with thread_id

    Returns:
        True if approved, False if rejected
    """

    print(f"\nOptions:")
    print("  (a)pprove - Execute the plan")
    print("  (r)eject  - Cancel execution")
    print("  (v)iew    - View current state")

    while True:
        choice = input("\nYour decision [a/r/v]: ").strip().lower()

        if choice == 'a':
            print("\n✓ Approved")
            return True

        elif choice == 'r':
            reason = input("Rejection reason (optional): ").strip()
            print(f"\n✗ Rejected: {reason or 'No reason given'}")
            return False

        elif choice == 'v':
            # View current state
            current_state = app.get_state(config)
            print(f"\n{'-'*70}")
            print("Current State:")
            for key, value in current_state.values.items():
                print(f"  {key}: {value}")
            print(f"{'-'*70}")

        else:
            print("Invalid choice. Use 'a', 'r', or 'v'.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_approval_workflow(user_request: str, auto_approve: Optional[bool] = None):
    """
    Runs the complete approval workflow.

    For AI/ML Scientists:
    This demonstrates the two-phase execution:
    1. Stream until interrupt (planner → approval)
    2. Update state with human decision
    3. Stream to completion (executor → end)

    Args:
        user_request: User's input
        auto_approve: If provided, automatically approve/reject (for testing)

    Returns:
        Final state
    """

    import uuid

    print("\n" + "="*70)
    print("Approval Workflow Demo")
    print("="*70)

    # Create agent
    app = create_approval_agent()

    # Configuration (thread_id allows state persistence)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Initial state
    initial_state: ApprovalState = {
        "input": user_request,
        "plan": "",
        "approved": None,
        "rejection_reason": None,
        "execution_result": "",
        "request_id": config["configurable"]["thread_id"]
    }

    # =========================================================================
    # PHASE 1: Execute until approval required
    # =========================================================================

    print("\n" + "="*70)
    print("PHASE 1: Planning")
    print("="*70)

    for event in app.stream(initial_state, config):
        # Print events as they occur
        node_name = list(event.keys())[0]
        print(f"\n[EVENT] {node_name}: {list(event[node_name].keys())}")

    # At this point, execution is paused at approval node

    # =========================================================================
    # PHASE 2: Get human approval
    # =========================================================================

    print("\n" + "="*70)
    print("PHASE 2: Awaiting Human Decision")
    print("="*70)

    if auto_approve is not None:
        # Automatic approval (for testing)
        approved = auto_approve
        print(f"\n[AUTO] {'Approved' if approved else 'Rejected'}")

    else:
        # Get human input
        approved = get_human_approval_cli(app, config)

    # Update state with approval decision
    app.update_state(
        config,
        {
            "approved": approved,
            "rejection_reason": None if approved else "User rejected"
        }
    )

    # =========================================================================
    # PHASE 3: Resume execution
    # =========================================================================

    print("\n" + "="*70)
    print("PHASE 3: Execution")
    print("="*70)

    # Resume from checkpoint (pass None as state to continue from saved state)
    for event in app.stream(None, config):
        node_name = list(event.keys())[0]
        print(f"\n[EVENT] {node_name}: Processed")

    # =========================================================================
    # Get final state
    # =========================================================================

    final_state = app.get_state(config)

    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"\nRequest: {final_state.values['input']}")
    print(f"Plan: {final_state.values['plan']}")
    print(f"Approved: {final_state.values['approved']}")
    print(f"Result: {final_state.values['execution_result']}")
    print("="*70 + "\n")

    return final_state.values


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Run approval workflow examples.

    Usage:
        # Interactive mode
        python approval_agent.py

        # Auto-approve (for testing)
        python approval_agent.py --auto-approve

        # Auto-reject (for testing)
        python approval_agent.py --auto-reject
    """

    import argparse

    parser = argparse.ArgumentParser(description="Approval Agent Demo")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve all requests"
    )
    parser.add_argument(
        "--auto-reject",
        action="store_true",
        help="Automatically reject all requests"
    )
    parser.add_argument(
        "--request",
        type=str,
        default="Send a welcome email to new team members",
        help="User request to process"
    )

    args = parser.parse_args()

    # Determine approval mode
    if args.auto_approve:
        auto_approve = True
    elif args.auto_reject:
        auto_approve = False
    else:
        auto_approve = None  # Interactive

    # Run workflow
    run_approval_workflow(args.request, auto_approve=auto_approve)
