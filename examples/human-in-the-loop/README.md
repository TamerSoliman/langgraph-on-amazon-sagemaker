# Human-in-the-Loop (HITL) Implementation

## Overview

This example demonstrates **human-in-the-loop** patterns where agents pause execution and wait for human approval or input before proceeding.

**For AI/ML Scientists:**
Think of this as adding a validation step in your pipeline - like reviewing model predictions before deployment. The agent generates outputs but humans verify/approve them before final execution.

**Why Human-in-the-Loop?**
- **Safety**: Prevent agents from taking harmful actions
- **Quality**: Human review improves output quality
- **Compliance**: Legal/regulatory requirements for human oversight
- **Learning**: Collect human feedback for fine-tuning
- **Cost control**: Approve expensive operations (API calls, database writes)

---

## Patterns Implemented

### 1. **Approval Pattern** (`approval_agent.py`)
Agent generates plan → Human approves/rejects → Agent executes

**Use case:** "Draft an email to all customers" → Human reviews → Send

### 2. **Input Pattern** (`input_agent.py`)
Agent needs information → Pauses → Human provides input → Agent continues

**Use case:** "Book a flight" → Agent asks: "Which airline?" → Human answers → Book

### 3. **Feedback Pattern** (`feedback_agent.py`)
Agent produces output → Human provides feedback → Agent revises → Repeat

**Use case:** "Write a report" → Human: "Add more data" → Agent revises → Repeat

### 4. **Multi-Step Approval** (`multi_step_agent.py`)
Complex workflow with multiple approval checkpoints

**Use case:** Purchase request → Manager approves → Finance approves → Execute

---

## Quick Start

### 1. Install Dependencies

```bash
cd examples/human-in-the-loop/
pip install -r requirements.txt
```

### 2. Run Basic Example

```bash
# Approval pattern (simplest)
python approval_agent.py
```

**Expected flow:**
```
[AGENT] Generating plan for: "Send email to team about meeting"
[AGENT] Plan: Draft email, review recipients, send

⏸️  HUMAN APPROVAL REQUIRED
Plan: Draft email about team meeting
Recipients: [team@company.com]
Approve? (yes/no): yes

[AGENT] ✓ Approved. Executing plan...
[AGENT] Email sent successfully
```

### 3. Run with SageMaker

```bash
export SAGEMAKER_ENDPOINT_NAME="your-endpoint"
python approval_agent.py
```

---

## Architecture

### State Machine with Human Checkpoints

```
User Request → Agent Plans → [WAIT] Human Approval → Agent Executes → Done
                                ↓
                              Reject → Modify → [WAIT] → Approve
```

**For AI/ML Scientists:**
This is like active learning - the model (agent) proposes labels (actions), humans verify/correct them, then the model updates based on feedback.

### LangGraph Implementation

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint import MemorySaver

# State includes human_approval field
class HITLState(TypedDict):
    input: str
    plan: str
    human_approval: Optional[bool]
    execution_result: str

# Create graph with checkpoints
graph = StateGraph(HITLState)

# Add nodes
graph.add_node("planner", plan_action)
graph.add_node("approval", wait_for_approval)  # Human checkpoint
graph.add_node("executor", execute_plan)

# Conditional routing based on approval
graph.add_conditional_edges(
    "approval",
    lambda state: "execute" if state["human_approval"] else "replan"
)

# CRITICAL: Use checkpointer to enable interrupts
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer, interrupt_before=["approval"])
```

**Key concept:** `interrupt_before=["approval"]` pauses execution before the approval node, allowing human intervention.

---

## Deployment Patterns

### Local Development (Synchronous)

**Simple prompt for approval:**

```python
def get_human_approval(plan: str) -> bool:
    print(f"Plan: {plan}")
    response = input("Approve? (yes/no): ")
    return response.lower() == "yes"
```

**Good for:** Testing, development, demos

### Production (Asynchronous)

**Three approaches:**

#### 1. **WebSocket/SSE (Server-Sent Events)**

```python
# Agent sends approval request via WebSocket
await websocket.send({
    "type": "approval_required",
    "plan": plan,
    "request_id": "abc123"
})

# Wait for human response
response = await websocket.receive()
approved = response["approved"]
```

**Good for:** Real-time web apps, chat interfaces

#### 2. **Message Queue (SQS/RabbitMQ)**

```python
# Agent publishes to approval queue
approval_queue.publish({
    "plan": plan,
    "callback_url": "/api/approval/abc123"
})

# Human reviews via web UI
# Clicks approve → POST to callback_url

# Agent polls or receives callback
approved = wait_for_callback("abc123")
```

**Good for:** Asynchronous workflows, multi-tenant systems

#### 3. **Database Polling**

```python
# Agent writes approval request to database
db.insert("approval_requests", {
    "id": "abc123",
    "plan": plan,
    "status": "pending"
})

# Poll until status changes
while True:
    status = db.get("approval_requests", "abc123")
    if status["status"] != "pending":
        break
    sleep(5)

approved = status["status"] == "approved"
```

**Good for:** Simple setups, low-traffic systems

---

## Example: Approval Pattern

### Basic Implementation

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from typing import TypedDict, Optional

class ApprovalState(TypedDict):
    input: str
    plan: str
    approved: Optional[bool]
    result: str

def planner_node(state: ApprovalState) -> ApprovalState:
    """Generate action plan"""
    plan = f"Plan: {state['input']}"
    return {**state, "plan": plan}

def approval_node(state: ApprovalState) -> ApprovalState:
    """Wait for human approval (interrupt point)"""
    # This node is where execution pauses
    # In production, this would be async
    print(f"\n⏸️  Approval required for: {state['plan']}")
    return state  # State unchanged, waiting for human input

def executor_node(state: ApprovalState) -> ApprovalState:
    """Execute approved plan"""
    if state.get("approved"):
        result = f"Executed: {state['plan']}"
    else:
        result = "Execution cancelled"
    return {**state, "result": result}

# Build graph
graph = StateGraph(ApprovalState)
graph.add_node("planner", planner_node)
graph.add_node("approval", approval_node)
graph.add_node("executor", executor_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "approval")

# Conditional edge based on approval
graph.add_conditional_edges(
    "approval",
    lambda s: "execute" if s.get("approved") else END,
    {"execute": "executor", END: END}
)

graph.add_edge("executor", END)

# Compile with checkpoint and interrupt
memory = MemorySaver()
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["approval"]  # Pause before approval
)

# Execute
config = {"configurable": {"thread_id": "1"}}
state = {"input": "Send email to team", "plan": "", "approved": None, "result": ""}

# First run - will stop at approval
for event in app.stream(state, config):
    print(event)

# At this point, execution is paused
# Human reviews and provides approval

# Resume with approval
app.update_state(config, {"approved": True})

# Continue execution
for event in app.stream(None, config):
    print(event)
```

### For AI/ML Scientists

**What's happening:**

1. **First stream()**: Runs until `interrupt_before=["approval"]` is hit
2. **Checkpoint saved**: State is persisted (plan, context, etc.)
3. **Human reviews**: External process (web UI, CLI, etc.) gets the state
4. **update_state()**: Human provides approval decision
5. **Second stream()**: Resumes from checkpoint with new state

This is like **checkpointing in distributed training** - save state, pause, resume later.

---

## Advanced Patterns

### 1. Timeout for Approval

```python
import asyncio
from datetime import datetime, timedelta

async def approval_with_timeout(state, timeout_seconds=300):
    """Wait for approval with timeout"""

    deadline = datetime.now() + timedelta(seconds=timeout_seconds)

    while datetime.now() < deadline:
        # Check if approval received
        approval_status = check_approval_status(state["request_id"])

        if approval_status is not None:
            return approval_status

        await asyncio.sleep(5)  # Poll every 5 seconds

    # Timeout - default to rejection
    return False
```

### 2. Multi-Level Approval

```python
def requires_approval(state) -> str:
    """Route based on approval level needed"""

    amount = state["transaction_amount"]

    if amount < 1000:
        return "auto_approve"
    elif amount < 10000:
        return "manager_approval"
    else:
        return "director_approval"

graph.add_conditional_edges(
    "analyze",
    requires_approval,
    {
        "auto_approve": "execute",
        "manager_approval": "manager_review",
        "director_approval": "director_review"
    }
)
```

### 3. Approval with Modifications

```python
def approval_node(state):
    """Allow human to modify plan, not just approve/reject"""

    print(f"Current plan: {state['plan']}")
    print("Options: (a)pprove, (r)eject, (m)odify")

    choice = input("Choice: ")

    if choice == 'a':
        return {**state, "approved": True}
    elif choice == 'r':
        return {**state, "approved": False}
    elif choice == 'm':
        new_plan = input("Enter modified plan: ")
        return {**state, "plan": new_plan, "approved": True}
```

---

## Production Considerations

### 1. Security

```python
# ✅ GOOD: Validate approval source
def validate_approval(approval_request, user_token):
    # Verify user is authorized to approve
    user = auth.verify_token(user_token)

    if not user.has_permission("approve_actions"):
        raise PermissionError("User not authorized")

    # Verify request hasn't been tampered with
    if approval_request.signature != compute_signature(approval_request):
        raise SecurityError("Invalid signature")

    return True
```

### 2. Audit Logging

```python
def log_approval_decision(state, approved, approver):
    """Log all approval decisions for compliance"""

    audit_log.write({
        "timestamp": datetime.now(),
        "request_id": state["request_id"],
        "plan": state["plan"],
        "approved": approved,
        "approver": approver,
        "ip_address": get_client_ip()
    })
```

### 3. State Persistence

```python
# Use PostgreSQL instead of MemorySaver for production
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(
    connection_string="postgresql://user:pass@host/db"
)

app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval"]
)

# State persists across server restarts
```

---

## Cost Analysis

### Synchronous (Blocking)

**Cost:** High - agent holds resources while waiting

**Example:** Lambda function waiting for approval
- Lambda execution time: 5 minutes (waiting)
- Cost: $0.00001667 * 5 * 60 = $0.005 per approval
- For 1000 approvals/day = $5/day = $150/month **just waiting**

### Asynchronous (Non-Blocking)

**Cost:** Low - agent pauses, releases resources

**Example:** Lambda + SQS + DynamoDB
- Lambda (plan): $0.000001 (0.1s)
- SQS (queue approval): $0.0000004
- DynamoDB (store state): $0.000001
- Lambda (execute): $0.000001 (0.1s)
- **Total: $0.0000034 per approval**
- For 1000 approvals/day = $0.10/month

**Savings: 99.93%** 🎉

**For AI/ML Scientists:**
This is like the difference between:
- **Synchronous**: Keeping GPU running while waiting for data (wasteful)
- **Asynchronous**: Release GPU, reload when data ready (efficient)

---

## Testing

### Unit Tests

```python
def test_approval_flow():
    """Test complete approval workflow"""

    app = create_approval_agent()
    config = {"configurable": {"thread_id": "test"}}

    state = {"input": "test task", "approved": None}

    # Run until interrupt
    events = list(app.stream(state, config))
    assert len(events) > 0

    # Get current state
    current = app.get_state(config)
    assert current.values["plan"] is not None

    # Approve
    app.update_state(config, {"approved": True})

    # Resume
    final_events = list(app.stream(None, config))
    final_state = app.get_state(config)

    assert final_state.values["result"] is not None
```

### Integration Tests

```python
def test_approval_timeout():
    """Test that approvals timeout correctly"""

    app = create_approval_agent(timeout=5)

    # Start workflow
    state = {"input": "test"}
    app.stream(state, config)

    # Don't provide approval
    time.sleep(6)

    # Should auto-reject after timeout
    final_state = app.get_state(config)
    assert final_state.values["approved"] == False
```

---

## Related Examples

- `../multi-agent/` - See how multiple agents can request approvals
- `../custom-tools/` - Tools that require approval before execution
- `../../deployment/cdk/` - Deploy HITL agents to Lambda

---

## Next Steps

1. **Run examples** to understand each pattern
2. **Choose pattern** based on your use case (sync vs async)
3. **Implement persistence** (PostgreSQL for production)
4. **Add auth** for approval endpoints
5. **Monitor** approval rates and response times

For questions, see the main repository README.
