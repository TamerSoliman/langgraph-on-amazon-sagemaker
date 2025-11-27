# LangGraph Component Reference Table

## Overview

This document maps LangGraph conceptual components to their implementation in this repository and the underlying AWS infrastructure.

---

## Core LangGraph Components

### 1. State

**LangGraph Concept:**
- Dictionary containing all information needed to execute the graph
- Passed between nodes
- Accumulated over execution (history of actions)

**Implementation in This Repo:**

| Aspect | Details |
|--------|---------|
| **Data Structure** | Python dictionary |
| **Schema** | Not explicitly defined (uses default from `create_agent_executor`) |
| **Keys** | `input`, `chat_history`, `agent_outcome`, `intermediate_steps` |
| **Storage** | In-memory (RAM) during execution |
| **Persistence** | ❌ None (state lost after execution completes) |
| **Location** | Agent Host (Lambda/ECS/EC2) |

**Default State Schema:**
```python
{
    "input": str,                      # Original user question
    "chat_history": List[BaseMessage], # Previous conversation turns
    "agent_outcome": Union[AgentAction, AgentFinish, None],
    "intermediate_steps": List[Tuple[AgentAction, str]]
}
```

**AWS Deployment Mapping:**

| State Aspect | Development (Notebook) | Production (Lambda) | Production (ECS) | Production with Persistence |
|--------------|------------------------|---------------------|------------------|------------------------------|
| **Storage** | Jupyter kernel memory | Lambda execution context | Container memory | DynamoDB or S3 |
| **Lifetime** | Until kernel restart | Until Lambda timeout (15 min) | Until container stops | Permanent |
| **Size Limit** | RAM limit (~GB) | 10GB | Task memory limit | DynamoDB: 400KB/item, S3: unlimited |
| **Cost** | Free (local) | Free (included in Lambda) | Free (included in task) | DynamoDB: $0.25/GB-month, S3: $0.023/GB-month |

**How to Add Persistence:**
```python
from langgraph.checkpoint.dynamodb import DynamoDBSaver

checkpointer = DynamoDBSaver(
    table_name="langgraph-checkpoints",
    region_name="us-east-1"
)

app = create_agent_executor(agent_runnable, tools, checkpointer=checkpointer)

# Now state is saved to DynamoDB after each step
# Can resume: app.stream(None, config={"thread_id": "conversation-123"})
```

---

### 2. Nodes

**LangGraph Concept:**
- Functions that process state
- Take state as input, return updated state
- Represent discrete steps in the workflow

**Implementation in This Repo:**

| Node Name | Function | Input | Output | Execution Location |
|-----------|----------|-------|--------|---------------------|
| **Agent Node** | Calls LLM, parses response | `state` dict | Updated state with `agent_outcome` | Agent Host (CPU) |
| **Tools Node** | Executes tools (Tavily search, etc.) | `state` dict with `agent_outcome` | Updated state with tool results in `intermediate_steps` | Agent Host (CPU) |
| **START** | No-op (entry point) | Initial inputs | Passed through | Agent Host |
| **END** | No-op (exit point) | Final state | Returned to caller | Agent Host |

**Agent Node Details:**
```python
# What happens inside the Agent Node (conceptual)
def agent_node(state):
    # 1. Format prompt with state
    prompt = format_prompt(
        input=state["input"],
        chat_history=state["chat_history"],
        intermediate_steps=state["intermediate_steps"]
    )

    # 2. Call LLM endpoint
    llm_response = llm.invoke(prompt)  # → SageMaker API call

    # 3. Parse response
    if "<tool>" in llm_response:
        outcome = AgentAction(tool="...", tool_input="...")
    elif "<final_answer>" in llm_response:
        outcome = AgentFinish(return_values={"output": "..."})

    # 4. Update state
    state["agent_outcome"] = outcome
    return state
```

**Tools Node Details:**
```python
# What happens inside the Tools Node (conceptual)
def tools_node(state):
    action = state["agent_outcome"]  # AgentAction object

    # 1. Look up tool by name
    tool = tools_dict[action.tool]  # e.g., TavilySearchResults

    # 2. Execute tool
    result = tool.run(action.tool_input)  # → External API call (Tavily)

    # 3. Update state
    state["intermediate_steps"].append((action, result))
    return state
```

**AWS Deployment Mapping:**

| Node | CPU Time | Network Calls | Cost (Lambda) | Cost (ECS Fargate) |
|------|----------|---------------|---------------|---------------------|
| **Agent Node** | ~100ms (formatting, parsing) | 1 LLM call (2-5 sec) | ~$0.0000002 | Included in task cost |
| **Tools Node** | ~50ms (tool lookup, logging) | 1 Tavily call (~500ms) | ~$0.0000001 | Included in task cost |

**Key Insight:** Nodes run on agent host (cheap CPU), but make network calls to expensive services (SageMaker LLM, Tavily API).

---

### 3. Edges

**LangGraph Concept:**
- Connections between nodes
- Define execution flow
- Can be conditional (route based on state)

**Implementation in This Repo:**

| Edge Type | From Node | To Node | Condition | Implementation |
|-----------|-----------|---------|-----------|----------------|
| **Unconditional** | START | Agent | Always | Default first step |
| **Conditional** | Agent | Tools | `agent_outcome` is `AgentAction` | `should_continue()` function |
| **Conditional** | Agent | END | `agent_outcome` is `AgentFinish` | `should_continue()` function |
| **Unconditional** | Tools | Agent | Always (loop back) | Continue reasoning after tool execution |

**Conditional Edge Logic:**
```python
# Inside create_agent_executor (conceptual)
def should_continue(state):
    agent_outcome = state.get("agent_outcome")

    if isinstance(agent_outcome, AgentFinish):
        return "end"  # Route to END node
    else:
        return "continue"  # Route to Tools node

# Graph construction:
# graph.add_conditional_edges(
#     "agent",
#     should_continue,
#     {
#         "continue": "tools",
#         "end": END
#     }
# )
```

**AWS Deployment Mapping:**

Edges are pure logic (no infrastructure cost). They run in the agent host's Python process.

| Edge Evaluation | CPU Time | Cost |
|-----------------|----------|------|
| Check `agent_outcome` type | ~1μs | Negligible |
| Decide next node | ~1μs | Negligible |

---

### 4. Graph

**LangGraph Concept:**
- The complete state machine
- Composed of nodes + edges + state schema
- Orchestrates execution flow

**Implementation in This Repo:**

| Aspect | Details |
|--------|---------|
| **Construction** | `create_agent_executor(agent_runnable, tools)` |
| **Type** | `CompiledStateGraph` (from LangGraph) |
| **Execution Method** | `app.stream(inputs)` - yields state after each node |
| **Graph Structure** | See diagram below |

**Graph Structure Visualization:**
```
START
  │
  ▼
┌────────┐
│ AGENT  │◀─────────┐
│  NODE  │          │
└───┬────┘          │
    │               │
    │ Decision      │
    │               │
   ┌┴┐              │
   │?│              │
   └┬┘              │
    │               │
    ├─AgentAction──▶│
    │               │
    │           ┌───┴────┐
    │           │ TOOLS  │
    │           │  NODE  │
    │           └────────┘
    │
    └─AgentFinish──▶END
```

**AWS Deployment Mapping:**

| Graph Aspect | Notebook | Lambda | ECS Fargate | Cost Implication |
|--------------|----------|--------|-------------|------------------|
| **Execution** | Single Python process | Lambda function | ECS task | Pay per execution |
| **State** | In-memory | In-memory (ephemeral) | In-memory (ephemeral) | Free |
| **Persistence** | None | Add DynamoDB checkpointer | Add DynamoDB checkpointer | $0.25/GB-month |
| **Concurrency** | 1 (notebook cell) | 1000s (auto-scale) | Configurable (1-100s) | Lambda free tier: 1M requests/month |

---

### 5. Tools

**LangGraph Concept:**
- External functions the agent can invoke
- Extend agent capabilities beyond LLM knowledge
- Defined using LangChain Tool interface

**Implementation in This Repo:**

| Tool | Purpose | API Provider | Cost | Rate Limit |
|------|---------|--------------|------|------------|
| **TavilySearchResults** | Web search | Tavily.com | Free tier: 1000 searches/month<br>Paid: $0.001/search | 60 requests/minute |

**Tool Configuration:**
```python
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [
    TavilySearchResults(
        max_results=1,  # Limit to 1 result to save tokens
        api_wrapper=TavilySearchAPIWrapper(
            tavily_api_key=os.environ["TAVILY_API_KEY"]
        )
    )
]
```

**Tool Interface (LangChain Standard):**
```python
class Tool:
    name: str              # "tavily_search_results_json"
    description: str       # "A search engine. Useful for when you need to answer questions about current events."
    func: Callable         # The function to execute

    def run(self, tool_input: str) -> str:
        # Execute and return result
        pass
```

**AWS Deployment Mapping:**

| Tool | Execution Location | Network Call | Timeout | Error Handling |
|------|--------------------|--------------|---------|----------------|
| **TavilySearch** | Agent Host (Lambda/ECS) | HTTPS to api.tavily.com | 30 seconds | Retry 3x with exponential backoff |

**Extensibility - Adding Custom Tools:**
```python
from langchain.tools import Tool

def query_database(query: str) -> str:
    """Query company database"""
    import boto3
    rds = boto3.client('rds-data')
    # Execute SQL query...
    return results

database_tool = Tool(
    name="company_database",
    description="Search company's internal database. Input should be a SQL query.",
    func=query_database
)

tools = [
    TavilySearchResults(max_results=1),
    database_tool  # Add custom tool
]
```

---

### 6. Checkpointer (State Persistence)

**LangGraph Concept:**
- Saves state to external storage after each step
- Enables pause/resume, human-in-the-loop, debugging

**Implementation in This Repo:**

| Aspect | Current Implementation | Production Recommendation |
|--------|------------------------|---------------------------|
| **Checkpointer** | ❌ None (in-memory only) | ✅ DynamoDB or S3 |
| **State Persistence** | ❌ Lost after execution | ✅ Survives restarts |
| **Human-in-the-Loop** | ❌ Not possible | ✅ Possible (pause before tool execution) |
| **Resume** | ❌ Must restart from beginning | ✅ Resume from last step |

**How to Add (DynamoDB Example):**
```python
from langgraph.checkpoint.dynamodb import DynamoDBSaver
import boto3

# Create DynamoDB table (one-time setup)
dynamodb = boto3.client('dynamodb')
dynamodb.create_table(
    TableName='langgraph-checkpoints',
    KeySchema=[
        {'AttributeName': 'thread_id', 'KeyType': 'HASH'},
        {'AttributeName': 'checkpoint_id', 'KeyType': 'RANGE'}
    ],
    AttributeDefinitions=[
        {'AttributeName': 'thread_id', 'AttributeType': 'S'},
        {'AttributeName': 'checkpoint_id', 'AttributeType': 'S'}
    ],
    BillingMode='PAY_PER_REQUEST'
)

# Use in graph creation
checkpointer = DynamoDBSaver(
    table_name='langgraph-checkpoints',
    region_name='us-east-1'
)

app = create_agent_executor(agent_runnable, tools, checkpointer=checkpointer)

# Execute with thread_id (conversation identifier)
config = {"configurable": {"thread_id": "user-123-conv-456"}}
for state in app.stream(inputs, config):
    # State automatically saved to DynamoDB after each step
    pass

# Resume later (e.g., after human approval)
for state in app.stream(None, config):  # None = resume from last checkpoint
    pass
```

**AWS Deployment Mapping:**

| Storage Backend | Write Latency | Read Latency | Cost (1M checkpoints) | Use Case |
|-----------------|---------------|--------------|------------------------|----------|
| **DynamoDB** | 5-20ms | 5-20ms | ~$250/month | Production (low latency) |
| **S3** | 50-200ms | 50-200ms | ~$23/month | Archival, debugging |
| **In-Memory** | <1ms | <1ms | $0 | Development only |

---

### 7. Prompt

**LangGraph Concept:**
- Template that instructs the LLM
- Includes placeholders for dynamic content
- Critical for agent behavior

**Implementation in This Repo:**

| Aspect | Details |
|--------|---------|
| **Source** | LangChain Hub (`hub.pull("hwchase17/xml-agent-convo")`) |
| **Format** | XML-based (uses `<tool>`, `<final_answer>` tags) |
| **Placeholders** | `{tools}`, `{input}`, `{agent_scratchpad}`, `{chat_history}` |
| **Storage** | Downloaded at runtime, cached in memory |

**Prompt Content (Simplified):**
```
You are a helpful assistant. Help the user answer any questions.

You have access to the following tools:
{tools}

In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags.
You will then get back a response in the form <observation></observation>

For example:
<tool>search</tool><tool_input>weather in SF</tool_input>
<observation>64 degrees</observation>

When you are done, respond with a final answer between <final_answer></final_answer>.

Question: {input}
{agent_scratchpad}
```

**AWS Deployment Mapping:**

Prompts are strings stored in the agent host's memory. No infrastructure cost.

**Custom Prompt Example:**
```python
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["tools", "input", "agent_scratchpad", "chat_history"],
    template="""
    You are a helpful assistant for [YOUR COMPANY].

    Available tools:
    {tools}

    Use XML tags for tool calls: <tool>name</tool><tool_input>input</tool_input>
    Use <final_answer>answer</final_answer> when done.

    Previous conversation:
    {chat_history}

    Question: {input}
    {agent_scratchpad}
    """
)

agent_runnable = create_xml_agent(llm, tools, custom_prompt)
```

---

## Infrastructure Mapping Summary

### Component Ownership Table

| LangGraph Component | Runs On | Deployed To | Requires GPU? | Billable Resource |
|---------------------|---------|-------------|---------------|-------------------|
| **State** | Agent Host | Lambda/ECS/EC2 | ❌ No | RAM (free or included in compute) |
| **Nodes** | Agent Host | Lambda/ECS/EC2 | ❌ No | CPU time (Lambda: $0.0000166667/GB-sec) |
| **Edges** | Agent Host | Lambda/ECS/EC2 | ❌ No | Negligible CPU |
| **Graph** | Agent Host | Lambda/ECS/EC2 | ❌ No | Execution time (Lambda: 15 min max) |
| **Tools** | Agent Host | Lambda/ECS/EC2 | ❌ No | CPU + external API costs (Tavily) |
| **Checkpointer** | AWS Service | DynamoDB/S3 | ❌ No | Storage ($0.25/GB-month DynamoDB) |
| **Prompt** | Agent Host | Lambda/ECS/EC2 (string in memory) | ❌ No | RAM (negligible) |
| **LLM** | SageMaker | SageMaker Endpoint | ✅ Yes | ml.g5.xlarge ($1.006/hour) |

---

### Data Flow Through Components

```
User Question
     │
     ▼
 ┌────────────────────────────────────────────────────────────┐
 │  STATE (initialized)                                        │
 │  {                                                          │
 │    "input": "What is the latest UK storm?",                │
 │    "chat_history": [],                                     │
 │    "intermediate_steps": [],                               │
 │    "agent_outcome": None                                   │
 │  }                                                          │
 └──────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────────┐
 │  NODE: Agent                                                │
 │  • Format PROMPT with state                                │
 │  • Call LLM (SageMaker endpoint)                           │
 │  • Parse response (XML → AgentAction)                      │
 └──────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────────┐
 │  STATE (updated)                                            │
 │  {                                                          │
 │    "input": "What is the latest UK storm?",                │
 │    "agent_outcome": AgentAction(tool="tavily_search",...), │
 │    ...                                                      │
 │  }                                                          │
 └──────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────────┐
 │  EDGE: Conditional                                          │
 │  • Check: agent_outcome is AgentAction?                    │
 │  • Decision: Route to Tools Node                           │
 └──────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────────┐
 │  NODE: Tools                                                │
 │  • Look up TOOL by name (tavily_search)                    │
 │  • Execute tool (API call to Tavily)                       │
 │  • Store result                                            │
 └──────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────────┐
 │  STATE (updated with tool result)                          │
 │  {                                                          │
 │    "intermediate_steps": [                                 │
 │      (AgentAction(...), "[{url:..., content:...}]")        │
 │    ],                                                       │
 │    ...                                                      │
 │  }                                                          │
 └──────────────────────┬─────────────────────────────────────┘
                        │
                        │ (Loop back to Agent Node)
                        ▼
 ┌────────────────────────────────────────────────────────────┐
 │  NODE: Agent (second call)                                 │
 │  • PROMPT now includes tool result in {agent_scratchpad}   │
 │  • LLM sees observation, generates final answer            │
 │  • Parse response (XML → AgentFinish)                      │
 └──────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────────┐
 │  STATE (final)                                              │
 │  {                                                          │
 │    "agent_outcome": AgentFinish(                           │
 │      return_values={"output": "Storm Henk..."}             │
 │    )                                                        │
 │  }                                                          │
 └──────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────────┐
 │  EDGE: Conditional                                          │
 │  • Check: agent_outcome is AgentFinish?                    │
 │  • Decision: Route to END                                  │
 └──────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────────┐
 │  END                                                        │
 │  • Return final state to caller                            │
 │  • Extract: state["agent_outcome"].return_values["output"] │
 └────────────────────────────────────────────────────────────┘
                        │
                        ▼
                User receives answer
```

---

## Cost Breakdown by Component

### Example: 1,000 Questions/Day

| Component | Resource | Usage | Cost/Month |
|-----------|----------|-------|------------|
| **State** | Lambda memory | 1GB × 5sec × 1000 questions/day | Included in execution |
| **Nodes (Agent)** | Lambda execution | 1GB × 3sec × 2 calls/question × 1000/day | ~$0.30 |
| **Nodes (Tools)** | Lambda execution | 1GB × 1sec × 1 call/question × 1000/day | ~$0.05 |
| **Edges** | Lambda execution | Negligible | ~$0.00 |
| **Graph** | Lambda execution | Included above | ~$0.00 |
| **Tools (Tavily)** | External API | 1000 searches/day × $0.001 | $30.00 |
| **Checkpointer (DynamoDB)** | Storage | 30,000 items × 10KB = 300MB | ~$0.08 |
| **Prompt** | Memory | Negligible | ~$0.00 |
| **LLM (SageMaker)** | ml.g5.xlarge 24/7 | $1.006/hour × 720 hours | ~$724.32 |
| **Total** | | | **~$754.75/month** |

**Key Insight:** LLM endpoint is 96% of the cost. All LangGraph components combined are only 4%.

---

## Reference: LangGraph vs. AWS Service Mapping

| LangGraph Abstraction | AWS Service (if externalized) | Notebook (local) |
|-----------------------|-------------------------------|------------------|
| **State** | DynamoDB, S3 | Python dict in memory |
| **Node execution** | Lambda function, ECS task | Function call |
| **Edge routing** | Step Functions (alternative) | If/else statement |
| **Graph orchestration** | Step Functions (alternative) | For loop with logic |
| **Tool execution** | Lambda (separate function) | API call from notebook |
| **LLM calls** | SageMaker endpoint | SageMaker endpoint |
| **Checkpointing** | DynamoDB, S3 | Not persistent |

**Note:** This repo uses LangGraph to handle state/nodes/edges/graph in-process (Lambda/ECS). An alternative would be AWS Step Functions for orchestration, but LangGraph provides more flexibility and is easier to develop/test.

---

## Quick Reference: Component Interaction Matrix

|             | State | Node | Edge | Graph | Tool | Checkpointer | Prompt | LLM |
|-------------|-------|------|------|-------|------|--------------|--------|-----|
| **State**   | -     | Passed to | Read by | Managed by | Passed to | Saved by | Injected into | - |
| **Node**    | Modifies | Calls | Returns to | Registered in | Calls | Triggers | Uses | Calls |
| **Edge**    | Reads | Connects | Conditional | Defined in | - | - | - | - |
| **Graph**   | Owns | Executes | Evaluates | Compiles | Provides | Uses | - | - |
| **Tool**    | Reads from | Called by | - | Registered in | - | - | Listed in | - |
| **Checkpointer** | Persists | - | - | Used by | - | - | - | - |
| **Prompt**  | Filled from | Used by | - | - | Includes | - | - | Sent to |
| **LLM**     | - | Called by | - | - | - | - | Receives | Returns to |

---

## Summary

This reference table demonstrates that:

1. **LangGraph components are lightweight:** State, nodes, edges, and graphs are just Python objects/functions running on CPU
2. **Infrastructure is decoupled:** LLM runs on expensive GPU (SageMaker), everything else on cheap CPU (Lambda/ECS)
3. **Persistence is optional but valuable:** Add checkpointers (DynamoDB) for production robustness
4. **Tools extend capabilities:** Easy to add custom tools without changing core architecture
5. **Cost is dominated by LLM:** Optimizing LangGraph component efficiency has minimal cost impact compared to LLM endpoint usage

**For developers:** Focus on prompt engineering, tool selection, and LLM configuration. LangGraph infrastructure is cheap and scalable by default.
