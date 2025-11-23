# Phase 1: Component Discovery & Identification

## Repository Overview

This repository contains a **single comprehensive Jupyter notebook** (`langgraph_sagemaker.ipynb`) that demonstrates how to integrate LangGraph with Amazon SageMaker LLM endpoints. The architecture follows a **decoupled pattern** where the LangGraph agent orchestration runs separately from the LLM inference endpoint.

---

## 1. Core Agent Scripts

### **LangGraph Graph Definition**
**Location:** `langgraph_sagemaker.ipynb` - Cell ID: `46d2fbde-72fe-498b-9ab9-474122a66104`

```python
from langgraph.prebuilt import create_agent_executor

tools = [TavilySearchResults(max_results=1)]
prompt = hub.pull("hwchase17/xml-agent-convo")
agent_runnable = create_xml_agent(llm, tools, prompt)
app = create_agent_executor(agent_runnable, tools)
```

**Key Characteristics:**
- Uses the **prebuilt** LangGraph agent executor (high-level abstraction)
- Wraps an XML-based agent created with `create_xml_agent`
- The graph is **stateful** - maintains conversation history and intermediate steps
- Implements the classic ReAct (Reasoning + Acting) pattern

---

### **Tool Definitions**
**Location:** `langgraph_sagemaker.ipynb` - Cell ID: `46d2fbde-72fe-498b-9ab9-474122a66104`

```python
tools = [TavilySearchResults(max_results=1)]
```

**Tool Details:**
- **TavilySearchResults**: Web search capability
  - External service (requires API key)
  - Limited to 1 result per query for efficiency
  - Integrated via LangChain's community tools
  - Returns JSON-formatted search results

**Extension Points:** The tools list is modular - additional tools can be added by appending to this list (e.g., calculators, database queries, API calls).

---

## 2. Agent-LLM Interface

### **LLM Client Implementation**
**Location:** `langgraph_sagemaker.ipynb` - Cell ID: `0625e0a9-cefc-4ec1-921c-5f96cf004ed0`

```python
from langchain_community.llms import SagemakerEndpoint

llm = SagemakerEndpoint(
    endpoint_name=endpoint_name,
    region_name="us-east-1",
    model_kwargs={"max_new_tokens": 500, "do_sample": True, "temperature": 0.001},
    content_handler=content_handler
)
```

**Key Configuration:**
- **endpoint_name**: Points to the SageMaker real-time inference endpoint
- **max_new_tokens: 500**: Critical parameter - must be high enough to allow complete XML responses (tool calls + reasoning)
- **temperature: 0.001**: Near-deterministic for reliable tool calling
- **content_handler**: Custom transformation logic (see below)

---

### **Structured Output / Function Calling Handler**
**Location:** `langgraph_sagemaker.ipynb` - Cell ID: `0625e0a9-cefc-4ec1-921c-5f96cf004ed0`

```python
class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        decoded_output = output.read().decode("utf-8")
        response_json = json.loads(decoded_output)
        response = response_json[0]["generated_text"]
        return response
```

**How Structured Output Works:**

1. **Prompt Format (XML-based)**: The agent uses XML tags for structured communication
   - `<tool>tool_name</tool>` - Specifies which tool to call
   - `<tool_input>query</tool_input>` - Provides tool arguments
   - `<final_answer>response</final_answer>` - Wraps final user-facing output

2. **Payload Transformation**:
   - **Input**: Converts LangChain prompt string → SageMaker JSON payload
   - **Output**: Extracts generated text from SageMaker response array

3. **Parsing Flow**:
   - LLM generates XML-tagged response
   - LangChain's XML agent parser extracts tool calls or final answers
   - LangGraph router uses parsed output to determine next node

---

## 3. Deployment Scripts

### **LLM Endpoint Provisioning**
**Location:** `langgraph_sagemaker.ipynb` - Cell IDs: `2e32a323-d521-4ef9-9ddb-73e0657d08ab` and `85d4f212-e99d-4dd9-b68e-439357c18e10`

**Deployment Method:** **SageMaker JumpStart (UI-based)**

The notebook **does not contain Python code to deploy the LLM endpoint**. Instead, it documents the manual steps:

```
1. Access SageMaker Studio through AWS Console
2. Navigate to "Studio Classic"
3. Open JumpStart
4. Search for "Mistral 7B Instruct"
5. Click "Deploy"
```

**Endpoint Reference in Code:**
```python
endpoint_name = input("SageMaker Endpoint Name:")
```

The endpoint is treated as a **pre-existing resource** that the notebook connects to.

---

### **Why This Separation Matters**

The notebook demonstrates a critical architectural pattern:

| Component | Location | Lifecycle | Cost Profile |
|-----------|----------|-----------|--------------|
| **LLM Endpoint** | SageMaker (GPU instance) | Long-running | High (ml.g5.xlarge+) |
| **LangGraph Agent** | Notebook/Client | On-demand | Low (CPU-only) |

**Key Insight:** By deploying the LLM endpoint separately, you can:
- Run the expensive GPU instance 24/7 for low-latency inference
- Run the agent logic on cheap compute (Lambda, EC2, ECS) that scales to zero
- Share one LLM endpoint across multiple agent applications
- Update agent logic without redeploying the model

---

## 4. Graph Execution Flow

**Location:** `langgraph_sagemaker.ipynb` - Cell ID: `67bc38b0-5896-4755-9f82-d02293b2d0b5`

```python
def ask_langgraph_question(question):
    inputs = {"input": question, "chat_history": []}

    for s in app.stream(inputs):
        agentValues = list(s.values())[0]
        if 'agent_outcome' in agentValues and isinstance(agentValues['agent_outcome'], AgentFinish):
            agentFinish = agentValues['agent_outcome']

    return agentFinish.return_values["output"]
```

**Execution Pattern:**
1. **Input**: User question + empty chat history
2. **Stream**: LangGraph yields state updates for each node execution
3. **Loop**: Continues until `AgentFinish` is detected
4. **Output**: Final answer extracted from terminal state

**Observed Execution Steps** (from notebook output):
```
Step 1: Agent decides to call tavily_search_results_json
Step 2: Tool executes, returns search results
Step 3: Agent processes results, generates final answer
Step 4: Complete state returned
```

---

## Summary: Code Location Reference

| Component | Cell ID | Line Reference |
|-----------|---------|----------------|
| **LangGraph Graph Definition** | `46d2fbde-72fe-498b-9ab9-474122a66104` | `app = create_agent_executor(...)` |
| **Tool Definitions** | `46d2fbde-72fe-498b-9ab9-474122a66104` | `tools = [TavilySearchResults(...)]` |
| **LLM Client** | `0625e0a9-cefc-4ec1-921c-5f96cf004ed0` | `llm = SagemakerEndpoint(...)` |
| **Structured Output Handler** | `0625e0a9-cefc-4ec1-921c-5f96cf004ed0` | `class ContentHandler(...)` |
| **SageMaker Endpoint Reference** | `85d4f212-e99d-4dd9-b68e-439357c18e10` | `endpoint_name = input(...)` |
| **Graph Execution** | `67bc38b0-5896-4755-9f82-d02293b2d0b5` | `app.stream(inputs)` |

---

**Next Steps:** Phase 2 will provide deeply annotated versions of the core components with line-by-line explanations.
