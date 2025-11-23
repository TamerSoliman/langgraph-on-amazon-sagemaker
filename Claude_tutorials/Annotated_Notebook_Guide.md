# Comprehensive Notebook Annotation Guide

## Overview

This repository contains **one primary notebook**: `langgraph_sagemaker.ipynb`. This guide breaks down the **10 most critical sections** of the notebook, providing deep annotations for each.

---

## Section 1: Dependency Installation
**Cell ID:** `fbf8619f-3cf1-4e4b-a3a8-9649f04bc70a`

### Code:
```python
%pip install --upgrade pip --root-user-action=ignore --quiet
%pip install sagemaker boto3 huggingface_hub --upgrade --root-user-action=ignore --quiet
%pip install -U langchain langgraph langchainhub --root-user-action=ignore --quiet
```

### Annotation:

**WHAT:** Install and upgrade all required Python packages

**HOW:**
- `%pip`: Jupyter magic command (alternative to `!pip`)
- `--upgrade`: Forces latest versions
- `--root-user-action=ignore`: Suppresses warnings when running as root (common in SageMaker notebooks)
- `--quiet`: Minimizes installation output

**WHY:**
1. **pip upgrade:** Ensures compatibility with newer package index features
2. **SageMaker SDK group** (`sagemaker`, `boto3`, `huggingface_hub`):
   - `sagemaker`: High-level SDK for endpoint management, training, etc.
   - `boto3`: Low-level AWS SDK for direct API calls (used for `invoke_endpoint`)
   - `huggingface_hub`: Utilities for HuggingFace models (optional here)
3. **LangChain group** (`langchain`, `langgraph`, `langchainhub`):
   - `langchain`: Core orchestration framework
   - `langgraph`: Stateful graph execution engine
   - `langchainhub`: Shared prompt repository

**CRITICAL NOTE:** The notebook warns "you may need to restart the kernel to use updated packages." This is because:
- Python loads packages into memory at import time
- Upgrading a package after import doesn't reload it
- Kernel restart ensures fresh imports

**PRODUCTION CONSIDERATION:**
Use a `requirements.txt` with pinned versions instead:
```
sagemaker==2.198.0
boto3==1.34.10
langchain==0.1.0
langgraph==0.0.20
```
This ensures reproducibility across environments.

---

## Section 2: Test Questions Definition
**Cell ID:** `d58b0f9e-0c05-4c65-9592-925d8c60eb7d`

### Code:
```python
questions_and_answers = [
    {
        "question": "What is the recipe of mayonnaise?",
        "answers": {}
    },
    {
        "question": "What is the name of the latest storm to hit the UK, and when and where did it cause the most damage?",
        "answers": {}
    }
]
```

### Annotation:

**WHAT:** Define test questions to compare LLM-only vs. LangGraph approaches

**HOW:**
- List of dictionaries with consistent schema
- `"answers"` key is an empty dict that gets populated with results from different approaches

**WHY THESE QUESTIONS:**

1. **"Recipe of mayonnaise"** (Knowledge-based):
   - **LLM-only:** Can answer from training data (common recipe)
   - **LangGraph:** May search web for additional/updated recipes
   - **Purpose:** Show LangGraph provides references/links vs. just reciting knowledge

2. **"Latest storm to hit the UK"** (Time-sensitive):
   - **LLM-only:** Will give outdated answer (training data cutoff)
   - **LangGraph:** Searches web for current information
   - **Purpose:** Demonstrate the critical value of real-time tool access

**DATA STRUCTURE EVOLUTION:**
After running all approaches, the structure becomes:
```python
{
    "question": "What is the recipe of mayonnaise?",
    "answers": {
        "llm": "1. Gather all ingredients...",           # From direct LLM call
        "langchain": "Ingredients: 2 large egg yolks...", # From LangChain wrapper
        "langgraph": "Here is a recipe: <a href=...>"    # From agent with tools
    }
}
```

**EXTENSION IDEA:**
Add questions that test multi-step reasoning:
```python
{
    "question": "What is the population of the capital city of the country that won the 2022 World Cup?",
    # Requires: 1) Search for 2022 World Cup winner (Argentina)
    #          2) Identify capital (Buenos Aires)
    #          3) Search population
}
```

---

## Section 3: Endpoint Name Input
**Cell ID:** `85d4f212-e99d-4dd9-b68e-439357c18e10`

### Code:
```python
import json
import boto3

endpoint_name = input("SageMaker Endpoint Name:")
```

### Annotation:

**WHAT:** Prompt user for the SageMaker endpoint name (manual input)

**HOW:**
- `input()` function blocks execution until user enters text
- No validation - assumes user provides correct endpoint name

**WHY:**
- Endpoint deployment is separate from notebook execution
- Endpoint name is dynamic (user-created via JumpStart)
- Makes notebook reusable across different endpoints

**SECURITY CONSIDERATION:**
This assumes the notebook's execution role has permission to call the endpoint:
```json
{
  "Effect": "Allow",
  "Action": "sagemaker:InvokeEndpoint",
  "Resource": "arn:aws:sagemaker:us-east-1:123456789012:endpoint/*"
}
```

**PRODUCTION ALTERNATIVE:**
Use environment variables or AWS Systems Manager Parameter Store:
```python
import os
endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
if not endpoint_name:
    raise ValueError("Set SAGEMAKER_ENDPOINT_NAME environment variable")

# OR using Parameter Store
import boto3
ssm = boto3.client('ssm')
endpoint_name = ssm.get_parameter(Name='/ml/mistral-endpoint')['Parameter']['Value']
```

**ERROR HANDLING ENHANCEMENT:**
```python
sm_client = boto3.client('sagemaker')
try:
    response = sm_client.describe_endpoint(EndpointName=endpoint_name)
    if response['EndpointStatus'] != 'InService':
        raise ValueError(f"Endpoint {endpoint_name} is {response['EndpointStatus']}, not InService")
except sm_client.exceptions.EndpointNotFound:
    raise ValueError(f"Endpoint {endpoint_name} does not exist")
```

---

## Section 4: Vanilla LLM Helper Functions
**Cell ID:** `76e09583-d585-47f7-b153-b9a4f5b32381`

### Code:
```python
client = boto3.client("runtime.sagemaker")

def query_endpoint(payload):
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8")
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response

def format_instructions(instructions: List[Dict[str, str]]) -> str:
    prompt: List[str] = []
    for user, answer in zip(instructions[::2], instructions[1::2]):
        prompt.extend(["<s>", "[INST] ", (user["content"]).strip(), " [/INST] ",
                      (answer["content"]).strip(), "</s>"])
    prompt.extend(["<s>", "[INST] ", (instructions[-1]["content"]).strip(), " [/INST] "])
    return "".join(prompt)

def print_instructions(prompt: str, response: str) -> None:
    bold, unbold = '\033[1m', '\033[0m'
    print(f"{bold}> Input{unbold}\n{prompt}\n\n{bold}> Output{unbold}\n{response[0]['generated_text']}\n")

def ask_question(question):
    instructions = [{"role": "user", "content": question}]
    prompt = format_instructions(instructions)
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 500, "do_sample": True, "temperature": 0.001}
    }
    response = query_endpoint(payload)
    print_instructions(prompt, response)
    return response[0]['generated_text']
```

### Annotation:

**PURPOSE:** Demonstrate baseline LLM capability **without** LangChain or LangGraph

**FUNCTION 1: `query_endpoint(payload)`**

**WHAT:** Low-level boto3 call to SageMaker endpoint

**HOW:**
```python
client.invoke_endpoint(
    EndpointName=endpoint_name,      # Target endpoint
    ContentType="application/json",  # Tell endpoint to expect JSON
    Body=json.dumps(payload).encode("utf-8")  # Convert dict → JSON string → bytes
)
```

**WHY:**
- Shows the raw API call that LangChain abstracts away
- Useful for debugging (what exactly is being sent?)
- Can be used when LangChain's abstraction is too limiting

**RETURN VALUE:**
```python
{
    "Body": StreamingBody object,  # Must call .read() to get bytes
    "ContentType": "application/json",
    "InvokedProductionVariant": "AllTraffic"
}
```

---

**FUNCTION 2: `format_instructions(instructions)`**

**WHAT:** Format conversation history into Mistral's expected prompt template

**HOW:**
Uses Mistral's chat template format:
```
<s>[INST] User message 1 [/INST] Assistant response 1</s>
<s>[INST] User message 2 [/INST] Assistant response 2</s>
<s>[INST] User message 3 [/INST]
```

**WHY:**
- Mistral 7B Instruct is a chat model fine-tuned on this specific format
- `<s>` = start of sequence token
- `[INST]` = instruction marker (tells model this is user input)
- `[/INST]` = end of instruction (assistant should respond here)
- Missing the last `</s>` signals model to continue generating

**KEY INSIGHT:**
```python
for user, answer in zip(instructions[::2], instructions[1::2]):
```
- `instructions[::2]`: Every even index (0, 2, 4...) = user messages
- `instructions[1::2]`: Every odd index (1, 3, 5...) = assistant messages
- `zip()` pairs them up: (user1, assistant1), (user2, assistant2)...

**LAST LINE:**
```python
prompt.extend(["<s>", "[INST] ", (instructions[-1]["content"]).strip(), " [/INST] "])
```
Handles the final user message (no assistant response yet)

---

**FUNCTION 3: `print_instructions(prompt, response)`**

**WHAT:** Pretty-print the input/output with ANSI formatting

**HOW:**
- `\033[1m`: ANSI escape code for bold text
- `\033[0m`: Reset formatting

**WHY:**
- Improves readability in Jupyter output
- Clearly separates prompt from response
- Useful for debugging prompt formatting issues

---

**FUNCTION 4: `ask_question(question)`**

**WHAT:** Complete pipeline for asking a question without LangChain

**FLOW:**
1. Wrap question in message format: `[{"role": "user", "content": question}]`
2. Format into Mistral template: `<s>[INST] {question} [/INST]`
3. Build payload with parameters
4. Send to endpoint via `query_endpoint()`
5. Print and return result

**WHY THIS EXISTS:**
- Establishes baseline for comparison
- Shows what LangChain is abstracting
- Useful for troubleshooting LangChain issues (does raw API work?)

---

## Section 5: LangChain ContentHandler & Endpoint Setup
**Cell ID:** `0625e0a9-cefc-4ec1-921c-5f96cf004ed0`

### Code:
```python
from langchain_community.llms import SagemakerEndpoint
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler

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

content_handler = ContentHandler()

llm = SagemakerEndpoint(
    endpoint_name=endpoint_name,
    region_name="us-east-1",
    model_kwargs={"max_new_tokens": 500, "do_sample": True, "temperature": 0.001},
    content_handler=content_handler
)
```

### Annotation:

**WHAT:** Create LangChain wrapper around SageMaker endpoint

**WHY THIS MATTERS:**
- **Without this:** Agent code would need boto3 calls, JSON serialization, error handling
- **With this:** Agent just calls `llm.invoke("question")` - framework handles the rest

**ContentHandler PATTERN:**

This is an **Adapter Pattern** - adapting SageMaker's interface to LangChain's interface:

```
LangChain API          ContentHandler          SageMaker API
━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━
llm.invoke(str)  →  transform_input()  →  {"inputs": str, ...}
                    transform_output() ←  [{"generated_text": ...}]
return str      ←                     ←
```

**WHY max_new_tokens=500 IS CRITICAL:**

Example of failure with max_new_tokens=128:
```
LLM output (truncated): "<tool>tavily_search_results_json</tool><tool_input>latest UK st"
                                                                             ↑ CUT OFF HERE
XML parser: ERROR - No closing </tool_input> tag found
Agent: Retries or fails
```

With max_new_tokens=500:
```
LLM output (complete): "<tool>tavily_search_results_json</tool><tool_input>latest UK storm</tool_input>"
XML parser: SUCCESS - Extracted tool call
Agent: Executes tool
```

**COST CALCULATION:**
- Endpoint: ml.g5.xlarge @ $1.006/hour = $0.00028/second
- Inference time: ~2 seconds for 500 tokens
- Cost per call: ~$0.00056
- 1000 questions/day: ~$0.56/day for inference (+ $24.14/day for 24/7 endpoint)

---

## Section 6: LangGraph Agent Construction
**Cell ID:** `46d2fbde-72fe-498b-9ab9-474122a66104`

### Code:
```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_xml_agent
from langgraph.prebuilt import create_agent_executor

tools = [TavilySearchResults(max_results=1)]
prompt = hub.pull("hwchase17/xml-agent-convo")
agent_runnable = create_xml_agent(llm, tools, prompt)
app = create_agent_executor(agent_runnable, tools)
```

### Annotation:

**WHAT:** Build the complete LangGraph agent with tools

**ARCHITECTURE LAYERS:**

```
Layer 1: Tools
├─ TavilySearchResults (max_results=1)
└─ [Future tools added here]

Layer 2: Prompt Template
├─ Instructions: "You are a helpful assistant..."
├─ Tool descriptions: Auto-injected
├─ XML format examples: <tool>name</tool><tool_input>input</tool_input>
└─ Placeholders: {input}, {chat_history}, {agent_scratchpad}

Layer 3: Agent Runnable (LLM + Prompt + Parsing)
├─ Format prompt with current state
├─ Call llm.invoke()
├─ Parse XML response
└─ Return: AgentAction or AgentFinish

Layer 4: LangGraph Executor (Stateful Graph)
├─ Agent Node: Calls agent_runnable
├─ Tools Node: Executes tools
├─ Conditional Edges: Route based on AgentAction/AgentFinish
└─ State: Maintains history, intermediate steps
```

**CRITICAL DESIGN DECISION: max_results=1**

**Why not more?**
1. **Context length:** Each result ~200-300 tokens
   - 5 results = 1000-1500 extra tokens
   - Longer context = higher cost + slower inference
2. **Relevance:** Tavily ranks by relevance - first result usually sufficient
3. **LLM focus:** More results = more info to parse = higher chance of errors

**When to increase:**
- Controversial topics (need multiple sources)
- Comparison tasks ("compare X vs Y")
- Multi-part questions requiring diverse sources

**XML vs JSON for Tool Calling:**

**Why XML?**
- Smaller models (7B params) struggle with strict JSON syntax
- XML is more forgiving: `<tool>name</tool>` vs `{"tool": "name"}`
- Missing quotes, extra commas → JSON parsing fails
- XML parsers use regex → partial matches work

**Example failure with JSON:**
```json
{"tool": "search", "input": "query}  ← Missing closing quote
```
Parser: Complete failure

**Example partial success with XML:**
```xml
<tool>search</tool><tool_input>query
```
Parser: Can extract `search` and `query` even without closing tag

---

## Section 7: LangGraph Execution with Streaming
**Cell ID:** `67bc38b0-5896-4755-9f82-d02293b2d0b5`

### Code:
```python
from langchain_core.agents import AgentFinish

def ask_langgraph_question(question):
    inputs = {"input": question, "chat_history": []}
    print("Asking question: " + question + "\n")

    stepIndex = 1
    agentFinish = None
    for s in app.stream(inputs):
        print("Step " + str(stepIndex) + ":")
        agentValues = list(s.values())[0]
        print(agentValues)
        print("----")
        stepIndex = stepIndex + 1
        if 'agent_outcome' in agentValues and isinstance(agentValues['agent_outcome'], AgentFinish):
            agentFinish = agentValues['agent_outcome']

    print("Final Outcome:\n")
    print(agentFinish)
    print("----\n")
    return agentFinish.return_values["output"]

for question in questions_and_answers:
    question["answers"]["langgraph"] = ask_langgraph_question(question["question"])
```

### Annotation:

**WHAT:** Execute the LangGraph agent and trace execution steps

**KEY METHOD: `app.stream(inputs)`**

**WHAT IT DOES:**
- Generator that yields state updates after each node execution
- Enables real-time monitoring of agent's thought process

**YIELDED VALUES:**
```python
# Step 1: Agent decides to call tool
{
    'agent': {  # ← Node name
        'agent_outcome': AgentAction(
            tool='tavily_search_results_json',
            tool_input='latest UK storm',
            log='<tool>tavily_search_results_json</tool>...'
        )
    }
}

# Step 2: Tool executes
{
    'tools': {  # ← Node name
        'intermediate_steps': [
            (AgentAction(...), '[{"url": "...", "content": "Storm Henk..."}]')
        ]
    }
}

# Step 3: Agent processes results
{
    'agent': {
        'agent_outcome': AgentFinish(
            return_values={'output': 'Storm Henk caused damage in...'},
            log='<final_answer>Storm Henk...</final_answer>'
        )
    }
}

# Step 4: Final state (complete history)
{
    'input': 'What is the latest storm...',
    'chat_history': [],
    'agent_outcome': AgentFinish(...),
    'intermediate_steps': [(AgentAction(...), "...")]
}
```

**WHY STREAMING MATTERS:**

1. **Debugging:** See exactly where agent fails
   ```
   Step 1: Agent calls tool ✓
   Step 2: Tool returns data ✓
   Step 3: Agent fails to parse → DEBUG HERE
   ```

2. **User experience:** Show progress indicators
   ```
   "Searching the web..."  (Step 1)
   "Analyzing results..."  (Step 2)
   "Generating answer..."  (Step 3)
   ```

3. **Monitoring:** Log each step for analytics
   ```python
   for s in app.stream(inputs):
       cloudwatch.log(event="agent_step", data=s)
   ```

**TERMINATION DETECTION:**
```python
if 'agent_outcome' in agentValues and isinstance(agentValues['agent_outcome'], AgentFinish):
    agentFinish = agentValues['agent_outcome']
```

**WHY THIS CHECK:**
- Not all steps contain `agent_outcome` (e.g., tools node has `intermediate_steps`)
- `agent_outcome` can be `AgentAction` (continue) or `AgentFinish` (done)
- Only `AgentFinish` has the final answer in `return_values["output"]`

**EXTENSION: Adding Timeouts**
```python
import time
start_time = time.time()
MAX_STEPS = 10
MAX_TIME = 30  # seconds

for s in app.stream(inputs):
    if stepIndex > MAX_STEPS:
        raise RuntimeError("Agent exceeded max steps")
    if time.time() - start_time > MAX_TIME:
        raise RuntimeError("Agent exceeded time limit")
    # ... rest of code
```

---

## Section 8: Tavily API Key Configuration
**Cell ID:** `6436f72b-a888-4449-a912-2567de5d438c`

### Code:
```python
import os
import getpass

os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key:")
```

### Annotation:

**WHAT:** Securely prompt for Tavily API key

**HOW:**
- `getpass.getpass()`: Like `input()` but hides typed characters (shows `········`)
- Sets environment variable: LangChain tools read from `os.environ`

**WHY ENVIRONMENT VARIABLES:**
- Tavily tool checks `os.getenv("TAVILY_API_KEY")` internally
- Environment variables don't persist after notebook restarts
- Won't accidentally commit secrets to Git (unlike hardcoded keys)

**SECURITY BEST PRACTICES:**

**❌ BAD:**
```python
os.environ["TAVILY_API_KEY"] = "tvly-abc123def456"  # Visible in notebook!
```

**✅ BETTER:**
```python
os.environ["TAVILY_API_KEY"] = getpass.getpass()
```

**✅ BEST (Production):**
```python
import boto3
secrets = boto3.client('secretsmanager')
secret = secrets.get_secret_value(SecretId='tavily-api-key')
os.environ["TAVILY_API_KEY"] = json.loads(secret['SecretString'])['api_key']
```

**TAVILY PRICING (as of 2024):**
- Free tier: 1,000 searches/month
- Pro: $0.001/search ($1 per 1,000 searches)
- This demo: 2 questions × ~1 search each = $0.002

**ALTERNATIVE TOOLS (no API key needed):**
```python
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun

tools = [
    DuckDuckGoSearchRun(),  # Free, but rate-limited and less reliable
    WikipediaQueryRun()     # Free, but only searches Wikipedia
]
```

---

## Section 9: Baseline LLM Testing
**Cell ID:** `6eb58038-3035-42b5-b4df-571332d856f2`

### Code:
```python
for question in questions_and_answers:
    question["answers"]["llm"] = ask_question(question["question"])
```

### Annotation:

**WHAT:** Test questions against raw LLM (no tools)

**WHY THIS STEP:**
- Establishes baseline performance
- Shows LLM's inherent knowledge vs. need for tools
- Demonstrates training data limitations

**OBSERVED RESULTS (from notebook):**

**Question 1: "What is the recipe of mayonnaise?"**
```
LLM Answer:
1. Gather all ingredients:
- 2 large egg yolks
- 1 tablespoon Dijon mustard
...
6. Serve chilled and enjoy your homemade mayonnaise!
```

**Analysis:**
- ✅ Correct recipe (from training data)
- ❌ No citations/sources
- ❌ Can't verify if recipe is current/popular

---

**Question 2: "Latest storm to hit the UK?"**
```
LLM Answer:
Storm Arwen: This storm hit the UK on November 25th and 26th, 2021...
```

**Analysis:**
- ❌ Outdated (training data cutoff)
- ❌ If asked in 2024, Storm Arwen is NOT the latest
- ❌ LLM has no awareness of current date

**KEY INSIGHT:**
This demonstrates the **critical need** for tool access:
- Static knowledge → outdated answers
- Tools → current, cited information

---

## Section 10: Results Comparison Table
**Cell ID:** `a0bd488a-368f-4346-85ec-7e20726fafd1`

### Code:
```python
import pandas as pd
from IPython.display import display, HTML

display_table = [["Question", "LLM Only", "LangChain", "LangGraph"]]
for qa in questions_and_answers:
    display_table.append([
        qa["question"],
        qa["answers"]["llm"],
        qa["answers"]["langchain"],
        qa["answers"]["langgraph"]
    ])

df = pd.DataFrame(display_table)
pd.set_option('display.max_colwidth', None)
display(HTML(df.to_html().replace("\\n","<br>")))
```

### Annotation:

**WHAT:** Visualize results from all three approaches side-by-side

**HOW:**
1. Build 2D list: rows = questions, columns = approaches
2. Convert to Pandas DataFrame
3. Render as HTML table with formatting

**KEY FORMATTING:**
```python
.replace("\\n","<br>")
```
- Converts escaped newlines in strings to HTML line breaks
- Makes multi-line responses readable in table cells

**COMPARISON INSIGHTS (from notebook):**

| Approach | Mayonnaise Question | Storm Question |
|----------|-------------------|----------------|
| **LLM Only** | Recites recipe from memory | Says "Storm Arwen (2021)" ❌ |
| **LangChain** | Nearly identical to LLM | Nearly identical to LLM ❌ |
| **LangGraph** | Returns link to recipe website ✅ | Says "Storm Henk" ✅ |

**WHY LANGCHAIN ≈ LLM ONLY:**
- LangChain is just a wrapper here
- No tools invoked → same underlying model call
- Minor differences due to non-deterministic sampling

**WHY LANGGRAPH SUCCEEDS:**
- Calls Tavily search tool
- Gets current data from web
- Provides citations/links

**PRODUCTION METRICS TO ADD:**
```python
display_table = [[
    "Question",
    "LLM Only",
    "LangChain",
    "LangGraph",
    "LangGraph Token Cost",
    "LangGraph Latency"
]]

# Then populate with:
# - token_count * $cost_per_token
# - time_to_completion in seconds
```

---

## Summary: Why These 10 Sections Matter

| Section | Purpose | Key Learning |
|---------|---------|--------------|
| 1. Dependencies | Environment setup | Version management, kernel restarts |
| 2. Test Questions | Benchmark design | Time-sensitive vs. static questions |
| 3. Endpoint Input | Infrastructure connection | Separation of model deployment from usage |
| 4. Vanilla LLM Helpers | Baseline approach | Understanding what frameworks abstract |
| 5. ContentHandler | Protocol translation | Model-specific format handling |
| 6. Agent Construction | Graph architecture | Tools, prompts, runnables, state |
| 7. Streaming Execution | Observability | Real-time monitoring, debugging |
| 8. API Key Config | Security | Secret management best practices |
| 9. Baseline Testing | Comparison | Demonstrating LLM limitations |
| 10. Results Table | Evaluation | Visual comparison of approaches |

---

## Next Steps

After understanding these sections, explore:

1. **Custom Tools:** Build a database query tool, calculator, etc.
2. **Multi-Agent:** Create specialized agents (research + writing)
3. **Human-in-the-Loop:** Add approval steps before tool execution
4. **Persistence:** Use checkpointers to save/resume agent state
5. **Production:** Containerize, add logging, monitoring, error handling
