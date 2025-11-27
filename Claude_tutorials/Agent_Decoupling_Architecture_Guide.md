# Agent Decoupling Architecture Guide

## Overview: The Two-Component Pattern

This guide explains the **decoupled architecture** pattern demonstrated in this repository, where LangGraph agent orchestration runs separately from LLM inference.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   USER                                       │
│                          (Web App / API Client)                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   │ HTTPS Request
                                   │ POST /ask {"question": "..."}
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPONENT 1: AGENT HOST                                                     │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                               │
│  Responsibilities:                                                            │
│  • Receive user questions                                                    │
│  • Orchestrate multi-step reasoning (LangGraph)                              │
│  • Decide when to call tools vs. LLM                                         │
│  • Parse LLM responses (XML → structured data)                               │
│  • Execute tools (Tavily, Wikipedia, databases, etc.)                        │
│  • Maintain conversation state                                               │
│  • Return final answer to user                                               │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  LangGraph State Machine                                             │   │
│  │                                                                       │   │
│  │   ┌──────────┐         ┌──────────┐         ┌──────────┐           │   │
│  │   │  START   │────────▶│  AGENT   │────────▶│   END    │           │   │
│  │   │          │         │   NODE   │         │          │           │   │
│  │   └──────────┘         └─────┬────┘         └──────────┘           │   │
│  │                              │                    ▲                  │   │
│  │                              │ AgentAction        │                  │   │
│  │                              │ (tool call)        │                  │   │
│  │                              ▼                    │                  │   │
│  │                        ┌──────────┐               │                  │   │
│  │                        │  TOOLS   │               │                  │   │
│  │                        │   NODE   │───────────────┘                  │   │
│  │                        └──────────┘   tool result                    │   │
│  │                              │         (loops back)                   │   │
│  │                              │                                        │   │
│  │                              │ (may call external APIs)               │   │
│  │                              ▼                                        │   │
│  │                        [ Tavily API ]                                 │   │
│  │                        [ Wikipedia  ]                                 │   │
│  │                        [ Your DB    ]                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  Technology Stack:                                                            │
│  • Python 3.10+                                                               │
│  • langchain, langgraph                                                       │
│  • boto3 (AWS SDK for SageMaker API calls)                                   │
│                                                                               │
│  Compute:                                                                     │
│  • AWS Lambda (serverless, scales to zero)           Cost: $0-5/month        │
│  • ECS Fargate (container, always-on)                Cost: ~$15/month        │
│  • EC2 (t3.small or larger)                          Cost: ~$15/month        │
│                                                                               │
│  Hardware Requirements:                                                       │
│  • CPU: 1-2 vCPUs (no GPU needed)                                            │
│  • RAM: 512MB - 2GB                                                           │
│  • Network: Must reach SageMaker endpoint (AWS internal network)             │
│                                                                               │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              │ Network Call (Agent → LLM)
                              │
                              │ Protocol: HTTPS (boto3.client('runtime.sagemaker').invoke_endpoint)
                              │
                              │ Request Payload:
                              │ {
                              │   "inputs": "<s>[INST] Your prompt here [/INST]",
                              │   "parameters": {
                              │     "max_new_tokens": 500,
                              │     "temperature": 0.001,
                              │     "do_sample": true
                              │   }
                              │ }
                              │
                              │ Latency: ~2-5 seconds (model inference time)
                              │ Cost: Included in endpoint hourly rate
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPONENT 2: LLM INFERENCE ENDPOINT                                         │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                               │
│  Responsibilities:                                                            │
│  • Receive text prompts from Agent Host                                      │
│  • Generate text completions (inference)                                     │
│  • Return generated text in JSON format                                      │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SageMaker Real-Time Inference Endpoint                              │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  HuggingFace Text Generation Inference (TGI) Container          │ │   │
│  │  │                                                                  │ │   │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │   │
│  │  │  │  Mistral 7B Instruct Model                               │  │ │   │
│  │  │  │  • 7 billion parameters                                  │  │ │   │
│  │  │  │  • ~14GB model weights (FP16)                            │  │ │   │
│  │  │  │  • Loaded in GPU VRAM for fast inference                 │  │ │   │
│  │  │  │  • Generates ~200 tokens/second                          │  │ │   │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │   │
│  │  │                                                                  │ │   │
│  │  │  Optimizations:                                                  │ │   │
│  │  │  • FlashAttention 2 (faster attention mechanism)                │ │   │
│  │  │  • Continuous batching (handle multiple requests in parallel)   │ │   │
│  │  │  • KV cache (reuse key-value computations)                      │ │   │
│  │  └──────────────────────────────────────────────────────────────┘ │   │
│  │                                                                       │   │
│  │  Response Format:                                                     │   │
│  │  [                                                                    │   │
│  │    {                                                                  │   │
│  │      "generated_text": "<tool>tavily_search...</tool>",              │   │
│  │      "details": {                                                     │   │
│  │        "finish_reason": "eos_token",                                  │   │
│  │        "generated_tokens": 487                                        │   │
│  │      }                                                                 │   │
│  │    }                                                                  │   │
│  │  ]                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  Deployment Method:                                                           │
│  • SageMaker JumpStart (UI-based, 1-click deploy)                            │
│  • SageMaker SDK (Python code for automation)                                │
│                                                                               │
│  Compute:                                                                     │
│  • ml.g5.xlarge (1x NVIDIA A10G GPU, 24GB VRAM)   Cost: $1.006/hour          │
│  • ml.g5.2xlarge (1x A10G, 24GB, more CPU)        Cost: $1.515/hour          │
│  • ml.g5.4xlarge (1x A10G, 24GB, even more CPU)   Cost: $2.03/hour           │
│                                                                               │
│  Monthly Cost (24/7 operation):                                               │
│  • ml.g5.xlarge: $1.006 × 24 × 30 = ~$723/month                              │
│  • ml.g5.2xlarge: ~$1,090/month                                              │
│                                                                               │
│  Hardware Requirements:                                                       │
│  • GPU: NVIDIA A10G or better (24GB+ VRAM)                                   │
│  • CPU: 4-8 cores (for pre/post-processing)                                  │
│  • RAM: 32GB+ (for model loading + requests)                                 │
│  • Storage: ~50GB (model weights + container image)                          │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Data & Control Flow

### Complete Request-Response Cycle

```
┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: USER SENDS QUESTION                                                │
└────────────────────────────────────────────────────────────────────────────┘

User → Agent Host:
  POST /ask
  {
    "question": "What is the latest storm to hit the UK?",
    "chat_history": []
  }

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: AGENT INITIALIZES STATE                                            │
└────────────────────────────────────────────────────────────────────────────┘

Agent Host (LangGraph):
  state = {
    "input": "What is the latest storm to hit the UK?",
    "chat_history": [],
    "intermediate_steps": [],
    "agent_outcome": None
  }

  → Execute graph: START node → AGENT node

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: AGENT NODE - FIRST LLM CALL (Decision Making)                     │
└────────────────────────────────────────────────────────────────────────────┘

Agent Host:
  1. Format prompt:
     prompt = """
     You are a helpful assistant...
     You have access to: tavily_search_results_json - A search engine...

     Question: What is the latest storm to hit the UK?
     """

  2. Prepare SageMaker payload:
     payload = {
       "inputs": "<s>[INST] {prompt} [/INST]",
       "parameters": {"max_new_tokens": 500, "temperature": 0.001}
     }

  3. Send to LLM Endpoint:
     Agent Host → LLM Endpoint:
       POST https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/mistral-7b/invocations
       Content-Type: application/json
       Body: {"inputs": "...", "parameters": {...}}

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: LLM INFERENCE (First Call)                                         │
└────────────────────────────────────────────────────────────────────────────┘

LLM Endpoint:
  1. Receive request
  2. Tokenize input: "What is the latest..." → [1, 2574, 354, ...]
  3. Load prompt into GPU memory
  4. Run transformer layers (32 layers × 7B params)
  5. Generate tokens autoregressively:
     <tool>tavily_search_results_json</tool><tool_input>latest UK storm</tool_input>
  6. Respond:
     [{"generated_text": "<tool>tavily_search_results_json</tool><tool_input>latest UK storm</tool_input>"}]

  Latency: ~2-3 seconds

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: AGENT PARSES LLM RESPONSE (Tool Call Detected)                    │
└────────────────────────────────────────────────────────────────────────────┘

Agent Host:
  1. Receive response from LLM
  2. Extract generated_text
  3. Parse XML:
     - Regex match: <tool>(.*?)</tool> → "tavily_search_results_json"
     - Regex match: <tool_input>(.*?)</tool_input> → "latest UK storm"
  4. Create AgentAction:
     agent_action = AgentAction(
       tool="tavily_search_results_json",
       tool_input="latest UK storm",
       log="<tool>tavily_search_results_json</tool>..."
     )
  5. Update state:
     state["agent_outcome"] = agent_action

  6. Conditional routing: Agent outcome is AgentAction → Route to TOOLS node

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: TOOLS NODE - EXECUTE TOOL                                          │
└────────────────────────────────────────────────────────────────────────────┘

Agent Host:
  1. Look up tool by name: "tavily_search_results_json"
  2. Execute tool:
     Agent Host → Tavily API:
       POST https://api.tavily.com/search
       {
         "query": "latest UK storm",
         "api_key": "tvly-...",
         "max_results": 1
       }

  3. Receive tool result:
     [
       {
         "url": "https://www.theguardian.com/uk-news/2024/jan/02/...",
         "content": "Storm Henk has been named and is forecast to bring very strong winds..."
       }
     ]

  4. Update state:
     state["intermediate_steps"].append(
       (agent_action, str(tool_result))
     )

  5. Route back to AGENT node (loop continues)

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: AGENT NODE - SECOND LLM CALL (Synthesis)                          │
└────────────────────────────────────────────────────────────────────────────┘

Agent Host:
  1. Format prompt WITH tool result in scratchpad:
     prompt = """
     You are a helpful assistant...

     Question: What is the latest storm to hit the UK?

     You have called: tavily_search_results_json
     Result: <observation>[{"url": "...", "content": "Storm Henk..."}]</observation>

     What is your next action?
     """

  2. Send to LLM Endpoint (second call):
     Agent Host → LLM Endpoint:
       POST /invocations
       {"inputs": "<s>[INST] {prompt with observation} [/INST]", ...}

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 8: LLM INFERENCE (Second Call - Final Answer)                        │
└────────────────────────────────────────────────────────────────────────────┘

LLM Endpoint:
  1. Process prompt (now includes tool result)
  2. Generate response:
     <final_answer>The name of the latest storm to hit the UK is Storm Henk, and it caused the most damage in the south-west of England.</final_answer>
  3. Return:
     [{"generated_text": "<final_answer>Storm Henk...</final_answer>"}]

  Latency: ~2-3 seconds

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 9: AGENT PARSES FINAL ANSWER                                          │
└────────────────────────────────────────────────────────────────────────────┘

Agent Host:
  1. Parse XML:
     - Regex match: <final_answer>(.*?)</final_answer> → "Storm Henk..."
  2. Create AgentFinish:
     agent_finish = AgentFinish(
       return_values={"output": "Storm Henk..."},
       log="<final_answer>...</final_answer>"
     )
  3. Update state:
     state["agent_outcome"] = agent_finish

  4. Conditional routing: Agent outcome is AgentFinish → Route to END

┌────────────────────────────────────────────────────────────────────────────┐
│  STEP 10: RETURN RESPONSE TO USER                                           │
└────────────────────────────────────────────────────────────────────────────┘

Agent Host → User:
  HTTP 200 OK
  {
    "answer": "The name of the latest storm to hit the UK is Storm Henk...",
    "metadata": {
      "steps": 2,  // Agent node → Tools node → Agent node
      "tools_used": ["tavily_search_results_json"],
      "total_latency_ms": 5200
    }
  }
```

---

## Component Responsibilities

### Agent Host (Component 1)

| Responsibility | Why Agent Handles This | Example |
|----------------|------------------------|---------|
| **Orchestration** | Needs to decide: LLM vs. tool vs. finish | "Should I search or can I answer from memory?" |
| **State Management** | Tracks conversation history, tool results | `intermediate_steps = [(action1, result1), ...]` |
| **Tool Execution** | Tools are often I/O-bound (APIs, DBs) - CPU is fine | Call Tavily API, query database, read file |
| **Prompt Engineering** | Inject tool descriptions, format conversations | Add `{tools}` list to prompt |
| **Response Parsing** | Extract structured data from LLM text | Parse `<tool>name</tool>` XML tags |
| **Error Handling** | Retry logic, fallbacks, timeouts | If tool fails → inform LLM → LLM tries different approach |
| **Routing** | Conditional edges in graph (continue vs. end) | `AgentAction → TOOLS`, `AgentFinish → END` |

**Why CPU is Sufficient:**
- String operations (parsing, formatting)
- HTTP requests (boto3, requests library)
- Graph traversal (Python dict/list operations)
- No matrix multiplications or heavy numerical compute

---

### LLM Endpoint (Component 2)

| Responsibility | Why Endpoint Handles This | Hardware Requirement |
|----------------|---------------------------|---------------------|
| **Text Generation** | Requires GPU-accelerated matrix math | GPU (A10G, A100, H100) |
| **Model Loading** | 14GB model weights must fit in VRAM | 24GB+ VRAM |
| **Tokenization** | Convert text ↔ token IDs using vocab | CPU |
| **Inference** | Forward pass through 32 transformer layers | GPU (99% of compute time) |
| **KV Cache** | Store attention keys/values for speed | GPU VRAM |
| **Batching** | Process multiple requests in parallel | GPU + VRAM |

**Why GPU is Required:**
- Model contains billions of parameters (weight matrices)
- Each generation step: multiply input by billions of weights
- GPUs have 10,000+ cores for parallel matrix operations
- CPU: ~10-100x slower for inference

---

## Why Decouple? Detailed Justification

### 1. Cost Optimization

**Scenario:** 1,000 questions/day, average 2 LLM calls per question

| Architecture | Agent Compute | LLM Compute | Total/Month |
|--------------|---------------|-------------|-------------|
| **Combined** (Agent + LLM in one GPU instance) | Included | ml.g5.xlarge 24/7 | $723 |
| **Decoupled** (Lambda + SageMaker) | Lambda: 2,000 invocations × 3s × 1GB = **$0.10** | ml.g5.xlarge 24/7 | $723.10 |
| **Decoupled** (Fargate + SageMaker) | Fargate: 1 task × 0.5 vCPU, 1GB = **$15** | ml.g5.xlarge 24/7 | $738 |

**Key Insight:** Agent compute is negligible compared to LLM endpoint cost

**Savings Example:**
- If you could scale LLM endpoint to only run 8 hours/day (not realistic due to cold starts):
  - Combined: Can't separate → still pay 24/7
  - Decoupled: Turn off endpoint 16 hours → save ~$482/month

---

### 2. Independent Scaling

**Problem:** Traffic spike (1,000 → 10,000 requests/hour)

| Architecture | Agent Scaling | LLM Scaling | Cold Start |
|--------------|---------------|-------------|------------|
| **Combined** | Must scale entire GPU instance | Expensive: Add more ml.g5.xlarge instances | 2-5 minutes |
| **Decoupled** | Lambda auto-scales to 10,000 instances | LLM unchanged (can handle 10-20 concurrent requests) | Agent: 1-3 sec <br> LLM: Already warm |

**Cost Comparison (10,000 req/hour for 1 hour):**
- Combined: Need 50+ GPU instances @ $1.006/hour = **$50+ (wasteful)**
- Decoupled: Lambda scales for free, LLM endpoint unchanged = **$1.006 (efficient)**

---

### 3. Flexibility in Agent Logic

**Decoupled = Easy Updates:**

```python
# Update agent code (add new tool, change prompt, etc.)
# 1. Build new Docker image
docker build -t agent:v2 .

# 2. Push to ECR
docker push <ecr-repo>:v2

# 3. Update Lambda/ECS
aws lambda update-function-code --function-name agent --image-uri <ecr-repo>:v2

# Total downtime: ~10 seconds (Lambda) or zero-downtime with ECS blue-green
```

**Combined = Harder Updates:**
```python
# Update combined container
# 1. Build new image with model + agent code (can't separate)
# 2. Upload 20GB image to ECR (~10 minutes)
# 3. Create new SageMaker endpoint with new image
# 4. Wait for endpoint to be InService (~5-10 minutes)
# 5. Swap traffic (or downtime)
# Total downtime: ~15-20 minutes
```

---

### 4. Multi-Tenancy and Reusability

**One LLM Endpoint, Multiple Agents:**

```
┌─────────────────┐       ┌─────────────────────────────┐
│  Customer       │──────▶│  SageMaker Endpoint         │
│  Support Agent  │       │  (Mistral 7B)               │
└─────────────────┘       │  Shared across all agents   │
                          └─────────────────────────────┘
┌─────────────────┐              ▲
│  Code           │──────────────┘
│  Review Agent   │
└─────────────────┘

┌─────────────────┐
│  Research       │──────────────┘
│  Agent          │
└─────────────────┘
```

**Benefits:**
- One endpoint cost (~$723/month) supports 3 different agents
- Each agent can use different tools, prompts, logic
- Update one agent without affecting others

**Combined Architecture:** Would need 3 separate endpoints (~$2,169/month)

---

## Networking and Security

### Network Connectivity Requirements

```
┌────────────────────────────────────────────────────────────────────┐
│  VPC (Optional but Recommended for Production)                     │
│                                                                     │
│  ┌────────────────────────┐         ┌─────────────────────────┐   │
│  │  Private Subnet        │         │  Private Subnet          │   │
│  │  ┌──────────────────┐  │         │  ┌───────────────────┐  │   │
│  │  │ Agent (Lambda/   │  │         │  │ SageMaker         │  │   │
│  │  │  ECS/EC2)        │  │────────▶│  │ Endpoint          │  │   │
│  │  └──────────────────┘  │  VPC    │  │ (ENI in VPC)      │  │   │
│  │         │               │  Link   │  └───────────────────┘  │   │
│  └─────────┼───────────────┘         └─────────────────────────┘   │
│            │ NAT Gateway                                            │
└────────────┼───────────────────────────────────────────────────────┘
             │
             ▼
        Internet (for Tavily API, etc.)
```

**Security Best Practices:**

1. **IAM Role for Agent:**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": "sagemaker:InvokeEndpoint",
         "Resource": "arn:aws:sagemaker:us-east-1:123456789012:endpoint/mistral-7b-instruct"
       },
       {
         "Effect": "Allow",
         "Action": "secretsmanager:GetSecretValue",
         "Resource": "arn:aws:secretsmanager:us-east-1:123456789012:secret:tavily-key"
       }
     ]
   }
   ```

2. **Endpoint Access Control:**
   - Option A: VPC-only endpoint (no internet access)
   - Option B: Resource policy to restrict callers
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {
           "AWS": "arn:aws:iam::123456789012:role/AgentRole"
         },
         "Action": "sagemaker:InvokeEndpoint",
         "Resource": "*"
       }
     ]
   }
   ```

---

## Monitoring and Observability

### Key Metrics to Track

| Metric | Where to Measure | Alert Threshold |
|--------|------------------|-----------------|
| **LLM Endpoint Latency** | SageMaker CloudWatch | >5 seconds |
| **LLM Endpoint Error Rate** | SageMaker CloudWatch | >1% |
| **Agent Execution Time** | Lambda/ECS CloudWatch | >30 seconds |
| **Tool Call Success Rate** | Custom metric (agent logs) | <95% |
| **Cost per Question** | CloudWatch + billing | >$0.05 |
| **Questions per Hour** | Agent logs | (capacity planning) |

### Distributed Tracing Example

```python
# agent/main.py
import boto3
import time
import json

cloudwatch = boto3.client('cloudwatch')

def lambda_handler(event, context):
    start_time = time.time()
    trace_id = context.request_id  # Unique per invocation

    try:
        # Execute agent
        result = execute_agent(event["question"], trace_id)

        # Log success metrics
        cloudwatch.put_metric_data(
            Namespace='LangGraphAgent',
            MetricData=[
                {
                    'MetricName': 'ExecutionTime',
                    'Value': time.time() - start_time,
                    'Unit': 'Seconds',
                    'Dimensions': [{'Name': 'Status', 'Value': 'Success'}]
                }
            ]
        )

        return {"statusCode": 200, "body": json.dumps(result)}

    except Exception as e:
        # Log failure metrics
        cloudwatch.put_metric_data(
            Namespace='LangGraphAgent',
            MetricData=[
                {
                    'MetricName': 'Errors',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [{'Name': 'ErrorType', 'Value': type(e).__name__}]
                }
            ]
        )
        raise
```

---

## Summary: Architecture Principles

| Principle | Reason | Benefit |
|-----------|--------|---------|
| **Separation of Concerns** | Agent (logic) ≠ LLM (compute) | Independent updates, scaling |
| **Right Tool for the Job** | CPU for agent, GPU for LLM | Cost optimization |
| **Network as Integration** | HTTP API between components | Language-agnostic, testable |
| **Stateless Agent** | State in request/response or external DB | Horizontal scaling |
| **Always-On LLM** | Eliminate cold starts | Low latency |
| **Serverless Agent** | Scale to zero during idle | Minimize cost |

---

## When NOT to Decouple

**Use Combined (Agent + LLM in one container) when:**

1. **Extreme Low Latency Required (<100ms):**
   - Trading algorithms
   - Real-time gaming
   - Network overhead (~50ms) is unacceptable

2. **Offline/Edge Deployment:**
   - No internet connectivity
   - Must run on a single device
   - Example: On-premise AI assistant

3. **Very Simple Agent (1-2 steps):**
   - Agent logic is trivial (just prompt formatting)
   - No tools, no complex routing
   - Overhead of separate components not justified

4. **Learning/Prototyping:**
   - Easier to run everything in a Jupyter notebook
   - Production can optimize later

**For this repository's use case (multi-step agent with web search), decoupling is the right choice.**
