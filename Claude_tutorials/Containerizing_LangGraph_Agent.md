# Containerizing the LangGraph Agent

## Critical Concept: There Is No Single "LangGraph Container"

**IMPORTANT:** When deploying this architecture, understand that:

❌ **MISCONCEPTION:** "Deploy LangGraph to SageMaker in a container"
✅ **REALITY:** LangGraph runs **separately** from the LLM, typically on cheap CPU compute

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT HOST (Your Container)                                     │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ LangGraph Orchestration Logic                        │       │
│  │ • Python runtime                                     │       │
│  │ • langchain, langgraph libraries                     │       │
│  │ • Your agent code (graph definition, tools)          │       │
│  │ • ContentHandler (SageMaker protocol adapter)        │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                   │
│  Deployment Options:                                             │
│  • Lambda Function (serverless, scales to zero)                  │
│  • ECS/Fargate Container (managed containers)                    │
│  • EC2 Instance (self-managed)                                   │
│  • Local Jupyter Notebook (development only)                     │
│                                                                   │
│  Hardware Requirements:                                          │
│  • CPU: 1-2 vCPUs (no GPU needed!)                               │
│  • RAM: 512MB - 2GB                                              │
│  • Storage: ~500MB (Python + dependencies)                       │
│                                                                   │
│  Cost: ~$5-20/month (or $0 with Lambda free tier)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTP POST (boto3.invoke_endpoint)
                             │ Payload: {"inputs": "...", "parameters": {...}}
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM INFERENCE ENDPOINT (SageMaker Managed)                      │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ SageMaker Endpoint                                   │       │
│  │ • Mistral 7B Instruct Model                          │       │
│  │ • HuggingFace TGI Container (pre-built)              │       │
│  │ • GPU-accelerated inference                          │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                   │
│  Deployment: SageMaker JumpStart or SDK                          │
│                                                                   │
│  Hardware Requirements:                                          │
│  • GPU: ml.g5.xlarge (24GB VRAM) or larger                       │
│  • Storage: ~20GB for model weights                              │
│                                                                   │
│  Cost: ~$720-1500/month (24/7 operation)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Agent Container Contents

### What Goes Into the Agent Container?

The agent container **does NOT contain**:
- ❌ The LLM model (Mistral 7B is in SageMaker endpoint)
- ❌ GPU drivers or CUDA
- ❌ HuggingFace model files

The agent container **DOES contain**:
- ✅ Python runtime (3.10+)
- ✅ LangChain/LangGraph libraries
- ✅ Your agent code (graph definitions, tools, prompts)
- ✅ AWS SDK (boto3) for calling SageMaker
- ✅ External tool integrations (Tavily, Wikipedia, etc.)

---

### Dockerfile Example

```dockerfile
# ============================================================================
# BASE IMAGE: Use official Python slim image (no unnecessary packages)
# ============================================================================
FROM python:3.10-slim

# ============================================================================
# METADATA
# ============================================================================
LABEL maintainer="your-team@example.com"
LABEL description="LangGraph Agent for SageMaker Mistral 7B"

# ============================================================================
# SYSTEM DEPENDENCIES (minimal - only what's needed for Python packages)
# ============================================================================
RUN apt-get update && apt-get install -y \
    gcc \
    # ↑ Required for compiling some Python packages (e.g., lxml, numpy)
    && rm -rf /var/lib/apt/lists/*
    # ↑ Clean up apt cache to reduce image size

# ============================================================================
# WORKING DIRECTORY
# ============================================================================
WORKDIR /app

# ============================================================================
# PYTHON DEPENDENCIES
# ============================================================================
# Copy requirements first (Docker layer caching - only rebuilds if requirements change)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir: Don't store pip cache (reduces image size by ~200MB)

# ============================================================================
# APPLICATION CODE
# ============================================================================
COPY agent/ /app/agent/
# agent/
# ├── __init__.py
# ├── graph.py          # LangGraph graph definition
# ├── tools.py          # Custom tools
# ├── prompts.py        # Prompt templates
# └── handlers.py       # ContentHandler for SageMaker

COPY main.py /app/
# Entry point for your application (FastAPI app, Lambda handler, etc.)

# ============================================================================
# ENVIRONMENT VARIABLES (defaults, can be overridden at runtime)
# ============================================================================
ENV SAGEMAKER_ENDPOINT_NAME="mistral-7b-instruct"
ENV AWS_DEFAULT_REGION="us-east-1"
ENV LOG_LEVEL="INFO"

# ============================================================================
# RUNTIME USER (security best practice - don't run as root)
# ============================================================================
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# ============================================================================
# EXPOSE PORT (if running as web service)
# ============================================================================
EXPOSE 8080

# ============================================================================
# ENTRYPOINT
# ============================================================================
CMD ["python", "main.py"]
# For FastAPI: uvicorn main:app --host 0.0.0.0 --port 8080
# For Lambda: Use AWS Lambda Python runtime (no CMD needed)
```

---

### requirements.txt

```txt
# ============================================================================
# CORE LANGCHAIN/LANGGRAPH
# ============================================================================
langchain==0.1.0
langgraph==0.0.20
langchain-community==0.0.10

# ============================================================================
# AWS SDK
# ============================================================================
boto3==1.34.10
# Required for:
# - sagemaker-runtime: invoke_endpoint()
# - secretsmanager: get_secret_value() for API keys
# - s3: Optional - for storing conversation history

# ============================================================================
# SAGEMAKER (optional - only if using high-level SDK)
# ============================================================================
# sagemaker==2.198.0  # Only needed if deploying endpoints from agent code

# ============================================================================
# EXTERNAL TOOLS
# ============================================================================
# Tavily search
tavily-python==0.2.0  # Or install via langchain-community

# Wikipedia (alternative free search)
wikipedia==1.4.0

# ============================================================================
# WEB FRAMEWORK (choose one based on deployment)
# ============================================================================
# For Lambda:
# (no web framework needed - use Lambda handler)

# For ECS/EC2 (FastAPI):
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# For Flask (alternative):
# flask==3.0.0

# ============================================================================
# UTILITIES
# ============================================================================
python-dotenv==1.0.0  # For local development (.env files)
structlog==23.2.0     # Structured logging for production

# ============================================================================
# MONITORING (optional but recommended for production)
# ============================================================================
# opentelemetry-api==1.21.0
# opentelemetry-sdk==1.21.0
# opentelemetry-instrumentation-fastapi==0.42b0
```

**Image size estimate:**
- Base Python 3.10-slim: ~150MB
- Dependencies: ~400MB
- Application code: <10MB
- **Total: ~560MB** (compare to LLM container: ~20GB!)

---

## Part 2: Why Deploy Agent Separately?

### Cost Comparison

| Deployment Model | Monthly Cost | Use Case |
|------------------|--------------|----------|
| **COMBINED** (agent + LLM in one container) | $1,500+ | ❌ Wasteful |
| **DECOUPLED** (agent on Lambda, LLM on SageMaker) | $730 (SageMaker) + $0 (Lambda free tier) = $730 | ✅ Optimal |
| **DECOUPLED** (agent on Fargate, LLM on SageMaker) | $730 (SageMaker) + $15 (Fargate) = $745 | ✅ Good |

**Why the difference?**

1. **Agent doesn't need GPU:**
   - LangGraph logic: String parsing, graph traversal, API calls
   - CPU is sufficient
   - GPU instances (ml.g5.xlarge): $1.006/hour
   - CPU instances (Lambda): $0.0000166667/GB-second (essentially free for low volume)

2. **Agent can scale to zero:**
   - No requests → no cost (Lambda, Fargate with scaling to 0)
   - LLM endpoint: Must run 24/7 for low latency (cold start = 2-5 minutes)

3. **Agent scales independently:**
   - 100 concurrent users → spin up 100 Lambda instances
   - LLM endpoint: One instance can handle 10-20 concurrent requests
   - Scaling agent is cheap, scaling LLM is expensive

---

### Latency Comparison

| Architecture | Cold Start | Request Latency |
|--------------|------------|-----------------|
| **Combined** | 2-5 minutes (load 7B model into VRAM) | 2-3 seconds |
| **Decoupled** | Agent: 1-3 seconds (Lambda) <br> LLM: Always warm | 2-3 seconds + network (~50ms) |

**Key Insight:** Decoupled adds ~50ms network latency, but eliminates cold starts

---

## Part 3: Deployment Options

### Option 1: AWS Lambda (Recommended for Low-Medium Volume)

**Pros:**
- ✅ Zero cost for <1M requests/month (free tier)
- ✅ Scales automatically (0 to 10,000 concurrent instances)
- ✅ No server management
- ✅ Pay only for compute time (billed in 1ms increments)

**Cons:**
- ❌ 15-minute execution timeout (fine for most agents)
- ❌ Cold starts (1-3 seconds for Python + dependencies)
- ❌ 10GB container image size limit (agent is ~560MB, well under limit)

**Deployment:**

```python
# lambda_handler.py
import json
import os
from agent.graph import create_agent_graph

# Initialize once (reused across invocations)
app = create_agent_graph(
    endpoint_name=os.environ["SAGEMAKER_ENDPOINT_NAME"]
)

def lambda_handler(event, context):
    """
    AWS Lambda handler for LangGraph agent

    Event format:
    {
        "question": "What is the latest UK storm?",
        "chat_history": []  # Optional
    }
    """
    question = event.get("question")
    chat_history = event.get("chat_history", [])

    # Execute graph
    inputs = {"input": question, "chat_history": chat_history}
    result = None

    for state in app.stream(inputs):
        # Could emit intermediate updates to EventBridge/SNS here
        if "agent_outcome" in state.get("agent", {}):
            from langchain_core.agents import AgentFinish
            outcome = state["agent"]["agent_outcome"]
            if isinstance(outcome, AgentFinish):
                result = outcome.return_values["output"]

    return {
        "statusCode": 200,
        "body": json.dumps({
            "answer": result,
            "metadata": {
                "steps": len(state.get("intermediate_steps", []))
            }
        })
    }
```

**Deploy with AWS CDK:**
```python
from aws_cdk import (
    aws_lambda as lambda_,
    aws_iam as iam,
    Duration
)

agent_lambda = lambda_.DockerImageFunction(
    self, "LangGraphAgent",
    code=lambda_.DockerImageCode.from_image_asset("./agent"),
    timeout=Duration.minutes(5),
    memory_size=1024,  # MB
    environment={
        "SAGEMAKER_ENDPOINT_NAME": "mistral-7b-instruct",
        "TAVILY_API_KEY": "{{resolve:secretsmanager:tavily-key:SecretString:api_key}}"
    }
)

# Grant permission to invoke SageMaker endpoint
agent_lambda.add_to_role_policy(iam.PolicyStatement(
    actions=["sagemaker:InvokeEndpoint"],
    resources=[f"arn:aws:sagemaker:{region}:{account}:endpoint/mistral-7b-instruct"]
))
```

---

### Option 2: ECS Fargate (Recommended for High Volume)

**Pros:**
- ✅ No cold starts (keep minimum 1 task running)
- ✅ Full container flexibility
- ✅ Scales to handle sustained high traffic
- ✅ Can run long-running processes (>15 minutes)

**Cons:**
- ❌ Always-on cost (minimum $15/month for 1 task)
- ❌ More complex deployment than Lambda
- ❌ Need to manage scaling policies

**Deployment:**

```python
# main.py - FastAPI application
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.graph import create_agent_graph
import os

app_api = FastAPI(title="LangGraph Agent API")

# Initialize agent graph once at startup
agent_graph = create_agent_graph(
    endpoint_name=os.environ["SAGEMAKER_ENDPOINT_NAME"]
)

class Question(BaseModel):
    question: str
    chat_history: list = []

@app_api.post("/ask")
async def ask_question(q: Question):
    """Execute LangGraph agent with a question"""
    inputs = {"input": q.question, "chat_history": q.chat_history}

    result = None
    steps = []

    for state in agent_graph.stream(inputs):
        # Log each step
        steps.append(str(state))

        # Check for completion
        if "agent" in state:
            from langchain_core.agents import AgentFinish
            outcome = state["agent"].get("agent_outcome")
            if isinstance(outcome, AgentFinish):
                result = outcome.return_values["output"]

    if result is None:
        raise HTTPException(status_code=500, detail="Agent failed to complete")

    return {
        "answer": result,
        "steps": steps
    }

@app_api.get("/health")
async def health_check():
    """Kubernetes/ALB health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_api, host="0.0.0.0", port=8080)
```

**Deploy with ECS:**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t langgraph-agent .
docker tag langgraph-agent:latest <account>.dkr.ecr.us-east-1.amazonaws.com/langgraph-agent:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/langgraph-agent:latest

# Create ECS task definition + service (via CDK/CloudFormation or Console)
```

---

### Option 3: EC2 Instance (For Learning/Custom Needs)

**Pros:**
- ✅ Full control over environment
- ✅ Can install any system dependencies
- ✅ Persistent storage

**Cons:**
- ❌ Most expensive (even t3.small is $15/month)
- ❌ Manual scaling and management
- ❌ You handle all security patches, monitoring, etc.

**When to use:**
- Learning/experimentation
- Need for specific OS-level dependencies
- Stateful applications (though prefer managed databases instead)

---

## Part 4: Agent Code Structure

### Recommended Project Layout

```
langgraph-agent/
├── Dockerfile
├── requirements.txt
├── main.py                  # Entry point (Lambda handler or FastAPI)
├── agent/
│   ├── __init__.py
│   ├── graph.py             # LangGraph graph construction
│   │   └── create_agent_graph() → app
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── tavily_search.py
│   │   ├── wikipedia.py
│   │   └── custom_tool.py   # Your custom tools
│   ├── handlers/
│   │   ├── __init__.py
│   │   └── sagemaker.py     # ContentHandler for SageMaker
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── agent_prompt.py  # Prompt templates
│   └── utils/
│       ├── __init__.py
│       ├── logging.py       # Structured logging setup
│       └── monitoring.py    # CloudWatch metrics
├── tests/
│   ├── test_graph.py
│   ├── test_tools.py
│   └── test_handlers.py
└── infrastructure/          # IaC (CDK/Terraform)
    ├── lambda_stack.py
    └── ecs_stack.py
```

---

### Example: graph.py (Extracting from Notebook)

```python
# agent/graph.py
from langchain import hub
from langchain.agents import create_xml_agent
from langchain_community.llms import SagemakerEndpoint
from langgraph.prebuilt import create_agent_executor
from .handlers.sagemaker import create_content_handler
from .tools import create_tools
import os

def create_agent_graph(endpoint_name: str = None):
    """
    Create the LangGraph agent executor

    Args:
        endpoint_name: SageMaker endpoint name (defaults to env var)

    Returns:
        LangGraph app (stateful graph executor)
    """
    # Configuration
    endpoint_name = endpoint_name or os.environ.get("SAGEMAKER_ENDPOINT_NAME")
    if not endpoint_name:
        raise ValueError("Must provide endpoint_name or set SAGEMAKER_ENDPOINT_NAME")

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    # Create LLM client with SageMaker endpoint
    llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name=region,
        model_kwargs={
            "max_new_tokens": 500,
            "do_sample": True,
            "temperature": 0.001
        },
        content_handler=create_content_handler()
    )

    # Create tools
    tools = create_tools()

    # Load prompt
    prompt = hub.pull("hwchase17/xml-agent-convo")

    # Build agent
    agent_runnable = create_xml_agent(llm, tools, prompt)

    # Create graph executor
    app = create_agent_executor(agent_runnable, tools)

    return app
```

---

## Part 5: Production Considerations

### Secret Management

```python
# ❌ BAD: Hardcoded secrets
os.environ["TAVILY_API_KEY"] = "tvly-abc123"

# ✅ GOOD: AWS Secrets Manager
import boto3
import json

def get_secret(secret_name: str) -> dict:
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

tavily_key = get_secret('prod/tavily-api-key')['api_key']
os.environ["TAVILY_API_KEY"] = tavily_key
```

---

### Logging and Monitoring

```python
# agent/utils/logging.py
import structlog
import logging

def setup_logging():
    """Configure structured logging for production"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

# Usage in graph.py
log = structlog.get_logger()

for state in app.stream(inputs):
    log.info("agent_step",
             step_type=list(state.keys())[0],
             has_outcome="agent_outcome" in state.get("agent", {}))
```

---

### Error Handling

```python
# main.py - Lambda handler with error handling
import traceback

def lambda_handler(event, context):
    try:
        question = event.get("question")
        if not question:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'question' field"})
            }

        result = execute_agent(question)

        return {
            "statusCode": 200,
            "body": json.dumps({"answer": result})
        }

    except Exception as e:
        log.error("agent_execution_failed",
                  error=str(e),
                  traceback=traceback.format_exc())

        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Agent execution failed",
                "message": str(e)
            })
        }
```

---

## Summary: Key Takeaways

1. **Agent ≠ LLM Container:**
   - Agent: Orchestration logic, CPU-only, cheap
   - LLM: Model inference, GPU-required, expensive

2. **Decoupling Saves Money:**
   - Combined: $1,500+/month
   - Decoupled: $730/month (50% savings)

3. **Choose Deployment Based on Scale:**
   - Low volume (<100K requests/month): **Lambda**
   - High volume: **ECS Fargate**
   - Learning: **Local/EC2**

4. **Keep It Simple:**
   - Agent container: ~560MB
   - Dependencies: langchain, boto3, fastapi
   - No CUDA, TensorFlow, PyTorch, or model files

5. **Production Checklist:**
   - ✅ Secrets in Secrets Manager (not env vars)
   - ✅ Structured logging (JSON format)
   - ✅ Error handling and retries
   - ✅ Health check endpoints
   - ✅ IAM least-privilege permissions
   - ✅ Monitoring (CloudWatch Logs + Metrics)
