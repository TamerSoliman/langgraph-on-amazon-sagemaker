# Interactive Tutorial Notebooks

## Overview

This directory contains 7 hands-on Jupyter notebooks that teach LangGraph concepts through interactive examples.

**For AI/ML Scientists:**
These notebooks are like ML course labs - you'll learn by running code, seeing outputs, and experimenting. Start with Notebook 1 and progress sequentially.

**Prerequisites:**
- Python 3.9+
- Basic Python knowledge
- AWS account (for deployment notebooks)
- Familiarity with ML concepts helpful but not required

---

## Notebook Index

### 🎯 Beginner Track (Notebooks 1-3)

**1. LangGraph Basics: Your First Agent** (`01_first_agent.ipynb`)
- State machines and graph concepts
- Creating a simple Q&A agent
- Understanding nodes and edges
- Running your first agent locally

**Estimated time:** 30 minutes
**Difficulty:** ⭐ Beginner

---

**2. Adding Tools: Web Search Integration** (`02_adding_tools.ipynb`)
- What are tools and why use them?
- Integrating Tavily web search
- Tool selection and execution
- Handling tool errors

**Estimated time:** 45 minutes
**Difficulty:** ⭐⭐ Beginner

---

**3. SageMaker Deployment: From Local to Cloud** (`03_sagemaker_deployment.ipynb`)
- Deploy Mistral-7B to SageMaker
- Connect agent to SageMaker endpoint
- Test cloud deployment
- Monitor and troubleshoot

**Estimated time:** 60 minutes
**Difficulty:** ⭐⭐ Intermediate

---

### 🚀 Intermediate Track (Notebooks 4-5)

**4. Multi-Agent Systems: Collaboration Patterns** (`04_multi_agent.ipynb`)
- Designing multi-agent workflows
- Researcher → Writer → Reviewer example
- Conditional routing and loops
- Managing shared state

**Estimated time:** 90 minutes
**Difficulty:** ⭐⭐⭐ Intermediate

---

**5. Human-in-the-Loop: Adding Control Points** (`05_human_in_the_loop.ipynb`)
- When and why to use HITL
- Implementing approval workflows
- Checkpointing and state persistence
- Production HITL patterns

**Estimated time:** 75 minutes
**Difficulty:** ⭐⭐⭐ Intermediate

---

### 🎓 Advanced Track (Notebooks 6-7)

**6. Production Deployment: Lambda + API Gateway** (`06_production_deployment.ipynb`)
- Complete AWS CDK deployment
- Lambda containerization
- API Gateway configuration
- Monitoring and logging setup

**Estimated time:** 120 minutes
**Difficulty:** ⭐⭐⭐⭐ Advanced

---

**7. Advanced Patterns: Streaming, Async, and Optimization** (`07_advanced_patterns.ipynb`)
- Response streaming for better UX
- Async execution for parallel operations
- Caching and optimization techniques
- Cost reduction strategies

**Estimated time:** 90 minutes
**Difficulty:** ⭐⭐⭐⭐ Advanced

---

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/aws-samples/langgraph-on-amazon-sagemaker.git
cd langgraph-on-amazon-sagemaker/tutorials

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Jupyter
pip install jupyter ipywidgets
```

### 2. Configure AWS Credentials

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-1"
```

### 3. Set API Keys

```bash
# For Tavily search (Notebooks 2+)
export TAVILY_API_KEY="tvly-your-key-here"

# Get free key from: https://app.tavily.com/
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Navigate to the notebook you want to start with (recommend starting with 01_first_agent.ipynb).

---

## Notebook Structure

Each notebook follows this structure:

### 1. **Learning Objectives**
Clear goals for what you'll learn

### 2. **Conceptual Explanation**
Theory with analogies for AI/ML scientists

### 3. **Code Examples**
Runnable code cells with detailed comments

### 4. **Exercises**
Hands-on tasks to reinforce learning

### 5. **Solutions**
Provided solutions with explanations

### 6. **Key Takeaways**
Summary of important concepts

### 7. **Next Steps**
What to learn next

---

## Notebook Summaries

### Notebook 1: LangGraph Basics

**What you'll build:**
A simple Q&A agent that answers questions using Claude/Mistral.

**Key concepts:**
- State dictionaries
- Graph nodes (functions)
- Graph edges (connections)
- StateGraph and compilation
- Streaming vs. invoke

**Code highlights:**
```python
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    input: str
    output: str

def agent_node(state):
    # Your agent logic
    return {"output": f"Answer to: {state['input']}"}

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()
result = app.invoke({"input": "Hello!"})
```

**What you'll learn:**
- How LangGraph represents workflows as graphs
- The difference between state, nodes, and edges
- How to run and debug agents

---

### Notebook 2: Adding Tools

**What you'll build:**
An agent that can search the web to answer current events questions.

**Key concepts:**
- Tools as external capabilities
- ReAct (Reasoning + Acting) pattern
- Tool selection and execution
- Error handling for tool failures

**Code highlights:**
```python
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Define tool
search_tool = TavilySearchResults(max_results=3)

# Create agent with tools
agent = create_xml_agent(llm, [search_tool], prompt)

# Agent decides when to use tools
result = agent.invoke({"input": "What happened in the news today?"})
```

**What you'll learn:**
- How agents decide which tools to use
- Debugging tool calls with verbose mode
- Creating custom tools for your needs

---

### Notebook 3: SageMaker Deployment

**What you'll build:**
Deploy Mistral-7B to a SageMaker endpoint and connect your agent.

**Key concepts:**
- SageMaker real-time inference
- Model deployment and endpoint configuration
- ContentHandler for request/response transformation
- Cost estimation and optimization

**Code highlights:**
```python
from sagemaker.huggingface import HuggingFaceModel

# Deploy model to SageMaker
model = HuggingFaceModel(
    model_data="s3://path/to/model",
    role=sagemaker_role,
    transformers_version="4.28",
    pytorch_version="2.0",
)

endpoint_name = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge"
)

# Connect agent
llm = SagemakerEndpoint(
    endpoint_name=endpoint_name,
    content_handler=MistralContentHandler()
)
```

**What you'll learn:**
- How to deploy LLMs to SageMaker
- Understanding instance types and costs
- Monitoring endpoint health
- Troubleshooting deployment issues

---

### Notebook 4: Multi-Agent Systems

**What you'll build:**
A three-agent system: Researcher finds facts, Writer drafts answer, Reviewer checks quality.

**Key concepts:**
- Agent specialization and collaboration
- Shared state across agents
- Conditional routing (if/else in graphs)
- Iteration and revision loops

**Code highlights:**
```python
# Define specialized agents
def researcher(state):
    facts = search_web(state["question"])
    return {"research": facts}

def writer(state):
    draft = llm.invoke(f"Write answer using: {state['research']}")
    return {"draft": draft}

def reviewer(state):
    if quality_check(state["draft"]):
        return {"approved": True}
    else:
        return {"needs_revision": True}

# Route based on review
graph.add_conditional_edges(
    "reviewer",
    lambda s: "done" if s["approved"] else "writer"  # Loop back if needed
)
```

**What you'll learn:**
- When to use multiple agents vs. one
- Designing agent collaboration patterns
- Preventing infinite loops with max iterations

---

### Notebook 5: Human-in-the-Loop

**What you'll build:**
An agent that pauses for human approval before taking actions.

**Key concepts:**
- Interrupts and checkpoints
- State persistence across pauses
- Approval workflows
- Production HITL architectures

**Code highlights:**
```python
from langgraph.checkpoint.memory import MemorySaver

# Create graph with checkpointing
checkpointer = MemorySaver()

app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute"]  # Pause before execution
)

# Phase 1: Run until interrupt
app.stream(initial_state, config)

# Human reviews and approves
app.update_state(config, {"approved": True})

# Phase 2: Resume execution
app.stream(None, config)
```

**What you'll learn:**
- When and why to add human checkpoints
- Implementing approval workflows
- Production deployment patterns (async, webhooks)

---

### Notebook 6: Production Deployment

**What you'll build:**
Complete production deployment with Lambda, API Gateway, CloudWatch.

**Key concepts:**
- AWS CDK infrastructure as code
- Lambda containerization
- API Gateway configuration
- CloudWatch monitoring
- Cost optimization

**Code highlights:**
```python
# CDK stack definition
class LangGraphStack(Stack):
    def __init__(self, scope, id):
        # SageMaker endpoint
        endpoint = sagemaker.CfnEndpoint(...)

        # Lambda function
        function = lambda_.DockerImageFunction(
            code=lambda_.DockerImageCode.from_image_asset("./agent"),
            environment={"ENDPOINT_NAME": endpoint.attr_endpoint_name}
        )

        # API Gateway
        api = apigateway.RestApi(self, "LangGraphAPI")
        api.root.add_method("POST", lambda_integration)

# Deploy
cdk deploy
```

**What you'll learn:**
- End-to-end production architecture
- Infrastructure as code with CDK
- Monitoring and alerting setup
- Security best practices

---

### Notebook 7: Advanced Patterns

**What you'll build:**
Optimized agent with streaming, caching, and async execution.

**Key concepts:**
- Response streaming for better UX
- Async/await for parallel operations
- Caching strategies
- Performance optimization
- Cost reduction techniques

**Code highlights:**
```python
# Streaming responses
for chunk in llm.stream(prompt):
    print(chunk, end='', flush=True)  # Show tokens as generated

# Async parallel execution
async def process_batch(questions):
    tasks = [agent_async.ainvoke(q) for q in questions]
    results = await asyncio.gather(*tasks)
    return results

# Caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_answer(question):
    return llm.invoke(question)  # Cached on repeated questions
```

**What you'll learn:**
- Improving user experience with streaming
- Parallel execution for throughput
- Caching for cost reduction
- A/B testing optimizations

---

## Exercises and Solutions

Each notebook includes:

**✏️ Guided Exercises:**
- Step-by-step tasks with hints
- Progressive difficulty
- Real-world scenarios

**✅ Solutions:**
- Complete working code
- Explanation of approach
- Alternative solutions discussed

**🎯 Challenge Exercises:**
- Open-ended problems
- Combine multiple concepts
- Encourage experimentation

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'langgraph'"

**Solution:**
```bash
pip install langgraph langchain langchain-community
```

### Issue: "SageMaker endpoint not found"

**Solution:**
- Check endpoint is deployed: `aws sagemaker list-endpoints`
- Verify region matches: `aws configure get region`
- Check IAM permissions for SageMaker access

### Issue: "TAVILY_API_KEY not set"

**Solution:**
```bash
export TAVILY_API_KEY="your-key"
# Or add to .env file
```

### Issue: High SageMaker costs

**Solution:**
- Delete endpoint when not in use: `aws sagemaker delete-endpoint --endpoint-name <name>`
- Use Serverless Inference for low traffic
- See Notebook 7 for cost optimization

---

## Learning Path

### For Beginners (No Agent Experience)

**Week 1:**
- Day 1-2: Notebook 1 (Basics)
- Day 3-4: Notebook 2 (Tools)
- Day 5: Review and experiment

**Week 2:**
- Day 1-2: Notebook 3 (SageMaker)
- Day 3-4: Notebook 4 (Multi-agent)
- Day 5: Build your own agent

**Week 3:**
- Day 1-2: Notebook 5 (HITL)
- Day 3-4: Notebook 6 (Production)
- Day 5: Deploy your agent

### For Experienced ML Engineers

**Week 1:**
- Day 1: Notebooks 1-2 (skim/review)
- Day 2: Notebook 3 (deploy SageMaker)
- Day 3: Notebook 4 (multi-agent patterns)
- Day 4: Notebook 5 (HITL)
- Day 5: Notebook 6-7 (production + optimization)

---

## Additional Resources

### Documentation
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangChain Docs](https://python.langchain.com/)
- [AWS SageMaker Docs](https://docs.aws.amazon.com/sagemaker/)

### Community
- [LangChain Discord](https://discord.gg/langchain)
- [AWS ML Community](https://repost.aws/tags/AWS-Machine-Learning)

### Related Tutorials
- `../examples/` - More advanced examples
- `../deployment/` - Production deployment guides
- `../monitoring/` - Observability setup

---

## Contributing

Found an issue or have improvements?

1. Open an issue on GitHub
2. Submit a pull request
3. Share your own notebook variations

---

## Next Steps After Completing Tutorials

1. **Build Your Own Agent**
   - Choose a use case relevant to your work
   - Start with Notebook 1 template
   - Add tools specific to your domain

2. **Deploy to Production**
   - Use Notebook 6 as template
   - Add monitoring (see `/monitoring/`)
   - Optimize costs (see `/guides/performance-optimization.md`)

3. **Join Community**
   - Share what you built
   - Help others learn
   - Contribute examples

Happy Learning! 🚀
