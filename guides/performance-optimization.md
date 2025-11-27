# Performance Tuning & Cost Optimization Guide

## Overview

This guide provides actionable strategies for optimizing LangGraph agents on SageMaker for both performance (latency/throughput) and cost.

**For AI/ML Scientists:**
This is like hyperparameter tuning and model compression for production ML systems. We're optimizing for:
- **Latency**: Time per inference (like reducing forward pass time)
- **Throughput**: Inferences per second (like batch size optimization)
- **Cost**: $ per 1000 inferences (like choosing GPU vs CPU)

---

## Quick Wins (Implement These First)

### 1. Use Smaller LLM for Simple Tasks (70% cost reduction)

**Problem:** Using Mistral-7B for everything, even simple tasks

**Solution:** Route simple tasks to smaller/cheaper models

```python
def select_model(question: str) -> str:
    """Route to appropriate model based on complexity"""

    # Simple questions → small model
    if len(question.split()) < 10 and "?" in question:
        return "mistral-3b"  # 50% cheaper, 2x faster

    # Complex questions → larger model
    else:
        return "mistral-7b"
```

**Savings:**
- Cost: 70% reduction (if 70% of questions are simple)
- Latency: 2x faster for simple questions

### 2. Cache Common Queries (90% cost reduction for repeated queries)

**Problem:** Re-running LLM for identical questions

**Solution:** Cache responses

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_answer_cached(question: str) -> str:
    """Cache answers to common questions"""
    return llm.invoke(question)
```

**Savings:**
- Cost: $0 for cache hits (vs $0.001 per LLM call)
- Latency: <1ms (vs 2000ms for LLM call)

### 3. Reduce Prompt Size (30% cost reduction)

**Problem:** Prompts with unnecessary verbosity

**Solution:** Optimize prompts

```python
# ❌ Verbose prompt (500 tokens)
prompt = """You are a helpful AI assistant. Your job is to answer
questions accurately and concisely. Please review the following
context carefully and provide a detailed answer...

Context: {context}
Question: {question}
...more instructions..."""

# ✅ Concise prompt (150 tokens)
prompt = """Answer based on context.

Context: {context}
Question: {question}

Answer:"""
```

**Savings:**
- Cost: 30% reduction in token costs
- Latency: 20% faster generation

### 4. Use Async for I/O-Bound Operations (3x throughput)

**Problem:** Waiting for APIs/databases sequentially

**Solution:** Parallel execution

```python
import asyncio

# ❌ Sequential (slow)
weather = get_weather("Paris")    # 500ms
stock = get_stock("AAPL")         # 500ms
# Total: 1000ms

# ✅ Parallel (fast)
async def get_all_data():
    weather, stock = await asyncio.gather(
        get_weather_async("Paris"),    # 500ms \
        get_stock_async("AAPL")        # 500ms  } parallel
    )                                  # Total: 500ms
    return weather, stock
```

**Savings:**
- Latency: 50% reduction
- Throughput: 2x increase

### 5. Terminate Lambda After Response (80% compute cost reduction)

**Problem:** Lambda waits for full timeout even after returning

**Solution:** Use response streaming or async invocation

```python
# In Lambda handler
import asyncio

def handler(event, context):
    # Process request
    result = process_request(event)

    # Return immediately
    response = {
        'statusCode': 200,
        'body': json.dumps(result)
    }

    # Don't wait for cleanup
    context.callbackWaitsForEmptyEventLoop = False

    return response
```

**Savings:**
- Compute cost: 80% reduction (5s billable vs 30s)

---

## Latency Optimization

### Understand Your Latency

**For AI/ML Scientists:**
Like profiling training time - first measure, then optimize.

**Typical latency breakdown:**
```
Total: 6200ms
├── LLM call: 4800ms (77%)    ← Biggest bottleneck
├── Tool calls: 1100ms (18%)
│   ├── Web search: 600ms
│   ├── Database: 400ms
│   └── Other: 100ms
└── Overhead: 300ms (5%)
    ├── Parsing: 150ms
    ├── State management: 100ms
    └── Logging: 50ms
```

**Optimization priority:** Focus on the 77% (LLM) first!

### 1. LLM Latency Optimization

#### A. Reduce max_new_tokens

```python
# ❌ Slow: Allow long responses
model_kwargs = {"max_new_tokens": 1000}  # Up to 1000 tokens @ 50 tokens/s = 20s!

# ✅ Fast: Limit response length
model_kwargs = {"max_new_tokens": 200}   # Max 200 tokens @ 50 tokens/s = 4s
```

**For AI/ML Scientists:**
Like early stopping in training - we stop generation once we have enough.

**Impact:**
- Latency: 5x faster (20s → 4s) for max-length responses
- Cost: 5x cheaper (fewer tokens generated)

#### B. Use Smaller Models for Subtasks

```python
# Researcher uses large model (needs accuracy)
researcher_llm = create_sagemaker_llm(endpoint="mistral-7b")

# Reviewer uses small model (simpler task)
reviewer_llm = create_sagemaker_llm(endpoint="mistral-3b")
```

**Impact:**
- Latency: 2x faster for reviewer
- Cost: 50% cheaper for reviewer

#### C. Batch Requests (if applicable)

```python
# If processing multiple independent questions
questions = ["Q1", "Q2", "Q3"]

# ❌ Sequential
answers = [llm.invoke(q) for q in questions]  # 3 x 2s = 6s

# ✅ Batch (if model supports)
answers = llm.batch(questions)  # ~3s (depends on model)
```

#### D. Use Streaming for Perceived Performance

```python
# Stream tokens as they're generated
# User sees response starting immediately instead of waiting 2s

for chunk in llm.stream(prompt):
    print(chunk, end='', flush=True)  # Display in real-time

# Actual latency: Same (2s)
# Perceived latency: Much better (user sees output at 0.1s)
```

### 2. Tool Call Optimization

#### A. Parallel Tool Execution

```python
# ❌ Sequential
weather = weather_tool.invoke("Paris")    # 600ms
stock = stock_tool.invoke("AAPL")        # 500ms
# Total: 1100ms

# ✅ Parallel
import asyncio
weather, stock = await asyncio.gather(
    weather_tool_async.invoke("Paris"),
    stock_tool_async.invoke("AAPL")
)
# Total: 600ms (max of the two)
```

#### B. Tool Result Caching

```python
from functools import lru_cache
import time

@lru_cache(maxsize=100)
def get_weather_cached(location: str, timestamp_hour: int):
    """Cache weather for 1 hour"""
    return weather_api.get(location)

# Usage
current_hour = int(time.time() / 3600)
weather = get_weather_cached("Paris", current_hour)
```

#### C. Faster APIs

```python
# ❌ Slow external API (600ms)
result = requests.get("https://slow-api.com/data")

# ✅ Fast alternative or self-hosted (100ms)
result = requests.get("https://your-cdn.com/cached-data")
```

### 3. State Management Optimization

#### A. Minimize State Size

```python
# ❌ Large state (slow to serialize/deserialize)
state = {
    "input": question,
    "full_context": huge_context,  # 100KB!
    "history": all_messages,       # 50KB!
    ...
}

# ✅ Compact state
state = {
    "input": question,
    "context_id": context_id,  # Reference, not full data
    "last_3_messages": recent,  # Only what's needed
    ...
}
```

#### B. Use References for Large Data

```python
# Store large data externally
s3_key = save_to_s3(large_context)

state = {
    "context_ref": s3_key,  # Just reference
    ...
}
```

---

## Cost Optimization

### Understand Your Costs

**For AI/ML Scientists:**
Like tracking GPU hours - know where money is spent.

**Typical cost breakdown (1000 questions/day):**
```
Total: $23.50/day = $705/month

├── SageMaker Endpoint: $23.20/day (99%)  ← Biggest cost
│   ml.g5.xlarge @ $1.006/hour x 24h = $24.14/day
│   Minus: Savings Plans discount = $23.20/day
│
├── Lambda: $0.20/day (1%)
│   1000 invocations x 5s avg x $0.0000166667/GB-second
│
├── API Gateway: $0.004/day (<1%)
│   1000 requests x $0.000004
│
└── Tool APIs: $0.10/day (<1%)
    Tavily: 1000 searches x $0.0001
```

**Optimization priority:** Focus on SageMaker (99%)!

### 1. SageMaker Endpoint Optimization

#### A. Use SageMaker Serverless (For Low Traffic)

**When:** <100 requests/hour

```python
# Switch from real-time endpoint to serverless
endpoint_config = {
    "ServerlessConfig": {
        "MemorySizeInMB": 4096,
        "MaxConcurrency": 10
    }
}

# Pricing
# Real-time: $23/day (always running)
# Serverless: $0.20/day for 100 req/day
#
# Savings: 99% for low traffic!
```

#### B. Use Auto-Scaling (Variable Traffic)

```python
# Scale down during low traffic
auto_scaling_config = {
    "MinInstanceCount": 1,     # Night/weekend
    "MaxInstanceCount": 5,     # Peak hours
    "TargetValue": 70.0,       # CPU utilization
}

# Savings: 60% reduction (average 2 instances vs 5)
```

#### C. Use Savings Plans or Reserved Instances

- **On-Demand:** $1.006/hour = $723/month
- **1-year Savings Plan:** $0.63/hour = $453/month (37% savings)
- **3-year Savings Plan:** $0.42/hour = $302/month (58% savings)

#### D. Use Smaller Instances Where Possible

```python
# Test if smaller instance handles your load

# ml.g5.xlarge: $1.006/hour, 4 vCPU, 24GB RAM, 1 GPU
# ml.g5.large:  $0.503/hour, 2 vCPU, 8GB RAM, 1 GPU (50% cheaper!)
# ml.g4dn.xlarge: $0.736/hour, 4 vCPU, 16GB RAM, 1 GPU (27% cheaper)

# If throughput is acceptable, use smaller instance
```

**For AI/ML Scientists:**
Like choosing GPU type - do you need A100 or will T4 suffice?

#### E. Use Spot Instances (For Batch Processing)

**Not for real-time APIs**, but for batch processing:

- On-demand: $1.006/hour
- Spot: ~$0.30/hour (70% savings)
- Trade-off: Can be interrupted

### 2. Lambda Optimization

#### A. Right-Size Memory

```python
# Test different memory sizes
# Higher memory = More CPU (faster) but more expensive

# 512MB: $0.0000083333/GB-s × 10s = $0.000042 per invoke
# 1024MB: $0.0000166667/GB-s × 5s = $0.000042 per invoke (same cost, 2x faster!)
# 2048MB: $0.0000333333/GB-s × 3s = $0.000050 per invoke (faster but pricier)

# Sweet spot: Usually 1024-1536MB
```

**For AI/ML Scientists:**
Like choosing batch size - balance speed vs. memory vs. cost.

#### B. Use Lambda SnapStart (For Java/Python 3.9+)

- Reduces cold start from 1-2s to <100ms
- Free feature
- Improves p99 latency significantly

### 3. Tool Cost Optimization

#### A. Cache API Results

```python
# Tavily search: $0.0001/search
# If 50% of searches are duplicates → 50% savings

@cache_for_hours(24)
def search_web(query):
    return tavily.search(query)
```

#### B. Use Free Tiers

- Tavily: 1000 searches/month free
- WeatherAPI: 1M calls/month free
- Ensure you're not paying for what's free!

#### C. Rate Limit Tool Usage

```python
# Prevent runaway tool usage
MAX_TOOLS_PER_REQUEST = 5

if tool_count >= MAX_TOOLS_PER_REQUEST:
    return "Tool limit reached"
```

---

## Throughput Optimization

**For AI/ML Scientists:**
Like increasing training batch size - how many samples/second can we process?

### 1. Increase SageMaker Endpoint Capacity

```python
# Add more instances
endpoint_config = {
    "InitialInstanceCount": 5,  # Instead of 1
}

# Or use larger instances
endpoint_config = {
    "InstanceType": "ml.g5.2xlarge",  # 2x capacity of g5.xlarge
}
```

**Cost vs. Throughput:**
- 1x ml.g5.xlarge: ~10 req/s, $723/month
- 5x ml.g5.xlarge: ~50 req/s, $3,615/month
- 1x ml.g5.12xlarge: ~100 req/s, $8,676/month

### 2. Use Async Processing

```python
# For non-real-time use cases
# User submits request → Gets request_id → Polls for result

# Allows request queuing and batching
# Much higher throughput with same infrastructure
```

### 3. Optimize Graph Execution

```python
# ❌ Sequential node execution
graph.add_edge("node1", "node2")
graph.add_edge("node2", "node3")
# Total: 6s (2s + 2s + 2s)

# ✅ Parallel where possible
graph.add_edge("start", "node1")
graph.add_edge("start", "node2")  # Runs in parallel with node1!
graph.add_edge(["node1", "node2"], "node3")
# Total: 4s (2s parallel + 2s)
```

---

## Monitoring-Driven Optimization

**For AI/ML Scientists:**
Like using TensorBoard to find bottlenecks during training.

### 1. Identify Bottlenecks

```python
from monitoring import LatencyAnalyzer

analyzer = LatencyAnalyzer("my-agent")
breakdown = analyzer.analyze(limit=1000)

# Output:
# LLM: 78% of latency → Optimize LLM first
# Tools: 15% → Optimize tools second
# Overhead: 7% → Optimize last
```

### 2. A/B Test Optimizations

```python
# Test optimization with 10% of traffic
if random.random() < 0.1:
    # Optimized version
    result = fast_llm.invoke(prompt)
    monitor.record_metric("version", "optimized")
else:
    # Original version
    result = llm.invoke(prompt)
    monitor.record_metric("version", "original")

# Compare metrics in CloudWatch
```

### 3. Set Performance Budgets

```python
# Alert if latency exceeds target
LATENCY_BUDGET_MS = 3000

@monitor.track_invocation
def run_agent(question):
    start = time.time()
    result = agent.invoke(question)
    latency = (time.time() - start) * 1000

    if latency > LATENCY_BUDGET_MS:
        monitor.alert(f"Latency budget exceeded: {latency}ms")

    return result
```

---

## Optimization Checklist

### ✅ Quick Wins (Do These First)
- [ ] Cache common queries (lru_cache)
- [ ] Reduce prompt verbosity
- [ ] Set appropriate max_new_tokens
- [ ] Use async for I/O operations
- [ ] Enable Lambda response streaming

### ✅ Cost Reduction
- [ ] Right-size SageMaker instance
- [ ] Consider Serverless for low traffic
- [ ] Use Savings Plans for predictable load
- [ ] Cache tool API results
- [ ] Optimize Lambda memory allocation

### ✅ Latency Reduction
- [ ] Parallelize tool calls
- [ ] Use smaller models for simple tasks
- [ ] Reduce state size
- [ ] Enable X-Ray tracing to find bottlenecks
- [ ] Optimize graph execution paths

### ✅ Monitoring
- [ ] Set up CloudWatch dashboards
- [ ] Create latency/error alarms
- [ ] Track cost metrics
- [ ] Monitor tool usage patterns
- [ ] Regular performance reviews

---

## Real-World Examples

### Example 1: E-commerce Q&A Agent

**Before optimization:**
- Latency: 8s (p99: 15s)
- Cost: $1,200/month
- Throughput: 5 req/s

**Optimizations applied:**
1. Cached product info (80% hit rate)
2. Used Mistral-3B for simple questions (60% of traffic)
3. Reduced prompt from 800 to 300 tokens
4. Parallel tool calls for inventory + pricing

**After optimization:**
- Latency: 2s (p99: 4s) - **75% improvement**
- Cost: $350/month - **71% reduction**
- Throughput: 15 req/s - **3x increase**

### Example 2: Customer Support Agent

**Before optimization:**
- Cost: $2,500/month (ml.g5.2xlarge always-on)
- Traffic: 200 req/day (highly variable)

**Optimizations applied:**
1. Switched to SageMaker Serverless
2. Cached common support questions

**After optimization:**
- Cost: $85/month - **97% reduction**
- Same quality, slightly higher cold-start latency (acceptable for support)

---

## Tools & Resources

### Profiling Tools

```bash
# AWS X-Ray for distributed tracing
aws xray get-trace-summaries --start-time <start> --end-time <end>

# CloudWatch Insights for log analysis
aws logs start-query --log-group-name "/aws/lambda/my-function" \
  --query-string "fields @timestamp, @message | filter latency > 5000"
```

### Cost Analysis Tools

```bash
# AWS Cost Explorer
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY --metrics BlendedCost \
  --group-by Type=SERVICE

# SageMaker endpoint utilization
aws sagemaker describe-endpoint --endpoint-name my-endpoint
```

---

## Next Steps

1. **Baseline**: Measure current performance and costs
2. **Quick wins**: Implement caching and prompt optimization
3. **Monitor**: Set up dashboards and track improvements
4. **Iterate**: Use data to find next bottleneck
5. **Document**: Record what works for your use case

Remember: **Measure, optimize, verify, repeat.**
