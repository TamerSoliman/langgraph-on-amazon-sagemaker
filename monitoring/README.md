# Monitoring & Observability for LangGraph Agents

## Overview

This package provides production-ready monitoring and observability for LangGraph agents running on AWS.

**For AI/ML Scientists:**
Monitoring agents is like monitoring model performance in production. You need to track:
- **Latency**: How long does inference take?
- **Throughput**: How many requests/second?
- **Errors**: What's failing and why?
- **Cost**: How much are we spending?
- **Quality**: Are outputs good?

---

## Metrics Tracked

### 1. **Invocation Metrics**
- Total invocations
- Invocations per minute/hour/day
- Success vs. error rate
- Average latency (p50, p90, p99)

### 2. **LLM Metrics**
- LLM calls per invocation
- Tokens consumed (input + output)
- LLM latency
- LLM errors

### 3. **Tool Metrics**
- Tool call frequency (which tools are used most)
- Tool success/failure rates
- Tool latency
- Tool-specific errors

### 4. **Cost Metrics**
- Cost per invocation
- Cost per hour/day/month
- Cost breakdown (LLM vs. compute vs. tools)

### 5. **Quality Metrics**
- User feedback (thumbs up/down)
- Response length
- Revision rate (for multi-agent systems)

---

## Quick Start

### 1. Install Dependencies

```bash
cd monitoring/
pip install -r requirements.txt
```

### 2. Add Instrumentation to Your Agent

```python
from monitoring import AgentMonitor

# Create monitor
monitor = AgentMonitor(
    namespace="LangGraphAgents",
    agent_name="my-agent"
)

# Wrap your agent execution
@monitor.track_invocation
def run_agent(question: str):
    app = create_agent_graph()
    result = app.invoke({"input": question})
    return result
```

### 3. View Metrics in CloudWatch

```bash
# AWS Console
# CloudWatch → Dashboards → "LangGraphAgents-my-agent"

# Or via CLI
aws cloudwatch get-dashboard \
  --dashboard-name "LangGraphAgents-my-agent"
```

---

## Architecture

### Monitoring Stack

```
Agent Execution
    ↓
Instrumentation Layer (Python decorators)
    ↓
CloudWatch Metrics + Logs
    ↓
CloudWatch Dashboards + Alarms
    ↓
SNS Notifications (email, Slack, PagerDuty)
```

**For AI/ML Scientists:**
This is similar to ML monitoring tools like Weights & Biases or MLflow:
- Collect metrics during execution
- Aggregate and visualize
- Alert on anomalies

---

## Instrumentation Guide

### Basic Instrumentation

```python
from monitoring import AgentMonitor

monitor = AgentMonitor(
    namespace="MyApp",
    agent_name="question-answering-agent"
)

# Track entire invocation
@monitor.track_invocation
def answer_question(question: str) -> str:
    # Your agent code here
    return answer

# Track individual LLM calls
@monitor.track_llm_call
def call_llm(prompt: str) -> str:
    llm = create_sagemaker_llm()
    response = llm.invoke(prompt)
    return response

# Track tool usage
@monitor.track_tool_call(tool_name="web_search")
def search_web(query: str) -> str:
    # Search logic
    return results
```

### Advanced Instrumentation

```python
# Manual metric tracking
with monitor.timer("custom_operation"):
    # Time-consuming operation
    result = expensive_computation()

# Track custom metrics
monitor.record_metric("cache_hit_rate", 0.85)
monitor.record_metric("retrieval_relevance", 0.92)

# Track errors with context
try:
    result = risky_operation()
except Exception as e:
    monitor.record_error(
        error_type="OperationFailed",
        error_message=str(e),
        context={"operation": "risky_operation"}
    )
    raise
```

---

## CloudWatch Dashboard

### Auto-Generated Dashboard

The monitor automatically creates a CloudWatch dashboard with:

1. **Overview Panel**
   - Total invocations (last hour/day)
   - Success rate
   - Average latency
   - Error count

2. **Latency Panel**
   - P50/P90/P99 latency over time
   - Latency by component (LLM, tools, total)

3. **Error Panel**
   - Error rate over time
   - Errors by type
   - Recent error logs

4. **Cost Panel**
   - Estimated cost per hour
   - Cost breakdown by component
   - Cost trend

5. **Tool Usage Panel**
   - Tool call frequency
   - Tool success rates
   - Tool latency

### Example Dashboard JSON

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["LangGraphAgents", "Invocations", {"stat": "Sum"}],
          [".", "Errors", {"stat": "Sum"}]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1",
        "title": "Invocations & Errors"
      }
    }
  ]
}
```

---

## Alerting

### Pre-Configured Alarms

```python
from monitoring import create_alarms

# Create standard alarms
alarms = create_alarms(
    agent_name="my-agent",
    sns_topic_arn="arn:aws:sns:us-east-1:123456789012:alerts",

    # Thresholds
    error_rate_threshold=0.05,      # 5% error rate
    latency_p99_threshold=10000,    # 10 seconds
    invocation_spike_threshold=2.0   # 2x normal rate
)
```

### Alarm Types

1. **High Error Rate**
   - Trigger: Error rate > 5% for 5 minutes
   - Action: Send SNS notification

2. **High Latency**
   - Trigger: P99 latency > 10 seconds for 5 minutes
   - Action: Send SNS notification

3. **No Invocations**
   - Trigger: No invocations for 30 minutes
   - Action: Send SNS notification (possible outage)

4. **Cost Spike**
   - Trigger: Hourly cost > 2x average
   - Action: Send SNS notification

---

## Logging

### Structured Logging

```python
from monitoring import setup_logging

# Configure structured logging
logger = setup_logging(
    log_level="INFO",
    include_trace_id=True
)

# Log with structured data
logger.info(
    "Agent invocation started",
    extra={
        "question": question,
        "user_id": user_id,
        "request_id": request_id
    }
)

# Logs are automatically sent to CloudWatch Logs
```

### Log Insights Queries

**Find slow requests:**
```sql
fields @timestamp, @message, latency_ms
| filter latency_ms > 5000
| sort latency_ms desc
| limit 20
```

**Find errors:**
```sql
fields @timestamp, @message, error_type
| filter level = "ERROR"
| stats count() by error_type
```

**Track tool usage:**
```sql
fields @timestamp, tool_name
| filter tool_name != ""
| stats count() by tool_name
```

---

## Distributed Tracing

### X-Ray Integration

**For AI/ML Scientists:**
Distributed tracing is like profiling your model - it shows where time is spent in each request.

```python
from monitoring import enable_xray_tracing
from aws_xray_sdk.core import xray_recorder

# Enable X-Ray
enable_xray_tracing()

# Trace subsegments automatically
@xray_recorder.capture("llm_call")
def call_llm(prompt):
    llm = create_sagemaker_llm()
    return llm.invoke(prompt)

# Trace shows:
# Request → Planner (2s) → LLM Call (4s) → Tool Call (1s) → Response
#           └─ breakdown of where time is spent
```

### Service Map

X-Ray automatically generates a service map showing:
- API Gateway → Lambda → SageMaker → Tavily API
- Latency for each hop
- Error rates between services

---

## Cost Tracking

### Real-Time Cost Estimation

```python
from monitoring import CostTracker

cost_tracker = CostTracker()

# Track costs automatically
@cost_tracker.track
def run_agent(question):
    # Automatically tracks:
    # - SageMaker inference costs
    # - Lambda compute costs
    # - API call costs (Tavily, etc.)
    result = app.invoke({"input": question})
    return result

# Get cost breakdown
costs = cost_tracker.get_costs(time_period="1h")
print(f"LLM: ${costs['llm']:.4f}")
print(f"Compute: ${costs['compute']:.4f}")
print(f"Tools: ${costs['tools']:.4f}")
print(f"Total: ${costs['total']:.4f}")
```

### Cost Optimization Alerts

```python
# Alert if daily cost exceeds budget
create_cost_alarm(
    daily_budget=50.00,  # $50/day
    sns_topic_arn="arn:aws:sns:..."
)
```

---

## Performance Analysis

### Latency Breakdown

```python
from monitoring import LatencyAnalyzer

analyzer = LatencyAnalyzer(agent_name="my-agent")

# Get latency breakdown for last 1000 requests
breakdown = analyzer.analyze(limit=1000)

print(f"Average total latency: {breakdown['total_avg']}ms")
print(f"  LLM calls: {breakdown['llm_avg']}ms ({breakdown['llm_pct']}%)")
print(f"  Tool calls: {breakdown['tools_avg']}ms ({breakdown['tools_pct']}%)")
print(f"  Overhead: {breakdown['overhead_avg']}ms ({breakdown['overhead_pct']}%)")
```

**Example output:**
```
Average total latency: 6200ms
  LLM calls: 4800ms (77%)
  Tool calls: 1100ms (18%)
  Overhead: 300ms (5%)

Recommendation: LLM calls dominate latency. Consider:
1. Reducing prompt size
2. Using smaller model
3. Caching common requests
```

---

## Quality Monitoring

### User Feedback Tracking

```python
from monitoring import QualityMonitor

quality = QualityMonitor()

# Track user feedback
@quality.track_feedback
def handle_feedback(request_id: str, feedback: str):
    # feedback: "positive" or "negative"
    quality.record_feedback(request_id, feedback)

# Get quality metrics
metrics = quality.get_metrics(period="7d")
print(f"Satisfaction rate: {metrics['satisfaction_rate']:.1%}")
print(f"Total feedback: {metrics['total_feedback']}")
```

### Output Quality Metrics

```python
# Track response length (detect truncation)
quality.record_metric("response_length", len(response))

# Track revision rate (multi-agent)
quality.record_metric("revision_rate", revisions / total_drafts)

# Track tool usage effectiveness
quality.record_metric("tool_success_rate", successes / total_calls)
```

---

## Production Checklist

### ✅ Before Deploying

- [ ] Instrumentation added to all critical paths
- [ ] CloudWatch dashboard created
- [ ] Alarms configured with SNS topics
- [ ] Log retention set (7-30 days recommended)
- [ ] X-Ray tracing enabled
- [ ] Cost tracking configured
- [ ] Error handling with proper logging
- [ ] Performance baselines established

### 📊 Monitoring After Deploy

**First 24 hours:**
- Check error rates every 2 hours
- Review latency percentiles
- Validate cost estimates
- Check alarm sensitivity (too noisy?)

**First week:**
- Analyze usage patterns
- Identify optimization opportunities
- Tune alarm thresholds
- Review log insights

**Ongoing:**
- Weekly cost review
- Monthly performance analysis
- Quarterly capacity planning

---

## Troubleshooting

### High Latency

1. **Check latency breakdown** - Where is time spent?
   ```python
   analyzer.analyze() # Shows LLM vs tools vs overhead
   ```

2. **Review CloudWatch Logs** - Any slow operations?
   ```sql
   fields @timestamp, operation, duration
   | filter duration > 5000
   | sort duration desc
   ```

3. **Check X-Ray traces** - Find bottlenecks
   - Look for long subsegments
   - Check for sequential operations that could be parallel

### High Error Rate

1. **Check error types**
   ```sql
   fields @timestamp, error_type
   | filter level = "ERROR"
   | stats count() by error_type
   ```

2. **Review recent errors**
   ```python
   monitor.get_recent_errors(count=10)
   ```

3. **Check dependencies** - SageMaker endpoint healthy? APIs responding?

### High Costs

1. **Cost breakdown**
   ```python
   costs = cost_tracker.get_breakdown(period="1d")
   # Identify most expensive component
   ```

2. **Check invocation volume** - Unexpected spike?
   ```sql
   fields @timestamp
   | stats count() by bin(5m)
   ```

3. **Review token usage** - Prompts too long?
   ```python
   monitor.get_token_stats()
   ```

---

## Example: Complete Instrumentation

```python
from monitoring import (
    AgentMonitor,
    setup_logging,
    enable_xray_tracing,
    create_alarms
)

# Setup
monitor = AgentMonitor(namespace="MyApp", agent_name="qa-agent")
logger = setup_logging(log_level="INFO")
enable_xray_tracing()

# Create alarms
create_alarms(
    agent_name="qa-agent",
    sns_topic_arn="arn:aws:sns:us-east-1:123456789012:alerts"
)

# Instrumented agent
@monitor.track_invocation
def answer_question(question: str) -> str:
    logger.info("Processing question", extra={"question": question})

    try:
        # Create agent
        app = create_agent_graph()

        # Execute with monitoring
        with monitor.timer("agent_execution"):
            result = app.invoke({"input": question})

        logger.info("Question answered successfully")
        return result["output"]

    except Exception as e:
        logger.error("Error answering question", exc_info=True)
        monitor.record_error(
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise

# Usage
answer = answer_question("What is LangGraph?")
```

---

## Next Steps

1. **Install** the monitoring package
2. **Instrument** your agent with basic metrics
3. **Create** CloudWatch dashboard
4. **Configure** alarms
5. **Monitor** for first week
6. **Tune** thresholds based on actual usage
7. **Optimize** based on insights

For more details, see individual files in this directory.
