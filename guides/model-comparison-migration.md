# Model Comparison & Migration Guide

## Overview

This guide helps you choose the right LLM for your LangGraph agent and migrate between models.

**For AI/ML Scientists:**
Choosing an LLM is like choosing a model architecture - balance accuracy, speed, and cost for your use case.

---

## Quick Comparison Table

| Model | Size | Speed | Cost/1K | Quality | Best For |
|-------|------|-------|---------|---------|----------|
| **Mistral 7B** | 7B | Fast | $$ | Good | General purpose, production |
| **Mistral 8x7B (Mixtral)** | 47B | Medium | $$$ | Excellent | Complex tasks, high quality |
| **LLaMA 2 7B** | 7B | Fast | $ | Good | Cost-sensitive, simple tasks |
| **LLaMA 2 13B** | 13B | Medium | $$ | Better | Balance of quality and cost |
| **LLaMA 2 70B** | 70B | Slow | $$$$ | Excellent | Highest quality needed |
| **Claude 3 Haiku** | ? | Very Fast | $ | Good | Speed-critical, simple tasks |
| **Claude 3 Sonnet** | ? | Fast | $$$ | Excellent | Production, high quality |
| **GPT-3.5 Turbo** | ? | Very Fast | $ | Good | Speed, cost optimization |
| **GPT-4** | ? | Slow | $$$$ | Excellent | Highest quality, complex reasoning |

**Cost Key:**
- $ = <$0.001/1K tokens
- $$ = $0.001-0.005/1K tokens
- $$$ = $0.005-0.02/1K tokens
- $$$$ = >$0.02/1K tokens

---

## Detailed Model Comparison

### Open-Source Models (SageMaker Deployment)

#### Mistral 7B Instruct

**Specs:**
- Parameters: 7 billion
- Context length: 8K tokens
- Inference speed: ~50 tokens/second (ml.g5.xlarge)

**Strengths:**
- ✅ Excellent quality for size
- ✅ Fast inference
- ✅ Good instruction following
- ✅ Multilingual (English, French, German, Spanish, Italian)

**Weaknesses:**
- ❌ Not as capable as larger models for very complex tasks
- ❌ Occasional XML tag issues (can generate malformed tool calls)

**Best for:**
- General-purpose Q&A
- Customer support chatbots
- Document summarization
- Code generation (basic)

**Cost (SageMaker ml.g5.xlarge):**
- Hosting: $1.006/hour = $723/month (24/7)
- Per request: ~$0.001 (assuming 2s latency)

**Example use case:**
"Answer customer questions about products" - high volume, need fast responses, quality is important but not critical.

---

#### Mixtral 8x7B (Mistral's MoE)

**Specs:**
- Parameters: 47B total (8 experts × 7B, activates 2 per token)
- Context length: 32K tokens
- Inference speed: ~30 tokens/second (ml.g5.xlarge)

**Strengths:**
- ✅ GPT-3.5 Turbo level quality
- ✅ Larger context window (32K vs 8K)
- ✅ Better reasoning than Mistral 7B
- ✅ Good for complex multi-step tasks

**Weaknesses:**
- ❌ Slower than Mistral 7B
- ❌ Higher memory requirements
- ❌ More expensive to run

**Best for:**
- Complex analysis tasks
- Long document processing
- Multi-turn conversations with memory
- Advanced reasoning

**Cost (SageMaker ml.g5.2xlarge):**
- Hosting: $1.515/hour = $1,090/month
- Per request: ~$0.002

**Example use case:**
"Analyze legal contracts and summarize key terms" - needs deep understanding, longer context, quality critical.

---

#### LLaMA 2 Models (7B, 13B, 70B)

**Specs:**
- 7B: Similar to Mistral 7B
- 13B: Better quality, slower
- 70B: Best open-source quality, very slow

**Strengths:**
- ✅ Free and open-source (no licensing fees)
- ✅ Well-documented and supported
- ✅ Good community fine-tunes available

**Weaknesses:**
- ❌ Mistral generally better at same size
- ❌ Less instruction-following capability
- ❌ Requires more prompt engineering

**Best for:**
- Cost-sensitive deployments
- Fine-tuning for specific domains
- Research and experimentation

**Cost:**
- 7B on ml.g5.xlarge: $723/month
- 13B on ml.g5.xlarge: $723/month (might need g5.2xlarge)
- 70B on ml.g5.12xlarge: $8,676/month

---

### Hosted API Models (No SageMaker Deployment)

#### Claude 3 (Anthropic via Bedrock)

**Haiku:**
- Fastest Claude model
- $0.00025/1K input tokens, $0.00125/1K output
- Best for: High-volume, speed-critical

**Sonnet:**
- Balanced speed and intelligence
- $0.003/1K input, $0.015/1K output
- Best for: Production applications

**Opus:**
- Most capable Claude model
- $0.015/1K input, $0.075/1K output
- Best for: Complex tasks requiring highest quality

**Advantages over SageMaker:**
- ✅ No infrastructure management
- ✅ Pay-per-use (no idle costs)
- ✅ Automatic scaling
- ✅ Very fast for low-traffic (<100 req/hour)

**Disadvantages:**
- ❌ More expensive for high traffic
- ❌ No model customization
- ❌ API rate limits

---

#### GPT Models (OpenAI)

**GPT-3.5 Turbo:**
- $0.0005/1K input, $0.0015/1K output
- Fast, cheap, good for simple tasks

**GPT-4:**
- $0.03/1K input, $0.06/1K output
- Best-in-class quality, slow, expensive

**GPT-4 Turbo:**
- $0.01/1K input, $0.03/1K output
- Faster than GPT-4, cheaper, good balance

---

## Decision Framework

### 1. Determine Your Requirements

**Quality Tier:**
- **Critical quality** (legal, medical, safety): GPT-4, Claude Opus, Mixtral 8x7B
- **High quality** (customer-facing): Mistral 7B, Claude Sonnet, GPT-3.5
- **Good enough** (internal tools): LLaMA 2 7B, Claude Haiku

**Latency Requirements:**
- **<500ms**: Claude Haiku, GPT-3.5, caching required
- **<2s**: Mistral 7B on SageMaker
- **<5s**: Mixtral, LLaMA 13B
- **>5s acceptable**: LLaMA 70B, GPT-4

**Traffic Volume:**
- **<100 req/hour**: Use hosted APIs (Claude, GPT)
- **100-1000 req/hour**: SageMaker with Mistral 7B
- **>1000 req/hour**: SageMaker with auto-scaling

**Cost Budget:**
- **<$100/month**: Hosted APIs or SageMaker Serverless
- **$100-500/month**: SageMaker ml.g5.xlarge (Mistral 7B)
- **$500-2000/month**: SageMaker ml.g5.2xlarge or multiple instances
- **>$2000/month**: Larger instances or GPT-4

### 2. Use This Decision Tree

```
START
  │
  ├─ Low traffic (<100 req/hour)?
  │  └─ YES → Use hosted API (Claude Sonnet or GPT-3.5)
  │  └─ NO → Continue
  │
  ├─ Need highest quality (legal, medical)?
  │  └─ YES → GPT-4 or Claude Opus
  │  └─ NO → Continue
  │
  ├─ Need fast responses (<2s)?
  │  └─ YES → Mistral 7B on SageMaker ml.g5.xlarge
  │  └─ NO → Continue
  │
  ├─ Complex reasoning required?
  │  └─ YES → Mixtral 8x7B or LLaMA 70B
  │  └─ NO → Mistral 7B
  │
  └─ Cost is primary concern?
     └─ YES → LLaMA 2 7B or Claude Haiku
     └─ NO → Mistral 7B (best balance)
```

---

## Migration Guide

### Scenario 1: Mistral 7B → Mixtral 8x7B

**Why migrate:** Need better quality for complex tasks

**Steps:**

**1. Deploy new model to SageMaker**
```bash
# Update instance type (needs more memory)
cdk deploy --parameters InstanceType=ml.g5.2xlarge \
           --parameters ModelId=mistralai/Mixtral-8x7B-Instruct-v0.1
```

**2. Update content handler (prompts may differ)**
```python
# Mixtral uses same format as Mistral, no changes needed
# But you may want to adjust max_new_tokens (Mixtral is slower)

model_kwargs = {
    "max_new_tokens": 400,  # Reduced from 500 to compensate for slower speed
    "temperature": 0.001,
}
```

**3. A/B test before full migration**
```python
# Send 10% of traffic to new model
if random.random() < 0.1:
    llm = create_sagemaker_llm(endpoint_name="mixtral-endpoint")
else:
    llm = create_sagemaker_llm(endpoint_name="mistral-endpoint")
```

**4. Monitor quality and cost**
- Compare answer quality manually
- Track latency increase (expect 1.5-2x slower)
- Monitor cost increase (expect 1.5x higher hosting cost)

**5. Full migration if satisfied**
```python
# Update default endpoint
os.environ["SAGEMAKER_ENDPOINT_NAME"] = "mixtral-endpoint"
```

**6. Delete old endpoint**
```bash
aws sagemaker delete-endpoint --endpoint-name mistral-7b-endpoint
```

**Cost impact:**
- Before: $723/month (ml.g5.xlarge)
- After: $1,090/month (ml.g5.2xlarge)
- Increase: +$367/month (+51%)

---

### Scenario 2: SageMaker → Claude Bedrock

**Why migrate:** Reduce infrastructure overhead, lower costs for low traffic

**Steps:**

**1. Get AWS Bedrock access**
```bash
# Request model access in Bedrock console
# Region: us-east-1 or us-west-2
```

**2. Update code to use Bedrock**
```python
# Old: SageMaker
from sagemaker_llm import create_sagemaker_llm
llm = create_sagemaker_llm(endpoint_name="mistral-endpoint")

# New: Bedrock
from langchain_aws import ChatBedrock
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1"
)
```

**3. Adjust prompts (Claude uses different format)**
```python
# Mistral uses XML for tools
# Claude has native tool calling

# Old: XML-based
tools = create_xml_agent(llm, tools, prompt)

# New: Native tool calling
from langchain.agents import AgentExecutor, create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)
```

**4. Test thoroughly (different model = different behaviors)**
- Claude is more cautious (may refuse some requests Mistral accepts)
- Claude has better instruction following
- Claude may be more verbose

**5. Delete SageMaker endpoint**
```bash
aws sagemaker delete-endpoint --endpoint-name mistral-endpoint
```

**Cost impact (100 requests/day):**
- Before: $723/month (SageMaker always-on)
- After: ~$15/month (Bedrock pay-per-use)
- Savings: 98%!

**Cost impact (10,000 requests/day):**
- Before: $723/month (SageMaker)
- After: ~$1,500/month (Bedrock)
- Increase: 107% (SageMaker cheaper for high volume)

---

### Scenario 3: GPT-3.5 → GPT-4 (Selected Queries)

**Why:** Upgrade quality for complex/important questions only

**Strategy:** Hybrid routing

**Implementation:**
```python
def route_to_model(question: str) -> str:
    """Route complex questions to GPT-4, simple to GPT-3.5"""

    # Heuristics for complexity
    word_count = len(question.split())
    has_context = "based on" in question.lower()
    is_reasoning = any(word in question.lower()
                      for word in ["why", "how", "explain", "analyze"])

    if (word_count > 20 and has_context) or is_reasoning:
        return "gpt-4"
    else:
        return "gpt-3.5-turbo"

# Usage
model = route_to_model(question)
llm = ChatOpenAI(model=model)
response = llm.invoke(question)
```

**Cost impact:**
- 80% questions → GPT-3.5 ($0.001 each) = $0.80
- 20% questions → GPT-4 ($0.03 each) = $6.00
- Total: $6.80 per 1000 questions

vs.

- 100% GPT-4: $30 per 1000 questions
- Savings: 77%

---

## Testing Models

### Benchmark Your Use Case

**1. Create test set**
```python
test_questions = [
    ("What is 2+2?", "4"),
    ("Explain quantum computing", "should mention superposition, qubits"),
    ("Summarize: [long text]", "key points: A, B, C"),
    ...
]
```

**2. Test each model**
```python
def evaluate_model(model_name, test_questions):
    results = []

    for question, expected in test_questions:
        start = time.time()
        answer = llm.invoke(question)
        latency = time.time() - start

        # Manual quality check
        quality_score = input(f"Rate answer 1-5: {answer}\n")

        results.append({
            "question": question,
            "answer": answer,
            "latency": latency,
            "quality": quality_score
        })

    return results
```

**3. Compare**
```python
mistral_results = evaluate_model("mistral-7b", test_questions)
gpt4_results = evaluate_model("gpt-4", test_questions)

# Compare avg quality and latency
print(f"Mistral: {avg_quality(mistral_results):.1f}/5, {avg_latency(mistral_results):.1f}s")
print(f"GPT-4: {avg_quality(gpt4_results):.1f}/5, {avg_latency(gpt4_results):.1f}s")
```

---

## Common Migration Pitfalls

### 1. Forgetting to Update Prompts

Different models need different prompts:

```python
# ❌ Using Mistral-style prompt for GPT-4
prompt = """<s>[INST] You are an assistant... [/INST]"""  # Mistral format

# ✅ Adapt prompt for target model
if model == "gpt-4":
    prompt = """You are an assistant..."""  # No special tokens
elif model == "mistral":
    prompt = """<s>[INST] You are an assistant... [/INST]"""
```

### 2. Not Accounting for Speed Differences

```python
# ❌ Same timeout for all models
timeout = 5  # May be too short for GPT-4, too long for Haiku

# ✅ Model-specific timeouts
timeouts = {
    "claude-haiku": 2,
    "mistral-7b": 5,
    "gpt-4": 30
}
timeout = timeouts.get(model, 10)
```

### 3. Ignoring Cost Changes

```python
# ❌ Migrate without cost tracking
new_llm = ChatBedrock(model_id="anthropic.claude-3-opus-20240229-v1:0")

# ✅ Track and alert on cost changes
from monitoring import CostTracker
cost_tracker = CostTracker()

@cost_tracker.track
def call_llm(question):
    return llm.invoke(question)

# Alert if daily cost > $100
if cost_tracker.daily_cost > 100:
    send_alert("High LLM costs!")
```

---

## Model-Specific Tips

### Mistral Models

**Prompt engineering:**
- Use XML tags for structure: `<context>...</context>`
- Be explicit about output format
- Include examples in prompt (few-shot)

**Common issues:**
- May generate malformed XML tool calls → Add validation
- Sometimes doesn't stop at `</final_answer>` → Set stop sequences

### Claude Models

**Prompt engineering:**
- Claude responds well to "Think step by step"
- Use markdown for structure
- Can handle very long prompts (100K+ tokens)

**Common issues:**
- May refuse valid requests if they seem potentially harmful → Rephrase
- Can be overly apologetic → Prompt: "Be concise"

### GPT Models

**Prompt engineering:**
- GPT-4 excellent at following complex instructions
- GPT-3.5 needs simpler, more direct prompts
- Both support function calling (better than XML for tools)

**Common issues:**
- Rate limits on API → Implement exponential backoff
- GPT-4 can be slow → Use streaming for better UX

---

## Recommended Starting Point

**For most production use cases:**

1. **Start with:** Mistral 7B on SageMaker ml.g5.xlarge
   - Good balance of cost, speed, quality
   - Easy to deploy and manage
   - Predictable costs

2. **Optimize after 1 month:**
   - Low traffic? → Switch to Claude Bedrock Sonnet
   - High traffic? → Add auto-scaling
   - Need better quality? → Upgrade to Mixtral
   - Cost too high? → Add caching, use LLaMA

3. **Monitor continuously:**
   - Track quality, latency, cost
   - A/B test improvements
   - Adjust based on real usage patterns

---

## Next Steps

1. **Choose model** using decision framework above
2. **Deploy** using deployment guides
3. **Test** with your actual use case
4. **Monitor** and iterate
5. **Migrate** when requirements change

**Remember:** The best model is the one that meets your requirements at acceptable cost. Start simple, optimize later.
