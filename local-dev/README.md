# Local Development Environment

## Overview

This directory contains a Docker Compose setup for local development and testing **without AWS costs**.

**For AI/ML Scientists:**
- No GPU needed (mock LLM responses)
- No SageMaker deployment needed
- No AWS credentials needed
- Perfect for: learning, experimentation, debugging

---

## What Gets Started

Running `docker-compose up` starts 3 containers:

1. **Mock SageMaker Endpoint** (port 8080)
   - Simulates SageMaker API
   - Returns predefined responses (not real LLM)
   - Instant response time (~10ms vs. 2-5 seconds)

2. **LangGraph Agent** (port 8000)
   - Runs your agent code
   - Calls mock endpoint instead of SageMaker
   - REST API for asking questions

3. **Streamlit UI** (port 8501) - Optional
   - Web UI for chatting with the agent
   - No coding required
   - Great for demos

---

## Quick Start

### 1. Prerequisites

Install Docker:
```bash
# macOS
brew install --cask docker

# Linux
sudo apt-get install docker.io docker-compose

# Verify
docker --version
docker-compose --version
```

### 2. Configure (Optional)

```bash
cd local-dev/

# Copy environment template
cp .env.example .env

# Edit .env and add your Tavily API key (or leave blank for mock)
nano .env
```

### 3. Start Services

```bash
docker-compose up
```

You should see:
```
✓ mock-sagemaker    Healthy
✓ agent             Started
✓ ui                Started
```

### 4. Test the Agent

**Option A: Web UI (easiest)**
```
Open browser: http://localhost:8501
Type a question and press Enter
```

**Option B: API (curl)**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the capital of France?"}'
```

**Option C: Python**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is 2+2?"}
)

print(response.json())
```

---

## Project Structure

```
local-dev/
├── docker-compose.yml          # Orchestrates all services
├── .env.example                # Environment variables template
├── mock-sagemaker/
│   ├── Dockerfile
│   ├── server.py               # Mock endpoint server
│   └── responses/
│       └── default.json        # Response templates
├── streamlit-ui/
│   ├── Dockerfile
│   ├── app.py                  # Chat UI
│   └── requirements.txt
└── README.md                   # This file
```

---

## How It Works

### Mock vs. Real LLM

**Mock (Local Development):**
```
User Question → Agent → Mock Endpoint → Pattern Matching → Canned Response
                                         (instant, free)
```

**Real (Production):**
```
User Question → Agent → SageMaker → GPU → Mistral 7B → Generated Text
                                    (2-5 sec, $0.001/call)
```

### Example Flow

**Input:** "What is the latest storm in the UK?"

**Agent calls mock endpoint:**
```json
{
  "inputs": "<s>[INST] What is the latest storm in the UK? [/INST]",
  "parameters": {"max_new_tokens": 500, "temperature": 0.001}
}
```

**Mock endpoint detects keywords:** `latest` → Assumes search needed

**Returns tool call:**
```json
[{
  "generated_text": "<tool>tavily_search_results_json</tool><tool_input>latest UK storm</tool_input>"
}]
```

**Agent executes Tavily search** (if TAVILY_API_KEY is set)

**Agent calls mock endpoint again** with search results

**Mock endpoint returns final answer:**
```json
[{
  "generated_text": "<final_answer>Based on search results, Storm Henk was the latest...</final_answer>"
}]
```

---

## Customizing Mock Responses

### Add New Response Pattern

Edit `mock-sagemaker/server.py`:

```python
def select_response(prompt):
    prompt_lower = prompt.lower()

    # Add your pattern
    if 'quantum' in prompt_lower:
        return {
            'generated_text': '<final_answer>Quantum computing uses quantum mechanics...</final_answer>'
        }

    # ... rest of function
```

### Add Response Templates

Create `mock-sagemaker/responses/my_response.json`:

```json
{
  "generated_text": "Your custom response here"
}
```

Load in `server.py`:
```python
responses = load_responses()  # Automatically loads all JSON files
```

---

## Development Workflow

### 1. Make Changes to Agent Code

```bash
# Edit agent code in ../agent/

# Restart just the agent container
docker-compose restart agent

# Or rebuild if you changed dependencies
docker-compose up --build agent
```

### 2. View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f agent
docker-compose logs -f mock-sagemaker

# Last 50 lines
docker-compose logs --tail=50 agent
```

### 3. Debug Inside Container

```bash
# Open shell in running container
docker-compose exec agent /bin/bash

# Run Python commands
python
>>> from graph import create_agent_graph
>>> app = create_agent_graph()
```

### 4. Run Tests

```bash
# Stop services
docker-compose down

# Run tests
cd ../tests
pytest

# Or test against running services
pytest integration/ --live
```

---

## Using Real SageMaker Endpoint

Want to test with real LLM but avoid deploying full stack?

**Edit `docker-compose.yml`:**

```yaml
agent:
  environment:
    # Comment out mock endpoint
    # - SAGEMAKER_ENDPOINT_URL=http://mock-sagemaker:8080

    # Add real endpoint
    - SAGEMAKER_ENDPOINT_NAME=your-real-endpoint-name
    - AWS_ACCESS_KEY_ID=your-key
    - AWS_SECRET_ACCESS_KEY=your-secret
    - AWS_DEFAULT_REGION=us-east-1
```

Now agent calls real SageMaker!

---

## Troubleshooting

### Issue: "Address already in use"

**Cause:** Port 8000, 8080, or 8501 already taken

**Solution:** Change ports in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # External:Internal
```

### Issue: Mock endpoint returns same response

**Cause:** Pattern matching didn't find keywords

**Solution:** Add more patterns in `mock-sagemaker/server.py`

### Issue: Tavily tool fails

**Cause:** TAVILY_API_KEY not set

**Solution:** Get key from https://app.tavily.com/ and add to `.env`

### Issue: Agent container keeps restarting

**Cause:** Error in agent code

**Solution:** Check logs:
```bash
docker-compose logs agent
```

### Issue: Can't connect to containers

**Cause:** Services not fully started

**Solution:** Wait for healthy status:
```bash
docker-compose ps
```

All should show "Up" or "Up (healthy)"

---

## Cleanup

### Stop Services (Keep Data)
```bash
docker-compose stop
```

### Stop and Remove Containers
```bash
docker-compose down
```

### Remove Everything (Including Images)
```bash
docker-compose down --rmi all --volumes
```

---

## Performance Comparison

| Metric | Mock | Real SageMaker |
|--------|------|----------------|
| **Response Time** | ~10ms | ~2-5 seconds |
| **Cost per Request** | $0 | ~$0.001 |
| **Quality** | Canned | Real AI |
| **GPU Needed** | No | Yes |
| **Internet Required** | Only for tools (Tavily) | Yes |

**When to use Mock:**
- Developing agent logic
- Testing tool integration
- Learning LangGraph
- Debugging edge cases
- Running tests in CI/CD

**When to use Real:**
- Need actual AI responses
- Testing prompt engineering
- Evaluating model quality
- Before production deployment

---

## Next Steps

1. **Experiment:** Try different questions, observe agent behavior
2. **Customize:** Add new tools, modify prompts, change graph logic
3. **Test:** Write tests using this local environment
4. **Deploy:** When ready, use CDK to deploy to AWS

For production deployment, see: `../deployment/cdk/README.md`
