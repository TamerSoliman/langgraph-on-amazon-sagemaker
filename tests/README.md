# Testing Suite for LangGraph on SageMaker

## Overview

This directory contains comprehensive tests for the LangGraph agent system.

**For AI/ML Scientists:**
Testing ensures your agent works correctly before deploying to production. Think of tests as "unit tests for your model API" - they validate behavior without manual testing.

---

## Test Types

### 1. Unit Tests (`tests/unit/`)
- **What:** Test individual functions in isolation
- **Speed:** Fast (milliseconds)
- **Dependencies:** None (all mocked)
- **Run frequency:** Every code change
- **Cost:** Free

**Examples:**
- Does ContentHandler correctly format SageMaker requests?
- Does Lambda handler parse API Gateway events correctly?

### 2. Integration Tests (`tests/integration/`)
- **What:** Test multiple components working together
- **Speed:** Medium (1-2 seconds)
- **Dependencies:** Mocked LLM, real graph execution
- **Run frequency:** Before commits
- **Cost:** Free

**Examples:**
- Does the agent execute the full ReAct loop?
- Are tool results passed back to the agent correctly?

### 3. End-to-End Tests (`tests/e2e/`)
- **What:** Test entire system with real AWS services
- **Speed:** Slow (5-15 seconds)
- **Dependencies:** Deployed SageMaker endpoint, Tavily API
- **Run frequency:** Before releases only
- **Cost:** ~$0.01 per test

**Examples:**
- Does the real SageMaker endpoint return valid responses?
- Can the agent answer factual questions correctly?

---

## Quick Start

### Install Dependencies

```bash
# Install test dependencies
cd tests/
pip install -r requirements-test.txt

# Also install agent dependencies (needed for testing)
pip install -r ../agent/requirements.txt
```

### Run All Tests (Except E2E)

```bash
# From tests/ directory
pytest

# With coverage report
pytest --cov=../agent --cov-report=html

# Verbose output
pytest -v

# Show print statements
pytest -s
```

### Run Specific Test Suites

```bash
# Unit tests only
pytest unit/

# Integration tests only
pytest integration/

# Specific test file
pytest unit/test_content_handler.py

# Specific test function
pytest unit/test_content_handler.py::TestContentHandlerInput::test_basic_prompt_transformation
```

### Run E2E Tests (Requires Deployed Endpoint)

```bash
# Set environment variables
export SAGEMAKER_ENDPOINT_NAME="your-endpoint-name"
export TAVILY_SECRET_ARN="arn:aws:secretsmanager:..."
export RUN_E2E_TESTS=1

# Run E2E tests
pytest e2e/
```

**WARNING:** E2E tests call real AWS services and cost money!

---

## Test Coverage

Generate HTML coverage report:

```bash
pytest --cov=../agent --cov-report=html

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

**Coverage goals:**
- Unit tests: >80% code coverage
- Integration tests: Critical paths (agent loop, tool execution)
- E2E tests: Happy path + key failure modes

---

## Writing New Tests

### Example: Unit Test

```python
# tests/unit/test_my_module.py

import pytest
import sys
sys.path.insert(0, '/home/user/langgraph-on-amazon-sagemaker/agent')
from my_module import my_function

def test_my_function():
    """Test my_function with valid input"""
    result = my_function("input")
    assert result == "expected_output"

def test_my_function_error():
    """Test my_function error handling"""
    with pytest.raises(ValueError):
        my_function("invalid_input")
```

### Example: Integration Test

```python
# tests/integration/test_my_integration.py

import pytest
from unittest.mock import patch

@patch('sagemaker_llm.SagemakerEndpoint')
def test_agent_with_mock_llm(mock_llm):
    """Test agent execution with mocked LLM"""
    mock_llm.return_value.invoke.return_value = "<final_answer>Test</final_answer>"

    from graph import create_agent_graph
    app = create_agent_graph()

    # Execute agent
    # ...assert results
```

### Using Fixtures

Fixtures are defined in `conftest.py` and automatically available:

```python
def test_with_fixtures(api_gateway_event, lambda_context, mock_llm_response):
    """Test using pre-defined test fixtures"""
    # Fixtures are injected automatically
    assert api_gateway_event['httpMethod'] == 'POST'
    assert lambda_context.request_id is not None
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r agent/requirements.txt
        pip install -r tests/requirements-test.txt

    - name: Run unit tests
      run: pytest tests/unit/ -v

    - name: Run integration tests
      run: pytest tests/integration/ -v

    - name: Generate coverage report
      run: pytest --cov=agent --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'agent'"

**Solution:** Add agent directory to Python path:
```python
import sys
sys.path.insert(0, '/home/user/langgraph-on-amazon-sagemaker/agent')
```

### Issue: "ImportError: cannot import name 'create_agent_graph'"

**Solution:** Ensure agent dependencies are installed:
```bash
pip install -r ../agent/requirements.txt
```

### Issue: E2E tests fail with "EndpointNotFound"

**Solution:** Deploy SageMaker endpoint first:
```bash
cd deployment/cdk
cdk deploy langgraph-dev-sagemaker
```

### Issue: Tests are slow

**Solution:**
- Skip E2E tests: `pytest -m "not e2e"`
- Run in parallel: `pytest -n 4` (requires pytest-xdist)
- Use mocks instead of real services

---

## Best Practices

### 1. Test Naming
- `test_<what>_<condition>_<expected>`
- Example: `test_handler_missing_question_returns_400`

### 2. One Assert Per Test (Ideally)
```python
# Bad: Multiple unrelated asserts
def test_everything():
    assert result.status == 200
    assert result.body == "ok"
    assert result.headers == {}

# Good: Separate tests
def test_status_code():
    assert result.status == 200

def test_response_body():
    assert result.body == "ok"
```

### 3. Use Fixtures for Setup
```python
# Bad: Repeated setup
def test_1():
    llm = create_llm()
    # ...

def test_2():
    llm = create_llm()
    # ...

# Good: Fixture
@pytest.fixture
def llm():
    return create_llm()

def test_1(llm):
    # ...

def test_2(llm):
    # ...
```

### 4. Mock External Dependencies
```python
# Never do this in tests (calls real API):
def test_search():
    result = tavily_search("query")  # ❌ Costs money!

# Do this instead (mocked):
@patch('tools.TavilySearchResults')
def test_search(mock_tavily):
    mock_tavily.return_value.run.return_value = "mock result"
    result = tavily_search("query")  # ✅ Free!
```

---

## Metrics and Reporting

### Test Execution Time

```bash
pytest --durations=10  # Show 10 slowest tests
```

### Flaky Test Detection

```bash
pytest --count=10  # Run each test 10 times
```

### Test Markers

```python
# Mark slow tests
@pytest.mark.slow
def test_something_slow():
    pass

# Run only fast tests
pytest -m "not slow"
```

---

## Next Steps

1. **Run tests locally:** `pytest`
2. **Check coverage:** `pytest --cov`
3. **Add new tests:** Copy examples from `conftest.py`
4. **Setup CI/CD:** Use GitHub Actions example above
5. **Before deploying:** Run full test suite including E2E

For questions, see the main repository README or file an issue.
