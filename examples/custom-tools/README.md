# Custom Tool Examples

## Overview

This example demonstrates how to create **custom tools** for your LangGraph agents beyond the built-in tools like Tavily search.

**For AI/ML Scientists:**
Tools are like APIs for your model - they extend what the LLM can do beyond text generation. Think of them as "function calls" the model can make to interact with external systems (databases, APIs, file systems, etc.).

**Why Custom Tools?**
- Access your own data sources (databases, data warehouses)
- Integrate with internal APIs (CRM, inventory systems, etc.)
- Perform specialized computations (math, simulations)
- Read/write files (logs, configs, reports)
- Call external services (weather APIs, payment processors)

---

## What Gets Built

**3 Custom Tool Examples:**

1. **Database Tool** (`database_tool.py`)
   - Queries PostgreSQL/MySQL databases
   - Executes SELECT queries safely
   - Returns results as formatted text
   - Use case: "What were our sales last quarter?"

2. **API Tool** (`api_tool.py`)
   - Calls external REST APIs
   - Handles authentication (API keys)
   - Parses JSON responses
   - Use case: "What's the weather in Paris?"

3. **File System Tool** (`file_system_tool.py`)
   - Reads files from specified directories
   - Lists directory contents
   - Writes reports/logs
   - Use case: "Summarize the contents of report.txt"

**Plus:** Agent that uses all three tools together

---

## Key Concepts

### Tool Anatomy

Every LangChain tool has:

```python
from langchain.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """
    Description of what this tool does.

    The LLM reads this docstring to decide when to use the tool!

    Args:
        query: What the user wants to know

    Returns:
        The result as a string
    """
    # Your logic here
    result = do_something(query)
    return result
```

**For AI/ML Scientists:**
The `@tool` decorator converts a normal Python function into something the LLM can call. The docstring is crucial - it's like a few-shot prompt teaching the model when/how to use the tool.

### Tool Selection Process

**How the LLM decides which tool to use:**

```
User: "What's the weather in Paris?"

LLM reasoning:
1. Reads available tools and their descriptions
2. Matches question intent to tool capabilities
3. Generates XML tool call:
   <tool>weather_api_tool</tool>
   <tool_input>Paris</tool_input>

LangGraph:
4. Executes weather_api_tool("Paris")
5. Passes result back to LLM

LLM:
6. Uses result to generate final answer
```

**For AI/ML Scientists:**
This is similar to a router model that selects which expert to use in a mixture-of-experts architecture. The LLM acts as the router, tools are the experts.

### Tool Safety

**Important: Tools execute arbitrary code!**

⚠️ **Security Considerations:**
- Never execute user-provided SQL directly (SQL injection risk)
- Validate/sanitize all inputs
- Limit tool permissions (read-only when possible)
- Use allow lists, not deny lists
- Log all tool executions for auditing

**Example: Safe vs. Unsafe SQL tool**

```python
# ❌ UNSAFE - vulnerable to SQL injection
@tool
def unsafe_query_db(sql: str) -> str:
    """Execute any SQL query"""
    return db.execute(sql)  # User could drop tables!

# ✅ SAFE - parameterized queries only
@tool
def safe_query_sales(product: str, date_range: str) -> str:
    """Query sales for a product in a date range"""
    # Validate inputs
    assert product.isalnum(), "Invalid product"

    # Use parameterized query
    sql = "SELECT * FROM sales WHERE product = ? AND date BETWEEN ? AND ?"
    return db.execute(sql, [product, date_start, date_end])
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd examples/custom-tools/

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env

# Edit .env to add your credentials:
# - Database connection strings
# - API keys
# - File paths
nano .env
```

### 3. Test Individual Tools

```bash
# Test database tool
python database_tool.py

# Test API tool
python api_tool.py

# Test file system tool
python file_system_tool.py
```

### 4. Run Agent with All Tools

```bash
python custom_tool_agent.py
```

**Expected interaction:**
```
Question: What were the sales for product X last month, and what's the current weather?

[AGENT] Planning...
[AGENT] Using tool: database_tool
  Input: product X sales last month
  Result: Sales: $15,234 (142 units)

[AGENT] Using tool: weather_api_tool
  Input: current weather
  Result: 72°F, partly cloudy

[AGENT] Final answer:
Product X had sales of $15,234 (142 units) last month. Current weather is 72°F and partly cloudy.
```

---

## Tool Examples

### Database Tool

**Use Case:** Query your production database for business metrics

```python
from langchain.tools import tool
import psycopg2

@tool
def query_sales_db(question: str) -> str:
    """
    Query the sales database for metrics like revenue, units sold, etc.

    Use this when the user asks about sales, revenue, customers, or orders.

    Args:
        question: Natural language question about sales data

    Returns:
        Sales metrics formatted as text
    """

    # Connect to database
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

    # Parse question to determine query
    # (In production, use NL-to-SQL model or predefined queries)
    if "revenue" in question.lower():
        query = "SELECT SUM(amount) as revenue FROM sales WHERE date >= NOW() - INTERVAL '30 days'"
    elif "units" in question.lower():
        query = "SELECT SUM(quantity) as units FROM sales WHERE date >= NOW() - INTERVAL '30 days'"
    else:
        return "I can query revenue or units sold. Please specify which you need."

    # Execute and return
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()

    return f"Result: {result[0]}"
```

**For AI/ML Scientists:**
This is like a retrieval component in RAG, but instead of retrieving documents, we're retrieving structured data from a database.

### API Tool

**Use Case:** Call external APIs (weather, stock prices, etc.)

```python
import requests
from langchain.tools import tool

@tool
def weather_api_tool(location: str) -> str:
    """
    Get current weather for a location.

    Use this when the user asks about weather, temperature, or forecasts.

    Args:
        location: City name or coordinates

    Returns:
        Current weather conditions
    """

    api_key = os.getenv("WEATHER_API_KEY")

    response = requests.get(
        f"https://api.weatherapi.com/v1/current.json",
        params={
            "key": api_key,
            "q": location
        }
    )

    if response.status_code != 200:
        return f"Error: Could not fetch weather for {location}"

    data = response.json()

    return f"{data['current']['temp_f']}°F, {data['current']['condition']['text']}"
```

### File System Tool

**Use Case:** Read configuration files, logs, or write reports

```python
import os
from langchain.tools import tool

@tool
def read_file_tool(filename: str) -> str:
    """
    Read contents of a file.

    Use this when the user asks about file contents, logs, or configs.

    Args:
        filename: Name of file to read (in allowed directory only)

    Returns:
        File contents or error message
    """

    # Security: Only allow reading from specific directory
    allowed_dir = os.getenv("ALLOWED_FILE_DIR", "/tmp/safe_files")
    filepath = os.path.join(allowed_dir, filename)

    # Prevent directory traversal attacks
    if not os.path.abspath(filepath).startswith(os.path.abspath(allowed_dir)):
        return "Error: Access denied (path traversal detected)"

    try:
        with open(filepath, 'r') as f:
            contents = f.read()
        return f"Contents of {filename}:\n{contents}"
    except FileNotFoundError:
        return f"Error: File {filename} not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

---

## Creating Your Own Tools

### Step 1: Define the Tool Function

```python
from langchain.tools import tool

@tool
def my_custom_tool(input_param: str) -> str:
    """
    [IMPORTANT] Write a clear description here!

    The LLM uses this docstring to decide when to call this tool.
    Be specific about:
    - What the tool does
    - When to use it
    - What input it expects

    Args:
        input_param: Description of the parameter

    Returns:
        Description of what gets returned
    """

    # Your implementation
    result = do_something(input_param)

    return str(result)  # Always return a string
```

**Key points:**
- Use the `@tool` decorator
- Write detailed docstring (LLM reads this!)
- Input and output should be strings (LLM works with text)
- Handle errors gracefully (return error message, don't raise)
- Keep it focused (one tool = one capability)

### Step 2: Add Tool to Agent

```python
from langchain_community.tools.tavily_search import TavilySearchResults
from my_tools import my_custom_tool

def create_tools() -> List[Tool]:
    """Create list of tools for the agent"""

    tools = [
        TavilySearchResults(max_results=1),  # Built-in tool
        my_custom_tool,                       # Your custom tool
    ]

    return tools

# In your graph setup:
tools = create_tools()
agent_runnable = create_xml_agent(llm, tools, prompt)
```

### Step 3: Test the Tool

```python
# Test tool directly
result = my_custom_tool.invoke("test input")
print(result)

# Test with agent
app = create_agent_graph()
response = app.invoke({
    "input": "Question that should trigger my custom tool"
})
```

---

## Advanced Patterns

### 1. Tools with Multiple Parameters

```python
from pydantic import BaseModel, Field
from langchain.tools import tool

class SearchInput(BaseModel):
    """Input schema for search tool"""
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Number of results to return")
    date_filter: str = Field(default="", description="Date range (e.g., 'last_week')")

@tool(args_schema=SearchInput)
def advanced_search_tool(query: str, max_results: int = 5, date_filter: str = "") -> str:
    """
    Search with advanced filters.

    Use this for searches that need specific result counts or date filtering.
    """

    # Implementation with multiple params
    results = search(query, limit=max_results, date=date_filter)
    return format_results(results)
```

**For AI/ML Scientists:**
Using Pydantic schemas is like defining the signature of a function in a typed language. It helps the LLM generate correct tool calls with proper parameter types.

### 2. Async Tools (for I/O-bound operations)

```python
from langchain.tools import tool

@tool
async def async_api_tool(query: str) -> str:
    """Async API call (faster for multiple parallel requests)"""

    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com?q={query}") as response:
            data = await response.json()
            return str(data)
```

**For AI/ML Scientists:**
Async tools are useful when you need to make multiple I/O-bound calls (API requests, database queries) in parallel. Like batching inference requests to maximize throughput.

### 3. Tools that Return Structured Data

```python
@tool
def get_user_profile(user_id: str) -> str:
    """Get user profile from database"""

    user = db.query(User).filter(User.id == user_id).first()

    # Format as structured text (the LLM will parse this)
    return f"""
    User Profile:
    - Name: {user.name}
    - Email: {user.email}
    - Plan: {user.subscription_plan}
    - Last Login: {user.last_login}
    """
```

**For AI/ML Scientists:**
Since tools must return strings, format structured data as readable text. The LLM is good at parsing semi-structured text (key-value pairs, tables, etc.).

### 4. Tool Chaining

```python
@tool
def lookup_user_tool(email: str) -> str:
    """Lookup user ID by email"""
    user = User.query.filter_by(email=email).first()
    return f"User ID: {user.id}"

@tool
def get_user_orders_tool(user_id: str) -> str:
    """Get orders for a user ID"""
    orders = Order.query.filter_by(user_id=user_id).all()
    return f"Found {len(orders)} orders"

# The LLM will automatically chain these:
# Question: "How many orders does alice@example.com have?"
# 1. Call lookup_user_tool("alice@example.com") → "User ID: 123"
# 2. Call get_user_orders_tool("123") → "Found 5 orders"
# 3. Answer: "Alice has 5 orders"
```

---

## Error Handling

### Best Practices

**1. Return error messages, don't raise exceptions:**

```python
# ❌ BAD
@tool
def bad_tool(input: str) -> str:
    if not valid(input):
        raise ValueError("Invalid input!")  # Breaks the agent

# ✅ GOOD
@tool
def good_tool(input: str) -> str:
    if not valid(input):
        return "Error: Invalid input. Please provide..."  # Agent can recover
```

**2. Provide helpful error messages:**

```python
@tool
def query_database(query: str) -> str:
    try:
        return db.execute(query)
    except DatabaseError as e:
        # Detailed error helps LLM understand what went wrong
        return f"Database error: {str(e)}. Try rephrasing your question."
```

**3. Add timeouts for external calls:**

```python
import requests

@tool
def api_tool(query: str) -> str:
    try:
        response = requests.get(
            f"https://api.example.com?q={query}",
            timeout=5  # Don't wait forever
        )
        return response.text
    except requests.Timeout:
        return "API request timed out. Please try again."
```

---

## Cost & Performance

### Tool Execution Costs

**Each tool call adds:**
1. **LLM call to generate tool call:** ~$0.001
2. **Tool execution cost:** Varies (API fees, database costs, etc.)
3. **LLM call to process result:** ~$0.001

**Example:**
```
Question: "What's the weather in Paris and sales for product X?"

Cost breakdown:
- LLM decides tools needed: $0.001
- Weather API call: $0.0001 (API fee)
- LLM processes weather: $0.001
- Database query: $0.00001 (negligible)
- LLM processes sales data: $0.001
- LLM generates final answer: $0.001

Total: ~$0.005 per question
```

### Optimization Strategies

**1. Batch tool calls when possible:**

```python
# Instead of calling tool 10 times:
for item in items:
    result = tool.invoke(item)  # 10 LLM calls = 10x cost

# Call once with batch input:
result = batch_tool.invoke(",".join(items))  # 1 LLM call
```

**2. Cache expensive tool results:**

```python
from functools import lru_cache

@tool
@lru_cache(maxsize=100)
def expensive_api_tool(query: str) -> str:
    """Cached API calls (same query = cached result)"""
    return call_expensive_api(query)
```

**3. Use cheaper tools when possible:**

```python
# Example: Use local file instead of database when feasible
@tool
def get_config(key: str) -> str:
    """Read config from local JSON (free) instead of database ($)"""
    with open('config.json') as f:
        config = json.load(f)
    return config.get(key, "Not found")
```

---

## Testing Custom Tools

### Unit Tests

```python
def test_my_tool():
    """Test tool in isolation"""

    result = my_custom_tool.invoke("test input")

    assert result != ""
    assert "expected" in result.lower()
```

### Integration Tests with Agent

```python
def test_agent_uses_tool():
    """Test that agent correctly uses the tool"""

    with patch('my_tools.external_api_call') as mock_api:
        mock_api.return_value = "mock result"

        app = create_agent_with_tools([my_custom_tool])

        response = app.invoke({
            "input": "Question that should trigger my_custom_tool"
        })

        # Verify tool was called
        mock_api.assert_called_once()

        # Verify tool result appears in final answer
        assert "mock result" in response["output"].lower()
```

---

## Next Steps

1. **Start simple:** Create one custom tool for your use case
2. **Test thoroughly:** Ensure tool handles errors gracefully
3. **Monitor usage:** Log tool calls to understand agent behavior
4. **Iterate:** Add more tools as needed, based on actual usage

**Related Examples:**
- `../multi-agent/` - See how multiple agents use tools together
- `../../tests/integration/test_tools.py` - More testing patterns
- `../../agent/tools.py` - Reference implementation of Tavily tool

For questions, see the main repository README.
