"""
Tool Definitions for LangGraph Agent

This module creates and configures all tools available to the agent.
Tools are external functions the agent can invoke during execution.

For AI/ML Scientists:
- Tools = function calls the agent can make (like API endpoints)
- Each tool has: name, description, and execution logic
- Agent decides when to use tools based on their descriptions
- Think of tools as giving the LLM "hands" to interact with the world

Common tool types:
- Search (Tavily, Google, Wikipedia)
- Databases (SQL queries, vector stores)
- APIs (REST calls, GraphQL)
- File systems (S3, local files)
- Computation (calculators, data processing)
"""

import os
import json
import logging
import boto3
from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool

logger = logging.getLogger(__name__)


def create_tools() -> List[Tool]:
    """
    Create and return all tools available to the agent

    Returns:
        List[Tool]: List of LangChain tool objects

    Environment Variables:
        TAVILY_SECRET_ARN: ARN of Tavily API key in AWS Secrets Manager

    For AI/ML Scientists:
        This function is called once when the Lambda container starts.
        The tools list is then used throughout the container's lifetime.
        Adding a new tool = add it to the return list here.
    """

    tools = []

    # ========================================================================
    # TOOL 1: Tavily Web Search
    # ========================================================================

    logger.info("Initializing Tavily search tool")

    try:
        # Get Tavily API key from AWS Secrets Manager
        # Why Secrets Manager: Never hardcode API keys in code!
        # Secrets Manager provides encrypted storage + access control
        tavily_secret_arn = os.environ.get('TAVILY_SECRET_ARN')

        if not tavily_secret_arn:
            logger.warning("TAVILY_SECRET_ARN not set - skipping Tavily tool")
        else:
            # Fetch secret from Secrets Manager
            secrets_client = boto3.client('secretsmanager')
            response = secrets_client.get_secret_value(SecretId=tavily_secret_arn)

            # Parse secret (stored as JSON: {"api_key": "tvly-..."})
            secret_dict = json.loads(response['SecretString'])
            tavily_api_key = secret_dict['api_key']

            # Set environment variable (TavilySearchResults reads from os.environ)
            os.environ['TAVILY_API_KEY'] = tavily_api_key

            # Create Tavily search tool
            tavily_tool = TavilySearchResults(
                max_results=1,  # Limit to 1 result (saves tokens + cost)
                # max_results=3,  # Uncomment for more comprehensive search
            )

            tools.append(tavily_tool)
            logger.info("Tavily search tool initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Tavily tool: {e}")
        # Don't crash the entire Lambda - just skip this tool
        # Agent will work but without web search capability

    # ========================================================================
    # TOOL 2: Wikipedia (Alternative Free Search)
    # ========================================================================

    # Uncomment to add Wikipedia search as a free alternative to Tavily:
    #
    # from langchain_community.tools import WikipediaQueryRun
    # from langchain_community.utilities import WikipediaAPIWrapper
    #
    # wikipedia_tool = WikipediaQueryRun(
    #     api_wrapper=WikipediaAPIWrapper(
    #         top_k_results=1,
    #         doc_content_chars_max=2000,
    #     ),
    #     name="wikipedia",
    #     description="Search Wikipedia for factual information about people, places, events, and concepts."
    # )
    # tools.append(wikipedia_tool)

    # ========================================================================
    # TOOL 3: Calculator (Simple Math)
    # ========================================================================

    # Simple calculator tool for arithmetic
    # No external API needed - pure Python
    #
    # def calculator(expression: str) -> str:
    #     """
    #     Evaluate a mathematical expression
    #
    #     Args:
    #         expression: Math expression like "2 + 2" or "3.14 * 10"
    #
    #     Returns:
    #         Result as a string
    #     """
    #     try:
    #         # SECURITY WARNING: eval() is dangerous!
    #         # In production, use a safe math parser like 'numexpr'
    #         result = eval(expression, {"__builtins__": {}}, {})
    #         return str(result)
    #     except Exception as e:
    #         return f"Error: {str(e)}"
    #
    # calculator_tool = Tool(
    #     name="calculator",
    #     description="Perform basic arithmetic calculations. Input should be a math expression like '2 + 2' or '3.14 * 10'.",
    #     func=calculator
    # )
    # tools.append(calculator_tool)

    # ========================================================================
    # TOOL 4: Custom Tool Template
    # ========================================================================

    # Template for adding your own custom tools
    #
    # def my_custom_tool(input_str: str) -> str:
    #     """
    #     Your tool's logic here
    #
    #     Args:
    #         input_str: Input from the agent (based on tool description)
    #
    #     Returns:
    #         Result string that the agent will see
    #     """
    #     # Example: Query a database
    #     # import boto3
    #     # dynamodb = boto3.resource('dynamodb')
    #     # table = dynamodb.Table('my-table')
    #     # response = table.get_item(Key={'id': input_str})
    #     # return str(response.get('Item', 'Not found'))
    #
    #     return f"Processed: {input_str}"
    #
    # custom_tool = Tool(
    #     name="my_tool",
    #     description="What this tool does and when to use it. Be specific! Example: 'Query the customer database for user information. Input should be a customer ID.'",
    #     func=my_custom_tool
    # )
    # tools.append(custom_tool)

    # ========================================================================
    # Return Final Tool List
    # ========================================================================

    if not tools:
        logger.warning("No tools initialized! Agent will have limited capabilities.")
        # Add a dummy tool so agent doesn't break
        tools.append(Tool(
            name="no_tools_available",
            description="No tools are currently available. Respond based on your training data only.",
            func=lambda x: "No tools available"
        ))

    logger.info(f"Initialized {len(tools)} tools: {[t.name for t in tools]}")
    return tools


# ============================================================================
# Example Custom Tools (Commented Out - Uncomment to Use)
# ============================================================================

def create_database_tool():
    """
    Example: Query a DynamoDB table

    For AI/ML Scientists:
        DynamoDB = AWS's NoSQL database (like MongoDB)
        Use this to give your agent access to structured data
    """
    import boto3

    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(os.environ.get('DYNAMODB_TABLE_NAME', 'my-table'))

    def query_database(item_id: str) -> str:
        try:
            response = table.get_item(Key={'id': item_id})
            item = response.get('Item')
            if item:
                return json.dumps(item, default=str)
            else:
                return f"No item found with ID: {item_id}"
        except Exception as e:
            return f"Database error: {str(e)}"

    return Tool(
        name="query_database",
        description="Query the database for information about items. Input should be an item ID (string).",
        func=query_database
    )


def create_s3_reader_tool():
    """
    Example: Read files from S3

    For AI/ML Scientists:
        S3 = AWS object storage (like Google Drive, but for code)
        Use this to give your agent access to documents, datasets, etc.
    """
    import boto3

    s3_client = boto3.client('s3')

    def read_s3_file(s3_uri: str) -> str:
        """
        Read a file from S3

        Args:
            s3_uri: S3 URI like "s3://bucket-name/path/to/file.txt"

        Returns:
            File contents as string
        """
        try:
            # Parse S3 URI
            if not s3_uri.startswith('s3://'):
                return "Error: Invalid S3 URI. Must start with 's3://'"

            parts = s3_uri[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''

            # Read file
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')

            # Truncate if too long (save tokens)
            max_chars = 5000
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n[Truncated. Total length: {len(content)} characters]"

            return content

        except Exception as e:
            return f"S3 read error: {str(e)}"

    return Tool(
        name="read_s3_file",
        description="Read the contents of a text file stored in S3. Input should be an S3 URI in the format: s3://bucket-name/path/to/file.txt",
        func=read_s3_file
    )


def create_rest_api_tool():
    """
    Example: Call an external REST API

    For AI/ML Scientists:
        This lets your agent interact with any HTTP API
        Examples: CRM systems, weather APIs, stock prices, etc.
    """
    import requests

    def call_api(endpoint_and_params: str) -> str:
        """
        Call a REST API

        Args:
            endpoint_and_params: JSON string like:
                {"url": "https://api.example.com/v1/users", "method": "GET", "params": {"id": "123"}}

        Returns:
            API response as string
        """
        try:
            config = json.loads(endpoint_and_params)
            url = config['url']
            method = config.get('method', 'GET')
            params = config.get('params', {})
            headers = config.get('headers', {})

            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=params, headers=headers, timeout=10)
            else:
                return f"Unsupported HTTP method: {method}"

            response.raise_for_status()
            return response.text

        except json.JSONDecodeError:
            return "Error: Input must be valid JSON"
        except requests.RequestException as e:
            return f"API call failed: {str(e)}"

    return Tool(
        name="call_rest_api",
        description='Call an external REST API. Input must be JSON with keys: "url", "method" (GET/POST), "params" (dict). Example: {"url":"https://api.example.com/users","method":"GET","params":{"id":"123"}}',
        func=call_api
    )
