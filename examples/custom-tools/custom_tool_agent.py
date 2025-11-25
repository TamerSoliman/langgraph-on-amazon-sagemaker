"""
Agent with Custom Tools

Demonstrates a LangGraph agent that uses all three custom tools:
- Database tool for querying sales data
- API tool for weather/stock information
- File system tool for reading/writing files

For AI/ML Scientists:
This shows how to compose multiple specialized tools into a single agent.
Think of it as a multi-modal model where each tool provides a different
"sense" or capability.
"""

import os
import sys
from typing import List

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../agent'))

from langchain.agents import AgentExecutor, create_xml_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.hub import pull

# Import custom tools
from database_tool import query_sales_database, initialize_test_database
from api_tool import get_current_weather, get_stock_price
from file_system_tool import read_file, write_file, list_files

# Import SageMaker LLM
from sagemaker_llm import create_sagemaker_llm


# =============================================================================
# TOOL CONFIGURATION
# =============================================================================

def create_custom_tools() -> List[Tool]:
    """
    Creates list of custom tools for the agent.

    For AI/ML Scientists:
    Each tool is a callable that the LLM can invoke. The LLM decides which
    tool to use based on the tool's description (docstring). This is similar
    to a routing mechanism in mixture-of-experts models.

    Returns:
        List of Tool objects
    """

    tools = [
        # Database tool
        query_sales_database,

        # API tools
        get_current_weather,
        get_stock_price,

        # File system tools
        read_file,
        write_file,
        list_files,
    ]

    return tools


# =============================================================================
# AGENT CREATION
# =============================================================================

def create_custom_tool_agent() -> AgentExecutor:
    """
    Creates a LangGraph agent with custom tools.

    Architecture:
    1. LLM receives question
    2. LLM decides which tool(s) to use
    3. Tools execute and return results
    4. LLM synthesizes final answer

    For AI/ML Scientists:
    This is a ReAct (Reasoning + Acting) loop. The agent:
    - Reasons about what to do
    - Acts by calling tools
    - Observes the results
    - Repeats until it has the answer

    Returns:
        Configured AgentExecutor
    """

    # Create LLM
    llm = create_sagemaker_llm()

    # Create tools
    tools = create_custom_tools()

    # Load prompt template
    # For AI/ML Scientists: This prompt teaches the LLM how to use tools
    # It's like few-shot learning - we show examples of tool usage
    prompt = pull("hwchase17/xml-agent-convo")

    # Create agent
    agent_runnable = create_xml_agent(llm, tools, prompt)

    # Create executor
    # For AI/ML Scientists: The executor manages the ReAct loop -
    # it repeatedly calls the agent, executes tools, and feeds results back
    agent_executor = AgentExecutor(
        agent=agent_runnable,
        tools=tools,
        verbose=True,  # Print tool usage for debugging
        max_iterations=10,  # Prevent infinite loops
        handle_parsing_errors=True  # Gracefully handle malformed tool calls
    )

    return agent_executor


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def run_example_queries():
    """
    Runs example queries demonstrating each tool.

    For AI/ML Scientists:
    These examples show the agent's multi-modal capabilities - it can
    seamlessly switch between data sources (DB, APIs, files) based on
    the question.
    """

    print("\n" + "="*70)
    print("Custom Tool Agent - Example Queries")
    print("="*70)

    # Initialize test database (if using SQLite)
    if os.getenv("DB_TYPE", "sqlite") == "sqlite":
        initialize_test_database()

    # Create agent
    agent = create_custom_tool_agent()

    # Example queries that use different tools
    queries = [
        # Database query
        {
            "query": "What was our total revenue last month?",
            "expected_tool": "query_sales_database",
            "description": "Tests database tool for business metrics"
        },

        # API query
        {
            "query": "What's the current weather in San Francisco?",
            "expected_tool": "get_current_weather",
            "description": "Tests weather API tool"
        },

        # File system query
        {
            "query": "What files are available in the directory?",
            "expected_tool": "list_files",
            "description": "Tests file listing tool"
        },

        # Multi-tool query (uses database + file write)
        {
            "query": "Get our sales data and save it to a file called sales_summary.txt",
            "expected_tool": "query_sales_database + write_file",
            "description": "Tests tool chaining - database read, then file write"
        },

        # Complex query requiring reasoning
        {
            "query": "What were our top products? Write a brief report about it.",
            "expected_tool": "query_sales_database + write_file",
            "description": "Tests agent's ability to synthesize data and generate output"
        }
    ]

    for i, example in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}/{len(queries)}")
        print(f"{'='*70}")
        print(f"Query: {example['query']}")
        print(f"Expected tool(s): {example['expected_tool']}")
        print(f"Description: {example['description']}")
        print(f"\n{'-'*70}")

        try:
            # Run agent
            result = agent.invoke({"input": example['query']})

            print(f"\n{'='*70}")
            print(f"Result:")
            print(f"{'='*70}")
            print(result['output'])

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("This might be due to missing API keys or configuration.")

    print(f"\n{'='*70}")
    print("All examples complete!")
    print(f"{'='*70}\n")


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_mode():
    """
    Interactive mode - chat with the agent using custom tools.

    For AI/ML Scientists:
    This is like a REPL for your agent. Useful for testing and
    understanding agent behavior.
    """

    print("\n" + "="*70)
    print("Custom Tool Agent - Interactive Mode")
    print("="*70)
    print("\nAvailable tools:")
    print("  - Database: Query sales data")
    print("  - Weather API: Get current weather")
    print("  - Stock API: Get stock prices")
    print("  - Files: Read, write, list files")
    print("\nType 'quit' to exit")
    print("="*70 + "\n")

    # Initialize
    if os.getenv("DB_TYPE", "sqlite") == "sqlite":
        initialize_test_database()

    agent = create_custom_tool_agent()

    # Interactive loop
    while True:
        query = input("\nYour question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not query:
            continue

        try:
            result = agent.invoke({"input": query})
            print(f"\n{result['output']}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
            continue

        except Exception as e:
            print(f"\n❌ Error: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    """
    Run the custom tool agent.

    Usage:
        # Run examples
        python custom_tool_agent.py

        # Interactive mode
        python custom_tool_agent.py --interactive

    Environment variables needed:
        - SAGEMAKER_ENDPOINT_NAME (or SAGEMAKER_ENDPOINT_URL for mock)
        - WEATHER_API_KEY (optional, for weather queries)
        - STOCK_API_KEY (optional, for stock queries)
        - DB_TYPE (optional, defaults to sqlite)
    """

    import argparse

    parser = argparse.ArgumentParser(description="Custom Tool Agent")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        run_example_queries()
