"""
Pytest Configuration and Fixtures

This file contains shared test fixtures used across all test files.

For AI/ML Scientists:
- Fixtures = reusable test setup code (like @pytest.fixture in pytest)
- conftest.py = special file that pytest automatically loads
- Fixtures here are available to all test files without imports

Think of fixtures as "test data generators" or "mock objects"
that you can inject into your tests.
"""

import pytest
import json
import os
from typing import Dict, Any
from unittest.mock import Mock, MagicMock


# ============================================================================
# Environment Setup Fixtures
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Set up environment variables for testing

    scope="session": Runs once for entire test session
    autouse=True: Runs automatically (don't need to request it)

    For AI/ML Scientists:
        This ensures all tests have required environment variables,
        even if running locally without AWS setup
    """
    os.environ['SAGEMAKER_ENDPOINT_NAME'] = 'test-mistral-endpoint'
    os.environ['TAVILY_SECRET_ARN'] = 'arn:aws:secretsmanager:us-east-1:123456789012:secret:tavily-test'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    os.environ['LOG_LEVEL'] = 'DEBUG'

    yield  # Tests run here

    # Cleanup (if needed)
    pass


# ============================================================================
# Mock LLM Response Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_tool_call_response() -> str:
    """
    Mock LLM response that calls a tool

    Returns:
        str: XML-formatted tool call

    For AI/ML Scientists:
        This simulates what the LLM returns when it wants to call a tool.
        Real LLM → SageMaker endpoint (slow, costs money)
        Mock LLM → returns this string (fast, free)
    """
    return "<tool>tavily_search_results_json</tool><tool_input>latest UK storm</tool_input>"


@pytest.fixture
def mock_llm_final_answer_response() -> str:
    """
    Mock LLM response with final answer

    Returns:
        str: XML-formatted final answer
    """
    return "<final_answer>The capital of France is Paris.</final_answer>"


@pytest.fixture
def mock_llm_response_with_reasoning() -> str:
    """
    Mock LLM response with reasoning before tool call

    Returns:
        str: More realistic LLM output with thought process
    """
    return """I need to search for information about the latest storm in the UK.

<tool>tavily_search_results_json</tool><tool_input>latest storm to hit UK</tool_input>"""


# ============================================================================
# Mock Tool Result Fixtures
# ============================================================================

@pytest.fixture
def mock_tavily_search_result() -> str:
    """
    Mock result from Tavily search tool

    Returns:
        str: JSON string that Tavily API would return
    """
    return json.dumps([{
        'url': 'https://www.example.com/storm-henk',
        'content': 'Storm Henk hit the UK on January 2, 2024, causing damage in south-west England.'
    }])


# ============================================================================
# Mock SageMaker Client Fixtures
# ============================================================================

@pytest.fixture
def mock_sagemaker_client():
    """
    Mock boto3 SageMaker client

    Returns:
        Mock: A mock SageMaker client that returns predefined responses

    Usage in tests:
        def test_something(mock_sagemaker_client):
            # Client is already configured
            response = mock_sagemaker_client.invoke_endpoint(...)
    """
    mock_client = MagicMock()

    # Mock invoke_endpoint response
    mock_response = {
        'Body': MagicMock(),
        'ContentType': 'application/json',
    }

    # Setup the Body.read() method to return mock LLM response
    mock_llm_output = json.dumps([{
        "generated_text": "<final_answer>Test answer</final_answer>"
    }])
    mock_response['Body'].read.return_value = mock_llm_output.encode('utf-8')

    mock_client.invoke_endpoint.return_value = mock_response

    return mock_client


# ============================================================================
# Mock Secrets Manager Fixtures
# ============================================================================

@pytest.fixture
def mock_secrets_manager_client():
    """
    Mock boto3 Secrets Manager client

    Returns:
        Mock: A mock Secrets Manager client
    """
    mock_client = MagicMock()

    # Mock get_secret_value response
    mock_client.get_secret_value.return_value = {
        'SecretString': json.dumps({'api_key': 'tvly-test-key-12345'})
    }

    return mock_client


# ============================================================================
# Test Event Fixtures (for Lambda testing)
# ============================================================================

@pytest.fixture
def api_gateway_event() -> Dict[str, Any]:
    """
    Mock API Gateway event

    Returns:
        dict: Event structure that API Gateway sends to Lambda

    For AI/ML Scientists:
        When a user calls your API:
        User → API Gateway → Lambda (receives this event)
    """
    return {
        'body': json.dumps({
            'question': 'What is the capital of France?',
            'chat_history': []
        }),
        'headers': {
            'Content-Type': 'application/json'
        },
        'requestContext': {
            'requestId': 'test-request-id-12345'
        },
        'httpMethod': 'POST',
        'path': '/ask'
    }


@pytest.fixture
def lambda_context():
    """
    Mock Lambda context

    Returns:
        Mock: A mock Lambda context object

    For AI/ML Scientists:
        Lambda context provides runtime information:
        - How much time remaining before timeout
        - Request ID for logging
        - Function name, version, etc.
    """
    context = Mock()
    context.request_id = 'test-request-id-12345'
    context.function_name = 'langgraph-agent'
    context.function_version = '$LATEST'
    context.memory_limit_in_mb = '1024'
    context.get_remaining_time_in_millis = Mock(return_value=300000)  # 5 minutes

    return context


# ============================================================================
# Mock Tool Fixtures
# ============================================================================

@pytest.fixture
def mock_tavily_tool():
    """
    Mock Tavily search tool

    Returns:
        Mock: A tool that returns predefined search results
    """
    from langchain.tools import Tool

    def mock_search(query: str) -> str:
        return json.dumps([{
            'url': 'https://www.example.com/result',
            'content': f'Mock search result for: {query}'
        }])

    return Tool(
        name="tavily_search_results_json",
        description="Mock search tool",
        func=mock_search
    )


# ============================================================================
# Parametrized Test Data
# ============================================================================

@pytest.fixture(params=[
    "What is the capital of France?",
    "Who won the 2022 World Cup?",
    "What is 2 + 2?",
    "Explain quantum computing in simple terms",
])
def sample_questions(request):
    """
    Parametrized fixture providing multiple test questions

    Usage:
        def test_agent_answers(sample_questions):
            # This test runs 4 times, once for each question
            response = agent.ask(sample_questions)
            assert response is not None
    """
    return request.param


# ============================================================================
# Performance Metrics Fixtures
# ============================================================================

@pytest.fixture
def performance_tracker():
    """
    Track performance metrics during tests

    Returns:
        dict: Dictionary to store metrics (latency, token counts, etc.)
    """
    return {
        'latency_ms': [],
        'token_counts': [],
        'tool_calls': [],
        'errors': []
    }


# ============================================================================
# Database Fixtures (if using database tools)
# ============================================================================

# Uncomment if you add database tools:
#
# @pytest.fixture
# def mock_dynamodb_table():
#     """Mock DynamoDB table for testing"""
#     from moto import mock_dynamodb
#     import boto3
#
#     with mock_dynamodb():
#         dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
#         table = dynamodb.create_table(
#             TableName='test-table',
#             KeySchema=[{'AttributeName': 'id', 'KeyType': 'HASH'}],
#             AttributeDefinitions=[{'AttributeName': 'id', 'AttributeType': 'S'}],
#             BillingMode='PAY_PER_REQUEST'
#         )
#
#         # Add test data
#         table.put_item(Item={'id': 'test-1', 'name': 'Test Item'})
#
#         yield table


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture
def temp_file(tmp_path):
    """
    Create a temporary file for testing

    Args:
        tmp_path: pytest built-in fixture for temp directories

    Returns:
        Path: Path to temp file

    For AI/ML Scientists:
        tmp_path is a pytest built-in that creates a unique temp directory
        for each test. It's automatically cleaned up after the test.
    """
    test_file = tmp_path / "test_data.json"
    test_file.write_text(json.dumps({'test': 'data'}))

    yield test_file

    # Cleanup happens automatically (tmp_path is deleted)
