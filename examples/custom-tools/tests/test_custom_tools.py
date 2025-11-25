"""
Unit Tests for Custom Tools

Tests each custom tool in isolation with mocked dependencies.

For AI/ML Scientists:
Testing tools separately ensures they work correctly before integrating
into the agent. Like testing dataset loaders before training.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# =============================================================================
# DATABASE TOOL TESTS
# =============================================================================

def test_database_tool_revenue_query():
    """Test database tool with revenue query."""

    from database_tool import query_sales_database, initialize_test_database

    # Initialize test database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        os.environ['SQLITE_FILE'] = tmp.name
        os.environ['DB_TYPE'] = 'sqlite'

    try:
        initialize_test_database()

        # Query total revenue
        result = query_sales_database.invoke("What was our total revenue?")

        # Should return revenue information
        assert "revenue" in result.lower() or "total" in result.lower()
        assert "$" in result  # Should have dollar amounts

    finally:
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def test_database_tool_units_query():
    """Test database tool with units sold query."""

    from database_tool import query_sales_database, initialize_test_database

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        os.environ['SQLITE_FILE'] = tmp.name
        os.environ['DB_TYPE'] = 'sqlite'

    try:
        initialize_test_database()

        result = query_sales_database.invoke("How many units did we sell?")

        assert "units" in result.lower()
        # Should contain numbers
        assert any(char.isdigit() for char in result)

    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def test_database_tool_error_handling():
    """Test database tool handles invalid queries gracefully."""

    from database_tool import query_sales_database

    # Set invalid database
    os.environ['SQLITE_FILE'] = '/nonexistent/path/db.sqlite'
    os.environ['DB_TYPE'] = 'sqlite'

    result = query_sales_database.invoke("test query")

    # Should return error message, not raise exception
    assert "error" in result.lower() or "can answer questions about" in result.lower()


# =============================================================================
# API TOOL TESTS
# =============================================================================

def test_weather_api_tool_success():
    """Test weather API tool with mocked API response."""

    from api_tool import get_current_weather

    # Mock successful API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'location': {
            'name': 'London',
            'country': 'UK'
        },
        'current': {
            'temp_f': 72.0,
            'temp_c': 22.0,
            'condition': {'text': 'Partly cloudy'},
            'humidity': 65,
            'wind_mph': 10.0
        }
    }

    with patch('api_tool.requests.get', return_value=mock_response):
        with patch.dict(os.environ, {'WEATHER_API_KEY': 'test_key'}):
            result = get_current_weather.invoke("London")

    # Verify result contains expected information
    assert "London" in result
    assert "72" in result or "72.0" in result  # Temperature
    assert "Partly cloudy" in result


def test_weather_api_tool_no_key():
    """Test weather API tool when API key is not configured."""

    from api_tool import get_current_weather

    # Clear API key
    with patch.dict(os.environ, {'WEATHER_API_KEY': ''}, clear=True):
        result = get_current_weather.invoke("London")

    # Should return helpful error message
    assert "API key not configured" in result or "not configured" in result.lower()


def test_weather_api_tool_timeout():
    """Test weather API tool handles timeouts."""

    from api_tool import get_current_weather
    import requests

    # Mock timeout
    with patch('api_tool.requests.get', side_effect=requests.Timeout("Timeout")):
        with patch.dict(os.environ, {'WEATHER_API_KEY': 'test_key'}):
            result = get_current_weather.invoke("London")

    assert "timeout" in result.lower() or "error" in result.lower()


def test_stock_api_tool_success():
    """Test stock API tool with mocked response."""

    from api_tool import get_stock_price

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'Global Quote': {
            '01. symbol': 'AAPL',
            '05. price': '150.25',
            '09. change': '2.50',
            '10. change percent': '+1.69%',
            '06. volume': '75000000'
        }
    }

    with patch('api_tool.requests.get', return_value=mock_response):
        with patch.dict(os.environ, {'STOCK_API_KEY': 'test_key'}):
            result = get_stock_price.invoke("AAPL")

    assert "AAPL" in result
    assert "150.25" in result
    assert "$" in result


# =============================================================================
# FILE SYSTEM TOOL TESTS
# =============================================================================

def test_read_file_success():
    """Test reading a file successfully."""

    from file_system_tool import read_file

    # Create temporary directory and file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set allowed directory
        os.environ['ALLOWED_FILE_DIR'] = tmpdir

        # Create test file
        test_file = os.path.join(tmpdir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write("Hello, this is a test file!")

        # Read file
        result = read_file.invoke("test.txt")

        assert "Hello, this is a test file!" in result
        assert "test.txt" in result


def test_read_file_not_found():
    """Test reading non-existent file."""

    from file_system_tool import read_file

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['ALLOWED_FILE_DIR'] = tmpdir

        result = read_file.invoke("nonexistent.txt")

        assert "not found" in result.lower() or "error" in result.lower()


def test_read_file_path_traversal_protection():
    """Test that path traversal attacks are blocked."""

    from file_system_tool import read_file

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['ALLOWED_FILE_DIR'] = tmpdir

        # Try to read file outside allowed directory
        result = read_file.invoke("../../../etc/passwd")

        assert "access denied" in result.lower() or "error" in result.lower()


def test_write_file_success():
    """Test writing a file successfully."""

    from file_system_tool import write_file

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['ALLOWED_WRITE_DIR'] = tmpdir

        # Write file
        content = "Test content for writing"
        result = write_file.invoke("output.txt", content)

        assert "success" in result.lower()

        # Verify file was created
        output_path = os.path.join(tmpdir, 'output.txt')
        assert os.path.exists(output_path)

        with open(output_path, 'r') as f:
            assert f.read() == content


def test_write_file_disallowed_extension():
    """Test that writing disallowed file types is blocked."""

    from file_system_tool import write_file

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['ALLOWED_WRITE_DIR'] = tmpdir

        # Try to write .exe file (not in allowed extensions)
        result = write_file.invoke("malware.exe", "bad content")

        assert "not allowed" in result.lower() or "error" in result.lower()


def test_list_files():
    """Test listing files in directory."""

    from file_system_tool import list_files

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['ALLOWED_FILE_DIR'] = tmpdir

        # Create some test files
        with open(os.path.join(tmpdir, 'file1.txt'), 'w') as f:
            f.write("test")
        with open(os.path.join(tmpdir, 'file2.json'), 'w') as f:
            f.write("{}")

        # List files
        result = list_files.invoke("")

        assert "file1.txt" in result
        assert "file2.json" in result


def test_search_files():
    """Test searching for text in files."""

    from file_system_tool import search_files

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['ALLOWED_FILE_DIR'] = tmpdir

        # Create files with searchable content
        with open(os.path.join(tmpdir, 'file1.txt'), 'w') as f:
            f.write("This file contains ERROR message")
        with open(os.path.join(tmpdir, 'file2.txt'), 'w') as f:
            f.write("This file is clean")

        # Search for "ERROR"
        result = search_files.invoke("ERROR")

        assert "file1.txt" in result
        assert "file2.txt" not in result or "0 occurrence" in result


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_custom_tool_agent_initialization():
    """Test that the agent can be initialized with all tools."""

    from custom_tool_agent import create_custom_tools

    # Mock SageMaker LLM
    with patch('custom_tool_agent.create_sagemaker_llm'):
        tools = create_custom_tools()

    # Should have all tools
    assert len(tools) >= 6  # database, weather, stock, read, write, list

    # Check tool names
    tool_names = [tool.name for tool in tools]
    assert "query_sales_database" in tool_names
    assert "get_current_weather" in tool_names
    assert "read_file" in tool_names


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    """
    Run tests directly (without pytest).

    Usage:
        python test_custom_tools.py
    """
    print("Running custom tool tests...")
    pytest.main([__file__, "-v", "--tb=short"])
