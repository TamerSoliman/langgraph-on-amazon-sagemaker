"""
API Tool Example

Demonstrates how to create a custom tool that calls external REST APIs.

For AI/ML Scientists:
This extends your model's knowledge beyond its training cutoff by fetching
real-time data from external services. Think of it as augmenting your model
with live data sources.
"""

import os
import json
from typing import Dict, Any, Optional
import requests
from langchain.tools import tool


# =============================================================================
# CONFIGURATION
# =============================================================================

# API credentials from environment variables
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
STOCK_API_KEY = os.getenv("STOCK_API_KEY", "")

# API endpoints
WEATHER_API_URL = "https://api.weatherapi.com/v1/current.json"
STOCK_API_URL = "https://www.alphavantage.co/query"

# Request timeout (seconds)
API_TIMEOUT = 5


# =============================================================================
# WEATHER TOOL
# =============================================================================

@tool
def get_current_weather(location: str) -> str:
    """
    Get current weather conditions for a location.

    Use this tool when the user asks about:
    - Current weather
    - Temperature
    - Weather conditions (sunny, rainy, etc.)
    - "What's the weather in [location]?"

    Args:
        location: City name, zip code, or coordinates (e.g., "Paris", "10001", "48.8567,2.3508")

    Returns:
        Current weather information formatted as text

    For AI/ML Scientists:
    This is a simple HTTP GET request to a weather API. The API returns JSON,
    which we parse and format into natural language for the LLM to use.
    """

    print(f"[WEATHER API] Fetching weather for: {location}")

    # Check if API key is configured
    if not WEATHER_API_KEY:
        return """Weather API key not configured. To use this tool:
        1. Get a free API key from https://www.weatherapi.com/
        2. Set environment variable: export WEATHER_API_KEY="your-key"
        """

    try:
        # Make API request
        # For AI/ML Scientists: This is an I/O-bound operation (waiting for network).
        # In production, consider using async/await for parallel requests.
        response = requests.get(
            WEATHER_API_URL,
            params={
                "key": WEATHER_API_KEY,
                "q": location,
                "aqi": "no"  # Don't need air quality data
            },
            timeout=API_TIMEOUT
        )

        # Check for errors
        if response.status_code == 400:
            return f"Error: Invalid location '{location}'. Please provide a valid city name or coordinates."
        elif response.status_code == 401:
            return "Error: Invalid API key. Please check your WEATHER_API_KEY environment variable."
        elif response.status_code != 200:
            return f"Error: Weather API returned status {response.status_code}"

        # Parse JSON response
        data = response.json()

        # Extract relevant fields
        location_name = data['location']['name']
        country = data['location']['country']
        temp_f = data['current']['temp_f']
        temp_c = data['current']['temp_c']
        condition = data['current']['condition']['text']
        humidity = data['current']['humidity']
        wind_mph = data['current']['wind_mph']

        # Format as natural language
        # For AI/ML Scientists: We're converting structured data (JSON) to
        # unstructured text for the LLM. The LLM is better at understanding
        # natural language than raw JSON.
        result = f"""Current weather in {location_name}, {country}:
        - Temperature: {temp_f}°F ({temp_c}°C)
        - Conditions: {condition}
        - Humidity: {humidity}%
        - Wind: {wind_mph} mph
        """

        print(f"[WEATHER API] ✓ Success")
        return result.strip()

    except requests.Timeout:
        return f"Error: Weather API request timed out after {API_TIMEOUT} seconds. Please try again."

    except requests.RequestException as e:
        return f"Error: Network error when calling weather API: {str(e)}"

    except (KeyError, ValueError) as e:
        return f"Error: Unexpected response format from weather API: {str(e)}"

    except Exception as e:
        return f"Error: Unexpected error fetching weather: {str(e)}"


# =============================================================================
# STOCK PRICE TOOL
# =============================================================================

@tool
def get_stock_price(symbol: str) -> str:
    """
    Get current stock price and basic info for a ticker symbol.

    Use this tool when the user asks about:
    - Stock prices ("What's the price of AAPL?")
    - Market data for companies
    - Trading information

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "GOOGL", "TSLA")

    Returns:
        Stock price and trading information

    For AI/ML Scientists:
    Similar to weather API but for financial data. Note that free APIs have
    rate limits (e.g., 5 calls/minute). In production, consider caching
    results or using paid APIs with higher limits.
    """

    print(f"[STOCK API] Fetching stock data for: {symbol}")

    if not STOCK_API_KEY:
        return """Stock API key not configured. To use this tool:
        1. Get a free API key from https://www.alphavantage.co/support/#api-key
        2. Set environment variable: export STOCK_API_KEY="your-key"
        Note: Free tier is limited to 5 API calls per minute.
        """

    try:
        # API request
        response = requests.get(
            STOCK_API_URL,
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": STOCK_API_KEY
            },
            timeout=API_TIMEOUT
        )

        if response.status_code != 200:
            return f"Error: Stock API returned status {response.status_code}"

        data = response.json()

        # Check for error messages in response
        if "Error Message" in data:
            return f"Error: {data['Error Message']}"

        if "Note" in data:
            # Rate limit message
            return f"Error: {data['Note']}"

        # Extract quote data
        quote = data.get("Global Quote", {})

        if not quote:
            return f"No data found for symbol '{symbol}'. Please check the ticker symbol."

        symbol_name = quote.get("01. symbol", symbol)
        price = float(quote.get("05. price", 0))
        change = float(quote.get("09. change", 0))
        change_percent = quote.get("10. change percent", "0%")
        volume = int(float(quote.get("06. volume", 0)))

        # Format result
        result = f"""Stock information for {symbol_name}:
        - Current Price: ${price:.2f}
        - Change: ${change:+.2f} ({change_percent})
        - Volume: {volume:,} shares
        """

        print(f"[STOCK API] ✓ Success")
        return result.strip()

    except requests.Timeout:
        return f"Error: Stock API request timed out after {API_TIMEOUT} seconds."

    except (KeyError, ValueError) as e:
        return f"Error: Unexpected response format from stock API: {str(e)}"

    except Exception as e:
        return f"Error: Unexpected error fetching stock data: {str(e)}"


# =============================================================================
# GENERIC API TOOL TEMPLATE
# =============================================================================

@tool
def call_generic_api(endpoint: str) -> str:
    """
    Call a generic REST API endpoint.

    This is a template for creating your own API tools.
    Customize this for your specific API needs.

    Use this when you need to call a custom internal API.

    Args:
        endpoint: API endpoint path (e.g., "/api/users/123")

    Returns:
        API response formatted as text

    For AI/ML Scientists:
    This is a generic wrapper you can customize for your internal APIs.
    Modify the base URL, authentication, and response parsing as needed.
    """

    # Configuration
    BASE_URL = os.getenv("CUSTOM_API_BASE_URL", "https://api.example.com")
    API_KEY = os.getenv("CUSTOM_API_KEY", "")

    try:
        # Build full URL
        url = f"{BASE_URL}{endpoint}"

        # Make request with authentication
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=API_TIMEOUT
        )

        if response.status_code != 200:
            return f"API error: HTTP {response.status_code}"

        # Parse response
        data = response.json()

        # Format for LLM (customize based on your API response structure)
        return json.dumps(data, indent=2)

    except Exception as e:
        return f"Error calling API: {str(e)}"


# =============================================================================
# ADVANCED: API TOOL WITH POST REQUEST
# =============================================================================

@tool
def create_api_resource(resource_type: str, data: str) -> str:
    """
    Create a resource via API POST request.

    Example of tool that modifies data (not just reads).

    For AI/ML Scientists:
    Be VERY careful with tools that modify data! Consider adding:
    1. Human-in-the-loop approval before executing
    2. Dry-run mode to preview changes
    3. Audit logging of all modifications
    4. Rollback capabilities

    Args:
        resource_type: Type of resource to create (e.g., "user", "order")
        data: JSON string with resource data

    Returns:
        Result of creation attempt
    """

    # Configuration
    BASE_URL = os.getenv("CUSTOM_API_BASE_URL", "https://api.example.com")
    API_KEY = os.getenv("CUSTOM_API_KEY", "")

    try:
        # Parse data
        payload = json.loads(data)

        # Make POST request
        url = f"{BASE_URL}/{resource_type}"

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=API_TIMEOUT
        )

        if response.status_code in [200, 201]:
            return f"Successfully created {resource_type}"
        else:
            return f"Error creating {resource_type}: HTTP {response.status_code}"

    except json.JSONDecodeError:
        return "Error: Invalid JSON data provided"

    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# HELPERS
# =============================================================================

def test_api_connectivity():
    """
    Test if APIs are reachable and credentials are valid.

    For AI/ML Scientists:
    Run this before deploying to production to catch configuration issues.
    Like checking if your data sources are accessible before training.
    """

    print("\nTesting API Connectivity...")
    print("="*70)

    # Test weather API
    if WEATHER_API_KEY:
        try:
            response = requests.get(
                WEATHER_API_URL,
                params={"key": WEATHER_API_KEY, "q": "London"},
                timeout=5
            )
            if response.status_code == 200:
                print("✓ Weather API: Connected")
            else:
                print(f"✗ Weather API: Error (HTTP {response.status_code})")
        except Exception as e:
            print(f"✗ Weather API: {str(e)}")
    else:
        print("⚠ Weather API: No API key configured")

    # Test stock API
    if STOCK_API_KEY:
        try:
            response = requests.get(
                STOCK_API_URL,
                params={"function": "GLOBAL_QUOTE", "symbol": "AAPL", "apikey": STOCK_API_KEY},
                timeout=5
            )
            if response.status_code == 200:
                print("✓ Stock API: Connected")
            else:
                print(f"✗ Stock API: Error (HTTP {response.status_code})")
        except Exception as e:
            print(f"✗ Stock API: {str(e)}")
    else:
        print("⚠ Stock API: No API key configured")

    print("="*70)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the API tools in isolation.

    Usage:
        export WEATHER_API_KEY="your-key"
        export STOCK_API_KEY="your-key"
        python api_tool.py
    """

    print("="*70)
    print("API Tool Test")
    print("="*70)

    # Test connectivity first
    test_api_connectivity()

    # Test weather tool
    if WEATHER_API_KEY:
        print("\n" + "="*70)
        print("Testing Weather Tool")
        print("="*70)

        test_locations = ["London", "New York", "Tokyo"]

        for location in test_locations:
            print(f"\nLocation: {location}")
            result = get_current_weather.invoke(location)
            print(result)
    else:
        print("\n⚠ Skipping weather tests (no API key)")

    # Test stock tool
    if STOCK_API_KEY:
        print("\n" + "="*70)
        print("Testing Stock Tool")
        print("="*70)

        test_symbols = ["AAPL", "GOOGL", "MSFT"]

        for symbol in test_symbols:
            print(f"\nSymbol: {symbol}")
            result = get_stock_price.invoke(symbol)
            print(result)

            # Rate limit: sleep between requests
            import time
            time.sleep(15)  # Free tier allows ~5 requests/minute
    else:
        print("\n⚠ Skipping stock tests (no API key)")

    print("\n" + "="*70)
    print("All tests complete!")
    print("="*70)
