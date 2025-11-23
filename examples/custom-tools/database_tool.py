"""
Database Tool Example

Demonstrates how to create a custom tool that queries a database.

For AI/ML Scientists:
This is like a retrieval component that fetches structured data instead of
unstructured documents. Think of it as RAG for databases.

Security Note:
This example uses safe, parameterized queries to prevent SQL injection.
NEVER execute user-provided SQL directly in production!
"""

import os
import sys
from typing import Optional
from langchain.tools import tool

# For production, use a real database client
# Supported options:
# - psycopg2 for PostgreSQL
# - pymysql for MySQL
# - sqlite3 for SQLite (built-in, no install needed)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database configuration from environment variables
# For AI/ML Scientists: Never hardcode credentials! Always use env vars or
# secrets management services (AWS Secrets Manager, etc.)

DB_TYPE = os.getenv("DB_TYPE", "sqlite")  # sqlite, postgres, mysql
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "sales_db")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# SQLite file for testing (no server needed)
SQLITE_FILE = os.getenv("SQLITE_FILE", "/tmp/test_sales.db")


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_db_connection():
    """
    Creates database connection based on configuration.

    For AI/ML Scientists:
    Connection pooling would be better for production (reuse connections
    instead of creating new ones). Similar to how you reuse model instances
    for multiple inferences instead of reloading for each prediction.

    Returns:
        Database connection object

    Raises:
        Exception if connection fails
    """

    if DB_TYPE == "sqlite":
        # SQLite: File-based database, perfect for testing
        import sqlite3
        conn = sqlite3.connect(SQLITE_FILE)
        return conn

    elif DB_TYPE == "postgres":
        # PostgreSQL: Production-grade relational database
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            return conn
        except ImportError:
            raise Exception("psycopg2 not installed. Run: pip install psycopg2-binary")

    elif DB_TYPE == "mysql":
        # MySQL: Another popular relational database
        try:
            import pymysql
            conn = pymysql.connect(
                host=DB_HOST,
                port=int(DB_PORT),
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            return conn
        except ImportError:
            raise Exception("pymysql not installed. Run: pip install pymysql")

    else:
        raise ValueError(f"Unsupported DB_TYPE: {DB_TYPE}")


# =============================================================================
# TOOL IMPLEMENTATION
# =============================================================================

@tool
def query_sales_database(question: str) -> str:
    """
    Query the sales database for business metrics.

    Use this tool when the user asks about:
    - Sales revenue or totals
    - Number of units sold
    - Top products or customers
    - Sales trends over time

    Examples of good questions:
    - "What was our total revenue last month?"
    - "How many units of product X did we sell?"
    - "Who are our top 5 customers by revenue?"

    Args:
        question: Natural language question about sales data

    Returns:
        Query result formatted as text, or error message

    For AI/ML Scientists:
    This tool translates natural language to SQL (like a semantic parser),
    executes the query, and returns results. In production, you might use
    a specialized NL-to-SQL model (e.g., T5 fine-tuned on Spider dataset).
    """

    print(f"[DATABASE TOOL] Processing query: {question}")

    try:
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()

        # =====================================================================
        # Query Selection Logic
        # =====================================================================
        # For AI/ML Scientists:
        # This is a simple rule-based NL-to-SQL system. In production, consider:
        # 1. Few-shot prompting with LLM to generate SQL
        # 2. Fine-tuned NL-to-SQL model
        # 3. Predefined query templates (safest)

        question_lower = question.lower()

        # Query 1: Total revenue
        if any(keyword in question_lower for keyword in ["revenue", "sales total", "total sales"]):
            query = """
                SELECT SUM(amount) as total_revenue
                FROM sales
                WHERE date >= date('now', '-30 days')
            """
            cursor.execute(query)
            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                return f"Total revenue in the last 30 days: ${result[0]:,.2f}"
            else:
                return "No sales data found for the last 30 days."

        # Query 2: Units sold
        elif any(keyword in question_lower for keyword in ["units", "quantity", "how many"]):
            # Check if asking about specific product
            # In production, use NER to extract product name
            if "product" in question_lower:
                # For demo, assume question contains product name after "product"
                words = question_lower.split()
                if "product" in words:
                    idx = words.index("product")
                    if idx + 1 < len(words):
                        product_name = words[idx + 1].strip("?.,")
                    else:
                        product_name = None
                else:
                    product_name = None

                if product_name:
                    # Parameterized query (prevents SQL injection)
                    query = """
                        SELECT SUM(quantity) as total_units
                        FROM sales
                        WHERE product_name = ?
                    """
                    cursor.execute(query, (product_name,))
                else:
                    query = "SELECT SUM(quantity) as total_units FROM sales"
                    cursor.execute(query)
            else:
                query = "SELECT SUM(quantity) as total_units FROM sales"
                cursor.execute(query)

            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                return f"Total units sold: {result[0]:,}"
            else:
                return "No sales data found."

        # Query 3: Top products
        elif "top" in question_lower and ("product" in question_lower or "item" in question_lower):
            limit = 5  # Default top 5

            query = """
                SELECT product_name, SUM(amount) as revenue
                FROM sales
                GROUP BY product_name
                ORDER BY revenue DESC
                LIMIT ?
            """
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            conn.close()

            if results:
                formatted = "Top products by revenue:\n"
                for i, (product, revenue) in enumerate(results, 1):
                    formatted += f"{i}. {product}: ${revenue:,.2f}\n"
                return formatted.strip()
            else:
                return "No sales data found."

        # Query 4: Top customers
        elif "top" in question_lower and "customer" in question_lower:
            limit = 5

            query = """
                SELECT customer_name, SUM(amount) as revenue
                FROM sales
                GROUP BY customer_name
                ORDER BY revenue DESC
                LIMIT ?
            """
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            conn.close()

            if results:
                formatted = "Top customers by revenue:\n"
                for i, (customer, revenue) in enumerate(results, 1):
                    formatted += f"{i}. {customer}: ${revenue:,.2f}\n"
                return formatted.strip()
            else:
                return "No sales data found."

        # Fallback: Unable to understand question
        else:
            conn.close()
            return """I can answer questions about:
            - Total revenue ("What was our revenue?")
            - Units sold ("How many units did we sell?")
            - Top products ("What are our top products?")
            - Top customers ("Who are our best customers?")

            Please rephrase your question to match one of these patterns."""

    except Exception as e:
        print(f"[DATABASE TOOL] Error: {e}")
        return f"Database error: {str(e)}. Please try rephrasing your question."


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def initialize_test_database():
    """
    Creates a test SQLite database with sample data.

    For AI/ML Scientists:
    This is like creating synthetic data for testing your model before
    deploying to production. Lets you test the tool without a real database.
    """

    import sqlite3

    print(f"[DATABASE TOOL] Initializing test database at {SQLITE_FILE}")

    conn = sqlite3.connect(SQLITE_FILE)
    cursor = conn.cursor()

    # Create sales table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            date DATE,
            product_name TEXT,
            customer_name TEXT,
            quantity INTEGER,
            amount REAL
        )
    """)

    # Check if data already exists
    cursor.execute("SELECT COUNT(*) FROM sales")
    if cursor.fetchone()[0] > 0:
        print("[DATABASE TOOL] Test data already exists")
        conn.close()
        return

    # Insert sample data
    sample_data = [
        ('2024-01-15', 'Widget A', 'Acme Corp', 100, 5000.00),
        ('2024-01-16', 'Widget B', 'TechCo', 50, 7500.00),
        ('2024-01-17', 'Widget A', 'InnovateLLC', 75, 3750.00),
        ('2024-01-18', 'Widget C', 'Acme Corp', 200, 10000.00),
        ('2024-01-19', 'Widget B', 'StartupXYZ', 30, 4500.00),
        ('2024-01-20', 'Widget A', 'TechCo', 120, 6000.00),
        ('2024-01-21', 'Widget C', 'InnovateLLC', 90, 4500.00),
        ('2024-01-22', 'Widget B', 'Acme Corp', 60, 9000.00),
        ('2024-01-23', 'Widget A', 'StartupXYZ', 85, 4250.00),
        ('2024-01-24', 'Widget C', 'TechCo', 150, 7500.00),
    ]

    cursor.executemany("""
        INSERT INTO sales (date, product_name, customer_name, quantity, amount)
        VALUES (?, ?, ?, ?, ?)
    """, sample_data)

    conn.commit()
    conn.close()

    print(f"[DATABASE TOOL] Inserted {len(sample_data)} sample records")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the database tool in isolation.

    Usage:
        python database_tool.py
    """

    print("="*70)
    print("Database Tool Test")
    print("="*70)

    # Initialize test database (if using SQLite)
    if DB_TYPE == "sqlite":
        initialize_test_database()

    # Test queries
    test_questions = [
        "What was our total revenue?",
        "How many units did we sell?",
        "What are our top products?",
        "Who are our best customers?",
        "How many units of product Widget A did we sell?",
    ]

    for question in test_questions:
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print(f"{'='*70}")

        result = query_sales_database.invoke(question)

        print(f"Result:\n{result}")

    print("\n" + "="*70)
    print("All tests complete!")
    print("="*70)
