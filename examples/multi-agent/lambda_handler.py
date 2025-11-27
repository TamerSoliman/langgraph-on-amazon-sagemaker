"""
AWS Lambda Handler for Multi-Agent Workflow

This module provides the Lambda entry point for the multi-agent system.
It handles API Gateway events, executes the agent workflow, and returns responses.

For AI/ML Scientists:
Think of Lambda as serverless compute - you write the code, AWS runs it when
needed. No servers to manage, pay only for execution time. Perfect for ML
inference APIs.
"""

import json
import os
import sys
from typing import Dict, Any
import traceback

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from multi_agent_graph import create_multi_agent_graph, MultiAgentState


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.

    This is the entry point called by AWS Lambda when the function is invoked.

    Args:
        event: API Gateway event containing request data
            Structure: {
                'body': '{"question": "What is X?"}',
                'headers': {...},
                'requestContext': {...}
            }
        context: Lambda context object with runtime information
            Contains: request_id, function_name, memory_limit, etc.

    Returns:
        API Gateway response dict:
            {
                'statusCode': 200,
                'headers': {...},
                'body': '{"answer": "...", "metadata": {...}}'
            }

    For AI/ML Scientists:
    This is like the main() function for your inference server. It:
    1. Receives HTTP request (question)
    2. Runs inference (multi-agent workflow)
    3. Returns HTTP response (answer)
    """

    print(f"[LAMBDA] Request ID: {context.request_id}")
    print(f"[LAMBDA] Memory limit: {context.memory_limit_in_mb}MB")

    try:
        # ---------------------------------------------------------------------
        # 1. Parse input
        # ---------------------------------------------------------------------

        # Extract request body
        if 'body' not in event:
            return error_response(400, "Missing request body")

        body = json.loads(event['body'])

        # Validate required field
        if 'question' not in body:
            return error_response(400, "Missing 'question' field in request body")

        question = body['question']

        print(f"[LAMBDA] Question: {question}")

        # Optional: Extract conversation history if provided
        chat_history = body.get('chat_history', [])

        # ---------------------------------------------------------------------
        # 2. Execute multi-agent workflow
        # ---------------------------------------------------------------------

        # Create the agent graph
        # For AI/ML Scientists: This is like loading your model
        app = create_multi_agent_graph()

        # Initialize state
        initial_state: MultiAgentState = {
            "question": question,
            "research": "",
            "draft": "",
            "review_feedback": "",
            "final_answer": "",
            "revision_count": 0
        }

        # Execute the workflow
        # For AI/ML Scientists: This is the forward pass through your pipeline
        print("[LAMBDA] Starting multi-agent execution...")

        final_state = app.invoke(
            initial_state,
            config={
                "recursion_limit": 20  # Prevent infinite loops (max 20 steps)
            }
        )

        print("[LAMBDA] Multi-agent execution complete")

        # ---------------------------------------------------------------------
        # 3. Format response
        # ---------------------------------------------------------------------

        # Extract final answer
        answer = final_state.get("final_answer", "")

        if not answer:
            # Fallback if workflow didn't produce final answer
            answer = final_state.get("draft", "Unable to generate answer")

        # Collect metadata about the execution
        metadata = {
            "revision_count": final_state.get("revision_count", 0),
            "research_length": len(final_state.get("research", "")),
            "draft_length": len(final_state.get("draft", "")),
            "request_id": context.request_id
        }

        # Build response
        response_body = {
            "answer": answer,
            "metadata": metadata
        }

        print(f"[LAMBDA] Response: {len(answer)} chars, {metadata['revision_count']} revisions")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"  # CORS
            },
            "body": json.dumps(response_body)
        }

    except json.JSONDecodeError as e:
        print(f"[LAMBDA] JSON parsing error: {e}")
        return error_response(400, f"Invalid JSON in request body: {str(e)}")

    except Exception as e:
        print(f"[LAMBDA] Error: {e}")
        print(traceback.format_exc())
        return error_response(500, f"Internal server error: {str(e)}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def error_response(status_code: int, message: str) -> Dict[str, Any]:
    """
    Creates a standardized error response.

    Args:
        status_code: HTTP status code (400, 500, etc.)
        message: Error message to return

    Returns:
        API Gateway response dict
    """
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "error": message
        })
    }


# =============================================================================
# HEALTH CHECK HANDLER
# =============================================================================

def health_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.

    This is a lightweight endpoint that confirms the Lambda function is
    running and can respond to requests. Used by load balancers and
    monitoring systems.

    Returns:
        200 OK with health status
    """
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "status": "healthy",
            "function_name": context.function_name,
            "memory_limit": context.memory_limit_in_mb
        })
    }


# =============================================================================
# VERSION HANDLER
# =============================================================================

def version_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Version information endpoint.

    Returns details about the deployed code version, dependencies, etc.
    Useful for debugging and ensuring correct version is deployed.

    Returns:
        200 OK with version information
    """

    # Try to get version from environment variable (set during deployment)
    version = os.getenv("APP_VERSION", "unknown")

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "version": version,
            "function_name": context.function_name,
            "python_version": sys.version
        })
    }


# =============================================================================
# LOCAL TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the Lambda handler locally.

    This simulates an API Gateway event and Lambda context for testing
    without deploying to AWS.

    Usage:
        python lambda_handler.py
    """

    # Mock Lambda context
    class MockContext:
        request_id = "test-request-123"
        function_name = "multi-agent-test"
        memory_limit_in_mb = 1024

    # Mock API Gateway event
    test_event = {
        "body": json.dumps({
            "question": "What were the main causes of the 2008 financial crisis?"
        }),
        "headers": {
            "Content-Type": "application/json"
        },
        "requestContext": {
            "requestId": "test-123"
        }
    }

    print("="*70)
    print("Testing Lambda Handler Locally")
    print("="*70)

    # Invoke handler
    context = MockContext()
    response = handler(test_event, context)

    # Display response
    print("\nResponse:")
    print(f"Status: {response['statusCode']}")
    print(f"Headers: {response['headers']}")

    body = json.loads(response['body'])
    print(f"\nAnswer: {body.get('answer', 'N/A')}")
    print(f"Metadata: {body.get('metadata', {})}")

    print("\n" + "="*70)

    # Test error case
    print("\nTesting Error Case (Missing Question):")
    error_event = {
        "body": json.dumps({}),  # Missing question
        "headers": {},
        "requestContext": {}
    }

    error_response = handler(error_event, context)
    print(f"Status: {error_response['statusCode']}")
    print(f"Body: {error_response['body']}")

    # Test health check
    print("\nTesting Health Check:")
    health_response = health_handler({}, context)
    print(f"Status: {health_response['statusCode']}")
    print(f"Body: {health_response['body']}")
