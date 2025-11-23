"""
AWS Lambda Handler for LangGraph Agent

This is the entry point for AWS Lambda. When API Gateway receives a request,
it triggers this Lambda function, which executes the LangGraph agent.

For AI/ML Scientists:
- Lambda handler = main() function that AWS calls
- event = input data (user's question from API Gateway)
- context = runtime information (request ID, time remaining, etc.)
- Return value = sent back to API Gateway → user

Flow:
  API Gateway → lambda_handler.handler(event, context) → LangGraph agent → response
"""

import json
import os
import logging
from typing import Dict, Any

from langchain_core.agents import AgentFinish
from graph import create_agent_graph

# Configure logging
# CloudWatch Logs will capture these messages
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function

    Args:
        event: Input event from API Gateway
            Expected format:
            {
                "body": "{\"question\":\"What is 2+2?\",\"chat_history\":[]}",
                "headers": {...},
                "requestContext": {...}
            }

        context: Lambda context object
            Provides runtime information:
            - context.request_id: Unique ID for this invocation
            - context.get_remaining_time_in_millis(): Time before timeout
            - context.function_name: Lambda function name

    Returns:
        dict: Response formatted for API Gateway
            {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": "{\"answer\":\"4\",\"metadata\":{...}}"
            }
    """

    # Log request details for debugging
    logger.info(f"Request ID: {context.request_id}")
    logger.info(f"Remaining time: {context.get_remaining_time_in_millis()}ms")

    try:
        # STEP 1: Parse input from API Gateway
        # API Gateway wraps the actual request body in 'body' field (as a JSON string)
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)  # Direct invocation (testing)

        question = body.get('question')
        chat_history = body.get('chat_history', [])

        # Validate input
        if not question:
            logger.error("Missing 'question' field in request")
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'error': 'Missing required field: question',
                    'usage': 'POST /ask with body: {"question":"Your question here","chat_history":[]}'
                })
            }

        logger.info(f"Question received: {question[:100]}...")  # Log first 100 chars

        # STEP 2: Initialize LangGraph agent
        # This loads the graph definition and connects to SageMaker endpoint
        logger.info("Initializing LangGraph agent")
        app = create_agent_graph()

        # STEP 3: Execute agent
        logger.info("Executing agent")
        inputs = {
            "input": question,
            "chat_history": chat_history
        }

        result = None
        intermediate_steps = []
        step_count = 0

        # Stream through agent execution
        # Each iteration is one node in the graph (agent node, tools node, etc.)
        for state in app.stream(inputs):
            step_count += 1
            logger.debug(f"Step {step_count}: {list(state.keys())}")

            # Check timeout (Lambda has 5-minute max)
            remaining_time = context.get_remaining_time_in_millis()
            if remaining_time < 10000:  # Less than 10 seconds remaining
                logger.warning(f"Approaching timeout! {remaining_time}ms remaining")

            # Extract state updates
            if 'agent' in state:
                agent_state = state['agent']
                if 'agent_outcome' in agent_state:
                    outcome = agent_state['agent_outcome']

                    # Check if agent is done
                    if isinstance(outcome, AgentFinish):
                        result = outcome.return_values.get('output')
                        logger.info("Agent finished successfully")
                        break

            # Track tool executions for metadata
            if 'intermediate_steps' in state.get('agent', {}):
                intermediate_steps = state['agent']['intermediate_steps']

        # STEP 4: Validate result
        if result is None:
            logger.error("Agent completed but no result found")
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'error': 'Agent execution failed to produce a result',
                    'steps': step_count
                })
            }

        # STEP 5: Build response
        response_body = {
            'answer': result,
            'metadata': {
                'steps': step_count,
                'tools_used': [step[0].tool for step in intermediate_steps],
                'request_id': context.request_id
            }
        }

        logger.info(f"Returning answer (length: {len(result)} chars)")

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                # Enable CORS (if API Gateway doesn't handle it)
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
            },
            'body': json.dumps(response_body)
        }

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {e}")
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': f'Invalid JSON: {str(e)}'
            })
        }

    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(f"Unexpected error: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e),
                'type': type(e).__name__
            })
        }


# For local testing (run this file directly)
if __name__ == '__main__':
    # Mock event and context
    class MockContext:
        request_id = 'test-request-id'
        function_name = 'langgraph-agent'

        def get_remaining_time_in_millis(self):
            return 300000  # 5 minutes

    test_event = {
        'body': json.dumps({
            'question': 'What is the capital of France?',
            'chat_history': []
        })
    }

    response = handler(test_event, MockContext())
    print(json.dumps(response, indent=2))
