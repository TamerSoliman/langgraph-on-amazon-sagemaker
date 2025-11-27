"""
Unit Tests for Lambda Handler

Tests the Lambda function entry point without executing the full agent.

For AI/ML Scientists:
- These tests validate input parsing, output formatting, error handling
- Agent execution is mocked (no actual LLM calls)
- Focus: Lambda-specific logic, not agent logic
"""

import pytest
import json
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, '/home/user/langgraph-on-amazon-sagemaker/agent')
from lambda_handler import handler


class TestLambdaHandlerInputParsing:
    """Test how handler parses API Gateway events"""

    def test_valid_request(self, api_gateway_event, lambda_context):
        """Test with valid question in request body"""
        with patch('lambda_handler.create_agent_graph') as mock_graph:
            # Mock the agent to return immediately
            mock_app = MagicMock()
            mock_graph.return_value = mock_app

            # Mock agent stream to return a final answer
            from langchain_core.agents import AgentFinish
            mock_finish = AgentFinish(
                return_values={'output': 'Paris'},
                log=''
            )
            mock_app.stream.return_value = [
                {'agent': {'agent_outcome': mock_finish}}
            ]

            response = handler(api_gateway_event, lambda_context)

            # Verify response structure
            assert response['statusCode'] == 200
            assert 'body' in response

            body = json.loads(response['body'])
            assert 'answer' in body
            assert body['answer'] == 'Paris'

    def test_missing_question_field(self, lambda_context):
        """Test error handling when question is missing"""
        event = {
            'body': json.dumps({'chat_history': []})  # Missing 'question'
        }

        response = handler(event, lambda_context)

        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert 'error' in body
        assert 'question' in body['error'].lower()

    def test_invalid_json_body(self, lambda_context):
        """Test error handling for malformed JSON"""
        event = {
            'body': 'not valid json'
        }

        response = handler(event, lambda_context)

        assert response['statusCode'] == 400
        body = json.loads(response['body'])
        assert 'error' in body

    def test_direct_invocation(self, lambda_context):
        """Test Lambda invoked directly (not through API Gateway)"""
        # Direct invocation passes body directly, not as JSON string
        event = {
            'question': 'What is 2+2?',
            'chat_history': []
        }

        with patch('lambda_handler.create_agent_graph') as mock_graph:
            mock_app = MagicMock()
            mock_graph.return_value = mock_app

            from langchain_core.agents import AgentFinish
            mock_finish = AgentFinish(return_values={'output': '4'}, log='')
            mock_app.stream.return_value = [
                {'agent': {'agent_outcome': mock_finish}}
            ]

            response = handler(event, lambda_context)

            assert response['statusCode'] == 200


class TestLambdaHandlerOutputFormatting:
    """Test response formatting"""

    def test_response_includes_metadata(self, api_gateway_event, lambda_context):
        """Verify metadata is included in response"""
        with patch('lambda_handler.create_agent_graph') as mock_graph:
            mock_app = MagicMock()
            mock_graph.return_value = mock_app

            from langchain_core.agents import AgentFinish, AgentAction
            # Simulate multi-step execution
            mock_action = AgentAction(tool='search', tool_input='query', log='')
            mock_finish = AgentFinish(return_values={'output': 'Answer'}, log='')

            mock_app.stream.return_value = [
                {'agent': {'agent_outcome': mock_action}},
                {'agent': {'intermediate_steps': [(mock_action, 'result')]}},
                {'agent': {'agent_outcome': mock_finish}}
            ]

            response = handler(api_gateway_event, lambda_context)

            body = json.loads(response['body'])
            assert 'metadata' in body
            assert 'steps' in body['metadata']
            assert 'tools_used' in body['metadata']
            assert 'request_id' in body['metadata']
            assert body['metadata']['request_id'] == lambda_context.request_id

    def test_cors_headers_present(self, api_gateway_event, lambda_context):
        """Verify CORS headers are set"""
        with patch('lambda_handler.create_agent_graph') as mock_graph:
            mock_app = MagicMock()
            mock_graph.return_value = mock_app

            from langchain_core.agents import AgentFinish
            mock_finish = AgentFinish(return_values={'output': 'Test'}, log='')
            mock_app.stream.return_value = [
                {'agent': {'agent_outcome': mock_finish}}
            ]

            response = handler(api_gateway_event, lambda_context)

            assert 'headers' in response
            assert 'Access-Control-Allow-Origin' in response['headers']
            assert response['headers']['Content-Type'] == 'application/json'


class TestLambdaHandlerErrorHandling:
    """Test error handling and edge cases"""

    def test_agent_execution_exception(self, api_gateway_event, lambda_context):
        """Test handling of unexpected errors during agent execution"""
        with patch('lambda_handler.create_agent_graph') as mock_graph:
            mock_graph.side_effect = Exception("Simulated failure")

            response = handler(api_gateway_event, lambda_context)

            assert response['statusCode'] == 500
            body = json.loads(response['body'])
            assert 'error' in body

    def test_agent_no_result(self, api_gateway_event, lambda_context):
        """Test when agent completes but produces no result"""
        with patch('lambda_handler.create_agent_graph') as mock_graph:
            mock_app = MagicMock()
            mock_graph.return_value = mock_app

            # Agent completes but no AgentFinish (shouldn't happen, but test it)
            mock_app.stream.return_value = [
                {'some_state': {}}
            ]

            response = handler(api_gateway_event, lambda_context)

            assert response['statusCode'] == 500

    def test_timeout_warning(self, api_gateway_event, lambda_context):
        """Test timeout warning logic"""
        # Set remaining time to < 10 seconds
        lambda_context.get_remaining_time_in_millis = Mock(return_value=5000)

        with patch('lambda_handler.create_agent_graph') as mock_graph:
            with patch('lambda_handler.logger.warning') as mock_warning:
                mock_app = MagicMock()
                mock_graph.return_value = mock_app

                from langchain_core.agents import AgentFinish
                mock_finish = AgentFinish(return_values={'output': 'Fast answer'}, log='')
                mock_app.stream.return_value = [
                    {'agent': {'agent_outcome': mock_finish}}
                ]

                response = handler(api_gateway_event, lambda_context)

                # Should have logged timeout warning
                mock_warning.assert_called()
