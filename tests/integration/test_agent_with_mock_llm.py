"""
Integration Tests with Mock LLM

These tests execute the full agent graph but with mocked LLM responses.

For AI/ML Scientists:
- Integration tests = test multiple components working together
- LLM is mocked (no SageMaker calls), but graph execution is real
- Tests the ReAct loop: agent → tools → agent → finish
- Faster than E2E, but more realistic than unit tests
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, '/home/user/langgraph-on-amazon-sagemaker/agent')


class TestAgentWithMockLLM:
    """Test agent execution with predefined LLM responses"""

    @patch('tools.boto3.client')  # Mock Secrets Manager
    @patch('sagemaker_llm.SagemakerEndpoint')  # Mock LLM
    def test_single_tool_call_flow(self,
                                    mock_llm_class,
                                    mock_boto_client,
                                    mock_tavily_search_result):
        """
        Test complete flow: question → tool call → answer

        Flow:
          1. User asks question
          2. Agent calls tool (Tavily search)
          3. Tool returns result
          4. Agent synthesizes final answer
        """
        # Setup: Mock Secrets Manager
        mock_secrets = MagicMock()
        mock_secrets.get_secret_value.return_value = {
            'SecretString': '{"api_key":"test-key"}'
        }
        mock_boto_client.return_value = mock_secrets

        # Setup: Mock LLM to return tool call, then final answer
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            # First call: decide to use tool
            "<tool>tavily_search_results_json</tool><tool_input>latest UK storm</tool_input>",
            # Second call: final answer after seeing tool result
            "<final_answer>Storm Henk caused damage in south-west England.</final_answer>"
        ]
        mock_llm_class.return_value = mock_llm

        # Execute: Create and run agent
        from graph import create_agent_graph

        app = create_agent_graph()
        inputs = {"input": "What is the latest UK storm?", "chat_history": []}

        result = None
        for state in app.stream(inputs):
            if 'agent' in state:
                from langchain_core.agents import AgentFinish
                outcome = state['agent'].get('agent_outcome')
                if isinstance(outcome, AgentFinish):
                    result = outcome.return_values['output']
                    break

        # Verify
        assert result is not None
        assert "Storm Henk" in result or "south-west England" in result
        # LLM should have been called twice
        assert mock_llm.invoke.call_count == 2

    @patch('tools.boto3.client')
    @patch('sagemaker_llm.SagemakerEndpoint')
    def test_direct_answer_no_tools(self, mock_llm_class, mock_boto_client):
        """
        Test when agent answers directly without calling tools

        Flow:
          1. User asks simple question (2+2)
          2. Agent answers directly (no tool needed)
        """
        mock_secrets = MagicMock()
        mock_secrets.get_secret_value.return_value = {
            'SecretString': '{"api_key":"test-key"}'
        }
        mock_boto_client.return_value = mock_secrets

        # Mock LLM to return direct answer
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "<final_answer>4</final_answer>"
        mock_llm_class.return_value = mock_llm

        from graph import create_agent_graph

        app = create_agent_graph()
        inputs = {"input": "What is 2+2?", "chat_history": []}

        result = None
        for state in app.stream(inputs):
            if 'agent' in state:
                from langchain_core.agents import AgentFinish
                outcome = state['agent'].get('agent_outcome')
                if isinstance(outcome, AgentFinish):
                    result = outcome.return_values['output']
                    break

        assert result == "4"
        # Should only call LLM once (no tool calls)
        assert mock_llm.invoke.call_count == 1


class TestAgentStateManagement:
    """Test state persistence and updates through execution"""

    @patch('tools.boto3.client')
    @patch('sagemaker_llm.SagemakerEndpoint')
    def test_intermediate_steps_tracked(self, mock_llm_class, mock_boto_client):
        """Verify intermediate steps are recorded in state"""
        mock_secrets = MagicMock()
        mock_secrets.get_secret_value.return_value = {
            'SecretString': '{"api_key":"test-key"}'
        }
        mock_boto_client.return_value = mock_secrets

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "<tool>tavily_search_results_json</tool><tool_input>test query</tool_input>",
            "<final_answer>Final result</final_answer>"
        ]
        mock_llm_class.return_value = mock_llm

        from graph import create_agent_graph

        app = create_agent_graph()
        inputs = {"input": "Test question", "chat_history": []}

        final_state = None
        for state in app.stream(inputs):
            final_state = state

        # Final state should have intermediate_steps
        if 'agent' in final_state:
            assert 'intermediate_steps' in final_state['agent'] or \
                   final_state.get('intermediate_steps') is not None
