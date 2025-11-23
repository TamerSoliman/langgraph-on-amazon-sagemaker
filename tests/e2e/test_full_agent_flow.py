"""
End-to-End Tests

These tests call the REAL SageMaker endpoint.
Only run these when endpoint is deployed and you're willing to pay for inference.

For AI/ML Scientists:
- E2E tests = test the entire system (no mocks)
- Requires: deployed SageMaker endpoint, valid Tavily API key
- Slow (2-10 seconds per test) and costs money
- Run sparingly (e.g., before releases, not on every commit)

Usage:
  pytest tests/e2e/test_full_agent_flow.py --e2e
  (requires --e2e flag to actually run)
"""

import pytest
import sys
import os

sys.path.insert(0, '/home/user/langgraph-on-amazon-sagemaker/agent')


@pytest.mark.skipif(
    not os.environ.get('RUN_E2E_TESTS'),
    reason="E2E tests skipped (set RUN_E2E_TESTS=1 to run)"
)
class TestE2EAgent:
    """
    End-to-end tests with real SageMaker endpoint

    IMPORTANT: These tests will:
    - Call SageMaker endpoint (costs ~$0.001 per test)
    - Call Tavily API (costs ~$0.001 per search)
    - Take 5-15 seconds each

    Before running:
    1. Deploy SageMaker endpoint
    2. Set SAGEMAKER_ENDPOINT_NAME environment variable
    3. Set TAVILY_SECRET_ARN environment variable
    4. Set RUN_E2E_TESTS=1
    """

    def test_real_question_with_search(self):
        """Test a real question that requires web search"""
        from graph import create_agent_graph

        app = create_agent_graph()

        # Question that definitely requires search (time-sensitive)
        inputs = {
            "input": "What is the latest storm to hit the UK?",
            "chat_history": []
        }

        result = None
        steps = 0
        tools_used = []

        for state in app.stream(inputs):
            steps += 1

            if 'agent' in state:
                from langchain_core.agents import AgentFinish, AgentAction
                outcome = state['agent'].get('agent_outcome')

                if isinstance(outcome, AgentAction):
                    tools_used.append(outcome.tool)

                if isinstance(outcome, AgentFinish):
                    result = outcome.return_values['output']
                    break

        # Assertions
        assert result is not None, "Agent should return an answer"
        assert len(result) > 0, "Answer should not be empty"
        assert steps > 1, "Should take multiple steps (agent + tools + agent)"
        assert 'tavily_search_results_json' in tools_used, "Should use search tool"

    def test_real_direct_answer(self):
        """Test a question that doesn't require tools"""
        from graph import create_agent_graph

        app = create_agent_graph()

        inputs = {
            "input": "What is 2 + 2?",
            "chat_history": []
        }

        result = None
        for state in app.stream(inputs):
            if 'agent' in state:
                from langchain_core.agents import AgentFinish
                outcome = state['agent'].get('agent_outcome')
                if isinstance(outcome, AgentFinish):
                    result = outcome.return_values['output']
                    break

        assert result is not None
        assert '4' in result or 'four' in result.lower()

    @pytest.mark.parametrize("question,expected_keyword", [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What is the largest planet?", "Jupiter"),
    ])
    def test_multiple_factual_questions(self, question, expected_keyword):
        """Test multiple factual questions"""
        from graph import create_agent_graph

        app = create_agent_graph()
        inputs = {"input": question, "chat_history": []}

        result = None
        for state in app.stream(inputs):
            if 'agent' in state:
                from langchain_core.agents import AgentFinish
                outcome = state['agent'].get('agent_outcome')
                if isinstance(outcome, AgentFinish):
                    result = outcome.return_values['output']
                    break

        assert result is not None
        assert expected_keyword.lower() in result.lower(), \
            f"Expected '{expected_keyword}' in answer: {result}"
