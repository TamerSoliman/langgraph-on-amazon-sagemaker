"""
LangGraph Agent Graph Definition

This module defines the LangGraph state machine for the agent.
It's extracted from langgraph_sagemaker.ipynb and adapted for production use.

For AI/ML Scientists:
- Graph = state machine with nodes (functions) and edges (transitions)
- State = dictionary passed between nodes (conversation history, tool results, etc.)
- Agent node = calls LLM to decide next action
- Tools node = executes the action (web search, database query, etc.)

Architecture:
  START → AGENT NODE ⟷ TOOLS NODE → END
           ↓ (Decision: call tool or finish?)
"""

import os
import logging
from typing import Dict, Any

from langchain import hub
from langchain.agents import create_xml_agent
from langgraph.prebuilt import create_agent_executor

from tools import create_tools
from sagemaker_llm import create_sagemaker_llm

logger = logging.getLogger(__name__)


def create_agent_graph():
    """
    Create the complete LangGraph agent executor

    This function assembles all components:
    1. Tools (Tavily search, etc.)
    2. LLM (SageMaker endpoint)
    3. Prompt (instructions for the LLM)
    4. Agent runnable (LLM + tools + prompt)
    5. Graph executor (state machine)

    Returns:
        CompiledStateGraph: The executable LangGraph agent

    For AI/ML Scientists:
        This is like model.compile() in Keras - it takes your defined
        architecture and creates an executable version

    Environment Variables Required:
        - SAGEMAKER_ENDPOINT_NAME: Name of the SageMaker endpoint
        - TAVILY_SECRET_ARN: ARN of Tavily API key in Secrets Manager
    """

    logger.info("Creating LangGraph agent")

    # STEP 1: Create Tools
    # Tools = external functions the agent can invoke
    # Example: Tavily web search, Wikipedia lookup, calculator, database query
    logger.info("Initializing tools")
    tools = create_tools()
    logger.info(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")

    # STEP 2: Create LLM Client
    # This connects to the SageMaker endpoint running Mistral 7B
    # The endpoint is separate infrastructure (GPU instance)
    # Agent just makes HTTP calls to it via boto3
    logger.info("Connecting to SageMaker LLM endpoint")
    llm = create_sagemaker_llm()
    logger.info(f"Connected to endpoint: {os.environ['SAGEMAKER_ENDPOINT_NAME']}")

    # STEP 3: Load Prompt Template
    # Prompts define the agent's behavior and capabilities
    # This specific prompt uses XML format for tool calling
    # (more reliable than JSON for smaller models like 7B)
    logger.info("Loading prompt template")
    prompt = hub.pull("hwchase17/xml-agent-convo")

    # The prompt contains placeholders:
    # - {tools}: List of available tools (auto-injected)
    # - {input}: User's question
    # - {agent_scratchpad}: History of tool calls and results
    # - {chat_history}: Previous conversation turns

    # STEP 4: Create Agent Runnable
    # This combines LLM + tools + prompt into a callable unit
    # Agent runnable:
    #   1. Formats prompt with current state
    #   2. Calls LLM (SageMaker endpoint)
    #   3. Parses LLM response (extracts tool calls or final answer)
    #   4. Returns AgentAction (call tool) or AgentFinish (done)
    logger.info("Creating XML agent runnable")
    agent_runnable = create_xml_agent(llm, tools, prompt)

    # STEP 5: Create Graph Executor
    # This wraps the agent runnable in a stateful graph
    # Graph manages:
    #   - State (input, chat_history, intermediate_steps, agent_outcome)
    #   - Nodes (agent node, tools node)
    #   - Edges (routing logic based on agent_outcome)
    #   - Execution loop (run until AgentFinish)
    logger.info("Creating agent executor (LangGraph)")
    app = create_agent_executor(agent_runnable, tools)

    # Behind the scenes, create_agent_executor builds this graph:
    #
    # ┌─────────┐
    # │  START  │
    # └────┬────┘
    #      │
    #      ▼
    # ┌──────────────┐
    # │ AGENT NODE   │ ← Calls agent_runnable.invoke(state)
    # │ (LLM)        │   Returns: AgentAction or AgentFinish
    # └─────┬────────┘
    #       │
    #       │ Conditional Edge (should_continue function)
    #       │
    #   ┌───┴───┐
    #   │       │
    #   ▼       ▼
    # AgentAction  AgentFinish
    # (tool call)     (done)
    #   │              │
    #   ▼              ▼
    # ┌──────────┐    END
    # │ TOOLS    │
    # │ NODE     │ ← Executes tool, adds result to state
    # └────┬─────┘
    #      │
    #      │ Loop back to AGENT NODE
    #      ▼
    # (Agent sees tool result in scratchpad, decides next action)

    # State schema:
    # {
    #   "input": "User's question",
    #   "chat_history": [],
    #   "agent_outcome": AgentAction(...) or AgentFinish(...),
    #   "intermediate_steps": [
    #     (AgentAction(...), "tool result"),
    #     ...
    #   ]
    # }

    logger.info("LangGraph agent created successfully")
    return app


# Example usage (for testing)
if __name__ == '__main__':
    # This won't work outside Lambda (no environment variables set)
    # But shows how to use the graph

    app = create_agent_graph()

    # Execute with a question
    inputs = {
        "input": "What is the capital of France?",
        "chat_history": []
    }

    print("Executing agent...")
    for state in app.stream(inputs):
        print(f"State update: {list(state.keys())}")

        if 'agent' in state:
            outcome = state['agent'].get('agent_outcome')
            if outcome:
                print(f"Agent outcome: {type(outcome).__name__}")

                from langchain_core.agents import AgentFinish
                if isinstance(outcome, AgentFinish):
                    print(f"Final answer: {outcome.return_values['output']}")
                    break
