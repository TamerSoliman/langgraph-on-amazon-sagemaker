"""
================================================================================
HEAVILY ANNOTATED LANGGRAPH AGENT DEFINITION
================================================================================

This file contains the core LangGraph agent setup from langgraph_sagemaker.ipynb
with comprehensive annotations explaining the What, How, and Why of each component.

Original Location: langgraph_sagemaker.ipynb, Cell ID: 46d2fbde-72fe-498b-9ab9-474122a66104

Architecture Pattern: ReAct (Reasoning + Acting) Agent with XML-based Tool Calling
================================================================================
"""

# ============================================================================
# IMPORTS: External Dependencies
# ============================================================================

from langchain_community.tools.tavily_search import TavilySearchResults
# WHAT: Import the Tavily web search tool
# HOW: This is a pre-built LangChain community tool that wraps the Tavily API
# WHY: Provides the agent with real-time web search capability, enabling it
#      to answer questions requiring current information beyond the LLM's
#      training data cutoff

from langchain import hub
# WHAT: LangChain Hub client for fetching shared prompts
# HOW: Downloads prompt templates from LangChain's centralized repository
# WHY: Reuses battle-tested prompt engineering patterns instead of writing
#      custom prompts from scratch. The hub ensures version control and
#      community-validated prompt quality

from langchain.agents import AgentExecutor, create_xml_agent
# WHAT: Agent creation utilities
# HOW:
#   - create_xml_agent: Factory function that builds an agent using XML tags
#                       for structured output (tool calls, final answers)
#   - AgentExecutor: (Imported but not used - legacy LangChain class)
# WHY: XML format is more reliable than JSON for tool calling with smaller
#      LLMs (like Mistral 7B) that may struggle with strict JSON formatting

from langgraph.prebuilt import create_agent_executor
# WHAT: High-level LangGraph agent wrapper
# HOW: Takes an agent runnable + tools, returns a stateful graph executor
# WHY: Abstracts away the complexity of building the graph manually
#      (nodes, edges, state management). This is the "batteries-included"
#      approach for simple agent workflows


# ============================================================================
# TOOL CONFIGURATION: Defining Agent Capabilities
# ============================================================================

tools = [TavilySearchResults(max_results=1)]
# WHAT: List of tools the agent can invoke during execution
# HOW:
#   - Python list containing tool instances
#   - Each tool must implement the LangChain Tool interface:
#       * name: String identifier for the tool
#       * description: Tells the LLM when/how to use it
#       * _run() or _arun(): Execution logic
# WHY:
#   - List format allows easy extension (add more tools by appending)
#   - max_results=1 balances usefulness vs. context length
#   - More results = more tokens sent back to LLM = higher cost + latency
#
# EXTENSION EXAMPLE:
#   from langchain.tools import WikipediaQueryRun
#   tools = [
#       TavilySearchResults(max_results=1),
#       WikipediaQueryRun(),
#       YourCustomTool()
#   ]


# ============================================================================
# PROMPT ENGINEERING: Instructing the Agent
# ============================================================================

prompt = hub.pull("hwchase17/xml-agent-convo")
# WHAT: Fetch the XML agent prompt template from LangChain Hub
# HOW:
#   - Sends HTTP request to hub.langchain.com
#   - Downloads prompt with placeholders for:
#       * {tools}: Auto-injected tool descriptions
#       * {input}: User's question
#       * {agent_scratchpad}: History of tool calls and results
#       * {chat_history}: Previous conversation turns
# WHY: This specific prompt teaches the LLM to:
#   1. Analyze the question and decide if it needs tools
#   2. Format tool calls using XML: <tool>search</tool><tool_input>query</tool_input>
#   3. Wrap final answers in: <final_answer>response</final_answer>
#
# PROMPT STRUCTURE (from notebook output):
#   """
#   You are a helpful assistant. Help the user answer any questions.
#
#   You have access to the following tools:
#   {tools}
#
#   In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags.
#   You will then get back a response in the form <observation></observation>
#
#   For example, if you have a tool called 'search' that could run a google search,
#   in order to search for the weather in SF you would respond:
#   <tool>search</tool><tool_input>weather in SF</tool_input>
#   <observation>64 degrees</observation>
#
#   When you are done, respond with a final answer between <final_answer></final_answer>.
#
#   Begin!
#
#   Previous Conversation:
#   {chat_history}
#
#   Question: {input}
#   {agent_scratchpad}
#   """

print(prompt)
# WHY: Debugging step - lets you verify the prompt structure before execution


# ============================================================================
# AGENT CREATION: Building the Reasoning Engine
# ============================================================================

agent_runnable = create_xml_agent(llm, tools, prompt)
# WHAT: Construct the core agent logic as a LangChain Runnable
# HOW:
#   - Takes the LLM instance (SagemakerEndpoint configured earlier)
#   - Injects tool descriptions into the prompt's {tools} placeholder
#   - Returns a Runnable that:
#       1. Formats the prompt with current inputs
#       2. Calls llm.invoke(formatted_prompt)
#       3. Parses the LLM's XML response to extract:
#          - Tool calls (tool name + input)
#          - Final answers
# WHY:
#   - Runnable interface allows chaining, streaming, and async execution
#   - XML parsing handles malformed outputs gracefully (more forgiving than JSON)
#   - The agent doesn't execute tools itself - it just decides WHAT to call
#
# UNDER THE HOOD:
#   When llm.invoke() is called, the LLM receives something like:
#   "You have access to: tavily_search_results_json - A search engine..."
#   "Question: What is the latest storm in the UK?"
#
#   LLM responds with:
#   "<tool>tavily_search_results_json</tool><tool_input>latest UK storm</tool_input>"
#
#   create_xml_agent's parser extracts:
#   AgentAction(tool='tavily_search_results_json', tool_input='latest UK storm')


# ============================================================================
# GRAPH CONSTRUCTION: Stateful Execution Framework
# ============================================================================

app = create_agent_executor(agent_runnable, tools)
# WHAT: Build the complete LangGraph execution graph
# HOW: Under the hood, this creates a StateGraph with:
#
#   ┌─────────────────────────────────────────────────────────┐
#   │                    GRAPH STRUCTURE                       │
#   └─────────────────────────────────────────────────────────┘
#
#   START
#     │
#     ▼
#   ┌─────────────────┐
#   │  Agent Node     │ ← Calls agent_runnable.invoke(state)
#   │  (LLM Reasoning)│   Returns AgentAction or AgentFinish
#   └────────┬────────┘
#            │
#            │ Decision Point (Conditional Edge)
#            │
#      ┌─────┴─────┐
#      │           │
#      ▼           ▼
#   AgentAction   AgentFinish
#   (tool call)   (done)
#      │           │
#      ▼           ▼
#   ┌─────────┐  END
#   │ Tools   │
#   │ Node    │ ← Executes tool, stores result in state
#   └────┬────┘
#        │
#        │ Loop back
#        ▼
#   ┌─────────────────┐
#   │  Agent Node     │ ← Sees tool result in {agent_scratchpad}
#   │  (Reasoning)    │   Decides next action
#   └─────────────────┘
#
# WHY: This graph structure implements the ReAct loop:
#   1. REASON: Agent analyzes question + previous tool results
#   2. ACT: Agent calls a tool (or finishes if answer is ready)
#   3. OBSERVE: Tool result added to state
#   4. REPEAT: Go back to step 1 with updated context
#
# STATE SCHEMA:
#   The graph maintains a dictionary state with keys:
#   {
#       'input': str,              # Original user question
#       'chat_history': List,      # Previous conversation turns
#       'agent_outcome': Union[AgentAction, AgentFinish],
#       'intermediate_steps': List[Tuple[AgentAction, str]]
#   }
#
# STATE UPDATES:
#   - After Agent Node: state['agent_outcome'] = AgentAction or AgentFinish
#   - After Tools Node: state['intermediate_steps'].append((action, result))
#   - Agent sees intermediate_steps via {agent_scratchpad} in prompt
#
# CONDITIONAL ROUTING LOGIC:
#   def should_continue(state):
#       if isinstance(state['agent_outcome'], AgentFinish):
#           return "end"
#       else:
#           return "continue"  # Execute tools node
#
# PERSISTENCE:
#   - This example uses in-memory state (lost after execution)
#   - For production, you can pass a checkpointer:
#       from langgraph.checkpoint.memory import MemorySaver
#       app = create_agent_executor(agent_runnable, tools,
#                                    checkpointer=MemorySaver())
#   - Checkpointers enable:
#       * Pausing/resuming execution
#       * Human-in-the-loop approval
#       * Debugging by replaying state
#
# HUMAN-IN-THE-LOOP:
#   Not implemented in this example, but can be added:
#   1. Add interrupt_before=["tools"] to create_agent_executor
#   2. After agent decides to call a tool, execution pauses
#   3. User can approve/reject/modify the tool call
#   4. Resume with: app.stream(None, config)


# ============================================================================
# USAGE EXAMPLE (from the notebook)
# ============================================================================

"""
def ask_langgraph_question(question):
    # WHAT: Execute the graph with a user question
    inputs = {"input": question, "chat_history": []}
    # HOW: Pass initial state - empty chat history for single-turn interaction

    for s in app.stream(inputs):
        # WHAT: Stream yields state updates after each node execution
        # WHY: Allows real-time monitoring of agent's reasoning process

        agentValues = list(s.values())[0]
        # STRUCTURE: Each update is a dict with one key (node name)
        # Example: {'agent': {'agent_outcome': AgentAction(...)}}

        if 'agent_outcome' in agentValues and isinstance(agentValues['agent_outcome'], AgentFinish):
            agentFinish = agentValues['agent_outcome']
            # WHAT: Detect when agent has generated final answer
            # WHY: Signals the end of the ReAct loop

    return agentFinish.return_values["output"]
    # WHAT: Extract final answer string from AgentFinish object


# EXAMPLE EXECUTION TRACE:
# Input: "What is the latest storm to hit the UK?"
#
# Step 1: Agent Node
#   State Update: {'agent_outcome': AgentAction(
#       tool='tavily_search_results_json',
#       tool_input='latest storm in UK'
#   )}
#
# Step 2: Tools Node
#   Tool Execution: tavily_search_results_json.run('latest storm in UK')
#   Tool Result: '[{"url": "...", "content": "Storm Henk..."}]'
#   State Update: {'intermediate_steps': [(action, result)]}
#
# Step 3: Agent Node (with tool result in scratchpad)
#   LLM sees: "<observation>[Storm Henk data...]</observation>"
#   State Update: {'agent_outcome': AgentFinish(
#       return_values={'output': 'Storm Henk caused damage in south-west England'}
#   )}
#
# Step 4: END
#   Final state returned with complete execution history
"""


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

"""
1. MODULARITY:
   - Tools, prompt, and LLM are separate components
   - Easy to swap Mistral 7B for GPT-4, Claude, etc.
   - Easy to add/remove tools without changing agent logic

2. DECOUPLING:
   - Agent graph runs on cheap CPU compute
   - LLM endpoint (llm variable) points to expensive GPU instance
   - Graph only makes network calls to SageMaker when needed

3. OBSERVABILITY:
   - app.stream() provides step-by-step execution visibility
   - Can log each tool call for debugging/auditing
   - State contains full history (intermediate_steps)

4. EXTENSIBILITY:
   - Replace create_agent_executor with custom StateGraph for:
       * Multi-agent collaboration
       * Parallel tool execution
       * Custom routing logic
   - Add memory with checkpointers
   - Add human-in-the-loop with interrupt_before

5. COST OPTIMIZATION:
   - max_results=1 minimizes tokens
   - temperature=0.001 reduces retries from bad tool calls
   - In-memory state avoids database costs (trade-off: no persistence)
"""
