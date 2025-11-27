"""
================================================================================
HEAVILY ANNOTATED AGENT-LLM COMMUNICATION INTERFACE
================================================================================

This file contains the SageMaker endpoint integration code from
langgraph_sagemaker.ipynb with comprehensive annotations explaining how
the LangGraph agent communicates with the remote LLM endpoint.

Original Location: langgraph_sagemaker.ipynb, Cell ID: 0625e0a9-cefc-4ec1-921c-5f96cf004ed0

Key Concepts:
- Protocol Translation: Converting between LangChain and SageMaker formats
- Payload Engineering: Crafting requests for optimal LLM performance
- Response Parsing: Extracting structured outputs from raw LLM responses
================================================================================
"""

# ============================================================================
# IMPORTS: LangChain SageMaker Integration
# ============================================================================

from langchain.prompts import PromptTemplate
# WHAT: Template class for dynamic prompt generation
# HOW: Not directly used in this cell, but available for custom prompts
# WHY: Imported for potential prompt customization in extended implementations

from langchain_community.llms import SagemakerEndpoint
# WHAT: LangChain wrapper for AWS SageMaker real-time inference endpoints
# HOW: Inherits from BaseLLM, providing:
#   - invoke(): Send a single prompt, get response
#   - stream(): Stream tokens as they're generated (if supported)
#   - batch(): Send multiple prompts in parallel
# WHY: Provides a unified interface - same code works with OpenAI, Anthropic, etc.
#      Just change the LLM class, keep agent logic unchanged

from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
# WHAT: Abstract base class for request/response transformation
# HOW: Defines two abstract methods that must be implemented:
#   - transform_input(): Prepare data for SageMaker endpoint
#   - transform_output(): Parse SageMaker response
# WHY: SageMaker endpoints accept arbitrary input/output formats
#      Different models (HuggingFace, Jumpstart, Custom) need different formats
#      This abstraction keeps the transformation logic separate and swappable

import json
# WHAT: Standard library for JSON encoding/decoding
# WHY: SageMaker expects JSON payloads, returns JSON responses


from typing import Dict
# WHAT: Type hints for better code documentation and IDE support
# WHY: Makes the transform methods' signatures clear


# ============================================================================
# CONTENT HANDLER: Protocol Translation Layer
# ============================================================================

class ContentHandler(LLMContentHandler):
    """
    WHAT: Custom transformer for Mistral 7B Instruct deployed via SageMaker Jumpstart

    HOW: Implements the required transform_input and transform_output methods
         to convert between LangChain strings and SageMaker's expected format

    WHY: Each SageMaker model has its own input/output format:
         - HuggingFace TGI: {"inputs": str, "parameters": dict}
         - AWS DeepSpeed: {"text_inputs": str, "max_length": int}
         - Custom containers: Your own schema

         This class encapsulates Mistral 7B Jumpstart's specific format,
         keeping it separate from the agent logic.
    """

    # ========================================================================
    # CLASS ATTRIBUTES: HTTP Headers
    # ========================================================================

    content_type = "application/json"
    # WHAT: HTTP Content-Type header sent to SageMaker endpoint
    # HOW: LangChain sets this in the POST request headers
    # WHY: Tells SageMaker how to parse the request body
    #      If you sent XML or plain text, you'd change this to "application/xml" or "text/plain"

    accepts = "application/json"
    # WHAT: HTTP Accept header sent to SageMaker endpoint
    # HOW: LangChain includes this to specify desired response format
    # WHY: Some endpoints support multiple output formats (JSON, XML, msgpack)
    #      This tells SageMaker we want JSON back


    # ========================================================================
    # INPUT TRANSFORMATION: LangChain → SageMaker
    # ========================================================================

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        """
        WHAT: Convert LangChain prompt string into SageMaker-compatible payload

        HOW: Called automatically by SagemakerEndpoint.invoke() before sending request

        WHY: SageMaker Jumpstart models expect a specific JSON structure.
             This method provides the bridge between LangChain's simple string
             and SageMaker's structured format.

        PARAMETERS:
            prompt (str): The complete prompt string, already formatted by the agent.
                         For Mistral, this includes the <s>[INST] ... [/INST] tags
                         Examples:
                           - "<s>[INST] What is 2+2? [/INST]"
                           - "<s>[INST] Search for latest news [/INST]"

            model_kwargs (Dict): Inference parameters passed from SagemakerEndpoint
                                Examples: {"max_new_tokens": 500, "temperature": 0.001}

        RETURNS:
            bytes: UTF-8 encoded JSON payload ready for HTTP POST
        """

        # --------------------------------------------------------------------
        # DEBUGGING HOOKS (commented out)
        # --------------------------------------------------------------------
        # print("transforming_input")  # Trace execution flow
        # print(prompt)                # Inspect actual prompt sent to model
        # WHY: Uncomment these during development to debug:
        #      - Prompt formatting issues (missing tags, wrong format)
        #      - Agent scratchpad content (verify tool results are included)
        #      - Token count estimation (long prompts → high latency/cost)


        # --------------------------------------------------------------------
        # PAYLOAD CONSTRUCTION
        # --------------------------------------------------------------------
        input_str = json.dumps({
            "inputs": prompt,
            "parameters": model_kwargs
        })
        # WHAT: Create the JSON payload required by Mistral 7B Jumpstart
        #
        # PAYLOAD STRUCTURE:
        # {
        #     "inputs": "<s>[INST] Your question [/INST]",
        #     "parameters": {
        #         "max_new_tokens": 500,
        #         "do_sample": True,
        #         "temperature": 0.001,
        #         "top_p": 0.9,          # (if specified in model_kwargs)
        #         "top_k": 50,           # (if specified)
        #         "repetition_penalty": 1.0  # (if specified)
        #     }
        # }
        #
        # WHY THIS FORMAT:
        # - "inputs": Jumpstart wraps HuggingFace's text-generation-inference (TGI)
        #             TGI expects "inputs" key for the text prompt
        # - "parameters": TGI's generation config - controls randomness, length, etc.
        #
        # MODEL_KWARGS DETAILS:
        # - max_new_tokens: Maximum tokens to generate (NOT total tokens)
        #                   Example: prompt=100 tokens, max_new_tokens=500 → max total=600
        #                   WHY 500: Agent needs room for:
        #                       * Reasoning (50-100 tokens)
        #                       * XML tags (20-30 tokens)
        #                       * Tool calls/answers (300-400 tokens)
        #                   Too low (e.g., 128) → responses cut mid-XML → parsing fails
        #
        # - temperature: Randomness (0.0=deterministic, 1.0=creative)
        #                WHY 0.001 (near-zero): Tool calling requires precise format
        #                Higher temp → more format violations → more retries
        #
        # - do_sample: Must be True for temperature to take effect
        #              False → greedy decoding (always pick highest probability token)
        #
        # ALTERNATIVE FORMATS (other models):
        # - AWS DeepSpeed container:
        #     {"text_inputs": prompt, "max_length": 500, "num_return_sequences": 1}
        # - SageMaker LMI container:
        #     {"inputs": [prompt], "parameters": {...}}  # Note: inputs is array
        # - Custom container: Whatever you defined in inference.py

        return input_str.encode("utf-8")
        # WHAT: Convert JSON string to bytes
        # WHY: HTTP POST body must be bytes, not string
        #      SageMaker SDK's invoke_endpoint() expects bytes input


    # ========================================================================
    # OUTPUT TRANSFORMATION: SageMaker → LangChain
    # ========================================================================

    def transform_output(self, output: bytes) -> str:
        """
        WHAT: Extract the generated text from SageMaker's JSON response

        HOW: Called automatically by SagemakerEndpoint.invoke() after receiving response

        WHY: SageMaker returns a structured response with metadata.
             LangChain just needs the raw text for the agent to parse.

        PARAMETERS:
            output (bytes): Raw HTTP response body from SageMaker endpoint
                           Example bytes: b'[{"generated_text": "The answer is..."}]'

        RETURNS:
            str: Extracted generated text (agent parses this for XML tags)
        """

        # --------------------------------------------------------------------
        # DEBUGGING HOOKS (commented out)
        # --------------------------------------------------------------------
        # print("transforming_output")
        # WHY: Uncomment to debug:
        #      - Empty responses (model timed out or failed)
        #      - Malformed JSON (container misconfiguration)
        #      - Unexpected response structure (wrong model version)


        # --------------------------------------------------------------------
        # STEP 1: DECODE BYTES TO STRING
        # --------------------------------------------------------------------
        decoded_output = output.read().decode("utf-8")
        # WHAT: Convert HTTP response bytes → UTF-8 string
        # HOW:
        #   - output is a StreamingBody object (from boto3)
        #   - .read() drains the stream into bytes
        #   - .decode("utf-8") converts bytes to string
        # WHY: JSON parsing requires string input, not bytes
        #
        # RESULT EXAMPLE:
        # '[{"generated_text": "<tool>tavily_search_results_json</tool>..."}]'

        # print("decoded: " + decoded_output)  # Uncomment to see raw JSON


        # --------------------------------------------------------------------
        # STEP 2: PARSE JSON
        # --------------------------------------------------------------------
        response_json = json.loads(decoded_output)
        # WHAT: Parse JSON string into Python data structure
        #
        # RESPONSE STRUCTURE (Mistral 7B Jumpstart):
        # [
        #     {
        #         "generated_text": "The actual model output text here",
        #         "details": {                    # (if return_full_text=False)
        #             "finish_reason": "length",  # or "eos_token" or "stop_sequence"
        #             "generated_tokens": 487,
        #             "seed": null,
        #             "prefill": [],
        #             "tokens": [...]             # Token IDs + probabilities
        #         }
        #     }
        # ]
        #
        # WHY ARRAY: HuggingFace TGI supports batch inference
        #            Even for 1 prompt, response is array with 1 element
        #
        # ALTERNATIVE FORMATS (other models):
        # - SageMaker LLaMA:
        #     {"generated_text": "...", "input_length": 42}
        # - Custom container:
        #     {"predictions": ["text1", "text2"]}


        # --------------------------------------------------------------------
        # STEP 3: EXTRACT GENERATED TEXT
        # --------------------------------------------------------------------
        response = response_json[0]["generated_text"]
        # WHAT: Extract the actual text from the first array element
        # HOW: Access array index 0, then "generated_text" key
        # WHY:
        #   - We only send 1 prompt, so only 1 response (index 0)
        #   - "generated_text" is the standard TGI key for output
        #   - Ignores metadata like finish_reason, tokens, etc.
        #
        # RESULT EXAMPLES:
        # 1. Tool call:
        #    "<tool>tavily_search_results_json</tool><tool_input>latest UK storm</tool_input>"
        #
        # 2. Final answer:
        #    "<final_answer>The recipe for mayonnaise is: ...</final_answer>"
        #
        # 3. Direct answer (no tools needed):
        #    "Mayonnaise is made by emulsifying oil and egg yolks..."

        # print("response: " + response)  # Uncomment to see extracted text


        # --------------------------------------------------------------------
        # RETURN VALUE
        # --------------------------------------------------------------------
        return response
        # WHAT: Return the raw generated text to LangChain
        # HOW: LangChain passes this to the XML agent parser
        # WHY: The parser extracts:
        #      - <tool> tags → AgentAction objects
        #      - <final_answer> tags → AgentFinish objects
        #      - Malformed responses → Parsing errors (agent retries or fails)


# ============================================================================
# CONTENT HANDLER INSTANTIATION
# ============================================================================

content_handler = ContentHandler()
# WHAT: Create an instance of our custom handler
# WHY: We'll pass this to SagemakerEndpoint to enable the transformations
#      Can reuse the same handler for multiple endpoint instances


# ============================================================================
# SAGEMAKER ENDPOINT CONFIGURATION
# ============================================================================

llm = SagemakerEndpoint(
    # ========================================================================
    # ENDPOINT IDENTIFICATION
    # ========================================================================
    endpoint_name=endpoint_name,
    # WHAT: Name of the SageMaker real-time inference endpoint
    # HOW: Obtained earlier via input("SageMaker Endpoint Name:")
    #      Example: "jumpstart-dft-hf-llm-mistral-7b-instruct"
    # WHY: This is the target for all LLM API calls
    #      The endpoint must:
    #        - Be in "InService" status
    #        - Be accessible from the execution role running this code
    #        - Use the Mistral 7B Jumpstart container (or compatible format)
    #
    # COST CONSIDERATION:
    # - Endpoint runs 24/7 once deployed (until explicitly deleted)
    # - Billing: $X/hour based on instance type (e.g., ml.g5.xlarge)
    # - This agent code makes on-demand calls → you pay for idle time too
    # - Serverless endpoints (preview) charge per inference instead


    # ========================================================================
    # AWS REGION
    # ========================================================================
    region_name="us-east-1",
    # WHAT: AWS region where the endpoint is deployed
    # HOW: Hardcoded here, could be dynamic:
    #      import boto3
    #      region_name = boto3.Session().region_name
    # WHY: SageMaker endpoints are regional resources
    #      Cross-region calls fail with "Endpoint not found"
    #
    # LATENCY CONSIDERATION:
    # - If agent runs in us-west-2, endpoint in us-east-1 → +50-100ms latency
    # - For production, deploy agent and endpoint in same region


    # ========================================================================
    # MODEL INFERENCE PARAMETERS
    # ========================================================================
    model_kwargs={
        "max_new_tokens": 500,
        # WHAT: Maximum number of tokens the model can generate
        # HOW: Passed to the model via ContentHandler.transform_input()
        # WHY: Critical for agent functionality:
        #      - Too low (128) → XML responses truncated → parsing fails
        #      - Too high (2048) → increased cost + latency
        #      - 500 balances completeness with efficiency
        #
        # CALCULATION:
        # Agent needs space for:
        #   - Reasoning: ~50 tokens ("Let me search for...")
        #   - XML tags: ~30 tokens ("<tool>...</tool><tool_input>...</tool_input>")
        #   - Tool input: ~50 tokens (search query)
        #   - OR final answer: ~300 tokens (comprehensive response)
        #   - Buffer: ~70 tokens (safety margin)
        # Total: ~500 tokens
        #
        # COST IMPACT (example pricing):
        # - Model: Mistral 7B on ml.g5.xlarge
        # - Cost: $1.006/hour endpoint + $0/token (cost in instance time)
        # - Inference time: ~200ms/token
        # - 500 tokens → ~100 seconds → ~$0.028/call
        #
        # NOTEBOOK COMMENT (from original):
        # "extending the max_tokens is VITAL, as the response will otherwise be cut,
        #  breaking the agent functionality by not giving it access to the LLM's full answer.
        #  The value has been picked empirically"

        "do_sample": True,
        # WHAT: Enable probabilistic sampling (vs. greedy decoding)
        # HOW: When True, model samples from top-k/top-p distribution
        #      When False, always picks highest probability token
        # WHY: Required for temperature to work
        #      Even with very low temp (0.001), need do_sample=True

        "temperature": 0.001
        # WHAT: Sampling randomness (0.0=deterministic, 2.0=very random)
        # HOW: Scales the logits before softmax:
        #      logits / temperature → softmax → sample
        # WHY: Agent needs consistent, predictable tool calling
        #      High temp → creative but unreliable XML formatting
        #      Low temp (0.001) → near-deterministic, reliable parsing
        #
        # EXAMPLE IMPACT:
        # Question: "Search for Python tutorials"
        #
        # Temperature 0.001 (reliable):
        # "<tool>tavily_search_results_json</tool><tool_input>Python tutorials</tool_input>"
        #
        # Temperature 0.9 (creative but risky):
        # "<tool>tavily_search_results_json<tool_input>awesome Python coding guides!!</tool_input></tool>"
        # ↑ Missing closing tag, extra words → parser fails
        #
        # EXTENSION:
        # For multi-step reasoning, might want:
        # - Step 1 (planning): temperature=0.7 (creative problem decomposition)
        # - Step 2 (tool call): temperature=0.001 (precise formatting)
        # - Step 3 (synthesis): temperature=0.5 (natural language)
        # This requires custom graph with different LLM configs per node
    },


    # ========================================================================
    # PROTOCOL HANDLER
    # ========================================================================
    content_handler=content_handler
    # WHAT: Instance of our ContentHandler class defined above
    # HOW: SagemakerEndpoint calls:
    #      - content_handler.transform_input() before invoke_endpoint()
    #      - content_handler.transform_output() after receiving response
    # WHY: Decouples model-specific format from agent logic
    #      To swap to a different model/container:
    #      1. Write a new ContentHandler subclass
    #      2. Pass it here
    #      3. Agent code remains unchanged
    #
    # EXAMPLE: Switching to Claude on SageMaker Bedrock
    # class ClaudeContentHandler(LLMContentHandler):
    #     def transform_input(self, prompt, model_kwargs):
    #         return json.dumps({
    #             "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
    #             "max_tokens_to_sample": model_kwargs["max_new_tokens"]
    #         }).encode("utf-8")
    #
    #     def transform_output(self, output):
    #         return json.loads(output.read())["completion"]
    #
    # llm = SagemakerEndpoint(endpoint_name="claude-v2",
    #                         content_handler=ClaudeContentHandler())
    # # Agent code (create_xml_agent, create_agent_executor) unchanged!
)

# WHAT: llm is now a configured LangChain LLM instance
# HOW: Implements the BaseLLM interface - works with all LangChain components
# WHY: Can pass this to:
#      - create_xml_agent() (as we do)
#      - LLMChain
#      - ConversationChain
#      - Any LangChain runnable expecting an LLM


# ============================================================================
# NETWORK FLOW: Complete Request/Response Cycle
# ============================================================================

"""
WHAT HAPPENS WHEN AGENT CALLS llm.invoke(prompt):

1. LangChain Layer (SagemakerEndpoint.invoke())
   ↓ Calls content_handler.transform_input()

2. Transform Input
   Input:  prompt = "<s>[INST] Search for UK storms [/INST]"
           model_kwargs = {"max_new_tokens": 500, ...}
   Output: b'{"inputs": "<s>[INST] Search for UK storms [/INST]",
              "parameters": {"max_new_tokens": 500, ...}}'

3. AWS SDK Layer (boto3 SageMaker Runtime)
   ↓ HTTP POST to https://runtime.sagemaker.us-east-1.amazonaws.com/
   Headers:
     Content-Type: application/json
     Accept: application/json
     X-Amzn-SageMaker-Target-Model: (endpoint name)
     Authorization: (AWS SigV4 signature)
   Body: (bytes from step 2)

4. SageMaker Service
   ↓ Routes request to endpoint's EC2/ECS instance
   ↓ Instance runs HuggingFace TGI container

5. Model Inference (Mistral 7B on GPU)
   ↓ Tokenize input: [1, 2574, 354, ...] (token IDs)
   ↓ GPU forward pass through 32 transformer layers
   ↓ Generate tokens iteratively (autoregressive):
       <tool>tav... → tavily_s... → tavily_search... → </tool>...
   ↓ Stop when:
       - max_new_tokens reached (500), OR
       - EOS token generated, OR
       - Custom stop sequence hit

6. TGI Container
   ↓ Detokenize output: [8432, 1039, ...] → "<tool>tavily_search..."
   ↓ Build JSON response: [{"generated_text": "...", "details": {...}}]

7. SageMaker Service
   ↓ HTTP 200 response
   Body: b'[{"generated_text": "<tool>tavily_search_results_json</tool>..."}]'

8. AWS SDK Layer (boto3)
   ↓ Returns StreamingBody to LangChain

9. Transform Output
   Input:  bytes from step 7
   Output: "<tool>tavily_search_results_json</tool><tool_input>UK storms</tool_input>"

10. LangChain Layer
    ↓ Returns string to agent

11. XML Agent Parser
    ↓ Regex: <tool>(.*?)</tool> → "tavily_search_results_json"
    ↓ Regex: <tool_input>(.*?)</tool_input> → "UK storms"
    ↓ Creates: AgentAction(tool="tavily_search_results_json",
                           tool_input="UK storms")

12. LangGraph Executor
    ↓ Routes to Tools Node
    ↓ Executes tool
    ↓ Adds result to state["intermediate_steps"]
    ↓ Routes back to Agent Node for next iteration

LATENCY BREAKDOWN (typical):
- Network (agent → SageMaker): 10-50ms
- Model inference (500 tokens): 2-5 seconds (depends on instance type)
- Network (SageMaker → agent): 10-50ms
- Total: ~2-5 seconds per LLM call

FAILURE MODES:
1. Endpoint not found → boto3.client exception → agent fails
2. Endpoint throttled → HTTP 429 → retry with exponential backoff
3. Model timeout (>60s) → SageMaker kills request → empty response
4. Malformed JSON → json.loads() exception → agent fails
5. Truncated XML (max_tokens too low) → Parser fails → agent retries or errors
"""


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

"""
1. SEPARATION OF CONCERNS:
   - ContentHandler: Format translation (model-specific)
   - SagemakerEndpoint: Network communication (AWS-specific)
   - Agent: Business logic (framework-agnostic)
   → Easy to swap any component independently

2. CRITICAL PARAMETERS:
   - max_new_tokens: Must be high enough for complete XML responses
   - temperature: Must be low for reliable tool calling
   - These are NOT just "performance tuning" - they're required for correctness

3. COST AWARENESS:
   - Every llm.invoke() call → SageMaker API call → GPU time
   - Agent may call LLM 3-5 times per user question (initial + per tool use)
   - Endpoint runs 24/7 → pay for idle time
   - Consider: Serverless endpoints, auto-scaling, or on-demand deployment

4. DEBUGGING STRATEGY:
   - Uncomment print() statements in ContentHandler
   - Log: prompt → input payload → output payload → parsed response
   - Check: Is XML well-formed? Are tool names correct? Is finish_reason="length"?

5. EXTENSIBILITY:
   - Add caching: Check if prompt seen before, return cached response
   - Add retry logic: If parsing fails, retry with higher max_tokens
   - Add fallback: If SageMaker fails, fall back to Bedrock/OpenAI
   - Add monitoring: Log latency, token counts, error rates to CloudWatch
"""
