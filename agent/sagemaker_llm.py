"""
SageMaker LLM Client

This module creates and configures the LangChain LLM client that connects
to the SageMaker endpoint running Mistral 7B.

For AI/ML Scientists:
- This is the "inference API" for your model
- SageMaker endpoint = GPU instance running the model
- This client = HTTP wrapper that sends requests to that endpoint
- Similar to: requests.post("http://gpu-server:8000/predict", json=data)
  but with AWS authentication, retry logic, and LangChain integration

Communication flow:
  Agent → SagemakerEndpoint → boto3 → HTTPS → SageMaker → GPU → Model
"""

import os
import json
import logging
from typing import Dict

from langchain_community.llms import SagemakerEndpoint
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler

logger = logging.getLogger(__name__)


class MistralContentHandler(LLMContentHandler):
    """
    Content handler for Mistral 7B Instruct model on SageMaker

    This class translates between LangChain's format and SageMaker's format.

    For AI/ML Scientists:
        Think of this as a data preprocessor and postprocessor:
        - transform_input: Converts LangChain prompt → SageMaker JSON payload
        - transform_output: Converts SageMaker JSON response → LangChain string

        Why needed: Different model containers have different expected formats.
        Mistral 7B uses HuggingFace TGI (Text Generation Inference) container,
        which expects specific JSON structure.

    Protocol:
        LangChain        ContentHandler      SageMaker Endpoint
        ─────────        ──────────────      ─────────────────
        str              transform_input()   {"inputs": str, "parameters": {...}}
        (prompt)         →                   (JSON payload)

        str              transform_output()  [{"generated_text": str}]
        (response)       ←                   (JSON response)
    """

    content_type = "application/json"
    # HTTP Content-Type header sent to SageMaker
    # Tells endpoint to expect JSON in request body

    accepts = "application/json"
    # HTTP Accept header sent to SageMaker
    # Tells endpoint we want JSON in response

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        """
        Transform LangChain prompt into SageMaker payload

        Args:
            prompt: The formatted prompt string (already includes chat template)
            model_kwargs: Generation parameters (max_tokens, temperature, etc.)

        Returns:
            bytes: JSON payload encoded as UTF-8 bytes

        For AI/ML Scientists:
            This is like creating a batch for inference:
            inputs = tokenizer(prompt)
            outputs = model.generate(**inputs, max_new_tokens=500)

            But here, we're sending it to a remote API instead of calling locally.

        Payload Structure (HuggingFace TGI):
            {
                "inputs": "<s>[INST] Your prompt here [/INST]",
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.001,
                    "do_sample": true,
                    "top_p": 0.9,
                    "top_k": 50
                }
            }
        """

        # Build payload dictionary
        payload = {
            "inputs": prompt,
            "parameters": model_kwargs
        }

        # Convert to JSON string, then to bytes
        # (HTTP POST body must be bytes)
        payload_bytes = json.dumps(payload).encode('utf-8')

        # Log for debugging (truncate long prompts)
        logger.debug(f"SageMaker request payload: {json.dumps(payload)[:200]}...")

        return payload_bytes

    def transform_output(self, output: bytes) -> str:
        """
        Extract generated text from SageMaker response

        Args:
            output: Raw HTTP response body (bytes) from SageMaker

        Returns:
            str: Generated text that LangChain will use

        For AI/ML Scientists:
            This is like:
            generated_ids = model.generate(...)
            generated_text = tokenizer.decode(generated_ids)

            But parsing from JSON instead of decoding tokens.

        Response Structure (HuggingFace TGI):
            [
                {
                    "generated_text": "The actual model output",
                    "details": {
                        "finish_reason": "length" | "eos_token" | "stop_sequence",
                        "generated_tokens": 487,
                        "tokens": [...]  // Optional token-level details
                    }
                }
            ]

        Why array: TGI supports batch inference.
        Even for 1 prompt, response is array with 1 element.
        """

        # Decode bytes to string
        response_str = output.read().decode('utf-8')

        # Parse JSON
        response_json = json.loads(response_str)

        # Extract generated text from first (and only) result
        generated_text = response_json[0]["generated_text"]

        # Log for debugging (truncate long responses)
        logger.debug(f"SageMaker response: {generated_text[:200]}...")

        return generated_text


def create_sagemaker_llm():
    """
    Create and configure the SageMaker LLM client

    Returns:
        SagemakerEndpoint: Configured LangChain LLM instance

    Environment Variables:
        SAGEMAKER_ENDPOINT_NAME: Name of the SageMaker endpoint
        AWS_DEFAULT_REGION: AWS region (defaults to us-east-1)

    For AI/ML Scientists:
        This function is called once when initializing the agent.
        The LLM instance is then reused for all inferences during
        the Lambda container's lifetime.

        Reusing the instance is important because:
        - Boto3 clients have connection pooling (faster subsequent calls)
        - ContentHandler instance is reused (no overhead)
        - Configuration is validated once, not on every call
    """

    # Get configuration from environment variables
    endpoint_name = os.environ.get('SAGEMAKER_ENDPOINT_NAME')
    region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

    if not endpoint_name:
        raise ValueError(
            "SAGEMAKER_ENDPOINT_NAME environment variable is required. "
            "This should be set by the Lambda deployment (CDK/CloudFormation)."
        )

    logger.info(f"Initializing SageMaker LLM client for endpoint: {endpoint_name}")

    # Create content handler instance
    content_handler = MistralContentHandler()

    # Create LangChain SageMaker LLM
    llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name=region,

        # Model generation parameters
        # These control the model's behavior during text generation
        model_kwargs={
            # ================================================================
            # max_new_tokens: Maximum number of tokens to generate
            # ================================================================
            # This is CRITICAL for agent functionality!
            #
            # Why 500:
            # - Agent needs to generate complete XML tags for tool calls
            # - Incomplete XML (e.g., "<tool>search</tool><tool_input>query")
            #   breaks the parser
            # - A complete response might be:
            #   "<tool>tavily_search</tool><tool_input>latest UK storm</tool_input>"
            #   (~30 tokens for XML + ~50 tokens for query = ~80 tokens)
            # - Final answers might be longer (200-400 tokens)
            # - 500 provides safe buffer
            #
            # Too low (e.g., 128): Agent breaks due to truncated XML
            # Too high (e.g., 2048): Higher cost + latency, no benefit
            #
            # Cost impact: ~200ms per token generated
            # 500 tokens ≈ 100 seconds ≈ $0.028 per call (on ml.g5.xlarge)
            "max_new_tokens": 500,

            # ================================================================
            # temperature: Sampling randomness (0.0 = deterministic)
            # ================================================================
            # Why 0.001 (near-zero):
            # - Tool calling requires precise XML formatting
            # - High temperature → creative but unreliable formatting
            # - Low temperature → consistent, predictable output
            #
            # Example comparison:
            # Temperature 0.001 (good):
            #   "<tool>search</tool><tool_input>query</tool_input>"
            #
            # Temperature 0.9 (bad):
            #   "<tool>search<tool_input>cool query!!</tool_input></tool>"
            #   ↑ Missing closing tag, extra words → parser fails
            #
            # For non-agent use cases (creative writing, brainstorming),
            # you'd want higher temperature (0.7-0.9)
            "temperature": 0.001,

            # ================================================================
            # do_sample: Enable probabilistic sampling
            # ================================================================
            # Why True:
            # - Required for temperature to have any effect
            # - False = greedy decoding (always pick highest probability token)
            # - True + low temp ≈ greedy, but allows some variation
            #
            # Even with very low temperature, do_sample must be True
            "do_sample": True,

            # ================================================================
            # Optional parameters (uncomment to use)
            # ================================================================

            # top_p (nucleus sampling): Alternative to temperature
            # - 0.9 = consider tokens comprising top 90% probability mass
            # - Good for creative tasks, not needed for agents
            # "top_p": 0.9,

            # top_k (top-k sampling): Limit to top K tokens
            # - 50 = only consider top 50 most likely tokens
            # - Reduces chance of off-topic generation
            # "top_k": 50,

            # repetition_penalty: Penalize repeated tokens
            # - 1.0 = no penalty (default)
            # - 1.2 = discourage repetition
            # - Useful for creative writing, not critical for agents
            # "repetition_penalty": 1.0,

            # stop_sequences: Auto-stop generation when sequence appears
            # - ["</final_answer>", "Human:"] = stop when these appear
            # - Can save tokens if final answer is short
            # "stop_sequences": ["</final_answer>"],
        },

        # Content handler for format translation
        content_handler=content_handler,
    )

    logger.info("SageMaker LLM client initialized successfully")
    return llm


# Example usage (for testing)
if __name__ == '__main__':
    # This won't work outside Lambda (no environment variables set)
    # But shows how to use the LLM client

    # Set dummy environment variable for testing
    os.environ['SAGEMAKER_ENDPOINT_NAME'] = 'test-endpoint'

    llm = create_sagemaker_llm()

    # Test prompt
    test_prompt = "<s>[INST] What is 2 + 2? [/INST]"

    print("Sending test prompt to LLM...")
    response = llm.invoke(test_prompt)
    print(f"Response: {response}")
