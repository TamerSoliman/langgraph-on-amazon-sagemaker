"""
Unit Tests for SageMaker Content Handler

These tests validate the request/response transformation logic
without actually calling SageMaker.

For AI/ML Scientists:
- Unit tests = test individual functions in isolation
- No external dependencies (no network, no AWS calls)
- Fast to run (~milliseconds per test)
- Run these frequently during development
"""

import pytest
import json
from unittest.mock import Mock, MagicMock

# Import from agent code
import sys
sys.path.insert(0, '/home/user/langgraph-on-amazon-sagemaker/agent')
from sagemaker_llm import MistralContentHandler


class TestContentHandlerInput:
    """Test suite for transform_input method"""

    @pytest.fixture
    def handler(self):
        return MistralContentHandler()

    def test_basic_prompt_transformation(self, handler):
        """Test that a simple prompt is correctly transformed"""
        prompt = "<s>[INST] What is 2+2? [/INST]"
        model_kwargs = {"max_new_tokens": 500, "temperature": 0.001}

        result = handler.transform_input(prompt, model_kwargs)

        # Should return bytes
        assert isinstance(result, bytes)

        # Decode and parse JSON
        payload = json.loads(result.decode('utf-8'))

        # Verify structure
        assert "inputs" in payload
        assert "parameters" in payload
        assert payload["inputs"] == prompt
        assert payload["parameters"]["max_new_tokens"] == 500
        assert payload["parameters"]["temperature"] == 0.001

    def test_empty_model_kwargs(self, handler):
        """Test with empty model_kwargs"""
        prompt = "Test prompt"
        model_kwargs = {}

        result = handler.transform_input(prompt, model_kwargs)
        payload = json.loads(result.decode('utf-8'))

        assert payload["parameters"] == {}

    def test_additional_parameters(self, handler):
        """Test with additional generation parameters"""
        prompt = "Test prompt"
        model_kwargs = {
            "max_new_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1
        }

        result = handler.transform_input(prompt, model_kwargs)
        payload = json.loads(result.decode('utf-8'))

        # Verify all parameters are preserved
        assert payload["parameters"] == model_kwargs

    def test_unicode_handling(self, handler):
        """Test handling of non-ASCII characters"""
        prompt = "What is café? 你好"
        model_kwargs = {"max_new_tokens": 100}

        result = handler.transform_input(prompt, model_kwargs)
        payload = json.loads(result.decode('utf-8'))

        # Unicode should be preserved
        assert payload["inputs"] == prompt


class TestContentHandlerOutput:
    """Test suite for transform_output method"""

    @pytest.fixture
    def handler(self):
        return MistralContentHandler()

    def test_basic_response_parsing(self, handler):
        """Test parsing a standard SageMaker response"""
        # Mock response from SageMaker
        response_data = [{
            "generated_text": "The answer is 4."
        }]

        # Create mock StreamingBody
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(response_data).encode('utf-8')

        result = handler.transform_output(mock_body)

        assert result == "The answer is 4."
        mock_body.read.assert_called_once()

    def test_response_with_xml_tags(self, handler):
        """Test parsing response with XML tool calls"""
        response_data = [{
            "generated_text": "<tool>search</tool><tool_input>query</tool_input>"
        }]

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(response_data).encode('utf-8')

        result = handler.transform_output(mock_body)

        assert "<tool>search</tool>" in result
        assert "<tool_input>query</tool_input>" in result

    def test_response_with_details(self, handler):
        """Test that details field is ignored (we only need generated_text)"""
        response_data = [{
            "generated_text": "The answer is 42.",
            "details": {
                "finish_reason": "eos_token",
                "generated_tokens": 10,
                "tokens": []
            }
        }]

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(response_data).encode('utf-8')

        result = handler.transform_output(mock_body)

        # Should extract only generated_text, ignore details
        assert result == "The answer is 42."

    def test_malformed_json_raises_error(self, handler):
        """Test that malformed JSON raises appropriate error"""
        mock_body = MagicMock()
        mock_body.read.return_value = b"not valid json"

        with pytest.raises(json.JSONDecodeError):
            handler.transform_output(mock_body)

    def test_missing_generated_text_key(self, handler):
        """Test error handling when generated_text key is missing"""
        response_data = [{
            "wrong_key": "some value"
        }]

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(response_data).encode('utf-8')

        with pytest.raises(KeyError):
            handler.transform_output(mock_body)


class TestContentHandlerHeaders:
    """Test suite for content type headers"""

    def test_content_type_header(self):
        """Verify content_type is application/json"""
        handler = MistralContentHandler()
        assert handler.content_type == "application/json"

    def test_accepts_header(self):
        """Verify accepts header is application/json"""
        handler = MistralContentHandler()
        assert handler.accepts == "application/json"
