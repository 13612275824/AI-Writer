# Unit tests for LLM Adapter (LangChain ChatOpenAI wrapper)

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure Worker2/ is on sys.path so `src` package can be found
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class TestLLMAdapter(unittest.TestCase):
    """Test LLMAdapter with mocked ChatOpenAI"""

    @patch("src.core.llm_adapter.ChatOpenAI")
    @patch("src.core.llm_adapter.get_config")
    def setUp(self, mock_get_config, mock_chat_openai):
        """Set up test fixtures"""
        cfg = MagicMock()
        cfg.default_model = "qwen-plus"
        cfg.temperature = 0.7
        cfg.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        cfg.api_key = "test-api-key"
        cfg.max_tokens = 2048
        mock_get_config.return_value = cfg

        # Reset singleton
        import src.core.llm_adapter as module
        module._llm_client_instance = None

        from src.core.llm_adapter import LLMAdapter
        self.adapter = LLMAdapter()

    def test_model_attribute(self):
        """Test model property returns configured model name"""
        self.assertEqual(self.adapter.model, "qwen-plus")

    @patch.object(
        __import__("src.core.llm_adapter", fromlist=["LLMAdapter"]).LLMAdapter,
        "chat_completion_simple",
    )
    def test_chat_completion_simple(self, mock_method):
        """Test simple chat completion returns string"""
        mock_method.return_value = "Hello, world!"
        result = self.adapter.chat_completion_simple("Hi")
        self.assertIsInstance(result, str)

    def test_chat_completion_stream_returns_generator(self):
        """Test streaming returns a generator-like object"""
        # Mock the internal _llm to simulate streaming
        mock_chunk = MagicMock()
        mock_chunk.content = "chunk"
        self.adapter._llm = MagicMock()
        self.adapter._llm.stream.return_value = [mock_chunk]

        result = self.adapter.chat_completion_stream([{"role": "user", "content": "Hi"}])
        # Should be iterable
        chunks = list(result)
        self.assertGreater(len(chunks), 0)


class TestLLMAdapterSingleton(unittest.TestCase):
    """Test singleton pattern"""

    @patch("src.core.llm_adapter.ChatOpenAI")
    @patch("src.core.llm_adapter.get_config")
    def test_get_llm_client_returns_same_instance(self, mock_get_config, mock_chat_openai):
        """Test get_llm_client returns singleton"""
        cfg = MagicMock()
        cfg.default_model = "qwen-plus"
        cfg.temperature = 0.7
        cfg.base_url = "https://test.com"
        cfg.api_key = "test-key"
        cfg.max_tokens = 2048
        mock_get_config.return_value = cfg

        import src.core.llm_adapter as module
        module._llm_client_instance = None

        from src.core.llm_adapter import get_llm_client
        client1 = get_llm_client()
        client2 = get_llm_client()
        self.assertIs(client1, client2)


if __name__ == "__main__":
    unittest.main()
