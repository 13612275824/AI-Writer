# Unit tests for Prompt Adapter (LangChain ChatPromptTemplate wrapper)

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure Worker2/ is on sys.path so `src` package can be found
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class TestPromptAdapter(unittest.TestCase):
    """Test PromptAdapter with mocked config"""

    @patch("src.core.prompt_adapter.get_config")
    def setUp(self, mock_get_config):
        cfg = MagicMock()
        cfg.get_system_prompt.return_value = "You are a helpful assistant."
        cfg.get_user_template.return_value = "{input}"
        mock_get_config.return_value = cfg

        import src.core.prompt_adapter as module
        module._prompt_adapter_instance = None

        from src.core.prompt_adapter import PromptAdapter
        self.adapter = PromptAdapter()

    def test_get_prompt_returns_string(self):
        """Test get_prompt returns a rendered string"""
        result = self.adapter.get_prompt("writing", input="test")
        self.assertIsInstance(result, str)

    def test_get_prompt_caches_templates(self):
        """Test templates are cached after first call"""
        self.adapter.get_prompt("writing", input="test1")
        self.assertIn("writing", self.adapter._templates)

    def test_get_chat_prompt_template_returns_object(self):
        """Test get_chat_prompt_template returns ChatPromptTemplate"""
        from langchain.prompts import ChatPromptTemplate
        # First cache the template
        self.adapter.get_prompt("writing", input="test")
        template = self.adapter.get_chat_prompt_template("writing")
        self.assertIsInstance(template, ChatPromptTemplate)

    def test_clear_cache(self):
        """Test clear_cache empties template cache"""
        self.adapter.get_prompt("writing", input="test")
        self.adapter.clear_cache()
        self.assertEqual(len(self.adapter._templates), 0)

    @patch("src.core.prompt_adapter.get_config")
    def test_missing_role_raises_error(self, mock_get_config):
        """Test missing role config raises ValueError"""
        cfg = MagicMock()
        cfg.get_system_prompt.return_value = ""
        cfg.get_user_template.return_value = ""
        mock_get_config.return_value = cfg

        from src.core.prompt_adapter import PromptAdapter
        adapter = PromptAdapter()
        with self.assertRaises(ValueError):
            adapter.get_prompt("nonexistent_role")


class TestPromptAdapterSingleton(unittest.TestCase):

    @patch("src.core.prompt_adapter.get_config")
    def test_get_prompt_manager_singleton(self, mock_get_config):
        cfg = MagicMock()
        cfg.get_system_prompt.return_value = "system"
        cfg.get_user_template.return_value = "user"
        mock_get_config.return_value = cfg

        import src.core.prompt_adapter as module
        module._prompt_adapter_instance = None

        from src.core.prompt_adapter import get_prompt_manager
        m1 = get_prompt_manager()
        m2 = get_prompt_manager()
        self.assertIs(m1, m2)


if __name__ == "__main__":
    unittest.main()
