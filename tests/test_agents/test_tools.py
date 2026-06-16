# Unit tests for Agent Tools (LangChain @tool)

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure Worker2/ is on sys.path so `src` package can be found
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class TestAgentTools(unittest.TestCase):
    """Test LangChain tool definitions"""

    def test_tool_imports(self):
        """Test all tools can be imported"""
        from src.agents.tools import (
            research_tool,
            write_article_tool,
            edit_content_tool,
            style_transfer_tool,
            ALL_TOOLS,
            RESEARCH_TOOLS,
            WRITER_TOOLS,
            EDITOR_TOOLS,
        )
        # All tool lists should be non-empty
        self.assertGreater(len(ALL_TOOLS), 0)
        self.assertGreater(len(RESEARCH_TOOLS), 0)
        self.assertGreater(len(WRITER_TOOLS), 0)
        self.assertGreater(len(EDITOR_TOOLS), 0)

    def test_tools_are_callable(self):
        """Test tools are LangChain tool objects"""
        from src.agents.tools import ALL_TOOLS
        for tool in ALL_TOOLS:
            self.assertTrue(callable(tool.func) or hasattr(tool, "invoke"))

    def test_tool_names(self):
        """Test tools have proper names"""
        from src.agents.tools import ALL_TOOLS
        names = [t.name for t in ALL_TOOLS]
        self.assertIn("research_tool", names)
        self.assertIn("write_article_tool", names)
        self.assertIn("edit_content_tool", names)
        self.assertIn("style_transfer_tool", names)


class TestAgentResult(unittest.TestCase):
    """Test AgentResult data class"""

    def test_agent_result_creation(self):
        """Test AgentResult dataclass"""
        from src.agents.researcher_agent import AgentResult
        result = AgentResult(
            output="test output",
            agent_name="researcher",
            metadata={"elapsed_ms": 100.0, "prompt_tokens": 30, "completion_tokens": 20},
        )
        self.assertEqual(result.output, "test output")
        self.assertEqual(result.agent_name, "researcher")
        self.assertEqual(result.elapsed_ms, 100.0)
        self.assertEqual(result.total_tokens, 50)


if __name__ == "__main__":
    unittest.main()
