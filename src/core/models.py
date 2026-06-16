# Compatibility module
#
# Re-exports LLMAdapter as LLMClient for writing modules that
# reference src.core.models.

from src.core.llm_adapter import LLMAdapter as LLMClient  # noqa: F401
from src.core.llm_adapter import get_llm_client  # noqa: F401
