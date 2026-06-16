# Prompts compatibility module
#
# Provides get_prompt() and build_messages() helpers used by writing modules.
# Bridges the gap between Config/PromptAdapter and the OpenAI message format
# expected by writing modules' _call_llm methods.

from typing import Any, Dict, List

from src.core.config import get_config


def get_prompt(role: str) -> Dict[str, str]:
    """Get prompt configuration for the specified role.

    Args:
        role: Role name (writing/editing/research/etc.)

    Returns:
        Dict with "system" and optionally "user_template" keys.
        Returns empty dict if role not found.
    """
    config = get_config()
    return config.get_prompt(role)


def build_messages(role: str, **variables: Any) -> List[Dict[str, str]]:
    """Build an OpenAI-format message list from prompts.yaml config.

    Reads the system prompt and user template for *role*, fills in
    *variables* in the user template, and returns a list of
    ``{"role": ..., "content": ...}`` dicts suitable for
    ``LLMAdapter.chat_completion()``.

    Args:
        role: Role name (must exist in prompts.yaml).
        **variables: Template variables to interpolate into the user template.

    Returns:
        List of message dicts, e.g.
        ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``
    """
    config = get_config()
    prompt_cfg = config.get_prompt(role)

    messages: List[Dict[str, str]] = []

    # System message
    system_text = prompt_cfg.get("system", "")
    if system_text:
        messages.append({"role": "system", "content": system_text})

    # User message – render template variables
    user_template = prompt_cfg.get("user_template", "")
    if user_template:
        try:
            user_text = user_template.format(**variables)
        except KeyError:
            # Leave unreplaced placeholders as-is when a variable is missing
            user_text = user_template
        messages.append({"role": "user", "content": user_text})

    return messages
