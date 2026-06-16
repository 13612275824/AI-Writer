# Agent 工具集（基于 LangChain Tools）
#
# 定义可供 LangChain Agent 使用的工具，用于研究、写作和编辑任务。

from typing import Optional

from langchain.tools import tool


@tool
def research_tool(query: str) -> str:
    """研究工具：从知识库检索相关信息

    使用此工具从文档知识库中搜索和检索信息，以支持研究任务。

    参数:
        query: 研究问题或搜索关键词

    返回:
        检索到的相关信息摘要
    """
    from src.rag.query_engine import get_generator

    query_engine = get_generator()
    result = query_engine.generate(query)
    return result.answer


@tool
def write_article_tool(topic: str, outline: str = "", style: str = "formal") -> str:
    """写作工具：根据主题生成文章内容

    使用此工具创建文章内容。提供主题，可选地提供大纲和风格偏好。

    参数:
        topic: 文章主题或议题
        outline: 可选的文章大纲/结构
        style: 写作风格（formal 正式 / casual 随意 / creative 创意）

    返回:
        生成的文章内容
    """
    from src.core.llm_adapter import get_llm_client
    from src.core.config import get_config

    config = get_config()
    llm = get_llm_client()

    system_prompt = config.get_system_prompt("writing") or (
        "You are a professional writer. Generate well-structured, "
        "informative content on the given topic."
    )

    user_prompt = f"Write an article about: {topic}"
    if outline:
        user_prompt += f"\n\nFollow this outline:\n{outline}"
    user_prompt += f"\n\nWriting style: {style}"

    return llm.chat_completion_simple(
        prompt=user_prompt,
        system_prompt=system_prompt,
    )


@tool
def edit_content_tool(content: str, edit_type: str = "polish") -> str:
    """编辑工具：润色和优化内容

    使用此工具通过多种编辑模式改进现有内容。

    参数:
        content: 待编辑的文本内容
        edit_type: 编辑类型 - polish（润色表达）、simplify（简化精炼）、
                   expand（扩展细节）、grammar_check（语法检查）、
                   restructure（重组结构）

    返回:
        编辑改进后的内容
    """
    from src.core.llm_adapter import get_llm_client
    from src.core.config import get_config

    config = get_config()
    llm = get_llm_client()

    system_prompt = config.get_system_prompt("editing") or (
        "You are a professional editor. Improve the given text while "
        "maintaining its original meaning and tone."
    )

    # 编辑指令映射
    edit_instructions = {
        "polish": "Polish and improve the expression of the following text",
        "simplify": "Simplify and make the following text more concise",
        "expand": "Expand the following text with more details",
        "grammar_check": "Check and fix grammar issues in the following text",
        "restructure": "Restructure and reorganize the following text",
    }

    instruction = edit_instructions.get(edit_type, edit_instructions["polish"])
    user_prompt = f"{instruction}:\n\n{content}"

    return llm.chat_completion_simple(
        prompt=user_prompt,
        system_prompt=system_prompt,
    )


@tool
def style_transfer_tool(content: str, target_style: str = "polished") -> str:
    """风格迁移工具：将内容转换为不同的写作风格

    使用此工具将现有内容转换为指定的目标风格。

    参数:
        content: 待转换的文本内容
        target_style: 目标风格（polished 润色 / academic 学术 / casual 随意 /
                      business 商务 / creative 创意 / journalistic 新闻 /
                      technical 技术 / narrative 叙事），默认 polished

    返回:
        转换为目标风格后的内容
    """
    from src.core.llm_adapter import get_llm_client

    llm = get_llm_client()

    system_prompt = (
        "You are a skilled writer who can adapt content to any style. "
        "Transform the given text while preserving its core meaning."
    )

    user_prompt = (
        f"Transform the following text into {target_style} style:\n\n{content}"
    )

    return llm.chat_completion_simple(
        prompt=user_prompt,
        system_prompt=system_prompt,
    )


# 各 Agent 可用的工具列表
ALL_TOOLS = [research_tool, write_article_tool, edit_content_tool, style_transfer_tool]
RESEARCH_TOOLS = [research_tool]                                        # 研究 Agent 专用工具
WRITER_TOOLS = [research_tool, write_article_tool, style_transfer_tool] # 写作 Agent 专用工具
EDITOR_TOOLS = [edit_content_tool, style_transfer_tool]                 # 编辑 Agent 专用工具
