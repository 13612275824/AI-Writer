# 编辑 Agent（基于 LangChain ReAct）
#
# 使用 LangChain ReAct Agent 配合编辑工具，对内容进行
# 润色和质量优化，替代 Worker1 中自行实现的 EditorAgent。

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents.tools import EDITOR_TOOLS
from src.agents.researcher_agent import AgentResult
from src.core.config import get_config
from src.core.exceptions import AgentError
from src.core.llm_adapter import get_llm_client


class EditorAgent:
    """编辑 Agent（基于 LangChain ReAct）

    使用 LangChain ReAct Agent 执行以下操作：
    1. 校对并修复语法问题
    2. 改进表达和可读性
    3. 调整结构和组织
    4. 生成编辑摘要

    示例:
        >>> agent = EditorAgent()
        >>> result = agent.execute(
        ...     "编辑并改进这篇文章",
        ...     context={"draft": "文章草稿内容..."}
        ... )
        >>> print(result.output)
    """

    def __init__(self):
        """初始化编辑 Agent"""
        config = get_config()
        self._llm_adapter = get_llm_client()

        # 获取 LangChain ChatOpenAI 实例
        self._llm = self._llm_adapter._llm

        # 创建 Tool Calling 提示词模板
        self._prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional editor and proofreader. Your task is to "
             "improve the quality of written content while maintaining its "
             "original meaning and tone.\n\n"
             "Apply multiple editing passes:\n"
             "1. First, check and fix grammar issues\n"
             "2. Then, improve expression and readability\n"
             "3. Finally, adjust structure if needed\n\n"
             "Return the final polished version of the content."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 创建 Tool Calling Agent（OpenAI 函数调用风格）
        agent = create_tool_calling_agent(
            llm=self._llm,
            tools=EDITOR_TOOLS,
            prompt=self._prompt,
        )
        # 创建 AgentExecutor（Agent 执行器）
        self._executor = AgentExecutor(
            agent=agent,
            tools=EDITOR_TOOLS,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        print("[EditorAgent] 已初始化 LangChain ReAct Agent")

    def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """执行编辑任务

        参数:
            task: 编辑任务描述
            context: 可选上下文（通常包含草稿内容）

        返回:
            AgentResult，包含编辑后的内容

        异常:
            AgentError: 执行失败时抛出
        """
        start_time = time.time()

        try:
            # 构建输入文本（合并草稿上下文）
            input_text = task
            if context:
                draft = context.get("draft", "")
                if draft:
                    input_text = (
                        f"Draft content to edit:\n{draft}\n\n"
                        f"---\n\n"
                        f"Editing task: {task}"
                    )

            # 执行 Agent（通过 callback 捕获 token 用量）
            with get_openai_callback() as cb:
                result = self._executor.invoke({"input": input_text})

            elapsed_ms = (time.time() - start_time) * 1000

            return AgentResult(
                output=result.get("output", ""),
                agent_name="editor",
                metadata={
                    "elapsed_ms": elapsed_ms,
                    "model": self._llm_adapter.model,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                }
            )

        except Exception as e:
            raise AgentError(f"编辑 Agent 执行失败: {str(e)}") from e
