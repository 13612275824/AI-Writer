# 写作 Agent（基于 LangChain ReAct）
#
# 使用 LangChain ReAct Agent 配合写作工具，根据研究材料
# 生成文章内容，替代 Worker1 中自行实现的 WriterAgent。

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents.tools import WRITER_TOOLS
from src.agents.researcher_agent import AgentResult
from src.core.config import get_config
from src.core.exceptions import AgentError
from src.core.llm_adapter import get_llm_client


class WriterAgent:
    """写作 Agent（基于 LangChain ReAct）

    使用 LangChain ReAct Agent 执行以下操作：
    1. 整合前序阶段的研究材料
    2. 生成文章大纲
    3. 根据大纲创作完整内容
    4. 应用合适的写作风格

    示例:
        >>> agent = WriterAgent()
        >>> result = agent.execute(
        ...     "撰写一篇关于 AI 在医疗领域应用的文章",
        ...     context={"research": "前期研究发现..."}
        ... )
        >>> print(result.output)
    """

    def __init__(self):
        """初始化写作 Agent"""
        config = get_config()
        self._llm_adapter = get_llm_client()

        # 获取 LangChain ChatOpenAI 实例
        self._llm = self._llm_adapter._llm

        # 创建 Tool Calling 提示词模板
        self._prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional writer and content creator. Your task is to "
             "write well-structured, engaging articles based on provided research "
             "materials and topic.\n\n"
             "Create comprehensive, well-organized content with clear structure."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 创建 Tool Calling Agent（OpenAI 函数调用风格）
        agent = create_tool_calling_agent(
            llm=self._llm,
            tools=WRITER_TOOLS,
            prompt=self._prompt,
        )
        # 创建 AgentExecutor（Agent 执行器）
        self._executor = AgentExecutor(
            agent=agent,
            tools=WRITER_TOOLS,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        print("[WriterAgent] 已初始化 LangChain ReAct Agent")

    def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """执行写作任务

        参数:
            task: 写作任务描述
            context: 可选上下文（通常包含研究结果）

        返回:
            AgentResult，包含生成的文章内容

        异常:
            AgentError: 执行失败时抛出
        """
        start_time = time.time()

        try:
            # 构建输入文本（合并研究上下文）
            input_text = task
            if context:
                research = context.get("research", "")
                if research:
                    input_text = (
                        f"Research materials:\n{research}\n\n"
                        f"---\n\n"
                        f"Task: {task}"
                    )

            # 执行 Agent（通过 callback 捕获 token 用量）
            with get_openai_callback() as cb:
                result = self._executor.invoke({"input": input_text})

            elapsed_ms = (time.time() - start_time) * 1000

            return AgentResult(
                output=result.get("output", ""),
                agent_name="writer",
                metadata={
                    "elapsed_ms": elapsed_ms,
                    "model": self._llm_adapter.model,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                }
            )

        except Exception as e:
            raise AgentError(f"写作 Agent 执行失败: {str(e)}") from e
