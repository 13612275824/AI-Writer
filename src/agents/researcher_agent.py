# 研究 Agent（基于 LangChain ReAct）
#
# 使用 LangChain ReAct Agent 配合 research_tool 从知识库中
# 检索并汇总信息，替代 Worker1 中自行实现的 ResearcherAgent。

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents.tools import RESEARCH_TOOLS, research_tool
from src.core.config import get_config
from src.core.exceptions import AgentError
from src.core.llm_adapter import get_llm_client


@dataclass
class AgentResult:
    """Agent 执行结果

    兼容 Worker1 的 AgentResult 接口。

    属性:
        output: Agent 最终输出文本
        agent_name: 产生此结果的 Agent 名称
        metadata: 附加元数据
    """
    output: str = ""
    agent_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """总 token 数（提示词 + 补全）"""
        prompt = self.metadata.get("prompt_tokens", 0) or 0
        completion = self.metadata.get("completion_tokens", 0) or 0
        return prompt + completion

    @property
    def elapsed_ms(self) -> float:
        """执行耗时（毫秒）"""
        return self.metadata.get("elapsed_ms", 0.0)

    @property
    def reflection_count(self) -> int:
        """反思次数"""
        return self.metadata.get("reflection_count", 0)

    def __repr__(self) -> str:
        output_preview = (
            self.output[:50] + "..." if len(self.output) > 50 else self.output
        )
        return (
            f"<AgentResult agent={self.agent_name!r} "
            f"output_len={len(self.output)} "
            f"elapsed={self.elapsed_ms:.1f}ms>"
        )


class ResearcherAgent:
    """研究 Agent（基于 LangChain ReAct）

    使用 LangChain ReAct Agent 自主执行以下操作：
    1. 从任务中提取搜索关键词
    2. 通过 research_tool 查询知识库
    3. 汇总和整理检索结果
    4. 如果信息不足则进行迭代搜索

    示例:
        >>> agent = ResearcherAgent()
        >>> result = agent.execute("研究 AI 在医疗领域的应用")
        >>> print(result.output)
    """

    def __init__(self):
        """初始化研究 Agent"""
        config = get_config()
        self._llm_adapter = get_llm_client()

        # 获取 LangChain ChatOpenAI 实例
        self._llm = self._llm_adapter._llm

        # 创建 Tool Calling 提示词模板
        self._prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional research assistant. Your task is to find "
             "and summarize relevant information from the knowledge base.\n\n"
             "Use the research_tool to search for information. Make multiple "
             "searches with different keywords if needed.\n\n"
             "Provide a comprehensive, well-organized summary of your findings."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 创建 Tool Calling Agent（OpenAI 函数调用风格）
        agent = create_tool_calling_agent(
            llm=self._llm,
            tools=RESEARCH_TOOLS,
            prompt=self._prompt,
        )
        # 创建 AgentExecutor（Agent 执行器）
        self._executor = AgentExecutor(
            agent=agent,
            tools=RESEARCH_TOOLS,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        print("[ResearcherAgent] 已初始化 LangChain ReAct Agent")

    def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """执行研究任务

        参数:
            task: 研究任务描述
            context: 来自前序阶段的可选上下文

        返回:
            AgentResult，包含研究发现

        异常:
            AgentError: 执行失败时抛出
        """
        start_time = time.time()

        try:
            # 构建输入文本（合并上下文）
            input_text = task
            if context:
                context_str = context.get("research", "")
                if context_str:
                    input_text = f"Previous research context:\n{context_str}\n\nNew task: {task}"

            # 执行 Agent（通过 callback 捕获 token 用量）
            with get_openai_callback() as cb:
                result = self._executor.invoke({"input": input_text})

            elapsed_ms = (time.time() - start_time) * 1000

            return AgentResult(
                output=result.get("output", ""),
                agent_name="researcher",
                metadata={
                    "elapsed_ms": elapsed_ms,
                    "model": self._llm_adapter.model,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                }
            )

        except Exception as e:
            raise AgentError(f"研究 Agent 执行失败: {str(e)}") from e
