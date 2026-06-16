# 查询引擎适配器（基于 LlamaIndex + LangChain LCEL）
#
# 本模块封装 LlamaIndex 的 RetrieverQueryEngine，提供端到端的
# RAG 问答能力，替代 Worker1 中自行实现的 Generator。
#
# 数据流：用户查询 -> 检索器搜索 -> 上下文构建 -> LLM 生成 -> GenerationResult
#
# 功能特性：
# - 单轮问答生成（检索 + 生成一体化，基于 LCEL Chain）
# - 流式生成（逐块输出，适合实时展示）
# - 直接生成模式（跳过检索，直接调用 LLM）
# - 多轮对话（chat() 方法，支持对话历史压缩）
# - 来源引用（附加参考来源）

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator as GenType, List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.core.config import get_config
from src.core.exceptions import ModelAPIError
from src.core.llm_adapter import get_llm_client
from src.rag.retriever import RetrievalResult, RetrieverAdapter, get_retriever
from src.rag.vector_store import get_vector_store


# ------------------------------------------------------------------ #
#                    数据结构                                          #
# ------------------------------------------------------------------ #

@dataclass
class GenerationResult:
    """生成结果

    兼容 Worker1 的 GenerationResult 接口。

    属性:
        answer: 模型生成的回答文本
        query: 原始用户查询
        sources: 参考来源列表（去重后的文档路径）
        context_used: 实际使用的检索上下文文本
        retrieval_result: 原始检索结果（可选，用于调试）
        elapsed_ms: 总耗时（毫秒），包含检索和生成
        model: 使用的模型名称
        prompt_tokens: 提示词 token 数量
        completion_tokens: 补全 token 数量
    """
    answer: str = ""
    query: str = ""
    sources: List[str] = field(default_factory=list)
    context_used: str = ""
    retrieval_result: Optional[RetrievalResult] = None
    elapsed_ms: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def has_sources(self) -> bool:
        """是否包含来源引用"""
        return len(self.sources) > 0

    @property
    def total_tokens(self) -> int:
        """总 token 数（提示词 + 补全）"""
        return self.prompt_tokens + self.completion_tokens

    def __repr__(self) -> str:
        answer_preview = (
            self.answer[:60] + "..." if len(self.answer) > 60 else self.answer
        )
        return (
            f"<GenerationResult query={self.query!r} "
            f"answer_len={len(self.answer)} "
            f"sources={len(self.sources)} "
            f"elapsed={self.elapsed_ms:.1f}ms>"
        )


@dataclass
class ChatMessage:
    """对话消息

    用于 chat() 多轮对话中记录每一轮的问答。

    属性:
        role: 消息角色（"user" 或 "assistant"）
        content: 消息内容
    """
    role: str = ""       # "user" 或 "assistant"
    content: str = ""

    def __repr__(self) -> str:
        preview = (
            self.content[:40] + "..." if len(self.content) > 40 else self.content
        )
        return f"<ChatMessage role={self.role!r} content={preview!r}>"


# ------------------------------------------------------------------ #
#                    查询引擎适配器                                     #
# ------------------------------------------------------------------ #

class QueryEngineAdapter:
    """LlamaIndex 查询引擎适配器

    职责：
    1. 整合 Retriever + LLM，实现端到端 RAG 问答
    2. 支持流式和非流式生成
    3. 支持直接生成模式（跳过检索）
    4. 提供与 Worker1 Generator 兼容的接口

    示例:
        >>> engine = QueryEngineAdapter()
        >>> result = engine.generate("AI 有哪些应用场景？")
        >>> print(result.answer)
        >>> for src in result.sources:
        ...     print(f"  来源: {src}")
    """

    def __init__(
        self,
        retriever: Optional[RetrieverAdapter] = None,
        system_prompt: Optional[str] = None,
        max_context_chars: Optional[int] = None,
        include_sources: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """初始化查询引擎适配器

        参数:
            retriever: RetrieverAdapter 实例（为 None 时使用全局单例）
            system_prompt: 默认系统提示词（为 None 时从配置读取）
            max_context_chars: 最大上下文字符数（为 None 时从配置读取）
            include_sources: 是否包含来源引用（为 None 时从配置读取）
            temperature: 生成温度（为 None 时从配置读取）
            max_tokens: 最大生成 token 数（为 None 时从配置读取）
        """
        config = get_config()

        self._retriever_adapter = retriever or get_retriever()
        self._llm_client = get_llm_client()

        self._system_prompt = (
            system_prompt
            if system_prompt is not None
            else config.generator_default_system_prompt
        )
        self._max_context_chars = (
            max_context_chars
            if max_context_chars is not None
            else config.generator_max_context_chars
        )
        self._include_sources = (
            include_sources
            if include_sources is not None
            else config.generator_include_sources
        )
        self._temperature = (
            temperature if temperature is not None else config.temperature
        )
        self._max_tokens = (
            max_tokens if max_tokens is not None else config.max_tokens
        )

        # 初始化 LlamaIndex OpenAI LLM，用于查询引擎
        self._llama_llm = LlamaOpenAI(
            model=config.default_model or "qwen-plus",
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            api_key=config.api_key,
            api_base=config.base_url,
        )

        # 创建 LlamaIndex 查询引擎组件
        self._index = get_vector_store().index

        # 构建 LCEL Chain：检索 → 上下文构建 → Prompt 填充 → LLM 生成 → 解析
        self._chain = self._build_lcel_chain()

        # 多轮对话状态：对话历史 + 历史压缩摘要
        self._chat_history: List[ChatMessage] = []
        self._chat_history_summary: str = ""
        # 对话历史压缩阈值：原始历史超过此数量时触发压缩（单位：条消息）
        self._chat_history_compress_threshold = 6

        print(
            f"[QueryEngineAdapter] 已初始化: model={config.default_model}, "
            f"include_sources={self._include_sources}, LCEL Chain 已构建"
        )

    # ------------------------------------------------------------------ #
    #                    核心方法：带检索的生成                              #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        use_retrieval: bool = True,
    ) -> GenerationResult:
        """使用 RAG 生成回答（基于 LCEL Chain：检索 + 生成）

        通过 LCEL Chain 将检索、上下文构建、Prompt 填充、LLM 生成
        串联为一条声明式管道，替代原来的手动分步调用。

        参数:
            query: 用户查询
            system_prompt: 覆盖系统提示词
            top_k: 检索结果数量
            score_threshold: 分数过滤阈值
            use_retrieval: 是否启用检索增强（False 时跳过检索，直接调用 LLM）

        返回:
            GenerationResult，包含回答和元数据

        异常:
            ModelAPIError: 生成失败时抛出
        """
        if not query or not query.strip():
            raise ModelAPIError("查询文本为空")

        # use_retrieval=False 时跳过检索，直接调用 LLM
        if not use_retrieval:
            return self.generate_direct(
                prompt=query,
                system_prompt=system_prompt,
            )

        start_time = time.time()
        sys_prompt = system_prompt or self._system_prompt

        try:
            # LCEL Chain 一次性执行：query → retrieve → build_context → prompt → LLM → answer
            answer = self._chain.invoke(
                {"query": query, "system_prompt": sys_prompt}
            )

            # 从链中提取中间结果（用于填充 GenerationResult 的元数据）
            retrieval_result = self._last_retrieval_result
            context = self._last_context

            # 如果启用来源引用，附加来源说明
            if self._include_sources and context:
                answer += "\n\n---\n*Reference materials from knowledge base were used to generate this answer.*"

            # 收集来源
            sources = self._collect_sources(retrieval_result) if retrieval_result else []

            elapsed_ms = (time.time() - start_time) * 1000

            return GenerationResult(
                answer=answer,
                query=query,
                sources=sources,
                context_used=context or "",
                retrieval_result=retrieval_result,
                elapsed_ms=elapsed_ms,
                model=self._llm_client.model,
            )

        except Exception as e:
            raise ModelAPIError(f"RAG 生成失败: {str(e)}") from e

    # ------------------------------------------------------------------ #
    #                    流式生成                                          #
    # ------------------------------------------------------------------ #

    def generate_stream(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        use_retrieval: bool = True,
    ) -> GenType[str, None, None]:
        """带 RAG 的流式生成

        参数:
            query: 用户查询
            system_prompt: 覆盖系统提示词
            top_k: 检索结果数量
            score_threshold: 分数阈值
            use_retrieval: 是否启用检索增强（False 时跳过检索）

        产出:
            生成的文本片段
        """
        sys_prompt = system_prompt or self._system_prompt

        # use_retrieval=False 时跳过检索，直接使用纯 LLM 流式生成
        if use_retrieval:
            # 先执行检索
            retrieval_result = self._retriever_adapter.retrieve(
                query=query,
                top_k=top_k,
            )
            context = self._build_context(retrieval_result)
        else:
            context = ""

        # 构建消息列表用于流式生成
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": self._build_user_prompt(query, context)},
        ]

        # 从 LLM 流式获取输出
        for chunk in self._llm_client.chat_completion_stream(messages):
            yield chunk

    # ------------------------------------------------------------------ #
    #                    直接生成（无检索）                                  #
    # ------------------------------------------------------------------ #

    def generate_direct(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """不经过检索直接生成（纯 LLM 调用）

        参数:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 温度覆盖值
            max_tokens: 最大 token 数覆盖值

        返回:
            GenerationResult，包含回答
        """
        if not prompt or not prompt.strip():
            raise ModelAPIError("提示词文本为空")

        start_time = time.time()
        sys_prompt = system_prompt or self._system_prompt

        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]

            kwargs = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            answer = self._llm_client.chat_completion_simple(
                prompt=prompt,
                system_prompt=sys_prompt,
                **kwargs,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            return GenerationResult(
                answer=answer,
                query=prompt,
                sources=[],
                elapsed_ms=elapsed_ms,
                model=self._llm_client.model,
            )

        except Exception as e:
            raise ModelAPIError(f"直接生成失败: {str(e)}") from e

    # ------------------------------------------------------------------ #
    #                    多轮对话（ChatEngine）                              #
    # ------------------------------------------------------------------ #

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """多轮对话生成（基于对话历史压缩 + RAG 检索）

        借鉴 LlamaIndex CondenseQuestionChatEngine 的思想：
        1. 当对话历史超过阈值时，用 LLM 将旧历史压缩为摘要
        2. 检索知识库获取相关上下文
        3. 将摘要 + 上下文 + 当前问题组合为 Prompt
        4. 调用 LLM 生成回答
        5. 将问答记录追加到对话历史

        参数:
            message: 用户当前输入
            system_prompt: 覆盖系统提示词

        返回:
            GenerationResult，包含回答和元数据

        异常:
            ModelAPIError: 生成失败时抛出
        """
        if not message or not message.strip():
            raise ModelAPIError("消息文本为空")

        start_time = time.time()
        sys_prompt = system_prompt or self._system_prompt

        try:
            # 第1步：压缩对话历史（如果过长）
            history_context = self._condense_history()

            # 第2步：RAG 检索
            retrieval_result = self._retriever_adapter.retrieve(query=message)
            context = self._build_context(retrieval_result)

            # 第3步：构建包含历史摘要和上下文的用户消息
            user_message = self._build_chat_prompt(message, context, history_context)

            # 第4步：调用 LLM 生成回答
            answer = self._llm_client.chat_completion_simple(
                prompt=user_message,
                system_prompt=sys_prompt,
            )

            # 第5步：追加本轮问答到对话历史
            self._chat_history.append(ChatMessage(role="user", content=message))
            self._chat_history.append(ChatMessage(role="assistant", content=answer))

            # 收集来源
            sources = self._collect_sources(retrieval_result)

            elapsed_ms = (time.time() - start_time) * 1000

            return GenerationResult(
                answer=answer,
                query=message,
                sources=sources,
                context_used=context,
                retrieval_result=retrieval_result,
                elapsed_ms=elapsed_ms,
                model=self._llm_client.model,
            )

        except Exception as e:
            raise ModelAPIError(f"多轮对话生成失败: {str(e)}") from e

    def reset_chat(self) -> None:
        """重置多轮对话状态

        清空对话历史和压缩摘要，开始一轮新的对话。
        """
        self._chat_history = []
        self._chat_history_summary = ""
        print("[QueryEngineAdapter] 多轮对话已重置")

    def _condense_history(self) -> str:
        """压缩对话历史

        当原始对话历史超过阈值时，用 LLM 将旧消息压缩为摘要，
        只保留最近几轮原始消息。这样可以控制 Prompt 长度，
        同时保留对话的关键上下文。

        压缩策略：
        - 原始历史 ≤ 阈值：直接拼接所有历史，不压缩
        - 原始历史 > 阈值：将前半部分压缩为摘要，保留后半部分原始消息

        返回:
            包含历史上下文的字符串（摘要 + 最近消息）
        """
        if len(self._chat_history) <= self._chat_history_compress_threshold:
            # 历史较短，直接拼接全部
            return self._format_history(self._chat_history)

        # 历史过长：压缩前半部分
        mid = len(self._chat_history) // 2
        old_messages = self._chat_history[:mid]
        recent_messages = self._chat_history[mid:]

        # 如果还没有摘要，先压缩旧消息生成摘要
        if not self._chat_history_summary:
            old_text = self._format_history(old_messages)
            self._chat_history_summary = self._llm_client.chat_completion_simple(
                prompt=(
                    f"请将以下对话历史压缩为简洁的摘要，保留关键信息和上下文：\n\n"
                    f"{old_text}"
                ),
                system_prompt="你是一个对话摘要助手，请用简洁的语言总结对话内容。",
            )
            print(
                f"[ChatEngine] 对话历史已压缩: "
                f"{len(old_messages)} 条消息 → {len(self._chat_history_summary)} 字符摘要"
            )

        # 拼接摘要 + 最近消息
        parts = []
        if self._chat_history_summary:
            parts.append(f"[Previous conversation summary]: {self._chat_history_summary}")
        recent = self._format_history(recent_messages)
        if recent:
            parts.append(f"[Recent messages]:\n{recent}")

        return "\n\n".join(parts)

    @staticmethod
    def _format_history(messages: List[ChatMessage]) -> str:
        """将对话消息列表格式化为字符串"""
        if not messages:
            return ""
        lines = []
        for msg in messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)

    def _build_chat_prompt(self, query: str, context: str, history: str) -> str:
        """构建多轮对话的用户 Prompt

        将对话历史摘要、RAG 检索上下文、当前问题组合为完整的 Prompt。
        """
        parts = []

        # 对话历史（摘要 + 最近消息）
        if history:
            parts.append(f"Conversation history:\n{history}")

        # RAG 检索上下文
        if context:
            parts.append(f"Reference materials:\n{context}")

        # 当前问题
        parts.append(f"Question: {query}")

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #                    LCEL Chain 构建                                     #
    # ------------------------------------------------------------------ #

    def _build_lcel_chain(self):
        """构建 LCEL 链式调用管道

        使用 LangChain LCEL (LangChain Expression Language) 将 RAG 流程
        串联为声明式管道：检索 → 上下文构建 → Prompt 填充 → LLM 生成 → 解析

        LCEL 核心组件：
        - RunnablePassthrough: 透传输入数据到下一步
        - RunnableLambda: 将普通函数包装为 LCEL 可执行单元
        - ChatPromptTemplate: 声明式 Prompt 模板，通过变量自动填充
        - StrOutputParser: 将 LLM 输出解析为纯字符串
        - | 管道符: 将多个 Runnable 串联为 Chain

        返回:
            可 invoke() 的 LCEL Chain
        """
        # LLM（从 LLMAdapter 中提取底层 ChatOpenAI 实例）
        llm = self._llm_client._llm

        # Prompt 模板：使用 ChatPromptTemplate 声明式定义系统/用户消息
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{user_message}"),
        ])

        # LCEL 管道：query → retrieve → build_context → format_prompt → LLM → parse
        chain = (
            RunnablePassthrough.assign(
                # 检索：query → RetrievalResult，同时缓存中间结果
                retrieval_result=RunnableLambda(self._lcel_retrieve),
            )
            .assign(
                # 上下文构建：RetrievalResult → context 字符串
                context=RunnableLambda(self._lcel_build_context),
            )
            .assign(
                # Prompt 填充：query + context → user_message
                user_message=RunnableLambda(self._lcel_format_user_message),
            )
            # 提取 LLM 所需的变量子集
            | RunnableLambda(self._lcel_extract_prompt_inputs)
            # Prompt 模板填充 → LLM 生成 → 字符串解析
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain

    # ------------------------------------------------------------------ #
    #                    LCEL 管道步骤函数                                    #
    # ------------------------------------------------------------------ #

    def _lcel_retrieve(self, inputs: Dict[str, Any]) -> RetrievalResult:
        """LCEL 步骤：执行检索并缓存结果"""
        query = inputs["query"]
        result = self._retriever_adapter.retrieve(query=query)
        # 缓存中间结果，供 generate() 组装 GenerationResult 元数据
        self._last_retrieval_result = result
        return result

    def _lcel_build_context(self, inputs: Dict[str, Any]) -> str:
        """LCEL 步骤：从检索结果构建上下文字符串"""
        retrieval_result = inputs.get("retrieval_result")
        if retrieval_result is None:
            return ""
        context = self._build_context(retrieval_result)
        # 缓存上下文，供 generate() 使用
        self._last_context = context
        return context

    def _lcel_format_user_message(self, inputs: Dict[str, Any]) -> str:
        """LCEL 步骤：将 query + context 组合为用户消息"""
        query = inputs["query"]
        context = inputs.get("context", "")
        return self._build_user_prompt(query, context)

    @staticmethod
    def _lcel_extract_prompt_inputs(inputs: Dict[str, Any]) -> Dict[str, str]:
        """LCEL 步骤：提取 Prompt 模板所需的变量子集

        ChatPromptTemplate 只需要 system_prompt 和 user_message，
        此步骤过滤掉其他中间变量（retrieval_result、context、query 等）
        """
        return {
            "system_prompt": inputs.get("system_prompt", ""),
            "user_message": inputs.get("user_message", ""),
        }

    # ------------------------------------------------------------------ #
    #                    内部辅助方法                                       #
    # ------------------------------------------------------------------ #

    def _build_context(self, retrieval_result: RetrievalResult) -> str:
        """从检索结果构建上下文字符串"""
        if retrieval_result.is_empty:
            return ""

        context_parts = []
        total_chars = 0

        for item in retrieval_result.items:
            if total_chars + len(item.text) > self._max_context_chars:
                break
            context_parts.append(item.text)
            total_chars += len(item.text)

        return "\n\n---\n\n".join(context_parts)

    def _build_user_prompt(self, query: str, context: str) -> str:
        """构建带 context 的用户提示词"""
        if context:
            return (
                f"Based on the following reference materials:\n\n"
                f"{context}\n\n"
                f"---\n\n"
                f"Please answer the following question: {query}"
            )
        else:
            return query

    def _collect_sources(self, retrieval_result: RetrievalResult) -> List[str]:
        """从检索结果中收集唯一的来源路径"""
        if not self._include_sources:
            return []

        sources = []
        seen = set()
        for item in retrieval_result.items:
            if item.source and item.source not in seen:
                seen.add(item.source)
                sources.append(item.source)

        return sources


# ------------------------------------------------------------------ #
#                     全局单例 & 辅助函数                                #
# ------------------------------------------------------------------ #

_generator_instance: Optional[QueryEngineAdapter] = None


def get_generator() -> QueryEngineAdapter:
    """获取全局 QueryEngine 适配器单例（懒加载）

    注意：函数名保留 get_generator 以保持向后兼容。

    返回:
        QueryEngineAdapter 单例
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = QueryEngineAdapter()
    return _generator_instance


def reload_generator() -> QueryEngineAdapter:
    """强制重新创建 QueryEngine 适配器

    返回:
        新的 QueryEngineAdapter 实例
    """
    global _generator_instance
    _generator_instance = QueryEngineAdapter()
    return _generator_instance
