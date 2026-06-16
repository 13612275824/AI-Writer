# ============================================================================
# 文章写作模块  |  src/writing/article_writer.py
# ============================================================================
#
# 功能：根据用户提供的主题，生成结构完整的文章
#
# 输入参数：
#     topic        : 文章主题（必填）
#     style        : 写作风格（可选，如：学术/通俗/正式/幽默）
#     requirements : 额外写作要求（可选）
#     outline      : 预生成的大纲（可选，None 时自动生成）
#     word_count   : 目标字数（可选）
#
# 输出格式：
#     ArticleResult 数据结构，包含文章正文、大纲、token 统计等
#
# 配置与依赖：
#     Prompt 模板：configs/prompts.yaml → writing 节点
#     依赖模块：src/core/models.py、src/core/prompts.py
#
# 使用方式：
#   from src.writing.article_writer import ArticleWriter, get_article_writer
#
#   writer = get_article_writer()
#   result = writer.write(topic="人工智能的发展")
#   print(result.article)
# ============================================================================

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.exceptions import ModelAPIError
from src.core.models import LLMClient, get_llm_client
from src.core.prompts import get_prompt
from src.utils.text_utils import debug_print


# ------------------------------------------------------------------ #
#                    数据结构                                         #
# ------------------------------------------------------------------ #

@dataclass
class ArticleResult:
    """文章写作结果

    Attributes:
        article: 生成的文章正文
        outline: 文章大纲（可选）
        topic: 原始主题
        style: 写作风格
        elapsed_ms: 生成耗时（毫秒）
        model: 使用的模型名称
        prompt_tokens: 提示词 token 数
        completion_tokens: 生成 token 数

    示例：
        >>> result = writer.write(topic="人工智能的发展")
        >>> print(result.article)
        >>> print(f"耗时: {result.elapsed_ms:.1f}ms")
    """
    article: str = ""
    outline: str = ""
    topic: str = ""
    style: str = ""
    elapsed_ms: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """返回总 token 数"""
        return self.prompt_tokens + self.completion_tokens

    def __repr__(self) -> str:
        article_preview = (
            self.article[:60] +
            "..." if len(self.article) > 60 else self.article
        )
        return (
            f"<ArticleResult topic={self.topic!r} "
            f"article_len={len(self.article)} "
            f"elapsed={self.elapsed_ms:.1f}ms>"
        )


# ------------------------------------------------------------------ #
#                    文章写作器                                       #
# ------------------------------------------------------------------ #

class ArticleWriter:
    """文章写作器 — 根据主题生成结构完整的文章

    设计原则：
    - 组合模式：持有 LLMClient 实例
    - Prompt 驱动：使用 prompts.yaml 中 writing 角色的模板
    - 防御性编程：参数校验 + 异常统一处理

    示例：
        >>> writer = ArticleWriter()
        >>> result = writer.write(topic="人工智能的发展")
        >>> print(result.article)
    """

    PROMPT_ROLE = "writing"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """初始化文章写作器

        Args:
            llm_client: LLMClient 实例（None 时使用全局单例）
            temperature: 生成温度（None 时使用 LLMClient 默认值）
            max_tokens: 最大生成 token 数（None 时使用 LLMClient 默认值）
        """
        self._llm = llm_client or get_llm_client()
        self._temperature = temperature
        self._max_tokens = max_tokens

        debug_print(
            f"ArticleWriter 初始化完成: model={self._llm.model}"
        )

    def write(
        self,
        topic: str,
        style: str = "",
        requirements: str = "",
        outline: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ArticleResult:
        """生成文章

        Args:
            topic: 文章主题（必填）
            style: 写作风格
            requirements: 额外写作要求
            outline: 预生成的大纲（None 时不使用）
            temperature: 生成温度
            max_tokens: 最大生成 token 数

        Returns:
            ArticleResult

        Raises:
            ValueError: topic 为空时抛出
            ModelAPIError: LLM 调用失败时抛出
        """
        if not topic or not topic.strip():
            raise ValueError("文章主题不能为空")

        start_time = time.time()

        messages = self._build_messages(
            topic=topic,
            style=style,
            requirements=requirements,
            outline=outline,
        )

        debug_print(
            f"开始生成文章: topic={topic!r}, style={style!r}"
        )

        tmp = temperature if temperature is not None else self._temperature
        mtk = max_tokens if max_tokens is not None else self._max_tokens

        article, prompt_tokens, completion_tokens = self._call_llm(
            messages=messages,
            temperature=tmp,
            max_tokens=mtk,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        result = ArticleResult(
            article=article,
            outline=outline or "",
            topic=topic,
            style=style,
            elapsed_ms=elapsed_ms,
            model=self._llm.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        debug_print(
            f"文章生成完成: article_len={len(article)}, "
            f"tokens={result.total_tokens}, elapsed={elapsed_ms:.1f}ms"
        )
        return result

    def _build_messages(
        self,
        topic: str,
        style: str,
        requirements: str,
        outline: Optional[str],
    ) -> List[Dict[str, str]]:
        """构建发送给 LLM 的消息列表

        Args:
            topic: 文章主题
            style: 写作风格
            requirements: 额外要求
            outline: 预生成大纲

        Returns:
            messages 列表
        """
        style_text = style or "无特殊要求"
        requirements_text = requirements or "无额外要求"
        outline_text = outline or "无预定义大纲，请自行规划结构"

        # 直接使用 system + 自定义 user 消息，不依赖 prompts.yaml 的 user_template
        system_prompt = self._get_system_prompt()
        user_content = self._build_user_content(
            topic=topic,
            style=style_text,
            requirements=requirements_text,
            outline=outline_text,
        )

        messages = [{"role": "system", "content": system_prompt}]
        if user_content:
            messages.append({"role": "user", "content": user_content})
        return messages

    def _get_system_prompt(self) -> str:
        """从 prompts.yaml 获取系统提示词

        Returns:
            系统提示词文本
        """
        try:
            prompt_config = get_prompt(self.PROMPT_ROLE)
            return prompt_config.get("system", "")
        except Exception:
            return "你是一个专业的写作助手，擅长根据用户需求生成高质量的文章。"

    def _build_user_content(
        self,
        topic: str,
        style: str,
        requirements: str,
        outline: str,
    ) -> str:
        """构建发送给 LLM 的用户消息内容

        Args:
            topic: 文章主题
            style: 写作风格
            requirements: 额外要求
            outline: 预生成大纲

        Returns:
            完整的用户消息文本
        """
        parts = [f"请写一篇关于 {topic} 的文章。"]
        parts.append(f"\n写作风格：{style}")
        parts.append(f"大纲：{outline}")
        parts.append(f"额外要求：{requirements}")
        return "\n".join(parts)

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple:
        """调用 LLM 并提取回答

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            (article_text, prompt_tokens, completion_tokens) 元组

        Raises:
            ModelAPIError: API 调用失败时抛出
        """
        kwargs: Dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._llm.chat_completion(messages, **kwargs)

        article_text = response.choices[0].message.content or ""

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = (
                getattr(response.usage, "completion_tokens", 0) or 0
            )

        return article_text, prompt_tokens, completion_tokens

    @property
    def llm_client(self) -> LLMClient:
        """当前使用的 LLMClient 实例"""
        return self._llm

    def __repr__(self) -> str:
        return f"<ArticleWriter model={self._llm.model!r}>"


# ------------------------------------------------------------------ #
#                    全局单例 & 便捷函数                               #
# ------------------------------------------------------------------ #

_article_writer_instance: Optional[ArticleWriter] = None


def get_article_writer() -> ArticleWriter:
    """获取全局 ArticleWriter 单例（懒加载）

    首次调用时会创建实例，后续调用直接返回同一实例。

    Returns:
        ArticleWriter 单例对象

    示例：
        >>> writer = get_article_writer()
        >>> result = writer.write(topic="人工智能的发展")
        >>> print(result.article)
    """
    global _article_writer_instance
    if _article_writer_instance is None:
        _article_writer_instance = ArticleWriter()
    return _article_writer_instance
