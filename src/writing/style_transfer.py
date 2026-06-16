# ============================================================================
# 风格转换模块  |  src/writing/style_transfer.py
# ============================================================================
#
# 功能：将文本从一种风格完整转换为另一种风格，保持原意不变
#
# 与 content_optimizer.py 的区别：
#     - content_optimizer: 在原风格基础上提升质量（润色/简化/扩写等）
#     - style_transfer:    彻底换一种风格重写（口语→学术、正式→通俗等）
#
# 输入参数：
#     content       : 待转换的原始文本（必填）
#     target_style  : 目标风格（必填：formal/casual/academic/professional/creative）
#     source_style  : 源风格（可选，None 时自动检测）
#     requirements  : 额外要求（可选）
#
# 输出格式：
#     StyleTransferResult 数据结构，包含转换后内容、token 统计等
#
# 配置与依赖：
#     Prompt 模板：configs/prompts.yaml → style_transfer 节点
#     依赖模块：src/core/models.py、src/core/prompts.py
#
# 使用方式：
#   from src.writing.style_transfer import StyleTransfer, get_style_transfer
#
#   transfer = get_style_transfer()
#   result = transfer.transfer(
#       content="这是一段口语化的文本",
#       target_style="academic",
#   )
#   print(result.transferred_content)
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
class StyleTransferResult:
    """风格转换结果

    Attributes:
        transferred_content: 转换后的内容
        original_content: 原始内容
        target_style: 目标风格
        source_style: 源风格
        char_diff: 字符数变化（转换后 - 转换前）
        elapsed_ms: 处理耗时（毫秒）
        model: 使用的模型名称
        prompt_tokens: 提示词 token 数
        completion_tokens: 生成 token 数

    示例：
        >>> result = transfer.transfer(content="口语文本", target_style="formal")
        >>> print(result.transferred_content)
        >>> print(f"耗时: {result.elapsed_ms:.1f}ms")
    """
    transferred_content: str = ""
    original_content: str = ""
    target_style: str = ""
    source_style: str = ""
    char_diff: int = 0
    elapsed_ms: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """返回总 token 数"""
        return self.prompt_tokens + self.completion_tokens

    @property
    def length_ratio(self) -> float:
        """转换后与原文的长度比率

        Returns:
            比率值，>1 表示转换后更长，<1 表示更短
        """
        if not self.original_content:
            return 0.0
        return len(self.transferred_content) / len(self.original_content)

    def __repr__(self) -> str:
        preview = (
            self.transferred_content[:60] + "..."
            if len(self.transferred_content) > 60
            else self.transferred_content
        )
        return (
            f"<StyleTransferResult target={self.target_style!r} "
            f"content_len={len(self.transferred_content)} "
            f"diff={self.char_diff:+d} "
            f"elapsed={self.elapsed_ms:.1f}ms>"
        )


# ------------------------------------------------------------------ #
#                    风格转换器                                       #
# ------------------------------------------------------------------ #

# 目标风格中英文映射
_STYLE_NAMES = {
    "formal": "正式书面",
    "casual": "轻松口语",
    "academic": "学术论文",
    "professional": "专业商务",
    "creative": "创意文艺",
    "news": "新闻报道",
    "storytelling": "叙事故事",
}


class StyleTransfer:
    """风格转换器 — 将文本从一种风格完整转换为另一种风格

    设计原则：
    - 组合模式：持有 LLMClient 实例
    - Prompt 驱动：使用 prompts.yaml 中 style_transfer 角色的模板
    - 防御性编程：参数校验 + 异常统一处理

    示例：
        >>> transfer = StyleTransfer()
        >>> result = transfer.transfer(
        ...     content="这个产品真的很好用啊！",
        ...     target_style="academic",
        ... )
        >>> print(result.transferred_content)
    """

    PROMPT_ROLE = "style_transfer"

    # 支持的转换风格
    SUPPORTED_STYLES = frozenset(
        ["formal", "casual", "academic", "professional",
            "creative", "news", "storytelling"]
    )

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """初始化风格转换器

        Args:
            llm_client: LLMClient 实例（None 时使用全局单例）
            temperature: 生成温度（None 时使用 LLMClient 默认值）
            max_tokens: 最大生成 token 数（None 时使用 LLMClient 默认值）
        """
        self._llm = llm_client or get_llm_client()
        self._temperature = temperature
        self._max_tokens = max_tokens

        debug_print(
            f"StyleTransfer 初始化完成: model={self._llm.model}"
        )

    def transfer(
        self,
        content: str,
        target_style: str,
        source_style: str = "",
        requirements: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> StyleTransferResult:
        """将文本转换为指定风格

        Args:
            content: 待转换的原始文本（必填）
            target_style: 目标风格（必填）
            source_style: 源风格（可选，为空时自动识别）
            requirements: 额外转换要求（可选）
            temperature: 生成温度（None 时使用默认值）
            max_tokens: 最大生成 token 数（None 时使用默认值）

        Returns:
            StyleTransferResult 包含转换后内容及元信息

        Raises:
            ValueError: content 为空或 target_style 不支持时抛出
            ModelAPIError: LLM 调用失败时抛出

        示例：
            >>> result = transfer.transfer(
            ...     content="口语化的文本内容",
            ...     target_style="formal",
            ... )
        """
        # ── 参数校验 ──
        if not content or not content.strip():
            raise ValueError("待转换内容不能为空")

        if target_style not in self.SUPPORTED_STYLES:
            raise ValueError(
                f"不支持的目标风格: {target_style!r}，"
                f"支持的类型: {sorted(self.SUPPORTED_STYLES)}"
            )

        start_time = time.time()

        messages = self._build_messages(
            content=content,
            target_style=target_style,
            source_style=source_style,
            requirements=requirements,
        )

        debug_print(
            f"开始风格转换: target={target_style!r}, "
            f"source={source_style!r}, content_len={len(content)}"
        )

        tmp = temperature if temperature is not None else self._temperature
        mtk = max_tokens if max_tokens is not None else self._max_tokens

        transferred, prompt_tokens, completion_tokens = self._call_llm(
            messages=messages,
            temperature=tmp,
            max_tokens=mtk,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        char_diff = len(transferred) - len(content)

        result = StyleTransferResult(
            transferred_content=transferred,
            original_content=content,
            target_style=target_style,
            source_style=source_style or "auto",
            char_diff=char_diff,
            elapsed_ms=elapsed_ms,
            model=self._llm.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        debug_print(
            f"风格转换完成: transferred_len={len(transferred)}, "
            f"diff={char_diff:+d}, tokens={result.total_tokens}, "
            f"elapsed={elapsed_ms:.1f}ms"
        )
        return result

    # ------------------------------------------------------------------ #
    #                    内部辅助方法                                     #
    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        content: str,
        target_style: str,
        source_style: str,
        requirements: str,
    ) -> List[Dict[str, str]]:
        """构建发送给 LLM 的消息列表

        Args:
            content: 待转换内容
            target_style: 目标风格
            source_style: 源风格
            requirements: 额外要求

        Returns:
            messages 列表
        """
        target_text = _STYLE_NAMES.get(target_style, target_style)
        source_text = (
            _STYLE_NAMES.get(
                source_style, source_style) if source_style else "自动识别"
        )
        req_text = requirements or "无额外要求"

        system_prompt = self._get_system_prompt()
        user_content = self._build_user_content(
            content=content,
            target_style=target_text,
            source_style=source_text,
            requirements=req_text,
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
            return (
                "你是一个专业的风格转换专家，擅长将文本从一种写作风格完整转换为另一种风格。"
                "在保持原文核心意思不变的前提下，将内容彻底转换为指定风格。"
            )

    def _build_user_content(
        self,
        content: str,
        target_style: str,
        source_style: str,
        requirements: str,
    ) -> str:
        """构建发送给 LLM 的用户消息内容

        Args:
            content: 待转换内容
            target_style: 目标风格（中文）
            source_style: 源风格（中文）
            requirements: 额外要求

        Returns:
            完整的用户消息文本
        """
        parts = [
            f"请将以下文本转换为【{target_style}】风格。",
            f"\n源风格：{source_style}",
            f"额外要求：{requirements}",
            f"\n待转换内容：\n{content}",
        ]
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
            (transferred_text, prompt_tokens, completion_tokens) 元组

        Raises:
            ModelAPIError: API 调用失败时抛出
        """
        kwargs: Dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._llm.chat_completion(messages, **kwargs)

        transferred_text = response.choices[0].message.content or ""

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = (
                getattr(response.usage, "completion_tokens", 0) or 0
            )

        return transferred_text, prompt_tokens, completion_tokens

    @property
    def llm_client(self) -> LLMClient:
        """当前使用的 LLMClient 实例"""
        return self._llm

    def __repr__(self) -> str:
        return f"<StyleTransfer model={self._llm.model!r}>"


# ------------------------------------------------------------------ #
#                    全局单例 & 便捷函数                               #
# ------------------------------------------------------------------ #

_style_transfer_instance: Optional[StyleTransfer] = None


def get_style_transfer() -> StyleTransfer:
    """获取全局 StyleTransfer 单例（懒加载）

    首次调用时会创建实例，后续调用直接返回同一实例。

    Returns:
        StyleTransfer 单例对象

    示例：
        >>> transfer = get_style_transfer()
        >>> result = transfer.transfer(content="口语文本", target_style="formal")
        >>> print(result.transferred_content)
    """
    global _style_transfer_instance
    if _style_transfer_instance is None:
        _style_transfer_instance = StyleTransfer()
    return _style_transfer_instance
