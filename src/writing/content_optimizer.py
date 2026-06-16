# ============================================================================
# 内容优化模块  |  src/writing/content_optimizer.py
# ============================================================================
#
# 功能：对已有文本进行润色、优化和改进
#
# 输入参数：
#     content       : 待优化的原始文本（必填）
#     optimize_type : 优化类型（可选：polish/simplify/expand/shorten/grammar）
#                     - polish  : 润色（提升表达质量，保持原意）
#                     - simplify: 简化（降低复杂度，更易理解）
#                     - expand  : 扩写（补充细节，丰富内容）
#                     - shorten : 缩写（精简内容，保留要点）
#                     - grammar : 语法校对（修正语法和错别字）
#     target_style  : 目标风格（可选：formal/casual/academic/professional）
#     focus_areas   : 重点关注方向（可选，逗号分隔或列表）
#     requirements  : 额外要求（可选）
#
# 输出格式：
#     OptimizationResult 数据结构，包含优化后内容、优化摘要、token 统计等
#
# 配置与依赖：
#     Prompt 模板：configs/prompts.yaml → editing 节点
#     依赖模块：src/core/models.py、src/core/prompts.py
#
# 使用方式：
#   from src.writing.content_optimizer import ContentOptimizer, get_content_optimizer
#
#   optimizer = get_content_optimizer()
#   result = optimizer.optimize(
#       content="原始文本内容...",
#       optimize_type="polish",
#       target_style="formal",
#   )
#   print(result.optimized_content)
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
class OptimizationResult:
    """内容优化结果

    Attributes:
        optimized_content: 优化后的内容
        original_content: 原始内容
        optimize_type: 优化类型
        target_style: 目标风格
        summary: 优化摘要（修改说明）
        char_diff: 字符数变化（优化后 - 优化前，正数表示增长）
        elapsed_ms: 处理耗时（毫秒）
        model: 使用的模型名称
        prompt_tokens: 提示词 token 数
        completion_tokens: 生成 token 数

    示例：
        >>> result = optimizer.optimize(content="原始文本")
        >>> print(result.optimized_content)
        >>> print(f"耗时: {result.elapsed_ms:.1f}ms")
    """
    optimized_content: str = ""
    original_content: str = ""
    optimize_type: str = ""
    target_style: str = ""
    summary: str = ""
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
    def improvement_ratio(self) -> float:
        """字符数变化比率（相对于原文长度）

        Returns:
            比率值，正数表示内容增长，负数表示精简
        """
        if not self.original_content:
            return 0.0
        return self.char_diff / len(self.original_content)

    def __repr__(self) -> str:
        preview = (
            self.optimized_content[:60] + "..."
            if len(self.optimized_content) > 60
            else self.optimized_content
        )
        return (
            f"<OptimizationResult type={self.optimize_type!r} "
            f"content_len={len(self.optimized_content)} "
            f"diff={self.char_diff:+d} "
            f"elapsed={self.elapsed_ms:.1f}ms>"
        )


# ------------------------------------------------------------------ #
#                    内容优化器                                       #
# ------------------------------------------------------------------ #

# 优化类型中英文映射
_OPTIMIZE_TYPE_NAMES = {
    "polish": "润色优化",
    "simplify": "简化改写",
    "expand": "扩写丰富",
    "shorten": "精简缩写",
    "grammar": "语法校对",
}

# 目标风格中英文映射
_TARGET_STYLE_NAMES = {
    "formal": "正式书面",
    "casual": "轻松口语",
    "academic": "学术论文",
    "professional": "专业商务",
    "creative": "创意文艺",
}


class ContentOptimizer:
    """内容优化器 — 对已有文本进行润色、优化和改进

    设计原则：
    - 组合模式：持有 LLMClient 实例
    - Prompt 驱动：使用 prompts.yaml 中 editing 角色的模板
    - 防御性编程：参数校验 + 异常统一处理

    示例：
        >>> optimizer = ContentOptimizer()
        >>> result = optimizer.optimize(
        ...     content="原始文本内容",
        ...     optimize_type="polish",
        ... )
        >>> print(result.optimized_content)
    """

    PROMPT_ROLE = "editing"

    # 支持的优化类型
    SUPPORTED_TYPES = frozenset(
        ["polish", "simplify", "expand", "shorten", "grammar"]
    )

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """初始化内容优化器

        Args:
            llm_client: LLMClient 实例（None 时使用全局单例）
            temperature: 生成温度（None 时使用 LLMClient 默认值）
            max_tokens: 最大生成 token 数（None 时使用 LLMClient 默认值）
        """
        self._llm = llm_client or get_llm_client()
        self._temperature = temperature
        self._max_tokens = max_tokens

        debug_print(
            f"ContentOptimizer 初始化完成: model={self._llm.model}"
        )

    def optimize(
        self,
        content: str,
        optimize_type: str = "polish",
        target_style: str = "",
        focus_areas: str = "",
        requirements: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> OptimizationResult:
        """对文本内容进行优化

        Args:
            content: 待优化的原始文本（必填）
            optimize_type: 优化类型（默认 polish）
            target_style: 目标风格（可选）
            focus_areas: 重点关注方向（可选，如"语法,表达,结构"）
            requirements: 额外优化要求（可选）
            temperature: 生成温度（None 时使用默认值）
            max_tokens: 最大生成 token 数（None 时使用默认值）

        Returns:
            OptimizationResult 包含优化后内容及元信息

        Raises:
            ValueError: content 为空或 optimize_type 不支持时抛出
            ModelAPIError: LLM 调用失败时抛出

        示例：
            >>> result = optimizer.optimize(
            ...     content="这是一段需要优化的文本",
            ...     optimize_type="polish",
            ...     target_style="formal",
            ... )
        """
        # ── 参数校验 ──
        if not content or not content.strip():
            raise ValueError("待优化内容不能为空")

        if optimize_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"不支持的优化类型: {optimize_type!r}，"
                f"支持的类型: {sorted(self.SUPPORTED_TYPES)}"
            )

        start_time = time.time()

        messages = self._build_messages(
            content=content,
            optimize_type=optimize_type,
            target_style=target_style,
            focus_areas=focus_areas,
            requirements=requirements,
        )

        debug_print(
            f"开始优化内容: type={optimize_type!r}, "
            f"style={target_style!r}, content_len={len(content)}"
        )

        tmp = temperature if temperature is not None else self._temperature
        mtk = max_tokens if max_tokens is not None else self._max_tokens

        optimized, prompt_tokens, completion_tokens = self._call_llm(
            messages=messages,
            temperature=tmp,
            max_tokens=mtk,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        char_diff = len(optimized) - len(content)

        result = OptimizationResult(
            optimized_content=optimized,
            original_content=content,
            optimize_type=optimize_type,
            target_style=target_style,
            summary=self._build_summary(
                optimize_type, target_style, char_diff, len(content)
            ),
            char_diff=char_diff,
            elapsed_ms=elapsed_ms,
            model=self._llm.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        debug_print(
            f"优化完成: optimized_len={len(optimized)}, "
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
        optimize_type: str,
        target_style: str,
        focus_areas: str,
        requirements: str,
    ) -> List[Dict[str, str]]:
        """构建发送给 LLM 的消息列表

        Args:
            content: 待优化内容
            optimize_type: 优化类型
            target_style: 目标风格
            focus_areas: 重点关注方向
            requirements: 额外要求

        Returns:
            messages 列表
        """
        type_text = _OPTIMIZE_TYPE_NAMES.get(optimize_type, optimize_type)
        style_text = _TARGET_STYLE_NAMES.get(target_style, target_style)
        focus_text = focus_areas or "无特殊要求"
        req_text = requirements or "无额外要求"

        system_prompt = self._get_system_prompt(optimize_type)
        user_content = self._build_user_content(
            content=content,
            optimize_type=type_text,
            target_style=style_text,
            focus_areas=focus_text,
            requirements=req_text,
        )

        messages = [{"role": "system", "content": system_prompt}]
        if user_content:
            messages.append({"role": "user", "content": user_content})
        return messages

    def _get_system_prompt(self, optimize_type: str = "polish") -> str:
        """从 prompts.yaml 获取系统提示词，并按优化类型增强

        Args:
            optimize_type: 优化类型，用于生成更精准的系统提示词

        Returns:
            系统提示词文本
        """
        # 优先使用 prompts.yaml 中 editing 角色的 system prompt
        try:
            prompt_config = get_prompt(self.PROMPT_ROLE)
            base_prompt = prompt_config.get("system", "")
        except Exception:
            base_prompt = "你是一个专业的内容编辑，擅长对文本进行润色、校对和优化。"

        # 根据优化类型追加指令
        type_instructions = {
            "polish": "你的任务是润色优化：提升语言表达质量、增强可读性，保持原意不变。",
            "simplify": "你的任务是简化改写：用更简洁易懂的表达方式重写内容，降低阅读门槛。",
            "expand": "你的任务是扩写丰富：在保持主题不变的前提下，补充细节、添加例证、丰富内容。",
            "shorten": "你的任务是精简缩写：去除冗余内容，精简表达，只保留核心要点。",
            "grammar": "你的任务是语法校对：修正错别字、语法错误、标点问题，不改变原文内容和风格。",
        }

        extra = type_instructions.get(optimize_type, "")
        if extra:
            return f"{base_prompt}\n\n{extra}"
        return base_prompt

    def _build_user_content(
        self,
        content: str,
        optimize_type: str,
        target_style: str,
        focus_areas: str,
        requirements: str,
    ) -> str:
        """构建发送给 LLM 的用户消息内容

        Args:
            content: 待优化内容
            optimize_type: 优化类型（中文）
            target_style: 目标风格（中文）
            focus_areas: 重点关注方向
            requirements: 额外要求

        Returns:
            完整的用户消息文本
        """
        parts = [f"请对以下文本进行{optimize_type}。"]
        if target_style and target_style != "无特殊要求":
            parts.append(f"\n目标风格：{target_style}")
        parts.append(f"\n重点关注：{focus_areas}")
        parts.append(f"额外要求：{requirements}")
        parts.append(f"\n待优化内容：\n{content}")
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
            (optimized_text, prompt_tokens, completion_tokens) 元组

        Raises:
            ModelAPIError: API 调用失败时抛出
        """
        kwargs: Dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._llm.chat_completion(messages, **kwargs)

        optimized_text = response.choices[0].message.content or ""

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = (
                getattr(response.usage, "completion_tokens", 0) or 0
            )

        return optimized_text, prompt_tokens, completion_tokens

    @staticmethod
    def _build_summary(
        optimize_type: str,
        target_style: str,
        char_diff: int,
        original_len: int,
    ) -> str:
        """构建优化摘要说明

        Args:
            optimize_type: 优化类型
            target_style: 目标风格
            char_diff: 字符数变化
            original_len: 原始文本长度

        Returns:
            摘要文本
        """
        type_text = _OPTIMIZE_TYPE_NAMES.get(optimize_type, optimize_type)
        parts = [f"优化类型: {type_text}"]
        if target_style:
            style_text = _TARGET_STYLE_NAMES.get(target_style, target_style)
            parts.append(f"目标风格: {style_text}")
        parts.append(f"字符变化: {char_diff:+d}")
        if original_len > 0:
            ratio = char_diff / original_len * 100
            parts.append(f"变化比例: {ratio:+.1f}%")
        return " | ".join(parts)

    @property
    def llm_client(self) -> LLMClient:
        """当前使用的 LLMClient 实例"""
        return self._llm

    def __repr__(self) -> str:
        return f"<ContentOptimizer model={self._llm.model!r}>"


# ------------------------------------------------------------------ #
#                    全局单例 & 便捷函数                               #
# ------------------------------------------------------------------ #

_content_optimizer_instance: Optional[ContentOptimizer] = None


def get_content_optimizer() -> ContentOptimizer:
    """获取全局 ContentOptimizer 单例（懒加载）

    首次调用时会创建实例，后续调用直接返回同一实例。

    Returns:
        ContentOptimizer 单例对象

    示例：
        >>> optimizer = get_content_optimizer()
        >>> result = optimizer.optimize(content="原始文本", optimize_type="polish")
        >>> print(result.optimized_content)
    """
    global _content_optimizer_instance
    if _content_optimizer_instance is None:
        _content_optimizer_instance = ContentOptimizer()
    return _content_optimizer_instance
