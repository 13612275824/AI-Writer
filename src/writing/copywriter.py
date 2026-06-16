# ============================================================================
# 文案写作模块  |  src/writing/copywriter.py
# ============================================================================
#
# 功能：根据产品/品牌信息，生成各类营销文案
#
# 输入参数：
#     product_name      : 产品/品牌名称（必填）
#     copy_type         : 文案类型（可选：slogan/product_description/ad_copy/
#                         social_media/landing_page/email）
#     target_audience   : 目标受众（可选）
#     brand_tone        : 品牌调性（可选：professional/casual/luxurious/
#                         playful/urgent/inspirational）
#     key_selling_points: 核心卖点（可选）
#     requirements      : 额外要求（可选）
#
# 输出格式：
#     CopywriterResult 数据结构，包含文案正文、token 统计等
#
# 配置与依赖：
#     Prompt 模板：configs/prompts.yaml → copywriting 节点
#     依赖模块：src/core/models.py、src/core/prompts.py
#
# 使用方式：
#   from src.writing.copywriter import Copywriter, get_copywriter
#
#   writer = get_copywriter()
#   result = writer.write(
#       product_name="智能手表",
#       copy_type="slogan",
#       target_audience="年轻运动爱好者",
#   )
#   print(result.copy)
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
class CopywriterResult:
    """文案写作结果

    Attributes:
        copy: 生成的文案正文
        copy_type: 文案类型
        product_name: 产品/品牌名称
        target_audience: 目标受众
        brand_tone: 品牌调性
        elapsed_ms: 生成耗时（毫秒）
        model: 使用的模型名称
        prompt_tokens: 提示词 token 数
        completion_tokens: 生成 token 数

    示例：
        >>> result = writer.write(product_name="智能手表", copy_type="slogan")
        >>> print(result.copy)
    """
    copy: str = ""
    copy_type: str = ""
    product_name: str = ""
    target_audience: str = ""
    brand_tone: str = ""
    elapsed_ms: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """返回总 token 数"""
        return self.prompt_tokens + self.completion_tokens

    def __repr__(self) -> str:
        copy_preview = (
            self.copy[:60] + "..." if len(self.copy) > 60 else self.copy
        )
        return (
            f"<CopywriterResult product={self.product_name!r} "
            f"type={self.copy_type!r} "
            f"copy_len={len(self.copy)} "
            f"elapsed={self.elapsed_ms:.1f}ms>"
        )


# ------------------------------------------------------------------ #
#                    文案写作器                                       #
# ------------------------------------------------------------------ #

# 文案类型中英文映射
_COPY_TYPE_NAMES = {
    "slogan": "广告语/口号",
    "product_description": "产品描述",
    "ad_copy": "广告文案",
    "social_media": "社交媒体文案",
    "landing_page": "落地页文案",
    "email": "营销邮件",
}

# 品牌调性中英文映射
_BRAND_TONE_NAMES = {
    "professional": "专业商务",
    "casual": "轻松随意",
    "luxurious": "奢华高端",
    "playful": "活泼有趣",
    "urgent": "紧迫促销",
    "inspirational": "激励鼓舞",
}


class Copywriter:
    """文案写作器 — 根据产品/品牌信息生成营销文案

    设计原则：
    - 组合模式：持有 LLMClient 实例
    - Prompt 驱动：使用 prompts.yaml 中 copywriting 角色的模板
    - 防御性编程：参数校验 + 异常统一处理

    示例：
        >>> writer = Copywriter()
        >>> result = writer.write(product_name="智能手表", copy_type="slogan")
        >>> print(result.copy)
    """

    PROMPT_ROLE = "copywriting"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """初始化文案写作器

        Args:
            llm_client: LLMClient 实例（None 时使用全局单例）
            temperature: 生成温度（None 时使用 LLMClient 默认值）
            max_tokens: 最大生成 token 数（None 时使用 LLMClient 默认值）
        """
        self._llm = llm_client or get_llm_client()
        self._temperature = temperature
        self._max_tokens = max_tokens

        debug_print(
            f"Copywriter 初始化完成: model={self._llm.model}"
        )

    def write(
        self,
        product_name: str,
        copy_type: str = "product_description",
        target_audience: str = "",
        brand_tone: str = "",
        key_selling_points: str = "",
        requirements: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> CopywriterResult:
        """生成营销文案

        Args:
            product_name: 产品/品牌名称（必填）
            copy_type: 文案类型（默认 product_description）
            target_audience: 目标受众
            brand_tone: 品牌调性
            key_selling_points: 核心卖点
            requirements: 额外要求
            temperature: 生成温度
            max_tokens: 最大生成 token 数

        Returns:
            CopywriterResult

        Raises:
            ValueError: product_name 为空时抛出
            ModelAPIError: LLM 调用失败时抛出
        """
        if not product_name or not product_name.strip():
            raise ValueError("产品/品牌名称不能为空")

        start_time = time.time()

        messages = self._build_messages(
            product_name=product_name,
            copy_type=copy_type,
            target_audience=target_audience,
            brand_tone=brand_tone,
            key_selling_points=key_selling_points,
            requirements=requirements,
        )

        debug_print(
            f"开始生成文案: product={product_name!r}, type={copy_type!r}"
        )

        tmp = temperature if temperature is not None else self._temperature
        mtk = max_tokens if max_tokens is not None else self._max_tokens

        copy_text, prompt_tokens, completion_tokens = self._call_llm(
            messages=messages,
            temperature=tmp,
            max_tokens=mtk,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        result = CopywriterResult(
            copy=copy_text,
            copy_type=copy_type,
            product_name=product_name,
            target_audience=target_audience,
            brand_tone=brand_tone,
            elapsed_ms=elapsed_ms,
            model=self._llm.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        debug_print(
            f"文案生成完成: copy_len={len(copy_text)}, "
            f"tokens={result.total_tokens}, elapsed={elapsed_ms:.1f}ms"
        )
        return result

    def _build_messages(
        self,
        product_name: str,
        copy_type: str,
        target_audience: str,
        brand_tone: str,
        key_selling_points: str,
        requirements: str,
    ) -> List[Dict[str, str]]:
        """构建发送给 LLM 的消息列表

        Args:
            product_name: 产品/品牌名称
            copy_type: 文案类型
            target_audience: 目标受众
            brand_tone: 品牌调性
            key_selling_points: 核心卖点
            requirements: 额外要求

        Returns:
            messages 列表
        """
        # 类型/调性翻译为中文
        copy_type_text = _COPY_TYPE_NAMES.get(copy_type, copy_type) or "产品描述"
        brand_tone_text = _BRAND_TONE_NAMES.get(
            brand_tone, brand_tone) or "无特殊要求"
        audience_text = target_audience or "通用受众"
        selling_points_text = key_selling_points or "未指定"
        requirements_text = requirements or "无额外要求"

        system_prompt = self._get_system_prompt()
        user_content = self._build_user_content(
            product_name=product_name,
            copy_type=copy_type_text,
            target_audience=audience_text,
            brand_tone=brand_tone_text,
            key_selling_points=selling_points_text,
            requirements=requirements_text,
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
            return "你是一个专业的文案创作者，擅长撰写各类营销文案。"

    def _build_user_content(
        self,
        product_name: str,
        copy_type: str,
        target_audience: str,
        brand_tone: str,
        key_selling_points: str,
        requirements: str,
    ) -> str:
        """构建发送给 LLM 的用户消息内容

        Args:
            product_name: 产品/品牌名称
            copy_type: 文案类型
            target_audience: 目标受众
            brand_tone: 品牌调性
            key_selling_points: 核心卖点
            requirements: 额外要求

        Returns:
            完整的用户消息文本
        """
        parts = [
            f"请为 {product_name} 撰写一段{copy_type}。",
            f"\n目标受众：{target_audience}",
            f"品牌调性：{brand_tone}",
            f"核心卖点：{key_selling_points}",
            f"额外要求：{requirements}",
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
            (copy_text, prompt_tokens, completion_tokens) 元组

        Raises:
            ModelAPIError: API 调用失败时抛出
        """
        kwargs: Dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._llm.chat_completion(messages, **kwargs)

        copy_text = response.choices[0].message.content or ""

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = (
                getattr(response.usage, "completion_tokens", 0) or 0
            )

        return copy_text, prompt_tokens, completion_tokens

    @property
    def llm_client(self) -> LLMClient:
        """当前使用的 LLMClient 实例"""
        return self._llm

    def __repr__(self) -> str:
        return f"<Copywriter model={self._llm.model!r}>"


# ------------------------------------------------------------------ #
#                    全局单例 & 便捷函数                               #
# ------------------------------------------------------------------ #

_copywriter_instance: Optional[Copywriter] = None


def get_copywriter() -> Copywriter:
    """获取全局 Copywriter 单例（懒加载）

    首次调用时会创建实例，后续调用直接返回同一实例。

    Returns:
        Copywriter 单例对象

    示例：
        >>> writer = get_copywriter()
        >>> result = writer.write(product_name="智能手表")
        >>> print(result.copy)
    """
    global _copywriter_instance
    if _copywriter_instance is None:
        _copywriter_instance = Copywriter()
    return _copywriter_instance
