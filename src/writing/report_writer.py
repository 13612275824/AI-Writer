# ============================================================================
# 报告写作模块  |  src/writing/report_writer.py
# ============================================================================
#
# 功能：根据用户提供的主题和素材，生成结构完整的专业报告
#
# 与 daily_report.py 的区别：
#     - daily_report:  每日工作日志（工作项→结构化日志）
#     - report_writer: 通用结构化报告（工作报告/项目报告/分析报告/调研报告）
#
# 输入参数：
#     title         : 报告标题（必填）
#     report_type   : 报告类型（可选：work_summary/project/analysis/research）
#                     - work_summary : 工作总结报告
#                     - project      : 项目报告
#                     - analysis     : 分析报告
#                     - research     : 调研报告
#     content       : 用户提供的素材或要点（可选）
#     sections      : 自定义章节结构（可选，逗号分隔或列表）
#     requirements  : 额外要求（可选）
#     word_count    : 目标字数（可选）
#
# 输出格式：
#     ReportResult 数据结构，包含报告正文、摘要、token 统计等
#
# 配置与依赖：
#     Prompt 模板：configs/prompts.yaml → report_writer 节点
#     依赖模块：src/core/models.py、src/core/prompts.py
#
# 使用方式：
#   from src.writing.report_writer import ReportWriter, get_report_writer
#
#   writer = get_report_writer()
#   result = writer.write(
#       title="2024年度工作总结报告",
#       report_type="work_summary",
#       content="完成了 RAG 模块、Writing 模块等核心功能开发",
#   )
#   print(result.report)
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
class ReportResult:
    """报告写作结果

    Attributes:
        report: 生成的报告正文
        summary: 报告摘要（自动生成）
        title: 原始标题
        report_type: 报告类型
        elapsed_ms: 处理耗时（毫秒）
        model: 使用的模型名称
        prompt_tokens: 提示词 token 数
        completion_tokens: 生成 token 数

    示例：
        >>> result = writer.write(title="项目进展报告", report_type="project")
        >>> print(result.report)
        >>> print(f"耗时: {result.elapsed_ms:.1f}ms")
    """
    report: str = ""
    summary: str = ""
    title: str = ""
    report_type: str = ""
    elapsed_ms: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """返回总 token 数"""
        return self.prompt_tokens + self.completion_tokens

    @property
    def word_count(self) -> int:
        """报告字数（中文字符 + 英文单词）"""
        if not self.report:
            return 0
        # 简单计算：中文字符数 + 英文单词数
        chinese = sum(1 for c in self.report if '\u4e00' <= c <= '\u9fff')
        english = sum(1 for c in self.report if c.isascii() and c.isalpha())
        return chinese + english

    def __repr__(self) -> str:
        preview = (
            self.report[:80] + "..."
            if len(self.report) > 80
            else self.report
        )
        return (
            f"<ReportResult title={self.title!r} type={self.report_type!r} "
            f"report_len={len(self.report)} "
            f"elapsed={self.elapsed_ms:.1f}ms>"
        )


# ------------------------------------------------------------------ #
#                    报告写作器                                       #
# ------------------------------------------------------------------ #

# 报告类型中英文映射
_REPORT_TYPE_NAMES = {
    "work_summary": "工作总结报告",
    "project": "项目报告",
    "analysis": "分析报告",
    "research": "调研报告",
}

# 各报告类型的默认章节结构
_DEFAULT_SECTIONS = {
    "work_summary": [
        "报告概述", "工作完成情况", "重点工作回顾", "问题与反思", "下一步计划",
    ],
    "project": [
        "项目概述", "项目背景", "实施过程", "项目成果", "问题与风险", "总结与建议",
    ],
    "analysis": [
        "分析概述", "分析背景", "数据与现状", "分析过程", "主要发现", "结论与建议",
    ],
    "research": [
        "调研概述", "调研背景与目的", "调研方法", "调研结果", "分析与讨论", "结论与建议",
    ],
}


class ReportWriter:
    """报告写作器 — 根据主题和素材生成结构完整的专业报告

    设计原则：
    - 组合模式：持有 LLMClient 实例
    - Prompt 驱动：使用 prompts.yaml 中 report_writer 角色的模板
    - 防御性编程：参数校验 + 异常统一处理

    示例：
        >>> writer = ReportWriter()
        >>> result = writer.write(
        ...     title="Q1 季度工作总结",
        ...     report_type="work_summary",
        ...     content="完成了系统架构设计和核心模块开发",
        ... )
        >>> print(result.report)
    """

    PROMPT_ROLE = "report_writer"

    # 支持的报告类型
    SUPPORTED_TYPES = frozenset(
        ["work_summary", "project", "analysis", "research"])

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """初始化报告写作器

        Args:
            llm_client: LLMClient 实例（None 时使用全局单例）
            temperature: 生成温度（None 时使用 LLMClient 默认值）
            max_tokens: 最大生成 token 数（None 时使用 LLMClient 默认值）
        """
        self._llm = llm_client or get_llm_client()
        self._temperature = temperature
        self._max_tokens = max_tokens

        debug_print(
            f"ReportWriter 初始化完成: model={self._llm.model}"
        )

    def write(
        self,
        title: str,
        report_type: str = "work_summary",
        content: str = "",
        sections: str = "",
        requirements: str = "",
        word_count: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ReportResult:
        """根据主题生成专业报告

        Args:
            title: 报告标题（必填）
            report_type: 报告类型（可选，默认 work_summary）
            content: 用户提供的素材或要点（可选）
            sections: 自定义章节结构（可选，逗号分隔或列表）
            requirements: 额外要求（可选）
            word_count: 目标字数（可选）
            temperature: 生成温度（None 时使用默认值）
            max_tokens: 最大生成 token 数（None 时使用默认值）

        Returns:
            ReportResult 包含报告正文及元信息

        Raises:
            ValueError: title 为空或 report_type 不支持时抛出
            ModelAPIError: LLM 调用失败时抛出

        示例：
            >>> result = writer.write(
            ...     title="年度项目总结",
            ...     report_type="project",
            ...     content="完成了三个核心模块的开发",
            ... )
        """
        # ── 参数校验 ──
        if not title or not title.strip():
            raise ValueError("报告标题不能为空")

        if report_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"不支持的报告类型: {report_type!r}，"
                f"支持的类型: {sorted(self.SUPPORTED_TYPES)}"
            )

        start_time = time.time()

        # 确定章节结构
        section_list = self._parse_sections(sections, report_type)

        messages = self._build_messages(
            title=title,
            report_type=report_type,
            content=content,
            sections=section_list,
            requirements=requirements,
            word_count=word_count,
        )

        type_name = _REPORT_TYPE_NAMES.get(report_type, report_type)
        debug_print(
            f"开始生成报告: type={type_name}, title={title!r}, "
            f"sections={len(section_list)}, content_len={len(content)}"
        )

        tmp = temperature if temperature is not None else self._temperature
        mtk = max_tokens if max_tokens is not None else self._max_tokens

        report_text, prompt_tokens, completion_tokens = self._call_llm(
            messages=messages,
            temperature=tmp,
            max_tokens=mtk,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # 提取摘要（取报告前 200 字或第一个段落）
        summary = self._extract_summary(report_text)

        result = ReportResult(
            report=report_text,
            summary=summary,
            title=title,
            report_type=report_type,
            elapsed_ms=elapsed_ms,
            model=self._llm.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        debug_print(
            f"报告生成完成: report_len={len(report_text)}, "
            f"tokens={result.total_tokens}, elapsed={elapsed_ms:.1f}ms"
        )
        return result

    # ------------------------------------------------------------------ #
    #                    内部辅助方法                                     #
    # ------------------------------------------------------------------ #

    def _parse_sections(self, sections: str, report_type: str) -> List[str]:
        """解析章节结构

        Args:
            sections: 用户自定义章节（逗号分隔字符串或列表）
            report_type: 报告类型（用于获取默认章节）

        Returns:
            章节列表
        """
        if sections and sections.strip():
            # 用户自定义章节：支持逗号分隔
            return [s.strip() for s in sections.split(",") if s.strip()]
        # 使用默认章节
        return _DEFAULT_SECTIONS.get(report_type, ["概述", "正文", "结论"])

    def _build_messages(
        self,
        title: str,
        report_type: str,
        content: str,
        sections: List[str],
        requirements: str,
        word_count: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """构建发送给 LLM 的消息列表

        Args:
            title: 报告标题
            report_type: 报告类型
            content: 用户素材
            sections: 章节列表
            requirements: 额外要求
            word_count: 目标字数

        Returns:
            messages 列表
        """
        type_name = _REPORT_TYPE_NAMES.get(report_type, report_type)
        req_text = requirements or "无额外要求"
        wc_text = f"约 {word_count} 字" if word_count else "无特定要求"
        sections_text = "\n".join(f"  - {s}" for s in sections)

        system_prompt = self._get_system_prompt()
        user_content = self._build_user_content(
            title=title,
            type_name=type_name,
            content=content or "由你根据报告类型和标题自行组织",
            sections_text=sections_text,
            requirements=req_text,
            word_count_text=wc_text,
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
                "你是一个专业的报告撰写专家，擅长根据用户提供的信息生成结构完整、"
                "逻辑清晰的专业报告。报告包含标题、摘要、正文、结论等完整结构。"
            )

    def _build_user_content(
        self,
        title: str,
        type_name: str,
        content: str,
        sections_text: str,
        requirements: str,
        word_count_text: str,
    ) -> str:
        """构建发送给 LLM 的用户消息内容

        Args:
            title: 报告标题
            type_name: 报告类型中文名称
            content: 用户素材
            sections_text: 章节结构文本
            requirements: 额外要求
            word_count_text: 目标字数文本

        Returns:
            完整的用户消息文本
        """
        parts = [
            f"请撰写一份【{type_name}】。",
            f"\n报告标题：{title}",
            f"\n报告章节结构：\n{sections_text}",
            f"\n目标字数：{word_count_text}",
            f"额外要求：{requirements}",
            f"\n用户提供的素材/要点：\n{content}",
        ]
        return "\n".join(parts)

    def _extract_summary(self, report_text: str) -> str:
        """从报告正文中提取摘要

        优先取第一段（标题后的第一段内容），最多 200 字。

        Args:
            report_text: 报告正文

        Returns:
            摘要文本
        """
        if not report_text:
            return ""

        # 跳过标题行，取第一个非空非标题段落
        lines = report_text.split("\n")
        for line in lines:
            stripped = line.strip()
            # 跳过空行和标题行
            if not stripped or stripped.startswith("#"):
                continue
            # 取这个段落，截断到 200 字
            if len(stripped) > 200:
                return stripped[:200] + "..."
            return stripped
        return ""

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
            (report_text, prompt_tokens, completion_tokens) 元组

        Raises:
            ModelAPIError: API 调用失败时抛出
        """
        kwargs: Dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._llm.chat_completion(messages, **kwargs)

        report_text = response.choices[0].message.content or ""

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = (
                getattr(response.usage, "completion_tokens", 0) or 0
            )

        return report_text, prompt_tokens, completion_tokens

    @property
    def llm_client(self) -> LLMClient:
        """当前使用的 LLMClient 实例"""
        return self._llm

    def __repr__(self) -> str:
        return f"<ReportWriter model={self._llm.model!r}>"


# ------------------------------------------------------------------ #
#                    全局单例 & 便捷函数                               #
# ------------------------------------------------------------------ #

_report_writer_instance: Optional[ReportWriter] = None


def get_report_writer() -> ReportWriter:
    """获取全局 ReportWriter 单例（懒加载）

    首次调用时会创建实例，后续调用直接返回同一实例。

    Returns:
        ReportWriter 单例对象

    示例：
        >>> writer = get_report_writer()
        >>> result = writer.write(title="项目进展报告")
        >>> print(result.report)
    """
    global _report_writer_instance
    if _report_writer_instance is None:
        _report_writer_instance = ReportWriter()
    return _report_writer_instance
