# ============================================================================
# 每日工作日志报告模块  |  src/writing/daily_report.py
# ============================================================================
#
# 功能：把用户输入的简单工作条目，变成一份结构完整、表达专业的每日工作日志。
#       用户只需要把今天做了什么、遇到什么问题、明天打算做什么等信息丢进来，
#       大模型会负责整理、补充细节并输出美观的 Markdown 格式报告。
#
# 输入参数（均可选，至少填一项）：
#     work_items   : 今天完成的工作事项
#     issues       : 遇到的问题或困难
#     plans        : 明天或后续的工作计划
#     project_name : 所属项目名称
#     author       : 作者姓名
#     date         : 报告日期，默认当天
#     extra_notes  : 其他补充说明
#
# 输出格式：
#     Markdown 格式，包含报告头部、工作摘要、详细记录、问题与风险、明日计划、备注。
#
# 配置与依赖：
#     Prompt 模板：configs/prompts.yaml → daily_report 节点
#     依赖模块：src/core/config.py、src/core/models.py
#
# 使用方式：
#   from src.writing.daily_report import DailyReportWriter, get_daily_report_writer
#
#   writer = get_daily_report_writer()
#   result = writer.write(
#       work_items=["完成 RAG 模块开发", "编写单元测试"],
#       issues=["向量数据库性能待优化"],
#       plans=["开始 writing 模块开发"],
#       project_name="AI写作助手",
#       author="张三",
#   )
#   print(result.report)
# ============================================================================

import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from src.core.exceptions import ModelAPIError
from src.core.models import LLMClient, get_llm_client
from src.core.prompts import build_messages, get_prompt
from src.utils.text_utils import debug_print


# ------------------------------------------------------------------ #
#                    数据结构                                         #
# ------------------------------------------------------------------ #

@dataclass
class DailyReportResult:
    """每日工作日志报告生成结果

    Attributes:
        report: 生成的 Markdown 格式报告文本
        work_items: 原始输入的工作事项
        issues: 原始输入的问题列表
        plans: 原始输入的计划列表
        project_name: 项目名称
        author: 作者姓名
        report_date: 报告日期（YYYY-MM-DD）
        elapsed_ms: 生成耗时（毫秒）
        model: 使用的模型名称
        prompt_tokens: 提示词 token 数
        completion_tokens: 生成 token 数

    示例：
        >>> result = writer.write(work_items=["完成功能开发"])
        >>> print(result.report)
        >>> print(f"耗时: {result.elapsed_ms:.1f}ms")
    """
    report: str = ""
    work_items: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    plans: List[str] = field(default_factory=list)
    project_name: str = ""
    author: str = ""
    report_date: str = ""
    elapsed_ms: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """总 token 数（提示 + 生成）"""
        return self.prompt_tokens + self.completion_tokens

    def __repr__(self) -> str:
        """调试友好的字符串表示"""
        report_preview = (
            self.report[:60] + "..." if len(self.report) > 60 else self.report
        )
        return (
            f"<DailyReportResult date={self.report_date!r} "
            f"report_len={len(self.report)} "
            f"elapsed={self.elapsed_ms:.1f}ms>"
        )


# ------------------------------------------------------------------ #
#                    每日工作日志报告写作器                             #
# ------------------------------------------------------------------ #

class DailyReportWriter:
    """每日工作日志报告写作器

    将用户输入的简单工作条目，通过大模型整理为结构完整、表达专业的
    Markdown 格式每日工作日志报告。

    设计原则：
    - 组合模式：持有 LLMClient 实例，复用其底层能力
    - Prompt 驱动：使用 prompts.yaml 中 daily_report 角色的模板
    - 防御性编程：参数校验 + 异常统一处理

    示例：
        >>> writer = DailyReportWriter()
        >>> result = writer.write(
        ...     work_items=["完成 RAG 模块", "编写测试"],
        ...     issues=["性能待优化"],
        ...     plans=["开始 writing 模块"],
        ...     project_name="AI写作助手",
        ...     author="张三",
        ... )
        >>> print(result.report)
    """

    # prompts.yaml 中对应的角色名称
    PROMPT_ROLE = "daily_report"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """初始化每日工作日志报告写作器

        Args:
            llm_client: LLMClient 实例（None 时使用全局单例）
            temperature: 生成温度（None 时使用 LLMClient 默认值）
            max_tokens: 最大生成 token 数（None 时使用 LLMClient 默认值）
        """
        self._llm = llm_client or get_llm_client()
        self._temperature = temperature
        self._max_tokens = max_tokens

        debug_print(
            f"DailyReportWriter 初始化完成: model={self._llm.model}"
        )

    # ------------------------------------------------------------------ #
    #                    核心方法：生成报告                               #
    # ------------------------------------------------------------------ #

    def write(
        self,
        work_items: Optional[List[str]] = None,
        issues: Optional[List[str]] = None,
        plans: Optional[List[str]] = None,
        project_name: str = "",
        author: str = "",
        report_date: Optional[str] = None,
        extra_notes: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> DailyReportResult:
        """生成每日工作日志报告

        将用户提供的工作信息通过大模型整理为结构化的 Markdown 格式报告。
        所有参数均可选，但至少需提供 work_items / issues / plans 中的一项。

        Args:
            work_items: 今日完成的工作事项列表
            issues: 遇到的问题或困难列表
            plans: 明日/后续工作计划列表
            project_name: 所属项目名称
            author: 报告作者姓名
            report_date: 报告日期（YYYY-MM-DD 格式，默认当天）
            extra_notes: 其他补充说明
            temperature: 生成温度（None 时使用默认值）
            max_tokens: 最大生成 token 数（None 时使用默认值）

        Returns:
            DailyReportResult 包含生成的报告及元信息

        Raises:
            ValueError: 当 work_items、issues、plans 全部为空时抛出
            ModelAPIError: LLM 调用失败时抛出

        示例：
            >>> result = writer.write(
            ...     work_items=["完成用户认证模块", "修复登录Bug"],
            ...     issues=["第三方接口响应慢"],
            ...     plans=["编写接口文档"],
            ...     project_name="电商平台",
            ...     author="李四",
            ... )
        """
        # ── 参数校验 ──
        work_items = work_items or []
        issues = issues or []
        plans = plans or []

        if not any([work_items, issues, plans]):
            raise ValueError(
                "至少需要提供 work_items、issues、plans 中的一项"
            )

        # 报告日期，默认当天
        if not report_date:
            report_date = date.today().strftime("%Y-%m-%d")

        start_time = time.time()
        tmp = temperature if temperature is not None else self._temperature
        mtk = max_tokens if max_tokens is not None else self._max_tokens

        # ── 构建 Prompt ──
        messages = self._build_messages(
            work_items=work_items,
            issues=issues,
            plans=plans,
            project_name=project_name,
            author=author,
            report_date=report_date,
            extra_notes=extra_notes,
        )

        debug_print(
            f"生成日志报告: date={report_date}, "
            f"work_items={len(work_items)}, "
            f"issues={len(issues)}, plans={len(plans)}"
        )

        # ── 调用 LLM ──
        report, prompt_tokens, completion_tokens = self._call_llm(
            messages=messages,
            temperature=tmp,
            max_tokens=mtk,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        result = DailyReportResult(
            report=report,
            work_items=work_items,
            issues=issues,
            plans=plans,
            project_name=project_name,
            author=author,
            report_date=report_date,
            elapsed_ms=elapsed_ms,
            model=self._llm.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        debug_print(
            f"报告生成完成: report_len={len(report)}, "
            f"tokens={result.total_tokens}, "
            f"elapsed={elapsed_ms:.1f}ms"
        )
        return result

    # ------------------------------------------------------------------ #
    #                    内部辅助方法                                     #
    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        work_items: List[str],
        issues: List[str],
        plans: List[str],
        project_name: str,
        author: str,
        report_date: str,
        extra_notes: str,
    ) -> List[Dict[str, str]]:
        """构建发送给 LLM 的消息列表

        使用 prompts.yaml 中 daily_report 角色的 system 和 user_template，
        将用户参数渲染到模板中。

        Args:
            work_items: 工作事项列表
            issues: 问题列表
            plans: 计划列表
            project_name: 项目名称
            author: 作者姓名
            report_date: 报告日期
            extra_notes: 补充说明

        Returns:
            messages 列表
        """
        # 将列表格式化为字符串，方便填入 Prompt 模板
        work_items_text = self._format_list(work_items, default_text="无")
        issues_text = self._format_list(issues, default_text="无")
        plans_text = self._format_list(plans, default_text="无")
        project_name = project_name or "未指定"
        author = author or "未署名"
        extra_notes = extra_notes or "无"

        # 使用 prompts.py 中的 build_messages 构建消息
        return build_messages(
            self.PROMPT_ROLE,
            work_items=work_items_text,
            issues=issues_text,
            plans=plans_text,
            project_name=project_name,
            author=author,
            date=report_date,
            extra_notes=extra_notes,
        )

    @staticmethod
    def _format_list(
        items: List[str],
        default_text: str = "无",
        numbered: bool = True,
    ) -> str:
        """将列表格式化为可读的文本

        Args:
            items: 字符串列表
            default_text: 列表为空时的默认文本
            numbered: 是否使用编号格式（默认 True）

        Returns:
            格式化后的文本

        示例：
            >>> DailyReportWriter._format_list(["A", "B"])
            '1. A\\n2. B'
            >>> DailyReportWriter._format_list([])
            '无'
        """
        if not items:
            return default_text

        if numbered:
            filtered = [item.strip() for item in items if item.strip()]
            if not filtered:
                return default_text
            return "\n".join(
                f"{i+1}. {text}" for i, text in enumerate(filtered)
            )
        else:
            filtered = [item.strip() for item in items if item.strip()]
            if not filtered:
                return default_text
            return "\n".join(f"- {text}" for text in filtered)

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

        # 提取回答文本
        report_text = response.choices[0].message.content or ""

        # 提取 token 用量统计
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = (
                getattr(response.usage, "completion_tokens", 0) or 0
            )

        return report_text, prompt_tokens, completion_tokens

    # ------------------------------------------------------------------ #
    #                    属性方法                                         #
    # ------------------------------------------------------------------ #

    @property
    def llm_client(self) -> LLMClient:
        """当前使用的 LLMClient 实例"""
        return self._llm

    def __repr__(self) -> str:
        """调试友好的字符串表示"""
        return f"<DailyReportWriter model={self._llm.model!r}>"


# ------------------------------------------------------------------ #
#                    全局单例 & 便捷函数                               #
# ------------------------------------------------------------------ #

_daily_report_writer_instance: Optional[DailyReportWriter] = None


def get_daily_report_writer() -> DailyReportWriter:
    """获取全局 DailyReportWriter 单例（懒加载）

    首次调用时会创建实例，后续调用直接返回同一实例。

    Returns:
        DailyReportWriter 单例对象

    示例：
        >>> writer = get_daily_report_writer()
        >>> result = writer.write(work_items=["完成功能开发"])
        >>> print(result.report)
    """
    global _daily_report_writer_instance
    if _daily_report_writer_instance is None:
        _daily_report_writer_instance = DailyReportWriter()
    return _daily_report_writer_instance
