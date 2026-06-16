# ============================================================================
# 每日工作日志报告模块测试  |  tests/test_writing/test_daily_report.py
# ============================================================================

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure Worker2/ is on sys.path so `src` package can be found
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.writing.daily_report import (
    DailyReportResult,
    DailyReportWriter,
    get_daily_report_writer,
)

# ── 确保项目根目录在 sys.path 中 ──────────────────────────────
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ------------------------------------------------------------------ #
#                    DailyReportResult 测试                           #
# ------------------------------------------------------------------ #

class TestDailyReportResult:
    """DailyReportResult 数据类测试"""

    def test_default_values(self):
        """默认值初始化"""
        result = DailyReportResult()
        assert result.report == ""
        assert result.work_items == []
        assert result.issues == []
        assert result.plans == []
        assert result.project_name == ""
        assert result.author == ""
        assert result.report_date == ""
        assert result.elapsed_ms == 0.0
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_total_tokens(self):
        """总 token 数计算"""
        result = DailyReportResult(prompt_tokens=100, completion_tokens=50)
        assert result.total_tokens == 150

    def test_total_tokens_zero(self):
        """无 token 信息时总数为 0"""
        result = DailyReportResult()
        assert result.total_tokens == 0

    def test_repr_short_report(self):
        """短报告的 repr"""
        result = DailyReportResult(report="短报告", report_date="2026-05-30")
        r = repr(result)
        assert "2026-05-30" in r
        assert "report_len=3" in r

    def test_repr_long_report(self):
        """长报告的 repr（截断显示）"""
        result = DailyReportResult(report="x" * 200, report_date="2026-05-30")
        r = repr(result)
        assert "report_len=200" in r


# ------------------------------------------------------------------ #
#                    DailyReportWriter 测试                           #
# ------------------------------------------------------------------ #

class TestDailyReportWriter:
    """DailyReportWriter 类测试"""

    # ── _format_list 静态方法测试 ──

    def test_format_list_numbered(self):
        """编号格式输出"""
        result = DailyReportWriter._format_list(["A", "B", "C"])
        assert result == "1. A\n2. B\n3. C"

    def test_format_list_unnumbered(self):
        """无序列表格式输出"""
        result = DailyReportWriter._format_list(["A", "B"], numbered=False)
        assert result == "- A\n- B"

    def test_format_list_empty(self):
        """空列表返回默认文本"""
        result = DailyReportWriter._format_list([])
        assert result == "无"

    def test_format_list_custom_default(self):
        """空列表使用自定义默认文本"""
        result = DailyReportWriter._format_list([], default_text="暂无")
        assert result == "暂无"

    def test_format_list_strips_whitespace(self):
        """自动去除条目两端空白"""
        result = DailyReportWriter._format_list(["  A  ", " B"])
        assert result == "1. A\n2. B"

    def test_format_list_skips_empty_items(self):
        """跳过空白条目"""
        result = DailyReportWriter._format_list(["A", "", "  ", "B"])
        assert result == "1. A\n2. B"

    def test_format_list_all_blank_returns_default(self):
        """全部为空白条目时返回默认文本"""
        result = DailyReportWriter._format_list(["", "  "])
        assert result == "无"

    # ── 参数校验测试 ──

    def test_write_all_empty_raises(self):
        """work_items/issues/plans 全为空时抛出 ValueError"""
        mock_llm = MagicMock()
        writer = DailyReportWriter(llm_client=mock_llm)

        with pytest.raises(ValueError, match="至少需要提供"):
            writer.write()

    def test_write_only_issues(self):
        """仅提供 issues 也能正常调用"""
        mock_llm = self._make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm)

        result = writer.write(issues=["接口响应慢"])
        assert isinstance(result, DailyReportResult)
        mock_llm.chat_completion.assert_called_once()

    def test_write_only_plans(self):
        """仅提供 plans 也能正常调用"""
        mock_llm = self._make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm)

        result = writer.write(plans=["编写文档"])
        assert isinstance(result, DailyReportResult)

    # ── write() 核心流程测试 ──

    def test_write_returns_result(self):
        """write() 返回 DailyReportResult，字段正确填充"""
        mock_llm = self._make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm)

        result = writer.write(
            work_items=["完成RAG模块", "编写测试"],
            issues=["性能待优化"],
            plans=["开始writing模块"],
            project_name="AI写作助手",
            author="张三",
            report_date="2026-05-30",
            extra_notes="无特别备注",
        )

        assert isinstance(result, DailyReportResult)
        assert result.report == "# 每日工作日志报告\n内容..."
        assert result.work_items == ["完成RAG模块", "编写测试"]
        assert result.issues == ["性能待优化"]
        assert result.plans == ["开始writing模块"]
        assert result.project_name == "AI写作助手"
        assert result.author == "张三"
        assert result.report_date == "2026-05-30"
        assert result.elapsed_ms > 0
        assert result.model == "qwen-plus"
        assert result.prompt_tokens == 120
        assert result.completion_tokens == 80

    def test_write_default_date(self):
        """未传 report_date 时默认为当天"""
        mock_llm = self._make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm)

        result = writer.write(work_items=["测试"])
        today = date.today().strftime("%Y-%m-%d")
        assert result.report_date == today

    def test_write_calls_llm_once(self):
        """write() 仅调用 LLM 一次"""
        mock_llm = self._make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm)

        writer.write(work_items=["测试"])
        assert mock_llm.chat_completion.call_count == 1

    def test_write_passes_messages_format(self):
        """传递给 LLM 的 messages 格式正确（包含 system 和 user）"""
        mock_llm = self._make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm)

        writer.write(
            work_items=["完成任务A"],
            issues=["问题B"],
            plans=["计划C"],
            project_name="项目X",
            author="作者Y",
            report_date="2026-05-30",
            extra_notes="备注Z",
        )

        call_args = mock_llm.chat_completion.call_args
        messages = call_args[0][0]  # 第一个位置参数

        # 应至少包含 system 和 user 两条消息
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"

        # user 消息应包含关键信息
        user_content = messages[-1]["content"]
        assert "2026-05-30" in user_content
        assert "作者Y" in user_content
        assert "项目X" in user_content
        assert "任务A" in user_content
        assert "问题B" in user_content
        assert "计划C" in user_content
        assert "备注Z" in user_content

    def test_write_with_temperature(self):
        """自定义 temperature 能传递给 LLM"""
        mock_llm = self._make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm, temperature=0.7)

        writer.write(work_items=["测试"])
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs.get("temperature") == 0.7

    def test_write_with_max_tokens(self):
        """自定义 max_tokens 能传递给 LLM"""
        mock_llm = self._make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm, max_tokens=2000)

        writer.write(work_items=["测试"])
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs.get("max_tokens") == 2000

    def test_write_override_temperature(self):
        """调用时的 temperature 覆盖初始化时的值"""
        mock_llm = self._make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm, temperature=0.5)

        writer.write(work_items=["测试"], temperature=0.9)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs.get("temperature") == 0.9

    # ── repr 测试 ──

    def test_repr(self):
        """repr 包含模型名称"""
        mock_llm = self._make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm)
        assert "qwen-plus" in repr(writer)

    # ── 辅助方法 ──

    @staticmethod
    def _make_mock_llm(report_text: str = "# 每日工作日志报告\n内容...") -> MagicMock:
        """创建模拟 LLMClient

        Args:
            report_text: 模拟生成的报告文本

        Returns:
            配置好的 MagicMock 对象
        """
        mock_llm = MagicMock()
        mock_llm.model = "qwen-plus"

        # 模拟 chat_completion 返回
        mock_choice = MagicMock()
        mock_choice.message.content = report_text

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 120
        mock_response.usage.completion_tokens = 80

        mock_llm.chat_completion.return_value = mock_response
        return mock_llm


# ------------------------------------------------------------------ #
#                    全局单例测试                                     #
# ------------------------------------------------------------------ #

class TestGetDailyReportWriter:
    """get_daily_report_writer 单例测试"""

    def setup_method(self):
        """每个测试前重置全局单例"""
        import src.writing.daily_report as mod
        mod._daily_report_writer_instance = None

    @patch("src.writing.daily_report.get_llm_client")
    def test_returns_instance(self, mock_get_llm):
        """返回 DailyReportWriter 实例"""
        mock_get_llm.return_value = MagicMock()
        writer = get_daily_report_writer()
        assert isinstance(writer, DailyReportWriter)

    @patch("src.writing.daily_report.get_llm_client")
    def test_singleton(self, mock_get_llm):
        """多次调用返回同一实例"""
        mock_get_llm.return_value = MagicMock()
        w1 = get_daily_report_writer()
        w2 = get_daily_report_writer()
        assert w1 is w2


# ------------------------------------------------------------------ #
#                    独立运行测试（python 直接执行）                    #
# ------------------------------------------------------------------ #

def _make_mock_llm(report_text: str = "# 每日工作日志报告\n内容...") -> MagicMock:
    """创建模拟 LLMClient"""
    mock_llm = MagicMock()
    mock_llm.model = "qwen-plus"

    mock_choice = MagicMock()
    mock_choice.message.content = report_text

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 120
    mock_response.usage.completion_tokens = 80

    mock_llm.chat_completion.return_value = mock_response
    return mock_llm


def check_format_list():
    """检查 _format_list 格式化"""
    print("=" * 60)
    print("测试 1: _format_list 列表格式化")
    print("=" * 60)

    try:
        # 编号格式
        result = DailyReportWriter._format_list(["完成RAG模块", "编写测试", "代码审查"])
        print(f"  编号格式:\n    {result.replace(chr(10), chr(10) + '    ')}")
        assert result == "1. 完成RAG模块\n2. 编写测试\n3. 代码审查"
        print("  ✓ 编号格式正确")

        # 无序格式
        result = DailyReportWriter._format_list(["A", "B"], numbered=False)
        print(f"  无序格式: {result!r}")
        assert result == "- A\n- B"
        print("  ✓ 无序格式正确")

        # 空列表
        result = DailyReportWriter._format_list([])
        assert result == "无"
        print(f"  空列表: {result!r} ✓")

        # 跳过空白条目并重新编号
        result = DailyReportWriter._format_list(["A", "", "  ", "B"])
        assert result == "1. A\n2. B"
        print(f"  跳过空白: {result!r} ✓")

        print()
        return True

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        print()
        return False


def check_write_basic():
    """检查 write() 基本流程"""
    print("=" * 60)
    print("测试 2: write() 基本生成流程")
    print("=" * 60)

    try:
        mock_llm = _make_mock_llm("# 每日工作日志\n\n## 工作摘要\n完成RAG模块开发...")
        writer = DailyReportWriter(llm_client=mock_llm)

        result = writer.write(
            work_items=["完成RAG模块开发", "编写单元测试"],
            issues=["向量检索性能待优化"],
            plans=["开始writing模块开发"],
            project_name="AI写作助手",
            author="张三",
            report_date="2026-05-30",
        )

        print(f"  报告日期: {result.report_date}")
        print(f"  报告长度: {len(result.report)} 字符")
        print(f"  使用模型: {result.model}")
        print(
            f"  Token统计: prompt={result.prompt_tokens}, completion={result.completion_tokens}")
        print(f"  耗时: {result.elapsed_ms:.1f}ms")
        print(f"  报告预览:\n    {result.report[:80]}...")

        assert isinstance(result, DailyReportResult)
        assert result.report_date == "2026-05-30"
        assert result.model == "qwen-plus"
        assert result.prompt_tokens == 120
        assert result.completion_tokens == 80
        assert result.elapsed_ms > 0
        mock_llm.chat_completion.assert_called_once()

        print("  ✓ 所有字段正确填充")
        print()
        return True

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        print()
        return False


def check_write_default_date():
    """检查 write() 默认日期"""
    print("=" * 60)
    print("测试 3: write() 默认日期为当天")
    print("=" * 60)

    try:
        mock_llm = _make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm)

        result = writer.write(work_items=["测试默认日期"])
        today = date.today().strftime("%Y-%m-%d")

        print(f"  今天日期: {today}")
        print(f"  报告日期: {result.report_date}")
        assert result.report_date == today
        print("  ✓ 默认日期正确")
        print()
        return True

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        print()
        return False


def check_write_validation():
    """检查 write() 参数校验"""
    print("=" * 60)
    print("测试 4: write() 参数校验")
    print("=" * 60)

    try:
        mock_llm = _make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm)

        # 全为空应报错
        try:
            writer.write()
            print("  ✗ 未抛出 ValueError")
            return False
        except ValueError as e:
            print(f"  全空参数: ValueError({e}) ✓")

        # 仅 issues 应正常
        result = writer.write(issues=["接口超时"])
        assert isinstance(result, DailyReportResult)
        print("  仅传 issues: 正常 ✓")

        # 仅 plans 应正常
        result = writer.write(plans=["编写文档"])
        assert isinstance(result, DailyReportResult)
        print("  仅传 plans: 正常 ✓")

        print()
        return True

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        print()
        return False


def check_messages_format():
    """检查 messages 构建格式"""
    print("=" * 60)
    print("测试 5: messages 消息格式验证")
    print("=" * 60)

    try:
        mock_llm = _make_mock_llm()
        writer = DailyReportWriter(llm_client=mock_llm)

        writer.write(
            work_items=["完成任务A"],
            issues=["问题B"],
            plans=["计划C"],
            project_name="项目X",
            author="作者Y",
            report_date="2026-05-30",
            extra_notes="备注Z",
        )

        messages = mock_llm.chat_completion.call_args[0][0]

        print(f"  消息数量: {len(messages)}")
        assert len(messages) >= 2
        print(f"  ✓ 消息数量正确")

        print(f"  首条角色: {messages[0]['role']}")
        assert messages[0]["role"] == "system"
        print(f"  ✓ system 消息存在")

        print(f"  末条角色: {messages[-1]['role']}")
        assert messages[-1]["role"] == "user"
        print(f"  ✓ user 消息存在")

        user_content = messages[-1]["content"]
        for keyword in ["2026-05-30", "作者Y", "项目X", "任务A", "问题B", "计划C", "备注Z"]:
            assert keyword in user_content, f"缺少: {keyword}"
            print(f"  ✓ user 消息包含 '{keyword}'")

        print()
        return True

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        print()
        return False


def check_temperature_override():
    """检查 temperature 参数传递与覆盖"""
    print("=" * 60)
    print("测试 6: temperature 参数传递与覆盖")
    print("=" * 60)

    try:
        mock_llm = _make_mock_llm()

        # 初始化时设置
        writer = DailyReportWriter(llm_client=mock_llm, temperature=0.7)
        writer.write(work_items=["测试"])
        kwargs = mock_llm.chat_completion.call_args[1]
        assert kwargs.get("temperature") == 0.7
        print(f"  初始化 temperature=0.7: 传递正确 ✓")

        # 调用时覆盖
        mock_llm.reset_mock()
        writer.write(work_items=["测试"], temperature=0.9)
        kwargs = mock_llm.chat_completion.call_args[1]
        assert kwargs.get("temperature") == 0.9
        print(f"  调用覆盖 temperature=0.9: 传递正确 ✓")

        # max_tokens
        mock_llm.reset_mock()
        writer2 = DailyReportWriter(llm_client=mock_llm, max_tokens=2000)
        writer2.write(work_items=["测试"])
        kwargs = mock_llm.chat_completion.call_args[1]
        assert kwargs.get("max_tokens") == 2000
        print(f"  max_tokens=2000: 传递正确 ✓")

        print()
        return True

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        print()
        return False


def check_result_dataclass():
    """检查 DailyReportResult 数据类"""
    print("=" * 60)
    print("测试 7: DailyReportResult 数据类")
    print("=" * 60)

    try:
        # 默认值
        r = DailyReportResult()
        assert r.report == ""
        assert r.total_tokens == 0
        print(f"  默认值初始化 ✓")

        # token 计算
        r = DailyReportResult(prompt_tokens=100, completion_tokens=50)
        assert r.total_tokens == 150
        print(f"  total_tokens = {r.total_tokens} ✓")

        # repr
        r = DailyReportResult(report="短报告", report_date="2026-05-30")
        s = repr(r)
        assert "2026-05-30" in s
        assert "report_len=3" in s
        print(f"  repr: {s} ✓")

        print()
        return True

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        print()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "DailyReportWriter 模块测试" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = []

    # 运行测试
    results.append(("列表格式化", check_format_list()))
    results.append(("基本生成流程", check_write_basic()))
    results.append(("默认日期", check_write_default_date()))
    results.append(("参数校验", check_write_validation()))
    results.append(("消息格式验证", check_messages_format()))
    results.append(("参数传递与覆盖", check_temperature_override()))
    results.append(("Result 数据类", check_result_dataclass()))

    # 打印测试结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")

    print()
    print(f"总计: {passed}/{total} 通过")
    print()

    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查错误信息")

    print()


if __name__ == "__main__":
    main()
