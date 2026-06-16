# ============================================================================
# 报告写作模块测试  |  tests/test_writing/test_report_writer.py
# ============================================================================

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure Worker2/ is on sys.path so `src` package can be found
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.exceptions import ModelAPIError
from src.writing.report_writer import (
    ReportWriter,
    ReportResult,
    get_report_writer,
    _REPORT_TYPE_NAMES,
    _DEFAULT_SECTIONS,
)

# ── 确保项目根目录在 sys.path 中 ──────────────────────────────
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ------------------------------------------------------------------ #
#                    辅助工具                                         #
# ------------------------------------------------------------------ #

def _make_mock_llm(model_name: str = "test-model"):
    """创建模拟的 LLMClient"""
    mock_llm = MagicMock()
    mock_llm.model = model_name
    return mock_llm


def _make_mock_response(content: str, prompt_tokens: int = 0, completion_tokens: int = 0):
    """创建模拟的 chat_completion 响应"""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    return mock_response


# ------------------------------------------------------------------ #
#                    ReportResult 测试                                #
# ------------------------------------------------------------------ #

class TestReportResult:
    """ReportResult 数据类测试"""

    def test_default_values(self):
        """默认值初始化"""
        result = ReportResult()
        assert result.report == ""
        assert result.summary == ""
        assert result.title == ""
        assert result.report_type == ""
        assert result.elapsed_ms == 0.0
        assert result.model == ""
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_total_tokens(self):
        """总 token 数计算"""
        result = ReportResult(prompt_tokens=100, completion_tokens=50)
        assert result.total_tokens == 150

    def test_total_tokens_zero(self):
        """无 token 信息时总数为 0"""
        result = ReportResult()
        assert result.total_tokens == 0

    def test_word_count_chinese(self):
        """中文字数计算"""
        result = ReportResult(report="这是一段中文报告内容")
        # 10 个中文字符
        assert result.word_count == 10

    def test_word_count_english(self):
        """英文字母计算"""
        result = ReportResult(report="This is a test report")
        # 17 个 ASCII 字母
        assert result.word_count == 17

    def test_word_count_mixed(self):
        """中英混合计算"""
        result = ReportResult(report="这是 test 报告")
        # 4 个中文字 + 4 个 ASCII 字母
        assert result.word_count == 8

    def test_word_count_empty(self):
        """空报告字数为 0"""
        result = ReportResult()
        assert result.word_count == 0

    def test_repr_short_content(self):
        """短内容的 repr"""
        result = ReportResult(
            report="报告内容",
            title="测试报告",
            report_type="work_summary",
            elapsed_ms=100.0,
        )
        r = repr(result)
        assert "title='测试报告'" in r
        assert "type='work_summary'" in r
        assert "elapsed=100.0ms" in r

    def test_repr_long_content(self):
        """长内容的 repr（截断显示）"""
        result = ReportResult(report="x" * 200)
        r = repr(result)
        assert "report_len=200" in r


# ------------------------------------------------------------------ #
#                    ReportWriter 初始化测试                          #
# ------------------------------------------------------------------ #

class TestReportWriterInit:
    """ReportWriter 初始化测试"""

    def test_init_with_mock_llm(self):
        """使用模拟 LLM 初始化"""
        mock_llm = _make_mock_llm("custom-model")
        writer = ReportWriter(llm_client=mock_llm)
        assert writer.llm_client is mock_llm
        assert writer.llm_client.model == "custom-model"

    def test_init_with_default_llm(self):
        """使用默认 LLM 初始化"""
        mock_llm = _make_mock_llm()
        writer = ReportWriter(llm_client=mock_llm)
        assert writer.llm_client is mock_llm

    def test_init_with_temperature(self):
        """指定温度参数"""
        mock_llm = _make_mock_llm()
        writer = ReportWriter(llm_client=mock_llm, temperature=0.5)
        assert writer._temperature == 0.5

    def test_init_with_max_tokens(self):
        """指定最大 token 数"""
        mock_llm = _make_mock_llm()
        writer = ReportWriter(llm_client=mock_llm, max_tokens=2000)
        assert writer._max_tokens == 2000

    def test_repr(self):
        """repr 方法"""
        mock_llm = _make_mock_llm("my-model")
        writer = ReportWriter(llm_client=mock_llm)
        assert "my-model" in repr(writer)

    def test_supported_types(self):
        """支持的报告类型集合"""
        expected = {"work_summary", "project", "analysis", "research"}
        assert ReportWriter.SUPPORTED_TYPES == expected


# ------------------------------------------------------------------ #
#                    write 方法测试                                   #
# ------------------------------------------------------------------ #

class TestWrite:
    """ReportWriter.write() 方法测试"""

    def _make_writer(self, response_content="# 报告标题\n\n这是报告概述内容。\n\n## 工作完成情况\n\n详细内容"):
        """创建带模拟响应的 writer"""
        mock_llm = _make_mock_llm("test-model")
        mock_llm.chat_completion.return_value = _make_mock_response(
            response_content, prompt_tokens=80, completion_tokens=120
        )
        return ReportWriter(llm_client=mock_llm)

    def test_basic_write(self):
        """基本报告写作"""
        writer = self._make_writer()
        result = writer.write(title="年度工作总结", report_type="work_summary")
        assert result.title == "年度工作总结"
        assert result.report_type == "work_summary"
        assert len(result.report) > 0
        assert result.elapsed_ms > 0

    def test_project_report(self):
        """项目报告"""
        writer = self._make_writer()
        result = writer.write(title="RAG 模块开发报告", report_type="project")
        assert result.report_type == "project"

    def test_analysis_report(self):
        """分析报告"""
        writer = self._make_writer()
        result = writer.write(title="用户行为分析", report_type="analysis")
        assert result.report_type == "analysis"

    def test_research_report(self):
        """调研报告"""
        writer = self._make_writer()
        result = writer.write(title="AI 技术趋势调研", report_type="research")
        assert result.report_type == "research"

    def test_with_content(self):
        """指定素材内容"""
        writer = self._make_writer()
        result = writer.write(
            title="项目进展报告",
            content="完成了模块 A 和模块 B 的开发",
        )
        assert result.report == writer._llm.chat_completion.return_value.choices[
            0].message.content

    def test_with_custom_sections(self):
        """自定义章节结构"""
        writer = self._make_writer()
        result = writer.write(
            title="自定义报告",
            sections="引言,主体内容,总结",
        )
        assert result.title == "自定义报告"

    def test_with_requirements(self):
        """指定额外要求"""
        writer = self._make_writer()
        result = writer.write(
            title="正式报告",
            requirements="使用正式语言，包含数据图表",
        )
        assert result.title == "正式报告"

    def test_with_word_count(self):
        """指定目标字数"""
        writer = self._make_writer()
        result = writer.write(title="长报告", word_count=5000)
        assert result.title == "长报告"

    def test_token_statistics(self):
        """token 统计"""
        writer = self._make_writer()
        result = writer.write(title="测试报告")
        assert result.prompt_tokens == 80
        assert result.completion_tokens == 120
        assert result.total_tokens == 200

    def test_summary_extraction(self):
        """摘要自动提取"""
        writer = self._make_writer("# 报告标题\n\n这是报告的摘要内容。\n\n## 第一章")
        result = writer.write(title="测试报告")
        assert "摘要" in result.summary or "报告" in result.summary

    def test_model_info(self):
        """模型信息记录"""
        writer = self._make_writer()
        result = writer.write(title="测试报告")
        assert result.model == "test-model"

    def test_all_supported_types(self):
        """所有支持的报告类型都能通过"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("报告内容")
        writer = ReportWriter(llm_client=mock_llm)
        for rtype in ReportWriter.SUPPORTED_TYPES:
            result = writer.write(title="测试报告", report_type=rtype)
            assert result.report_type == rtype

    def test_temperature_override(self):
        """温度参数覆盖"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("报告内容")
        writer = ReportWriter(llm_client=mock_llm, temperature=0.3)
        writer.write(title="报告", temperature=0.8)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.8

    def test_temperature_default(self):
        """温度参数使用默认值"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("报告内容")
        writer = ReportWriter(llm_client=mock_llm, temperature=0.3)
        writer.write(title="报告")
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.3

    def test_max_tokens_override(self):
        """max_tokens 参数覆盖"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("报告内容")
        writer = ReportWriter(llm_client=mock_llm, max_tokens=500)
        writer.write(title="报告", max_tokens=2000)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 2000


# ------------------------------------------------------------------ #
#                    参数校验测试                                     #
# ------------------------------------------------------------------ #

class TestWriteValidation:
    """参数校验测试"""

    def _make_writer(self):
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("报告内容")
        return ReportWriter(llm_client=mock_llm)

    def test_empty_title_raises(self):
        """空标题抛出 ValueError"""
        writer = self._make_writer()
        with pytest.raises(ValueError, match="报告标题不能为空"):
            writer.write(title="")

    def test_whitespace_title_raises(self):
        """纯空格标题抛出 ValueError"""
        writer = self._make_writer()
        with pytest.raises(ValueError, match="报告标题不能为空"):
            writer.write(title="   ")

    def test_none_title_raises(self):
        """None 标题抛出 ValueError"""
        writer = self._make_writer()
        with pytest.raises(ValueError):
            writer.write(title=None)

    def test_invalid_type_raises(self):
        """不支持的类型抛出 ValueError"""
        writer = self._make_writer()
        with pytest.raises(ValueError, match="不支持的报告类型"):
            writer.write(title="报告", report_type="invalid_type")

    def test_empty_type_raises(self):
        """空类型抛出 ValueError"""
        writer = self._make_writer()
        with pytest.raises(ValueError, match="不支持的报告类型"):
            writer.write(title="报告", report_type="")


# ------------------------------------------------------------------ #
#                    内部方法测试                                     #
# ------------------------------------------------------------------ #

class TestParseSections:
    """_parse_sections 方法测试"""

    def _make_writer(self):
        mock_llm = _make_mock_llm()
        return ReportWriter(llm_client=mock_llm)

    def test_custom_sections(self):
        """自定义章节（逗号分隔）"""
        writer = self._make_writer()
        sections = writer._parse_sections("引言,主体,结论", "work_summary")
        assert sections == ["引言", "主体", "结论"]

    def test_custom_sections_with_spaces(self):
        """自定义章节（带空格）"""
        writer = self._make_writer()
        sections = writer._parse_sections(" 引言 , 主体 , 结论 ", "work_summary")
        assert sections == ["引言", "主体", "结论"]

    def test_empty_sections_uses_default(self):
        """空章节使用默认"""
        writer = self._make_writer()
        sections = writer._parse_sections("", "work_summary")
        assert sections == _DEFAULT_SECTIONS["work_summary"]

    def test_whitespace_sections_uses_default(self):
        """纯空格章节使用默认"""
        writer = self._make_writer()
        sections = writer._parse_sections("   ", "project")
        assert sections == _DEFAULT_SECTIONS["project"]

    def test_unknown_type_fallback(self):
        """未知类型使用兜底章节"""
        writer = self._make_writer()
        # 手动调用时传入不支持的类型（绕过 write 方法的校验）
        sections = writer._parse_sections("", "unknown_type")
        assert sections == ["概述", "正文", "结论"]


class TestBuildMessages:
    """_build_messages 方法测试"""

    def _make_writer(self):
        mock_llm = _make_mock_llm()
        return ReportWriter(llm_client=mock_llm)

    def test_message_structure(self):
        """消息列表包含 system 和 user 两条消息"""
        writer = self._make_writer()
        messages = writer._build_messages(
            title="测试报告",
            report_type="work_summary",
            content="测试内容",
            sections=["概述", "正文", "结论"],
            requirements="",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_non_empty(self):
        """系统提示词非空"""
        writer = self._make_writer()
        messages = writer._build_messages(
            title="测试报告",
            report_type="work_summary",
            content="内容",
            sections=["概述"],
            requirements="",
        )
        assert len(messages[0]["content"]) > 0

    def test_user_content_contains_title(self):
        """用户消息包含标题"""
        writer = self._make_writer()
        messages = writer._build_messages(
            title="年度工作总结",
            report_type="work_summary",
            content="内容",
            sections=["概述"],
            requirements="",
        )
        assert "年度工作总结" in messages[1]["content"]

    def test_user_content_contains_sections(self):
        """用户消息包含章节"""
        writer = self._make_writer()
        messages = writer._build_messages(
            title="报告",
            report_type="project",
            content="内容",
            sections=["项目概述", "实施过程", "项目成果"],
            requirements="",
        )
        assert "项目概述" in messages[1]["content"]
        assert "实施过程" in messages[1]["content"]


class TestGetSystemPrompt:
    """_get_system_prompt 方法测试"""

    def _make_writer(self):
        mock_llm = _make_mock_llm()
        return ReportWriter(llm_client=mock_llm)

    def test_prompt_non_empty(self):
        """系统提示词非空"""
        writer = self._make_writer()
        prompt = writer._get_system_prompt()
        assert len(prompt) > 0

    def test_prompt_contains_report_keywords(self):
        """提示词包含报告相关关键词"""
        writer = self._make_writer()
        prompt = writer._get_system_prompt()
        assert "报告" in prompt or "撰写" in prompt

    def test_prompt_fallback_on_error(self):
        """Prompt 加载失败时使用兜底"""
        writer = self._make_writer()
        with patch("src.writing.report_writer.get_prompt", side_effect=Exception("mock")):
            prompt = writer._get_system_prompt()
            assert len(prompt) > 0
            assert "报告" in prompt


class TestExtractSummary:
    """_extract_summary 方法测试"""

    def _make_writer(self):
        mock_llm = _make_mock_llm()
        return ReportWriter(llm_client=mock_llm)

    def test_extract_first_paragraph(self):
        """提取第一段作为摘要"""
        writer = self._make_writer()
        report = "# 标题\n\n这是摘要内容。\n\n## 第一章\n\n详细内容"
        summary = writer._extract_summary(report)
        assert "摘要" in summary

    def test_skip_empty_lines(self):
        """跳过空行"""
        writer = self._make_writer()
        report = "# 标题\n\n\n这是第一段落。"
        summary = writer._extract_summary(report)
        assert "段落" in summary

    def test_truncate_long_summary(self):
        """长摘要截断"""
        writer = self._make_writer()
        report = "# 标题\n\n" + "x" * 300
        summary = writer._extract_summary(report)
        assert len(summary) <= 203  # 200 + "..."

    def test_empty_report(self):
        """空报告返回空摘要"""
        writer = self._make_writer()
        summary = writer._extract_summary("")
        assert summary == ""


# ------------------------------------------------------------------ #
#                    LLM 调用异常测试                                 #
# ------------------------------------------------------------------ #

class TestLLMCallErrors:
    """LLM 调用异常场景测试"""

    def test_api_error_propagated(self):
        """API 调用异常向上抛出"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.side_effect = ModelAPIError("API 调用失败")
        writer = ReportWriter(llm_client=mock_llm)
        with pytest.raises(ModelAPIError):
            writer.write(title="测试报告")

    def test_usage_none(self):
        """usage 为 None 时 token 数为 0"""
        mock_llm = _make_mock_llm()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "报告内容"
        mock_response.usage = None
        mock_llm.chat_completion.return_value = mock_response

        writer = ReportWriter(llm_client=mock_llm)
        result = writer.write(title="测试报告")
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_empty_response_content(self):
        """模型返回空内容"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("")
        writer = ReportWriter(llm_client=mock_llm)
        result = writer.write(title="测试报告")
        assert result.report == ""


# ------------------------------------------------------------------ #
#                    映射常量测试                                     #
# ------------------------------------------------------------------ #

class TestMappings:
    """类型映射测试"""

    def test_type_names_complete(self):
        """类型映射包含所有支持的类型"""
        for rtype in ReportWriter.SUPPORTED_TYPES:
            assert rtype in _REPORT_TYPE_NAMES

    def test_type_names_chinese(self):
        """所有映射值都是中文"""
        for name in _REPORT_TYPE_NAMES.values():
            assert any('\u4e00' <= c <= '\u9fff' for c in name)

    def test_default_sections_complete(self):
        """默认章节包含所有支持的类型"""
        for rtype in ReportWriter.SUPPORTED_TYPES:
            assert rtype in _DEFAULT_SECTIONS

    def test_work_summary_mapping(self):
        """work_summary 映射正确"""
        assert _REPORT_TYPE_NAMES["work_summary"] == "工作总结报告"

    def test_project_mapping(self):
        """project 映射正确"""
        assert _REPORT_TYPE_NAMES["project"] == "项目报告"


# ------------------------------------------------------------------ #
#                    全局单例测试                                     #
# ------------------------------------------------------------------ #

class TestGetReportWriter:
    """get_report_writer 全局单例测试"""

    def test_returns_instance(self):
        """返回 ReportWriter 实例"""
        import src.writing.report_writer as mod
        mod._report_writer_instance = None
        writer = get_report_writer()
        assert isinstance(writer, ReportWriter)

    def test_singleton(self):
        """多次调用返回同一实例"""
        import src.writing.report_writer as mod
        mod._report_writer_instance = None
        w1 = get_report_writer()
        w2 = get_report_writer()
        assert w1 is w2

    def test_singleton_reset(self):
        """重置后创建新实例"""
        import src.writing.report_writer as mod
        mod._report_writer_instance = None
        w1 = get_report_writer()
        mod._report_writer_instance = None
        w2 = get_report_writer()
        assert w1 is not w2
