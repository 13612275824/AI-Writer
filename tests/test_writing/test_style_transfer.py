# ============================================================================
# 风格转换模块测试  |  tests/test_writing/test_style_transfer.py
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
from src.writing.style_transfer import (
    StyleTransfer,
    StyleTransferResult,
    get_style_transfer,
    _STYLE_NAMES,
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
#                    StyleTransferResult 测试                         #
# ------------------------------------------------------------------ #

class TestStyleTransferResult:
    """StyleTransferResult 数据类测试"""

    def test_default_values(self):
        """默认值初始化"""
        result = StyleTransferResult()
        assert result.transferred_content == ""
        assert result.original_content == ""
        assert result.target_style == ""
        assert result.source_style == ""
        assert result.char_diff == 0
        assert result.elapsed_ms == 0.0
        assert result.model == ""
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_total_tokens(self):
        """总 token 数计算"""
        result = StyleTransferResult(prompt_tokens=100, completion_tokens=50)
        assert result.total_tokens == 150

    def test_total_tokens_zero(self):
        """无 token 信息时总数为 0"""
        result = StyleTransferResult()
        assert result.total_tokens == 0

    def test_length_ratio(self):
        """长度比率计算"""
        result = StyleTransferResult(
            original_content="abcde",  # 5 chars
            transferred_content="abcdefgh",  # 8 chars
        )
        assert result.length_ratio == 8 / 5

    def test_length_ratio_shorter(self):
        """转换后更短时的比率"""
        result = StyleTransferResult(
            original_content="abcdefghij",  # 10 chars
            transferred_content="abcde",  # 5 chars
        )
        assert result.length_ratio == 0.5

    def test_length_ratio_empty_original(self):
        """原始内容为空时比率为 0"""
        result = StyleTransferResult(
            original_content="",
            transferred_content="some text",
        )
        assert result.length_ratio == 0.0

    def test_repr_short_content(self):
        """短内容的 repr"""
        result = StyleTransferResult(
            transferred_content="转换后文本",
            target_style="formal",
            char_diff=2,
            elapsed_ms=100.0,
        )
        r = repr(result)
        assert "target='formal'" in r
        assert "diff=+2" in r
        assert "elapsed=100.0ms" in r

    def test_repr_long_content(self):
        """长内容的 repr（截断显示）"""
        result = StyleTransferResult(transferred_content="x" * 200)
        r = repr(result)
        assert "content_len=200" in r

    def test_repr_empty(self):
        """空内容的 repr"""
        result = StyleTransferResult()
        r = repr(result)
        assert "content_len=0" in r


# ------------------------------------------------------------------ #
#                    StyleTransfer 初始化测试                         #
# ------------------------------------------------------------------ #

class TestStyleTransferInit:
    """StyleTransfer 初始化测试"""

    def test_init_with_mock_llm(self):
        """使用模拟 LLM 初始化"""
        mock_llm = _make_mock_llm("custom-model")
        transfer = StyleTransfer(llm_client=mock_llm)
        assert transfer.llm_client is mock_llm
        assert transfer.llm_client.model == "custom-model"

    def test_init_with_default_llm(self):
        """使用默认 LLM 初始化"""
        mock_llm = _make_mock_llm()
        transfer = StyleTransfer(llm_client=mock_llm)
        assert transfer.llm_client is mock_llm

    def test_init_with_temperature(self):
        """指定温度参数"""
        mock_llm = _make_mock_llm()
        transfer = StyleTransfer(llm_client=mock_llm, temperature=0.5)
        assert transfer._temperature == 0.5

    def test_init_with_max_tokens(self):
        """指定最大 token 数"""
        mock_llm = _make_mock_llm()
        transfer = StyleTransfer(llm_client=mock_llm, max_tokens=1000)
        assert transfer._max_tokens == 1000

    def test_repr(self):
        """repr 方法"""
        mock_llm = _make_mock_llm("my-model")
        transfer = StyleTransfer(llm_client=mock_llm)
        assert "my-model" in repr(transfer)

    def test_supported_styles(self):
        """支持的转换风格集合"""
        expected = {"formal", "casual", "academic", "professional",
                    "creative", "news", "storytelling"}
        assert StyleTransfer.SUPPORTED_STYLES == expected


# ------------------------------------------------------------------ #
#                    transfer 方法测试                                #
# ------------------------------------------------------------------ #

class TestTransfer:
    """StyleTransfer.transfer() 方法测试"""

    def _make_transfer(self, response_content="转换后的内容"):
        """创建带模拟响应的 transfer"""
        mock_llm = _make_mock_llm("test-model")
        mock_llm.chat_completion.return_value = _make_mock_response(
            response_content, prompt_tokens=50, completion_tokens=30
        )
        return StyleTransfer(llm_client=mock_llm)

    def test_basic_transfer(self):
        """基本风格转换"""
        transfer = self._make_transfer("正式书面风格的文本")
        result = transfer.transfer(content="口语化文本", target_style="formal")
        assert result.transferred_content == "正式书面风格的文本"
        assert result.original_content == "口语化文本"
        assert result.target_style == "formal"
        assert result.elapsed_ms > 0

    def test_casual_to_academic(self):
        """口语到学术风格转换"""
        transfer = self._make_transfer("学术风格的严谨表述")
        result = transfer.transfer(content="口语化文本", target_style="academic")
        assert result.target_style == "academic"

    def test_formal_to_casual(self):
        """正式到口语风格转换"""
        transfer = self._make_transfer("轻松随意的表达")
        result = transfer.transfer(
            content="正式文件内容",
            target_style="casual",
            source_style="formal",
        )
        assert result.target_style == "casual"
        assert result.source_style == "formal"

    def test_auto_source_style(self):
        """源风格自动识别"""
        transfer = self._make_transfer("转换后内容")
        result = transfer.transfer(content="原始文本", target_style="formal")
        assert result.source_style == "auto"

    def test_explicit_source_style(self):
        """显式指定源风格"""
        transfer = self._make_transfer("转换后内容")
        result = transfer.transfer(
            content="原始文本",
            target_style="formal",
            source_style="casual",
        )
        assert result.source_style == "casual"

    def test_with_requirements(self):
        """指定额外转换要求"""
        transfer = self._make_transfer("转换后内容")
        result = transfer.transfer(
            content="原始文本",
            target_style="academic",
            requirements="保留专业术语",
        )
        assert result.transferred_content == "转换后内容"

    def test_token_statistics(self):
        """token 统计"""
        transfer = self._make_transfer("转换内容")
        result = transfer.transfer(content="原始内容", target_style="formal")
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 30
        assert result.total_tokens == 80

    def test_char_diff(self):
        """字符数变化计算"""
        transfer = self._make_transfer("转换后的更长文本内容")
        result = transfer.transfer(content="短文本", target_style="formal")
        assert result.char_diff == len("转换后的更长文本内容") - len("短文本")

    def test_model_info(self):
        """模型信息记录"""
        transfer = self._make_transfer("转换内容")
        result = transfer.transfer(content="原始内容", target_style="formal")
        assert result.model == "test-model"

    def test_all_supported_styles(self):
        """所有支持的转换风格都能通过"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("结果")
        transfer = StyleTransfer(llm_client=mock_llm)
        for style in StyleTransfer.SUPPORTED_STYLES:
            result = transfer.transfer(content="测试内容", target_style=style)
            assert result.target_style == style

    def test_temperature_override(self):
        """温度参数覆盖"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("结果")
        transfer = StyleTransfer(llm_client=mock_llm, temperature=0.3)
        transfer.transfer(content="内容", target_style="formal", temperature=0.8)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.8

    def test_temperature_default(self):
        """温度参数使用默认值"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("结果")
        transfer = StyleTransfer(llm_client=mock_llm, temperature=0.3)
        transfer.transfer(content="内容", target_style="formal")
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.3

    def test_max_tokens_override(self):
        """max_tokens 参数覆盖"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("结果")
        transfer = StyleTransfer(llm_client=mock_llm, max_tokens=500)
        transfer.transfer(content="内容", target_style="formal", max_tokens=1000)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 1000


# ------------------------------------------------------------------ #
#                    参数校验测试                                     #
# ------------------------------------------------------------------ #

class TestTransferValidation:
    """参数校验测试"""

    def _make_transfer(self):
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("结果")
        return StyleTransfer(llm_client=mock_llm)

    def test_empty_content_raises(self):
        """空内容抛出 ValueError"""
        transfer = self._make_transfer()
        with pytest.raises(ValueError, match="待转换内容不能为空"):
            transfer.transfer(content="", target_style="formal")

    def test_whitespace_content_raises(self):
        """纯空格内容抛出 ValueError"""
        transfer = self._make_transfer()
        with pytest.raises(ValueError, match="待转换内容不能为空"):
            transfer.transfer(content="   ", target_style="formal")

    def test_none_content_raises(self):
        """None 内容抛出 ValueError"""
        transfer = self._make_transfer()
        with pytest.raises(ValueError):
            transfer.transfer(content=None, target_style="formal")

    def test_invalid_style_raises(self):
        """不支持的风格抛出 ValueError"""
        transfer = self._make_transfer()
        with pytest.raises(ValueError, match="不支持的目标风格"):
            transfer.transfer(content="内容", target_style="invalid_style")

    def test_empty_target_style_raises(self):
        """空目标风格抛出 ValueError"""
        transfer = self._make_transfer()
        with pytest.raises(ValueError, match="不支持的目标风格"):
            transfer.transfer(content="内容", target_style="")


# ------------------------------------------------------------------ #
#                    内部方法测试                                     #
# ------------------------------------------------------------------ #

class TestBuildMessages:
    """_build_messages 方法测试"""

    def _make_transfer(self):
        mock_llm = _make_mock_llm()
        return StyleTransfer(llm_client=mock_llm)

    def test_message_structure(self):
        """消息列表包含 system 和 user 两条消息"""
        transfer = self._make_transfer()
        messages = transfer._build_messages(
            content="测试内容",
            target_style="正式书面",
            source_style="",
            requirements="",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_non_empty(self):
        """系统提示词非空"""
        transfer = self._make_transfer()
        messages = transfer._build_messages(
            content="测试内容",
            target_style="正式书面",
            source_style="",
            requirements="",
        )
        assert len(messages[0]["content"]) > 0

    def test_user_content_contains_target(self):
        """用户消息包含目标风格"""
        transfer = self._make_transfer()
        messages = transfer._build_messages(
            content="内容",
            target_style="正式书面",
            source_style="",
            requirements="",
        )
        assert "正式书面" in messages[1]["content"]

    def test_user_content_contains_original(self):
        """用户消息包含原始内容"""
        transfer = self._make_transfer()
        messages = transfer._build_messages(
            content="这是原始文本",
            target_style="正式书面",
            source_style="",
            requirements="",
        )
        assert "这是原始文本" in messages[1]["content"]

    def test_user_content_contains_source(self):
        """用户消息包含源风格"""
        transfer = self._make_transfer()
        messages = transfer._build_messages(
            content="内容",
            target_style="正式书面",
            source_style="轻松口语",
            requirements="",
        )
        assert "轻松口语" in messages[1]["content"]

    def test_user_content_auto_source(self):
        """无源风格时显示自动识别"""
        transfer = self._make_transfer()
        messages = transfer._build_messages(
            content="内容",
            target_style="正式书面",
            source_style="",
            requirements="",
        )
        assert "自动识别" in messages[1]["content"]


class TestGetSystemPrompt:
    """_get_system_prompt 方法测试"""

    def _make_transfer(self):
        mock_llm = _make_mock_llm()
        return StyleTransfer(llm_client=mock_llm)

    def test_prompt_non_empty(self):
        """系统提示词非空"""
        transfer = self._make_transfer()
        prompt = transfer._get_system_prompt()
        assert len(prompt) > 0

    def test_prompt_contains_style_transfer(self):
        """提示词包含风格转换关键词"""
        transfer = self._make_transfer()
        prompt = transfer._get_system_prompt()
        assert "风格转换" in prompt or "风格" in prompt

    def test_prompt_fallback_on_error(self):
        """Prompt 加载失败时使用兜底"""
        transfer = self._make_transfer()
        with patch("src.writing.style_transfer.get_prompt", side_effect=Exception("mock")):
            prompt = transfer._get_system_prompt()
            assert len(prompt) > 0
            assert "风格" in prompt


class TestBuildUserContent:
    """_build_user_content 方法测试"""

    def _make_transfer(self):
        mock_llm = _make_mock_llm()
        return StyleTransfer(llm_client=mock_llm)

    def test_contains_all_parts(self):
        """用户内容包含所有部分"""
        transfer = self._make_transfer()
        content = transfer._build_user_content(
            content="原始文本",
            target_style="正式书面",
            source_style="轻松口语",
            requirements="保留关键信息",
        )
        assert "正式书面" in content
        assert "轻松口语" in content
        assert "保留关键信息" in content
        assert "原始文本" in content

    def test_no_requirements(self):
        """无额外要求时包含默认文本"""
        transfer = self._make_transfer()
        content = transfer._build_user_content(
            content="原始文本",
            target_style="正式书面",
            source_style="自动识别",
            requirements="无额外要求",
        )
        assert "无额外要求" in content


# ------------------------------------------------------------------ #
#                    LLM 调用异常测试                                 #
# ------------------------------------------------------------------ #

class TestLLMCallErrors:
    """LLM 调用异常场景测试"""

    def test_api_error_propagated(self):
        """API 调用异常向上抛出"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.side_effect = ModelAPIError("API 调用失败")
        transfer = StyleTransfer(llm_client=mock_llm)
        with pytest.raises(ModelAPIError):
            transfer.transfer(content="测试内容", target_style="formal")

    def test_usage_none(self):
        """usage 为 None 时 token 数为 0"""
        mock_llm = _make_mock_llm()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "转换后"
        mock_response.usage = None
        mock_llm.chat_completion.return_value = mock_response

        transfer = StyleTransfer(llm_client=mock_llm)
        result = transfer.transfer(content="原始内容", target_style="formal")
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_empty_response_content(self):
        """模型返回空内容"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("")
        transfer = StyleTransfer(llm_client=mock_llm)
        result = transfer.transfer(content="原始内容", target_style="formal")
        assert result.transferred_content == ""


# ------------------------------------------------------------------ #
#                    映射常量测试                                     #
# ------------------------------------------------------------------ #

class TestMappings:
    """风格映射测试"""

    def test_style_names_complete(self):
        """风格映射包含所有支持的类型"""
        for style in StyleTransfer.SUPPORTED_STYLES:
            assert style in _STYLE_NAMES

    def test_style_names_chinese(self):
        """所有映射值都是中文"""
        for name in _STYLE_NAMES.values():
            # 至少包含一个中文字符
            assert any('\u4e00' <= c <= '\u9fff' for c in name)

    def test_formal_mapping(self):
        """formal 映射正确"""
        assert _STYLE_NAMES["formal"] == "正式书面"

    def test_casual_mapping(self):
        """casual 映射正确"""
        assert _STYLE_NAMES["casual"] == "轻松口语"

    def test_academic_mapping(self):
        """academic 映射正确"""
        assert _STYLE_NAMES["academic"] == "学术论文"


# ------------------------------------------------------------------ #
#                    全局单例测试                                     #
# ------------------------------------------------------------------ #

class TestGetStyleTransfer:
    """get_style_transfer 全局单例测试"""

    def test_returns_instance(self):
        """返回 StyleTransfer 实例"""
        import src.writing.style_transfer as mod
        mod._style_transfer_instance = None
        transfer = get_style_transfer()
        assert isinstance(transfer, StyleTransfer)

    def test_singleton(self):
        """多次调用返回同一实例"""
        import src.writing.style_transfer as mod
        mod._style_transfer_instance = None
        t1 = get_style_transfer()
        t2 = get_style_transfer()
        assert t1 is t2

    def test_singleton_reset(self):
        """重置后创建新实例"""
        import src.writing.style_transfer as mod
        mod._style_transfer_instance = None
        t1 = get_style_transfer()
        mod._style_transfer_instance = None
        t2 = get_style_transfer()
        assert t1 is not t2
