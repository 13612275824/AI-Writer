# ============================================================================
# 内容优化模块测试  |  tests/test_writing/test_content_optimizer.py
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
from src.writing.content_optimizer import (
    ContentOptimizer,
    OptimizationResult,
    get_content_optimizer,
    _OPTIMIZE_TYPE_NAMES,
    _TARGET_STYLE_NAMES,
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
#                    OptimizationResult 测试                          #
# ------------------------------------------------------------------ #

class TestOptimizationResult:
    """OptimizationResult 数据类测试"""

    def test_default_values(self):
        """默认值初始化"""
        result = OptimizationResult()
        assert result.optimized_content == ""
        assert result.original_content == ""
        assert result.optimize_type == ""
        assert result.target_style == ""
        assert result.summary == ""
        assert result.char_diff == 0
        assert result.elapsed_ms == 0.0
        assert result.model == ""
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_total_tokens(self):
        """总 token 数计算"""
        result = OptimizationResult(prompt_tokens=100, completion_tokens=50)
        assert result.total_tokens == 150

    def test_total_tokens_zero(self):
        """无 token 信息时总数为 0"""
        result = OptimizationResult()
        assert result.total_tokens == 0

    def test_improvement_ratio(self):
        """字符变化比率计算"""
        result = OptimizationResult(
            original_content="abcde",  # 5 chars
            optimized_content="abcdef",  # 6 chars
            char_diff=1,
        )
        assert result.improvement_ratio == 0.2  # 1/5

    def test_improvement_ratio_negative(self):
        """内容精简时的比率"""
        result = OptimizationResult(
            original_content="abcdefghij",  # 10 chars
            optimized_content="abcde",  # 5 chars
            char_diff=-5,
        )
        assert result.improvement_ratio == -0.5

    def test_improvement_ratio_empty_original(self):
        """原始内容为空时比率为 0"""
        result = OptimizationResult(original_content="", char_diff=5)
        assert result.improvement_ratio == 0.0

    def test_repr_short_content(self):
        """短内容的 repr"""
        result = OptimizationResult(
            optimized_content="优化后文本",
            optimize_type="polish",
            char_diff=2,
            elapsed_ms=100.0,
        )
        r = repr(result)
        assert "type='polish'" in r
        assert "diff=+2" in r
        assert "elapsed=100.0ms" in r

    def test_repr_long_content(self):
        """长内容的 repr（截断显示）"""
        result = OptimizationResult(optimized_content="x" * 200)
        r = repr(result)
        assert "content_len=200" in r

    def test_repr_empty(self):
        """空内容的 repr"""
        result = OptimizationResult()
        r = repr(result)
        assert "content_len=0" in r


# ------------------------------------------------------------------ #
#                    ContentOptimizer 初始化测试                      #
# ------------------------------------------------------------------ #

class TestContentOptimizerInit:
    """ContentOptimizer 初始化测试"""

    def test_init_with_mock_llm(self):
        """使用模拟 LLM 初始化"""
        mock_llm = _make_mock_llm("custom-model")
        optimizer = ContentOptimizer(llm_client=mock_llm)
        assert optimizer.llm_client is mock_llm
        assert optimizer.llm_client.model == "custom-model"

    def test_init_with_default_llm(self):
        """使用默认 LLM 初始化"""
        mock_llm = _make_mock_llm()
        optimizer = ContentOptimizer(llm_client=mock_llm)
        assert optimizer.llm_client is mock_llm

    def test_init_with_temperature(self):
        """指定温度参数"""
        mock_llm = _make_mock_llm()
        optimizer = ContentOptimizer(llm_client=mock_llm, temperature=0.5)
        assert optimizer._temperature == 0.5

    def test_init_with_max_tokens(self):
        """指定最大 token 数"""
        mock_llm = _make_mock_llm()
        optimizer = ContentOptimizer(llm_client=mock_llm, max_tokens=1000)
        assert optimizer._max_tokens == 1000

    def test_repr(self):
        """repr 方法"""
        mock_llm = _make_mock_llm("my-model")
        optimizer = ContentOptimizer(llm_client=mock_llm)
        assert "my-model" in repr(optimizer)


# ------------------------------------------------------------------ #
#                    optimize 方法测试                                #
# ------------------------------------------------------------------ #

class TestOptimize:
    """ContentOptimizer.optimize() 方法测试"""

    def _make_optimizer(self, response_content="优化后的内容"):
        """创建带模拟响应的 optimizer"""
        mock_llm = _make_mock_llm("test-model")
        mock_llm.chat_completion.return_value = _make_mock_response(
            response_content, prompt_tokens=50, completion_tokens=30
        )
        return ContentOptimizer(llm_client=mock_llm)

    def test_basic_polish(self):
        """基本润色优化"""
        optimizer = self._make_optimizer("润色后的文本内容")
        result = optimizer.optimize(content="原始文本内容", optimize_type="polish")
        assert result.optimized_content == "润色后的文本内容"
        assert result.original_content == "原始文本内容"
        assert result.optimize_type == "polish"
        assert result.elapsed_ms > 0

    def test_simplify(self):
        """简化改写"""
        optimizer = self._make_optimizer("简化版文本")
        result = optimizer.optimize(
            content="这是一段非常复杂的文本", optimize_type="simplify")
        assert result.optimized_content == "简化版文本"
        assert result.optimize_type == "simplify"

    def test_expand(self):
        """扩写丰富"""
        optimizer = self._make_optimizer("扩写后的丰富内容，包含更多细节和例证")
        result = optimizer.optimize(content="简短内容", optimize_type="expand")
        assert result.optimize_type == "expand"
        assert result.char_diff > 0  # 扩写后内容应更长

    def test_shorten(self):
        """精简缩写"""
        optimizer = self._make_optimizer("精简内容")
        result = optimizer.optimize(
            content="这是一段非常冗长的文本内容", optimize_type="shorten")
        assert result.optimize_type == "shorten"

    def test_grammar(self):
        """语法校对"""
        optimizer = self._make_optimizer("修正后的文本内容")
        result = optimizer.optimize(
            content="有错误的文本内容", optimize_type="grammar")
        assert result.optimize_type == "grammar"

    def test_with_target_style(self):
        """指定目标风格"""
        optimizer = self._make_optimizer("正式风格的文本")
        result = optimizer.optimize(
            content="随意文本",
            optimize_type="polish",
            target_style="formal",
        )
        assert result.target_style == "formal"

    def test_with_focus_areas(self):
        """指定重点关注方向"""
        optimizer = self._make_optimizer("优化后内容")
        result = optimizer.optimize(
            content="原始内容",
            focus_areas="语法,表达,结构",
        )
        assert result.optimized_content == "优化后内容"

    def test_with_requirements(self):
        """指定额外要求"""
        optimizer = self._make_optimizer("优化后内容")
        result = optimizer.optimize(
            content="原始内容",
            requirements="保持专业术语不变",
        )
        assert result.optimized_content == "优化后内容"

    def test_token_statistics(self):
        """token 统计"""
        optimizer = self._make_optimizer("优化内容")
        result = optimizer.optimize(content="原始内容")
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 30
        assert result.total_tokens == 80

    def test_char_diff_positive(self):
        """字符数变化（内容增长）"""
        optimizer = self._make_optimizer("优化后的更长文本内容")
        result = optimizer.optimize(content="短文本")
        assert result.char_diff == len("优化后的更长文本内容") - len("短文本")

    def test_char_diff_negative(self):
        """字符数变化（内容精简）"""
        optimizer = self._make_optimizer("短")
        result = optimizer.optimize(content="很长的原始文本内容")
        assert result.char_diff < 0

    def test_summary_contains_type(self):
        """摘要包含优化类型"""
        optimizer = self._make_optimizer("优化内容")
        result = optimizer.optimize(content="原始内容", optimize_type="polish")
        assert "润色优化" in result.summary

    def test_summary_contains_style(self):
        """摘要包含目标风格"""
        optimizer = self._make_optimizer("优化内容")
        result = optimizer.optimize(
            content="原始内容",
            target_style="formal",
        )
        assert "正式书面" in result.summary

    def test_model_info(self):
        """模型信息记录"""
        optimizer = self._make_optimizer("优化内容")
        result = optimizer.optimize(content="原始内容")
        assert result.model == "test-model"

    def test_temperature_override(self):
        """温度参数覆盖"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("结果")
        optimizer = ContentOptimizer(llm_client=mock_llm, temperature=0.3)
        optimizer.optimize(content="内容", temperature=0.8)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.8

    def test_temperature_default(self):
        """温度参数使用默认值"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("结果")
        optimizer = ContentOptimizer(llm_client=mock_llm, temperature=0.3)
        optimizer.optimize(content="内容")
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.3

    def test_max_tokens_override(self):
        """max_tokens 参数覆盖"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("结果")
        optimizer = ContentOptimizer(llm_client=mock_llm, max_tokens=500)
        optimizer.optimize(content="内容", max_tokens=1000)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 1000


# ------------------------------------------------------------------ #
#                    参数校验测试                                     #
# ------------------------------------------------------------------ #

class TestOptimizeValidation:
    """参数校验测试"""

    def _make_optimizer(self):
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("结果")
        return ContentOptimizer(llm_client=mock_llm)

    def test_empty_content_raises(self):
        """空内容抛出 ValueError"""
        optimizer = self._make_optimizer()
        with pytest.raises(ValueError, match="待优化内容不能为空"):
            optimizer.optimize(content="")

    def test_whitespace_content_raises(self):
        """纯空格内容抛出 ValueError"""
        optimizer = self._make_optimizer()
        with pytest.raises(ValueError, match="待优化内容不能为空"):
            optimizer.optimize(content="   ")

    def test_none_content_raises(self):
        """None 内容抛出 ValueError"""
        optimizer = self._make_optimizer()
        with pytest.raises(ValueError):
            optimizer.optimize(content=None)

    def test_invalid_optimize_type_raises(self):
        """不支持的优化类型抛出 ValueError"""
        optimizer = self._make_optimizer()
        with pytest.raises(ValueError, match="不支持的优化类型"):
            optimizer.optimize(content="内容", optimize_type="invalid_type")

    def test_valid_types_accepted(self):
        """所有支持的优化类型都能通过校验"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("结果")
        optimizer = ContentOptimizer(llm_client=mock_llm)
        for opt_type in ContentOptimizer.SUPPORTED_TYPES:
            result = optimizer.optimize(content="测试内容", optimize_type=opt_type)
            assert result.optimize_type == opt_type


# ------------------------------------------------------------------ #
#                    内部方法测试                                     #
# ------------------------------------------------------------------ #

class TestBuildMessages:
    """_build_messages 方法测试"""

    def _make_optimizer(self):
        mock_llm = _make_mock_llm()
        return ContentOptimizer(llm_client=mock_llm)

    def test_message_structure(self):
        """消息列表包含 system 和 user 两条消息"""
        optimizer = self._make_optimizer()
        messages = optimizer._build_messages(
            content="测试内容",
            optimize_type="润色优化",
            target_style="",
            focus_areas="",
            requirements="",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_contains_role(self):
        """系统提示词包含编辑角色"""
        optimizer = self._make_optimizer()
        messages = optimizer._build_messages(
            content="测试内容",
            optimize_type="润色优化",
            target_style="",
            focus_areas="",
            requirements="",
        )
        system = messages[0]["content"]
        assert len(system) > 0  # 系统提示词非空

    def test_user_content_contains_original(self):
        """用户消息包含原始内容"""
        optimizer = self._make_optimizer()
        messages = optimizer._build_messages(
            content="这是原始文本",
            optimize_type="润色优化",
            target_style="",
            focus_areas="",
            requirements="",
        )
        assert "这是原始文本" in messages[1]["content"]

    def test_user_content_contains_style(self):
        """用户消息包含目标风格"""
        optimizer = self._make_optimizer()
        messages = optimizer._build_messages(
            content="内容",
            optimize_type="润色优化",
            target_style="正式书面",
            focus_areas="",
            requirements="",
        )
        assert "正式书面" in messages[1]["content"]

    def test_user_content_contains_type(self):
        """用户消息包含优化类型"""
        optimizer = self._make_optimizer()
        messages = optimizer._build_messages(
            content="内容",
            optimize_type="简化改写",
            target_style="",
            focus_areas="",
            requirements="",
        )
        assert "简化改写" in messages[1]["content"]


class TestGetSystemPrompt:
    """_get_system_prompt 方法测试"""

    def _make_optimizer(self):
        mock_llm = _make_mock_llm()
        return ContentOptimizer(llm_client=mock_llm)

    def test_polish_prompt(self):
        """润色类型的提示词包含润色指令"""
        optimizer = self._make_optimizer()
        prompt = optimizer._get_system_prompt("polish")
        assert "润色" in prompt or len(prompt) > 0

    def test_simplify_prompt(self):
        """简化类型的提示词"""
        optimizer = self._make_optimizer()
        prompt = optimizer._get_system_prompt("simplify")
        assert "简化" in prompt or len(prompt) > 0

    def test_expand_prompt(self):
        """扩写类型的提示词"""
        optimizer = self._make_optimizer()
        prompt = optimizer._get_system_prompt("expand")
        assert "扩写" in prompt or len(prompt) > 0

    def test_shorten_prompt(self):
        """缩写类型的提示词"""
        optimizer = self._make_optimizer()
        prompt = optimizer._get_system_prompt("shorten")
        assert "精简" in prompt or len(prompt) > 0

    def test_grammar_prompt(self):
        """语法校对类型的提示词"""
        optimizer = self._make_optimizer()
        prompt = optimizer._get_system_prompt("grammar")
        assert "语法" in prompt or len(prompt) > 0

    def test_prompt_fallback_on_error(self):
        """Prompt 加载失败时使用兜底"""
        optimizer = self._make_optimizer()
        with patch("src.writing.content_optimizer.get_prompt", side_effect=Exception("mock")):
            prompt = optimizer._get_system_prompt("polish")
            assert len(prompt) > 0


class TestBuildSummary:
    """_build_summary 方法测试"""

    def test_basic_summary(self):
        """基本摘要包含类型和变化"""
        summary = ContentOptimizer._build_summary("polish", "", 5, 100)
        assert "润色优化" in summary
        assert "+5" in summary

    def test_summary_with_style(self):
        """带风格的摘要"""
        summary = ContentOptimizer._build_summary("polish", "formal", 5, 100)
        assert "正式书面" in summary

    def test_summary_negative_diff(self):
        """精简时的摘要"""
        summary = ContentOptimizer._build_summary("shorten", "", -20, 100)
        assert "-20" in summary

    def test_summary_ratio(self):
        """摘要包含变化比例"""
        summary = ContentOptimizer._build_summary("expand", "", 30, 100)
        assert "+30.0%" in summary


# ------------------------------------------------------------------ #
#                    LLM 调用异常测试                                 #
# ------------------------------------------------------------------ #

class TestLLMCallErrors:
    """LLM 调用异常场景测试"""

    def test_api_error_propagated(self):
        """API 调用异常向上抛出"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.side_effect = ModelAPIError("API 调用失败")
        optimizer = ContentOptimizer(llm_client=mock_llm)
        with pytest.raises(ModelAPIError):
            optimizer.optimize(content="测试内容")

    def test_usage_none(self):
        """usage 为 None 时 token 数为 0"""
        mock_llm = _make_mock_llm()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "优化后"
        mock_response.usage = None
        mock_llm.chat_completion.return_value = mock_response

        optimizer = ContentOptimizer(llm_client=mock_llm)
        result = optimizer.optimize(content="原始内容")
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_empty_response_content(self):
        """模型返回空内容"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.return_value = _make_mock_response("")
        optimizer = ContentOptimizer(llm_client=mock_llm)
        result = optimizer.optimize(content="原始内容")
        assert result.optimized_content == ""


# ------------------------------------------------------------------ #
#                    映射常量测试                                     #
# ------------------------------------------------------------------ #

class TestMappings:
    """优化类型和风格映射测试"""

    def test_optimize_type_names(self):
        """优化类型映射完整"""
        expected_types = {"polish", "simplify", "expand", "shorten", "grammar"}
        assert set(_OPTIMIZE_TYPE_NAMES.keys()) == expected_types

    def test_target_style_names(self):
        """目标风格映射完整"""
        expected_styles = {"formal", "casual",
                           "academic", "professional", "creative"}
        assert set(_TARGET_STYLE_NAMES.keys()) == expected_styles

    def test_all_types_in_supported(self):
        """映射中的类型都在 SUPPORTED_TYPES 中"""
        for t in _OPTIMIZE_TYPE_NAMES:
            assert t in ContentOptimizer.SUPPORTED_TYPES


# ------------------------------------------------------------------ #
#                    全局单例测试                                     #
# ------------------------------------------------------------------ #

class TestGetContentOptimizer:
    """get_content_optimizer 全局单例测试"""

    def test_returns_instance(self):
        """返回 ContentOptimizer 实例"""
        import src.writing.content_optimizer as mod
        mod._content_optimizer_instance = None  # 重置单例
        optimizer = get_content_optimizer()
        assert isinstance(optimizer, ContentOptimizer)

    def test_singleton(self):
        """多次调用返回同一实例"""
        import src.writing.content_optimizer as mod
        mod._content_optimizer_instance = None
        opt1 = get_content_optimizer()
        opt2 = get_content_optimizer()
        assert opt1 is opt2

    def test_singleton_reset(self):
        """重置后创建新实例"""
        import src.writing.content_optimizer as mod
        mod._content_optimizer_instance = None
        opt1 = get_content_optimizer()
        mod._content_optimizer_instance = None
        opt2 = get_content_optimizer()
        assert opt1 is not opt2
