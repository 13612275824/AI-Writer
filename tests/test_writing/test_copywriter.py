# ============================================================================
# 文案写作模块测试  |  tests/test_writing/test_copywriter.py
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
from src.writing.copywriter import (
    Copywriter,
    CopywriterResult,
    get_copywriter,
    _COPY_TYPE_NAMES,
    _BRAND_TONE_NAMES,
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
#                    CopywriterResult 测试                            #
# ------------------------------------------------------------------ #

class TestCopywriterResult:
    """CopywriterResult 数据类测试"""

    def test_default_values(self):
        """默认值初始化"""
        result = CopywriterResult()
        assert result.copy == ""
        assert result.copy_type == ""
        assert result.product_name == ""
        assert result.target_audience == ""
        assert result.brand_tone == ""
        assert result.elapsed_ms == 0.0
        assert result.model == ""
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_total_tokens(self):
        """总 token 数计算"""
        result = CopywriterResult(prompt_tokens=80, completion_tokens=40)
        assert result.total_tokens == 120

    def test_total_tokens_zero(self):
        """无 token 信息时总数为 0"""
        result = CopywriterResult()
        assert result.total_tokens == 0

    def test_repr_short_copy(self):
        """短文案的 repr"""
        result = CopywriterResult(
            copy="探索无限可能",
            product_name="智能手表",
            copy_type="slogan",
            elapsed_ms=150.0,
        )
        r = repr(result)
        assert "product='智能手表'" in r
        assert "type='slogan'" in r
        assert "copy_len=6" in r
        assert "elapsed=150.0ms" in r

    def test_repr_long_copy(self):
        """长文案的 repr"""
        result = CopywriterResult(copy="x" * 200, product_name="产品")
        r = repr(result)
        assert "copy_len=200" in r

    def test_repr_empty(self):
        """空文案的 repr"""
        result = CopywriterResult()
        r = repr(result)
        assert "copy_len=0" in r

    def test_custom_values(self):
        """自定义值初始化"""
        result = CopywriterResult(
            copy="限时特惠，不容错过",
            copy_type="ad_copy",
            product_name="新款手机",
            target_audience="年轻白领",
            brand_tone="urgent",
            elapsed_ms=300.0,
            model="gpt-4",
            prompt_tokens=150,
            completion_tokens=200,
        )
        assert result.copy == "限时特惠，不容错过"
        assert result.copy_type == "ad_copy"
        assert result.product_name == "新款手机"
        assert result.target_audience == "年轻白领"
        assert result.brand_tone == "urgent"
        assert result.elapsed_ms == 300.0
        assert result.model == "gpt-4"
        assert result.prompt_tokens == 150
        assert result.completion_tokens == 200
        assert result.total_tokens == 350


# ------------------------------------------------------------------ #
#                    映射常量测试                                     #
# ------------------------------------------------------------------ #

class TestMappingConstants:
    """文案类型和调性映射常量测试"""

    def test_copy_type_names_keys(self):
        """文案类型映射包含所有预定义类型"""
        expected = {
            "slogan", "product_description", "ad_copy",
            "social_media", "landing_page", "email",
        }
        assert set(_COPY_TYPE_NAMES.keys()) == expected

    def test_copy_type_names_values(self):
        """文案类型映射值非空"""
        for v in _COPY_TYPE_NAMES.values():
            assert isinstance(v, str)
            assert len(v) > 0

    def test_brand_tone_names_keys(self):
        """品牌调性映射包含所有预定义类型"""
        expected = {
            "professional", "casual", "luxurious",
            "playful", "urgent", "inspirational",
        }
        assert set(_BRAND_TONE_NAMES.keys()) == expected

    def test_brand_tone_names_values(self):
        """品牌调性映射值非空"""
        for v in _BRAND_TONE_NAMES.values():
            assert isinstance(v, str)
            assert len(v) > 0


# ------------------------------------------------------------------ #
#                    Copywriter 初始化测试                             #
# ------------------------------------------------------------------ #

class TestCopywriterInit:
    """Copywriter 初始化测试"""

    def test_init_with_mock_client(self):
        """使用 mock 客户端初始化"""
        mock_llm = _make_mock_llm("my-model")
        writer = Copywriter(llm_client=mock_llm)
        assert writer._llm is mock_llm
        assert writer._temperature is None
        assert writer._max_tokens is None

    def test_init_with_custom_params(self):
        """使用自定义参数初始化"""
        mock_llm = _make_mock_llm()
        writer = Copywriter(
            llm_client=mock_llm,
            temperature=0.9,
            max_tokens=500,
        )
        assert writer._temperature == 0.9
        assert writer._max_tokens == 500

    def test_init_default_client(self):
        """默认客户端初始化"""
        with patch("src.writing.copywriter.get_llm_client") as mock_get:
            mock_get.return_value = _make_mock_llm()
            writer = Copywriter()
            mock_get.assert_called_once()

    def test_llm_client_property(self):
        """llm_client 属性访问"""
        mock_llm = _make_mock_llm("prop-model")
        writer = Copywriter(llm_client=mock_llm)
        assert writer.llm_client.model == "prop-model"

    def test_repr(self):
        """__repr__ 输出"""
        mock_llm = _make_mock_llm("test-model")
        writer = Copywriter(llm_client=mock_llm)
        assert "test-model" in repr(writer)

    def test_prompt_role(self):
        """PROMPT_ROLE 类属性"""
        assert Copywriter.PROMPT_ROLE == "copywriting"


# ------------------------------------------------------------------ #
#                    write() 核心方法测试                             #
# ------------------------------------------------------------------ #

class TestCopywriterWrite:
    """write() 方法测试"""

    def _make_writer(self, response_content="这是生成的文案内容"):
        """创建带 mock 的 writer"""
        mock_llm = _make_mock_llm("test-model")
        mock_response = _make_mock_response(
            response_content,
            prompt_tokens=60,
            completion_tokens=80,
        )
        mock_llm.chat_completion.return_value = mock_response
        return Copywriter(llm_client=mock_llm)

    def test_basic_write(self):
        """基本文案写作"""
        writer = self._make_writer()
        result = writer.write(product_name="智能手表")

        assert isinstance(result, CopywriterResult)
        assert result.copy == "这是生成的文案内容"
        assert result.product_name == "智能手表"
        assert result.copy_type == "product_description"
        assert result.prompt_tokens == 60
        assert result.completion_tokens == 80
        assert result.total_tokens == 140
        assert result.elapsed_ms > 0

    def test_write_with_slogan_type(self):
        """指定 slogan 类型"""
        writer = self._make_writer("探索无限可能")
        result = writer.write(product_name="智能手表", copy_type="slogan")
        assert result.copy_type == "slogan"
        assert result.copy == "探索无限可能"

    def test_write_with_all_params(self):
        """所有参数"""
        writer = self._make_writer()
        result = writer.write(
            product_name="智能手表",
            copy_type="ad_copy",
            target_audience="年轻运动爱好者",
            brand_tone="playful",
            key_selling_points="超长续航，精准心率监测",
            requirements="突出性价比",
        )
        assert result.product_name == "智能手表"
        assert result.copy_type == "ad_copy"
        assert result.target_audience == "年轻运动爱好者"
        assert result.brand_tone == "playful"

    def test_write_temperature_override(self):
        """write 中 temperature 覆盖实例默认值"""
        writer = self._make_writer()
        writer._temperature = 0.5
        writer.write(product_name="产品", temperature=0.9)

        call_kwargs = writer._llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    def test_write_max_tokens_override(self):
        """write 中 max_tokens 覆盖实例默认值"""
        writer = self._make_writer()
        writer._max_tokens = 1000
        writer.write(product_name="产品", max_tokens=500)

        call_kwargs = writer._llm.chat_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 500

    def test_write_uses_instance_defaults(self):
        """不传参数时使用实例默认值"""
        writer = self._make_writer()
        writer._temperature = 0.8
        writer._max_tokens = 800
        writer.write(product_name="产品")

        call_kwargs = writer._llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 800

    def test_write_no_optional_params(self):
        """不传可选参数时 kwargs 不包含 temperature/max_tokens"""
        writer = self._make_writer()
        writer.write(product_name="产品")

        call_kwargs = writer._llm.chat_completion.call_args[1]
        assert "temperature" not in call_kwargs
        assert "max_tokens" not in call_kwargs

    def test_write_empty_product_raises(self):
        """空产品名抛出 ValueError"""
        writer = self._make_writer()
        with pytest.raises(ValueError, match="产品/品牌名称不能为空"):
            writer.write(product_name="")

    def test_write_whitespace_product_raises(self):
        """纯空格产品名抛出 ValueError"""
        writer = self._make_writer()
        with pytest.raises(ValueError, match="产品/品牌名称不能为空"):
            writer.write(product_name="   ")

    def test_write_none_product_raises(self):
        """None 产品名抛出 ValueError"""
        writer = self._make_writer()
        with pytest.raises(ValueError, match="产品/品牌名称不能为空"):
            writer.write(product_name=None)

    def test_write_model_name_in_result(self):
        """结果中包含模型名称"""
        mock_llm = _make_mock_llm("special-model")
        mock_response = _make_mock_response("文案")
        mock_llm.chat_completion.return_value = mock_response
        writer = Copywriter(llm_client=mock_llm)

        result = writer.write(product_name="产品")
        assert result.model == "special-model"

    def test_write_api_error_propagated(self):
        """API 错误向上传播"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.side_effect = ModelAPIError("API 调用失败")
        writer = Copywriter(llm_client=mock_llm)

        with pytest.raises(ModelAPIError):
            writer.write(product_name="产品")

    def test_write_default_copy_type(self):
        """默认 copy_type 是 product_description"""
        writer = self._make_writer()
        result = writer.write(product_name="产品")
        assert result.copy_type == "product_description"


# ------------------------------------------------------------------ #
#                    _build_messages 测试                             #
# ------------------------------------------------------------------ #

class TestCopywriterBuildMessages:
    """_build_messages 方法测试"""

    def _make_writer(self):
        mock_llm = _make_mock_llm()
        return Copywriter(llm_client=mock_llm)

    def test_messages_structure(self):
        """消息列表包含 system 和 user"""
        writer = self._make_writer()
        messages = writer._build_messages(
            product_name="智能手表",
            copy_type="slogan",
            target_audience="年轻人",
            brand_tone="playful",
            key_selling_points="时尚外观",
            requirements="简洁有力",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_content_includes_all_params(self):
        """用户消息包含所有参数"""
        writer = self._make_writer()
        messages = writer._build_messages(
            product_name="智能手表",
            copy_type="ad_copy",
            target_audience="运动爱好者",
            brand_tone="urgent",
            key_selling_points="超长续航",
            requirements="突出性价比",
        )
        user_text = messages[1]["content"]
        assert "智能手表" in user_text
        assert "运动爱好者" in user_text
        assert "超长续航" in user_text
        assert "突出性价比" in user_text

    def test_copy_type_translated(self):
        """文案类型被翻译为中文"""
        writer = self._make_writer()
        messages = writer._build_messages(
            product_name="产品",
            copy_type="slogan",
            target_audience="",
            brand_tone="",
            key_selling_points="",
            requirements="",
        )
        user_text = messages[1]["content"]
        assert "广告语/口号" in user_text

    def test_brand_tone_translated(self):
        """品牌调性被翻译为中文"""
        writer = self._make_writer()
        messages = writer._build_messages(
            product_name="产品",
            copy_type="product_description",
            target_audience="",
            brand_tone="luxurious",
            key_selling_points="",
            requirements="",
        )
        user_text = messages[1]["content"]
        assert "奢华高端" in user_text

    def test_empty_audience_fallback(self):
        """空受众使用默认文本"""
        writer = self._make_writer()
        messages = writer._build_messages(
            product_name="产品",
            copy_type="product_description",
            target_audience="",
            brand_tone="",
            key_selling_points="",
            requirements="",
        )
        user_text = messages[1]["content"]
        assert "通用受众" in user_text
        assert "无特殊要求" in user_text
        assert "未指定" in user_text
        assert "无额外要求" in user_text

    def test_system_prompt_from_config(self):
        """system 提示词来自配置"""
        writer = self._make_writer()
        with patch("src.writing.copywriter.get_prompt") as mock_get:
            mock_get.return_value = {"system": "专业文案创作者"}
            messages = writer._build_messages(
                product_name="产品",
                copy_type="slogan",
                target_audience="",
                brand_tone="",
                key_selling_points="",
                requirements="",
            )
        assert messages[0]["content"] == "专业文案创作者"


# ------------------------------------------------------------------ #
#                    _build_user_content 测试                         #
# ------------------------------------------------------------------ #

class TestCopywriterBuildUserContent:
    """_build_user_content 方法测试"""

    def _make_writer(self):
        mock_llm = _make_mock_llm()
        return Copywriter(llm_client=mock_llm)

    def test_basic_content(self):
        """基本用户内容构建"""
        writer = self._make_writer()
        content = writer._build_user_content(
            product_name="智能手表",
            copy_type="产品描述",
            target_audience="年轻白领",
            brand_tone="专业商务",
            key_selling_points="超长续航",
            requirements="简洁明了",
        )
        assert "智能手表" in content
        assert "产品描述" in content
        assert "年轻白领" in content
        assert "专业商务" in content
        assert "超长续航" in content
        assert "简洁明了" in content

    def test_content_starts_with_request(self):
        """内容以写作请求开头"""
        writer = self._make_writer()
        content = writer._build_user_content(
            product_name="产品",
            copy_type="广告文案",
            target_audience="受众",
            brand_tone="调性",
            key_selling_points="卖点",
            requirements="要求",
        )
        assert content.startswith("请为")


# ------------------------------------------------------------------ #
#                    _get_system_prompt 测试                          #
# ------------------------------------------------------------------ #

class TestCopywriterGetSystemPrompt:
    """_get_system_prompt 方法测试"""

    def _make_writer(self):
        mock_llm = _make_mock_llm()
        return Copywriter(llm_client=mock_llm)

    def test_get_system_prompt_success(self):
        """正常获取系统提示词"""
        writer = self._make_writer()
        with patch("src.writing.copywriter.get_prompt") as mock_get:
            mock_get.return_value = {"system": "专业文案创作者"}
            result = writer._get_system_prompt()
        assert result == "专业文案创作者"

    def test_get_system_prompt_empty_system(self):
        """system 为空时返回空字符串"""
        writer = self._make_writer()
        with patch("src.writing.copywriter.get_prompt") as mock_get:
            mock_get.return_value = {"system": ""}
            result = writer._get_system_prompt()
        assert result == ""

    def test_get_system_prompt_no_system_key(self):
        """缺少 system 键时返回空字符串"""
        writer = self._make_writer()
        with patch("src.writing.copywriter.get_prompt") as mock_get:
            mock_get.return_value = {}
            result = writer._get_system_prompt()
        assert result == ""

    def test_get_system_prompt_exception(self):
        """get_prompt 抛异常时返回默认值"""
        writer = self._make_writer()
        with patch("src.writing.copywriter.get_prompt") as mock_get:
            mock_get.side_effect = Exception("加载失败")
            result = writer._get_system_prompt()
        assert "文案" in result

    def test_get_system_prompt_uses_correct_role(self):
        """使用正确的 PROMPT_ROLE"""
        writer = self._make_writer()
        with patch("src.writing.copywriter.get_prompt") as mock_get:
            mock_get.return_value = {"system": "test"}
            writer._get_system_prompt()
            mock_get.assert_called_once_with("copywriting")


# ------------------------------------------------------------------ #
#                    _call_llm 测试                                   #
# ------------------------------------------------------------------ #

class TestCopywriterCallLLM:
    """_call_llm 方法测试"""

    def _make_writer_with_response(self, content="文案", pt=10, ct=20):
        mock_llm = _make_mock_llm()
        mock_response = _make_mock_response(content, pt, ct)
        mock_llm.chat_completion.return_value = mock_response
        return Copywriter(llm_client=mock_llm), mock_llm

    def test_basic_call(self):
        """基本 LLM 调用"""
        writer, mock_llm = self._make_writer_with_response("生成的文案", 50, 100)
        messages = [{"role": "system", "content": "test"},
                    {"role": "user", "content": "test"}]

        copy_text, pt, ct = writer._call_llm(messages)
        assert copy_text == "生成的文案"
        assert pt == 50
        assert ct == 100

    def test_call_with_temperature(self):
        """带 temperature 参数"""
        writer, mock_llm = self._make_writer_with_response()
        messages = [{"role": "user", "content": "test"}]

        writer._call_llm(messages, temperature=0.9)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    def test_call_with_max_tokens(self):
        """带 max_tokens 参数"""
        writer, mock_llm = self._make_writer_with_response()
        messages = [{"role": "user", "content": "test"}]

        writer._call_llm(messages, max_tokens=300)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 300

    def test_call_without_optional_params(self):
        """不传可选参数时 kwargs 为空"""
        writer, mock_llm = self._make_writer_with_response()
        messages = [{"role": "user", "content": "test"}]

        writer._call_llm(messages)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs == {}

    def test_call_no_usage(self):
        """响应没有 usage 时 token 数为 0"""
        mock_llm = _make_mock_llm()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "文案"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_llm.chat_completion.return_value = mock_response

        writer = Copywriter(llm_client=mock_llm)
        messages = [{"role": "user", "content": "test"}]
        copy_text, pt, ct = writer._call_llm(messages)

        assert copy_text == "文案"
        assert pt == 0
        assert ct == 0

    def test_call_none_content(self):
        """message.content 为 None 时返回空字符串"""
        mock_llm = _make_mock_llm()
        mock_response = _make_mock_response(None)
        mock_llm.chat_completion.return_value = mock_response

        writer = Copywriter(llm_client=mock_llm)
        messages = [{"role": "user", "content": "test"}]
        copy_text, _, _ = writer._call_llm(messages)
        assert copy_text == ""

    def test_call_api_error(self):
        """API 错误直接传播"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.side_effect = ModelAPIError("API 失败")

        writer = Copywriter(llm_client=mock_llm)
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ModelAPIError):
            writer._call_llm(messages)


# ------------------------------------------------------------------ #
#                    全局单例函数测试                                  #
# ------------------------------------------------------------------ #

class TestGetCopywriter:
    """get_copywriter() 单例函数测试"""

    def setup_method(self):
        """每个测试前重置单例"""
        import src.writing.copywriter as mod
        mod._copywriter_instance = None

    def test_returns_copywriter(self):
        """返回 Copywriter 实例"""
        with patch("src.writing.copywriter.get_llm_client") as mock_get:
            mock_get.return_value = _make_mock_llm()
            writer = get_copywriter()
            assert isinstance(writer, Copywriter)

    def test_singleton_same_instance(self):
        """多次调用返回同一实例"""
        with patch("src.writing.copywriter.get_llm_client") as mock_get:
            mock_get.return_value = _make_mock_llm()
            w1 = get_copywriter()
            w2 = get_copywriter()
            assert w1 is w2

    def test_singleton_only_creates_once(self):
        """get_llm_client 只调用一次"""
        with patch("src.writing.copywriter.get_llm_client") as mock_get:
            mock_get.return_value = _make_mock_llm()
            get_copywriter()
            get_copywriter()
            mock_get.assert_called_once()


# ------------------------------------------------------------------ #
#                    集成测试                                          #
# ------------------------------------------------------------------ #

class TestCopywriterIntegration:
    """端到端集成测试（使用 mock LLM）"""

    def test_full_flow_slogan(self):
        """完整 slogan 生成流程"""
        mock_llm = _make_mock_llm("gpt-4")
        mock_response = _make_mock_response(
            "探索无限可能，从手腕开始",
            prompt_tokens=80,
            completion_tokens=30,
        )
        mock_llm.chat_completion.return_value = mock_response

        writer = Copywriter(llm_client=mock_llm,
                            temperature=0.9, max_tokens=200)
        result = writer.write(
            product_name="智能手表",
            copy_type="slogan",
            target_audience="年轻运动爱好者",
            brand_tone="inspirational",
            key_selling_points="超长续航，精准心率监测",
        )

        assert result.product_name == "智能手表"
        assert result.copy_type == "slogan"
        assert result.copy == "探索无限可能，从手腕开始"
        assert result.target_audience == "年轻运动爱好者"
        assert result.brand_tone == "inspirational"
        assert result.model == "gpt-4"
        assert result.prompt_tokens == 80
        assert result.completion_tokens == 30
        assert result.total_tokens == 110
        assert result.elapsed_ms > 0

        # 验证 LLM 被正确调用
        mock_llm.chat_completion.assert_called_once()
        messages = mock_llm.chat_completion.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "智能手表" in messages[1]["content"]
        assert "广告语/口号" in messages[1]["content"]

    def test_full_flow_product_description(self):
        """完整产品描述生成流程"""
        mock_llm = _make_mock_llm()
        mock_response = _make_mock_response(
            "这款智能手表采用先进的传感器技术...",
        )
        mock_llm.chat_completion.return_value = mock_response

        writer = Copywriter(llm_client=mock_llm)
        result = writer.write(product_name="智能手表")

        assert result.copy_type == "product_description"
        assert "智能手表" in result.copy

    def test_minimal_flow(self):
        """最小参数写作流程"""
        mock_llm = _make_mock_llm()
        mock_response = _make_mock_response("简单文案")
        mock_llm.chat_completion.return_value = mock_response

        writer = Copywriter(llm_client=mock_llm)
        result = writer.write(product_name="产品")

        assert result.product_name == "产品"
        assert result.copy == "简单文案"


# ------------------------------------------------------------------ #
#                    入口                                              #
# ------------------------------------------------------------------ #

def main():
    """运行全部测试"""
    print("=" * 70)
    print("文案写作模块测试")
    print("=" * 70)
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
