# ============================================================================
# 文章写作模块测试  |  tests/test_writing/test_article_writer.py
# ============================================================================

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure Worker2/ is on sys.path so `src` package can be found
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.exceptions import ModelAPIError
from src.writing.article_writer import (
    ArticleResult,
    ArticleWriter,
    get_article_writer,
)

# ── 确保项目根目录在 sys.path 中 ──────────────────────────────
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# --------------------------------------------------------6---------- #
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
#                    ArticleResult 测试                               #
# ------------------------------------------------------------------ #

class TestArticleResult:
    """ArticleResult 数据类测试"""

    def test_default_values(self):
        """默认值初始化"""
        result = ArticleResult()
        assert result.article == ""
        assert result.outline == ""
        assert result.topic == ""
        assert result.style == ""
        assert result.elapsed_ms == 0.0
        assert result.model == ""
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_total_tokens(self):
        """总 token 数计算"""
        result = ArticleResult(prompt_tokens=100, completion_tokens=50)
        assert result.total_tokens == 150

    def test_total_tokens_zero(self):
        """无 token 信息时总数为 0"""
        result = ArticleResult()
        assert result.total_tokens == 0

    def test_repr_short_article(self):
        """短文章的 repr"""
        result = ArticleResult(article="短文章", topic="AI", elapsed_ms=100.0)
        r = repr(result)
        assert "topic='AI'" in r
        assert "article_len=3" in r
        assert "elapsed=100.0ms" in r

    def test_repr_long_article(self):
        """长文章的 repr（截断显示）"""
        result = ArticleResult(article="x" * 200, topic="AI")
        r = repr(result)
        assert "article_len=200" in r

    def test_repr_empty(self):
        """空文章的 repr"""
        result = ArticleResult()
        r = repr(result)
        assert "article_len=0" in r

    def test_custom_values(self):
        """自定义值初始化"""
        result = ArticleResult(
            article="AI文章",
            outline="1. 引言\n2. 正文",
            topic="AI",
            style="学术",
            elapsed_ms=500.5,
            model="gpt-4",
            prompt_tokens=200,
            completion_tokens=300,
        )
        assert result.article == "AI文章"
        assert result.outline == "1. 引言\n2. 正文"
        assert result.topic == "AI"
        assert result.style == "学术"
        assert result.elapsed_ms == 500.5
        assert result.model == "gpt-4"
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 300
        assert result.total_tokens == 500


# ------------------------------------------------------------------ #
#                    ArticleWriter 初始化测试                         #
# ------------------------------------------------------------------ #

class TestArticleWriterInit:
    """ArticleWriter 初始化测试"""

    def test_init_with_mock_client(self):
        """使用 mock 客户端初始化"""
        mock_llm = _make_mock_llm("my-model")
        writer = ArticleWriter(llm_client=mock_llm)
        assert writer._llm is mock_llm
        assert writer._temperature is None
        assert writer._max_tokens is None

    def test_init_with_custom_params(self):
        """使用自定义参数初始化"""
        mock_llm = _make_mock_llm()
        writer = ArticleWriter(
            llm_client=mock_llm,
            temperature=0.8,
            max_tokens=2000,
        )
        assert writer._temperature == 0.8
        assert writer._max_tokens == 2000

    def test_init_default_client(self):
        """默认客户端初始化（使用全局单例）"""
        with patch("src.writing.article_writer.get_llm_client") as mock_get:
            mock_get.return_value = _make_mock_llm()
            writer = ArticleWriter()
            mock_get.assert_called_once()

    def test_llm_client_property(self):
        """llm_client 属性访问"""
        mock_llm = _make_mock_llm("prop-model")
        writer = ArticleWriter(llm_client=mock_llm)
        assert writer.llm_client.model == "prop-model"

    def test_repr(self):
        """__repr__ 输出"""
        mock_llm = _make_mock_llm("test-model")
        writer = ArticleWriter(llm_client=mock_llm)
        assert "test-model" in repr(writer)

    def test_prompt_role(self):
        """PROMPT_ROLE 类属性"""
        assert ArticleWriter.PROMPT_ROLE == "writing"


# ------------------------------------------------------------------ #
#                    write() 核心方法测试                             #
# ------------------------------------------------------------------ #

class TestArticleWriterWrite:
    """write() 方法测试"""

    def _make_writer(self, response_content="这是生成的文章内容"):
        """创建带 mock 的 writer"""
        mock_llm = _make_mock_llm("test-model")
        mock_response = _make_mock_response(
            response_content,
            prompt_tokens=50,
            completion_tokens=100,
        )
        mock_llm.chat_completion.return_value = mock_response
        writer = ArticleWriter(llm_client=mock_llm)
        return writer

    def test_basic_write(self):
        """基本写作调用"""
        writer = self._make_writer()
        result = writer.write(topic="人工智能的发展")

        assert isinstance(result, ArticleResult)
        assert result.article == "这是生成的文章内容"
        assert result.topic == "人工智能的发展"
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 100
        assert result.total_tokens == 150
        assert result.elapsed_ms > 0

    def test_write_with_style(self):
        """指定写作风格"""
        writer = self._make_writer()
        result = writer.write(topic="量子计算", style="学术")
        assert result.style == "学术"

    def test_write_with_requirements(self):
        """指定额外要求"""
        writer = self._make_writer()
        result = writer.write(topic="AI", requirements="1000字以上")
        assert result.article != ""

    def test_write_with_outline(self):
        """指定大纲"""
        writer = self._make_writer()
        result = writer.write(
            topic="AI",
            outline="1. 引言\n2. 技术背景\n3. 应用场景\n4. 总结",
        )
        assert result.outline == "1. 引言\n2. 技术背景\n3. 应用场景\n4. 总结"

    def test_write_with_no_outline(self):
        """不指定大纲时默认空字符串"""
        writer = self._make_writer()
        result = writer.write(topic="AI")
        assert result.outline == ""

    def test_write_temperature_override(self):
        """write 中 temperature 覆盖实例默认值"""
        writer = self._make_writer()
        writer._temperature = 0.5
        writer.write(topic="AI", temperature=0.9)

        call_kwargs = writer._llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    def test_write_max_tokens_override(self):
        """write 中 max_tokens 覆盖实例默认值"""
        writer = self._make_writer()
        writer._max_tokens = 1000
        writer.write(topic="AI", max_tokens=2000)

        call_kwargs = writer._llm.chat_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 2000

    def test_write_uses_instance_defaults(self):
        """不传参数时使用实例默认值"""
        writer = self._make_writer()
        writer._temperature = 0.7
        writer._max_tokens = 1500
        writer.write(topic="AI")

        call_kwargs = writer._llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1500

    def test_write_no_optional_params(self):
        """不传可选参数时 kwargs 不包含 temperature/max_tokens"""
        writer = self._make_writer()
        writer.write(topic="AI")

        call_kwargs = writer._llm.chat_completion.call_args[1]
        assert "temperature" not in call_kwargs
        assert "max_tokens" not in call_kwargs

    def test_write_empty_topic_raises(self):
        """空主题抛出 ValueError"""
        writer = self._make_writer()
        with pytest.raises(ValueError, match="文章主题不能为空"):
            writer.write(topic="")

    def test_write_whitespace_topic_raises(self):
        """纯空格主题抛出 ValueError"""
        writer = self._make_writer()
        with pytest.raises(ValueError, match="文章主题不能为空"):
            writer.write(topic="   ")

    def test_write_none_topic_raises(self):
        """None 主题抛出 ValueError"""
        writer = self._make_writer()
        with pytest.raises(ValueError, match="文章主题不能为空"):
            writer.write(topic=None)

    def test_write_model_name_in_result(self):
        """结果中包含模型名称"""
        mock_llm = _make_mock_llm("my-special-model")
        mock_response = _make_mock_response("文章")
        mock_llm.chat_completion.return_value = mock_response
        writer = ArticleWriter(llm_client=mock_llm)

        result = writer.write(topic="AI")
        assert result.model == "my-special-model"

    def test_write_elapsed_time(self):
        """elapsed_ms 应该大于 0"""
        writer = self._make_writer()
        result = writer.write(topic="AI")
        assert result.elapsed_ms >= 0

    def test_write_api_error_propagated(self):
        """API 错误向上传播"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.side_effect = ModelAPIError("API 调用失败")
        writer = ArticleWriter(llm_client=mock_llm)

        with pytest.raises(ModelAPIError):
            writer.write(topic="AI")


# ------------------------------------------------------------------ #
#                    _build_messages 测试                             #
# ------------------------------------------------------------------ #

class TestArticleWriterBuildMessages:
    """_build_messages 方法测试"""

    def _make_writer(self):
        mock_llm = _make_mock_llm()
        return ArticleWriter(llm_client=mock_llm)

    def test_messages_structure(self):
        """消息列表包含 system 和 user"""
        writer = self._make_writer()
        messages = writer._build_messages(
            topic="AI", style="学术", requirements="详细", outline="1. 引言",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_content_includes_all_params(self):
        """用户消息包含所有参数"""
        writer = self._make_writer()
        messages = writer._build_messages(
            topic="量子计算",
            style="通俗",
            requirements="举例说明",
            outline="1. 概念\n2. 应用",
        )
        user_text = messages[1]["content"]
        assert "量子计算" in user_text
        assert "通俗" in user_text
        assert "举例说明" in user_text
        assert "概念" in user_text

    def test_empty_style_fallback(self):
        """空 style 使用默认文本"""
        writer = self._make_writer()
        messages = writer._build_messages(
            topic="AI", style="", requirements="", outline=None,
        )
        user_text = messages[1]["content"]
        assert "无特殊要求" in user_text
        assert "无额外要求" in user_text
        assert "无预定义大纲" in user_text

    def test_system_prompt_from_config(self):
        """system 提示词来自配置"""
        writer = self._make_writer()
        with patch("src.writing.article_writer.get_prompt") as mock_get:
            mock_get.return_value = {"system": "自定义系统提示"}
            messages = writer._build_messages(
                topic="AI", style="", requirements="", outline=None,
            )
        assert messages[0]["content"] == "自定义系统提示"

    def test_system_prompt_fallback(self):
        """配置加载失败时使用默认 system"""
        writer = self._make_writer()
        with patch("src.writing.article_writer.get_prompt") as mock_get:
            mock_get.side_effect = Exception("配置加载失败")
            messages = writer._build_messages(
                topic="AI", style="", requirements="", outline=None,
            )
        assert "写作助手" in messages[0]["content"]


# ------------------------------------------------------------------ #
#                    _build_user_content 测试                         #
# ------------------------------------------------------------------ #

class TestArticleWriterBuildUserContent:
    """_build_user_content 方法测试"""

    def _make_writer(self):
        mock_llm = _make_mock_llm()
        return ArticleWriter(llm_client=mock_llm)

    def test_basic_content(self):
        """基本用户内容构建"""
        writer = self._make_writer()
        content = writer._build_user_content(
            topic="AI", style="学术", requirements="详细分析", outline="1. 引言",
        )
        assert "AI" in content
        assert "学术" in content
        assert "详细分析" in content
        assert "1. 引言" in content

    def test_all_defaults(self):
        """所有默认值"""
        writer = self._make_writer()
        content = writer._build_user_content(
            topic="AI", style="无特殊要求", requirements="无额外要求",
            outline="无预定义大纲",
        )
        assert "AI" in content
        assert "无特殊要求" in content


# ------------------------------------------------------------------ #
#                    _get_system_prompt 测试                          #
# ------------------------------------------------------------------ #

class TestArticleWriterGetSystemPrompt:
    """_get_system_prompt 方法测试"""

    def _make_writer(self):
        mock_llm = _make_mock_llm()
        return ArticleWriter(llm_client=mock_llm)

    def test_get_system_prompt_success(self):
        """正常获取系统提示词"""
        writer = self._make_writer()
        with patch("src.writing.article_writer.get_prompt") as mock_get:
            mock_get.return_value = {"system": "专业的写作助手"}
            result = writer._get_system_prompt()
        assert result == "专业的写作助手"

    def test_get_system_prompt_empty_system(self):
        """system 为空时返回空字符串"""
        writer = self._make_writer()
        with patch("src.writing.article_writer.get_prompt") as mock_get:
            mock_get.return_value = {"system": ""}
            result = writer._get_system_prompt()
        assert result == ""

    def test_get_system_prompt_no_system_key(self):
        """缺少 system 键时返回默认值"""
        writer = self._make_writer()
        with patch("src.writing.article_writer.get_prompt") as mock_get:
            mock_get.return_value = {}
            result = writer._get_system_prompt()
        assert result == ""

    def test_get_system_prompt_exception(self):
        """get_prompt 抛异常时返回默认值"""
        writer = self._make_writer()
        with patch("src.writing.article_writer.get_prompt") as mock_get:
            mock_get.side_effect = Exception("加载失败")
            result = writer._get_system_prompt()
        assert "写作助手" in result

    def test_get_system_prompt_uses_correct_role(self):
        """使用正确的 PROMPT_ROLE"""
        writer = self._make_writer()
        with patch("src.writing.article_writer.get_prompt") as mock_get:
            mock_get.return_value = {"system": "test"}
            writer._get_system_prompt()
            mock_get.assert_called_once_with("writing")


# ------------------------------------------------------------------ #
#                    _call_llm 测试                                   #
# ------------------------------------------------------------------ #

class TestArticleWriterCallLLM:
    """_call_llm 方法测试"""

    def _make_writer_with_response(self, content="文章", pt=10, ct=20):
        mock_llm = _make_mock_llm()
        mock_response = _make_mock_response(content, pt, ct)
        mock_llm.chat_completion.return_value = mock_response
        return ArticleWriter(llm_client=mock_llm), mock_llm

    def test_basic_call(self):
        """基本 LLM 调用"""
        writer, mock_llm = self._make_writer_with_response("生成的文章", 50, 100)
        messages = [{"role": "system", "content": "test"},
                    {"role": "user", "content": "test"}]

        article, pt, ct = writer._call_llm(messages)
        assert article == "生成的文章"
        assert pt == 50
        assert ct == 100

    def test_call_with_temperature(self):
        """带 temperature 参数"""
        writer, mock_llm = self._make_writer_with_response()
        messages = [{"role": "user", "content": "test"}]

        writer._call_llm(messages, temperature=0.8)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.8

    def test_call_with_max_tokens(self):
        """带 max_tokens 参数"""
        writer, mock_llm = self._make_writer_with_response()
        messages = [{"role": "user", "content": "test"}]

        writer._call_llm(messages, max_tokens=500)
        call_kwargs = mock_llm.chat_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 500

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
        mock_choice.message.content = "文章"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_llm.chat_completion.return_value = mock_response

        writer = ArticleWriter(llm_client=mock_llm)
        messages = [{"role": "user", "content": "test"}]
        article, pt, ct = writer._call_llm(messages)

        assert article == "文章"
        assert pt == 0
        assert ct == 0

    def test_call_none_content(self):
        """message.content 为 None 时返回空字符串"""
        mock_llm = _make_mock_llm()
        mock_response = _make_mock_response(None)
        mock_llm.chat_completion.return_value = mock_response

        writer = ArticleWriter(llm_client=mock_llm)
        messages = [{"role": "user", "content": "test"}]
        article, _, _ = writer._call_llm(messages)
        assert article == ""

    def test_call_api_error(self):
        """API 错误直接传播"""
        mock_llm = _make_mock_llm()
        mock_llm.chat_completion.side_effect = ModelAPIError("API 失败")

        writer = ArticleWriter(llm_client=mock_llm)
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(ModelAPIError):
            writer._call_llm(messages)


# ------------------------------------------------------------------ #
#                    全局单例函数测试                                  #
# ------------------------------------------------------------------ #

class TestGetArticleWriter:
    """get_article_writer() 单例函数测试"""

    def setup_method(self):
        """每个测试前重置单例"""
        import src.writing.article_writer as mod
        mod._article_writer_instance = None

    def test_returns_writer(self):
        """返回 ArticleWriter 实例"""
        with patch("src.writing.article_writer.get_llm_client") as mock_get:
            mock_get.return_value = _make_mock_llm()
            writer = get_article_writer()
            assert isinstance(writer, ArticleWriter)

    def test_singleton_same_instance(self):
        """多次调用返回同一实例"""
        with patch("src.writing.article_writer.get_llm_client") as mock_get:
            mock_get.return_value = _make_mock_llm()
            w1 = get_article_writer()
            w2 = get_article_writer()
            assert w1 is w2

    def test_singleton_only_creates_once(self):
        """get_llm_client 只调用一次"""
        with patch("src.writing.article_writer.get_llm_client") as mock_get:
            mock_get.return_value = _make_mock_llm()
            get_article_writer()
            get_article_writer()
            mock_get.assert_called_once()


# ------------------------------------------------------------------ #
#                    集成测试                                          #
# ------------------------------------------------------------------ #

class TestArticleWriterIntegration:
    """端到端集成测试（使用 mock LLM）"""

    def test_full_flow(self):
        """完整写作流程"""
        mock_llm = _make_mock_llm("gpt-4")
        mock_response = _make_mock_response(
            "人工智能是当今最具变革性的技术之一...",
            prompt_tokens=100,
            completion_tokens=200,
        )
        mock_llm.chat_completion.return_value = mock_response

        writer = ArticleWriter(llm_client=mock_llm,
                               temperature=0.7, max_tokens=2000)
        result = writer.write(
            topic="人工智能的发展趋势",
            style="专业",
            requirements="重点分析2026年的发展方向",
            outline="1. 引言\n2. 当前现状\n3. 未来趋势\n4. 总结",
        )

        assert result.topic == "人工智能的发展趋势"
        assert result.style == "专业"
        assert result.article == "人工智能是当今最具变革性的技术之一..."
        assert result.outline == "1. 引言\n2. 当前现状\n3. 未来趋势\n4. 总结"
        assert result.model == "gpt-4"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 200
        assert result.total_tokens == 300
        assert result.elapsed_ms > 0

        # 验证 LLM 被正确调用
        mock_llm.chat_completion.assert_called_once()
        messages = mock_llm.chat_completion.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "人工智能的发展趋势" in messages[1]["content"]
        assert "专业" in messages[1]["content"]

    def test_minimal_flow(self):
        """最小参数写作流程"""
        mock_llm = _make_mock_llm()
        mock_response = _make_mock_response("一篇简单文章")
        mock_llm.chat_completion.return_value = mock_response

        writer = ArticleWriter(llm_client=mock_llm)
        result = writer.write(topic="简单主题")

        assert result.topic == "简单主题"
        assert result.article == "一篇简单文章"
        assert result.style == ""
        assert result.outline == ""


# ------------------------------------------------------------------ #
#                    入口                                              #
# ------------------------------------------------------------------ #

def main():
    """运行全部测试"""
    print("=" * 70)
    print("文章写作模块测试")
    print("=" * 70)
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
