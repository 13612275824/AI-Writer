"""测试 text_utils.py 文本处理工具模块"""

import io
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.text_utils import debug_print, format_text, safe_get

# 导入必须在路径设置之后


# ==================== debug_print 测试 ====================

class TestDebugPrint:
    """debug_print 函数测试"""

    def test_output_contains_message(self):
        """测试输出包含消息内容"""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            debug_print("测试消息")
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "测试消息" in output

    def test_output_contains_level(self):
        """测试输出包含默认级别 DEBUG"""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            debug_print("hello")
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "[DEBUG]" in output

    def test_custom_level(self):
        """测试自定义级别"""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            debug_print("info msg", level="INFO")
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "[INFO]" in output

    def test_output_contains_filename(self):
        """测试输出包含调用者文件名"""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            debug_print("check filename")
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "test_text_utils.py" in output

    def test_output_format(self):
        """测试输出格式 [file:line] [LEVEL] message"""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            debug_print("格式检查")
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue().strip()
        # 格式: [文件名:行号] [DEBUG] 格式检查
        assert output.startswith("[")
        assert "]" in output


# ==================== format_text 测试 ====================

class TestFormatText:
    """format_text 函数测试"""

    def test_empty_string(self):
        """测试空字符串"""
        assert format_text("") == ""

    def test_none_input(self):
        """测试 None 输入"""
        assert format_text(None) == ""

    def test_short_text_unchanged(self):
        """测试短文本不变"""
        text = "短文本"
        assert format_text(text, max_length=100) == "短文本"

    def test_exact_length_unchanged(self):
        """测试刚好等于最大长度的文本"""
        text = "a" * 10
        assert format_text(text, max_length=10) == "a" * 10

    def test_long_text_truncated(self):
        """测试长文本被截断"""
        text = "a" * 200
        result = format_text(text, max_length=100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_whitespace_stripped(self):
        """测试前后空白被去除"""
        text = "  hello  "
        assert format_text(text) == "hello"

    def test_custom_max_length(self):
        """测试自定义最大长度"""
        text = "abcdefghij"  # 10 chars
        result = format_text(text, max_length=5)
        assert result == "abcde..."

    def test_default_max_length(self):
        """测试默认最大长度 100"""
        text = "x" * 150
        result = format_text(text)
        assert result.endswith("...")
        assert len(result) == 103


# ==================== safe_get 测试 ====================

class TestSafeGet:
    """safe_get 函数测试"""

    def test_simple_key(self):
        """测试简单键获取"""
        d = {"name": "AI助手"}
        assert safe_get(d, "name") == "AI助手"

    def test_missing_key_returns_default(self):
        """测试缺失键返回默认值"""
        d = {"name": "AI助手"}
        assert safe_get(d, "age") is None
        assert safe_get(d, "age", default=0) == 0

    def test_nested_key(self):
        """测试嵌套键获取"""
        d = {"app": {"name": "AI助手", "version": "1.0"}}
        assert safe_get(d, "app.name") == "AI助手"
        assert safe_get(d, "app.version") == "1.0"

    def test_deeply_nested_key(self):
        """测试深层嵌套键获取"""
        d = {"a": {"b": {"c": "deep_value"}}}
        assert safe_get(d, "a.b.c") == "deep_value"

    def test_nested_missing_key(self):
        """测试嵌套缺失键"""
        d = {"app": {"name": "AI助手"}}
        assert safe_get(d, "app.missing") is None
        assert safe_get(d, "app.missing", "default") == "default"

    def test_non_dict_intermediate(self):
        """测试中间层非字典"""
        d = {"app": "not_a_dict"}
        assert safe_get(d, "app.name") is None

    def test_empty_dict(self):
        """测试空字典"""
        assert safe_get({}, "key") is None
        assert safe_get({}, "key", "fallback") == "fallback"


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 16 + "Text Utils 模块测试" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = []

    # ---- debug_print ----
    print("=" * 60)
    print("debug_print 测试")
    print("=" * 60)
    t1 = TestDebugPrint()
    for name, fn in [
        ("输出包含消息内容", t1.test_output_contains_message),
        ("输出包含默认级别 DEBUG", t1.test_output_contains_level),
        ("自定义级别", t1.test_custom_level),
        ("输出包含调用者文件名", t1.test_output_contains_filename),
        ("输出格式检查", t1.test_output_format),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append((name, False))
    print()

    # ---- format_text ----
    print("=" * 60)
    print("format_text 测试")
    print("=" * 60)
    t2 = TestFormatText()
    for name, fn in [
        ("空字符串", t2.test_empty_string),
        ("None 输入", t2.test_none_input),
        ("短文本不变", t2.test_short_text_unchanged),
        ("刚好等于最大长度", t2.test_exact_length_unchanged),
        ("长文本被截断", t2.test_long_text_truncated),
        ("前后空白被去除", t2.test_whitespace_stripped),
        ("自定义最大长度", t2.test_custom_max_length),
        ("默认最大长度 100", t2.test_default_max_length),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append((name, False))
    print()

    # ---- safe_get ----
    print("=" * 60)
    print("safe_get 测试")
    print("=" * 60)
    t3 = TestSafeGet()
    for name, fn in [
        ("简单键获取", t3.test_simple_key),
        ("缺失键返回默认值", t3.test_missing_key_returns_default),
        ("嵌套键获取", t3.test_nested_key),
        ("深层嵌套键获取", t3.test_deeply_nested_key),
        ("嵌套缺失键", t3.test_nested_missing_key),
        ("中间层非字典", t3.test_non_dict_intermediate),
        ("空字典", t3.test_empty_dict),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append((name, False))
    print()

    # 打印汇总
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, r in results:
        status = "✓ 通过" if r else "✗ 失败"
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
