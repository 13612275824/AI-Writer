"""测试 api_utils.py API调用工具模块"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.api_utils import validate_api_key, retry, validate_response
import time

# 导入必须在路径设置之后


# ==================== validate_api_key 测试 ====================

class TestValidateApiKey:
    """validate_api_key 函数测试"""

    def test_valid_key(self):
        """测试合法 API Key"""
        assert validate_api_key("sk-abcdef1234567890") is True

    def test_empty_key(self):
        """测试空 API Key"""
        assert validate_api_key("") is False

    def test_none_key(self):
        """测试 None API Key"""
        assert validate_api_key(None) is False

    def test_short_key(self):
        """测试过短的 API Key"""
        assert validate_api_key("sk-abc") is False

    def test_wrong_prefix(self):
        """测试错误前缀"""
        assert validate_api_key("xx-abcdef1234567890") is False

    def test_custom_prefix(self):
        """测试自定义前缀"""
        assert validate_api_key("api-abcdef1234567890", prefix="api-") is True

    def test_no_prefix(self):
        """测试无前缀校验"""
        assert validate_api_key("abcdef1234567890", prefix="") is True

    def test_non_string_input(self):
        """测试非字符串输入"""
        assert validate_api_key(12345) is False


# ==================== retry 装饰器测试 ====================

class TestRetry:
    """retry 装饰器测试"""

    def test_success_no_retry(self):
        """测试成功时不重试"""
        call_count = 0

        @retry(max_retries=3, delay=0.01)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = success_func()
        assert result == "ok"
        assert call_count == 1

    def test_retry_then_success(self):
        """测试重试后成功"""
        call_count = 0

        @retry(max_retries=3, delay=0.01)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("暂时失败")
            return "ok"

        result = flaky_func()
        assert result == "ok"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """测试超过最大重试次数抛出异常"""
        call_count = 0

        @retry(max_retries=2, delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("永久失败")

        try:
            always_fail()
            assert False, "应该抛出异常"
        except RuntimeError as e:
            assert "永久失败" in str(e)
            assert call_count == 3  # 1次首次 + 2次重试

    def test_specific_exception_only(self):
        """测试只捕获指定异常"""

        @retry(max_retries=3, delay=0.01, exceptions=(ValueError,))
        def wrong_exception():
            raise TypeError("类型错误")

        try:
            wrong_exception()
            assert False, "应该抛出异常"
        except TypeError:
            pass  # TypeError 不在捕获列表中，直接抛出


# ==================== validate_response 测试 ====================

class TestValidateResponse:
    """validate_response 函数测试"""

    def test_valid_response(self):
        """测试合法响应"""
        resp = {"code": 200, "data": "hello", "message": "ok"}
        assert validate_response(resp, ["code", "data"]) is True

    def test_missing_field(self):
        """测试缺少必需字段"""
        resp = {"code": 200}
        try:
            validate_response(resp, ["code", "data"])
            assert False, "应该抛出 ValueError"
        except ValueError as e:
            assert "data" in str(e)

    def test_non_dict_response(self):
        """测试非字典响应"""
        try:
            validate_response("not a dict", ["code"])
            assert False, "应该抛出 ValueError"
        except ValueError as e:
            assert "字典" in str(e)

    def test_empty_required_fields(self):
        """测试空必需字段列表"""
        resp = {"code": 200}
        assert validate_response(resp, []) is True


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 16 + "API Utils 模块测试" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # ---- validate_api_key ----
    print("=" * 60)
    print("validate_api_key 测试")
    print("=" * 60)
    t = TestValidateApiKey()
    results = []
    for name, fn in [
        ("合法 API Key", t.test_valid_key),
        ("空 API Key", t.test_empty_key),
        ("None API Key", t.test_none_key),
        ("过短 API Key", t.test_short_key),
        ("错误前缀", t.test_wrong_prefix),
        ("自定义前缀", t.test_custom_prefix),
        ("无前缀校验", t.test_no_prefix),
        ("非字符串输入", t.test_non_string_input),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append((name, False))
    print()

    # ---- retry ----
    print("=" * 60)
    print("retry 装饰器测试")
    print("=" * 60)
    t2 = TestRetry()
    for name, fn in [
        ("成功不重试", t2.test_success_no_retry),
        ("重试后成功", t2.test_retry_then_success),
        ("超过最大重试次数", t2.test_max_retries_exceeded),
        ("只捕获指定异常", t2.test_specific_exception_only),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append((name, False))
    print()

    # ---- validate_response ----
    print("=" * 60)
    print("validate_response 测试")
    print("=" * 60)
    t3 = TestValidateResponse()
    for name, fn in [
        ("合法响应", t3.test_valid_response),
        ("缺少必需字段", t3.test_missing_field),
        ("非字典响应", t3.test_non_dict_response),
        ("空必需字段列表", t3.test_empty_required_fields),
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
