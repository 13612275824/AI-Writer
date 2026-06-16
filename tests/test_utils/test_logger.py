"""测试 logger.py 日志工具模块"""

import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.logger import setup_logging, get_logger, reset_logging, _ensure_log_dirs

# 导入必须在路径设置之后


def test_setup_logging():
    """测试日志系统初始化"""
    print("=" * 60)
    print("测试 1: 日志系统初始化 (setup_logging)")
    print("=" * 60)

    try:
        # 先重置状态
        reset_logging()

        # 执行初始化
        setup_logging()
        print("✓ setup_logging() 调用成功")

        # 再次调用应该是幂等的（不报错）
        setup_logging()
        print("✓ 重复调用 setup_logging() 幂等无异常")
        print()

    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        print()
        raise AssertionError(f"初始化失败: {e}")


def test_get_logger():
    """测试获取 Logger 实例"""
    print("=" * 60)
    print("测试 2: 获取 Logger 实例 (get_logger)")
    print("=" * 60)

    try:
        # 重置状态以确保自动初始化能正常工作
        reset_logging()

        # 通过 get_logger 获取（应自动触发 setup_logging）
        logger = get_logger("test_module")
        print(f"✓ Logger 创建成功: {logger}")
        print(f"  - 名称: {logger.name}")
        print(f"  - 类型: {type(logger).__name__}")

        # 验证是 logging.Logger 实例
        assert isinstance(logger, logging.Logger), "Logger 类型不正确"
        print("✓ 类型验证通过 (logging.Logger)")
        print()

    except Exception as e:
        print(f"✗ 获取 Logger 失败: {e}")
        print()
        raise AssertionError(f"获取 Logger 失败: {e}")


def test_log_levels():
    """测试各级别日志输出"""
    print("=" * 60)
    print("测试 3: 各级别日志输出")
    print("=" * 60)

    try:
        logger = get_logger("test_levels")

        print("--- 以下日志应同时输出到控制台和文件 ---")
        logger.debug("这是一条 DEBUG 日志（控制台不显示，仅写入文件）")
        logger.info("这是一条 INFO 日志")
        logger.warning("这是一条 WARNING 日志")
        logger.error("这是一条 ERROR 日志")
        logger.critical("这是一条 CRITICAL 日志")

        print("✓ 所有级别日志输出完成")
        print("  提示: 请检查 logs/app.log 确认 DEBUG 消息已写入")
        print()

    except Exception as e:
        print(f"✗ 日志输出失败: {e}")
        print()
        raise AssertionError(f"日志输出失败: {e}")


def test_log_file_written():
    """测试日志文件是否正确写入"""
    print("=" * 60)
    print("测试 4: 日志文件写入验证")
    print("=" * 60)

    try:
        log_file = project_root / "logs" / "app.log"

        if not log_file.exists():
            print(f"✗ 日志文件不存在: {log_file}")
            raise AssertionError(f"日志文件不存在: {log_file}")

        print(f"✓ 日志文件存在: {log_file}")

        # 读取文件内容
        content = log_file.read_text(encoding="utf-8")
        lines = [line for line in content.strip().split("\n") if line]

        print(f"  - 日志行数: {len(lines)}")

        # 检查是否包含 DEBUG 日志
        has_debug = any("DEBUG" in line for line in lines)
        if has_debug:
            print("✓ 文件中包含 DEBUG 级别日志")
        else:
            print("✗ 文件中未找到 DEBUG 级别日志")

        # 显示最近几行
        print("  - 最近 3 行日志:")
        for line in lines[-3:]:
            print(f"    {line}")

        print()

    except Exception as e:
        print(f"✗ 文件读取失败: {e}")
        print()
        raise AssertionError(f"文件读取失败: {e}")


def test_ensure_log_dirs():
    """测试日志目录自动创建"""
    print("=" * 60)
    print("测试 5: 日志目录自动创建 (_ensure_log_dirs)")
    print("=" * 60)

    try:
        # 模拟配置
        test_config = {
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": "logs/app.log",
                },
                "console": {
                    "class": "logging.StreamHandler",
                    # 没有 filename，应该被跳过
                },
            }
        }

        _ensure_log_dirs(test_config)
        print("✓ _ensure_log_dirs() 执行成功")

        # 验证目录存在
        log_dir = Path("logs")
        if log_dir.exists():
            print(f"✓ 日志目录已创建: {log_dir.resolve()}")
        else:
            print("✗ 日志目录未创建")
            raise AssertionError("日志目录未创建")

        print()

    except Exception as e:
        print(f"✗ 目录创建失败: {e}")
        print()
        raise AssertionError(f"目录创建失败: {e}")


def test_reset_logging():
    """测试日志系统重置"""
    print("=" * 60)
    print("测试 6: 日志系统重置 (reset_logging)")
    print("=" * 60)

    try:
        from src.utils.logger import _logging_initialized

        # 先确保已初始化
        setup_logging()
        print(f"  - 重置前状态: _logging_initialized = {_logging_initialized}")

        # 执行重置
        reset_logging()

        # 重新导入以获取最新值
        from src.utils import logger as logger_module
        print(
            f"  - 重置后状态: _logging_initialized = {logger_module._logging_initialized}")

        assert logger_module._logging_initialized is False, "重置后应为 False"
        print("✓ 重置功能正常")
        print()

    except Exception as e:
        print(f"✗ 重置失败: {e}")
        print()
        raise AssertionError(f"重置失败: {e}")


def test_error_logging():
    """测试异常日志记录（exc_info）"""
    print("=" * 60)
    print("测试 7: 异常日志记录 (exc_info)")
    print("=" * 60)

    try:
        logger = get_logger("test_error")

        try:
            # 故意制造一个异常
            result = 1 / 0
        except ZeroDivisionError:
            logger.error("发生除零错误", exc_info=True)
            print("✓ exc_info=True 异常日志记录完成")
            print("  提示: 请检查 logs/app.log 确认异常堆栈已写入")

        print()

    except Exception as e:
        print(f"✗ 异常日志记录失败: {e}")
        print()
        raise AssertionError(f"异常日志记录失败: {e}")


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 18 + "Logger 模块测试" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = []

    # 运行测试
    results.append(("日志系统初始化", test_setup_logging()))
    results.append(("获取 Logger 实例", test_get_logger()))
    results.append(("各级别日志输出", test_log_levels()))
    results.append(("日志文件写入验证", test_log_file_written()))
    results.append(("日志目录自动创建", test_ensure_log_dirs()))
    results.append(("日志系统重置", test_reset_logging()))
    results.append(("异常日志记录", test_error_logging()))

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
