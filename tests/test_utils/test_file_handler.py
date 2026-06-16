"""测试 file_handler.py 文件处理工具模块"""

import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到 Python 路径
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.file_handler import (
    ensure_dir,
    read_text_file,
    write_text_file,
    safe_write_text,
    get_file_info,
    list_files,
)

# 导入必须在路径设置之后


# ==================== ensure_dir 测试 ====================

class TestEnsureDir:
    """ensure_dir 函数测试"""

    def test_create_new_dir(self):
        """测试创建新目录"""
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = Path(tmp) / "sub" / "deep"
            result = ensure_dir(new_dir)
            assert result.exists()
            assert result.is_dir()

    def test_existing_dir(self):
        """测试已存在的目录"""
        with tempfile.TemporaryDirectory() as tmp:
            result = ensure_dir(tmp)
            assert result.exists()

    def test_returns_path(self):
        """测试返回 Path 对象"""
        with tempfile.TemporaryDirectory() as tmp:
            result = ensure_dir(tmp)
            assert isinstance(result, Path)


# ==================== read_text_file 测试 ====================

class TestReadTextFile:
    """read_text_file 函数测试"""

    def test_read_utf8_file(self):
        """测试读取 UTF-8 文件"""
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "test.txt"
            file_path.write_text("测试内容", encoding="utf-8")
            content = read_text_file(file_path)
            assert content == "测试内容"

    def test_read_nonexistent_file(self):
        """测试读取不存在的文件"""
        try:
            read_text_file("/nonexistent/path/file.txt")
            assert False, "应该抛出 FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_read_directory_raises(self):
        """测试读取目录抛出异常"""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                read_text_file(tmp)
                assert False, "应该抛出 ValueError"
            except ValueError:
                pass


# ==================== write_text_file 测试 ====================

class TestWriteTextFile:
    """write_text_file 函数测试"""

    def test_write_new_file(self):
        """测试写入新文件"""
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "output.txt"
            result = write_text_file(file_path, "写入内容")
            assert result.exists()
            assert result.read_text(encoding="utf-8") == "写入内容"

    def test_auto_create_parent_dir(self):
        """测试自动创建父目录"""
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "sub" / "dir" / "output.txt"
            result = write_text_file(file_path, "hello")
            assert result.exists()
            assert result.parent.exists()

    def test_overwrite_existing_file(self):
        """测试覆盖已有文件"""
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "output.txt"
            write_text_file(file_path, "第一次")
            write_text_file(file_path, "第二次")
            assert file_path.read_text(encoding="utf-8") == "第二次"


# ==================== safe_write_text 测试 ====================

class TestSafeWriteText:
    """safe_write_text 函数测试"""

    def test_safe_write(self):
        """测试安全写入"""
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "safe.txt"
            result = safe_write_text(file_path, "安全写入")
            assert result.exists()
            assert result.read_text(encoding="utf-8") == "安全写入"

    def test_safe_write_overwrite(self):
        """测试安全写入覆盖"""
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "safe.txt"
            safe_write_text(file_path, "第一次")
            safe_write_text(file_path, "第二次")
            assert file_path.read_text(encoding="utf-8") == "第二次"

    def test_no_tmp_file_left(self):
        """测试写入后无临时文件残留"""
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "safe.txt"
            safe_write_text(file_path, "内容")
            tmp_files = list(Path(tmp).glob("*.tmp"))
            assert len(tmp_files) == 0


# ==================== get_file_info 测试 ====================

class TestGetFileInfo:
    """get_file_info 函数测试"""

    def test_basic_info(self):
        """测试基本信息"""
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "test.txt"
            file_path.write_bytes(b"test content")
            info = get_file_info(file_path)
            assert info["name"] == "test.txt"
            assert info["extension"] == ".txt"
            assert info["is_file"] is True
            assert info["is_dir"] is False
            assert info["size"] > 0

    def test_nonexistent_file(self):
        """测试不存在的文件"""
        try:
            get_file_info("/nonexistent/file.txt")
            assert False, "应该抛出 FileNotFoundError"
        except FileNotFoundError:
            pass


# ==================== list_files 测试 ====================

class TestListFiles:
    """list_files 函数测试"""

    def test_list_all_files(self):
        """测试列出所有文件"""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("a")
            (Path(tmp) / "b.txt").write_text("b")
            files = list_files(tmp)
            assert len(files) == 2

    def test_filter_by_extension(self):
        """测试按扩展名过滤"""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("a")
            (Path(tmp) / "b.pdf").write_text("b")
            (Path(tmp) / "c.txt").write_text("c")
            files = list_files(tmp, extension=".txt")
            assert len(files) == 2
            assert all(f.suffix == ".txt" for f in files)

    def test_recursive(self):
        """测试递归搜索"""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("a")
            sub = Path(tmp) / "sub"
            sub.mkdir()
            (sub / "b.txt").write_text("b")
            files = list_files(tmp, recursive=True)
            assert len(files) == 2

    def test_nonexistent_dir(self):
        """测试不存在的目录"""
        try:
            list_files("/nonexistent/dir")
            assert False, "应该抛出 FileNotFoundError"
        except FileNotFoundError:
            pass


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 14 + "File Handler 模块测试" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = []

    # ---- ensure_dir ----
    print("=" * 60)
    print("ensure_dir 测试")
    print("=" * 60)
    t1 = TestEnsureDir()
    for name, fn in [
        ("创建新目录", t1.test_create_new_dir),
        ("已存在的目录", t1.test_existing_dir),
        ("返回 Path 对象", t1.test_returns_path),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append((name, False))
    print()

    # ---- read_text_file ----
    print("=" * 60)
    print("read_text_file 测试")
    print("=" * 60)
    t2 = TestReadTextFile()
    for name, fn in [
        ("读取 UTF-8 文件", t2.test_read_utf8_file),
        ("读取不存在的文件", t2.test_read_nonexistent_file),
        ("读取目录抛出异常", t2.test_read_directory_raises),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append((name, False))
    print()

    # ---- write_text_file ----
    print("=" * 60)
    print("write_text_file 测试")
    print("=" * 60)
    t3 = TestWriteTextFile()
    for name, fn in [
        ("写入新文件", t3.test_write_new_file),
        ("自动创建父目录", t3.test_auto_create_parent_dir),
        ("覆盖已有文件", t3.test_overwrite_existing_file),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append((name, False))
    print()

    # ---- safe_write_text ----
    print("=" * 60)
    print("safe_write_text 测试")
    print("=" * 60)
    t4 = TestSafeWriteText()
    for name, fn in [
        ("安全写入", t4.test_safe_write),
        ("安全写入覆盖", t4.test_safe_write_overwrite),
        ("无临时文件残留", t4.test_no_tmp_file_left),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append((name, False))
    print()

    # ---- get_file_info ----
    print("=" * 60)
    print("get_file_info 测试")
    print("=" * 60)
    t5 = TestGetFileInfo()
    for name, fn in [
        ("基本信息", t5.test_basic_info),
        ("不存在的文件", t5.test_nonexistent_file),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
            results.append((name, True))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append((name, False))
    print()

    # ---- list_files ----
    print("=" * 60)
    print("list_files 测试")
    print("=" * 60)
    t6 = TestListFiles()
    for name, fn in [
        ("列出所有文件", t6.test_list_all_files),
        ("按扩展名过滤", t6.test_filter_by_extension),
        ("递归搜索", t6.test_recursive),
        ("不存在的目录", t6.test_nonexistent_dir),
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
