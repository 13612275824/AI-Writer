# 文件处理工具
"""
文件处理工具模块

提供通用的文件操作辅助函数：
- 文件读写（支持编码检测）
- 目录自动创建
- 文件信息查询
- 安全写入（原子操作）
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(dir_path: str | Path) -> Path:
    """确保目录存在，不存在则创建

    Args:
        dir_path: 目录路径

    Returns:
        目录的 Path 对象
    """
    path = Path(dir_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"创建目录: {path}")
    return path


def read_text_file(file_path: str | Path, encoding: str = "utf-8") -> str:
    """读取文本文件内容

    Args:
        file_path: 文件路径
        encoding: 文件编码（默认 utf-8）

    Returns:
        文件文本内容

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件无法读取
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件: {path}")

    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        # 尝试 GBK 编码
        logger.warning(f"UTF-8 解码失败，尝试 GBK: {path}")
        return path.read_text(encoding="gbk")


def write_text_file(
    file_path: str | Path,
    content: str,
    encoding: str = "utf-8",
    ensure_directory: bool = True,
) -> Path:
    """写入文本文件

    Args:
        file_path: 文件路径
        content: 写入内容
        encoding: 文件编码（默认 utf-8）
        ensure_directory: 是否自动创建父目录

    Returns:
        文件的 Path 对象
    """
    path = Path(file_path)
    if ensure_directory:
        ensure_dir(path.parent)

    path.write_text(content, encoding=encoding)
    logger.debug(f"写入文件: {path} ({len(content)} 字符)")
    return path


def safe_write_text(
    file_path: str | Path,
    content: str,
    encoding: str = "utf-8",
) -> Path:
    """安全写入文件（原子操作）

    先写入临时文件，再原子性替换目标文件，避免写入中断导致文件损坏。

    Args:
        file_path: 目标文件路径
        content: 写入内容
        encoding: 文件编码

    Returns:
        文件的 Path 对象
    """
    path = Path(file_path)
    ensure_dir(path.parent)

    # 写入临时文件，再原子替换
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        shutil.move(tmp_path, str(path))
        logger.debug(f"安全写入文件: {path} ({len(content)} 字符)")
    except Exception:
        # 失败时清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return path


def get_file_info(file_path: str | Path) -> Dict[str, Any]:
    """获取文件基本信息

    Args:
        file_path: 文件路径

    Returns:
        包含文件信息的字典：name, size, extension, modified_time
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "extension": path.suffix,
        "modified_time": stat.st_mtime,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }


def list_files(
    dir_path: str | Path,
    extension: Optional[str] = None,
    recursive: bool = False,
) -> List[Path]:
    """列出目录下的文件

    Args:
        dir_path: 目录路径
        extension: 过滤文件扩展名（如 ".txt"、".pdf"）
        recursive: 是否递归搜索子目录

    Returns:
        文件 Path 对象列表
    """
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"目录不存在: {path}")
    if not path.is_dir():
        raise ValueError(f"路径不是目录: {path}")

    pattern = f"**/*" if recursive else "*"
    files = [f for f in path.glob(pattern) if f.is_file()]

    if extension:
        ext = extension if extension.startswith(".") else f".{extension}"
        files = [f for f in files if f.suffix == ext]

    return sorted(files)
