# 文本处理工具
import os
import sys
from typing import Any


def debug_print(msg: Any, level: str = "DEBUG") -> None:
    """打印带文件名和行号的调试信息

    Args:
        msg: 要打印的消息内容
        level: 日志级别标签（默认 DEBUG）

    示例:
        >>> from src.utils.text_utils import debug_print
        >>> debug_print("配置加载开始")
        [config.py:30] [DEBUG] 配置加载开始
    """
    # 获取调用者的栈帧（frame=1 表示上一级调用者）
    frame = sys._getframe(1)
    filename = os.path.basename(frame.f_code.co_filename)
    lineno = frame.f_lineno
    print(f"[{filename}:{lineno}] [{level}] {msg}")


def format_text(text: str, max_length: int = 100) -> str:
    """格式化文本，超过指定长度时截断并添加省略号

    Args:
        text: 原始文本
        max_length: 最大显示长度

    Returns:
        格式化后的文本
    """
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def safe_get(d: dict, key: str, default: Any = None) -> Any:
    """安全地从嵌套字典中获取值

    Args:
        d: 字典对象
        key: 键名（支持点分隔的嵌套键，如 "app.name"）
        default: 默认值

    Returns:
        对应的值或默认值

    示例:
        >>> config = {"app": {"name": "AI助手"}}
        >>> safe_get(config, "app.name")
        'AI助手'
    """
    keys = key.split(".")
    current = d
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current
