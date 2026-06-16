# 日志工具
#
# 本模块负责统一项目的日志系统，提供：
# - setup_logging(): 从 configs/logging.yaml 加载配置并初始化日志系统
# - get_logger(name): 获取指定名称的 Logger 实例
#
# 使用方式：
#   from src.utils.logger import get_logger
#   logger = get_logger(__name__)
#   logger.info("操作完成")
#   logger.error("操作失败", exc_info=True)
#
# 日志输出目标（由 configs/logging.yaml 控制）：
# - 控制台（stdout）：INFO 及以上级别，便于开发调试
# - 文件（logs/app.log）：DEBUG 及以上级别，便于问题排查
#
# 注意事项：
# - 首次调用 get_logger() 时会自动执行 setup_logging()
# - setup_logging() 是幂等的，多次调用只生效一次
# - 日志文件目录（logs/）不存在时会自动创建

import logging
import logging.config
from pathlib import Path
from typing import Optional

# 模块级标志位：确保日志系统只初始化一次
_logging_initialized: bool = False


def setup_logging(config: Optional[object] = None) -> None:
    """初始化项目日志系统

    从 Config 加载 logging.yaml 配置，通过 logging.config.dictConfig() 完成初始化。
    此函数是幂等的，重复调用不会重复初始化。

    Args:
        config: Config 实例（可选）。
                若为 None，则自动通过 get_config() 获取全局单例。
                显式传入可避免循环导入（例如在 Config 类内部使用时）。

    执行流程：
    1. 检查是否已初始化（通过 _logging_initialized 标志）
    2. 获取 Config 实例并读取 logging_config
    3. 确保日志文件目录存在（FileHandler 不会自动创建父目录）
    4. 调用 dictConfig() 应用配置
    5. 输出初始化完成的 INFO 日志

    示例：
        >>> from src.utils.logger import setup_logging
        >>> setup_logging()  # 使用全局 Config
        >>> setup_logging(config=my_config)  # 使用指定 Config
    """
    global _logging_initialized

    # 幂等检查：避免重复初始化
    if _logging_initialized:
        return

    # 获取 Config 实例
    if config is None:
        from src.core.config import get_config
        config = get_config()

    logging_config = config.logging_config

    # 确保日志文件目录存在
    # logging.yaml 中配置的 filename 可能包含目录（如 logs/app.log）
    # FileHandler 不会自动创建目录，需要提前处理
    _ensure_log_dirs(logging_config)

    # 应用日志配置
    logging.config.dictConfig(logging_config)

    # 标记初始化完成
    _logging_initialized = True

    # 输出一条启动日志，确认系统正常工作
    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成")
    logger.warning("日志系统初始化完成")


def _ensure_log_dirs(logging_config: dict) -> None:
    """检查并创建日志文件所需的目录

    遍历 logging.yaml 中所有 FileHandler 的 filename 配置，
    若目录不存在则自动创建。

    Args:
        logging_config: 从 logging.yaml 加载的配置字典

    说明：
        - 只处理包含 filename 字段的 handler（即文件类处理器）
        - 使用 parents=True 递归创建多级目录
        - 使用 exist_ok=True 避免目录已存在时报错
    """
    handlers = logging_config.get("handlers", {})
    for handler_cfg in handlers.values():
        filename = handler_cfg.get("filename")
        if filename:
            log_dir = Path(filename).parent
            log_dir.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 Logger 实例

    首次调用时会自动执行 setup_logging()，确保日志系统已初始化。
    后续调用直接返回 Logger 实例，无额外开销。

    Args:
        name: Logger 名称，通常传入 __name__ 以标识调用模块。
              例如：src.core.config、src.writing.copywriter

    Returns:
        配置好的 logging.Logger 实例

    示例：
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.debug("调试信息，仅写入文件")
        >>> logger.info("普通信息，同时输出到控制台和文件")
        >>> logger.warning("警告信息")
        >>> logger.error("错误信息", exc_info=True)  # 附带异常堆栈
    """
    # 延迟初始化：首次调用 get_logger 时才设置日志系统
    if not _logging_initialized:
        setup_logging()

    return logging.getLogger(name)


def reset_logging() -> None:
    """重置日志系统状态（仅供测试使用）

    将 _logging_initialized 标志重置为 False，
    使下次调用 setup_logging() 时可以重新初始化。

    注意：此方法仅用于单元测试场景，生产代码中不应调用。
    """
    global _logging_initialized
    _logging_initialized = False
