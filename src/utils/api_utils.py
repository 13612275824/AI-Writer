# API调用工具
"""
API 调用工具模块

提供通用的 API 请求辅助函数：
- 重试机制
- 响应验证
- API Key 格式校验
"""

import time
import functools
from typing import Any, Callable, Optional, TypeVar, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def validate_api_key(api_key: str, prefix: str = "sk-") -> bool:
    """校验 API Key 格式是否合法

    Args:
        api_key: 待校验的 API Key
        prefix: 期望的前缀（默认 "sk-"）

    Returns:
        True 表示格式合法，False 表示格式异常
    """
    if not api_key or not isinstance(api_key, str):
        return False
    if len(api_key) < 10:
        return False
    if prefix and not api_key.startswith(prefix):
        return False
    return True


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """重试装饰器

    当被装饰函数抛出指定异常时，自动重试并等待指数退避时间。

    Args:
        max_retries: 最大重试次数（不含首次调用）
        delay: 初始等待秒数
        backoff: 退避系数（每次重试等待时间乘以此值）
        exceptions: 需要捕获重试的异常类型

    Returns:
        装饰后的函数

    示例::

        @retry(max_retries=3, delay=0.5)
        def call_api():
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"[{func.__name__}] 第 {attempt + 1} 次调用失败: {e}，"
                            f"{current_delay:.1f}s 后重试..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"[{func.__name__}] 已达到最大重试次数 ({max_retries})，"
                            f"放弃重试"
                        )

            raise last_exception  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def validate_response(response_dict: Dict[str, Any], required_fields: list) -> bool:
    """验证 API 响应字典是否包含必需字段

    Args:
        response_dict: API 返回的字典
        required_fields: 必需包含的字段名列表

    Returns:
        True 表示所有必需字段都存在

    Raises:
        ValueError: 当缺少必需字段时
    """
    if not isinstance(response_dict, dict):
        raise ValueError(f"响应应为字典类型，实际为 {type(response_dict).__name__}")

    missing = [field for field in required_fields if field not in response_dict]
    if missing:
        raise ValueError(f"响应缺少必需字段: {', '.join(missing)}")

    return True
