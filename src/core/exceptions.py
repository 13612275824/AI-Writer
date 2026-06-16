# 自定义异常处理
#
# 本模块定义项目中所有自定义异常类
# 用于统一错误处理和提高代码可读性
#
# 异常层次结构：
# - AppBaseError（基础异常）
#   |- ConfigurationError（配置错误）
#   |- ModelAPIError（模型API错误）
#   |- DocumentError（文档处理错误）
#   |- VectorStoreError（向量存储错误）
#   |- AgentError（Agent执行错误）

from typing import Any, Dict, Optional


class AppBaseError(Exception):
    """应用基础异常类

    所有自定义异常的基类，提供统一的错误信息结构

    Attributes:
        message: 错误描述信息
        error_code: 错误代码（可选）
        details: 额外详细信息（可选）
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """初始化基础异常

        Args:
            message: 错误描述信息
            error_code: 错误代码（如 "CONFIG_001"）
            details: 额外详细信息字典
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """将异常信息转换为字典格式

        Returns:
            包含错误信息的字典
        """
        result = {
            "error": self.__class__.__name__,
            "message": self.message,
        }
        if self.error_code:
            result["error_code"] = self.error_code
        if self.details:
            result["details"] = self.details
        return result

    def __str__(self) -> str:
        """友好的字符串表示"""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(AppBaseError):
    """配置错误异常

    当配置文件加载失败、配置项缺失或格式错误时抛出

    示例：
        >>> raise ConfigurationError("API Key未设置", "CONFIG_001")
    """

    def __init__(self, message: str, error_code: str = "CONFIG_ERROR") -> None:
        super().__init__(message, error_code)


class ModelAPIError(AppBaseError):
    """模型API调用错误异常

    当与大模型API交互失败时抛出（网络错误、认证失败、限流等）

    Attributes:
        provider: 模型提供商名称
        model: 模型名称
        status_code: HTTP状态码（如果有）

    示例：
        >>> raise ModelAPIError("API调用超时", provider="openai", model="qwen-plus")
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        error_code: str = "MODEL_API_ERROR",
    ) -> None:
        super().__init__(message, error_code)
        self.provider = provider
        self.model = model
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        """重写：包含模型信息"""
        result = super().to_dict()
        if self.provider:
            result["provider"] = self.provider
        if self.model:
            result["model"] = self.model
        if self.status_code:
            result["status_code"] = self.status_code
        return result


class DocumentError(AppBaseError):
    """文档处理错误异常

    当文档加载、解析、处理失败时抛出

    示例：
        >>> raise DocumentError("PDF文件损坏", "DOC_001")
    """

    def __init__(self, message: str, error_code: str = "DOC_ERROR") -> None:
        super().__init__(message, error_code)


class VectorStoreError(AppBaseError):
    """向量存储错误异常

    当向量数据库操作失败时抛出（连接失败、索引错误等）

    示例：
        >>> raise VectorStoreError("向量数据库连接失败", "VECTOR_001")
    """

    def __init__(self, message: str, error_code: str = "VECTOR_ERROR") -> None:
        super().__init__(message, error_code)


class AgentError(AppBaseError):
    """Agent执行错误异常

    当Agent任务执行失败时抛出

    示例：
        >>> raise AgentError("写作Agent执行超时", "AGENT_001")
    """

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        error_code: str = "AGENT_ERROR",
    ) -> None:
        super().__init__(message, error_code)
        self.agent_name = agent_name

    def to_dict(self) -> Dict[str, Any]:
        """重写：包含Agent名称"""
        result = super().to_dict()
        if self.agent_name:
            result["agent_name"] = self.agent_name
        return result


class PromptError(AppBaseError):
    """Prompt模板错误异常

    当Prompt模板加载或渲染失败时抛出

    示例：
        >>> raise PromptError("Prompt模板缺少占位符", "PROMPT_001")
    """

    def __init__(self, message: str, error_code: str = "PROMPT_ERROR") -> None:
        super().__init__(message, error_code)
