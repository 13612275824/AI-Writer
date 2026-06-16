# LangChain LLM 适配器
#
# 本模块使用 LangChain 的 ChatOpenAI 封装大模型调用，
# 替代 Worker1 中自实现的 LLMClient，确保写作模块无需修改即可使用。
#
# 核心功能：
# - 同步/异步/流式三种调用模式
# - Mock ChatCompletion 对象兼容 Worker1 的 response.choices[0].message.content 访问方式
# - 自动读取 Config 单例中的模型配置（API Key、模型名、温度等）

from typing import Any, Dict, Generator, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.core.config import get_config


class LLMAdapter:
    """LangChain LLM 适配器
    
    职责：
    1. 封装 ChatOpenAI，提供与 Worker1 的 LLMClient 兼容的接口
    2. 从 Config 单例中读取配置（API Key、模型名、温度等）
    3. 支持同步、异步、流式三种调用模式
    
    示例：
        >>> adapter = LLMAdapter()
        >>> response = adapter.chat_completion_simple("你好")
        >>> print(response)
    """
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> None:
        """初始化 LLM 适配器
        
        Args:
            provider: 模型提供商名称（默认 "openai"，实际使用阿里云百炼 DashScope）
            api_key: API 密钥（None 时从配置读取）
            base_url: API 端点地址（None 时从配置读取）
            model: 模型名称（None 时从配置读取）
            temperature: 温度参数（None 时从配置读取）
            max_tokens: 最大生成 token 数（None 时从配置读取）
            top_p: 核采样参数（None 时从配置读取）
            frequency_penalty: 频率惩罚（None 时从配置读取）
            presence_penalty: 存在惩罚（None 时从配置读取）
        """
        self._provider = provider
        self._config = get_config()
        
        # 获取 API 凭证和端点地址
        self._api_key = api_key or self._config.api_key
        self._base_url = base_url or self._config.base_url
        
        # 获取模型配置（优先级：参数 > 环境变量 > YAML 配置 > 默认值）
        provider_config = self._config.get_provider_config(provider)
        self._model = model or self._config.default_model or provider_config.get("model", "qwen-plus")
        self._temperature = temperature if temperature is not None else self._config.temperature
        self._max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        self._top_p = top_p if top_p is not None else self._config.top_p
        self._frequency_penalty = frequency_penalty if frequency_penalty is not None else self._config.frequency_penalty
        self._presence_penalty = presence_penalty if presence_penalty is not None else self._config.presence_penalty
        
        # 初始化 LangChain ChatOpenAI 实例
        self._llm = ChatOpenAI(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            openai_api_key=self._api_key,
            openai_api_base=self._base_url,
            model_kwargs={
                k: v for k, v in {
                    "top_p": self._top_p,
                    "frequency_penalty": self._frequency_penalty,
                    "presence_penalty": self._presence_penalty,
                }.items() if v is not None
            }
        )
        
        # 上一次调用的 Token 用量统计
        self._last_prompt_tokens: int = 0
        self._last_completion_tokens: int = 0
        
        print(f"[LLMAdapter] 初始化完成: provider={provider}, model={self._model}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: bool = False,
    ) -> Any:
        """发送聊天请求并获取模型响应
        
        兼容 Worker1 的 LLMClient.chat_completion 接口。
        
        Args:
            messages: 消息列表，格式: [{"role": "user", "content": "..."}]
            model: 模型名称（覆盖默认值）
            temperature: 温度（覆盖默认值）
            max_tokens: 最大 token 数（覆盖默认值）
            top_p: 核采样（覆盖默认值）
            frequency_penalty: 频率惩罚（覆盖默认值）
            presence_penalty: 存在惩罚（覆盖默认值）
            stream: 是否使用流式输出（默认 False）
            
        Returns:
            非流式: Mock ChatCompletion 对象（兼容 response.choices[0].message.content）
            流式: 文本片段的生成器
            
        Raises:
            Exception: API 调用失败时抛出
        """
        # 将消息列表转换为 LangChain 消息格式
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        
        if stream:
            return self._stream_completion(lc_messages)
        else:
            return self._non_stream_completion(lc_messages)
    
    def _non_stream_completion(self, messages: List) -> Any:
        """非流式完成：一次性返回完整响应"""
        response = self._llm.invoke(messages)
        
        # 创建 Mock ChatCompletion 对象以保持与 Worker1 的接口兼容性
        # Worker1 中使用 response.choices[0].message.content 访问响应内容
        class MockUsage:
            """模拟 OpenAI API 的用量统计"""
            def __init__(self, prompt_tokens: int, completion_tokens: int):
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = completion_tokens
        
        class MockMessage:
            """模拟消息对象，提供 content 属性"""
            def __init__(self, content: str):
                self.content = content
        
        class MockChoice:
            """模拟选择对象，提供 message 属性"""
            def __init__(self, message: MockMessage):
                self.message = message
        
        class MockChatCompletion:
            """模拟 OpenAI ChatCompletion 响应对象"""
            def __init__(self, content: str):
                self.choices = [MockChoice(MockMessage(content))]
                # 估算 Token 用量（实际值应从 API 响应中获取）
                self.usage = MockUsage(
                    prompt_tokens=len(str(messages)) // 4,
                    completion_tokens=len(content) // 4
                )
        
        return MockChatCompletion(response.content)
    
    def _stream_completion(self, messages: List) -> Generator:
        """流式完成：逐块返回文本片段"""
        for chunk in self._llm.stream(messages):
            yield chunk
    
    def chat_completion_simple(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """简化的聊天接口（单轮对话）
        
        兼容 Worker1 的 LLMClient.chat_completion_simple。
        
        Args:
            prompt: 用户输入的提问或指令
            system_prompt: 系统提示词（可选，用于设定角色和行为）
            **kwargs: 其他参数传递给 chat_completion
            
        Returns:
            模型响应的文本内容
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat_completion(messages, **kwargs)
        
        # 提取并保存 Token 用量
        if hasattr(response, "usage") and response.usage:
            self._last_prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            self._last_completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        
        return response.choices[0].message.content
    
    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Generator[str, None, None]:
        """流式聊天接口（逐块返回文本片段）
        
        兼容 Worker1 的 LLMClient.chat_completion_stream。
        适用于 CLI 实时显示、Web SSE 推送等场景。
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数传递给 chat_completion
            
        Yields:
            每个生成的文本片段
        """
        for chunk in self.chat_completion(messages, stream=True, **kwargs):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
    
    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Any:
        """异步聊天接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数传递给 chat_completion
            
        Returns:
            Mock ChatCompletion 对象
        """
        # 将消息列表转换为 LangChain 消息格式
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        
        response = await self._llm.ainvoke(lc_messages)
        
        # 创建 Mock 对象（与非流式相同结构）
        class MockUsage:
            def __init__(self, prompt_tokens: int, completion_tokens: int):
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = completion_tokens
        
        class MockMessage:
            def __init__(self, content: str):
                self.content = content
        
        class MockChoice:
            def __init__(self, message: MockMessage):
                self.message = message
        
        class MockChatCompletion:
            def __init__(self, content: str):
                self.choices = [MockChoice(MockMessage(content))]
                self.usage = MockUsage(
                    prompt_tokens=len(str(messages)) // 4,
                    completion_tokens=len(content) // 4
                )
        
        return MockChatCompletion(response.content)
    
    @property
    def model(self) -> str:
        """当前使用的模型名称"""
        return self._model
    
    @property
    def provider(self) -> str:
        """当前模型提供商"""
        return self._provider
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型配置信息
        
        Returns:
            包含模型配置的字典
        """
        return {
            "provider": self._provider,
            "model": self._model,
            "base_url": self._base_url,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
        }
    
    def __repr__(self) -> str:
        """友好的字符串表示，便于调试和日志记录"""
        return f"<LLMAdapter provider={self._provider!r} model={self._model!r}>"


# ------------------------------------------------------------------ #
#                     全局单例 & 便捷获取函数                           #
# ------------------------------------------------------------------ #

_llm_adapter_instance: Optional[LLMAdapter] = None


def get_llm_client(provider: str = "openai") -> LLMAdapter:
    """获取全局 LLM 适配器单例（懒加载）
    
    首次调用时创建 LLMAdapter 实例。
    后续调用返回同一实例，避免重复初始化。
    
    注意：函数名保持为 get_llm_client 以兼容 Worker1 的写作模块。
    
    Args:
        provider: 模型提供商名称（默认 "openai"）
        
    Returns:
        LLMAdapter 单例对象
        
    示例：
        >>> client = get_llm_client()
        >>> response = client.chat_completion_simple("你好")
    """
    global _llm_adapter_instance
    if _llm_adapter_instance is None:
        _llm_adapter_instance = LLMAdapter(provider=provider)
    return _llm_adapter_instance


def reload_llm_client(provider: str = "openai") -> LLMAdapter:
    """强制重新创建 LLM 客户端
    
    适用场景：
    - 配置文件修改后需要重新初始化
    - 切换到不同的模型提供商
    
    Args:
        provider: 模型提供商名称
        
    Returns:
        新的 LLMAdapter 实例
    """
    global _llm_adapter_instance
    _llm_adapter_instance = LLMAdapter(provider=provider)
    return _llm_adapter_instance
