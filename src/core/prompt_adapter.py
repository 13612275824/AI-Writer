# LangChain Prompt 适配器
#
# 本模块使用 LangChain 的 ChatPromptTemplate 封装 Prompt 模板管理，
# 替代 Worker1 中自实现的 PromptManager，使用 prompts.yaml 配置文件。
#
# 核心功能：
# - 从 Config 单例加载 prompts.yaml 中的角色模板
# - 创建并缓存 LangChain ChatPromptTemplate 对象
# - 支持带变量的模板渲染

from typing import Dict, Optional

from langchain.prompts import ChatPromptTemplate

from src.core.config import get_config


class PromptAdapter:
    """LangChain Prompt 适配器
    
    职责：
    1. 从 prompts.yaml 配置中加载不同角色的 Prompt 模板
    2. 创建并缓存 LangChain ChatPromptTemplate 对象
    3. 支持带变量的模板渲染
    
    示例：
        >>> adapter = PromptAdapter()
        >>> prompt = adapter.get_prompt("writing", topic="AI")
        >>> print(prompt)
    """
    
    def __init__(self):
        """初始化 Prompt 适配器"""
        self._config = get_config()
        self._templates: Dict[str, ChatPromptTemplate] = {}  # 缓存已编译的模板对象
    
    def get_prompt(self, role: str, **variables) -> str:
        """获取渲染后的 Prompt 文本
        
        Args:
            role: 角色名称（writing/editing/research 等）
            **variables: 模板变量
            
        Returns:
            渲染后的 Prompt 字符串
        """
        # 获取或创建模板（懒加载 + 缓存）
        if role not in self._templates:
            system_prompt = self._config.get_system_prompt(role)
            user_template = self._config.get_user_template(role)
            
            if not system_prompt and not user_template:
                raise ValueError(f"未找到角色 '{role}' 的 Prompt 配置")
            
            # 创建 LangChain ChatPromptTemplate
            messages = []
            if system_prompt:
                messages.append(("system", system_prompt))
            if user_template:
                messages.append(("user", user_template))
            
            template = ChatPromptTemplate.from_messages(messages)
            self._templates[role] = template
        
        # 使用变量渲染模板
        try:
            return self._templates[role].format(**variables)
        except KeyError as e:
            raise ValueError(f"角色 '{role}' 缺少变量 {e}") from e
    
    def get_chat_prompt_template(self, role: str) -> ChatPromptTemplate:
        """获取 LangChain ChatPromptTemplate 对象
        
        当需要在 LangChain Chain 中使用模板对象时调用此方法。
        
        Args:
            role: 角色名称
            
        Returns:
            ChatPromptTemplate 对象
        """
        if role not in self._templates:
            # 触发模板创建
            self.get_prompt(role)
        
        return self._templates[role]
    
    def clear_cache(self):
        """清除模板缓存（配置文件修改后可调用此方法刷新）"""
        self._templates.clear()


# ------------------------------------------------------------------ #
#                     全局单例 & 便捷获取函数                           #
# ------------------------------------------------------------------ #

_prompt_adapter_instance: Optional[PromptAdapter] = None


def get_prompt_manager() -> PromptAdapter:
    """获取全局 Prompt 适配器单例（懒加载）
    
    首次调用时创建 PromptAdapter 实例。
    后续调用返回同一实例。
    
    注意：函数名保持为 get_prompt_manager 以向后兼容。
    
    Returns:
        PromptAdapter 单例对象
        
    示例：
        >>> manager = get_prompt_manager()
        >>> prompt = manager.get_prompt("writing", topic="AI")
    """
    global _prompt_adapter_instance
    if _prompt_adapter_instance is None:
        _prompt_adapter_instance = PromptAdapter()
    return _prompt_adapter_instance


def reload_prompt_manager() -> PromptAdapter:
    """强制重新创建 Prompt 适配器
    
    适用场景：
    - prompts.yaml 配置文件修改后需要刷新
    
    Returns:
        新的 PromptAdapter 实例
    """
    global _prompt_adapter_instance
    _prompt_adapter_instance = PromptAdapter()
    return _prompt_adapter_instance
