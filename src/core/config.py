# 配置加载与管理
#
# 本模块负责统一管理和加载项目的所有配置信息，包括：
# - .env 环境变量（API密钥、模型参数等敏感信息）
# - configs/config.yaml（应用主配置：名称、版本、服务器设置）
# - configs/models.yaml（大模型提供商配置）
# - configs/prompts.yaml（Prompt模板配置）
# - configs/logging.yaml（日志系统配置）
#
# 使用方式：
#   from src.core.config import get_config
#   cfg = get_config()  # 获取全局单例
#   api_key = cfg.api_key
#   model = cfg.default_model

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
# from src.utils.text_utils import debug_print


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


class Config:

    """统一配置管理器（单例模式）

    职责：
    1. 在初始化时自动加载所有配置文件和环境变量
    2. 提供统一的属性访问接口，隐藏底层配置来源差异
    3. 支持配置热更新（reload方法）
    4. 提供常用路径的便捷访问

    设计原则：
    - 环境变量优先于 YAML 配置（便于不同环境灵活覆盖）
    - 所有路径均转换为绝对路径（避免相对路径问题）
    - 缺失配置时提供合理默认值

    示例：
        >>> cfg = get_config()
        >>> print(cfg.app_name)        # 'AI写作助手系统'
        >>> print(cfg.api_key)         # 从 .env 读取
        >>> print(cfg.server_port)     # 8000
        >>> cfg.ensure_dirs()          # 创建所需目录
    """

    # 项目根目录（configs/ 的上一级）
    # 通过 __file__ 动态计算，确保无论从哪里导入都能正确定位
    _BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    _CONFIG_DIR: Path = _BASE_DIR / "configs"

    # debug_print(f"配置目录: {_CONFIG_DIR}")

    def __init__(self) -> None:
        """初始化配置管理器

        执行顺序：
        1. 加载 .env 文件到环境变量（python-dotenv）
        2. 依次加载四个 YAML 配置文件到内存字典

        注意：如果某个 YAML 文件不存在会抛出 FileNotFoundError
        """
        # debug_print(f"正在初始化配置管理器，项目根目录: {self._BASE_DIR}")

        # 1. 加载 .env 环境变量
        #    python-dotenv 会将 .env 中的键值对注入到 os.environ
        env_path = self._BASE_DIR / ".env"
        debug_print(f"加载env_path环境变量文件: {env_path}")
        load_dotenv(env_path, encoding="utf-8")

        # 2. 加载各 YAML 配置文件
        #    每个文件的内容被解析为字典，存储在实例变量中
        self._app_config: Dict[str, Any] = self._load_yaml("config.yaml")
        self._models_config: Dict[str, Any] = self._load_yaml("models.yaml")
        self._prompts_config: Dict[str, Any] = self._load_yaml("prompts.yaml")
        self._logging_config: Dict[str, Any] = self._load_yaml("logging.yaml")

        debug_print("配置管理器初始化完成")

    # ------------------------------------------------------------------ #
    #                         YAML 加载工具                               #
    # ------------------------------------------------------------------ #
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """加载指定 YAML 配置文件并返回字典

        Args:
            filename: YAML 文件名（相对于 configs/ 目录）

        Returns:
            解析后的字典，如果文件为空或非字典类型则返回空字典

        Raises:
            FileNotFoundError: 当配置文件不存在时抛出
        """
        filepath = self._CONFIG_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # yaml.safe_load 对于空文件返回 None，需要处理
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------ #
    #                         环境变量快捷访问                            #
    # ------------------------------------------------------------------ #
    # 以下属性直接从 .env 文件读取，用于敏感信息和运行时可变的配置
    # 优先级：环境变量 > 代码默认值
    # ------------------------------------------------------------------ #
    @property
    def api_key(self) -> str:
        """阿里云百炼 API Key（必填）

        Returns:
            API 密钥字符串

        Raises:
            EnvironmentError: 当 DASHSCOPE_API_KEY 未设置时抛出
        """
        value = os.getenv("DASHSCOPE_API_KEY", "")
        if not value:
            raise EnvironmentError("环境变量 DASHSCOPE_API_KEY 未设置，请检查 .env 文件")
        return value

    @property
    def base_url(self) -> str:
        """大模型 API 端点 URL

        默认指向阿里云百炼的兼容 OpenAI 接口
        可通过 .env 中的 DASHSCOPE_BASE_URL 自定义
        """
        return os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    @property
    def default_model(self) -> str:
        """默认使用的大模型名称

        优先级：
        1. .env 中的 DEFAULT_MODEL 环境变量
        2. models.yaml 中的 models.default 配置
        3. 兜底默认值 "qwen-plus"

        可选值示例：qwen-plus, qwen-turbo, qwen-max
        具体可用模型参考阿里云百炼文档
        """
        # 优先从环境变量获取
        env_model = os.getenv("DEFAULT_MODEL")
        if env_model:
            return env_model

        # 其次从配置文件获取
        yaml_model = self._models_config.get("models", {}).get("default")
        if yaml_model:
            return yaml_model

        # 兜底默认值
        return "qwen-plus"

    @property
    def temperature(self) -> float:
        """温度参数 (0.0-2.0)

        控制输出的随机性和创造性：
        - 值越低（接近0）：输出更确定、保守
        - 值越高（接近2）：输出更多样、有创意
        推荐值：0.7（平衡创造性和稳定性）
        """
        return float(os.getenv("TEMPERATURE", "0.7"))

    @property
    def max_tokens(self) -> int:
        """最大生成 token 数

        限制模型单次输出的最大长度。
        注意：token 不等于字符数，中文约 1 token ≈ 1-2 字符
        """
        return int(os.getenv("MAX_TOKENS", "2048"))

    @property
    def top_p(self) -> float:
        """核采样参数 (0.0-1.0)

        只考虑累积概率超过该值的 token。
        - 值越低：从更少的高概率词中选择，输出更集中
        - 值越高：考虑更多低概率词，输出更多样
        与 temperature 配合使用，通常不建议同时调整两者
        """
        return float(os.getenv("TOP_P", "0.9"))

    @property
    def frequency_penalty(self) -> float:
        """频率惩罚 (-2.0 到 2.0)

        根据 token 在已生成文本中出现的频率进行惩罚：
        - 正值：降低重复内容出现的概率
        - 负值：鼓励重复
        适用于减少啰嗦和重复表达
        """
        return float(os.getenv("FREQUENCY_PENALTY", "0.0"))

    @property
    def presence_penalty(self) -> float:
        """存在惩罚 (-2.0 到 2.0)

        根据 token 是否已在文本中出现过进行惩罚：
        - 正值：鼓励模型谈论新话题
        - 负值：鼓励重复已有话题
        适用于增加内容的多样性和覆盖面
        """
        return float(os.getenv("PRESENCE_PENALTY", "0.0"))

    # ------------------------------------------------------------------ #
    #                         文本分割器配置                              #
    # ------------------------------------------------------------------ #
    # 从 configs/config.yaml 的 text_splitter 节读取默认参数
    # ------------------------------------------------------------------ #
    @property
    def chunk_size(self) -> int:
        """文本块的目标字符数（默认 500）

        每个分割后的文本块尽量接近该值。
        较小的值有利于向量检索精度，较大的值保留更多上下文。
        """
        return int(
            self._app_config.get("text_splitter", {}).get("chunk_size", 500)
        )

    @property
    def chunk_overlap(self) -> int:
        """相邻文本块的重叠字符数（默认 50）

        重叠可保留上下文连贯性，防止语义在边界处被截断。
        必须小于 chunk_size。
        """
        return int(
            self._app_config.get("text_splitter", {}).get("chunk_overlap", 50)
        )

    @property
    def max_chunk_size(self) -> int:
        """文本块的最大字符数上限（默认 1000）

        当单个自然段落超过此值时会被强制截断。
        必须大于等于 chunk_size。
        """
        return int(
            self._app_config.get("text_splitter", {}).get(
                "max_chunk_size", 1000)
        )

    # ------------------------------------------------------------------ #
    #                         向量数据库配置                              #
    # ------------------------------------------------------------------ #
    @property
    def chroma_persist_dir(self) -> str:
        """ChromaDB 向量数据库持久化目录（绝对路径）

        ChromaDB 将向量嵌入和元数据持久化存储在此目录。
        首次使用时会自动创建该目录。

        Returns:
            绝对路径字符串，例如：D:\\AiStudy\\Worker1\\data\\vectors
        """
        raw = os.getenv("CHROMA_PERSIST_DIR", "./data/vectors")
        # 如果是相对路径，转换为基于项目根目录的绝对路径
        return str(self._BASE_DIR / raw) if not os.path.isabs(raw) else raw

    @property
    def embedding_model(self) -> str:
        """Embedding 模型名称

        用于将文本转换为向量表示。
        优先级：环境变量 EMBEDDING_MODEL > 默认值 text-embedding-v3

        常用可选值（阿里云百炼）：
        - text-embedding-v3: 最新版，1024维，推荐
        - text-embedding-v2: 上一版，1536维
        - text-embedding-v1: 初版，1536维
        """
        return os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

    @property
    def embedding_dimensions(self) -> int:
        """Embedding 向量维度

        text-embedding-v3 默认 1024 维
        text-embedding-v2/v1 默认 1536 维
        """
        return int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

    @property
    def chroma_collection_name(self) -> str:
        """ChromaDB 默认集合名称

        一个集合类似一张表，存储同一批文档的向量。
        可通过环境变量 CHROMA_COLLECTION_NAME 自定义
        """
        return os.getenv("CHROMA_COLLECTION_NAME", "documents")

    # ------------------------------------------------------------------ #
    #                         检索器配置                                  #
    # ------------------------------------------------------------------ #
    # 从 configs/config.yaml 的 retriever 节读取，可被环境变量覆盖
    # ------------------------------------------------------------------ #
    @property
    def retriever_top_k(self) -> int:
        """检索器默认返回结果数量

        优先级：环境变量 RETRIEVER_TOP_K > config.yaml > 默认值 5
        """
        env_val = os.getenv("RETRIEVER_TOP_K")
        if env_val is not None:
            return int(env_val)
        return int(
            self._app_config.get("retriever", {}).get("top_k", 5)
        )

    @property
    def retriever_score_threshold(self) -> float:
        """检索器相似度得分阈值（余弦距离，越小越相似）

        仅返回得分低于此阈值的结果。
        设为 None 时不过滤。
        优先级：环境变量 RETRIEVER_SCORE_THRESHOLD > config.yaml > 默认值 0.7
        """
        env_val = os.getenv("RETRIEVER_SCORE_THRESHOLD")
        if env_val is not None:
            return float(env_val)
        raw = self._app_config.get(
            "retriever", {}).get("score_threshold", 0.7)
        if raw is None:
            return None  # type: ignore[return-value]
        return float(raw)

    @property
    def retriever_deduplicate(self) -> bool:
        """是否启用检索结果去重

        优先级：环境变量 RETRIEVER_DEDUPLICATE > config.yaml > 默认值 True
        """
        env_val = os.getenv("RETRIEVER_DEDUPLICATE")
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes")
        return bool(
            self._app_config.get("retriever", {}).get(
                "deduplicate", True)
        )

    @property
    def retriever_dedup_similarity(self) -> float:
        """去重相似度阈值（0~1，文本重合度超过此值则合并）

        优先级：环境变量 RETRIEVER_DEDUP_SIMILARITY > config.yaml > 默认值 0.9
        """
        env_val = os.getenv("RETRIEVER_DEDUP_SIMILARITY")
        if env_val is not None:
            return float(env_val)
        return float(
            self._app_config.get("retriever", {}).get(
                "dedup_similarity", 0.9)
        )

    @property
    def retriever_rerank_enabled(self) -> bool:
        """是否启用 LLMRerank 检索后处理

        优先级：环境变量 RETRIEVER_RERANK_ENABLED > config.yaml > 默认值 False
        """
        env_val = os.getenv("RETRIEVER_RERANK_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes")
        return bool(
            self._app_config.get("retriever", {}).get(
                "rerank", {}).get("enabled", False)
        )

    @property
    def retriever_rerank_top_n(self) -> int:
        """LLMRerank 重排序后保留的 Top N 结果数

        优先级：环境变量 RETRIEVER_RERANK_TOP_N > config.yaml > 默认值 3
        """
        env_val = os.getenv("RETRIEVER_RERANK_TOP_N")
        if env_val is not None:
            return int(env_val)
        return int(
            self._app_config.get("retriever", {}).get(
                "rerank", {}).get("top_n", 3)
        )

    # ------------------------------------------------------------------ #
    #                         生成器配置                                  #
    # ------------------------------------------------------------------ #
    # 从 configs/config.yaml 的 generator 节读取默认参数
    # ------------------------------------------------------------------ #
    @property
    def generator_max_context_chars(self) -> int:
        """生成器检索上下文最大字符数

        将检索结果拼接为 Prompt 上下文时的字符上限，超出则截断。
        优先级：环境变量 GENERATOR_MAX_CONTEXT_CHARS > config.yaml > 默认值 3000
        """
        env_val = os.getenv("GENERATOR_MAX_CONTEXT_CHARS")
        if env_val is not None:
            return int(env_val)
        return int(
            self._app_config.get("generator", {}).get(
                "max_context_chars", 3000)
        )

    @property
    def generator_default_system_prompt(self) -> str:
        """生成器默认系统提示词

        当 prompts.yaml 中无对应角色时使用的兑底系统提示词。
        优先级：config.yaml > 默认值
        """
        return self._app_config.get("generator", {}).get(
            "default_system_prompt",
            "你是一个专业的写作助手，请根据提供的参考资料回答用户问题。"
        )

    @property
    def generator_include_sources(self) -> bool:
        """是否在生成回答中附带引用来源标注

        优先级：环境变量 GENERATOR_INCLUDE_SOURCES > config.yaml > 默认值 True
        """
        env_val = os.getenv("GENERATOR_INCLUDE_SOURCES")
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes")
        return bool(
            self._app_config.get("generator", {}).get(
                "include_sources", True)
        )

    @property
    def generator_temperature(self) -> Optional[float]:
        """生成器温度参数（None 时使用 LLMClient 默认值）

        优先级：环境变量 GENERATOR_TEMPERATURE > config.yaml > None
        """
        env_val = os.getenv("GENERATOR_TEMPERATURE")
        if env_val is not None:
            return float(env_val)
        raw = self._app_config.get("generator", {}).get("temperature")
        return float(raw) if raw is not None else None

    @property
    def generator_max_tokens(self) -> Optional[int]:
        """生成器单次最大 token 数（None 时使用 LLMClient 默认值）

        优先级：环境变量 GENERATOR_MAX_TOKENS > config.yaml > None
        """
        env_val = os.getenv("GENERATOR_MAX_TOKENS")
        if env_val is not None:
            return int(env_val)
        raw = self._app_config.get("generator", {}).get("max_tokens")
        return int(raw) if raw is not None else None

    # ------------------------------------------------------------------ #
    #                         Agents 配置                                #
    # ------------------------------------------------------------------ #
    # 以下属性从 configs/config.yaml 的 agents 配置节读取
    # ------------------------------------------------------------------ #

    @property
    def agents_pipeline(self) -> list:
        """流水线默认阶段顺序

        来源：config.yaml 的 agents.pipeline 字段
        默认：["research", "write", "edit"]
        """
        return self._app_config.get("agents", {}).get(
            "pipeline", ["research", "write", "edit"]
        )

    @property
    def agents_max_reflections(self) -> Optional[int]:
        """Agent 最大反思迭代次数（None 时使用 Agent 内部默认值 2）

        来源：config.yaml 的 agents.max_reflections 字段
        """
        raw = self._app_config.get("agents", {}).get("max_reflections")
        return int(raw) if raw is not None else None

    @property
    def agents_research_top_k(self) -> Optional[int]:
        """研究 Agent 默认检索结果数量（None 时使用 retriever 的 top_k）

        来源：config.yaml 的 agents.research_top_k 字段
        """
        raw = self._app_config.get("agents", {}).get("research_top_k")
        return int(raw) if raw is not None else None

    @property
    def agents_write_style(self) -> Optional[str]:
        """写作 Agent 默认写作风格（None 时不指定）

        来源：config.yaml 的 agents.write_style 字段
        """
        return self._app_config.get("agents", {}).get("write_style")

    @property
    def agents_edit_mode(self) -> str:
        """编辑 Agent 默认编辑模式

        可选值：full / grammar / expression / structure
        来源：config.yaml 的 agents.edit_mode 字段
        默认：full
        """
        return self._app_config.get("agents", {}).get("edit_mode", "full")

    @property
    def agents_edit_quality_threshold(self) -> float:
        """编辑 Agent 反思质量阈值（0.0~1.0）

        来源：config.yaml 的 agents.edit_quality_threshold 字段
        默认：0.75
        """
        return float(
            self._app_config.get("agents", {}).get(
                "edit_quality_threshold", 0.75
            )
        )

    @property
    def agents_validation_enabled(self) -> bool:
        """是否启用研究结果验证（条件分支工作流）

        启用后在 research 和 write 之间插入 validate 节点，
        使用 LLM 评估研究结果是否充分，不充分则回到 research。

        优先级：环境变量 AGENTS_VALIDATION_ENABLED > config.yaml > 默认值 False
        """
        env_val = os.getenv("AGENTS_VALIDATION_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes")
        return bool(
            self._app_config.get("agents", {}).get(
                "validation", {}).get("enabled", False)
        )

    @property
    def agents_validation_max_retries(self) -> int:
        """验证最大重试次数（验证不通过时回到 research 的次数上限）

        来源：config.yaml 的 agents.validation.max_retries 字段
        默认：2
        """
        env_val = os.getenv("AGENTS_VALIDATION_MAX_RETRIES")
        if env_val is not None:
            return int(env_val)
        return int(
            self._app_config.get("agents", {}).get(
                "validation", {}).get("max_retries", 2)
        )

    @property
    def agents_validation_quality_threshold(self) -> float:
        """验证通过的质量阈值（0.0~1.0，LLM 评分归一化后高于此值视为通过）

        来源：config.yaml 的 agents.validation.quality_threshold 字段
        默认：0.6
        """
        env_val = os.getenv("AGENTS_VALIDATION_QUALITY_THRESHOLD")
        if env_val is not None:
            return float(env_val)
        return float(
            self._app_config.get("agents", {}).get(
                "validation", {}).get("quality_threshold", 0.6)
        )

    # ------------------------------------------------------------------ #
    #                         应用配置                                    #
    # ------------------------------------------------------------------ #
    # 以下属性从 configs/config.yaml 读取，也可被环境变量覆盖
    # ------------------------------------------------------------------ #
    @property
    def app_name(self) -> str:
        """应用名称，来自 config.yaml 的 app.name 字段"""
        return self._app_config.get("app", {}).get("name", "AI写作助手系统")

    @property
    def app_version(self) -> str:
        """应用版本号，来自 config.yaml 的 app.version 字段"""
        return self._app_config.get("app", {}).get("version", "0.1.0")

    @property
    def debug(self) -> bool:
        """调试模式开关

        优先级：
        1. .env 中的 DEBUG 环境变量（true/false/1/0/yes/no）
        2. config.yaml 中的 app.debug 字段
        3. 默认值 False

        生产环境务必设置为 False，防止敏感信息泄露
        """
        env_val = os.getenv("DEBUG")
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes")
        return self._app_config.get("app", {}).get("debug", False)

    @property
    def app_env(self) -> str:
        """运行环境标识

        可选值：
        - development：开发环境（详细日志、热重载）
        - staging：预发布环境
        - production：生产环境（优化性能、关闭调试）
        """
        return os.getenv("APP_ENV", "development")

    @property
    def log_level(self) -> str:
        """日志级别（大写字符串）

        可选值：DEBUG < INFO < WARNING < ERROR < CRITICAL
        只有等于或高于设定级别的日志才会被记录
        """
        return os.getenv("LOG_LEVEL", "INFO").upper()

    # ------------------------------------------------------------------ #
    #                         服务器配置                                  #
    # ------------------------------------------------------------------ #
    # 用于 FastAPI/Uvicorn Web 服务的启动参数
    # ------------------------------------------------------------------ #
    @property
    def server_host(self) -> str:
        """服务器监听地址

        - 0.0.0.0：接受所有网络接口的连接（局域网可访问）
        - 127.0.0.1：仅本地访问
        """
        return self._app_config.get("server", {}).get("host", "0.0.0.0")

    @property
    def server_port(self) -> int:
        """服务器监听端口号

        默认 8000，如被占用可在 config.yaml 中修改
        """
        return int(self._app_config.get("server", {}).get("port", 8000))

    # ------------------------------------------------------------------ #
    #                         模型配置                                    #
    # ------------------------------------------------------------------ #
    # 从 configs/models.yaml 读取多模型提供商配置
    # 支持扩展：OpenAI、Anthropic、本地部署模型等
    # ------------------------------------------------------------------ #
    @property
    def models_config(self) -> Dict[str, Any]:
        """返回完整模型配置字典

        结构示例：
        {
            "default": "qwen-plus",
            "providers": {
                "openai": {"model": "", "temperature": 0.7, ...},
                "anthropic": {"model": "", "temperature": 0.7, ...}
            }
        }
        """
        return self._models_config.get("models", {})

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """获取指定模型提供商的配置

        Args:
            provider: 提供商名称，如 "openai"、"anthropic"

        Returns:
            该提供商的配置字典，包含 model、temperature、max_tokens 等
            如果提供商不存在，返回空字典
        """
        return self.models_config.get("providers", {}).get(provider, {})

    # ------------------------------------------------------------------ #
    #                         Prompt 配置                                 #
    # ------------------------------------------------------------------ #
    # 从 configs/prompts.yaml 读取不同角色的 Prompt 模板
    # 角色分为：writing（写作）、editing（编辑）、research（研究）
    # ------------------------------------------------------------------ #
    @property
    def prompts_config(self) -> Dict[str, Any]:
        """返回完整 Prompt 配置字典

        结构示例：
        {
            "writing": {"system": "...", "user_template": "..."},
            "editing": {"system": "...", "user_template": "..."},
            "research": {"system": "...", "user_template": "..."}
        }
        """
        return self._prompts_config.get("prompts", {})

    def get_prompt(self, role: str) -> Dict[str, str]:
        """获取指定角色的 Prompt 配置

        Args:
            role: 角色名称，可选值：
                  - "writing"：写作 Agent 的系统提示词
                  - "editing"：编辑 Agent 的系统提示词
                  - "research"：研究 Agent 的系统提示词

        Returns:
            包含 system 和 user_template 的字典
            如果角色不存在，返回空字典
        """
        return self.prompts_config.get(role, {})

    def get_system_prompt(self, role: str) -> str:
        """获取指定角色的系统提示词
        
        Args:
            role: 角色名称（writing/editing/research 等）
            
        Returns:
            系统提示词字符串（未找到时返回空字符串）
        """
        prompt_dict = self.get_prompt(role)
        return prompt_dict.get("system", "")

    def get_user_template(self, role: str) -> str:
        """获取指定角色的用户模板
        
        Args:
            role: 角色名称（writing/editing/research 等）
            
        Returns:
            用户模板字符串（未找到时返回空字符串）
        """
        prompt_dict = self.get_prompt(role)
        return prompt_dict.get("user_template", "")

    # ------------------------------------------------------------------ #
    #                         日志配置                                    #
    # ------------------------------------------------------------------ #
    # 从 configs/logging.yaml 读取，用于 Python logging 模块初始化
    # ------------------------------------------------------------------ #
    @property
    def logging_config(self) -> Dict[str, Any]:
        """返回完整日志配置字典

        该字典可直接传递给 logging.config.dictConfig() 进行日志系统初始化。
        包含 formatter、handler、root logger 等完整配置。
        """
        return self._logging_config

    # ------------------------------------------------------------------ #
    #                         路径工具                                    #
    # ------------------------------------------------------------------ #
    # 提供项目中常用目录的绝对路径，避免硬编码路径字符串
    # 所有路径均为 pathlib.Path 对象，支持跨平台操作
    # ------------------------------------------------------------------ #
    @property
    def base_dir(self) -> Path:
        """项目根目录（绝对路径）

        例如：D:\\AiStudy\\Worker1
        """
        return self._BASE_DIR

    @property
    def data_dir(self) -> Path:
        """data/ 目录（存放所有数据文件）"""
        return self._BASE_DIR / "data"

    @property
    def documents_dir(self) -> Path:
        """用户上传的原始参考文档目录

        支持格式：PDF、Word (.docx)、纯文本 (.txt)
        系统会自动对这些文档进行解析和向量化
        """
        return self.data_dir / "documents"

    @property
    def outputs_dir(self) -> Path:
        """AI 生成的写作成果输出目录

        所有生成的文章、报告、文案都会保存到此目录
        """
        return self.data_dir / "outputs"

    @property
    def templates_dir(self) -> Path:
        """写作模板文件目录

        可存放预定义的写作模板，供用户快速选择
        """
        return self.data_dir / "templates"

    # ------------------------------------------------------------------ #
    #                         辅助方法                                    #
    # ------------------------------------------------------------------ #
    def ensure_dirs(self) -> None:
        """确保运行时所需的目录存在

        会在以下目录不存在时自动创建（包括父目录）：
        - data/documents/  （用户上传的文档）
        - data/outputs/    （AI 生成的内容）
        - data/templates/  （写作模板）
        - data/vectors/    （ChromaDB 向量数据库）

        建议在应用启动时调用此方法，避免因目录缺失导致运行时错误
        """
        for d in [self.documents_dir, self.outputs_dir, self.templates_dir,
                  Path(self.chroma_persist_dir)]:
            d.mkdir(parents=True, exist_ok=True)

    def reload(self) -> None:
        """重新加载所有配置文件（热更新）

        适用场景：
        - 修改了 .env 或 YAML 配置文件后，无需重启应用
        - 在开发调试时快速应用配置变更

        注意：调用后所有基于旧配置的缓存可能失效，需谨慎使用
        """
        self.__init__()

    def __repr__(self) -> str:
        """友好的字符串表示，便于调试和日志记录"""
        return (
            f"<Config app={self.app_name!r} v{self.app_version} "
            f"env={self.app_env} model={self.default_model}>"
        )


# ------------------------------------------------------------------ #
#                     全局单例 & 便捷函数                               #
# ------------------------------------------------------------------ #
# 采用单例模式，确保整个应用中只有一个 Config 实例
# 避免重复加载配置文件，提升性能并保持配置一致性
# ------------------------------------------------------------------ #
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置单例（懒加载）

    首次调用时会创建 Config 实例并加载所有配置。
    后续调用直接返回同一实例，保证配置一致性。

    Returns:
        Config 单例对象

    示例：
        >>> cfg = get_config()
        >>> print(cfg.api_key)
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reload_config() -> Config:
    """强制重新加载配置并返回新实例

    适用场景：
    - 配置文件被外部修改后需要立即生效
    - 测试环境中需要切换不同配置

    Returns:
        新的 Config 实例（同时替换全局单例）

    注意：
        调用后之前持有的旧 Config 引用将失效，
        应重新调用 get_config() 获取最新实例
    """
    global _config_instance
    _config_instance = Config()
    return _config_instance
