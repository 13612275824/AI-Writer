# LangChain & LlamaIndex 详细学习计划

> 基于项目课程资料（第10课 LlamaIndex + 第11课 LangChain）及 Worker2 实战项目，制定以下系统学习路线。
> 建议学习周期：**4～6 周**，每天投入 1.5～2 小时。

---

## 第一阶段：基础概念与环境准备（第 1 周）

### 学习目标
理解大语言模型开发框架（SDK）的核心价值，搭建好开发环境。

### 1.1 理解 SDK 的价值
- **学习内容**：
  - SDK（Software Development Kit）的概念与意义
  - 大模型开发框架的三大核心价值：第三方能力抽象、常用方案封装、底层实现封装
  - 好框架的四大特点：可靠性、可维护性、可扩展性、低学习成本
- **参考资料**：
  - 第10课 notebook "大语言模型开发框架的价值是什么？" 章节
  - [什么是 SDK](https://aws.amazon.com/cn/what-is/sdk/)
  - [SDK 和 API 的区别](https://aws.amazon.com/cn/compare/the-difference-between-sdk-and-api/)

### 1.2 环境搭建
- **任务清单**：
  - [ ] 安装 Python 3.10+ 环境
  - [ ] 创建虚拟环境：`python -m venv .venv`
  - [ ] 安装 LangChain：`pip install langchain langchain-openai langchain-community langchain-text-splitters`
  - [ ] 安装 LlamaIndex：`pip install llama-index`
  - [ ] 安装 Jupyter：`pip install jupyter`
  - [ ] 配置 API Key（OpenAI / DeepSeek 等）
  - [ ] 安装向量数据库相关：`pip install faiss-cpu qdrant-client`
- **验证方式**：成功运行 notebook 中第一个"4行代码实现RAG"的示例

### 1.3 两大框架对比概览
| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| **侧重点** | LLM 交互封装 | 数据交互封装 |
| **强项** | Prompt/LLM/OutputParser/LCEL | 数据加载/切割/索引/检索/排序 |
| **RAG 能力** | 工具相对粗糙 | 工具丰富且精细 |
| **工作流** | LangGraph（状态机） | Workflow（事件驱动） |
| **部署** | LangServe | LlamaDeploy |
| **监控** | LangSmith | 三方合作 |

---

## 第二阶段：LangChain 系统学习（第 2～3 周）

### 学习目标
掌握 LangChain 五大核心组件，能独立构建 LCEL Chain 和 LangGraph 工作流。

### 2.1 模型 I/O 封装（3天）

#### 2.1.1 ChatModel 统一接口
- **学习内容**：
  - `init_chat_model()` 统一初始化不同模型（OpenAI、DeepSeek 等）
  - `AIMessage`、`HumanMessage`、`SystemMessage` 消息类型
  - 多轮对话 Session 封装
  - 流式输出 `model.stream()`
- **代码练习**：
  - [ ] 分别用 OpenAI 和 DeepSeek 模型完成多轮对话
  - [ ] 实现流式输出并对比两种模型的体验
- **对应课件**：第11课 notebook 1.1 节

#### 2.1.2 Prompt 模板
- **学习内容**：
  - `PromptTemplate` —— 自定义变量模板
  - `ChatPromptTemplate` —— 多轮对话模板
  - `MessagesPlaceholder` —— 将多轮对话变成模板
  - 从文件加载 Prompt 模板 `PromptTemplate.from_file()`
- **核心概念**：把 Prompt 模板看作**带有参数的函数**
- **代码练习**：
  - [ ] 创建一个客服场景的 ChatPromptTemplate
  - [ ] 使用 MessagesPlaceholder 实现带历史的对话
- **对应课件**：第11课 notebook 1.2 节

#### 2.1.3 结构化输出
- **学习内容**：
  - `with_structured_output()` 直接输出 Pydantic 对象
  - JSON Schema 格式的结构化输出
  - `OutputParser` 系列：`JsonOutputParser`、`PydanticOutputParser`
  - `OutputFixingParser` 利用大模型自动纠错
- **代码练习**：
  - [ ] 实现一个从自然语言中提取日期信息的结构化输出
  - [ ] 对比三种输出方式的优劣
- **对应课件**：第11课 notebook 1.3 节

#### 2.1.4 Function Calling
- **学习内容**：
  - `@tool` 装饰器定义工具
  - `bind_tools()` 绑定工具到模型
  - 工具调用的消息流转（tool_calls → tool_msg → 最终回复）
- **代码练习**：
  - [ ] 实现一个简单的计算器工具（add + multiply）
  - [ ] 完成完整的 Function Calling 流程
- **对应课件**：第11课 notebook 1.4 节

### 2.2 数据连接封装（2天）

#### 2.2.1 文档加载与处理
- **学习内容**：
  - `Document Loaders`：`PyMuPDFLoader` 等文件加载器
  - `Document Transformers`：`RecursiveCharacterTextSplitter` 文本切分
  - chunk_size 和 chunk_overlap 的含义与调优
- **代码练习**：
  - [ ] 加载一个 PDF 文件并切分为文档块
- **对应课件**：第11课 notebook 2.1～2.2 节

#### 2.2.2 向量数据库与检索
- **学习内容**：
  - `OpenAIEmbeddings` 文本向量化
  - `FAISS` 向量存储
  - `as_retriever()` 构建检索器
  - `search_kwargs` 控制检索数量
- **代码练习**：
  - [ ] 构建一个基于 FAISS 的文档检索系统
- **对应课件**：第11课 notebook 2.3 节

### 2.3 LCEL 链式调用（3天）⭐ 重点

#### 2.3.1 LCEL 基础
- **学习内容**：
  - LCEL（LangChain Expression Language）管道语法 `|`
  - `RunnablePassthrough` 透传输入
  - Pipeline 式调用：`input | prompt | llm | parser`
  - LCEL 的八大亮点：流支持、异步、并行、重试/回退、中间结果、模式、LangSmith、LangServe
- **代码练习**：
  - [ ] 实现一个语义解析器 Chain（PromptTemplate + StructuredLLM）
- **对应课件**：第11课 notebook 4.1 节

#### 2.3.2 用 LCEL 实现 RAG
- **学习内容**：
  - 完整的 RAG Chain：`{"question": ..., "context": retriever} | prompt | llm | parser`
  - 理解 LCEL 中数据流转的方式
- **代码练习**：
  - [ ] 用 LCEL 实现一个基于 PDF 的 RAG 问答系统
- **对应课件**：第11课 notebook 4.2 节

#### 2.3.3 LCEL 高级特性（选学）
- **学习内容**：
  - 工厂模式：`configurable_alternatives` 运行时切换模型
  - 故障回退：`fallbacks`
  - 并行调用
  - 逻辑分支 / 路由
- **对应课件**：第11课 notebook 4.3 节
- **参考文档**：
  - [配置运行时变量](https://python.langchain.com/docs/how_to/configure/)
  - [故障回退](https://python.langchain.com/docs/how_to/fallbacks/)
  - [并行调用](https://python.langchain.com/docs/how_to/parallel/)

### 2.4 对话历史管理（1天）
- **学习内容**：
  - 对话历史的存储、加载与剪裁
  - `memory.db` 持久化方案
- **对应课件**：第11课 "三、对话历史管理" 概念

### 2.5 LangGraph 工作流（3天）⭐ 重点

#### 2.5.1 基础概念
- **学习内容**：
  - `StateGraph` —— 将工作流定义为状态机
  - `Node` —— 工作流节点
  - `Edge` —— 节点之间的跳转
  - `State` —— 可更新的状态（`TypedDict` + `Annotated`）
  - `START` / `END` 特殊节点
  - `compile()` 编译图
- **代码练习**：
  - [ ] 实现一个基础的 Chatbot 工作流
- **对应课件**：第11课 notebook 5.1～5.2 节

#### 2.5.2 集成 RAG
- **学习内容**：
  - 在 LangGraph 中添加检索节点
  - 节点间通过 State 传递数据
- **代码练习**：
  - [ ] 在 Chatbot 工作流中加入 RAG 检索节点
- **对应课件**：第11课 notebook 5.3 节

#### 2.5.3 条件分支与人工介入
- **学习内容**：
  - `add_conditional_edges` 条件路由
  - `interrupt()` 人工介入（转人工）
  - `Command(resume=...)` 恢复执行
  - `MemorySaver` checkpointer 状态持久化
  - `thread_id` 线程配置
- **代码练习**：
  - [ ] 实现一个"找不到答案则转人工"的 RAG 工作流
- **对应课件**：第11课 notebook 5.4 节

---

## 第三阶段：LlamaIndex 系统学习（第 3～4 周）

### 学习目标
掌握 LlamaIndex 六大核心模块，能构建功能完整的 RAG 系统和工作流。

### 3.1 数据加载 Loading（2天）

#### 3.1.1 本地文件加载
- **学习内容**：
  - `SimpleDirectoryReader` 基本用法
  - 支持的文件类型（csv/docx/pdf/pptx/md 等）
  - `recursive`、`required_exts` 参数
  - 更换文件加载器（如 `LlamaParse` 提升 PDF 解析效果）
- **代码练习**：
  - [ ] 加载不同格式的文档并对比效果
- **对应课件**：第10课 notebook 3.1 节

#### 3.1.2 Data Connectors
- **学习内容**：
  - `SimpleWebPageReader` 读取网页
  - LlamaHub 上的更多数据连接器
  - 数据库连接器
- **代码练习**：
  - [ ] 使用 WebPageReader 读取一个网页并提取内容
- **对应课件**：第10课 notebook 3.2 节
- **参考资源**：
  - [内置文件加载器](https://llamahub.ai/l/readers/llama-index-readers-file)
  - [三方数据加载器](https://docs.llamaindex.ai/en/stable/module_guides/loading/connector/modules/)

### 3.2 文本切分与解析 Chunking（2天）

#### 3.2.1 TextSplitters
- **学习内容**：
  - `TokenTextSplitter` —— 按 token 数切分
  - `SentenceSplitter` —— 尽量保证句子边界
  - `CodeSplitter` —— 根据 AST 切分代码
  - `SemanticSplitterNodeParser` —— 基于语义切分
  - `chunk_size` 和 `chunk_overlap` 参数调优
- **代码练习**：
  - [ ] 用不同 Splitter 切分同一文档，对比结果差异
- **对应课件**：第10课 notebook 4.1 节

#### 3.2.2 NodeParsers 结构化解析
- **学习内容**：
  - `HTMLNodeParser` —— 解析 HTML 文档
  - `MarkdownNodeParser` —— 解析 Markdown
  - `JSONNodeParser` —— 解析 JSON
  - `Document` → `Node` 的转换关系
- **代码练习**：
  - [ ] 解析一个网页的 HTML 结构
- **对应课件**：第10课 notebook 4.2 节

### 3.3 索引 Indexing 与检索 Retrieval（3天）⭐ 重点

#### 3.3.1 向量索引与检索
- **学习内容**：
  - `VectorStoreIndex` 构建向量索引
  - `SimpleVectorStore` 内存向量存储
  - 自定义 Vector Store（以 Qdrant 为例）
  - `StorageContext` 关联存储空间
  - `as_retriever(similarity_top_k=N)` 构建检索器
- **代码练习**：
  - [ ] 分别用内存存储和 Qdrant 构建向量检索系统
- **对应课件**：第10课 notebook 5.1 节

#### 3.3.2 多种检索方式
- **学习内容**：
  - 关键字检索：`BM25Retriever`、`KeywordTableGPTRetriever`
  - RAG-Fusion：`QueryFusionRetriever`（多 query 融合）
  - KnowledgeGraph 检索
  - SQL / Text-to-SQL 检索
- **代码练习**：
  - [ ] 实现 RAG-Fusion 检索并对比普通向量检索的效果
- **对应课件**：第10课 notebook 5.2 节

#### 3.3.3 检索后处理
- **学习内容**：
  - `LLMRerank` —— 利用大模型重排序
  - `Node Postprocessors` 模块
  - top_n 参数控制最终返回结果数
- **代码练习**：
  - [ ] 在检索后加入 Rerank 步骤，对比排序前后效果
- **对应课件**：第10课 notebook 5.3 节

### 3.4 生成回复 QA & Chat（2天）

#### 3.4.1 单轮问答 Query Engine
- **学习内容**：
  - `index.as_query_engine()` 基本用法
  - 流式输出 `streaming=True`
  - `print_response_stream()` 流式打印
- **对应课件**：第10课 notebook 6.1 节

#### 3.4.2 多轮对话 Chat Engine
- **学习内容**：
  - `index.as_chat_engine()` 多轮对话
  - `stream_chat()` 流式对话
  - `CondenseQuestionChatEngine` 压缩问题对话引擎
- **代码练习**：
  - [ ] 实现一个支持多轮对话的问答系统
- **对应课件**：第10课 notebook 6.2 节

### 3.5 底层接口：Prompt、LLM 与 Embedding（2天）

#### 3.5.1 Prompt 模板
- **学习内容**：
  - `PromptTemplate` 基本用法
  - `ChatPromptTemplate` 多轮消息模板
  - `ChatMessage` 与 `MessageRole`
- **对应课件**：第10课 notebook 7.1 节

#### 3.5.2 语言模型与 Embedding
- **学习内容**：
  - `llama_index.llms.openai.OpenAI` 模型连接
  - 连接 DeepSeek：`llama_index.llms.deepseek.DeepSeek`
  - `Settings.llm` 全局模型设置
  - `OpenAIEmbedding` 与 `Settings.embed_model`
  - `structured_predict()` 结构化预测
- **代码练习**：
  - [ ] 配置全局 LLM 和 Embedding 模型
  - [ ] 使用 DeepSeek 替换 OpenAI
- **对应课件**：第10课 notebook 7.2～7.3 节

### 3.6 完整 RAG 系统实战（2天）⭐ 综合练习

- **任务**：构建一个功能较完整的 RAG 系统，包含：
  - [ ] 加载指定目录的文件
  - [ ] 使用 Qdrant 向量数据库并持久化
  - [ ] 支持 RAG-Fusion
  - [ ] 支持检索后重排序（LLMRerank）
  - [ ] 支持多轮对话（CondenseQuestionChatEngine）
  - [ ] 使用 IngestionPipeline 处理文档
- **对应课件**：第10课 notebook 第8节

---

## 第四阶段：工作流进阶（第 4～5 周）

### 学习目标
掌握 LlamaIndex Workflow 事件驱动工作流，并理解两大框架工作流的差异。

### 4.1 LlamaIndex Workflow（3天）

#### 4.1.1 核心概念
- **学习内容**：
  - 事件驱动架构：`Event` → `step` → `Event` → ... → `StopEvent`
  - `Workflow` 基类
  - `@step` 装饰器定义步骤
  - `StartEvent` / `StopEvent` 起止事件
  - `Context` 上下文管理
  - 自定义 Event 类
- **对应课件**：第10课 notebook 9.1～9.3 节

#### 4.1.2 Text-to-SQL 工作流实战
- **学习内容**：
  - 数据准备：加载 CSV → 生成表描述 → 存入 SQLite
  - `ObjectIndex` 通过索引检索任意 Python 对象
  - `SQLTableNodeMapping` 表映射
  - `SQLRetriever` SQL 查询器
  - Text-to-SQL Prompt 与 SQL 解析
  - 完整工作流：表检索 → SQL 生成 → SQL 执行 → 自然语言回复
- **代码练习**：
  - [ ] 完整实现 Text-to-SQL 工作流
  - [ ] 使用 `draw_all_possible_flows` 可视化工作流
- **对应课件**：第10课 notebook 9.2～9.4 节

#### 4.1.3 工作流管理框架的意义
- **思考题**：
  - step 的执行有逻辑分支怎么办？
  - step 之间有循环怎么处理？
  - 多个 step 可以并行执行吗？
  - 一个 step 的触发依赖多个前置 step 的结果？
- **参考文档**：[LlamaIndex Workflow Cookbook](https://docs.llamaindex.ai/en/stable/examples/workflow/workflows_cookbook/)

### 4.2 两大框架工作流对比（1天）

| 特性 | LangGraph | LlamaIndex Workflow |
|------|-----------|---------------------|
| **驱动方式** | 状态机（State Machine） | 事件驱动（Event Driven） |
| **核心抽象** | State + Node + Edge | Event + Step + Context |
| **状态管理** | TypedDict + Annotated | Context 对象 |
| **条件路由** | `add_conditional_edges` | Event 类型自动路由 |
| **人工介入** | `interrupt()` + `Command(resume)` | 内置支持 |
| **可视化** | `draw_mermaid_png()` | `draw_all_possible_flows()` |
| **适用场景** | 通用工作流编排 | 数据密集型工作流 |

### 4.3 LangSmith 调试与监控（1天）
- **学习内容**：
  - LangSmith 平台注册与配置
  - `LANGCHAIN_API_KEY`、`LANGCHAIN_TRACING_V2` 环境变量
  - 追踪 Chain 执行过程
  - 调试与性能分析

---

## 第五阶段：综合实战与项目整合（第 5～6 周）

### 学习目标
将所学知识应用到 Worker2 项目中，完成端到端的实战项目。

### 5.1 Worker2 项目实战
- **项目路径**：`d:\AiStudy\Worker2`
- **项目结构**：
  ```
  Worker2/
  ├── src/
  │   ├── agents/      # 智能体模块
  │   ├── core/        # 核心模块（LLM/Prompt 适配器等）
  │   ├── rag/         # RAG 模块
  │   ├── utils/       # 工具模块
  │   └── writing/     # 写作模块
  ├── configs/         # 配置文件
  ├── tests/           # 测试
  └── app.py           # 应用入口
  ```
- **实践任务**：
  - [ ] 阅读 Worker2 的 RAG 模块源码，理解其架构
  - [ ] 使用 LlamaIndex 优化 RAG 模块的检索能力
  - [ ] 使用 LangChain LCEL 重构 Chain 调用逻辑
  - [ ] 为 Agents 模块添加 LangGraph 工作流支持

### 5.2 综合练习项目建议

#### 项目 A：智能文档问答系统
- 技术栈：LlamaIndex + Qdrant + DeepSeek
- 功能要求：
  - 支持多种文档格式（PDF/Word/HTML）
  - RAG-Fusion + Rerank
  - 多轮对话
  - 流式输出

#### 项目 B：多步骤智能助手
- 技术栈：LangChain + LangGraph + OpenAI
- 功能要求：
  - 工具调用（搜索、计算、文件操作）
  - 条件分支与错误回退
  - 人工介入机制
  - LangSmith 监控

#### 项目 C：Text-to-SQL 数据分析师
- 技术栈：LlamaIndex Workflow + SQLite
- 功能要求：
  - 自然语言查询数据库
  - 自动表检索与 SQL 生成
  - 结果可视化

### 5.3 学习扩展方向
- **智能体（Agent）**：
  - LangChain Agent：[文档](https://python.langchain.com/docs/tutorials/agents/)
  - LlamaIndex Agent：[文档](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)
- **RAG 评测**：[LlamaIndex 评测指南](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/)
- **生产级 RAG 优化**：[Advanced Topics](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- **LangServe 部署**：[文档](https://python.langchain.com/docs/langserve/)

---

## 附录：学习检查清单

### LangChain 核心掌握度
- [ ] 能使用 `init_chat_model` 初始化不同模型
- [ ] 能创建 `PromptTemplate` 和 `ChatPromptTemplate`
- [ ] 能实现结构化输出（Pydantic / JSON）
- [ ] 能定义和使用 Function Calling
- [ ] 能使用 Document Loader 和 TextSplitter
- [ ] 能构建 FAISS 向量存储和检索器
- [ ] **能使用 LCEL 构建完整 Chain** ⭐
- [ ] 能使用 LangGraph 构建状态机工作流 ⭐
- [ ] 能实现条件分支和人工介入

### LlamaIndex 核心掌握度
- [ ] 能使用 `SimpleDirectoryReader` 和 Data Connectors
- [ ] 能选择合适的 TextSplitter 和 NodeParser
- [ ] 能构建 VectorStoreIndex（内存 + Qdrant）
- [ ] 能实现多种检索方式（向量/BM25/RAG-Fusion）
- [ ] 能使用 LLMRerank 等后处理
- [ ] 能构建 QueryEngine 和 ChatEngine
- [ ] 能配置全局 LLM 和 Embedding
- [ ] **能构建完整的 RAG 系统** ⭐
- [ ] **能使用 Workflow 实现事件驱动工作流** ⭐

### 官方文档收藏
| 资源 | 链接 |
|------|------|
| LangChain 功能模块 | https://python.langchain.com/docs/tutorials |
| LangChain API 文档 | https://python.langchain.com/api_reference/ |
| LangChain 三方集成 | https://python.langchain.com/docs/integrations/providers/ |
| LangGraph 文档 | https://langchain-ai.github.io/langgraph/ |
| LlamaIndex Python 文档 | https://docs.llamaindex.ai/en/stable/ |
| LlamaIndex API 文档 | https://docs.llamaindex.ai/en/stable/api_reference/ |
| LlamaHub 组件库 | https://llamahub.ai/ |
| LlamaIndex 生产级 RAG | https://docs.llamaindex.ai/en/stable/optimizing/production_rag/ |
