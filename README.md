# Worker2 - AI写作助手 (LangChain + LlamaIndex 改造版)

## 项目简介

基于 Worker1 项目，使用 **LangChain** 和 **LlamaIndex** 框架替代原有的自实现功能模块。
Worker1 中未做任何修改，所有改造均在本项目目录下完成。

## 技术栈

| 模块 | Worker1（自实现） | Worker2（框架替代） |
|------|-------------------|---------------------|
| LLM 调用 | 自实现 LLMClient | LangChain ChatOpenAI |
| Prompt 管理 | 自实现 PromptManager | LangChain ChatPromptTemplate |
| 文档加载 | 自实现 DocumentLoader | LlamaIndex PDFReader / DocxReader |
| 文本分割 | 自实现 TextSplitter | LangChain RecursiveCharacterTextSplitter |
| 向量存储 | 自实现 VectorStore | LlamaIndex ChromaVectorStore + ChromaDB |
| 检索器 | 自实现 Retriever | LlamaIndex VectorIndexRetriever |
| 查询引擎 | 自实现 Generator | LlamaIndex RetrieverQueryEngine |
| Agent 工具 | 自实现 Tool 类 | LangChain @tool 装饰器 |
| Agent 执行 | 自实现 AgentRunner | LangChain ReAct Agent + AgentExecutor |
| 工作流编排 | 自实现 Orchestrator | LangGraph StateGraph |

## 目录结构

```
Worker2/
├── configs/                  # 配置文件（prompts.yaml 等）
├── data/                     # 数据目录
│   ├── documents/            # 待导入文档
│   └── vectors/              # ChromaDB 持久化存储
├── src/
│   ├── core/
│   │   ├── llm_adapter.py        # LangChain LLM 适配器
│   │   ├── prompt_adapter.py     # LangChain Prompt 适配器
│   │   ├── config.py             # 配置管理
│   │   └── exceptions.py         # 异常定义
│   ├── rag/
│   │   ├── document_ingestion.py # LlamaIndex 文档入库管线
│   │   ├── vector_store.py       # LlamaIndex 向量存储封装
│   │   ├── retriever.py          # LlamaIndex 检索器封装
│   │   └── query_engine.py       # LlamaIndex 查询引擎封装
│   ├── agents/
│   │   ├── tools.py              # LangChain @tool 定义
│   │   ├── researcher_agent.py   # ReAct 研究型 Agent
│   │   ├── writer_agent.py       # ReAct 写作型 Agent
│   │   ├── editor_agent.py       # ReAct 编辑型 Agent
│   │   └── orchestrator.py       # LangGraph 工作流编排
│   ├── writing/                  # 写作模块（从 Worker1 复制，未修改）
│   └── utils/                    # 工具模块（从 Worker1 复制，未修改）
├── tests/                        # 单元测试（29 个测试用例）
├── app.py                        # FastAPI Web 应用
├── cli.py                        # 命令行交互入口
├── requirements.txt              # Python 依赖
└── .env                          # 环境变量（API Key 等）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置环境变量

编辑 `.env` 文件，填入阿里云百炼 API Key：

```
DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 使用 CLI

```bash
# 交互模式
python cli.py

# 导入文档
python cli.py import data/documents/

# 知识问答
python cli.py ask "什么是深度学习？"

# Agent 协作写作
python cli.py agent "写一篇关于 AI 的文章"

# 对话模式
python cli.py chat
```

### 4. 启动 Web 服务

```bash
python -m uvicorn app:app --reload --port 8000
```

## 运行测试

```bash
python -m pytest tests/ -v
```

## 设计原则

1. **适配器模式**：所有新模块通过适配器类封装，保持与 Worker1 兼容的接口
2. **单例模式**：所有模块通过 `get_xxx()` 工厂函数懒加载，避免重复创建
3. **向后兼容**：函数名、数据结构、返回值格式与 Worker1 保持一致
4. **框架替代**：凡是 LangChain / LlamaIndex 已有的实现，均直接使用框架实现
