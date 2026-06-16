# ============================================================================
# Web 应用入口  |  app.py
# ============================================================================
#
# 功能：基于 FastAPI 的 Web API 服务
#
# API 端点：
#   GET  /                      应用信息
#   GET  /health                健康检查
#   POST /api/chat              LLM 对话（支持流式）
#   POST /api/ask               RAG 知识问答
#   POST /api/write/article     文章写作
#   POST /api/write/copy        文案写作
#   POST /api/write/report      每日工作日志
#   POST /api/optimize           内容优化
#   POST /api/transfer           风格转换
#   POST /api/write/report2     报告写作
#   POST /api/agent             多 Agent 流水线
#   POST /api/documents/import  上传并导入文档
#   GET  /api/documents         列出已导入文档
#   DELETE /api/documents/{src} 删除指定文档
#
# 启动方式：
#   python app.py                    启动服务（默认端口 8000）
#   uvicorn app:app --port 8000      直接用 uvicorn
#
# 配置：
#   configs/config.yaml → server.host / server.port
# ============================================================================

from src.agents.orchestrator import get_orchestrator
from src.writing.daily_report import get_daily_report_writer
from src.writing.copywriter import get_copywriter
from src.writing.article_writer import get_article_writer
from src.writing.content_optimizer import get_content_optimizer
from src.writing.style_transfer import get_style_transfer
from src.writing.report_writer import get_report_writer
from src.rag.vector_store import get_vector_store
from src.rag.document_ingestion import get_document_loader
from src.rag.query_engine import get_generator
from src.core.llm_adapter import get_llm_client
from src.core.exceptions import ModelAPIError
from src.core.config import get_config
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ------------------------------------------------------------------ #
#                    请求/响应模型                                    #
# ------------------------------------------------------------------ #

class ChatRequest(BaseModel):
    """对话请求"""
    model_config = {
        "json_schema_extra": {
            "title": "对话请求",
            "examples": [
                {
                    "messages": [
                        {"role": "user", "content": "你好，请帮我写一段关于人工智能的介绍"}
                    ],
                    "role": None,
                    "stream": False
                },
                {
                    "messages": [
                        {"role": "user", "content": "请帮我润色以下文字"},
                        {"role": "assistant", "content": "好的，请把需要润色的文字发给我。"},
                        {"role": "user", "content": "人工智能是未来的发展方向，很重要。"}
                    ],
                    "role": "editor",
                    "stream": False
                }
            ]
        }
    }
    messages: List[Dict[str, str]] = Field(
        ..., description="对话消息列表，格式：[{\"role\": \"user/assistant\", \"content\": \"消息内容\"}]")
    role: Optional[str] = Field(None, description="角色预设（可选），如：writer、editor等")
    stream: bool = Field(False, description="是否启用流式输出")


class ChatResponse(BaseModel):
    """对话响应"""
    model_config = {"json_schema_extra": {"title": "对话响应"}}
    reply: str = Field(..., description="AI 回复的内容")
    model: str = Field(..., description="使用的模型名称")
    prompt_tokens: int = Field(0, description="提示词消耗的 token 数")
    completion_tokens: int = Field(0, description="生成内容消耗的 token 数")


class AskRequest(BaseModel):
    """RAG 问答请求"""
    model_config = {
        "json_schema_extra": {
            "title": "RAG 问答请求",
            "examples": [
                {
                    "query": "项目中使用了哪些技术栈？",
                    "top_k": 3,
                    "use_retrieval": True,
                    "stream": False
                },
                {
                    "query": "RAG 检索增强生成的工作原理是什么？",
                    "top_k": 5,
                    "use_retrieval": True,
                    "stream": False
                }
            ]
        }
    }
    query: str = Field(..., description="用户查询问题")
    top_k: Optional[int] = Field(None, description="返回最相关的 K 个文档片段（默认使用配置值）")
    use_retrieval: bool = Field(True, description="是否启用检索增强")
    stream: bool = Field(False, description="是否启用流式输出")


class AskResponse(BaseModel):
    """RAG 问答响应"""
    model_config = {"json_schema_extra": {"title": "RAG 问答响应"}}
    answer: str = Field(..., description="AI 生成的答案")
    sources: List[str] = Field(default_factory=list, description="参考来源文档列表")
    model: str = Field(..., description="使用的模型名称")
    elapsed_ms: float = Field(0.0, description="处理耗时（毫秒）")
    total_tokens: int = Field(0, description="总消耗 token 数")


class ArticleRequest(BaseModel):
    """文章写作请求"""
    model_config = {
        "json_schema_extra": {
            "title": "文章写作请求",
            "examples": [
                {
                    "topic": "人工智能在医疗领域的应用前景",
                    "style": "专业",
                    "requirements": "字数约1000字，包含具体案例和数据",
                    "outline": None,
                    "temperature": None,
                    "max_tokens": None
                },
                {
                    "topic": "如何培养良好的阅读习惯",
                    "style": "轻松",
                    "requirements": "面向大学生读者，语言通俗易懂",
                    "outline": "一、阅读的意义\n二、常见的阅读误区\n三、实用的阅读方法\n四、如何坚持长期阅读",
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            ]
        }
    }
    topic: str = Field(..., description="文章主题")
    style: str = Field("", description="写作风格（如：正式、轻松、专业等）")
    requirements: str = Field("", description="特殊要求或说明")
    outline: Optional[str] = Field(None, description="自定义大纲（可选）")
    temperature: Optional[float] = Field(
        None, description="生成温度参数（0.0-1.0，越高越随机）")
    max_tokens: Optional[int] = Field(None, description="最大生成 token 数")


class ArticleResponse(BaseModel):
    """文章写作响应"""
    model_config = {"json_schema_extra": {"title": "文章写作响应"}}
    article: str = Field(..., description="生成的完整文章内容")
    outline: str = Field("", description="文章大纲")
    topic: str = Field(..., description="文章主题")
    style: str = Field(..., description="使用的写作风格")
    elapsed_ms: float = Field(0.0, description="处理耗时（毫秒）")
    total_tokens: int = Field(0, description="总消耗 token 数")


class CopyRequest(BaseModel):
    """文案写作请求"""
    model_config = {
        "json_schema_extra": {
            "title": "文案写作请求",
            "examples": [
                {
                    "product_name": "星辰智能手表",
                    "copy_type": "product_description",
                    "target_audience": "25-40岁科技爱好者",
                    "brand_tone": "科技感、高端",
                    "key_selling_points": "心率监测、睡眠分析、7天续航、IP68防水",
                    "requirements": "文案控制在200字以内",
                    "temperature": None,
                    "max_tokens": None
                },
                {
                    "product_name": "鲜果日记果汁",
                    "copy_type": "ad_slogan",
                    "target_audience": "年轻白领",
                    "brand_tone": "清新、健康",
                    "key_selling_points": "100%鲜榨、无添加、当日配送",
                    "requirements": "需要5条广告语供选择",
                    "temperature": 0.8,
                    "max_tokens": 500
                }
            ]
        }
    }
    product_name: str = Field(..., description="产品或品牌名称")
    copy_type: str = Field(
        "product_description", description="文案类型（product_description、ad_slogan、social_media等）")
    target_audience: str = Field("", description="目标受众群体")
    brand_tone: str = Field("", description="品牌语调（如：专业、亲和、高端等）")
    key_selling_points: str = Field("", description="核心卖点")
    requirements: str = Field("", description="特殊要求")
    temperature: Optional[float] = Field(None, description="生成温度参数")
    max_tokens: Optional[int] = Field(None, description="最大生成 token 数")


class CopyResponse(BaseModel):
    """文案写作响应"""
    model_config = {"populate_by_name": True,
                    "json_schema_extra": {"title": "文案写作响应"}}

    content: str = Field(..., alias="copy", description="生成的文案内容")
    copy_type: str = Field(..., description="文案类型")
    product_name: str = Field(..., description="产品名称")
    target_audience: str = Field(..., description="目标受众")
    elapsed_ms: float = Field(0.0, description="处理耗时（毫秒）")
    total_tokens: int = Field(0, description="总消耗 token 数")


class ReportRequest(BaseModel):
    """每日工作日志请求"""
    model_config = {
        "json_schema_extra": {
            "title": "每日工作日志请求",
            "examples": [
                {
                    "work_items": [
                        "完成用户登录模块的接口开发",
                        "修复订单状态更新的并发问题",
                        "参加需求评审会议，确认二期功能范围"
                    ],
                    "issues": [
                        "第三方支付接口响应超时，需要增加重试机制",
                        "测试环境数据库连接池偶尔耗尽"
                    ],
                    "plans": [
                        "开发用户权限管理模块",
                        "优化首页加载性能",
                        "编写单元测试覆盖核心业务逻辑"
                    ],
                    "project_name": "电商平台2.0",
                    "author": "张三",
                    "report_date": "2025-01-15",
                    "extra_notes": "本周整体进度正常，预计周五完成第一阶段交付",
                    "temperature": None,
                    "max_tokens": None
                },
                {
                    "work_items": [
                        "整理项目技术文档",
                        "完成RAG模块的代码评审"
                    ],
                    "issues": [],
                    "plans": [
                        "继续完善API文档中文化"
                    ],
                    "project_name": "AI写作助手",
                    "author": "李四",
                    "report_date": None,
                    "extra_notes": "",
                    "temperature": None,
                    "max_tokens": None
                }
            ]
        }
    }
    work_items: Optional[List[str]] = Field(None, description="完成的工作事项列表")
    issues: Optional[List[str]] = Field(None, description="遇到的问题列表")
    plans: Optional[List[str]] = Field(None, description="明日计划列表")
    project_name: str = Field("", description="项目名称")
    author: str = Field("", description="作者姓名")
    report_date: Optional[str] = Field(None, description="报告日期（YYYY-MM-DD格式）")
    extra_notes: str = Field("", description="额外备注")
    temperature: Optional[float] = Field(None, description="生成温度参数")
    max_tokens: Optional[int] = Field(None, description="最大生成 token 数")


class ReportResponse(BaseModel):
    """每日工作日志响应"""
    model_config = {"json_schema_extra": {"title": "每日工作日志响应"}}
    report: str = Field(..., description="生成的完整工作报告")
    report_date: str = Field(..., description="报告日期")
    elapsed_ms: float = Field(0.0, description="处理耗时（毫秒）")
    total_tokens: int = Field(0, description="总消耗 token 数")


class OptimizeRequest(BaseModel):
    """内容优化请求 (Content optimization request)"""
    model_config = {"json_schema_extra": {
        "title": "内容优化请求 (Content optimization request)"}}
    content: str = Field(...,
                         description="待优化的原始文本 (Original text to optimize)")
    optimize_type: str = Field(
        "polish", description="优化类型: polish(润色)/simplify(简化)/expand(扩写)/shorten(缩写)/grammar(语法校对)")
    target_style: str = Field(
        "", description="目标风格: formal/casual/academic/professional/creative")
    focus_areas: str = Field("", description="重点关注方向 (Focus areas)")
    requirements: str = Field("", description="额外要求 (Additional requirements)")
    temperature: Optional[float] = Field(
        None, description="生成温度参数 (Generation temperature)")
    max_tokens: Optional[int] = Field(
        None, description="最大生成 token 数 (Max generation tokens)")


class OptimizeResponse(BaseModel):
    """内容优化响应 (Content optimization response)"""
    model_config = {"json_schema_extra": {
        "title": "内容优化响应 (Content optimization response)"}}
    optimized_content: str = Field(...,
                                   description="优化后的内容 (Optimized content)")
    optimize_type: str = Field(..., description="优化类型 (Optimization type)")
    target_style: str = Field("", description="目标风格 (Target style)")
    summary: str = Field("", description="优化摘要 (Optimization summary)")
    char_diff: int = Field(0, description="字符数变化 (Character count difference)")
    elapsed_ms: float = Field(0.0, description="处理耗时（毫秒）(Elapsed time in ms)")
    total_tokens: int = Field(0, description="总消耗 token 数 (Total tokens used)")


class TransferRequest(BaseModel):
    """风格转换请求 (Style transfer request)"""
    model_config = {"json_schema_extra": {
        "title": "风格转换请求 (Style transfer request)"}}
    content: str = Field(...,
                         description="待转换的原始文本 (Original text to transfer)")
    target_style: str = Field(
        ..., description="目标风格: formal/casual/academic/professional/creative/news/storytelling")
    source_style: str = Field(
        "", description="源风格（留空自动识别）(Source style, empty for auto-detect)")
    requirements: str = Field("", description="额外要求 (Additional requirements)")
    temperature: Optional[float] = Field(
        None, description="生成温度参数 (Generation temperature)")
    max_tokens: Optional[int] = Field(
        None, description="最大生成 token 数 (Max generation tokens)")


class TransferResponse(BaseModel):
    """风格转换响应 (Style transfer response)"""
    model_config = {"json_schema_extra": {
        "title": "风格转换响应 (Style transfer response)"}}
    transferred_content: str = Field(...,
                                     description="转换后的内容 (Transferred content)")
    target_style: str = Field(..., description="目标风格 (Target style)")
    source_style: str = Field("", description="源风格 (Source style)")
    char_diff: int = Field(0, description="字符数变化 (Character count difference)")
    elapsed_ms: float = Field(0.0, description="处理耗时（毫秒）(Elapsed time in ms)")
    total_tokens: int = Field(0, description="总消耗 token 数 (Total tokens used)")


class ReportWriteRequest(BaseModel):
    """报告写作请求 (Report writing request)"""
    model_config = {"json_schema_extra": {
        "title": "报告写作请求 (Report writing request)"}}
    title: str = Field(..., description="报告标题 (Report title)")
    report_type: str = Field(
        "work_summary", description="报告类型: work_summary(工作总结)/project(项目)/analysis(分析)/research(调研)")
    content: str = Field(
        "", description="用户提供的素材或要点 (User-provided materials or key points)")
    sections: str = Field(
        "", description="自定义章节结构（逗号分隔）(Custom sections, comma-separated)")
    requirements: str = Field("", description="额外要求 (Additional requirements)")
    word_count: Optional[int] = Field(
        None, description="目标字数 (Target word count)")
    temperature: Optional[float] = Field(
        None, description="生成温度参数 (Generation temperature)")
    max_tokens: Optional[int] = Field(
        None, description="最大生成 token 数 (Max generation tokens)")


class ReportWriteResponse(BaseModel):
    """报告写作响应 (Report writing response)"""
    model_config = {"json_schema_extra": {
        "title": "报告写作响应 (Report writing response)"}}
    report: str = Field(..., description="生成的报告正文 (Generated report content)")
    summary: str = Field("", description="报告摘要 (Report summary)")
    title: str = Field(..., description="报告标题 (Report title)")
    report_type: str = Field(..., description="报告类型 (Report type)")
    elapsed_ms: float = Field(0.0, description="处理耗时（毫秒）(Elapsed time in ms)")
    total_tokens: int = Field(0, description="总消耗 token 数 (Total tokens used)")


class AgentRequest(BaseModel):
    """多智能体流水线请求"""
    model_config = {
        "json_schema_extra": {
            "title": "多智能体流水线请求",
            "examples": [
                {
                    "task": "撰写一篇关于大语言模型发展趋势的技术博客文章",
                    "skip_research": False,
                    "skip_edit": False,
                    "style": "专业",
                    "edit_mode": "full"
                },
                {
                    "task": "写一篇公司简介，突出科技创新和团队实力",
                    "skip_research": True,
                    "skip_edit": False,
                    "style": "正式",
                    "edit_mode": "expression"
                },
                {
                    "task": "帮我润色以下段落：人工智能正在改变我们的生活方式",
                    "skip_research": True,
                    "skip_edit": False,
                    "style": None,
                    "edit_mode": "grammar"
                }
            ]
        }
    }
    task: str = Field(..., description="任务描述")
    skip_research: bool = Field(False, description="是否跳过研究阶段")
    skip_edit: bool = Field(False, description="是否跳过编辑阶段")
    style: Optional[str] = Field(None, description="写作风格")
    edit_mode: Optional[str] = Field(
        None, description="编辑模式（full/grammar/expression/structure）")


class AgentResponse(BaseModel):
    """多智能体流水线响应"""
    model_config = {"json_schema_extra": {"title": "多智能体流水线响应"}}
    output: str = Field(..., description="最终输出内容")
    success: bool = Field(..., description="是否执行成功")
    stage_count: int = Field(0, description="执行的阶段数")
    elapsed_ms: float = Field(0.0, description="总处理耗时（毫秒）")
    total_tokens: int = Field(0, description="总消耗 token 数")
    error: Optional[str] = Field(None, description="错误信息（如果失败）")


class DocumentInfo(BaseModel):
    """文档信息"""
    model_config = {"json_schema_extra": {"title": "文档信息"}}
    source: str = Field(..., description="文档来源路径或名称")
    chunk_count: int = Field(..., description="文本块数量")
    total_chars: int = Field(..., description="总字符数")
    file_type: str = Field("", description="文件类型")


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    model_config = {"json_schema_extra": {"title": "文档列表响应"}}
    documents: List[DocumentInfo] = Field(..., description="文档信息列表")
    collection: str = Field(..., description="集合名称")


# ------------------------------------------------------------------ #
#                    FastAPI 应用实例                                 #
# ------------------------------------------------------------------ #

# 获取配置
try:
    _cfg = get_config()
    _app_name = _cfg.app_name
    _app_version = _cfg.app_version
except Exception:
    _app_name = "AI写作助手"
    _app_version = "0.1.0"

app = FastAPI(
    title=_app_name,
    version=_app_version,
    description="AI Writing Assistant Web API - supports conversation, RAG Q&A, article writing, copywriting, report generation, and multi-agent collaboration (AI 写作助手 Web API - 支持对话、RAG问答、文章写作、文案创作、报告生成、多Agent协作)",
    docs_url=None,  # 使用自定义中文 Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_tags=[
        {"name": "System (系统)",
         "description": "System information and health check (系统信息和健康检查)"},
        {"name": "Chat (对话)",
         "description": "LLM intelligent conversation (LLM 智能对话功能)"},
        {"name": "RAG Q&A (RAG问答)",
         "description": "Knowledge base intelligent Q&A (基于知识库的智能问答)"},
        {"name": "Writing (写作)",
         "description": "Article, copywriting, and report writing (文章、文案、报告写作)"},
        {"name": "Agent (智能体)",
         "description": "Multi-agent collaborative pipeline (多智能体协作流水线)"},
        {"name": "Documents (文档管理)",
         "description": "Document upload, query, and deletion (文档上传、查询和删除)"},
    ],
)


# ------------------------------------------------------------------ #
#                    系统端点                                         #
# ------------------------------------------------------------------ #

@app.get("/", tags=["System (系统)"], summary="App Info (应用信息)", response_description="App basic info and available endpoints (应用基本信息和接口列表)")
async def root():
    """Get app basic info and available endpoints (获取应用基本信息和可用接口列表)"""
    return {
        "name": _app_name,
        "version": _app_version,
        "description": "AI写作助手系统提供多种智能写作和问答功能",
        "接口列表": {
            "健康检查": "GET /health",
            "智能对话": "POST /api/chat",
            "知识问答": "POST /api/ask",
            "文章写作": "POST /api/write/article",
            "文案创作": "POST /api/write/copy",
            "工作日志": "POST /api/write/report",
            "内容优化": "POST /api/optimize",
            "风格转换": "POST /api/transfer",
            "报告写作": "POST /api/write/report2",
            "前端界面": "GET /app",
            "智能体协作": "POST /api/agent",
            "文档管理": {
                "导入文档": "POST /api/documents/import",
                "查看文档": "GET /api/documents",
                "删除文档": "DELETE /api/documents?source=xxx",
            },
        },
        "使用提示": "访问 /app 使用 Web 界面，访问 /docs 查看交互式 API 文档",
    }


@app.get("/health", tags=["System (系统)"], summary="Health Check (健康检查)", response_description="Service running status (服务运行状态)")
async def health():
    """Health check - verify service is running (健康检查 - 验证服务是否正常运行)"""
    return {"status": "ok", "service": _app_name, "message": "服务运行正常"}


# ------------------------------------------------------------------ #
#                    前端页面                                         #
# ------------------------------------------------------------------ #

@app.get("/app", response_class=HTMLResponse, tags=["System (系统)"], summary="Web Frontend (前端界面)", response_description="HTML frontend for all features (全功能 HTML 前端操作界面)")
async def frontend():
    """Built-in Web frontend for all features (内置 Web 前端界面，支持对话、问答、写作、文案、日志、Agent 协作等全部功能)"""
    return _get_frontend_html()


def _get_frontend_html() -> str:
    """Return the complete frontend HTML content (loaded from template file)."""
    template_path = _project_root / "data" / "templates" / "frontend.html"
    try:
        html = template_path.read_text(encoding="utf-8")
        return html.replace("{{VERSION}}", _app_version)
    except FileNotFoundError:
        return "<h1>前端模板文件未找到</h1><p>请检查 data/templates/frontend.html</p>"


# ------------------------------------------------------------------ #
#                    对话端点                                         #
# ------------------------------------------------------------------ #

@app.post("/api/chat", tags=["Chat (对话)"], summary="Chat (智能对话)", description="Multi-turn conversation with AI assistant, supporting streaming output (与 AI 助手进行多轮对话，支持流式输出)", response_description="AI chat reply (AI 对话回复)")
async def chat(request: ChatRequest):
    """LLM conversation with streaming support (LLM 对话，支持流式输出)

    - stream=False: Returns complete JSON response (返回完整 JSON 响应)
    - stream=True: Returns SSE stream response (返回 SSE 流式响应)
    """
    client = get_llm_client()

    # 加载角色系统提示词
    messages = request.messages.copy()
    if request.role:
        try:
            config = get_config()
            system_prompt = config.get_system_prompt(request.role)
            if system_prompt:
                messages.insert(
                    0, {"role": "system", "content": system_prompt})
        except Exception:
            pass  # 角色不存在时使用默认

    if request.stream:
        # 流式输出
        def generate():
            full_response = ""
            for chunk in client.chat_completion_stream(messages):
                full_response += chunk
                data = json.dumps(
                    {"chunk": chunk, "done": False}, ensure_ascii=False)
                yield f"data: {data}\n\n"
            # 结束信号
            data = json.dumps({"chunk": "", "done": True,
                              "reply": full_response}, ensure_ascii=False)
            yield f"data: {data}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # 非流式
        try:
            response = client.chat_completion(messages)
            reply = response.choices[0].message.content or ""
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = getattr(
                    response.usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(
                    response.usage, "completion_tokens", 0) or 0

            return ChatResponse(
                reply=reply,
                model=client.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except ModelAPIError as e:
            raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------ #
#                    RAG 问答端点                                     #
# ------------------------------------------------------------------ #

@app.post("/api/ask", tags=["RAG Q&A (RAG问答)"], summary="Knowledge Q&A (知识问答)", description="RAG-based intelligent Q&A system with retrieval-augmented generation (基于知识库的智能问答系统，支持检索增强生成)", response_description="RAG Q&A result (RAG 问答结果)")
async def ask(request: AskRequest):
    """RAG knowledge Q&A - retrieval + LLM generation (RAG 知识问答 - 基于检索 + LLM 生成)"""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")

    gen = get_generator()

    if request.stream:
        # 流式输出
        def generate():
            full_answer = ""
            for chunk in gen.generate_stream(
                query=request.query,
                use_retrieval=request.use_retrieval,
                top_k=request.top_k,
            ):
                full_answer += chunk
                data = json.dumps(
                    {"chunk": chunk, "done": False}, ensure_ascii=False)
                yield f"data: {data}\n\n"
            data = json.dumps({"chunk": "", "done": True,
                              "answer": full_answer}, ensure_ascii=False)
            yield f"data: {data}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        try:
            result = gen.generate(
                query=request.query,
                use_retrieval=request.use_retrieval,
                top_k=request.top_k,
            )
            return AskResponse(
                answer=result.answer,
                sources=result.sources if result.has_sources else [],
                model=gen._llm_client.model,
                elapsed_ms=result.elapsed_ms,
                total_tokens=result.total_tokens,
            )
        except ModelAPIError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


# ------------------------------------------------------------------ #
#                    写作端点                                         #
# ------------------------------------------------------------------ #

@app.post("/api/write/article", response_model=ArticleResponse, tags=["Writing (写作)"], summary="Article Writing (文章写作)", description="Auto-generate structured articles based on topic (根据主题自动生成结构化文章)", response_description="Generated article content (生成的文章内容)")
async def write_article(request: ArticleRequest):
    """Article writing (文章写作)"""
    if not request.topic or not request.topic.strip():
        raise HTTPException(status_code=400, detail="文章主题不能为空")

    writer = get_article_writer()

    try:
        result = writer.write(
            topic=request.topic,
            style=request.style,
            requirements=request.requirements,
            outline=request.outline,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return ArticleResponse(
            article=result.article,
            outline=result.outline,
            topic=result.topic,
            style=result.style,
            elapsed_ms=result.elapsed_ms,
            total_tokens=result.total_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelAPIError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/write/copy", response_model=CopyResponse, tags=["Writing (写作)"], summary="Copywriting (文案创作)", description="Generate product copy, slogans, and marketing content (生成产品文案、广告语等营销内容)", response_description="Generated copy content (生成的文案内容)")
async def write_copy(request: CopyRequest):
    """Copywriting (文案写作)"""
    if not request.product_name or not request.product_name.strip():
        raise HTTPException(status_code=400, detail="产品/品牌名称不能为空")

    writer = get_copywriter()

    try:
        result = writer.write(
            product_name=request.product_name,
            copy_type=request.copy_type,
            target_audience=request.target_audience,
            brand_tone=request.brand_tone,
            key_selling_points=request.key_selling_points,
            requirements=request.requirements,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return CopyResponse(
            content=result.copy,
            copy_type=result.copy_type,
            product_name=result.product_name,
            target_audience=result.target_audience,
            elapsed_ms=result.elapsed_ms,
            total_tokens=result.total_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelAPIError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/write/report", response_model=ReportResponse, tags=["Writing (写作)"], summary="Daily Report (工作日志)", description="Auto-generate daily work log report (自动生成每日工作日志报告)", response_description="Generated work report (生成的工作报告)")
async def write_report(request: ReportRequest):
    """Generate daily work log report (生成每日工作日志报告)"""
    # 校验：至少提供一项工作内容
    if not any([request.work_items, request.issues, request.plans]):
        raise HTTPException(
            status_code=400,
            detail="至少需要提供 work_items / issues / plans 中的一项",
        )

    writer = get_daily_report_writer()

    try:
        result = writer.write(
            work_items=request.work_items,
            issues=request.issues,
            plans=request.plans,
            project_name=request.project_name,
            author=request.author,
            report_date=request.report_date,
            extra_notes=request.extra_notes,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return ReportResponse(
            report=result.report,
            report_date=result.report_date,
            elapsed_ms=result.elapsed_ms,
            total_tokens=result.total_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelAPIError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize", response_model=OptimizeResponse, tags=["Writing (写作)"], summary="Content Optimization (内容优化)", description="Polish, simplify, expand, shorten or grammar-check existing text (对已有文本进行润色、简化、扩写、缩写或语法校对)", response_description="Optimized content (优化后的内容)")
async def optimize_content(request: OptimizeRequest):
    """Content optimization - polish/simplify/expand/shorten/grammar (内容优化 - 润色/简化/扩写/缩写/语法校对)"""
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="待优化内容不能为空")

    optimizer = get_content_optimizer()

    try:
        result = optimizer.optimize(
            content=request.content,
            optimize_type=request.optimize_type,
            target_style=request.target_style,
            focus_areas=request.focus_areas,
            requirements=request.requirements,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return OptimizeResponse(
            optimized_content=result.optimized_content,
            optimize_type=result.optimize_type,
            target_style=result.target_style,
            summary=result.summary,
            char_diff=result.char_diff,
            elapsed_ms=result.elapsed_ms,
            total_tokens=result.total_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelAPIError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------ #
#                    Style Transfer 端点                              #
# ------------------------------------------------------------------ #

@app.post("/api/transfer", response_model=TransferResponse, tags=["Writing (写作)"], summary="Style Transfer (风格转换)", description="Transfer text completely from one style to another while preserving meaning (将文本从一种风格完整转换为另一种风格，保持原意不变)", response_description="Transferred content (转换后的内容)")
async def transfer_style(request: TransferRequest):
    """Style transfer - convert text between styles while preserving meaning (风格转换 - 将文本从一种风格完整转换为另一种风格)"""
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="待转换内容不能为空")

    transfer = get_style_transfer()

    try:
        result = transfer.transfer(
            content=request.content,
            target_style=request.target_style,
            source_style=request.source_style,
            requirements=request.requirements,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return TransferResponse(
            transferred_content=result.transferred_content,
            target_style=result.target_style,
            source_style=result.source_style,
            char_diff=result.char_diff,
            elapsed_ms=result.elapsed_ms,
            total_tokens=result.total_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelAPIError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------ #
#                    Report Write 端点                                #
# ------------------------------------------------------------------ #

@app.post("/api/write/report2", response_model=ReportWriteResponse, tags=["Writing (写作)"], summary="Report Writing (报告写作)", description="Generate structured professional reports: work summary, project, analysis, research (生成结构化的专业报告：工作总结/项目/分析/调研)", response_description="Generated report content (生成的报告内容)")
async def write_report2(request: ReportWriteRequest):
    """Report writing - generate structured professional reports (报告写作 - 生成结构化的专业报告)"""
    if not request.title or not request.title.strip():
        raise HTTPException(status_code=400, detail="报告标题不能为空")

    writer = get_report_writer()

    try:
        result = writer.write(
            title=request.title,
            report_type=request.report_type,
            content=request.content,
            sections=request.sections,
            requirements=request.requirements,
            word_count=request.word_count,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return ReportWriteResponse(
            report=result.report,
            summary=result.summary,
            title=result.title,
            report_type=result.report_type,
            elapsed_ms=result.elapsed_ms,
            total_tokens=result.total_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelAPIError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------ #
#                    Agent 端点                                       #
# ------------------------------------------------------------------ #

@app.post("/api/agent", response_model=AgentResponse, tags=["Agent (智能体)"], summary="Agent Pipeline (智能体协作)", description="Multi-agent pipeline: Research → Write → Edit (多智能体流水线：研究 → 写作 → 编辑)", response_description="Agent collaboration result (智能体协作结果)")
async def agent_pipeline(request: AgentRequest):
    """Multi-agent collaborative writing - Research → Write → Edit pipeline (多 Agent 协作写作 - 研究 → 写作 → 编辑 流水线)"""
    if not request.task or not request.task.strip():
        raise HTTPException(status_code=400, detail="任务描述不能为空")

    # 构建执行选项
    options = {}
    if request.skip_research:
        options["skip_research"] = True
    if request.skip_edit:
        options["skip_edit"] = True
    if request.style:
        options.setdefault("write_context", {})
        options["write_context"]["style"] = request.style
    if request.edit_mode:
        options.setdefault("edit_context", {})
        options["edit_context"]["edit_mode"] = request.edit_mode

    try:
        orch = get_orchestrator()
        result = orch.run(request.task, options=options)

        return AgentResponse(
            output=result.final_output,
            success=result.success,
            stage_count=result.stage_count,
            elapsed_ms=result.total_elapsed_ms,
            total_tokens=result.total_tokens,
            error=result.error if not result.success else None,
        )
    except Exception as e:
        return AgentResponse(
            output="",
            success=False,
            error=str(e),
        )


# ------------------------------------------------------------------ #
#                    文档管理端点                                     #
# ------------------------------------------------------------------ #

@app.post("/api/documents/import", tags=["Documents (文档管理)"], summary="Import Document (导入文档)", description="Upload document and auto-import into vector knowledge base (上传文档并自动导入到向量知识库)", response_description="Document import result (文档导入结果)")
async def import_document(file: UploadFile = File(..., description="Document file to upload (要上传的文档文件)"), collection: Optional[str] = Query(None, description="Vector DB collection name, optional (向量数据库集合名称，可选)")):
    """Upload and import document into vector database (上传并导入文档到向量数据库)"""
    # 保存上传文件到临时目录
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Load and split document using unified pipeline
        loader = get_document_loader()
        try:
            chunks = loader.load_and_split(str(file_path))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"文档加载失败: {e}")

        if not chunks:
            raise HTTPException(status_code=400, detail="无有效文本块")

        # 存入向量数据库
        store = get_vector_store()
        count = store.add_chunks(chunks, collection_name=collection)

        actual_collection = collection or store.collection_name

        return {
            "message": "导入成功",
            "filename": file.filename,
            "chunks": count,
            "collection": actual_collection,
        }


@app.get("/api/documents", response_model=DocumentListResponse, tags=["Documents (文档管理)"], summary="List Documents (查看文档列表)", description="List all documents imported into knowledge base (列出已导入到知识库的所有文档)", response_description="Document list info (文档列表信息)")
async def list_documents(collection: Optional[str] = Query(None, description="Vector DB collection name, optional (向量数据库集合名称，可选)")):
    """List imported documents (列出已导入的文档)"""
    store = get_vector_store()
    sources = store.get_sources(collection_name=collection)

    documents = [
        DocumentInfo(
            source=src["source"],
            chunk_count=src["chunk_count"],
            total_chars=src["total_chars"],
            file_type=src.get("file_type", ""),
        )
        for src in sources
    ]

    actual_collection = collection or store.collection_name

    return DocumentListResponse(
        documents=documents,
        collection=actual_collection,
    )


@app.delete("/api/documents", tags=["Documents (文档管理)"], summary="Delete Document (删除文档)", description="Delete specified document and all its text chunks from knowledge base (从知识库中删除指定文档及其所有文本块)", response_description="Deletion result (删除结果)")
async def delete_document(source: str = Query(..., description="文档来源路径或名称"), collection: Optional[str] = Query(None, description="Vector DB collection name, optional (向量数据库集合名称，可选)")):
    """Delete all text chunks of specified document (删除指定文档的所有文本块)"""
    store = get_vector_store()

    # 检查文档是否存在
    sources = store.get_sources(collection_name=collection)
    if not any(s["source"] == source for s in sources):
        raise HTTPException(status_code=404, detail=f"文档不存在: {source}")

    store.delete_by_source(source=source, collection_name=collection)

    return {"message": "删除成功", "source": source}


# ------------------------------------------------------------------ #
#                    自定义 OpenAPI 模式（中文描述）                     #
# ------------------------------------------------------------------ #

def custom_openapi():
    """自定义 OpenAPI 模式，将自动生成的英文描述替换为中文"""
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    # 将所有 422 响应描述改为中文
    for path_item in openapi_schema.get("paths", {}).values():
        for operation in path_item.values():
            if isinstance(operation, dict) and "responses" in operation:
                responses = operation["responses"]
                # 修改 422 验证错误描述
                if "422" in responses:
                    responses["422"]["description"] = "请求参数验证错误"
                # 修改 200 成功响应描述（如果没有自定义）
                if "200" in responses and responses["200"].get("description") == "Successful Response":
                    responses["200"]["description"] = "成功响应"

    # 修改组件中的模型标题为中文
    components = openapi_schema.get("components", {})
    schemas = components.get("schemas", {})

    # 重命名自动生成的上传文件 Body 模型
    body_key = "Body_import_document_api_documents_import_post"
    if body_key in schemas:
        schemas[body_key]["title"] = "文档上传参数"
        props = schemas[body_key].get("properties", {})
        if "file" in props:
            props["file"]["title"] = "文件"
            props["file"]["description"] = "要上传的文档文件"
        if "collection" in props:
            props["collection"]["title"] = "集合名称"
            props["collection"]["description"] = "向量数据库集合名称（可选）"

    # 重命名所有包含 Body_ 的自动生成模型
    for key in list(schemas.keys()):
        if key.startswith("Body_"):
            schemas[key].setdefault("title", key)
            if schemas[key]["title"] == key:
                schemas[key]["title"] = "上传参数"

    if "HTTPValidationError" in schemas:
        schemas["HTTPValidationError"]["title"] = "参数验证错误"
        if "properties" in schemas["HTTPValidationError"]:
            props = schemas["HTTPValidationError"]["properties"]
            if "detail" in props:
                props["detail"]["title"] = "错误详情"
    if "ValidationError" in schemas:
        schemas["ValidationError"]["title"] = "验证错误"
        if "properties" in schemas["ValidationError"]:
            props = schemas["ValidationError"]["properties"]
            for key, val in props.items():
                title_map = {
                    "loc": "错误位置", "msg": "错误信息",
                    "type": "错误类型", "input": "输入值",
                    "ctx": "上下文",
                }
                if key in title_map:
                    val["title"] = title_map[key]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ------------------------------------------------------------------ #
#                    自定义中文 Swagger UI 文档页                      #
# ------------------------------------------------------------------ #

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """自定义 Swagger UI 文档页面（中文界面）"""
    return HTMLResponse(
        _build_swagger_html(
            openapi_url=app.openapi_url,
            title=f"{_app_name} - API 文档",
        )
    )


def _build_swagger_html(openapi_url: str, title: str) -> str:
    """构建带中文翻译的 Swagger UI HTML 页面"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
const translations = {{
    'dom_id': '#swagger-ui',
    'deepLinking': true,
    'presets': [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
    'layout': 'BaseLayout',
    'url': '{openapi_url}',
    'translations': {{
        'components': {{
            'OnlineValidatorBadge': {{ 'online': '在线验证' }},
            'operation': {{
                'operationId': '操作ID',
                'tags': '标签',
                'tryItOut': '试一试',
                'cancel': '取消',
                'execute': '执行',
                'clear': '清除',
                'close': '关闭',
                'responses': '响应',
                'parameters': '参数',
                'required': '必填',
                'noParameters': '无参数',
                'noContent': '无内容',
                'requestBody': '请求体',
                'headers': '请求头',
                'security': '安全认证',
                'deprecated': '已弃用',
                'links': '链接',
                'noLinks': '无链接'
            }},
            'responses': {{
                'successfulResponse': '成功响应',
                'validationError': '验证错误',
                'responseCode': '响应码',
                'responseDescription': '描述',
                'responseLinks': '链接',
                'mediaType': '媒体类型',
                'controlsAcceptHeader': '控制 Accept 请求头'
            }},
            'parameters': {{
                'name': '名称',
                'description': '描述',
                'required': '必填',
                'defaultValue': '默认值',
                'type': '类型',
                'in': '位置',
                'enum': '枚举值'
            }},
            'schemas': {{
                'schemas': '数据模型',
                'expandAll': '展开全部',
                'collapseAll': '收起全部',
                'exampleValue': '示例值',
                'schema': '模型',
                'noBody': '无请求体',
                'string': '字符串',
                'number': '数字',
                'integer': '整数',
                'boolean': '布尔值',
                'array': '数组',
                'object': '对象',
                'required': '必填'
            }},
            'authBtn': {{ 'authorize': '授权', 'close': '关闭' }},
            'model': {{
                'exampleValue': '示例值',
                'schema': '模型',
                'or': '或',
                'required': '必填',
                'expandAll': '展开全部',
                'collapseAll': '收起全部'
            }},
            'servers': {{
                'servers': '服务地址',
                'noServersAvailable': '无可用服务'
            }}
        }}
    }}
}};
window.onload = function() {{
    SwaggerUIBundle(translations);
}};
</script>
</body>
</html>"""


# ------------------------------------------------------------------ #
#                    启动入口                                         #
# ------------------------------------------------------------------ #

def main():
    """启动 Web 服务"""
    import uvicorn

    cfg = get_config()
    host = cfg.server_host
    port = cfg.server_port

    print(f"\n  {_app_name} v{_app_version}")
    print(f"  启动服务器: http://{host}:{port}")
    print(f"  前端界面: http://{host}:{port}/app")
    print(f"  API 文档: http://{host}:{port}/docs\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
