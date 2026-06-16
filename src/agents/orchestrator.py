# 编排器（基于 LlamaIndex Workflow）
#
# 使用 LlamaIndex 事件驱动工作流协调 Agent 流水线，
# 替代原 LangGraph StateGraph 实现。
#
# 工作流：研究 -> [验证] -> 写作 -> 编辑
# 验证节点可选：启用后通过事件路由判断研究结果是否充分，
# 不充分则回到研究阶段重新检索（带最大重试次数保护）。
#
# 核心概念对比：
# - LangGraph：状态机模式（state dict 在节点间传递）
# - LlamaIndex Workflow：事件驱动模式（Event 在 @step 方法间流转）

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from src.agents.researcher_agent import AgentResult, ResearcherAgent
from src.agents.writer_agent import WriterAgent
from src.agents.editor_agent import EditorAgent
from src.core.config import get_config
from src.core.exceptions import AgentError
from src.core.llm_adapter import get_llm_client


# ------------------------------------------------------------------ #
#                    数据结构                                          #
# ------------------------------------------------------------------ #

@dataclass
class StageResult:
    """单阶段执行结果

    属性:
        stage_name: 阶段名称（research 研究 / write 写作 / edit 编辑）
        agent_name: 执行此阶段的 Agent 名称
        output: 此阶段的输出文本
        elapsed_ms: 执行耗时（毫秒）
    """
    stage_name: str = ""
    agent_name: str = ""
    output: str = ""
    elapsed_ms: float = 0.0

    def __repr__(self) -> str:
        return (
            f"<StageResult stage={self.stage_name!r} "
            f"output_len={len(self.output)} elapsed={self.elapsed_ms:.1f}ms>"
        )


@dataclass
class OrchestratorResult:
    """流水线编排器执行结果

    兼容 Worker1 的 OrchestratorResult 接口。

    属性:
        final_output: 最终输出文本（通常为编辑后的最终版本）
        stages: 各阶段执行结果列表
        total_elapsed_ms: 总执行耗时（毫秒）
        total_tokens: 总 token 消耗
        task: 原始任务描述
        success: 是否执行成功
        error: 错误信息（如果失败）
    """
    final_output: str = ""
    stages: List[StageResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    total_tokens: int = 0
    task: str = ""
    success: bool = True
    error: str = ""

    @property
    def stage_count(self) -> int:
        """执行的阶段数量"""
        return len(self.stages)

    def __repr__(self) -> str:
        return (
            f"<OrchestratorResult stages={self.stage_count} "
            f"success={self.success} elapsed={self.total_elapsed_ms:.1f}ms>"
        )


# ------------------------------------------------------------------ #
#                    工作流事件定义                                     #
# ------------------------------------------------------------------ #
# LlamaIndex Workflow 使用事件驱动：每个 @step 接收一种 Event，
# 返回另一种 Event，运行时自动将事件路由到对应的 step。
# ------------------------------------------------------------------ #

class ResearchEvent(Event):
    """研究阶段事件：携带任务描述，触发研究 step"""
    task: str
    validation_retries: int = 0      # 当前验证重试次数（重试时携带）


class ResearchDoneEvent(Event):
    """研究完成事件：携带研究结果"""
    task: str
    research_result: str
    elapsed_ms: float
    validation_retries: int = 0


class WriteEvent(Event):
    """写作阶段事件：携带任务和研究结果"""
    task: str
    research_result: str


class WriteDoneEvent(Event):
    """写作完成事件：携带草稿"""
    task: str
    draft: str
    elapsed_ms: float


class EditEvent(Event):
    """编辑阶段事件：携带任务和草稿"""
    task: str
    draft: str


class EditDoneEvent(Event):
    """编辑完成事件：携带最终输出"""
    final_output: str
    elapsed_ms: float


# ------------------------------------------------------------------ #
#                    LlamaIndex 事件驱动工作流                           #
# ------------------------------------------------------------------ #

class AgentWorkflow(Workflow):
    """基于 LlamaIndex Workflow 的 Agent 流水线

    事件驱动工作流：
    - StartEvent → research_step → ResearchDoneEvent
    - ResearchDoneEvent → (验证通过) → WriteEvent → write_step → WriteDoneEvent
    - ResearchDoneEvent → (验证不通过) → ResearchEvent → research_step（循环）
    - WriteDoneEvent → edit_step → EditDoneEvent → StopEvent

    每个 @step 方法接收一种 Event 类型，返回下一种 Event 类型，
    LlamaIndex 运行时自动将事件路由到匹配的 step。
    """

    def __init__(
        self,
        validation_enabled: bool = False, #是否启用研究结果的质量验证环节。
        validation_max_retries: int = 2, #验证环节的最大重试次数。
        validation_quality_threshold: float = 0.6, #验证环节的评分阈值。
        **kwargs,
    ):
        super().__init__(timeout=300, verbose=False, **kwargs)
        self._validation_enabled = validation_enabled
        self._validation_max_retries = validation_max_retries
        self._validation_quality_threshold = validation_quality_threshold

        # 创建各阶段 Agent
        self._researcher = ResearcherAgent()
        self._writer = WriterAgent()
        self._editor = EditorAgent()

        # 收集阶段结果（供 OrchestratorAdapter 读取）
        self._collected_stages: List[Dict[str, Any]] = []

        # 跳过阶段标志（由 OrchestratorAdapter.run() 设置）
        self._skip_research: bool = False
        self._skip_edit: bool = False

    @step
    async def start_step(self, ev: StartEvent) -> ResearchEvent | WriteEvent:
        """入口步骤：从 StartEvent 提取任务，触发研究或直接写作"""
        task = ev.get("task", "")
        self._collected_stages = []
        print(f"[AgentWorkflow] 开始工作流，任务: {task[:50]}...")

        # 跳过研究时直接进入写作
        if self._skip_research:
            print("[AgentWorkflow] 跳过研究阶段")
            return WriteEvent(task=task, research_result="")

        return ResearchEvent(task=task)

    @step
    async def research_step(self, ev: ResearchEvent) -> ResearchDoneEvent:
        """研究步骤：执行 ResearcherAgent，产出研究结果"""
        start_time = time.time()

        # 在线程中执行同步的 Agent（避免阻塞事件循环）
        agent_result = await asyncio.to_thread(
            self._researcher.execute, ev.task
        )
        elapsed_ms = (time.time() - start_time) * 1000

        self._collected_stages.append({
            "stage": "research", "elapsed_ms": elapsed_ms,
            "tokens": agent_result.total_tokens,
        })

        print(f"[AgentWorkflow] 研究完成，耗时 {elapsed_ms:.0f}ms")

        return ResearchDoneEvent(
            task=ev.task,
            research_result=agent_result.output,
            elapsed_ms=elapsed_ms,
            validation_retries=ev.validation_retries,
        )

    @step
    async def validate_and_route(
        self, ev: ResearchDoneEvent
    ) -> WriteEvent | ResearchEvent:
        """验证与路由步骤：评估研究结果，决定进入写作或回到研究

        条件分支逻辑（对应 LangGraph 的 add_conditional_edges）：
        - 验证未启用 → 直接进入写作
        - 重试次数已达上限 → 强制进入写作
        - LLM 评分 ≥ 阈值 → 进入写作
        - LLM 评分 < 阈值 → 回到研究（重新检索）
        """
        # 验证未启用，直接进入写作
        if not self._validation_enabled:
            return WriteEvent(task=ev.task, research_result=ev.research_result)

        retries = ev.validation_retries

        # 重试次数已达上限，强制通过
        if retries >= self._validation_max_retries:
            print(
                f"[Validate] 已达最大重试次数 {self._validation_max_retries}，"
                f"强制通过验证"
            )
            self._collected_stages.append({
                "stage": "validate", "elapsed_ms": 0, "forced_pass": True
            })
            return WriteEvent(task=ev.task, research_result=ev.research_result)

        # 使用 LLM 评估研究结果
        start_time = time.time()
        try:
            llm = get_llm_client()
            eval_prompt = (
                f"你是一个研究质量评估专家。请评估以下研究结果对于完成任务的充分程度。\n\n"
                f"任务：{ev.task}\n\n"
                f"研究结果：\n{ev.research_result}\n\n"
                f"请给出 0-10 的评分（仅输出数字，不要其他内容）：\n"
                f"0 = 完全不相关，10 = 非常充分且高质量"
            )
            score_text = await asyncio.to_thread(
                llm.chat_completion_simple, eval_prompt
            )

            # 解析评分（容错处理）
            score = 5.0
            match = re.search(r"\d+(?:\.\d+)?", score_text.strip())
            if match:
                score = float(match.group())
                score = max(0.0, min(10.0, score))

            normalized_score = score / 10.0
            passed = normalized_score >= self._validation_quality_threshold

            elapsed_ms = (time.time() - start_time) * 1000
            new_retries = retries + (0 if passed else 1)

            self._collected_stages.append({
                "stage": "validate", "elapsed_ms": elapsed_ms,
                "score": normalized_score, "passed": passed,
                "retries": new_retries,
            })

            print(
                f"[Validate] 评分={normalized_score:.2f}, "
                f"阈值={self._validation_quality_threshold}, "
                f"通过={passed}, 重试次数={new_retries}"
            )

            if passed:
                return WriteEvent(task=ev.task, research_result=ev.research_result)
            else:
                return ResearchEvent(task=ev.task, validation_retries=new_retries)

        except Exception as e:
            print(f"[Validate] 验证失败，降级为通过: {e}")
            self._collected_stages.append({
                "stage": "validate", "elapsed_ms": 0, "error": str(e)
            })
            return WriteEvent(task=ev.task, research_result=ev.research_result)

    @step
    async def write_step(self, ev: WriteEvent) -> WriteDoneEvent | EditDoneEvent:
        """写作步骤：基于研究结果生成草稿，支持跳过编辑"""
        start_time = time.time()

        context = {"research": ev.research_result}
        agent_result = await asyncio.to_thread(
            self._writer.execute, task=ev.task, context=context
        )
        elapsed_ms = (time.time() - start_time) * 1000

        self._collected_stages.append({
            "stage": "write", "elapsed_ms": elapsed_ms,
            "tokens": agent_result.total_tokens,
        })

        print(f"[AgentWorkflow] 写作完成，耗时 {elapsed_ms:.0f}ms")

        # 跳过编辑时直接返回 EditDoneEvent
        if self._skip_edit:
            print("[AgentWorkflow] 跳过编辑阶段")
            return EditDoneEvent(
                final_output=agent_result.output,
                elapsed_ms=elapsed_ms,
            )

        return WriteDoneEvent(
            task=ev.task,
            draft=agent_result.output,
            elapsed_ms=elapsed_ms,
        )

    @step
    async def edit_step(self, ev: WriteDoneEvent) -> EditDoneEvent:
        """编辑步骤：对草稿进行编辑优化"""
        start_time = time.time()

        context = {"draft": ev.draft}
        agent_result = await asyncio.to_thread(
            self._editor.execute, task=ev.task, context=context
        )
        elapsed_ms = (time.time() - start_time) * 1000

        self._collected_stages.append({
            "stage": "edit", "elapsed_ms": elapsed_ms,
            "tokens": agent_result.total_tokens,
        })

        print(f"[AgentWorkflow] 编辑完成，耗时 {elapsed_ms:.0f}ms")

        return EditDoneEvent(
            final_output=agent_result.output,
            elapsed_ms=elapsed_ms,
        )

    @step
    async def finish_step(self, ev: EditDoneEvent) -> StopEvent:
        """结束步骤：将最终输出包装为 StopEvent 终止工作流"""
        return StopEvent(result=ev.final_output)


# ------------------------------------------------------------------ #
#                    编排器适配器（同步包装器）                            #
# ------------------------------------------------------------------ #

class OrchestratorAdapter:
    """LlamaIndex Workflow 工作流编排器

    使用 LlamaIndex 事件驱动工作流协调「研究 -> 写作 -> 编辑」流水线。
    提供同步 run() 接口，内部通过 asyncio.run() 桥接异步工作流。

    示例:
        >>> orchestrator = OrchestratorAdapter()
        >>> result = orchestrator.run("撰写一篇关于 AI 趋势的文章")
        >>> print(result.final_output)
    """

    def __init__(self):
        """初始化编排器"""
        config = get_config()

        # 创建事件驱动工作流实例
        self._workflow = AgentWorkflow(
            validation_enabled=config.agents_validation_enabled,
            validation_max_retries=config.agents_validation_max_retries,
            validation_quality_threshold=config.agents_validation_quality_threshold,
        )

        if config.agents_validation_enabled:
            print(
                f"[OrchestratorAdapter] 已初始化 LlamaIndex 事件驱动工作流"
                f"（max_retries={config.agents_validation_max_retries}, "
                f"threshold={config.agents_validation_quality_threshold}）"
            )
        else:
            print("[OrchestratorAdapter] 已初始化 LlamaIndex 事件驱动工作流")

    def run(
        self,
        task: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        """运行完整的事件驱动工作流

        参数:
            task: 任务描述
            options: 可选配置（如跳过某些阶段）

        返回:
            OrchestratorResult，包含最终输出和各阶段详情
        """
        start_time = time.time()

        try:
            # 处理 options 参数
            skip_research = options.get("skip_research", False) if options else False
            skip_edit = options.get("skip_edit", False) if options else False

            # 运行工作流（支持跳过特定阶段）
            final_output = self._run_workflow_sync(
                task=task,
                skip_research=skip_research,
                skip_edit=skip_edit,
            )

            total_elapsed_ms = (time.time() - start_time) * 1000

            # 构建阶段结果列表（无论成功失败都收集）并汇总 tokens
            stages = []
            total_tokens = 0
            for stage_data in self._workflow._collected_stages:
                tokens = stage_data.get("tokens", 0)
                total_tokens += tokens
                stages.append(StageResult(
                    stage_name=stage_data["stage"],
                    agent_name=stage_data["stage"],
                    elapsed_ms=stage_data.get("elapsed_ms", 0),
                ))

            return OrchestratorResult(
                final_output=final_output,
                stages=stages,
                total_elapsed_ms=total_elapsed_ms,
                total_tokens=total_tokens,
                task=task,
                success=True,
            )

        except Exception as e:
            total_elapsed_ms = (time.time() - start_time) * 1000

            # 失败时也收集已完成的阶段并汇总 tokens
            stages = []
            total_tokens = 0
            for stage_data in self._workflow._collected_stages:
                tokens = stage_data.get("tokens", 0)
                total_tokens += tokens
                stages.append(StageResult(
                    stage_name=stage_data["stage"],
                    agent_name=stage_data["stage"],
                    elapsed_ms=stage_data.get("elapsed_ms", 0),
                ))

            return OrchestratorResult(
                final_output="",
                stages=stages,
                task=task,
                success=False,
                error=str(e),
                total_elapsed_ms=total_elapsed_ms,
                total_tokens=total_tokens,
            )

    def _run_workflow_sync(
        self,
        task: str,
        skip_research: bool = False,
        skip_edit: bool = False,
    ) -> str:
        """同步运行工作流（处理事件循环兼容性问题）

        FastAPI 已在运行事件循环中，asyncio.run() 无法嵌套调用。
        在独立线程中创建新的事件循环来运行 LlamaIndex 工作流。
        """
        # 设置跳过标志
        self._workflow._skip_research = skip_research
        self._workflow._skip_edit = skip_edit

        # 在独立线程中运行，避免与 FastAPI 事件循环冲突
        import concurrent.futures

        def _run_in_thread():
            async def _async_wrapper():
                handler = self._workflow.run(task=task)
                return await handler
            return asyncio.run(_async_wrapper())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_thread)
            return future.result()

    def run_stream(
        self,
        task: str,
        options: Optional[Dict[str, Any]] = None,
    ):
        """以流式输出运行工作流

        每个阶段完成时产出进度更新。

        参数:
            task: 任务描述
            options: 可选配置

        产出:
            包含阶段进度信息的字典
        """
        async def _run_and_stream():
            handler = self._workflow.run(task=task)
            async for event in handler.stream_events():
                yield {"event": type(event).__name__, "data": str(event)}
            result = await handler
            yield {"event": "done", "data": result}

        # 同步包装异步流式生成器
        loop = asyncio.new_event_loop()
        try:
            gen = _run_and_stream()
            agen = gen.__aiter__()
            while True:
                try:
                    chunk = loop.run_until_complete(agen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()


# ------------------------------------------------------------------ #
#                     全局单例 & 辅助函数                                #
# ------------------------------------------------------------------ #

_orchestrator_instance: Optional[OrchestratorAdapter] = None


def get_orchestrator() -> OrchestratorAdapter:
    """获取全局编排器适配器单例（懒加载）

    返回:
        OrchestratorAdapter 单例
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestratorAdapter()
    return _orchestrator_instance


def reload_orchestrator() -> OrchestratorAdapter:
    """强制重新创建编排器适配器

    返回:
        新的 OrchestratorAdapter 实例
    """
    global _orchestrator_instance
    _orchestrator_instance = OrchestratorAdapter()
    return _orchestrator_instance
