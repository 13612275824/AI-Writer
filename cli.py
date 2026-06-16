# ============================================================================
# 命令行交互入口  |  cli.py
# ============================================================================
#
# 功能：提供命令行交互界面，整合已有模块功能
#
# 子命令：
#   chat     - 与大模型进行交互式对话（流式输出）
#   ask      - 基于 RAG 的知识问答（检索 + 生成）
#   report   - 生成每日工作日志报告
#   optimize - 内容优化（润色/简化/扩写/缩写/语法校对）
#   agent    - 多 Agent 协作写作（Research → Write → Edit 流水线）
#   import  - 导入文档到向量数据库
#   docs    - 查看/删除已导入的文档
#   info    - 查看系统配置信息
#
# 使用方式：
#   python cli.py chat                              交互对话
#   python cli.py ask "什么是深度学习"                RAG 知识问答
#   python cli.py report                             生成日志报告
#   python cli.py optimize                            交互式内容优化
#   python cli.py optimize -t polish "待优化文本"       快捷润色
#   python cli.py agent "写一篇关于 AI 的文章"          Agent 流水线
#   python cli.py agent --skip-research "润色内容"     跳过研究阶段
#   python cli.py agent --stream "写深度学习的报告"    流式显示进度
#   python cli.py import data/documents/              导入文档
#   python cli.py info                                查看系统信息
# ============================================================================

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ------------------------------------------------------------------ #
#                    帮助信息                                         #
# ------------------------------------------------------------------ #

_HELP_TEXT = """
  === 使用流程 ===

  第一步：导入文档（构建知识库）
    import data/documents/              导入目录下所有文档（.pdf/.docx/.txt/.md）
    import data/documents/孔令辉.pdf    导入单个文件
    import /data/documents/             支持以 / 开头的相对路径
    import data/docs -r                 -r 递归扫描子目录
    import data/docs -c my_collection   -c 指定集合名称（默认 documents）

  第二步：知识问答（基于已导入的文档）
    ask 孔令辉的职业是什么？             RAG 检索 + 大模型生成
    ask -k 3 什么是深度学习？            -k 指定检索结果数量（默认从配置读取）
    ask -s 孔令辉的职业是什么？            -s 流式输出（实时逐字显示）
    ask -d 写一首关于春天的诗            -d 跳过检索，直接调用大模型生成
    ask -s -d 讲个笑话                   -s -d 可组合使用

  第三步：多 Agent 协作写作
    agent "写一篇关于 AI 的文章"          完整流水线（Research → Write → Edit）
    agent --skip-research "润色以下内容"   跳过研究阶段
    agent --skip-edit "写一篇初稿"        跳过编辑阶段
    agent --stream "写深度学习的报告"     流式显示各阶段进度
    agent --style 学术 "写论文摘要"       指定写作风格
    agent --edit-mode grammar "校对内容"  指定编辑模式（full/grammar/expression/structure）
    agent -o output/article.md           输出保存到文件

  第四步：对话 & 写作
    chat                                与大模型自由对话（流式输出）
    chat -r writing                     使用指定角色对话（可选: writing/editing/research/daily_report）
    report                              交互式输入工作内容，生成每日工作日志
    report -w "完成模块A" "修复Bug" -s   快捷生成（-w 指定工作项，-s 跳过问题和计划输入）
    report -o output/report.md          生成并保存到文件
    optimize                             交互式内容优化（输入文本，空行结束）
    optimize -t polish "待优化文本"        快捷润色优化
    optimize -t grammar -s formal "..." 语法校对 + 正式风格
    optimize -o output/optimized.md     优化并保存到文件
    transfer                             交互式风格转换（输入文本，空行结束）
    transfer --target formal "待转换文本"   快捷转换为正式风格
    transfer --target academic --source casual "..."  口语转学术
    transfer -o output/transferred.md    转换并保存到文件
    report-write "报告标题"                交互式报告写作（工作总结）
    report-write -t project "项目报告"     快捷生成项目报告
    report-write -t analysis -w 5000 "分析报告"  分析报告，目标 5000 字
    report-write -o output/report.md       报告并保存到文件

  文档管理:
    docs                                列出已导入的文档（文件名、文本块数、字符数、类型）
    docs view 1                         查看第 1 个文档的文本块内容
    docs view 1 -n 3                    仅显示前 3 个文本块
    docs delete 1                       删除第 1 个文档的所有文本块
    docs -c my_collection               操作指定集合

  系统信息:
    info                                查看模型、RAG 配置、向量数据库状态

  通用操作:
    help / h / ?                        显示此帮助信息
    quit / exit / q                     退出程序
"""


# ------------------------------------------------------------------ #
#                    子命令实现                                       #
# ------------------------------------------------------------------ #

def cmd_chat(args):
    """交互对话模式（流式输出）"""
    from src.core.llm_adapter import get_llm_client
    from src.core.config import get_config

    client = get_llm_client()
    print(f"\n  模型: {client.model}")
    print("  输入消息与大模型对话，输入 quit/exit 退出，输入 clear 清空历史\n")

    # 加载系统提示词
    system_prompt = None
    if args.role:
        try:
            config = get_config()
            system_prompt = config.get_system_prompt(args.role)
            print(f"  已加载角色: {args.role}\n")
        except Exception:
            print(f"  警告: 角色 '{args.role}' 不存在，使用默认模式\n")

    history = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n再见！")
            break

        if user_input.lower() == "clear":
            history = [h for h in history if h["role"] == "system"]
            print("  [历史已清空]\n")
            continue

        history.append({"role": "user", "content": user_input})

        print("AI: ", end="", flush=True)
        full_response = ""
        try:
            for chunk in client.chat_completion_stream(history):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()  # 换行
        except Exception as e:
            print(f"\n  [错误] {e}\n")
            history.pop()  # 移除失败的用户消息
            continue

        history.append({"role": "assistant", "content": full_response})
        print()


def cmd_ask(args):
    """RAG 知识问答"""
    from src.rag.query_engine import get_generator

    gen = get_generator()
    query = " ".join(args.query)

    print(f"\n  问题: {query}")
    print(f"  模型: {gen.llm_client.model}")
    print("  " + "-" * 50)

    try:
        if args.stream:
            # 流式输出：实时逐字显示
            print()
            full_answer = ""
            for chunk in gen.generate_stream(
                query=query,
                use_retrieval=not args.direct,
                top_k=args.top_k,
            ):
                print(chunk, end="", flush=True)
                full_answer += chunk
            print()

            # 流式完成后读取检索来源（generate_stream 内部已保存）
            if not args.direct and hasattr(gen, '_last_sources') and gen._last_sources:
                print("\n  引用来源:")
                for src in gen._last_sources:
                    print(f"    - {src}")
        else:
            # 非流式输出（默认）
            result = gen.generate(
                query=query,
                use_retrieval=not args.direct,
                top_k=args.top_k,
            )

            print(f"\n{result.answer}\n")

            if result.has_sources:
                print("  引用来源:")
                for src in result.sources:
                    print(f"    - {src}")

            print(f"\n  耗时: {result.elapsed_ms:.1f}ms | "
                  f"Token: {result.total_tokens}")

    except Exception as e:
        print(f"\n  [错误] {e}")


def cmd_report(args):
    """生成每日工作日志报告"""
    from src.writing.daily_report import get_daily_report_writer

    writer = get_daily_report_writer()

    print("\n  === 每日工作日志报告生成 ===\n")

    # 收集工作事项
    work_items = []
    if args.work:
        work_items = args.work
    else:
        print("  请输入今日完成的工作事项（每行一条，空行结束）:")
        while True:
            try:
                line = input("    > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            work_items.append(line)

    # 收集问题
    issues = []
    if args.issues:
        issues = args.issues
    elif not args.skip_optional:
        print("\n  请输入遇到的问题（每行一条，空行跳过）:")
        while True:
            try:
                line = input("    > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            issues.append(line)

    # 收集计划
    plans = []
    if args.plans:
        plans = args.plans
    elif not args.skip_optional:
        print("\n  请输入明日计划（每行一条，空行跳过）:")
        while True:
            try:
                line = input("    > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            plans.append(line)

    # 校验
    if not any([work_items, issues, plans]):
        print("\n  [错误] 至少需要提供一项工作内容")
        return

    print("\n  正在生成报告...")

    try:
        result = writer.write(
            work_items=work_items,
            issues=issues,
            plans=plans,
            project_name=args.project or "",
            author=args.author or "",
            report_date=args.date or None,
        )

        print(f"\n{'=' * 60}")
        print(result.report)
        print(f"{'=' * 60}")

        print(f"\n  日期: {result.report_date} | "
              f"耗时: {result.elapsed_ms:.1f}ms | "
              f"Token: {result.total_tokens}")

        # 保存到文件
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.report, encoding="utf-8")
            print(f"  已保存到: {output_path}")

    except Exception as e:
        print(f"\n  [错误] {e}")


def cmd_import(args):
    """导入文档到向量数据库"""
    from src.rag.document_ingestion import get_document_loader
    from src.rag.vector_store import VectorStoreAdapter, get_vector_store

    source_path = Path(args.path)

    # 路径不存在时，尝试相对于项目根目录解析
    # 兼容用户输入 /data/... 或 data/... 两种形式
    if not source_path.exists():
        relative_path = args.path.lstrip("/\\")
        candidate = _project_root / relative_path
        if candidate.exists():
            source_path = candidate
        else:
            print(f"  [错误] 路径不存在: {args.path}")
            return

    print(f"\n  路径: {source_path}")
    print("  正在加载文档...")

    loader = get_document_loader()

    if source_path.is_file():
        try:
            all_chunks = loader.load_and_split(str(source_path))
            print(f"  加载 1 个文档: {source_path.name}")
            print(f"  共 {len(all_chunks)} 个文本块")
        except Exception as e:
            print(f"  [错误] 加载失败: {e}")
            return
    else:
        try:
            all_chunks = []
            supported_exts = {".pdf", ".docx", ".txt", ".md", ".markdown"}
            files_to_process = []
            if args.recursive:
                for ext in supported_exts:
                    files_to_process.extend(source_path.rglob(f"*{ext}"))
            else:
                files_to_process = [
                    f for f in source_path.iterdir()
                    if f.is_file() and f.suffix.lower() in supported_exts
                ]
            for file_path in files_to_process:
                try:
                    chunks = loader.load_and_split(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"  [警告] 跳过 {file_path.name}: {e}")
            print(f"  加载 {len(files_to_process)} 个文档")
        except Exception as e:
            print(f"  [错误] 加载失败: {e}")
            return

    if not all_chunks:
        print("  未找到可加载的文档或无有效文本块")
        return

    # 向量化存储
    print("  正在存入向量数据库...")
    store = get_vector_store()

    # collection 为 None 时由 VectorStore 使用配置文件中的默认集合名
    collection = args.collection  # None 表示使用配置默认值
    count = store.add_chunks(all_chunks, collection_name=collection)

    # 获取实际使用的集合名（用于打印）
    actual_collection = collection or store.collection_name

    print(f"\n  导入完成!")
    print(f"  集合: {actual_collection}")
    print(f"  新增: {count} 个文本块")
    print(f"  文档: {len(files_to_process) if not source_path.is_file() else 1} 个")


def cmd_docs(args):
    """查看已导入的文档"""
    from src.rag.vector_store import get_vector_store

    store = get_vector_store()
    action = args.action
    collection = args.collection

    try:
        if action == "list":
            # 列出所有来源文档
            sources = store.get_sources(collection_name=collection)

            if not sources:
                coll_name = collection or store.collection_name
                print(f"\n  集合 '{coll_name}' 中暂无已导入的文档")
                return

            print(f"\n  已导入的文档列表 ({len(sources)} 个来源):")
            print(f"  {'-' * 56}")
            print(f"  {'序号':<4} {'文件名':<20} {'文本块':<6} {'字符数':<8} {'类型'}")
            print(f"  {'-' * 56}")

            for i, src_info in enumerate(sources, 1):
                source_path = src_info["source"]
                # 提取文件名显示
                file_name = source_path.rsplit(
                    "/", 1)[-1] if "/" in source_path else source_path.rsplit("\\", 1)[-1]
                if len(file_name) > 18:
                    file_name = file_name[:15] + "..."
                file_type = src_info.get("file_type", "")
                print(
                    f"  {i:<4} {file_name:<20} "
                    f"{src_info['chunk_count']:<6} "
                    f"{src_info['total_chars']:<8} "
                    f"{file_type}"
                )

            print(f"  {'-' * 56}")
            print(f"\n  提示: docs view <序号> 查看文档内容")
            print(f"       docs delete <序号> 删除文档")

        elif action == "view":
            # 查看指定文档内容
            target = args.target
            if not target:
                print("  [错误] 请指定文档序号或来源路径")
                return

            # 尝试解析为序号
            sources = store.get_sources(collection_name=collection)
            if not sources:
                print("  集合中暂无已导入的文档")
                return

            source_path = None
            if target.isdigit():
                idx = int(target) - 1
                if 0 <= idx < len(sources):
                    source_path = sources[idx]["source"]
                else:
                    print(f"  [错误] 序号超出范围 (1-{len(sources)})")
                    return
            else:
                source_path = target

            chunks = store.get_by_source(
                source=source_path,
                limit=args.limit,
                collection_name=collection,
            )

            if not chunks:
                print(f"  未找到来源为 '{source_path}' 的文档")
                return

            print(f"\n  文档来源: {source_path}")
            print(f"  文本块数: {len(chunks)}")
            print(f"  {'=' * 56}")

            for i, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                idx = meta.get("chunk_index", i)
                text = chunk["text"]
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"\n  [块 #{idx}] ({len(text)} 字符)")
                print(f"  {preview}")

        elif action == "delete":
            # 删除指定文档
            target = args.target
            if not target:
                print("  [错误] 请指定文档序号或来源路径")
                return

            sources = store.get_sources(collection_name=collection)
            if not sources:
                print("  集合中暂无已导入的文档")
                return

            source_path = None
            if target.isdigit():
                idx = int(target) - 1
                if 0 <= idx < len(sources):
                    source_path = sources[idx]["source"]
                else:
                    print(f"  [错误] 序号超出范围 (1-{len(sources)})")
                    return
            else:
                source_path = target

            file_name = source_path.rsplit(
                "/", 1)[-1] if "/" in source_path else source_path.rsplit("\\", 1)[-1]
            print(f"\n  即将删除文档: {file_name}")
            print(f"  来源路径: {source_path}")

            store.delete_by_source(
                source=source_path,
                collection_name=collection,
            )
            print(f"  删除成功!")

    except Exception as e:
        print(f"\n  [错误] {e}")


# 优化类型中文映射
_OPTIMIZE_TYPES = {
    "polish": "润色优化",
    "simplify": "简化改写",
    "expand": "扩写丰富",
    "shorten": "精简缩写",
    "grammar": "语法校对",
}

# 目标风格中文映射
_TARGET_STYLES = {
    "formal": "正式书面",
    "casual": "轻松口语",
    "academic": "学术论文",
    "professional": "专业商务",
    "creative": "创意文艺",
}


def cmd_optimize(args):
    """内容优化（润色/简化/扩写/缩写/语法校对）"""
    from src.writing.content_optimizer import get_content_optimizer

    # 获取待优化内容
    if args.content:
        content = " ".join(args.content)
    else:
        print("\n  === 内容优化 ===\n")
        print("  请输入待优化的内容（空行结束）:")
        lines = []
        while True:
            try:
                line = input("    > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            lines.append(line)
        content = "\n".join(lines)

    if not content.strip():
        print("\n  [错误] 待优化内容不能为空")
        return

    # 优化类型
    opt_type = args.type or "polish"
    type_name = _OPTIMIZE_TYPES.get(opt_type, opt_type)

    # 目标风格
    style = args.style or ""
    style_name = _TARGET_STYLES.get(style, style) if style else ""

    print(f"\n  优化类型: {type_name}")
    if style_name:
        print(f"  目标风格: {style_name}")
    print(f"  内容长度: {len(content)} 字符")
    print("  " + "-" * 50)
    print("  正在优化...")

    optimizer = get_content_optimizer()

    try:
        result = optimizer.optimize(
            content=content,
            optimize_type=opt_type,
            target_style=style,
            focus_areas=args.focus or "",
            requirements=args.requirements or "",
        )

        print(f"\n{'=' * 60}")
        print(result.optimized_content)
        print(f"{'=' * 60}")

        print(f"\n  {result.summary}")
        print(f"  耗时: {result.elapsed_ms:.1f}ms | Token: {result.total_tokens}")

        # 保存到文件
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.optimized_content, encoding="utf-8")
            print(f"  已保存到: {output_path}")

    except Exception as e:
        print(f"\n  [错误] {e}")


# 转换风格中文映射
_TRANSFER_STYLES = {
    "formal": "正式书面",
    "casual": "轻松口语",
    "academic": "学术论文",
    "professional": "专业商务",
    "creative": "创意文艺",
    "news": "新闻报道",
    "storytelling": "叙事故事",
}


def cmd_transfer(args):
    """风格转换（将文本从一种风格完整转换为另一种）"""
    from src.writing.style_transfer import get_style_transfer

    # 获取待转换内容
    if args.content:
        content = " ".join(args.content)
    else:
        print("\n  === 风格转换 ===\n")
        print("  请输入待转换的内容（空行结束）:")
        lines = []
        while True:
            try:
                line = input("    > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            lines.append(line)
        content = "\n".join(lines)

    if not content.strip():
        print("\n  [错误] 待转换内容不能为空")
        return

    # 目标风格（必填）
    target = args.target or "formal"
    target_name = _TRANSFER_STYLES.get(target, target)

    # 源风格（可选）
    source = args.source or ""
    source_name = _TRANSFER_STYLES.get(source, source) if source else "自动识别"

    print(f"\n  目标风格: {target_name}")
    print(f"  源风格: {source_name}")
    print(f"  内容长度: {len(content)} 字符")
    print("  " + "-" * 50)
    print("  正在转换...")

    transfer = get_style_transfer()

    try:
        result = transfer.transfer(
            content=content,
            target_style=target,
            source_style=source,
            requirements=args.requirements or "",
        )

        print(f"\n{'=' * 60}")
        print(result.transferred_content)
        print(f"{'=' * 60}")
        print(
            f"\n  耗时: {result.elapsed_ms:.1f}ms | Token: {result.total_tokens}")

        # 保存到文件
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                result.transferred_content, encoding="utf-8")
            print(f"  已保存到: {output_path}")

    except Exception as e:
        print(f"\n  [错误] {e}")


# 报告类型中文映射
_REPORT_TYPES = {
    "work_summary": "工作总结报告",
    "project": "项目报告",
    "analysis": "分析报告",
    "research": "调研报告",
}


def cmd_report_write(args):
    """报告写作（工作总结/项目/分析/调研报告）"""
    from src.writing.report_writer import get_report_writer

    # 获取标题
    if args.title:
        title = " ".join(args.title)
    else:
        print("\n  === 报告写作 ===\n")
        title = input("  请输入报告标题: ").strip()

    if not title.strip():
        print("\n  [错误] 报告标题不能为空")
        return

    # 报告类型
    rtype = args.type or "work_summary"
    type_name = _REPORT_TYPES.get(rtype, rtype)

    # 素材内容
    content = ""
    if args.content:
        content = " ".join(args.content)
    elif not args.title:
        # 交互模式下才询问素材
        print("  请输入素材/要点（空行结束）:")
        lines = []
        while True:
            try:
                line = input("    > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            lines.append(line)
        content = "\n".join(lines)

    print(f"\n  报告标题: {title}")
    print(f"  报告类型: {type_name}")
    if content:
        print(f"  素材长度: {len(content)} 字符")
    print("  " + "-" * 50)
    print("  正在生成报告...")

    writer = get_report_writer()

    try:
        result = writer.write(
            title=title,
            report_type=rtype,
            content=content,
            sections=args.sections or "",
            requirements=args.requirements or "",
            word_count=args.word_count,
        )

        print(f"\n{'=' * 60}")
        print(result.report)
        print(f"{'=' * 60}")

        if result.summary:
            print(f"\n  📋 摘要: {result.summary}")
        print(f"  耗时: {result.elapsed_ms:.1f}ms | Token: {result.total_tokens}")

        # 保存到文件
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.report, encoding="utf-8")
            print(f"  已保存到: {output_path}")

    except Exception as e:
        print(f"\n  [错误] {e}")


# 阶段名称中文映射
_STAGE_NAMES = {
    "research": "研究",
    "write": "写作",
    "edit": "编辑",
}


def cmd_agent(args):
    """多 Agent 协作写作（Research → Write → Edit 流水线）"""
    from src.agents.orchestrator import get_orchestrator

    task = " ".join(args.task)

    # 构建执行选项
    options = {}
    if args.skip_research:
        options["skip_research"] = True
    if args.skip_edit:
        options["skip_edit"] = True
    if args.style:
        options.setdefault("write_context", {})
        options["write_context"]["style"] = args.style
    if args.edit_mode:
        options.setdefault("edit_context", {})
        options["edit_context"]["edit_mode"] = args.edit_mode

    try:
        orch = get_orchestrator()

        print(f"\n  任务: {task}")
        print(f"  流水线: Research → Write → Edit")
        print("  " + "-" * 50)

        if args.stream:
            # 流式输出：逐阶段显示进度
            print()
            final_output = ""
            for event in orch.run_stream(task, options=options):
                event_type = event["type"]

                if event_type == "pipeline_start":
                    total = event["data"]["total_stages"]
                    pipeline = event["data"]["pipeline"]
                    stages_display = " → ".join(
                        _STAGE_NAMES.get(s, s) for s in pipeline
                    )
                    print(f"  [启动] 共 {total} 个阶段: {stages_display}")

                elif event_type == "stage_start":
                    stage = event["stage"]
                    idx = event["data"]["stage_index"] + 1
                    desc = event["data"].get("description", "")
                    name_cn = _STAGE_NAMES.get(stage, stage)
                    print(f"\n  [{idx}] {name_cn}: {desc}")
                    print(f"      执行中...", end="", flush=True)

                elif event_type == "stage_end":
                    stage = event["stage"]
                    data = event["data"]
                    if data["success"]:
                        ms = data["elapsed_ms"]
                        out_len = data["output_length"]
                        print(f" ✓ ({ms:.0f}ms, {out_len} 字符)")
                    else:
                        err = data.get("error", "未知错误")
                        print(f" ✗ 失败: {err}")

                elif event_type == "complete":
                    final_output = event["data"]["final_output"]
                    total_ms = event["data"]["total_elapsed_ms"]
                    print(f"\n  {'=' * 50}")
                    print(f"  完成！共 {event['data']['total_stages']} 个阶段, "
                          f"耗时 {total_ms:.0f}ms")
                    print(f"  {'=' * 50}")

                elif event_type == "error":
                    print(f"\n  [错误] {event['data']}")
                    return

            # 输出最终内容
            if final_output:
                print(f"\n{final_output}\n")

        else:
            # 非流式输出（默认）：执行完整流水线
            print("  正在执行流水线...")

            # 使用回调显示进度
            def on_stage_start(stage_name):
                name_cn = _STAGE_NAMES.get(stage_name, stage_name)
                print(f"  → {name_cn}阶段开始...", flush=True)

            def on_stage_end(stage_name, result):
                name_cn = _STAGE_NAMES.get(stage_name, stage_name)
                if result.success:
                    print(f"  ✓ {name_cn}阶段完成 "
                          f"({result.elapsed_ms:.0f}ms)")
                else:
                    print(f"  ✗ {name_cn}阶段失败: {result.error}")

            options["on_stage_start"] = on_stage_start
            options["on_stage_end"] = on_stage_end

            result = orch.run(task, options=options)

            if result.success:
                print(f"\n{'=' * 60}")
                print(result.final_output)
                print(f"{'=' * 60}")

                print(f"\n  阶段数: {result.stage_count} | "
                      f"耗时: {result.total_elapsed_ms:.0f}ms | "
                      f"Token: {result.total_tokens}")
            else:
                print(f"\n  [失败] {result.error}")
                return

            # 输出变量（供 -o 使用）
            final_output = result.final_output

        # 保存到文件
        if args.output and final_output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(final_output, encoding="utf-8")
            print(f"  已保存到: {output_path}")

    except Exception as e:
        print(f"\n  [错误] {e}")


def cmd_info(args):
    """查看系统配置信息"""
    from src.core.config import get_config

    cfg = get_config()

    print(f"\n  {'=' * 50}")
    print(f"  {cfg.app_name}  v{cfg.app_version}")
    print(f"  {'=' * 50}")

    print(f"\n  [应用配置]")
    print(f"    名称:     {cfg.app_name}")
    print(f"    版本:     {cfg.app_version}")
    print(f"    调试模式: {cfg.debug}")

    print(f"\n  [模型配置]")
    print(f"    提供商:   openai (阿里云百炼)")
    print(f"    模型:     {cfg.default_model}")
    print(f"    温度:     {cfg.temperature}")
    print(f"    API端点:  {cfg.base_url}")

    print(f"\n  [RAG 配置]")
    print(f"    检索 Top-K:       {cfg.retriever_top_k}")
    print(f"    文本块大小:       {cfg.chunk_size}")
    print(f"    文本块重叠:       {cfg.chunk_overlap}")
    print(f"    生成上下文上限:   {cfg.generator_max_context_chars} 字符")

    # 检查向量数据库
    vectors_dir = Path("data/vectors")
    db_file = vectors_dir / "chroma.sqlite3"
    print(f"\n  [数据状态]")
    print(f"    向量数据库: {'存在' if db_file.exists() else '未创建'}")

    docs_dir = Path("data/documents")
    if docs_dir.exists():
        doc_files = list(docs_dir.iterdir())
        doc_count = len([f for f in doc_files if f.is_file()
                        and f.name != ".gitkeep"])
        print(f"    文档目录:   {doc_count} 个文件")
    else:
        print(f"    文档目录:   不存在")

    print()


# ------------------------------------------------------------------ #
#                    参数解析                                         #
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""

    parser = argparse.ArgumentParser(
        prog="ai-writer",
        description="AI写作助手 - 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cli.py chat                              交互对话（流式输出）
  python cli.py chat --role writing               使用写作角色对话
  python cli.py ask "什么是深度学习"                RAG 知识问答
  python cli.py ask -s "孔令辉的职业是什么"          流式输出（实时逐字显示）
  python cli.py ask --direct "写一首诗"             跳过检索，直接生成
  python cli.py agent "写一篇关于 AI 的文章"          Agent 流水线写作
  python cli.py agent --stream --skip-research "润色内容"  流式+跳过研究
  python cli.py agent -o output/article.md         保存到文件
  python cli.py report                            交互式生成日志报告
  python cli.py report -w "完成模块A" "修复Bug" -s   快捷生成（-s 跳过可选项输入）
  python cli.py report -o output/report.md        生成并保存到文件
  python cli.py optimize                            交互式内容优化
  python cli.py optimize -t simplify "待优化文本"     快捷简化改写
  python cli.py optimize -t grammar -s formal "..."  语法校对+正式风格
  python cli.py transfer --target formal "待转换文本"  风格转换（正式风格）
  python cli.py transfer --target academic --source casual "..."  口语转学术
  python cli.py report-write "年度工作总结"             报告写作
  python cli.py report-write -t project "项目报告"       生成项目报告
  python cli.py import data/documents/             导入目录下所有文档
  python cli.py import report.pdf                  导入单个文档
  python cli.py docs                               列出已导入的文档
  python cli.py docs view 1                        查看第 1 个文档内容
  python cli.py docs delete 1                      删除第 1 个文档
  python cli.py info                               查看系统配置和数据状态
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ── chat 子命令 ──
    chat_parser = subparsers.add_parser(
        "chat", help="与大模型进行交互式对话"
    )
    chat_parser.add_argument(
        "--role", "-r", type=str, default=None,
        help="Prompt 角色名称（writing/editing/research/daily_report）"
    )

    # ── ask 子命令 ──
    ask_parser = subparsers.add_parser(
        "ask", help="基于 RAG 的知识问答"
    )
    ask_parser.add_argument(
        "query", nargs="+", type=str,
        help="查询问题"
    )
    ask_parser.add_argument(
        "--direct", "-d", action="store_true",
        help="跳过检索，直接调用 LLM 生成"
    )
    ask_parser.add_argument(
        "--top-k", "-k", type=int, default=None,
        help="检索结果数量（默认从配置读取）"
    )
    ask_parser.add_argument(
        "--stream", "-s", action="store_true",
        help="启用流式输出（实时显示生成过程）"
    )

    # ── report 子命令 ──
    report_parser = subparsers.add_parser(
        "report", help="生成每日工作日志报告"
    )
    report_parser.add_argument(
        "--work", "-w", nargs="+", type=str, default=None,
        help="今日完成的工作事项"
    )
    report_parser.add_argument(
        "--issues", "-i", nargs="+", type=str, default=None,
        help="遇到的问题"
    )
    report_parser.add_argument(
        "--plans", "-p", nargs="+", type=str, default=None,
        help="明日工作计划"
    )
    report_parser.add_argument(
        "--project", type=str, default=None,
        help="项目名称"
    )
    report_parser.add_argument(
        "--author", "-a", type=str, default=None,
        help="作者姓名"
    )
    report_parser.add_argument(
        "--date", type=str, default=None,
        help="报告日期（YYYY-MM-DD，默认当天）"
    )
    report_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出文件路径（可选）"
    )
    report_parser.add_argument(
        "--skip-optional", "-s", action="store_true",
        help="跳过问题和计划的交互输入"
    )

    # ── import 子命令 ──
    import_parser = subparsers.add_parser(
        "import", help="导入文档到向量数据库"
    )
    import_parser.add_argument(
        "path", type=str,
        help="文件或目录路径"
    )
    import_parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="递归扫描子目录"
    )
    import_parser.add_argument(
        "--collection", "-c", type=str, default=None,
        help="向量集合名称（默认使用配置文件中的 documents）"
    )

    # ── optimize 子命令 ──
    optimize_parser = subparsers.add_parser(
        "optimize", help="内容优化（润色/简化/扩写/缩写/语法校对）"
    )
    optimize_parser.add_argument(
        "content", nargs="*", type=str, default=None,
        help="待优化的文本内容（可选，不提供时交互式输入）"
    )
    optimize_parser.add_argument(
        "--type", "-t", type=str, default=None,
        choices=["polish", "simplify", "expand", "shorten", "grammar"],
        help="优化类型（默认 polish）"
    )
    optimize_parser.add_argument(
        "--style", "-s", type=str, default=None,
        choices=["formal", "casual", "academic", "professional", "creative"],
        help="目标风格"
    )
    optimize_parser.add_argument(
        "--focus", "-f", type=str, default=None,
        help="重点关注方向（如：语法,表达,结构）"
    )
    optimize_parser.add_argument(
        "--requirements", "-r", type=str, default=None,
        help="额外优化要求"
    )
    optimize_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出文件路径（可选）"
    )

    # ── transfer 子命令 ──
    transfer_parser = subparsers.add_parser(
        "transfer", help="风格转换（将文本从一种风格完整转换为另一种风格）"
    )
    transfer_parser.add_argument(
        "content", nargs="*", type=str, default=None,
        help="待转换的文本内容（可选，不提供时交互式输入）"
    )
    transfer_parser.add_argument(
        "--target", "-t", type=str, default="formal",
        choices=["formal", "casual", "academic",
                 "professional", "creative", "news", "storytelling"],
        help="目标风格（默认 formal）"
    )
    transfer_parser.add_argument(
        "--source", "-s", type=str, default="",
        choices=["formal", "casual", "academic",
                 "professional", "creative", "news", "storytelling"],
        help="源风格（留空自动识别）"
    )
    transfer_parser.add_argument(
        "--requirements", "-r", type=str, default=None,
        help="额外转换要求"
    )
    transfer_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出文件路径（可选）"
    )

    # ── report-write 子命令 ──
    report_write_parser = subparsers.add_parser(
        "report-write", help="报告写作（工作总结/项目/分析/调研报告）"
    )
    report_write_parser.add_argument(
        "title", nargs="+", type=str,
        help="报告标题"
    )
    report_write_parser.add_argument(
        "--type", "-t", type=str, default="work_summary",
        choices=["work_summary", "project", "analysis", "research"],
        help="报告类型（默认 work_summary）"
    )
    report_write_parser.add_argument(
        "--content", "-c", nargs="*", type=str, default=None,
        help="素材内容（可选，不提供时交互式输入）"
    )
    report_write_parser.add_argument(
        "--sections", type=str, default=None,
        help="自定义章节（逗号分隔，如：引言,主体,结论）"
    )
    report_write_parser.add_argument(
        "--requirements", "-r", type=str, default=None,
        help="额外要求"
    )
    report_write_parser.add_argument(
        "--word-count", "-w", type=int, default=None,
        help="目标字数"
    )
    report_write_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出文件路径（可选）"
    )

    # ── agent 子命令 ──
    agent_parser = subparsers.add_parser(
        "agent", help="多 Agent 协作写作（Research → Write → Edit）"
    )
    agent_parser.add_argument(
        "task", nargs="+", type=str,
        help="写作任务描述"
    )
    agent_parser.add_argument(
        "--skip-research", action="store_true",
        help="跳过研究阶段"
    )
    agent_parser.add_argument(
        "--skip-edit", action="store_true",
        help="跳过编辑阶段"
    )
    agent_parser.add_argument(
        "--stream", "-s", action="store_true",
        help="流式显示各阶段进度"
    )
    agent_parser.add_argument(
        "--style", type=str, default=None,
        help="写作风格（如: 学术/通俗/正式/幽默）"
    )
    agent_parser.add_argument(
        "--edit-mode", type=str, default=None,
        choices=["full", "grammar", "expression", "structure"],
        help="编辑模式（full/grammar/expression/structure）"
    )
    agent_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出文件路径（可选）"
    )

    # ── info 子命令 ──
    subparsers.add_parser(
        "info", help="查看系统配置信息"
    )

    # ── docs 子命令 ──
    docs_parser = subparsers.add_parser(
        "docs", help="查看已导入的文档"
    )
    docs_parser.add_argument(
        "action", nargs="?", type=str, default="list",
        choices=["list", "view", "delete"],
        help="操作类型: list(列表) / view(查看) / delete(删除)"
    )
    docs_parser.add_argument(
        "target", nargs="?", type=str, default=None,
        help="文档序号或来源路径（用于 view/delete）"
    )
    docs_parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="限制显示的文本块数量（仅 view 生效）"
    )
    docs_parser.add_argument(
        "--collection", "-c", type=str, default=None,
        help="向量集合名称（默认使用配置文件中的 documents）"
    )

    return parser


def _interactive_mode():
    """交互模式：等待用户输入命令并执行"""
    try:
        from src.core.config import get_config
        cfg = get_config()
        app_name = cfg.app_name
        app_version = cfg.app_version
    except Exception:
        app_name = "AI写作助手"
        app_version = "0.1.0"

    line = f"  {app_name}  v{app_version}"
    padding = 52 - len(line)
    left = padding // 2
    right = padding - left
    print(f"""
  +{'=' * 54}+
  |{' ' * left}{line}{' ' * right}|
  +{'=' * 54}+""")
    print(_HELP_TEXT)

    parser = build_parser()

    while True:
        try:
            user_input = input("ai-writer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n再见！")
            break

        if user_input.lower() in ("help", "h", "?"):
            print(_HELP_TEXT)
            continue

        # 解析用户输入的命令参数
        import shlex
        try:
            argv = shlex.split(user_input)
        except ValueError as e:
            print(f"  [错误] 参数解析失败: {e}")
            continue

        try:
            args = parser.parse_args(argv)
        except SystemExit:
            # argparse 在参数错误时会调用 sys.exit()，这里拦截
            continue

        if not args.command:
            print("  请输入命令，如: chat, ask, report, import, docs, info")
            continue

        commands = {
            "chat": cmd_chat,
            "ask": cmd_ask,
            "report": cmd_report,
            "optimize": cmd_optimize,
            "transfer": cmd_transfer,
            "report-write": cmd_report_write,
            "agent": cmd_agent,
            "import": cmd_import,
            "info": cmd_info,
            "docs": cmd_docs,
        }

        handler = commands.get(args.command)
        if handler:
            try:
                handler(args)
            except KeyboardInterrupt:
                print("\n\n  操作已取消")
            except Exception as e:
                print(f"\n  [错误] {e}")
        else:
            print(f"  [错误] 未知命令: {args.command}")

        print()  # 空行分隔


def main():
    """CLI 主入口"""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        # 无命令时进入交互模式
        _interactive_mode()
        return

    # 命令路由
    commands = {
        "chat": cmd_chat,
        "ask": cmd_ask,
        "report": cmd_report,
        "optimize": cmd_optimize,
        "transfer": cmd_transfer,
        "report-write": cmd_report_write,
        "agent": cmd_agent,
        "import": cmd_import,
        "info": cmd_info,
        "docs": cmd_docs,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\n\n  操作已取消")
        except Exception as e:
            print(f"\n  [错误] {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
