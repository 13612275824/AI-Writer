# 检索器适配器（LlamaIndex）
#
# 本模块封装 LlamaIndex 的 VectorIndexRetriever，提供高级检索策略，
# 替代 Worker1 中自实现的 Retriever。
#
# 支持的检索策略：
# - 单查询检索（带得分阈值过滤）
# - 多查询检索（关键词拆分 + 结果合并去重）
# - 来源过滤检索（限定在特定文档中检索）
# - LLMRerank 后处理（用 LLM 对检索结果重新排序，提升相关性）
# - 结构化 RetrievalResult 返回

import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from src.core.config import get_config
from src.core.exceptions import VectorStoreError
from src.rag.vector_store import VectorStoreAdapter, get_vector_store


# ------------------------------------------------------------------ #
#                    数据结构定义                                      #
# ------------------------------------------------------------------ #

@dataclass
class RetrievedItem:
    """单条检索结果

    兼容 Worker1 的 RetrievedItem 接口。

    Attributes:
        text: 匹配到的文本内容
        score: 原始相似度得分（距离值，越小越相似）
        normalized_score: 归一化得分（0~1，越高越相似）
        id: 文档唯一 ID
        source: 来源文档路径
        metadata: 额外元数据
        relevance: 相关性等级（"high" / "medium" / "low"）
    """
    text: str
    score: float = 0.0
    normalized_score: float = 0.0
    id: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance: str = ""

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"<RetrievedItem score={self.score:.4f} "
            f"relevance={self.relevance!r} text={text_preview!r}>"
        )


@dataclass
class RetrievalResult:
    """检索结果集

    兼容 Worker1 的 RetrievalResult 接口。

    Attributes:
        query: 原始查询文本
        items: 检索结果列表（按相似度排序）
        total: 结果总数
        elapsed_ms: 检索耗时（毫秒）
        collection_name: 搜索的集合名称
        filters_applied: 已应用的过滤条件描述
    """
    query: str
    items: List[RetrievedItem] = field(default_factory=list)
    total: int = 0
    elapsed_ms: float = 0.0
    collection_name: str = ""
    filters_applied: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """是否未找到任何结果"""
        return len(self.items) == 0

    @property
    def texts(self) -> List[str]:
        """提取所有结果文本（用于构建上下文）"""
        return [item.text for item in self.items]

    @property
    def high_relevance_items(self) -> List[RetrievedItem]:
        """仅返回高相关性结果"""
        return [item for item in self.items if item.relevance == "high"]

    def __repr__(self) -> str:
        return (
            f"<RetrievalResult query={self.query!r} "
            f"total={self.total} elapsed={self.elapsed_ms:.1f}ms>"
        )


# ------------------------------------------------------------------ #
#                    检索器适配器                                      #
# ------------------------------------------------------------------ #

class RetrieverAdapter:
    """LlamaIndex 检索器适配器

    职责：
    1. 封装 LlamaIndex VectorIndexRetriever
    2. 提供 retrieve / retrieve_multi_query / retrieve_from_source 方法
    3. 应用得分过滤和去重策略

    示例：
        >>> retriever = RetrieverAdapter()
        >>> result = retriever.retrieve("AI有哪些应用？", top_k=3)
        >>> for item in result.items:
        ...     print(f"  [{item.relevance}] {item.text[:60]}")
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreAdapter] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        deduplicate: Optional[bool] = None,
        dedup_similarity: Optional[float] = None,
        rerank: Optional[bool] = None,
        rerank_top_n: Optional[int] = None,
    ):
        """初始化检索器适配器

        Args:
            vector_store: VectorStoreAdapter 实例（None 时使用全局单例）
            top_k: 默认返回结果数量（None 时从配置读取）
            score_threshold: 得分阈值（None 时从配置读取）
            deduplicate: 是否启用去重（None 时从配置读取）
            dedup_similarity: 去重相似度阈值（None 时从配置读取）
            rerank: 是否启用 LLMRerank 后处理（None 时从配置读取）
            rerank_top_n: LLMRerank 保留的 Top N 结果数（None 时从配置读取）
        """
        config = get_config()

        self._store = vector_store or get_vector_store()
        self._index = self._store.index
        self._top_k = top_k if top_k is not None else config.retriever_top_k
        self._score_threshold = (
            score_threshold
            if score_threshold is not None
            else config.retriever_score_threshold
        )
        self._deduplicate = (
            deduplicate
            if deduplicate is not None
            else config.retriever_deduplicate
        )
        self._dedup_similarity = (
            dedup_similarity
            if dedup_similarity is not None
            else config.retriever_dedup_similarity
        )

        # LLMRerank 后处理配置
        self._rerank_enabled = (
            rerank
            if rerank is not None
            else config.retriever_rerank_enabled
        )
        self._rerank_top_n = (
            rerank_top_n
            if rerank_top_n is not None
            else config.retriever_rerank_top_n
        )
        # 懒加载：仅在启用时创建 LLMRerank 实例
        self._reranker: Optional[LLMRerank] = None

        print(
            f"[RetrieverAdapter] 初始化完成: top_k={self._top_k}, "
            f"score_threshold={self._score_threshold}, deduplicate={self._deduplicate}, "
            f"rerank={self._rerank_enabled}, rerank_top_n={self._rerank_top_n}"
        )

    # ------------------------------------------------------------------ #
    #                    核心：单查询检索                                  #
    # ------------------------------------------------------------------ #

    def _get_reranker(self) -> LLMRerank:
        """获取 LLMRerank 实例（懒加载单例）

        使用与 query_engine 相同的 OpenAI 兼容 API（DashScope）创建 LLM，
        避免重复初始化的开销。

        Returns:
            LLMRerank 实例
        """
        if self._reranker is None:
            config = get_config()
            llm = LlamaOpenAI(
                model=config.default_model,
                api_key=config.api_key,
                api_base=config.base_url,
                temperature=0.0,  # 重排序需要确定性输出
                max_tokens=512,
            )
            self._reranker = LLMRerank(
                top_n=self._rerank_top_n,
                llm=llm,
            )
            print(
                f"[RetrieverAdapter] LLMRerank 初始化完成: "
                f"model={config.default_model}, top_n={self._rerank_top_n}"
            )
        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
    ) -> RetrievalResult:
        """单查询检索

        Args:
            query: 查询文本
            top_k: 返回结果数量（None 使用默认值）
            rerank: 是否启用 LLMRerank（None 使用默认配置，True/False 临时覆盖）

        Returns:
            包含过滤后结果的 RetrievalResult

        Raises:
            VectorStoreError: 检索失败时抛出
        """
        if not query or not query.strip():
            raise VectorStoreError("查询文本为空")

        k = top_k or self._top_k
        # 确定本次是否启用 LLMRerank
        use_rerank = rerank if rerank is not None else self._rerank_enabled
        # 启用 rerank 时多检索一些结果供 LLM 重排，然后截取 top_n
        retrieve_k = k * 2 if use_rerank else k
        start_time = time.time()

        try:
            # 1. 创建 LlamaIndex 向量检索器，获取候选节点
            retriever = VectorIndexRetriever(
                index=self._index,
                similarity_top_k=retrieve_k,
            )
            nodes = retriever.retrieve(query)

            # 2. LLMRerank 后处理：用 LLM 重新评估节点与查询的相关性
            if use_rerank and nodes:
                try:
                    reranker = self._get_reranker()
                    nodes = reranker.postprocess_nodes(
                        nodes,
                        query_str=query,
                    )
                    print(
                        f"[RetrieverAdapter] LLMRerank 完成: "
                        f"{retrieve_k} 条候选 -> {len(nodes)} 条重排结果"
                    )
                except Exception as rerank_err:
                    # Rerank 失败时降级为原始结果，不阻断主流程
                    print(
                        f"[RetrieverAdapter] LLMRerank 失败，降级为原始结果: "
                        f"{rerank_err}"
                    )

            # 3. 将节点转换为 RetrievedItem，应用得分过滤和去重
            items = self._process_nodes(nodes)
            elapsed_ms = (time.time() - start_time) * 1000

            filters = []
            if use_rerank:
                filters.append("LLMRerank")

            result = RetrievalResult(
                query=query,
                items=items,
                total=len(items),
                elapsed_ms=elapsed_ms,
                collection_name=self._store.collection_name,
                filters_applied=filters,
            )

            return result

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"检索失败: {str(e)}") from e

    # ------------------------------------------------------------------ #
    #                    多查询检索                                       #
    # ------------------------------------------------------------------ #

    def retrieve_multi_query(
        self,
        query: str,
        extra_queries: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """多查询检索（原始查询 + 关键词查询）

        执行多个查询并合并结果，自动去重。

        Args:
            query: 原始查询文本
            extra_queries: 额外的查询列表
            top_k: 最终返回的结果数量

        Returns:
            包含合并结果的 RetrievalResult
        """
        k = top_k or self._top_k
        start_time = time.time()

        # 构建所有查询
        all_queries = [query]
        if extra_queries:
            all_queries.extend(extra_queries)

        # 执行每个查询
        all_items: List[RetrievedItem] = []
        seen_texts = set()  # 用于去重的文本集合

        for q in all_queries:
            try:
                retriever = VectorIndexRetriever(
                    index=self._index,
                    similarity_top_k=k,
                )
                nodes = retriever.retrieve(q)
                items = self._process_nodes(nodes)

                # 合并时去重（使用前 100 个字符作为去重键）
                for item in items:
                    text_key = item.text[:100]
                    if text_key not in seen_texts:
                        seen_texts.add(text_key)
                        all_items.append(item)
            except Exception:
                continue

        # 按得分排序（最佳在前）并限制返回数量
        all_items.sort(key=lambda x: x.score, reverse=True)
        all_items = all_items[:k]

        elapsed_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            items=all_items,
            total=len(all_items),
            elapsed_ms=elapsed_ms,
            collection_name=self._store.collection_name,
            filters_applied=["multi_query"],
        )

    # ------------------------------------------------------------------ #
    #                    来源过滤检索                                     #
    # ------------------------------------------------------------------ #

    def retrieve_from_source(
        self,
        query: str,
        source: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """限定来源文档的检索

        Args:
            query: 查询文本
            source: 来源文件路径（用于过滤）
            top_k: 返回结果数量

        Returns:
            包含来源过滤结果的 RetrievalResult
        """
        k = top_k or self._top_k
        start_time = time.time()

        try:
            # 使用 ChromaDB 直接查询，带 where 过滤条件
            results = self._store.query(
                query_text=query,
                top_k=k,
                where={"source": source},
            )

            items = [
                RetrievedItem(
                    text=r.text,
                    score=r.score,
                    normalized_score=max(0, 1 - r.score),
                    id=r.id,
                    source=r.source,
                    metadata=r.metadata,
                    relevance=self._classify_relevance(r.score),
                )
                for r in results
            ]

            elapsed_ms = (time.time() - start_time) * 1000

            return RetrievalResult(
                query=query,
                items=items,
                total=len(items),
                elapsed_ms=elapsed_ms,
                collection_name=self._store.collection_name,
                filters_applied=[f"source={source}"],
            )

        except Exception as e:
            raise VectorStoreError(f"来源过滤检索失败: {str(e)}") from e

    # ------------------------------------------------------------------ #
    #                    内部辅助方法                                     #
    # ------------------------------------------------------------------ #

    def _process_nodes(self, nodes) -> List[RetrievedItem]:
        """将 LlamaIndex 检索节点转换为 RetrievedItem 列表

        应用得分过滤和去重策略。
        """
        items = []
        for node_with_score in nodes:
            node = node_with_score.node
            score = node_with_score.score if node_with_score.score is not None else 0.0

            # 应用得分阈值过滤
            if score < self._score_threshold:
                continue

            items.append(RetrievedItem(
                text=node.text or "",
                score=score,
                normalized_score=score,  # 已经是 0-1 范围
                id=node.id_ or "",
                source=node.metadata.get("source", "") if node.metadata else "",
                metadata=dict(node.metadata) if node.metadata else {},
                relevance=self._classify_relevance(score),
            ))

        # 去重处理
        if self._deduplicate:
            items = self._deduplicate_items(items)

        return items

    def _deduplicate_items(self, items: List[RetrievedItem]) -> List[RetrievedItem]:
        """使用 SequenceMatcher 去除重复结果"""
        if not items:
            return items

        unique = [items[0]]
        for item in items[1:]:
            is_dup = False
            for existing in unique:
                # 计算文本相似度，超过阈值则视为重复
                ratio = SequenceMatcher(None, item.text, existing.text).ratio()
                if ratio >= self._dedup_similarity:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(item)

        return unique

    def _classify_relevance(self, score: float) -> str:
        """根据得分分类相关性等级

        对于 LlamaIndex 得分（余弦相似度，越高越相似）：
        - >= 0.8: 高相关性
        - >= 0.5: 中等相关性
        - < 0.5:  低相关性
        """
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"


# ------------------------------------------------------------------ #
#                     全局单例 & 便捷获取函数                           #
# ------------------------------------------------------------------ #

_retriever_instance: Optional[RetrieverAdapter] = None


def get_retriever() -> RetrieverAdapter:
    """获取全局检索器适配器单例（懒加载）

    Returns:
        RetrieverAdapter 单例
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RetrieverAdapter()
    return _retriever_instance


def reload_retriever() -> RetrieverAdapter:
    """强制重新创建检索器适配器

    Returns:
        新的 RetrieverAdapter 实例
    """
    global _retriever_instance
    _retriever_instance = RetrieverAdapter()
    return _retriever_instance
