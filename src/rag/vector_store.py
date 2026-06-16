# 向量存储适配器（LlamaIndex + ChromaDB）
#
# 本模块封装 LlamaIndex 的 ChromaVectorStore 和 VectorStoreIndex，
# 提供向量存储和相似度搜索功能，替代 Worker1 中自实现的 VectorStore。
#
# 核心流程：
# - 文档入库：LangChain Document 文本块 → LlamaIndex 节点 → VectorStoreIndex → ChromaDB
# - 查询检索：用户查询 → Embedding 向量化 → 相似度搜索 → SearchResult 列表
#
# Embedding 模型：DashScope text-embedding-v3（通过 OpenAI 兼容 API 调用）

import os
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chromadb
from langchain_core.documents import Document as LangChainDocument
from llama_index.core import Document as LlamaDocument
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.core.config import get_config
from src.core.exceptions import VectorStoreError

# ------------------------------------------------------------------ #
#          兼容 DashScope 等非 OpenAI 原生 Embedding 模型              #
# ------------------------------------------------------------------ #
# LlamaIndex OpenAIEmbedding 的 get_engine() 会校验模型名是否在其
# 内置枚举中（仅支持 text-embedding-ada-002 / 3-small / 3-large），
# DashScope 的 text-embedding-v3 不在其中，会导致 ValueError。
# 这里对 get_engine 做最小化补丁：遇到未知模型时直接返回模型名，
# 使 OpenAIEmbedding 能正常初始化并将模型名透传给 API。

import llama_index.embeddings.openai.base as _openai_emb_base

_original_get_engine = _openai_emb_base.get_engine


def _patched_get_engine(mode, model, mode_model_dict):
    try:
        return _original_get_engine(mode, model, mode_model_dict)
    except ValueError:
        # 非 OpenAI 原生模型（如 DashScope text-embedding-v3），直接返回模型名
        return model


_openai_emb_base.get_engine = _patched_get_engine


# ------------------------------------------------------------------ #
#                    数据结构定义                                      #
# ------------------------------------------------------------------ #

@dataclass
class SearchResult:
    """搜索结果数据对象

    表示单条相似度搜索结果，兼容 Worker1 的 SearchResult 接口。

    Attributes:
        text: 匹配到的文本内容
        score: 相似度得分（距离值，越小越相似）
        id: 文档在 ChromaDB 中的唯一 ID
        source: 来源文档路径
        metadata: 额外元数据
    """
    text: str
    score: float = 0.0
    id: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"<SearchResult score={self.score:.4f} text={text_preview!r}>"


# ------------------------------------------------------------------ #
#                    向量存储适配器                                    #
# ------------------------------------------------------------------ #

class VectorStoreAdapter:
    """LlamaIndex 向量存储适配器

    职责：
    1. 封装 LlamaIndex ChromaVectorStore + VectorStoreIndex
    2. 提供与 Worker1 兼容的 add/query/delete 方法
    3. 通过 OpenAI 兼容 API 使用 DashScope Embedding 模型

    示例：
        >>> store = VectorStoreAdapter()
        >>> store.add_documents(chunks)
        >>> results = store.query("什么是人工智能？", top_k=5)
    """

    def __init__(self):
        """初始化向量存储适配器"""
        config = get_config()

        # ChromaDB 持久化目录
        persist_dir = config.chroma_persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # 初始化 ChromaDB 持久化客户端
        self._chroma_client = chromadb.PersistentClient(path=persist_dir)

        # 集合名称（类似数据库中的表名）
        self._collection_name = config.chroma_collection_name or "default_collection"
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
        )

        # 创建 LlamaIndex ChromaVectorStore（连接 ChromaDB 集合）
        self._chroma_vector_store = ChromaVectorStore(
            chroma_collection=self._chroma_collection,
        )

        # 初始化 Embedding 模型（DashScope 通过 OpenAI 兼容 API 调用）
        self._embed_model = OpenAIEmbedding(
            model=config.embedding_model,
            api_key=config.api_key,
            api_base=config.base_url,
            dimensions=config.embedding_dimensions,
        )

        # 节点解析器：将文档转换为 LlamaIndex 节点
        self._node_parser = SimpleNodeParser.from_defaults()

        # 创建 StorageContext 和 VectorStoreIndex
        # 使用空文档列表初始化，后续通过 add_documents 添加数据
        storage_context = StorageContext.from_defaults(
            vector_store=self._chroma_vector_store,
        )
        self._index = VectorStoreIndex.from_documents(
            documents=[],
            storage_context=storage_context,
            embed_model=self._embed_model,
        )

        print(
            f"[VectorStoreAdapter] 初始化完成: collection={self._collection_name}, "
            f"embedding={config.embedding_model}, dims={config.embedding_dimensions}"
        )

    @property
    def index(self) -> VectorStoreIndex:
        """获取 LlamaIndex VectorStoreIndex（供检索器/查询引擎使用）"""
        return self._index

    @property
    def collection_name(self) -> str:
        """当前集合名称"""
        return self._collection_name

    # ------------------------------------------------------------------ #
    #                         添加文档                                    #
    # ------------------------------------------------------------------ #

    def add_documents(
        self,
        documents: List[LangChainDocument],
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """将 LangChain Document 添加到向量存储

        每个文档会被转换为 LlamaIndex TextNode 并插入索引。

        Args:
            documents: LangChain Document 文本块列表
            collection_name: 目标集合（None 使用当前集合）

        Returns:
            已添加文档的 ID 列表

        Raises:
            VectorStoreError: 添加失败时抛出
            ValueError: 文档列表为空时抛出
        """
        if not documents:
            raise ValueError("文档列表为空")

        try:
            nodes = []
            for i, doc in enumerate(documents):
                # 基于来源路径、序号和内容前缀生成唯一 ID
                doc_id = hashlib.md5(
                    f"{doc.metadata.get('source', 'unknown')}_{i}_{doc.page_content[:50]}".encode()
                ).hexdigest()

                node = TextNode(
                    text=doc.page_content,
                    id_=doc_id,
                    metadata=doc.metadata,
                )
                # 使用 Embedding 模型生成文本向量
                node.embedding = self._embed_model.get_text_embedding(doc.page_content)
                nodes.append(node)

            # 将节点批量插入索引
            self._index.insert_nodes(nodes)

            ids = [n.id_ for n in nodes]
            print(f"[VectorStoreAdapter] 已添加 {len(ids)} 个文档到集合")
            return ids

        except Exception as e:
            raise VectorStoreError(f"添加文档失败: {str(e)}") from e

    def add_chunks(
        self,
        chunks: List[LangChainDocument],
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """add_documents 的别名，向后兼容 Worker1

        Args:
            chunks: LangChain Document 文本块列表
            collection_name: 目标集合

        Returns:
            已添加文档的 ID 列表
        """
        return self.add_documents(chunks, collection_name)

    # ------------------------------------------------------------------ #
    #                         查询 / 搜索                                #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        collection_name: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """相似度搜索

        Args:
            query_text: 查询文本
            top_k: 返回结果数量（默认 5）
            collection_name: 搜索的集合（None 使用当前集合）
            where: 元数据过滤条件

        Returns:
            SearchResult 列表，按相似度排序（最佳在前）

        Raises:
            VectorStoreError: 查询失败时抛出
            ValueError: 查询文本为空时抛出
        """
        if not query_text or not query_text.strip():
            raise ValueError("查询文本为空")

        try:
            # 使用 LlamaIndex 内置检索器进行相似度搜索
            retriever = self._index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query_text)

            results = []
            for node_with_score in nodes:
                node = node_with_score.node
                score = node_with_score.score if node_with_score.score is not None else 0.0
                results.append(SearchResult(
                    text=node.text or "",
                    score=score,
                    id=node.id_ or "",
                    source=node.metadata.get("source", ""),
                    metadata=dict(node.metadata) if node.metadata else {},
                ))

            return results

        except Exception as e:
            raise VectorStoreError(f"查询失败: {str(e)}") from e

    # ------------------------------------------------------------------ #
    #                         集合管理                                    #
    # ------------------------------------------------------------------ #

    def create_collection(self, name: str) -> None:
        """创建新集合

        Args:
            name: 集合名称
        """
        try:
            self._chroma_client.get_or_create_collection(name=name)
            print(f"[VectorStoreAdapter] 集合已创建: {name}")
        except Exception as e:
            raise VectorStoreError(f"创建集合失败: {str(e)}") from e

    def delete_collection(self, name: Optional[str] = None) -> None:
        """删除集合

        Args:
            name: 要删除的集合（None 删除当前集合）
        """
        target = name or self._collection_name
        try:
            self._chroma_client.delete_collection(name=target)
            print(f"[VectorStoreAdapter] 集合已删除: {target}")
        except Exception as e:
            raise VectorStoreError(f"删除集合失败: {str(e)}") from e

    # ------------------------------------------------------------------ #
    #                         删除操作                                    #
    # ------------------------------------------------------------------ #

    def delete_by_ids(self, ids: List[str]) -> None:
        """按 ID 删除文档

        Args:
            ids: 要删除的文档 ID 列表
        """
        if not ids:
            return
        try:
            self._chroma_collection.delete(ids=ids)
            print(f"[VectorStoreAdapter] 按 ID 删除了 {len(ids)} 个文档")
        except Exception as e:
            raise VectorStoreError(f"按 ID 删除失败: {str(e)}") from e

    def delete_by_source(self, source: str, collection_name: Optional[str] = None) -> None:
        """删除指定来源文件的所有文档

        Args:
            source: 来源文件路径
            collection_name: 要删除的集合（None 使用当前集合）
        """
        try:
            collection = self._chroma_collection
            if collection_name and collection_name != self._collection_name:
                collection = self._chroma_client.get_or_create_collection(name=collection_name)
            collection.delete(
                where={"source": source}
            )
            print(f"[VectorStoreAdapter] 已删除来源文档: {source}")
        except Exception as e:
            raise VectorStoreError(f"按来源删除失败: {str(e)}") from e

    def get_by_source(
        self,
        source: str,
        limit: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取指定来源文件的所有文本块

        兼容 Worker1 的 VectorStore.get_by_source 接口。

        Args:
            source: 来源文件路径（用于过滤）
            limit: 最大返回文本块数（None 返回全部）
            collection_name: 查询的集合（None 使用当前集合）

        Returns:
            包含 'text' 和 'metadata' 键的字典列表
        """
        try:
            collection = self._chroma_collection
            if collection_name and collection_name != self._collection_name:
                collection = self._chroma_client.get_or_create_collection(name=collection_name)

            results = collection.get(
                where={"source": source},
                limit=limit,
            )

            if not results or not results.get("documents"):
                return []

            chunks = []
            for i, doc_text in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results.get("metadatas") else {}
                chunks.append({
                    "text": doc_text,
                    "metadata": meta,
                })
            return chunks

        except Exception as e:
            print(f"[VectorStoreAdapter] get_by_source 失败: {str(e)}")
            return []

    # ------------------------------------------------------------------ #
    #                         工具方法                                    #
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        """获取当前集合中的文档总数

        Returns:
            文档数量
        """
        return self._chroma_collection.count()

    def list_collections(self) -> List[str]:
        """列出所有集合名称

        Returns:
            集合名称列表
        """
        return [c.name for c in self._chroma_client.list_collections()]

    def get_sources(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取所有唯一来源的统计信息列表

        兼容 Worker1 的 VectorStore.get_sources 接口。

        Args:
            collection_name: 查询的集合（None 使用当前集合）

        Returns:
            包含 source、chunk_count、total_chars、file_type 的字典列表
        """
        try:
            # 从集合中获取所有文档
            results = self._chroma_collection.get()
            
            if not results or not results.get("metadatas"):
                return []

            # 按来源聚合统计
            source_stats: Dict[str, Dict[str, Any]] = {}
            for i, metadata in enumerate(results["metadatas"]):
                source = metadata.get("source", "unknown")
                if source not in source_stats:
                    source_stats[source] = {
                        "source": source,
                        "chunk_count": 0,
                        "total_chars": 0,
                        "file_type": metadata.get("file_type", ""),
                    }
                source_stats[source]["chunk_count"] += 1
                if results.get("documents") and i < len(results["documents"]):
                    source_stats[source]["total_chars"] += len(results["documents"][i])

            return list(source_stats.values())

        except Exception as e:
            print(f"[VectorStoreAdapter] get_sources 失败: {str(e)}")
            return []

    def __repr__(self) -> str:
        return (
            f"<VectorStoreAdapter collection={self._collection_name!r} "
            f"count={self.count()}>"
        )


# ------------------------------------------------------------------ #
#                     全局单例 & 便捷获取函数                           #
# ------------------------------------------------------------------ #

_vector_store_instance: Optional[VectorStoreAdapter] = None


def get_vector_store() -> VectorStoreAdapter:
    """获取全局向量存储适配器单例（懒加载）

    Returns:
        VectorStoreAdapter 单例
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreAdapter()
    return _vector_store_instance


def reload_vector_store() -> VectorStoreAdapter:
    """强制重新创建向量存储适配器

    Returns:
        新的 VectorStoreAdapter 实例
    """
    global _vector_store_instance
    _vector_store_instance = VectorStoreAdapter()
    return _vector_store_instance

