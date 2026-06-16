# 文档入库管线（Document Ingestion Pipeline）
#
# 本模块使用 LlamaIndex Readers 进行文档加载，LangChain TextSplitters 进行文本分割，
# 替代 Worker1 中的 DocumentLoader + TextSplitter 自实现。
#
# 工作流程：
# 1. LlamaIndex Reader 加载文档 → List[LlamaDocument]
# 2. 转换为 LangChain Document 格式
# 3. LangChain TextSplitter 分割文本 → List[LangChain Document chunks]
# 4. 存入向量数据库

import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document as LlamaDocument
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    UnstructuredReader,
)

from src.core.config import get_config
from src.core.exceptions import DocumentError


class DocumentIngestionPipeline:
    """文档入库管线（LlamaIndex + LangChain）
    
    职责：
    1. 使用 LlamaIndex Readers 加载文档（支持更多格式）
    2. 使用 LangChain TextSplitters 分割文本（成熟的分割策略）
    3. 提供与 Worker1 的 DocumentLoader 兼容的接口
    
    示例：
        >>> pipeline = DocumentIngestionPipeline()
        >>> chunks = pipeline.load_and_split("path/to/file.pdf")
        >>> print(f"生成了 {len(chunks)} 个文本块")
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """初始化文档入库管线
        
        Args:
            chunk_size: 每个文本块的目标字符数（None 时从配置读取）
            chunk_overlap: 相邻文本块的重叠字符数（None 时从配置读取）
        """
        config = get_config()
        
        self._chunk_size = chunk_size or config.chunk_size
        self._chunk_overlap = chunk_overlap or config.chunk_overlap
        
        # 初始化 LangChain 递归文本分割器
        # 使用中英文分隔符，优先按段落、句子、标点符号分割
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        
        # 初始化 LlamaIndex 文档读取器
        self._pdf_reader = PDFReader()    # PDF 文件读取
        self._docx_reader = DocxReader()  # Word 文档读取
        
        print(f"[DocumentIngestionPipeline] 初始化完成: chunk_size={self._chunk_size}, overlap={self._chunk_overlap}")
    
    def load_and_split(self, file_path: str) -> List[LangChainDocument]:
        """加载并分割单个文件
        
        将文档加载和文本分割合并为一步操作，简化调用流程。
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            LangChain Document 文本块列表
            
        Raises:
            DocumentError: 文件格式不支持或加载失败时抛出
        """
        path = Path(file_path)
        
        if not path.exists():
            raise DocumentError(f"文件不存在: {file_path}")
        
        ext = path.suffix.lower()
        
        try:
            # 第 1 步：使用 LlamaIndex Reader 加载文档
            llama_docs = self._load_with_llamaindex(file_path, ext)
            
            if not llama_docs:
                raise DocumentError(f"无法从文件中提取内容: {file_path}")
            
            # 第 2 步：转换为 LangChain Document 格式
            # 保留原始元数据并添加来源路径和文件类型
            lc_docs = [
                LangChainDocument(
                    page_content=doc.text,
                    metadata={"source": str(path), "file_type": ext, **doc.metadata}
                )
                for doc in llama_docs
            ]
            
            # 第 3 步：使用 LangChain 分割器将文本切分为小块
            chunks = self._splitter.split_documents(lc_docs)
            
            print(f"[DocumentIngestionPipeline] 加载 {file_path}: {len(chunks)} 个文本块")
            return chunks
            
        except Exception as e:
            raise DocumentError(f"加载/分割失败 {file_path}: {str(e)}") from e
    
    def _load_with_llamaindex(self, file_path: str, ext: str) -> List[LlamaDocument]:
        """使用 LlamaIndex Reader 加载文档
        
        根据文件扩展名自动选择合适的 Reader：
        - .pdf → PDFReader
        - .docx → DocxReader
        - .txt/.md/.markdown → 直接读取
        
        Args:
            file_path: 文件路径
            ext: 文件扩展名
            
        Returns:
            LlamaIndex Document 列表
        """
        if ext == ".pdf":
            return self._pdf_reader.load_data(file_path)
        elif ext == ".docx":
            return self._docx_reader.load_data(file_path)
        elif ext in [".txt", ".md", ".markdown"]:
            # 纯文本文件直接读取
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return [LlamaDocument(text=content, metadata={"source": file_path})]
        else:
            raise DocumentError(f"不支持的文件格式: {ext}")
    
    def ingest_directory(self, dir_path: str) -> int:
        """批量导入目录下的所有文档
        
        Args:
            dir_path: 包含文档的目录路径
            
        Returns:
            生成的文本块总数
            
        Raises:
            DocumentError: 目录不存在时抛出
        """
        dir_path_obj = Path(dir_path)
        
        if not dir_path_obj.exists() or not dir_path_obj.is_dir():
            raise DocumentError(f"目录不存在: {dir_path}")
        
        # 支持的文件扩展名
        supported_exts = {".pdf", ".docx", ".txt", ".md", ".markdown"}
        
        all_chunks = []
        failed_files = []
        
        # 遍历目录中的文件
        for file_path in dir_path_obj.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                try:
                    chunks = self.load_and_split(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"[DocumentIngestionPipeline] 处理文件失败 {file_path}: {e}")
                    failed_files.append(str(file_path))
        
        print(f"[DocumentIngestionPipeline] 目录导入完成:")
        print(f"  - 总文本块数: {len(all_chunks)}")
        print(f"  - 失败文件数: {len(failed_files)}")
        
        if failed_files:
            print(f"  - 失败文件: {failed_files}")
        
        return len(all_chunks)
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式列表
        
        Returns:
            支持的文件扩展名列表
        """
        return [".pdf", ".docx", ".txt", ".md", ".markdown"]


# ------------------------------------------------------------------ #
#                     全局单例 & 便捷获取函数                           #
# ------------------------------------------------------------------ #

_pipeline_instance: Optional[DocumentIngestionPipeline] = None


def get_document_loader() -> DocumentIngestionPipeline:
    """获取全局文档入库管线单例
    
    注意：函数名保持为 get_document_loader 以向后兼容 Worker1。
    
    Returns:
        DocumentIngestionPipeline 单例
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DocumentIngestionPipeline()
    return _pipeline_instance


def reload_document_loader() -> DocumentIngestionPipeline:
    """强制重新创建文档入库管线
    
    Returns:
        新的 DocumentIngestionPipeline 实例
    """
    global _pipeline_instance
    _pipeline_instance = DocumentIngestionPipeline()
    return _pipeline_instance
