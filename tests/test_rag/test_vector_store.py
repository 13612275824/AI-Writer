# Unit tests for Vector Store Adapter (LlamaIndex + ChromaDB)

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure Worker2/ is on sys.path so `src` package can be found
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class TestVectorStoreAdapter(unittest.TestCase):
    """Test VectorStoreAdapter with fully mocked dependencies"""

    @patch("src.rag.vector_store.VectorStoreIndex")
    @patch("src.rag.vector_store.StorageContext")
    @patch("src.rag.vector_store.SimpleNodeParser")
    @patch("src.rag.vector_store.OpenAIEmbedding")
    @patch("src.rag.vector_store.ChromaVectorStore")
    @patch("src.rag.vector_store.chromadb.PersistentClient")
    @patch("src.rag.vector_store.get_config")
    def setUp(self, mock_get_config, mock_persistent_client, mock_chroma_vs,
              mock_embedding, mock_node_parser, mock_storage_ctx, mock_index):
        cfg = MagicMock()
        cfg.chroma_persist_dir = "data/vectors"
        cfg.chroma_collection_name = "test_collection"
        cfg.embedding_model = "text-embedding-v3"
        cfg.api_key = "test-key"
        cfg.base_url = "https://test.com/v1"
        cfg.embedding_dimensions = 1024
        mock_get_config.return_value = cfg

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_persistent_client.return_value.get_or_create_collection.return_value = mock_collection

        from src.rag.vector_store import VectorStoreAdapter
        self.adapter = VectorStoreAdapter()

    def test_collection_name_property(self):
        """Test collection_name property"""
        self.assertEqual(self.adapter.collection_name, "test_collection")

    def test_count(self):
        """Test count method"""
        self.adapter._chroma_collection.count.return_value = 42
        self.assertEqual(self.adapter.count(), 42)

    def test_add_chunks(self):
        """Test add_chunks with LangChain documents"""
        from langchain_core.documents import Document
        docs = [
            Document(page_content="Test content 1", metadata={"source": "test1.txt"}),
            Document(page_content="Test content 2", metadata={"source": "test2.txt"}),
        ]
        result = self.adapter.add_chunks(docs)
        # add_chunks returns list of IDs
        self.assertEqual(len(result), 2)

    def test_get_sources_empty(self):
        """Test get_sources with empty collection"""
        self.adapter._chroma_collection.get.return_value = {"metadatas": [], "documents": []}
        sources = self.adapter.get_sources()
        self.assertEqual(sources, [])

    def test_get_sources_with_data(self):
        """Test get_sources aggregates by source"""
        self.adapter._chroma_collection.get.return_value = {
            "metadatas": [
                {"source": "file1.txt", "file_type": ".txt"},
                {"source": "file1.txt", "file_type": ".txt"},
                {"source": "file2.pdf", "file_type": ".pdf"},
            ],
            "documents": ["content1", "content2", "content3"],
        }
        sources = self.adapter.get_sources()
        self.assertEqual(len(sources), 2)

    def test_get_by_source(self):
        """Test get_by_source returns chunks"""
        self.adapter._chroma_collection.get.return_value = {
            "documents": ["chunk1", "chunk2"],
            "metadatas": [
                {"source": "test.txt", "chunk_index": 0},
                {"source": "test.txt", "chunk_index": 1},
            ],
        }
        chunks = self.adapter.get_by_source(source="test.txt")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "chunk1")

    def test_delete_by_source(self):
        """Test delete_by_source calls collection.delete"""
        self.adapter.delete_by_source(source="test.txt")
        self.adapter._chroma_collection.delete.assert_called_once()

    def test_repr(self):
        """Test repr output"""
        self.adapter._chroma_collection.count.return_value = 10
        r = repr(self.adapter)
        self.assertIn("test_collection", r)
        self.assertIn("10", r)


class TestVectorStoreSingleton(unittest.TestCase):

    @patch("src.rag.vector_store.VectorStoreIndex")
    @patch("src.rag.vector_store.StorageContext")
    @patch("src.rag.vector_store.SimpleNodeParser")
    @patch("src.rag.vector_store.OpenAIEmbedding")
    @patch("src.rag.vector_store.ChromaVectorStore")
    @patch("src.rag.vector_store.chromadb.PersistentClient")
    @patch("src.rag.vector_store.get_config")
    def test_singleton(self, mock_get_config, mock_pc, mock_cvs,
                       mock_emb, mock_np, mock_sc, mock_idx):
        cfg = MagicMock()
        cfg.chroma_persist_dir = "data/vectors"
        cfg.chroma_collection_name = "test"
        cfg.embedding_model = "text-embedding-v3"
        cfg.api_key = "test-key"
        cfg.base_url = "https://test.com/v1"
        cfg.embedding_dimensions = 1024
        mock_get_config.return_value = cfg

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_pc.return_value.get_or_create_collection.return_value = mock_collection

        import src.rag.vector_store as module
        module._vector_store_instance = None

        from src.rag.vector_store import get_vector_store
        s1 = get_vector_store()
        s2 = get_vector_store()
        self.assertIs(s1, s2)


if __name__ == "__main__":
    unittest.main()
