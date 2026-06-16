# Unit tests for Document Ingestion Pipeline

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure Worker2/ is on sys.path so `src` package can be found
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class TestDocumentIngestionPipeline(unittest.TestCase):
    """Test DocumentIngestionPipeline with mocked dependencies"""

    @patch("src.rag.document_ingestion.get_config")
    def setUp(self, mock_get_config):
        cfg = MagicMock()
        cfg.chunk_size = 500
        cfg.chunk_overlap = 50
        mock_get_config.return_value = cfg

        from src.rag.document_ingestion import DocumentIngestionPipeline
        self.pipeline = DocumentIngestionPipeline()

    def test_get_supported_formats(self):
        """Test supported file formats"""
        formats = self.pipeline.get_supported_formats()
        self.assertIn(".pdf", formats)
        self.assertIn(".docx", formats)
        self.assertIn(".txt", formats)
        self.assertIn(".md", formats)

    def test_load_and_split_txt_file(self):
        """Test loading and splitting a .txt file"""
        # Create a temporary txt file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("This is test content. " * 50)
            tmp_path = f.name

        try:
            chunks = self.pipeline.load_and_split(tmp_path)
            self.assertGreater(len(chunks), 0)
            # Each chunk should be a LangChain Document
            from langchain_core.documents import Document
            self.assertIsInstance(chunks[0], Document)
            self.assertIn("source", chunks[0].metadata)
        finally:
            os.unlink(tmp_path)

    def test_load_and_split_nonexistent_file(self):
        """Test loading nonexistent file raises DocumentError"""
        from src.core.exceptions import DocumentError
        with self.assertRaises(DocumentError):
            self.pipeline.load_and_split("/nonexistent/file.txt")

    def test_load_and_split_unsupported_format(self):
        """Test loading unsupported format raises DocumentError"""
        from src.core.exceptions import DocumentError
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"dummy content")
            tmp_path = f.name

        try:
            with self.assertRaises(DocumentError):
                self.pipeline.load_and_split(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_ingest_directory(self):
        """Test directory ingestion"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                with open(os.path.join(tmpdir, f"test{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Test content {i}. " * 30)

            total_chunks = self.pipeline.ingest_directory(tmpdir)
            self.assertGreater(total_chunks, 0)


class TestDocumentIngestionSingleton(unittest.TestCase):

    @patch("src.rag.document_ingestion.get_config")
    def test_singleton(self, mock_get_config):
        cfg = MagicMock()
        cfg.chunk_size = 500
        cfg.chunk_overlap = 50
        mock_get_config.return_value = cfg

        import src.rag.document_ingestion as module
        module._pipeline_instance = None

        from src.rag.document_ingestion import get_document_loader
        loader1 = get_document_loader()
        loader2 = get_document_loader()
        self.assertIs(loader1, loader2)


if __name__ == "__main__":
    unittest.main()
