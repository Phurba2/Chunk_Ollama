"""Minimal entry points for Markdown indexing and semantic search."""

from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
from src.embeddings import EmbeddingGenerator
from src.markdown_processor import create_default_processor
from src.embedding_pipeline import EmbeddingPipeline
from src.search import PaperSearchEngine, SearchMode

DB_CONFIG = {
    "host": DB_HOST,
    "port": DB_PORT,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
}


def ingest_and_embed(limit: int = 100):
    """Register Markdown files in markdown/, chunk them, and generate embeddings."""
    registration = create_default_processor().ingest()
    pipeline = EmbeddingPipeline(DB_CONFIG, EmbeddingGenerator())
    processing = pipeline.process_pending_papers(limit=limit)
    return {"registration": registration, "processing": processing}


def search(query: str, mode: SearchMode = SearchMode.HYBRID, limit: int = 5):
    """Search indexed Markdown files and return ranked matching chunks."""
    return PaperSearchEngine(DB_CONFIG, EmbeddingGenerator()).search(query=query, mode=mode, limit=limit)
