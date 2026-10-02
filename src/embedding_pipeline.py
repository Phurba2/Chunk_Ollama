import logging
import re
from typing import Dict, List

import numpy as np
import psycopg2
from psycopg2.extras import execute_batch

from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, EMBEDDING_BATCH_SIZE
from src.embeddings import EmbeddingGenerator
from src.markdown_extractor import MarkdownExtractor
from src.text_chunker import TextChunker

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(self, db_config: dict, embedding_generator: EmbeddingGenerator, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.db_config = db_config
        self.embedding_generator = embedding_generator
        self.batch_size = batch_size

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def process_paper(self, paper_id: int, chunks: List[Dict]) -> Dict[str, object]:
        if not chunks:
            return {"paper_id": paper_id, "chunks": 0, "embedded": 0, "status": "empty"}
        embeddings = self.embedding_generator.generate_embeddings([chunk["text"] for chunk in chunks], show_progress=True)
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (paper_id,))
                rows = []
                for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    text = chunk["text"]
                    rows.append((paper_id, index, text, chunk.get("token_count"), embedding.tolist(), chunk.get("section_name"), self._detect_math(text), self._detect_code(text), self._detect_references(text)))
                execute_batch(cursor, """
                    INSERT INTO paper_chunks
                    (paper_id, chunk_index, chunk_text, chunk_tokens, embedding, section_name, has_math, has_code, has_references)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, rows, page_size=self.batch_size)
                cursor.execute("""
                    UPDATE papers SET embedding_generated = TRUE, processed = TRUE,
                    processing_error = NULL, updated_at = CURRENT_TIMESTAMP WHERE id = %s
                """, (paper_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {"paper_id": paper_id, "chunks": len(chunks), "embedded": len(embeddings), "status": "completed"}

    @staticmethod
    def _detect_math(text):
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in [r"\\frac", r"\\sum", r"\\int", r"∑", r"∫", r"≤", r"≥", r"≈", r"\bEquation\s+\d+"])

    @staticmethod
    def _detect_code(text):
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in [r"\bdef\s+\w+\(", r"\bclass\s+\w+", r"\bimport\s+\w+", r"```"])

    @staticmethod
    def _detect_references(text):
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in [r"\[\d+\]", r"\(\w+\s+et al\.,?\s+\d{4}\)", r"\bdoi:\s*10\."])

    def process_pending_papers(self, limit: int = 10) -> Dict[str, object]:
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, file_path FROM papers WHERE embedding_generated = FALSE ORDER BY id LIMIT %s", (limit,))
                papers = cursor.fetchall()
        finally:
            conn.close()

        processed = failed = 0
        for paper_id, file_path in papers:
            try:
                extracted = MarkdownExtractor().extract(file_path)
                chunks = TextChunker().chunk_paper(text=extracted["text"], sections=extracted["sections"])
                self.process_paper(paper_id, chunks)
                processed += 1
            except Exception as error:
                failed += 1
                logger.exception("Failed to process document %d: %s", paper_id, error)
                self._record_error(paper_id, str(error))
        return {"requested": limit, "found": len(papers), "processed": processed, "failed": failed}

    def _record_error(self, paper_id: int, error: str):
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("UPDATE papers SET processing_error = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s", (error, paper_id))
            conn.commit()
        finally:
            conn.close()


def create_default_pipeline():
    return EmbeddingPipeline(
        db_config={"host": DB_HOST, "port": DB_PORT, "dbname": DB_NAME, "user": DB_USER, "password": DB_PASSWORD},
        embedding_generator=EmbeddingGenerator(),
    )