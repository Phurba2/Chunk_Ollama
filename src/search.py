import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from src.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


@dataclass
class SearchResult:
    paper_id: int
    filename: str
    title: str
    score: float
    matched_chunks: List[Dict]


class PaperSearchEngine:
    def __init__(self, db_config: dict, embedding_generator: EmbeddingGenerator):
        self.db_config = db_config
        self.embedding_generator = embedding_generator

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def search(self, query: str, mode=SearchMode.VECTOR, limit=10, filters: Optional[Dict] = None):
        if not query or limit <= 0:
            return []
        if mode == SearchMode.VECTOR:
            return self._vector_search(query, limit)
        if mode == SearchMode.KEYWORD:
            return self._keyword_search(query, limit)
        if mode == SearchMode.HYBRID:
            return self._combine_results(self._vector_search(query, limit * 2), self._keyword_search(query, limit * 2), limit)
        raise ValueError(f"Unknown search mode: {mode}")

    def _vector_search(self, query, limit):
        embedding = self.embedding_generator.generate_query_embedding(query)
        sql = """
            SELECT p.id AS paper_id, p.filename, p.title,
                   1 - (c.embedding <=> %s::vector) AS score,
                   c.id AS chunk_id, c.chunk_index, c.chunk_text,
                   c.section_name
            FROM paper_chunks c JOIN papers p ON p.id = c.paper_id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s::vector LIMIT %s
        """
        with self._get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (embedding.tolist(), embedding.tolist(), max(limit * 10, 50)))
            return self._group(cur.fetchall(), limit)

    def _keyword_search(self, query, limit):
        sql = """
            SELECT p.id AS paper_id, p.filename, p.title,
                   similarity(c.chunk_text, %s) AS score,
                   c.id AS chunk_id, c.chunk_index, c.chunk_text,
                   c.section_name
            FROM paper_chunks c JOIN papers p ON p.id = c.paper_id
            WHERE c.chunk_text %% %s
            ORDER BY similarity(c.chunk_text, %s) DESC LIMIT %s
        """
        with self._get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (query, query, query, max(limit * 10, 50)))
            return self._group(cur.fetchall(), limit)

    def _group(self, rows, limit):
        grouped = {}
        for row in rows:
            item = grouped.setdefault(row["paper_id"], {
                "paper_id": row["paper_id"], "filename": row["filename"],
                "title": row["title"], "score": float(row["score"]), "matched_chunks": []
            })
            item["score"] = max(item["score"], float(row["score"]))
            item["matched_chunks"].append({
                "chunk_id": row["chunk_id"], "chunk_index": row["chunk_index"],
                "text": row["chunk_text"], "section_name": row["section_name"],
                "score": float(row["score"])
            })
        results = sorted(grouped.values(), key=lambda x: x["score"], reverse=True)
        for result in results:
            result["matched_chunks"] = sorted(result["matched_chunks"], key=lambda x: x["score"], reverse=True)[:3]
        return [SearchResult(**result) for result in results[:limit]]

    def _combine_results(self, vectors, keywords, limit):
        combined = {r.paper_id: [r, r.score, 0.0] for r in vectors}
        for result in keywords:
            if result.paper_id in combined:
                combined[result.paper_id][2] = result.score
            else:
                combined[result.paper_id] = [result, 0.0, result.score]
        output = []
        for result, vector_score, keyword_score in combined.values():
            result.score = 0.7 * vector_score + 0.3 * keyword_score
            output.append(result)
        return sorted(output, key=lambda x: x.score, reverse=True)[:limit]

    def find_similar_papers(self, paper_id, limit=10):
        with self._get_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT embedding FROM paper_chunks WHERE paper_id = %s AND embedding IS NOT NULL ORDER BY id LIMIT 1", (paper_id,))
            row = cur.fetchone()
            if not row:
                return []
            cur.execute("""
                SELECT p.id AS paper_id, p.filename, p.title,
                       1 - (c.embedding <=> %s::vector) AS score,
                       c.id AS chunk_id, c.chunk_index, c.chunk_text,
                       c.section_name
                FROM paper_chunks c JOIN papers p ON p.id = c.paper_id
                WHERE c.embedding IS NOT NULL AND p.id != %s
                ORDER BY c.embedding <=> %s::vector LIMIT %s
            """, (row["embedding"], paper_id, row["embedding"], max(limit * 10, 50)))
            return self._group(cur.fetchall(), limit)
