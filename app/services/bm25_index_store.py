from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
from threading import Lock

import jieba


@dataclass
class BM25ChunkRecord:
    chunk_id: str
    document_id: str
    title: str
    content: str
    metadata: dict[str, str]


@dataclass
class BM25SearchResult:
    chunk_id: str
    document_id: str
    title: str
    content: str
    metadata: dict[str, str]
    score: float


class SQLiteBM25IndexStore:
    _QUERY_STOPWORDS = {
        "什么",
        "怎么",
        "如何",
        "为何",
        "吗",
        "么",
        "呢",
        "请问",
    }

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        if not self._db_path.is_absolute():
            self._db_path = Path.cwd() / self._db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def setup(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunk_fts
                USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    title UNINDEXED,
                    content UNINDEXED,
                    metadata_json UNINDEXED,
                    title_terms,
                    content_terms,
                    tokenize = 'unicode61'
                )
                """
            )
            connection.commit()

    def count_chunks(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT count(*) FROM knowledge_chunk_fts").fetchone()
            return int(row[0]) if row is not None else 0

    def upsert_chunks(self, chunks: list[BM25ChunkRecord]) -> None:
        if not chunks:
            return

        rows = [
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.title,
                chunk.content,
                json.dumps(chunk.metadata, ensure_ascii=False),
                self._tokenize(chunk.title),
                self._tokenize(chunk.content),
            )
            for chunk in chunks
        ]

        with self._lock:
            with self._connect() as connection:
                connection.executemany(
                    "DELETE FROM knowledge_chunk_fts WHERE chunk_id = ?",
                    [(chunk.chunk_id,) for chunk in chunks],
                )
                connection.executemany(
                    """
                    INSERT INTO knowledge_chunk_fts (
                        chunk_id,
                        document_id,
                        title,
                        content,
                        metadata_json,
                        title_terms,
                        content_terms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                connection.commit()

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            with self._connect() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM knowledge_chunk_fts WHERE document_id = ? LIMIT 1",
                    (document_id,),
                ).fetchone()
                if exists is None:
                    return False
                connection.execute(
                    "DELETE FROM knowledge_chunk_fts WHERE document_id = ?",
                    (document_id,),
                )
                connection.commit()
                return True

    def search(
        self,
        query: str,
        *,
        limit: int,
        title_boost: float = 2.0,
        knowledge_base_id: str | None = None,
    ) -> list[BM25SearchResult]:
        match_query = self._to_match_query(query)
        if not match_query:
            return []

        with self._connect() as connection:
            escaped_match = match_query.replace("'", "''")
            fetch_limit = int(limit) if not knowledge_base_id else max(int(limit) * 10, 50)
            rows = connection.execute(
                f"""
                SELECT
                    chunk_id,
                    document_id,
                    title,
                    content,
                    metadata_json,
                    bm25(knowledge_chunk_fts, ?, 1.0) AS raw_score
                FROM knowledge_chunk_fts
                WHERE knowledge_chunk_fts MATCH '{escaped_match}'
                ORDER BY raw_score ASC
                LIMIT ?
                """,
                (float(title_boost), fetch_limit),
            ).fetchall()

        results: list[BM25SearchResult] = []
        for row in rows:
            metadata = self._load_metadata(row["metadata_json"])
            if knowledge_base_id and metadata.get("knowledge_base_id", "default") != knowledge_base_id:
                continue
            raw_score = float(row["raw_score"])
            results.append(
                BM25SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    title=str(row["title"]),
                    content=str(row["content"]),
                    metadata=metadata,
                    score=self._normalize_score(raw_score),
                )
            )
            if len(results) >= limit:
                break
        return results

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _load_metadata(value: str | None) -> dict[str, str]:
        if not value:
            return {}
        try:
            raw = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return {str(key): str(item) for key, item in raw.items()}

    @staticmethod
    def _tokenize(text: str) -> str:
        return " ".join(
            token.strip()
            for token in jieba.cut_for_search(text)
            if SQLiteBM25IndexStore._is_meaningful_token(token.strip())
        )

    @classmethod
    def _to_match_query(cls, text: str) -> str:
        tokens = [
            token
            for token in cls._tokenize(text).split()
            if token and token not in cls._QUERY_STOPWORDS
        ]
        if not tokens:
            return ""
        return " OR ".join(f'"{token.replace(chr(34), chr(32)).strip()}"' for token in dict.fromkeys(tokens))

    @staticmethod
    def _normalize_score(raw_score: float) -> float:
        bounded = max(raw_score, 0.0)
        return 1.0 / (1.0 + bounded)

    @staticmethod
    def _is_meaningful_token(token: str) -> bool:
        if not token:
            return False
        if re.search(r"[0-9A-Za-z\u4e00-\u9fff]", token) is None:
            return False
        if token in SQLiteBM25IndexStore._QUERY_STOPWORDS:
            return False
        if token.isascii() and len(token) == 1 and not token.isdigit():
            return False
        return True
