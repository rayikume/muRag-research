from pathlib import Path

import duckdb
from loguru import logger

from models.schemas import RetrievedChunk


class DuckDBStore:
    def __init__(self, db_path: Path, embedding_dim: int) -> None:
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(db_path))
        self._setup()

    def _setup(self) -> None:
        self.conn.execute("INSTALL vss;")
        self.conn.execute("LOAD vss;")
        self.conn.execute("SET hnsw_enable_experimental_persistence = true;")

        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS chunks_id_seq;")
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id        INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
                source    VARCHAR,
                text      VARCHAR,
                embedding FLOAT[{self.embedding_dim}]
            );
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_hnsw
            ON chunks USING HNSW (embedding)
            WITH (metric = 'cosine');
        """)

        logger.debug(
            "DuckDB store ready at {} (dim={})", self.db_path, self.embedding_dim
        )

    def insert_chunks(self, rows: list[tuple[str, str, list[float]]]) -> None:
        self.conn.executemany(
            "INSERT INTO chunks (source, text, embedding) VALUES (?, ?, ?);",
            rows,
        )
        logger.info("Inserted {} chunks", len(rows))

    def similarity_search(
        self, query_vector: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        sql = f"""
            SELECT text, source,
                   array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) AS score
            FROM chunks
            ORDER BY score DESC
            LIMIT ?;
        """
        rows = self.conn.execute(sql, [query_vector, top_k]).fetchall()
        return [RetrievedChunk(text=t, source=s, score=sc) for t, s, sc in rows]

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()
        return row[0] if row is not None else 0

    def clear(self) -> None:
        self.conn.execute("DELETE FROM chunks;")
        logger.warning("Cleared all chunks from store")

    def close(self) -> None:
        self.conn.close()
