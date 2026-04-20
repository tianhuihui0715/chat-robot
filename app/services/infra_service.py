from __future__ import annotations

from dataclasses import dataclass

from minio import Minio
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.config import Settings


@dataclass
class InfraService:
    settings: Settings
    postgres_engine: Engine | None = None
    qdrant_client: QdrantClient | None = None
    minio_client: Minio | None = None

    def setup(self) -> None:
        self.postgres_engine = create_engine(
            self.settings.postgres_dsn,
            future=True,
            pool_pre_ping=True,
        )
        self.qdrant_client = QdrantClient(url=self.settings.qdrant_url)
        self.minio_client = Minio(
            endpoint=self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
            secure=self.settings.minio_secure,
        )

    def shutdown(self) -> None:
        if self.postgres_engine is not None:
            self.postgres_engine.dispose()
            self.postgres_engine = None
        self.qdrant_client = None
        self.minio_client = None

    def check_postgres(self) -> bool:
        if self.postgres_engine is None:
            return False
        try:
            with self.postgres_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def check_qdrant(self) -> bool:
        if self.qdrant_client is None:
            return False
        try:
            self.qdrant_client.get_collections()
            return True
        except Exception:
            return False

    def check_minio(self) -> bool:
        if self.minio_client is None:
            return False
        try:
            self.minio_client.list_buckets()
            return True
        except Exception:
            return False

    def ensure_minio_bucket(self) -> bool:
        if self.minio_client is None:
            return False
        try:
            if not self.minio_client.bucket_exists(self.settings.minio_bucket):
                self.minio_client.make_bucket(self.settings.minio_bucket)
            return True
        except Exception:
            return False
