import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "config"
CONFIG_FILES = (
    CONFIG_DIR / "app.toml",
    CONFIG_DIR / "models.toml",
    CONFIG_DIR / "rag.toml",
)

ENV_MAPPING = {
    "APP_NAME": "app_name",
    "APP_ENV": "app_env",
    "LOG_LEVEL": "log_level",
    "API_PREFIX": "api_prefix",
    "RUNTIME_MODE": "runtime_mode",
    "HOST_MODEL_ROOT": "host_model_root",
    "LLM_MODEL_PATH": "llm_model_path",
    "INTENT_MODEL_PATH": "intent_model_path",
    "EMBEDDING_MODEL_PATH": "embedding_model_path",
    "RERANKER_MODEL_PATH": "reranker_model_path",
    "LLM_MAX_INPUT_TOKENS": "llm_max_input_tokens",
    "LLM_MAX_NEW_TOKENS": "llm_max_new_tokens",
    "GPU_QUEUE_MAXSIZE": "gpu_queue_maxsize",
    "RAG_TOP_K": "rag_top_k",
    "RAG_SCORE_THRESHOLD": "rag_score_threshold",
    "POSTGRES_HOST": "postgres_host",
    "POSTGRES_PORT": "postgres_port",
    "POSTGRES_DB": "postgres_db",
    "POSTGRES_USER": "postgres_user",
    "POSTGRES_PASSWORD": "postgres_password",
    "POSTGRES_DSN": "postgres_dsn",
    "QDRANT_URL": "qdrant_url",
}


class Settings(BaseModel):
    app_name: str = "Chat Robot"
    app_env: str = "dev"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    runtime_mode: Literal["mock", "local_hf"] = "mock"

    host_model_root: str = "/root/models"
    llm_model_path: str = ""
    intent_model_path: str = ""
    embedding_model_path: str = ""
    reranker_model_path: str = ""

    llm_max_input_tokens: int = 8192
    llm_max_new_tokens: int = 512
    gpu_queue_maxsize: int = 100

    rag_top_k: int = 4
    rag_score_threshold: float = 0.1

    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "chat_robot"
    postgres_user: str = "chat_robot"
    postgres_password: str = "chat_robot"
    postgres_dsn: str = "postgresql+psycopg://chat_robot:chat_robot@postgres:5432/chat_robot"

    qdrant_url: str = "http://qdrant:6333"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as file:
        return tomllib.load(file)


def _load_config_defaults() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in CONFIG_FILES:
        data = _read_toml(path)
        if path.name == "app.toml":
            merged.update(
                {
                    "app_name": data.get("app", {}).get("name"),
                    "app_env": data.get("app", {}).get("env"),
                    "log_level": data.get("app", {}).get("log_level"),
                    "api_prefix": data.get("app", {}).get("api_prefix"),
                    "runtime_mode": data.get("runtime", {}).get("mode"),
                    "postgres_host": data.get("database", {}).get("host"),
                    "postgres_port": data.get("database", {}).get("port"),
                    "postgres_db": data.get("database", {}).get("name"),
                    "postgres_user": data.get("database", {}).get("user"),
                    "postgres_password": data.get("database", {}).get("password"),
                    "postgres_dsn": data.get("database", {}).get("dsn"),
                    "qdrant_url": data.get("vector_store", {}).get("qdrant_url"),
                }
            )
        elif path.name == "models.toml":
            merged.update(
                {
                    "host_model_root": data.get("paths", {}).get("host_model_root"),
                    "llm_model_path": data.get("paths", {}).get("llm_model_path"),
                    "intent_model_path": data.get("paths", {}).get("intent_model_path"),
                    "embedding_model_path": data.get("paths", {}).get("embedding_model_path"),
                    "reranker_model_path": data.get("paths", {}).get("reranker_model_path"),
                    "llm_max_input_tokens": data.get("generation", {}).get("llm_max_input_tokens"),
                    "llm_max_new_tokens": data.get("generation", {}).get("llm_max_new_tokens"),
                    "gpu_queue_maxsize": data.get("generation", {}).get("gpu_queue_maxsize"),
                }
            )
        elif path.name == "rag.toml":
            merged.update(
                {
                    "rag_top_k": data.get("rag", {}).get("top_k"),
                    "rag_score_threshold": data.get("rag", {}).get("score_threshold"),
                }
            )
    return {key: value for key, value in merged.items() if value is not None}


def _load_env_overrides() -> dict[str, Any]:
    load_dotenv(BASE_DIR / ".env", override=False)
    overrides: dict[str, Any] = {}
    for env_name, field_name in ENV_MAPPING.items():
        value = os.getenv(env_name)
        if value is not None:
            overrides[field_name] = value
    return overrides


@lru_cache
def get_settings() -> Settings:
    merged = {}
    merged.update(_load_config_defaults())
    merged.update(_load_env_overrides())
    return Settings.model_validate(merged)
