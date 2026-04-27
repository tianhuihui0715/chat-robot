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
    CONFIG_DIR / "intents.toml",
)

ENV_MAPPING = {
    "APP_NAME": "app_name",
    "APP_ENV": "app_env",
    "LOG_LEVEL": "log_level",
    "API_PREFIX": "api_prefix",
    "RUNTIME_MODE": "runtime_mode",
    "INFERENCE_RUNTIME_MODE": "inference_runtime_mode",
    "INFERENCE_SERVICE_URL": "inference_service_url",
    "INFERENCE_TIMEOUT_SECONDS": "inference_timeout_seconds",
    "TRACE_STORE_DSN": "trace_store_dsn",
    "HOST_MODEL_ROOT": "host_model_root",
    "LLM_MODEL_PATH": "llm_model_path",
    "INTENT_MODEL_PATH": "intent_model_path",
    "EMBEDDING_MODEL_PATH": "embedding_model_path",
    "RERANKER_MODEL_PATH": "reranker_model_path",
    "LLM_MAX_INPUT_TOKENS": "llm_max_input_tokens",
    "LLM_MAX_NEW_TOKENS": "llm_max_new_tokens",
    "LLM_TEMPERATURE": "llm_temperature",
    "GPU_QUEUE_MAXSIZE": "gpu_queue_maxsize",
    "RAG_TOP_K": "rag_top_k",
    "RAG_SCORE_THRESHOLD": "rag_score_threshold",
    "RAG_RERANK_CANDIDATE_LIMIT": "rag_rerank_candidate_limit",
    "RAG_COLLECTION_NAME": "rag_collection_name",
    "RAG_RETRIEVAL_MODE": "rag_retrieval_mode",
    "RAG_BM25_TOP_K": "rag_bm25_top_k",
    "RAG_BM25_TITLE_BOOST": "rag_bm25_title_boost",
    "RAG_RRF_K": "rag_rrf_k",
    "RAG_RRF_MIN_SCORE": "rag_rrf_min_score",
    "RAG_LEXICAL_INDEX_PATH": "rag_lexical_index_path",
    "RAG_CHUNK_SIZE": "rag_chunk_size",
    "RAG_CHUNK_OVERLAP": "rag_chunk_overlap",
    "POSTGRES_HOST": "postgres_host",
    "POSTGRES_PORT": "postgres_port",
    "POSTGRES_DB": "postgres_db",
    "POSTGRES_USER": "postgres_user",
    "POSTGRES_PASSWORD": "postgres_password",
    "POSTGRES_DSN": "postgres_dsn",
    "QDRANT_URL": "qdrant_url",
    "MINIO_ENDPOINT": "minio_endpoint",
    "MINIO_ACCESS_KEY": "minio_access_key",
    "MINIO_SECRET_KEY": "minio_secret_key",
    "MINIO_SECURE": "minio_secure",
    "MINIO_BUCKET": "minio_bucket",
    "LANGSMITH_ENABLED": "langsmith_enabled",
    "LANGSMITH_PROJECT": "langsmith_project",
    "LANGSMITH_ENDPOINT": "langsmith_endpoint",
    "LANGSMITH_API_KEY": "langsmith_api_key",
}


class Settings(BaseModel):
    app_name: str = "Chat Robot"
    app_env: str = "dev"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    runtime_mode: Literal["mock", "remote_inference"] = "mock"
    inference_runtime_mode: Literal["mock", "local_hf"] = "mock"
    inference_service_url: str = "http://localhost:8001"
    inference_timeout_seconds: float = 60.0
    trace_store_dsn: str = "sqlite:///./data/chat_robot.db"

    host_model_root: str = "/root/models"
    llm_model_path: str = ""
    intent_model_path: str = ""
    embedding_model_path: str = ""
    reranker_model_path: str = ""

    llm_max_input_tokens: int = 8192
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.0
    gpu_queue_maxsize: int = 100

    rag_top_k: int = 4
    rag_score_threshold: float = 0.1
    rag_rerank_candidate_limit: int = 12
    rag_collection_name: str = "knowledge_chunks"
    rag_retrieval_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    rag_bm25_top_k: int = 8
    rag_bm25_title_boost: float = 2.0
    rag_rrf_k: int = 60
    rag_rrf_min_score: float = 0.016
    rag_lexical_index_path: str = "./data/rag_lexical.db"
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 80

    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "chat_robot"
    postgres_user: str = "chat_robot"
    postgres_password: str = "chat_robot"
    postgres_dsn: str = "postgresql+psycopg://chat_robot:chat_robot@postgres:5432/chat_robot"

    qdrant_url: str = "http://qdrant:6333"
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket: str = "chat-robot"
    langsmith_enabled: bool = False
    langsmith_project: str = "chat-robot"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str | None = None
    intent_prompt_role: str = "你是一个中文对话系统的意图识别器，只能输出一个 JSON 对象，不能输出解释。"
    intent_prompt_task: str = "根据完整对话历史，判断最后一条用户消息的 intent、是否需要检索、检索词改写和简短依据。"
    intent_prompt_available_intents: list[str] = [
        "chat",
        "knowledge_qa",
        "task",
        "follow_up",
        "reject",
    ]
    intent_prompt_decision_rules: list[str] = [
        "knowledge_qa: 需要查询项目、文档、配置、接口、事实知识。",
        "task: 明确要求执行任务、编写代码、制定方案、产出结构化结果。",
        "follow_up: 明显依赖上一轮上下文的追问、补充、澄清；如果脱离上文无法完整理解，应优先判为 follow_up。",
        "如果最后一句同时像知识问答又明显依赖上文，请优先使用 follow_up，而不是 knowledge_qa。",
        "reject: 涉及违法、危险或不应回答的请求。",
        "其余情况一律使用 chat。",
    ]
    intent_prompt_rewrite_rules: list[str] = [
        "绝不能凭空改写成新的问题。",
        "对 follow_up 允许结合上一轮上下文补全主语，使其变成可独立理解的完整问题。",
        "如果 need_rag=true，rewrite_query 应尽量保留用户原意，只做轻微压缩和规范化。",
        "如果 need_rag=false，rewrite_query 直接等于最后一条用户消息。",
        "不要输出“您有什么问题吗”“请提供更多信息”之类泛化句子。",
    ]
    intent_prompt_rationale_rule: str = "rationale 用一句简短中文说明依据。"
    intent_prompt_output_schema: str = (
        '{"intent":"chat","need_rag":false,"rewrite_query":"原问题","rationale":"判断依据"}'
    )
    intent_prompt_examples: list[dict[str, str]] = []


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
                    "inference_runtime_mode": data.get("runtime", {}).get("inference_mode"),
                    "inference_service_url": data.get("runtime", {}).get("inference_service_url"),
                    "inference_timeout_seconds": data.get("runtime", {}).get("inference_timeout_seconds"),
                    "trace_store_dsn": data.get("observability", {}).get("trace_store_dsn"),
                    "langsmith_enabled": data.get("observability", {}).get("langsmith_enabled"),
                    "langsmith_project": data.get("observability", {}).get("langsmith_project"),
                    "langsmith_endpoint": data.get("observability", {}).get("langsmith_endpoint"),
                    "postgres_host": data.get("database", {}).get("host"),
                    "postgres_port": data.get("database", {}).get("port"),
                    "postgres_db": data.get("database", {}).get("name"),
                    "postgres_user": data.get("database", {}).get("user"),
                    "postgres_password": data.get("database", {}).get("password"),
                    "postgres_dsn": data.get("database", {}).get("dsn"),
                    "qdrant_url": data.get("vector_store", {}).get("qdrant_url"),
                    "minio_endpoint": data.get("object_store", {}).get("endpoint"),
                    "minio_access_key": data.get("object_store", {}).get("access_key"),
                    "minio_secret_key": data.get("object_store", {}).get("secret_key"),
                    "minio_secure": data.get("object_store", {}).get("secure"),
                    "minio_bucket": data.get("object_store", {}).get("bucket"),
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
                    "llm_temperature": data.get("generation", {}).get("llm_temperature"),
                    "gpu_queue_maxsize": data.get("generation", {}).get("gpu_queue_maxsize"),
                }
            )
        elif path.name == "rag.toml":
            merged.update(
                {
                    "rag_top_k": data.get("rag", {}).get("top_k"),
                    "rag_score_threshold": data.get("rag", {}).get("score_threshold"),
                    "rag_rerank_candidate_limit": data.get("rag", {}).get("rerank_candidate_limit"),
                    "rag_collection_name": data.get("rag", {}).get("collection_name"),
                    "rag_retrieval_mode": data.get("rag", {}).get("retrieval_mode"),
                    "rag_bm25_top_k": data.get("rag", {}).get("bm25_top_k"),
                    "rag_bm25_title_boost": data.get("rag", {}).get("bm25_title_boost"),
                    "rag_rrf_k": data.get("rag", {}).get("rrf_k"),
                    "rag_rrf_min_score": data.get("rag", {}).get("rrf_min_score"),
                    "rag_chunk_size": data.get("rag", {}).get("chunk_size"),
                    "rag_chunk_overlap": data.get("rag", {}).get("chunk_overlap"),
                }
            )
        elif path.name == "intents.toml":
            merged.update(
                {
                    "intent_prompt_role": data.get("intent_prompt", {}).get("role"),
                    "intent_prompt_task": data.get("intent_prompt", {}).get("task"),
                    "intent_prompt_available_intents": data.get("intent_prompt", {}).get("available_intents"),
                    "intent_prompt_decision_rules": data.get("intent_prompt", {}).get("decision_rules"),
                    "intent_prompt_rewrite_rules": data.get("intent_prompt", {}).get("rewrite_rules"),
                    "intent_prompt_rationale_rule": data.get("intent_prompt", {}).get("rationale_rule"),
                    "intent_prompt_output_schema": data.get("intent_prompt", {}).get("output_schema"),
                    "intent_prompt_examples": data.get("intent_examples"),
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
