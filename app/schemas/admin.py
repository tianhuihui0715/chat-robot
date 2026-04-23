from pydantic import BaseModel, Field, model_validator

from app.schemas.chat import SourceChunk
from app.schemas.traces import TraceDetail, TraceSummary


class RAGRuntimeConfig(BaseModel):
    top_k: int = Field(ge=1, le=20)
    score_threshold: float = Field(ge=0.0, le=1.0)
    candidate_multiplier: int = Field(ge=1, le=10)
    chunk_size: int = Field(ge=100, le=4000)
    chunk_overlap: int = Field(ge=0, le=1000)
    reranker_enabled: bool = True
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    chunking_applies_to: str = "future_ingestion_only"


class RAGRuntimeConfigUpdate(BaseModel):
    top_k: int = Field(ge=1, le=20)
    score_threshold: float = Field(ge=0.0, le=1.0)
    candidate_multiplier: int = Field(ge=1, le=10)
    chunk_size: int = Field(ge=100, le=4000)
    chunk_overlap: int = Field(ge=0, le=1000)
    reranker_enabled: bool
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class TraceDetailPreview(BaseModel):
    trace: TraceDetail
    step_limit: int
    has_more_steps: bool = False
    output_truncated: bool = False


class TracePageResponse(BaseModel):
    items: list[TraceSummary]
    total: int
    page: int
    page_size: int


class RAGCompareVariant(BaseModel):
    name: str
    top_k: int = Field(ge=1, le=20)
    score_threshold: float = Field(ge=0.0, le=1.0)
    candidate_multiplier: int = Field(ge=1, le=10)
    reranker_enabled: bool = True


class RAGCompareRequest(BaseModel):
    query: str = Field(min_length=1)
    variants: list[RAGCompareVariant] = Field(min_length=1, max_length=4)
    generate_answer: bool = True


class RAGCompareResult(BaseModel):
    name: str
    answer: str | None = None
    sources: list[SourceChunk] = Field(default_factory=list)


class RAGCompareResponse(BaseModel):
    query: str
    results: list[RAGCompareResult] = Field(default_factory=list)


class RAGEvaluationCase(BaseModel):
    query: str = Field(min_length=1)
    expected_sources: list[str] = Field(default_factory=list)
    expected_answer_keywords: list[str] = Field(default_factory=list)


class RAGEvaluationVariantResult(BaseModel):
    name: str
    answer: str | None = None
    sources: list[SourceChunk] = Field(default_factory=list)
    source_hit: bool = False
    answer_hit: bool = False
    matched_sources: list[str] = Field(default_factory=list)
    matched_answer_keywords: list[str] = Field(default_factory=list)


class RAGEvaluationCaseResult(BaseModel):
    query: str
    expected_sources: list[str] = Field(default_factory=list)
    expected_answer_keywords: list[str] = Field(default_factory=list)
    variants: list[RAGEvaluationVariantResult] = Field(default_factory=list)


class RAGEvaluationSummary(BaseModel):
    name: str
    total_cases: int
    source_hit_cases: int
    answer_hit_cases: int
    source_hit_rate: float
    answer_hit_rate: float
    average_returned_sources: float


class RAGEvaluationRequest(BaseModel):
    cases: list[RAGEvaluationCase] = Field(min_length=1, max_length=50)
    variants: list[RAGCompareVariant] = Field(min_length=1, max_length=4)
    generate_answer: bool = True


class RAGEvaluationResponse(BaseModel):
    cases: list[RAGEvaluationCaseResult] = Field(default_factory=list)
    summaries: list[RAGEvaluationSummary] = Field(default_factory=list)


class RAGLabDocument(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class RAGLabVariant(BaseModel):
    variant_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)
    chunk_size: int = Field(ge=100, le=4000)
    chunk_overlap_ratio: float = Field(ge=0.0, le=0.8)
    retrieval_k: int = Field(ge=1, le=20)
    rerank_k: int = Field(ge=0, le=20)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_rerank_k(self):
        if self.rerank_k > self.retrieval_k:
            raise ValueError("rerank_k cannot be greater than retrieval_k")
        return self


class RAGLabRequest(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=30)
    documents: list[RAGLabDocument] = Field(min_length=1, max_length=20)
    variants: list[RAGLabVariant] = Field(min_length=1, max_length=4)


class RAGLabSourcePreview(BaseModel):
    document_id: str
    title: str
    preview: str
    content_full: str
    score: float


class RAGLabQuestionResult(BaseModel):
    question: str
    answer_preview: str
    answer_full: str
    sources: list[RAGLabSourcePreview] = Field(default_factory=list)


class RAGLabVariantResult(BaseModel):
    variant_id: str
    name: str
    chunk_size: int
    chunk_overlap_ratio: float
    retrieval_k: int
    rerank_k: int
    temperature: float
    questions: list[RAGLabQuestionResult] = Field(default_factory=list)


class RAGLabSessionResponse(BaseModel):
    session_id: str
    document_count: int
    question_count: int
    variants: list[RAGLabVariantResult] = Field(default_factory=list)


class RAGLabApplyResponse(BaseModel):
    session_id: str
    variant_id: str
    applied_config: RAGRuntimeConfig
    note: str
