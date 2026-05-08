from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.schemas.chat import SourceChunk
from app.services.knowledge_base import KnowledgeBase
from app.services.retriever_service import RetrieverService
from app.services.trace_service import TraceService


@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()
    enabled: bool = True
    exposed_to_planner: bool = False
    requires_knowledge_base: bool = False


@dataclass
class ToolExecutionContext:
    request_id: str | None
    session_id: str | None
    knowledge_base_id: str | None
    use_reranker: bool | None
    retrieval_profile: dict[str, int] = field(default_factory=dict)


def render_tool_catalog(definitions: list[ToolDefinition]) -> str:
    sections: list[str] = []
    for definition in definitions:
        parameter_parts: list[str] = []
        for parameter in definition.parameters:
            required = "required" if parameter.required else "optional"
            parameter_parts.append(
                f"{parameter.name}:{parameter.type}:{required}:{parameter.description}"
            )
        parameters = "; ".join(parameter_parts) or "none"
        sections.append(
            f"- {definition.name}: {definition.description} | parameters={parameters}"
        )
    return "\n".join(sections)


@dataclass
class ToolExecutionResult:
    tool_name: str
    ok: bool
    payload: dict[str, Any]
    summary: str
    sources: list[SourceChunk] = field(default_factory=list)


class Tool(Protocol):
    @property
    def definition(self) -> ToolDefinition:
        ...

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self, *, planner_only: bool = False) -> list[ToolDefinition]:
        definitions = []
        for tool in self._tools.values():
            definition = tool.definition
            if not definition.enabled:
                continue
            if planner_only and not definition.exposed_to_planner:
                continue
            definitions.append(definition)
        return sorted(definitions, key=lambda item: item.name)

    def planner_tool_names(self) -> list[str]:
        return [definition.name for definition in self.list_tools(planner_only=True)]


class ToolRuntime:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        tool = self._registry.get(tool_name)
        if tool is None or not tool.definition.enabled:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                payload={"error": f"Unknown tool: {tool_name}"},
                summary=f"工具 {tool_name} 不存在或未启用。",
            )
        return await tool.execute(arguments, context)


class RetrievalSearchTool:
    def __init__(self, retriever_service: RetrieverService) -> None:
        self._retriever_service = retriever_service
        self._definition = ToolDefinition(
            name="retrieval.search",
            description="根据查询从当前知识库中检索相关片段。",
            parameters=(
                ToolParameter(name="query", type="string", description="检索查询语句"),
                ToolParameter(name="top_k", type="integer", description="返回片段数", required=False),
            ),
            exposed_to_planner=True,
            requires_knowledge_base=True,
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolExecutionResult(
                tool_name=self._definition.name,
                ok=False,
                payload={"error": "query is required"},
                summary="缺少检索 query。",
            )
        profile = context.retrieval_profile or {}
        explicit_top_k = arguments.get("top_k")
        top_k_override = _coerce_int(explicit_top_k) or profile.get("top_k")
        sources = await self._retriever_service.retrieve(
            query,
            use_reranker=context.use_reranker,
            knowledge_base_id=context.knowledge_base_id,
            top_k_override=top_k_override,
            candidate_multiplier_override=profile.get("candidate_multiplier"),
            rerank_candidate_limit_override=profile.get("rerank_candidate_limit"),
            bm25_top_k_override=profile.get("bm25_top_k"),
        )
        return ToolExecutionResult(
            tool_name=self._definition.name,
            ok=True,
            payload={
                "query": query,
                "retrieved_count": len(sources),
                "source_ids": [source.document_id for source in sources],
            },
            summary=f"检索完成，返回 {len(sources)} 条来源。",
            sources=sources,
        )


class AnswerDirectTool:
    def __init__(self) -> None:
        self._definition = ToolDefinition(
            name="answer.direct",
            description="显式表示该问题可直接回答，不需要外部工具执行。",
            parameters=(
                ToolParameter(name="query", type="string", description="当前问题或子任务描述"),
            ),
            exposed_to_planner=True,
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        query = str(arguments.get("query", "")).strip()
        return ToolExecutionResult(
            tool_name=self._definition.name,
            ok=True,
            payload={"query": query, "mode": "direct"},
            summary="该步骤标记为直接回答。",
        )


class KnowledgeBaseDocumentLookupTool:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._knowledge_base = knowledge_base
        self._definition = ToolDefinition(
            name="kb.document_lookup",
            description="按标题或正文关键词查找当前知识库中的文档。",
            parameters=(
                ToolParameter(name="keyword", type="string", description="文档关键词"),
                ToolParameter(name="limit", type="integer", description="返回文档数", required=False),
            ),
            requires_knowledge_base=True,
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        keyword = str(arguments.get("keyword", "")).strip().lower()
        if not keyword:
            return ToolExecutionResult(
                tool_name=self._definition.name,
                ok=False,
                payload={"error": "keyword is required"},
                summary="缺少文档查找 keyword。",
            )
        limit = _coerce_int(arguments.get("limit")) or 5
        matches: list[dict[str, str]] = []
        for document in self._knowledge_base.list_documents():
            haystack = f"{document.title}\n{document.content}".lower()
            if keyword not in haystack:
                continue
            matches.append(
                {
                    "document_id": document.document_id,
                    "title": document.title,
                    "knowledge_base_id": document.metadata.get("knowledge_base_id", "default"),
                    "knowledge_base_name": document.metadata.get("knowledge_base_name", "默认知识库"),
                }
            )
            if len(matches) >= limit:
                break
        return ToolExecutionResult(
            tool_name=self._definition.name,
            ok=True,
            payload={"keyword": keyword, "matches": matches, "match_count": len(matches)},
            summary=f"找到 {len(matches)} 篇相关文档。",
        )


class TraceLookupTool:
    def __init__(self, trace_service: TraceService) -> None:
        self._trace_service = trace_service
        self._definition = ToolDefinition(
            name="trace.lookup",
            description="按 request_id 查看一次请求的执行摘要。",
            parameters=(
                ToolParameter(name="request_id", type="string", description="请求 ID"),
            ),
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        request_id = str(arguments.get("request_id", "")).strip()
        if not request_id:
            return ToolExecutionResult(
                tool_name=self._definition.name,
                ok=False,
                payload={"error": "request_id is required"},
                summary="缺少 request_id。",
            )
        detail = self._trace_service.get_trace_detail(request_id)
        if detail is None:
            return ToolExecutionResult(
                tool_name=self._definition.name,
                ok=False,
                payload={"request_id": request_id, "found": False},
                summary=f"没有找到 request_id={request_id} 的 trace。",
            )
        return ToolExecutionResult(
            tool_name=self._definition.name,
            ok=True,
            payload={
                "request_id": detail.request_id,
                "status": detail.status,
                "intent": detail.intent,
                "need_rag": detail.need_rag,
                "step_count": detail.step_count,
                "total_latency_ms": detail.total_latency_ms,
            },
            summary=f"找到 request_id={request_id} 的 trace，共 {detail.step_count} 个步骤。",
        )


def build_default_tool_runtime(
    *,
    retriever_service: RetrieverService,
    knowledge_base: KnowledgeBase,
    trace_service: TraceService,
) -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(RetrievalSearchTool(retriever_service))
    registry.register(AnswerDirectTool())
    registry.register(KnowledgeBaseDocumentLookupTool(knowledge_base))
    registry.register(TraceLookupTool(trace_service))
    return ToolRuntime(registry)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None
