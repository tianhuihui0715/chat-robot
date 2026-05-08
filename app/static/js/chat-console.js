const state = {
  messages: [],
  lastResponse: null,
  sessionId: crypto.randomUUID(),
  plannerPreview: null,
  executionPreview: [],
  sourcePagination: {
    items: [],
    page: 1,
    pageSize: 4,
  },
};

const feed = document.getElementById("chat-feed");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const clearButton = document.getElementById("clear-chat");
const requestId = document.getElementById("request-id");
const intentValue = document.getElementById("intent-value");
const needRagValue = document.getElementById("need-rag-value");
const rewriteQueryValue = document.getElementById("rewrite-query-value");
const executionModeValue = document.getElementById("execution-mode-value");
const rationaleValue = document.getElementById("intent-rationale");
const rawResponse = document.getElementById("raw-response");
const traceLink = document.getElementById("trace-link");
const sourceList = document.getElementById("source-list");
const executionList = document.getElementById("execution-list");
const sessionForm = document.getElementById("session-form");
const sessionIdInput = document.getElementById("session-id");
const newSessionButton = document.getElementById("new-session");
const sessionHint = document.getElementById("session-hint");
const streamMode = document.getElementById("stream-mode");
const rerankMode = document.getElementById("rerank-mode");
const executionStrategy = document.getElementById("execution-strategy");
const chatHint = document.getElementById("chat-hint");

function appendBubble(role, text) {
  const article = document.createElement("article");
  article.className = `bubble bubble--${role}`;
  article.innerHTML = `
    <span class="bubble__role">${role === "user" ? "User" : "Assistant"}</span>
    <p></p>
  `;
  article.querySelector("p").textContent = text;
  feed.appendChild(article);
  feed.scrollTop = feed.scrollHeight;
  return article;
}

function updateBubble(article, text) {
  const paragraph = article.querySelector("p");
  if (paragraph) {
    paragraph.textContent = text;
  }
  feed.scrollTop = feed.scrollHeight;
}

function appendBubbleText(article, text) {
  const paragraph = article.querySelector("p");
  if (paragraph) {
    paragraph.textContent += text;
  }
  feed.scrollTop = feed.scrollHeight;
}

function setSessionHint() {
  sessionHint.textContent = `当前会话：${state.sessionId}`;
  sessionIdInput.value = state.sessionId;
}

function updateChatHint() {
  const endpointText = streamMode.checked
    ? "当前模式：流式返回 /api/v1/chat/stream"
    : "当前模式：普通返回 /api/v1/chat";
  const rerankText = rerankMode.checked ? "启用重排，答案更稳但更慢" : "跳过重排，响应更快";
  const strategyTextMap = {
    off: "关闭规划执行",
    auto: "自动判断规划执行",
    force: "强制走规划执行",
  };
  chatHint.textContent = `${endpointText}；${rerankText}；${strategyTextMap[executionStrategy.value] ?? "自动判断规划执行"}`;
}

function buildChatPayload() {
  return {
    session_id: state.sessionId,
    messages: state.messages,
    use_reranker: rerankMode.checked,
    execution_strategy: executionStrategy.value,
  };
}

function resetConversation() {
  state.messages = [];
  state.lastResponse = null;
  state.plannerPreview = null;
  state.executionPreview = [];
  state.sourcePagination.items = [];
  state.sourcePagination.page = 1;
  feed.innerHTML = `
    <article class="bubble bubble--assistant">
      <span class="bubble__role">Assistant</span>
      <p>可以直接在这里测试问答、流式输出和来源引用效果。</p>
    </article>
  `;
  requestId.textContent = "-";
  intentValue.textContent = "-";
  needRagValue.textContent = "-";
  rewriteQueryValue.textContent = "-";
  executionModeValue.textContent = "-";
  rationaleValue.textContent = "等待请求。";
  rawResponse.textContent = "{}";
  traceLink.href = "/admin/traces";
  renderSources([]);
  renderExecutionSteps([]);
}

function renderSources(sources) {
  state.sourcePagination.items = Array.isArray(sources) ? sources : [];
  state.sourcePagination.page = 1;
  renderSourcePage();
}

function renderSourcePage() {
  const { items, pageSize } = state.sourcePagination;
  if (!items.length) {
    sourceList.innerHTML = '<div class="empty-state">当前还没有命中来源片段。</div>';
    return;
  }

  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  const currentPage = Math.min(state.sourcePagination.page, totalPages);
  const start = (currentPage - 1) * pageSize;
  const visibleItems = items.slice(start, start + pageSize);

  state.sourcePagination.page = currentPage;
  sourceList.innerHTML = "";

  for (const source of visibleItems) {
    const item = document.createElement("article");
    item.className = "mini-list__item";
    const mode = source.metadata?.retrieval_mode;
    const fusion = source.metadata?.fusion_sources;
    const chunkId = source.metadata?.chunk_id;
    const chunkIds = source.metadata?.chunk_ids;
    const citationIndex = source.metadata?.citation_index;
    const mergedCount = Number(source.metadata?.merged_chunk_count ?? 1);
    item.innerHTML = `
      <div class="mini-list__row">
        <strong>${source.title || source.document_id}</strong>
        <div class="mini-list__actions">
          ${citationIndex ? `<span class="badge badge--success">来源 ${citationIndex}</span>` : ""}
          <span class="badge badge--neutral">${Number(source.score ?? 0).toFixed(3)}</span>
        </div>
      </div>
      <div class="console-tags">
        ${mode ? `<span class="console-tag">${mode}</span>` : ""}
        ${fusion ? `<span class="console-tag">融合 ${fusion}</span>` : ""}
        ${mergedCount > 1 ? `<span class="console-tag">合并 ${mergedCount} 段</span>` : ""}
        ${chunkId ? `<span class="console-tag">${chunkId}</span>` : ""}
        ${chunkIds && chunkIds !== chunkId ? `<span class="console-tag">${chunkIds}</span>` : ""}
      </div>
      <p>${source.content || ""}</p>
    `;
    sourceList.appendChild(item);
  }

  const footer = document.createElement("div");
  footer.className = "mini-list__footer";

  const summary = document.createElement("span");
  summary.className = "hint";
  summary.textContent = `共 ${items.length} 条来源，当前第 ${currentPage} / ${totalPages} 页`;

  const pager = document.createElement("div");
  pager.className = "mini-list__pager";

  const previousButton = document.createElement("button");
  previousButton.type = "button";
  previousButton.className = "button button--ghost button--small";
  previousButton.textContent = "上一页";
  previousButton.disabled = currentPage <= 1;
  previousButton.addEventListener("click", () => {
    state.sourcePagination.page = Math.max(1, currentPage - 1);
    renderSourcePage();
  });

  const nextButton = document.createElement("button");
  nextButton.type = "button";
  nextButton.className = "button button--ghost button--small";
  nextButton.textContent = "下一页";
  nextButton.disabled = currentPage >= totalPages;
  nextButton.addEventListener("click", () => {
    state.sourcePagination.page = Math.min(totalPages, currentPage + 1);
    renderSourcePage();
  });

  pager.appendChild(previousButton);
  pager.appendChild(nextButton);
  footer.appendChild(summary);
  footer.appendChild(pager);
  sourceList.appendChild(footer);
}

function renderExecutionSteps(steps, planner) {
  const items = [];
  if (planner) {
    items.push({
      type: "planner",
      status: "completed",
      ...planner,
    });
  }
  if (Array.isArray(steps)) {
    items.push(...steps.filter((step) => !(planner && step.type === "planner")));
  }

  if (!items.length) {
    executionList.innerHTML = '<div class="empty-state">当前还没有执行过程。</div>';
    return;
  }

  executionList.innerHTML = "";
  for (const step of items) {
    const item = document.createElement("article");
    item.className = "mini-list__item";
    const title = formatExecutionTitle(step);
    const detail = formatExecutionDetail(step);
    item.innerHTML = `
      <div class="mini-list__row">
        <strong>${title}</strong>
        <div class="mini-list__actions">
          <span class="badge badge--neutral">${step.status ?? "info"}</span>
        </div>
      </div>
      <p>${detail}</p>
    `;
    executionList.appendChild(item);
  }
}

function formatExecutionTitle(step) {
  if (step.type === "planner") {
    return `Planner: ${step.mode ?? "-"}`;
  }
  if (step.type === "subtask") {
    return `${step.task_id ?? "task"}: ${step.status ?? "planned"}`;
  }
  if (step.type === "retrieval") {
    const queryIndex = step.query_index ? `子查询 ${step.query_index}` : "检索";
    return `${queryIndex}: ${step.status ?? "planned"}`;
  }
  if (step.type === "aggregate") {
    return `聚合: ${step.mode ?? "-"}`;
  }
  if (step.type === "merge") {
    return `合并: ${step.merge_strategy ?? "union"}`;
  }
  return step.type ?? "step";
}

function formatExecutionDetail(step) {
  if (step.type === "planner") {
    const subqueries = Array.isArray(step.subqueries) && step.subqueries.length
      ? `；子查询：${step.subqueries.join(" / ")}`
      : "";
    const source = step.planner_source ? `；来源：${step.planner_source}` : "";
    return `原因：${step.reason ?? "-"}；主查询：${step.primary_query ?? "-"}；策略：${step.merge_strategy ?? "-"}${source}${subqueries}`;
  }
  if (step.type === "subtask") {
    const items = Array.isArray(step.items) && step.items.length
      ? `；候选项：${step.items.slice(0, 6).join(" / ")}`
      : "";
    const count = Number.isFinite(Number(step.count)) ? `；数量：${step.count}` : "";
    const confidence = Number.isFinite(Number(step.confidence)) ? `；置信度：${Number(step.confidence).toFixed(2)}` : "";
    const coverage = step.coverage_hint ? `；覆盖：${step.coverage_hint}` : "";
    return `目标：${step.goal ?? "-"}；查询：${step.query ?? "-"}${count}${confidence}${coverage}${items}`;
  }
  if (step.type === "retrieval") {
    const retrievedText = Number.isFinite(Number(step.retrieved_count))
      ? `；召回 ${step.retrieved_count} 条`
      : "";
    return `查询：${step.query ?? "-"}${retrievedText}`;
  }
  if (step.type === "aggregate") {
    const items = Array.isArray(step.items) && step.items.length
      ? `；结果：${step.items.slice(0, 6).join(" / ")}`
      : "";
    const counts = Number.isFinite(Number(step.left_count)) || Number.isFinite(Number(step.right_count))
      ? `；计数：${step.left_count ?? "-"} vs ${step.right_count ?? "-"}`
      : "";
    const confidence = Number.isFinite(Number(step.confidence)) ? `；置信度：${Number(step.confidence).toFixed(2)}` : "";
    const ranked = Array.isArray(step.ranked_task_ids) && step.ranked_task_ids.length
      ? `；排序：${step.ranked_task_ids.join(" > ")}`
      : "";
    const grouped = step.grouped_items && Object.keys(step.grouped_items).length
      ? `；分组：${Object.entries(step.grouped_items).slice(0, 3).map(([key, value]) => `${key}:${(value || []).slice(0, 3).join("/")}`).join(" | ")}`
      : "";
    return `聚合模式：${step.mode ?? "-"}${counts}${confidence}${ranked}${grouped}${items}`;
  }
  if (step.type === "merge") {
    return `合并策略：${step.merge_strategy ?? "-"}；来源数：${step.retrieved_count ?? 0}`;
  }
  return JSON.stringify(step);
}

function applyFinalResponse(finalResponse) {
  state.lastResponse = finalResponse;
  state.plannerPreview = finalResponse.planner ?? null;
  state.executionPreview = finalResponse.execution_steps ?? [];
  state.messages.push({ role: "assistant", content: finalResponse.answer });
  requestId.textContent = finalResponse.request_id ?? "-";
  intentValue.textContent = finalResponse.intent?.intent ?? "-";
  needRagValue.textContent = String(finalResponse.intent?.need_rag ?? "-");
  rewriteQueryValue.textContent = finalResponse.intent?.rewrite_query ?? "-";
  executionModeValue.textContent = finalResponse.intent?.execution_mode ?? "-";
  rationaleValue.textContent = finalResponse.intent?.rationale ?? "-";
  rawResponse.textContent = JSON.stringify(finalResponse, null, 2);
  traceLink.href = finalResponse.request_id
    ? `/admin/traces?request_id=${finalResponse.request_id}`
    : "/admin/traces";
  renderSources(finalResponse.sources ?? []);
  renderExecutionSteps(finalResponse.execution_steps ?? [], finalResponse.planner ?? null);
}

async function sendMessageNonStream(assistantBubble) {
  const response = await fetch("/api/v1/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(buildChatPayload()),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  const data = await response.json();
  updateBubble(assistantBubble, data.answer ?? "");
  applyFinalResponse(data);
}

async function sendMessageStream(assistantBubble) {
  const response = await fetch("/api/v1/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(buildChatPayload()),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("浏览器未返回可读取的数据流。");
  }

  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalResponse = null;
  let hasStreamedContent = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const eventBlock of events) {
      const payloadLine = eventBlock
        .split("\n")
        .find((line) => line.startsWith("data: "));
      if (!payloadLine) {
        continue;
      }

      const payload = JSON.parse(payloadLine.slice(6));
      if (payload.type === "status") {
        if (!hasStreamedContent) {
          const elapsed = Number(payload.elapsed_seconds ?? 0);
          updateBubble(
            assistantBubble,
            elapsed > 0 ? `正在思考 ${elapsed} 秒...` : "正在思考...",
          );
        }
      } else if (payload.type === "planner") {
        state.plannerPreview = payload.data ?? null;
        renderExecutionSteps(state.executionPreview, state.plannerPreview);
      } else if (payload.type === "execution") {
        state.executionPreview = [
          ...state.executionPreview,
          {
            type: payload.stage?.includes("aggregate")
              ? "aggregate"
              : payload.stage?.includes("subtask")
                ? "subtask"
                : payload.stage?.includes("merge")
                  ? "merge"
                  : "retrieval",
            status: payload.stage?.includes("completed") ? "completed" : "planned",
            query_index: payload.query_index,
            task_id: payload.task_id,
            goal: payload.goal,
            query: payload.query,
            items: payload.items,
            count: payload.count ?? payload.item_count,
            confidence: payload.confidence,
            coverage_hint: payload.coverage_hint,
            mode: payload.mode,
            merge_strategy: payload.merge_strategy,
            retrieved_count: payload.retrieved_count,
            left_count: payload.left_count,
            right_count: payload.right_count,
          },
        ];
        renderExecutionSteps(state.executionPreview, state.plannerPreview);
      } else if (payload.type === "delta") {
        if (!hasStreamedContent) {
          updateBubble(assistantBubble, "");
          hasStreamedContent = true;
        }
        appendBubbleText(assistantBubble, payload.delta ?? "");
      } else if (payload.type === "response") {
        finalResponse = payload.data;
      } else if (payload.type === "error") {
        throw new Error(payload.message || "流式请求失败。");
      }
    }
  }

  if (!finalResponse) {
    throw new Error("未收到最终响应。");
  }

  updateBubble(assistantBubble, finalResponse.answer ?? "");
  applyFinalResponse(finalResponse);
}

async function sendMessage(event) {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) {
    return;
  }

  const outgoing = { role: "user", content: text };
  state.messages.push(outgoing);
  appendBubble("user", text);
  const assistantBubble = appendBubble("assistant", "正在思考...");
  input.value = "";
  sendButton.disabled = true;

  try {
    if (streamMode.checked) {
      await sendMessageStream(assistantBubble);
    } else {
      await sendMessageNonStream(assistantBubble);
    }
  } catch (error) {
    updateBubble(assistantBubble, `请求失败：${error.message}`);
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", sendMessage);
clearButton.addEventListener("click", resetConversation);
sessionForm.addEventListener("submit", (event) => event.preventDefault());
sessionIdInput.addEventListener("change", () => {
  const nextId = sessionIdInput.value.trim();
  if (nextId) {
    state.sessionId = nextId;
  }
  setSessionHint();
});
newSessionButton.addEventListener("click", () => {
  state.sessionId = crypto.randomUUID();
  setSessionHint();
  resetConversation();
});
streamMode.addEventListener("change", updateChatHint);
rerankMode.addEventListener("change", updateChatHint);
executionStrategy.addEventListener("change", updateChatHint);

setSessionHint();
updateChatHint();
resetConversation();
