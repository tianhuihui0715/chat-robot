const state = {
  messages: [],
  lastResponse: null,
  sessionId: crypto.randomUUID(),
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
const rationaleValue = document.getElementById("intent-rationale");
const rawResponse = document.getElementById("raw-response");
const traceLink = document.getElementById("trace-link");
const sourceList = document.getElementById("source-list");
const sessionForm = document.getElementById("session-form");
const sessionIdInput = document.getElementById("session-id");
const newSessionButton = document.getElementById("new-session");
const sessionHint = document.getElementById("session-hint");
const streamMode = document.getElementById("stream-mode");
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
  chatHint.textContent = streamMode.checked
    ? "当前模式：流式返回 /api/v1/chat/stream"
    : "当前模式：普通返回 /api/v1/chat";
}

function resetConversation() {
  state.messages = [];
  state.lastResponse = null;
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
  rationaleValue.textContent = "等待请求。";
  rawResponse.textContent = "{}";
  traceLink.href = "/admin/traces";
  renderSources([]);
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
    item.innerHTML = `
      <div class="mini-list__row">
        <strong>${source.title || source.document_id}</strong>
        <span class="badge badge--neutral">${Number(source.score ?? 0).toFixed(3)}</span>
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

function applyFinalResponse(finalResponse) {
  state.lastResponse = finalResponse;
  state.messages.push({ role: "assistant", content: finalResponse.answer });
  requestId.textContent = finalResponse.request_id ?? "-";
  intentValue.textContent = finalResponse.intent?.intent ?? "-";
  needRagValue.textContent = String(finalResponse.intent?.need_rag ?? "-");
  rewriteQueryValue.textContent = finalResponse.intent?.rewrite_query ?? "-";
  rationaleValue.textContent = finalResponse.intent?.rationale ?? "-";
  rawResponse.textContent = JSON.stringify(finalResponse, null, 2);
  traceLink.href = finalResponse.request_id
    ? `/admin/traces?request_id=${finalResponse.request_id}`
    : "/admin/traces";
  renderSources(finalResponse.sources ?? []);
}

async function sendMessageNonStream(assistantBubble) {
  const response = await fetch("/api/v1/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: state.sessionId,
      messages: state.messages,
    }),
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
    body: JSON.stringify({
      session_id: state.sessionId,
      messages: state.messages,
    }),
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

setSessionHint();
updateChatHint();
resetConversation();
