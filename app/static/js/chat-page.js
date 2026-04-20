const state = {
  messages: [],
  lastResponse: null,
  sessionId: crypto.randomUUID(),
};

const feed = document.getElementById("chat-feed");
const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const sendButton = document.getElementById("send-button");
const clearButton = document.getElementById("clear-chat");
const apiStatus = document.getElementById("api-status");
const runtimeMode = document.getElementById("runtime-mode");
const traceCount = document.getElementById("trace-count");
const knowledgeCount = document.getElementById("knowledge-count");
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
const knowledgeForm = document.getElementById("knowledge-form");
const knowledgeFiles = document.getElementById("knowledge-files");
const knowledgeTitle = document.getElementById("knowledge-title");
const knowledgeContent = document.getElementById("knowledge-content");
const knowledgeSubmit = document.getElementById("knowledge-submit");
const knowledgeHint = document.getElementById("knowledge-hint");
const documentList = document.getElementById("document-list");
const refreshDocumentsButton = document.getElementById("refresh-documents");
const postgresStatus = document.getElementById("postgres-status");
const qdrantStatus = document.getElementById("qdrant-status");
const minioStatus = document.getElementById("minio-status");

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
}

function setSessionHint() {
  sessionHint.textContent = `当前会话：${state.sessionId}`;
  sessionIdInput.value = state.sessionId;
}

function resetConversation() {
  state.messages = [];
  state.lastResponse = null;
  feed.innerHTML = `
    <article class="bubble bubble--assistant">
      <span class="bubble__role">Assistant</span>
      <p>你好，我已经准备好了。你可以先导入文档，再直接在这里测试检索和问答链路。</p>
    </article>
  `;
  requestId.textContent = "-";
  intentValue.textContent = "-";
  needRagValue.textContent = "-";
  rewriteQueryValue.textContent = "-";
  rationaleValue.textContent = "等待请求。";
  rawResponse.textContent = "{}";
  traceLink.href = "/traces";
  renderSources([]);
}

function setInfraStatus(element, isConnected) {
  element.textContent = isConnected ? "connected" : "disconnected";
  element.className = isConnected ? "status-inline status-inline--ok" : "status-inline status-inline--error";
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/v1/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    apiStatus.textContent = data.status;
    runtimeMode.textContent = data.runtime_mode;
    traceCount.textContent = String(data.trace_requests);
    knowledgeCount.textContent = String(data.knowledge_documents);
    setInfraStatus(postgresStatus, data.postgres_connected);
    setInfraStatus(qdrantStatus, data.qdrant_connected);
    setInfraStatus(minioStatus, data.minio_connected);
  } catch (error) {
    apiStatus.textContent = "error";
    runtimeMode.textContent = "-";
    traceCount.textContent = "-";
    knowledgeCount.textContent = "-";
    setInfraStatus(postgresStatus, false);
    setInfraStatus(qdrantStatus, false);
    setInfraStatus(minioStatus, false);
  }
}

function renderSources(sources) {
  if (!sources?.length) {
    sourceList.innerHTML = '<div class="empty-state">当前还没有来源片段。</div>';
    return;
  }

  sourceList.innerHTML = "";
  for (const source of sources) {
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
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error(`读取文件失败：${file.name}`));
    reader.readAsText(file, "utf-8");
  });
}

async function refreshDocuments() {
  try {
    const response = await fetch("/api/v1/knowledge/documents");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const documents = await response.json();
    if (!documents.length) {
      documentList.innerHTML = '<div class="empty-state">还没有已入库文档。</div>';
      return;
    }

    documentList.innerHTML = "";
    for (const document of documents) {
      const item = document.createElement("article");
      item.className = "mini-list__item";
      item.innerHTML = `
        <div class="mini-list__row">
          <strong>${document.title}</strong>
          <span class="hint">${document.document_id}</span>
        </div>
      `;
      documentList.appendChild(item);
    }
  } catch (error) {
    documentList.innerHTML = `<div class="empty-state">加载文档失败：${error.message}</div>`;
  }
}

async function ingestKnowledge(event) {
  event.preventDefault();
  knowledgeSubmit.disabled = true;

  try {
    const documents = [];
    const files = Array.from(knowledgeFiles.files ?? []);

    for (const file of files) {
      const content = (await readFileAsText(file)).trim();
      if (!content) {
        continue;
      }
      documents.push({
        title: file.name,
        content,
        metadata: {
          source: "upload",
          file_name: file.name,
        },
      });
    }

    const manualTitle = knowledgeTitle.value.trim();
    const manualContent = knowledgeContent.value.trim();
    if (manualContent) {
      documents.push({
        title: manualTitle || `manual-${new Date().toISOString()}`,
        content: manualContent,
        metadata: {
          source: "manual",
        },
      });
    }

    if (!documents.length) {
      throw new Error("请先选择文件或输入知识内容。");
    }

    const response = await fetch("/api/v1/knowledge/ingest", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ documents }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    knowledgeHint.textContent = `导入成功：${data.ingested_count} 篇，当前共 ${data.total_documents} 篇文档。`;
    knowledgeFiles.value = "";
    knowledgeTitle.value = "";
    knowledgeContent.value = "";
    await refreshDocuments();
    await refreshHealth();
  } catch (error) {
    knowledgeHint.textContent = `导入失败：${error.message}`;
  } finally {
    knowledgeSubmit.disabled = false;
  }
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
  input.value = "";
  sendButton.disabled = true;

  try {
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
    state.lastResponse = data;
    state.messages.push({ role: "assistant", content: data.answer });
    appendBubble("assistant", data.answer);

    requestId.textContent = data.request_id ?? "-";
    intentValue.textContent = data.intent?.intent ?? "-";
    needRagValue.textContent = String(data.intent?.need_rag ?? "-");
    rewriteQueryValue.textContent = data.intent?.rewrite_query ?? "-";
    rationaleValue.textContent = data.intent?.rationale ?? "-";
    rawResponse.textContent = JSON.stringify(data, null, 2);
    traceLink.href = data.request_id ? `/traces?request_id=${data.request_id}` : "/traces";
    renderSources(data.sources ?? []);
    await refreshHealth();
  } catch (error) {
    appendBubble("assistant", `请求失败：${error.message}`);
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
knowledgeForm.addEventListener("submit", ingestKnowledge);
refreshDocumentsButton.addEventListener("click", refreshDocuments);

setSessionHint();
resetConversation();
refreshHealth();
refreshDocuments();
setInterval(refreshHealth, 15000);
