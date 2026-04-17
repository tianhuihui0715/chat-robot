const state = {
  messages: [],
  lastResponse: null,
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

function resetConversation() {
  state.messages = [];
  state.lastResponse = null;
  feed.innerHTML = `
    <article class="bubble bubble--assistant">
      <span class="bubble__role">Assistant</span>
      <p>你好，我已经准备好了。你可以直接在这里测试本地聊天链路。</p>
    </article>
  `;
  requestId.textContent = "-";
  intentValue.textContent = "-";
  needRagValue.textContent = "-";
  rewriteQueryValue.textContent = "-";
  rationaleValue.textContent = "等待请求。";
  rawResponse.textContent = "{}";
  traceLink.href = "/traces";
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
  } catch (error) {
    apiStatus.textContent = "error";
    runtimeMode.textContent = "-";
    traceCount.textContent = "-";
    knowledgeCount.textContent = "-";
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

resetConversation();
refreshHealth();
setInterval(refreshHealth, 15000);
