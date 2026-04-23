const traceList = document.getElementById("trace-list");
const traceSummary = document.getElementById("trace-summary");
const traceTimeline = document.getElementById("trace-timeline");
const traceOutput = document.getElementById("trace-output");
const traceRaw = document.getElementById("trace-raw");
const selectedTrace = document.getElementById("selected-trace");
const selectedStatus = document.getElementById("selected-status");
const selectedLatency = document.getElementById("selected-latency");
const selectedSteps = document.getElementById("selected-steps");
const refreshButton = document.getElementById("refresh-admin");
const tracePrevPage = document.getElementById("trace-prev-page");
const traceNextPage = document.getElementById("trace-next-page");
const tracePageSummary = document.getElementById("trace-page-summary");
const traceStatusFilter = document.getElementById("trace-status-filter");
const traceSessionFilter = document.getElementById("trace-session-filter");
const traceViewAll = document.getElementById("trace-view-all");
const ragConfigForm = document.getElementById("rag-config-form");
const ragConfigHint = document.getElementById("rag-config-hint");
const ragCompareForm = document.getElementById("rag-compare-form");
const ragCompareHint = document.getElementById("rag-compare-hint");
const compareResults = document.getElementById("compare-results");

const pageState = {
  traces: [],
  selectedId: new URLSearchParams(window.location.search).get("request_id"),
  page: 1,
  pageSize: 10,
  total: 0,
  viewAll: false,
  ragConfig: null,
};

function formatDateTime(value) {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString("zh-CN", { hour12: false });
}

function statusBadge(status) {
  if (status === "completed") {
    return "badge badge--success";
  }
  if (status === "error" || status === "failed") {
    return "badge badge--error";
  }
  return "badge badge--neutral";
}

function switchTab(tabName) {
  document.querySelectorAll(".tabbar__button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.tab === tabName);
  });
  document.querySelectorAll(".admin-tab").forEach((panel) => {
    panel.classList.toggle("is-active", panel.id === `tab-${tabName}`);
  });
}

function renderTraceList() {
  if (!pageState.traces.length) {
    traceList.innerHTML = '<div class="empty-state">当前没有可展示的请求日志。</div>';
    tracePageSummary.textContent = "第 0 / 0 页";
    return;
  }

  traceList.innerHTML = "";
  for (const trace of pageState.traces) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `trace-item ${trace.request_id === pageState.selectedId ? "is-active" : ""}`;

    const titleRow = document.createElement("div");
    titleRow.className = "trace-item__title-row";

    const title = document.createElement("strong");
    title.textContent = trace.intent ?? "pending";

    const badge = document.createElement("span");
    badge.className = statusBadge(trace.status);
    badge.textContent = trace.status ?? "pending";

    const preview = document.createElement("p");
    preview.className = "trace-item__preview";
    preview.textContent = trace.user_input ?? "-";

    const meta = document.createElement("p");
    meta.className = "trace-item__meta";
    meta.textContent = `${formatDateTime(trace.created_at)} · ${trace.total_latency_ms ?? "-"} ms`;

    titleRow.append(title, badge);
    button.append(titleRow, preview, meta);
    button.addEventListener("click", async () => {
      pageState.selectedId = trace.request_id;
      pageState.viewAll = false;
      renderTraceList();
      await loadTraceDetail(trace.request_id);
    });
    traceList.appendChild(button);
  }

  const totalPages = Math.max(1, Math.ceil(pageState.total / pageState.pageSize));
  tracePageSummary.textContent = `第 ${pageState.page} / ${totalPages} 页，共 ${pageState.total} 条`;
  tracePrevPage.disabled = pageState.page <= 1;
  traceNextPage.disabled = pageState.page >= totalPages;
}

function renderTraceDetail(wrapper) {
  const trace = wrapper.trace;

  selectedTrace.textContent = trace.request_id ?? "-";
  selectedStatus.textContent = trace.status ?? "-";
  selectedLatency.textContent = trace.total_latency_ms ? `${trace.total_latency_ms} ms` : "-";
  selectedSteps.textContent = String(trace.steps?.length ?? 0);
  traceViewAll.hidden = !wrapper.has_more_steps && !wrapper.output_truncated;

  traceSummary.innerHTML = "";
  const summaryCard = document.createElement("div");
  summaryCard.className = "trace-summary";
  [
    ["用户输入", trace.user_input ?? "-"],
    ["意图", trace.intent ?? "-"],
    ["改写查询", trace.intent_record?.rewrite_query ?? "-"],
    ["创建时间", formatDateTime(trace.created_at)],
  ].forEach(([label, value]) => {
    const line = document.createElement("p");
    const strong = document.createElement("strong");
    strong.textContent = `${label}：`;
    line.append(strong, document.createTextNode(value));
    summaryCard.appendChild(line);
  });
  traceSummary.appendChild(summaryCard);

  traceTimeline.innerHTML = "";
  if (!trace.steps?.length) {
    traceTimeline.innerHTML = '<div class="empty-state">当前请求还没有步骤明细。</div>';
  } else {
    for (const step of trace.steps) {
      const article = document.createElement("article");
      article.className = "timeline-item";

      const titleRow = document.createElement("div");
      titleRow.className = "timeline-item__title-row";

      const title = document.createElement("strong");
      title.textContent = step.step_type ?? "step";

      const badge = document.createElement("span");
      badge.className = statusBadge(step.status);
      badge.textContent = step.status ?? "pending";

      const meta = document.createElement("p");
      meta.className = "timeline-item__meta";
      meta.textContent = `顺序 ${step.step_order} · ${step.latency_ms ?? "-"} ms`;

      const code = document.createElement("pre");
      code.className = "code-block";
      code.textContent = JSON.stringify(step, null, 2);

      titleRow.append(title, badge);
      article.append(titleRow, meta, code);
      traceTimeline.appendChild(article);
    }
  }

  traceOutput.textContent = trace.final_output || trace.generation_record?.llm_output || "-";
  traceRaw.textContent = JSON.stringify(trace, null, 2);
}

function fillRagConfigForm(config) {
  pageState.ragConfig = config;
  document.getElementById("rag-top-k").value = config.top_k;
  document.getElementById("rag-score-threshold").value = config.score_threshold;
  document.getElementById("rag-candidate-multiplier").value = config.candidate_multiplier;
  document.getElementById("rag-chunk-size").value = config.chunk_size;
  document.getElementById("rag-chunk-overlap").value = config.chunk_overlap;
  document.getElementById("rag-reranker-enabled").checked = Boolean(config.reranker_enabled);

  document.getElementById("compare-a-top-k").value = config.top_k;
  document.getElementById("compare-a-threshold").value = config.score_threshold;
  document.getElementById("compare-a-multiplier").value = config.candidate_multiplier;
  document.getElementById("compare-a-rerank").checked = Boolean(config.reranker_enabled);

  document.getElementById("compare-b-top-k").value = config.top_k;
  document.getElementById("compare-b-threshold").value = config.score_threshold;
  document.getElementById("compare-b-multiplier").value = config.candidate_multiplier;
  document.getElementById("compare-b-rerank").checked = false;
}

function renderCompareResults(data) {
  compareResults.innerHTML = "";
  if (!data.results?.length) {
    compareResults.innerHTML = '<div class="empty-state">当前没有可展示的对比结果。</div>';
    return;
  }

  for (const result of data.results) {
    const card = document.createElement("div");
    card.className = "detail-card";

    const title = document.createElement("h3");
    title.textContent = result.name;

    const answer = document.createElement("pre");
    answer.className = "code-block";
    answer.textContent = result.answer || "[未生成回答]";

    const sources = document.createElement("div");
    sources.className = "mini-list";
    if (!result.sources?.length) {
      sources.innerHTML = '<div class="empty-state">没有命中来源。</div>';
    } else {
      for (const source of result.sources) {
        const item = document.createElement("article");
        item.className = "mini-list__item";
        item.innerHTML = `
          <div class="mini-list__row">
            <strong>${source.title || source.document_id}</strong>
            <span class="badge badge--neutral">${Number(source.score ?? 0).toFixed(3)}</span>
          </div>
          <p>${source.content || ""}</p>
        `;
        sources.appendChild(item);
      }
    }

    card.append(title, answer, sources);
    compareResults.appendChild(card);
  }
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function loadTraceList() {
  const query = new URLSearchParams({
    page: String(pageState.page),
    page_size: String(pageState.pageSize),
  });
  const sessionId = traceSessionFilter.value.trim();
  const status = traceStatusFilter.value.trim();
  if (sessionId) {
    query.set("session_id", sessionId);
  }
  if (status) {
    query.set("status", status);
  }
  const data = await fetchJson(`/api/v1/traces?${query.toString()}`);
  pageState.traces = data.items ?? [];
  pageState.total = data.total ?? 0;
  if (!pageState.selectedId && pageState.traces.length) {
    pageState.selectedId = pageState.traces[0].request_id;
  }
  renderTraceList();
  if (pageState.selectedId) {
    await loadTraceDetail(pageState.selectedId);
  }
}

async function loadTraceDetail(requestId) {
  const query = new URLSearchParams({
    step_limit: "3",
    view_all: String(pageState.viewAll),
  });
  const data = await fetchJson(`/api/v1/traces/${requestId}?${query.toString()}`);
  renderTraceDetail(data);
}

async function loadRagConfig() {
  const data = await fetchJson("/api/v1/admin/rag/config");
  fillRagConfigForm(data);
}

async function refreshDashboard() {
  try {
    await Promise.all([loadTraceList(), loadRagConfig()]);
  } catch (error) {
    traceList.innerHTML = `<div class="empty-state">加载后台失败：${error.message}</div>`;
    compareResults.innerHTML = "";
  }
}

document.querySelectorAll(".tabbar__button").forEach((button) => {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
});

refreshButton.addEventListener("click", refreshDashboard);
tracePrevPage.addEventListener("click", async () => {
  pageState.page = Math.max(1, pageState.page - 1);
  await loadTraceList();
});
traceNextPage.addEventListener("click", async () => {
  const totalPages = Math.max(1, Math.ceil(pageState.total / pageState.pageSize));
  pageState.page = Math.min(totalPages, pageState.page + 1);
  await loadTraceList();
});
traceStatusFilter.addEventListener("change", async () => {
  pageState.page = 1;
  await loadTraceList();
});
traceSessionFilter.addEventListener("change", async () => {
  pageState.page = 1;
  await loadTraceList();
});
traceViewAll.addEventListener("click", async () => {
  if (!pageState.selectedId) {
    return;
  }
  pageState.viewAll = true;
  await loadTraceDetail(pageState.selectedId);
  traceViewAll.hidden = true;
});

ragConfigForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const payload = {
      top_k: Number(document.getElementById("rag-top-k").value),
      score_threshold: Number(document.getElementById("rag-score-threshold").value),
      candidate_multiplier: Number(document.getElementById("rag-candidate-multiplier").value),
      chunk_size: Number(document.getElementById("rag-chunk-size").value),
      chunk_overlap: Number(document.getElementById("rag-chunk-overlap").value),
      reranker_enabled: document.getElementById("rag-reranker-enabled").checked,
    };
    const data = await fetchJson("/api/v1/admin/rag/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    fillRagConfigForm(data);
    ragConfigHint.textContent = "RAG 运行时参数已保存并立即生效。Chunk 参数对后续新导入文档生效。";
  } catch (error) {
    ragConfigHint.textContent = `保存失败：${error.message}`;
  }
});

ragCompareForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const payload = {
      query: document.getElementById("compare-query").value.trim(),
      generate_answer: true,
      variants: [
        {
          name: document.getElementById("compare-a-name").value.trim() || "方案 A",
          top_k: Number(document.getElementById("compare-a-top-k").value),
          score_threshold: Number(document.getElementById("compare-a-threshold").value),
          candidate_multiplier: Number(document.getElementById("compare-a-multiplier").value),
          reranker_enabled: document.getElementById("compare-a-rerank").checked,
        },
        {
          name: document.getElementById("compare-b-name").value.trim() || "方案 B",
          top_k: Number(document.getElementById("compare-b-top-k").value),
          score_threshold: Number(document.getElementById("compare-b-threshold").value),
          candidate_multiplier: Number(document.getElementById("compare-b-multiplier").value),
          reranker_enabled: document.getElementById("compare-b-rerank").checked,
        },
      ],
    };
    const data = await fetchJson("/api/v1/admin/rag/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderCompareResults(data);
    ragCompareHint.textContent = "对比完成，可以直接比较不同流程下的来源与回答差异。";
  } catch (error) {
    ragCompareHint.textContent = `对比失败：${error.message}`;
    compareResults.innerHTML = "";
  }
});

refreshDashboard();
