const traceList = document.getElementById("trace-list");
const traceSummary = document.getElementById("trace-summary");
const traceTimeline = document.getElementById("trace-timeline");
const traceOutput = document.getElementById("trace-output");
const traceRaw = document.getElementById("trace-raw");
const selectedTrace = document.getElementById("selected-trace");
const selectedStatus = document.getElementById("selected-status");
const selectedLatency = document.getElementById("selected-latency");
const selectedSteps = document.getElementById("selected-steps");
const refreshButton = document.getElementById("refresh-traces");

const pageState = {
  traces: [],
  selectedId: new URLSearchParams(window.location.search).get("request_id"),
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
  if (status === "error") {
    return "badge badge--error";
  }
  return "badge badge--neutral";
}

function renderTraceList() {
  if (!pageState.traces.length) {
    traceList.innerHTML = '<div class="empty-state">还没有可展示的请求轨迹。</div>';
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
    meta.textContent = formatDateTime(trace.created_at);

    titleRow.append(title, badge);
    button.append(titleRow, preview, meta);
    button.addEventListener("click", () => {
      pageState.selectedId = trace.request_id;
      renderTraceList();
      loadTraceDetail(trace.request_id);
    });
    traceList.appendChild(button);
  }
}

function renderTraceDetail(trace) {
  selectedTrace.textContent = trace.request_id ?? "-";
  selectedStatus.textContent = trace.status ?? "-";
  selectedLatency.textContent = trace.total_latency_ms ? `${trace.total_latency_ms} ms` : "-";
  selectedSteps.textContent = String(trace.steps?.length ?? 0);

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

  if (!trace.steps?.length) {
    traceTimeline.innerHTML = '<div class="empty-state">当前请求还没有步骤明细。</div>';
  } else {
    traceTimeline.innerHTML = "";
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

  traceOutput.textContent =
    trace.final_output ||
    trace.generation_record?.llm_output ||
    "-";
  traceRaw.textContent = JSON.stringify(trace, null, 2);
}

async function loadTraceList() {
  const response = await fetch("/api/v1/traces");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const data = await response.json();
  pageState.traces = data.items ?? data;
  if (!pageState.selectedId && pageState.traces.length) {
    pageState.selectedId = pageState.traces[0].request_id;
  }
  renderTraceList();
  if (pageState.selectedId) {
    await loadTraceDetail(pageState.selectedId);
  }
}

async function loadTraceDetail(requestId) {
  const response = await fetch(`/api/v1/traces/${requestId}`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const data = await response.json();
  renderTraceDetail(data);
}

async function refreshDashboard() {
  try {
    await loadTraceList();
  } catch (error) {
    traceList.innerHTML = `<div class="empty-state">加载日志失败：${error.message}</div>`;
    traceTimeline.innerHTML = "";
    traceOutput.textContent = "-";
    traceRaw.textContent = "{}";
  }
}

refreshButton.addEventListener("click", refreshDashboard);
refreshDashboard();
