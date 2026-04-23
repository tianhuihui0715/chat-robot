(() => {
  const { fetchJson, formatDateTime, statusBadgeClass } = window.AdminCommon;

  const traceList = document.getElementById("trace-list");
  const traceSummary = document.getElementById("trace-summary");
  const traceTimeline = document.getElementById("trace-timeline");
  const traceOutput = document.getElementById("trace-output");
  const traceRaw = document.getElementById("trace-raw");
  const selectedTrace = document.getElementById("selected-trace");
  const selectedStatus = document.getElementById("selected-status");
  const selectedLatency = document.getElementById("selected-latency");
  const selectedSteps = document.getElementById("selected-steps");
  const refreshButton = document.getElementById("trace-refresh");
  const tracePrevPage = document.getElementById("trace-prev-page");
  const traceNextPage = document.getElementById("trace-next-page");
  const tracePageSummary = document.getElementById("trace-page-summary");
  const traceStatusFilter = document.getElementById("trace-status-filter");
  const traceSessionFilter = document.getElementById("trace-session-filter");
  const traceViewAll = document.getElementById("trace-view-all");

  const pageState = {
    traces: [],
    selectedId: new URLSearchParams(window.location.search).get("request_id"),
    page: 1,
    pageSize: 10,
    total: 0,
    viewAll: false,
  };

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
      title.textContent = trace.intent || "pending";

      const badge = document.createElement("span");
      badge.className = statusBadgeClass(trace.status);
      badge.textContent = trace.status || "pending";

      const preview = document.createElement("p");
      preview.className = "trace-item__preview";
      preview.textContent = trace.user_input || "-";

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

    selectedTrace.textContent = trace.request_id || "-";
    selectedStatus.textContent = trace.status || "-";
    selectedLatency.textContent = trace.total_latency_ms ? `${trace.total_latency_ms} ms` : "-";
    selectedSteps.textContent = String(trace.steps?.length ?? 0);
    traceViewAll.hidden = !wrapper.has_more_steps && !wrapper.output_truncated;

    traceSummary.innerHTML = "";
    const summaryCard = document.createElement("div");
    summaryCard.className = "trace-summary";
    [
      ["用户输入", trace.user_input || "-"],
      ["意图", trace.intent || "-"],
      ["改写查询", trace.intent_record?.rewrite_query || "-"],
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
        title.textContent = step.step_type || "step";

        const badge = document.createElement("span");
        badge.className = statusBadgeClass(step.status);
        badge.textContent = step.status || "pending";

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
    pageState.traces = data.items || [];
    pageState.total = data.total || 0;

    if (!pageState.selectedId && pageState.traces.length) {
      pageState.selectedId = pageState.traces[0].request_id;
    }

    renderTraceList();
    if (pageState.selectedId) {
      await loadTraceDetail(pageState.selectedId);
    } else {
      traceSummary.innerHTML = "<p>当前没有可展示的历史请求。</p>";
      traceTimeline.innerHTML = "";
      traceOutput.textContent = "-";
      traceRaw.textContent = "-";
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

  async function refreshTraces() {
    try {
      await loadTraceList();
    } catch (error) {
        traceList.innerHTML = `<div class="empty-state">加载日志失败：${error.message}</div>`;
    }
  }

  refreshButton.addEventListener("click", refreshTraces);
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

  refreshTraces();
})();
