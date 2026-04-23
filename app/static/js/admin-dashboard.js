(() => {
  const { fetchJson, formatDateTime, statusBadgeClass } = window.AdminCommon;

  const refreshButton = document.getElementById("dashboard-refresh");
  const overviewStatus = document.getElementById("overview-status");
  const overviewQueue = document.getElementById("overview-queue");
  const overviewDocuments = document.getElementById("overview-documents");
  const overviewTopK = document.getElementById("overview-top-k");
  const runtimeDetails = document.getElementById("runtime-details");
  const ragDetails = document.getElementById("rag-details");
  const documentPreview = document.getElementById("document-preview");
  const recentTraces = document.getElementById("recent-traces");

  function setDefinitionList(target, entries) {
    target.innerHTML = "";
    for (const [label, value] of entries) {
      const wrapper = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = label;
      dd.textContent = value;
      wrapper.append(dt, dd);
      target.appendChild(wrapper);
    }
  }

  function renderDocuments(documents) {
    documentPreview.innerHTML = "";
    if (!documents.length) {
      documentPreview.innerHTML = '<div class="empty-state">当前还没有已入库文档。</div>';
      return;
    }

    for (const item of documents.slice(0, 5)) {
      const article = document.createElement("article");
      article.className = "mini-list__item";
      article.innerHTML = `
        <div class="mini-list__row">
          <strong>${item.title || item.document_id}</strong>
          <span class="badge badge--neutral">${item.document_id}</span>
        </div>
      `;
      documentPreview.appendChild(article);
    }
  }

  function renderRecentTraces(traces) {
    recentTraces.innerHTML = "";
    if (!traces.length) {
      recentTraces.innerHTML = '<div class="empty-state">当前还没有流程日志。</div>';
      return;
    }

    for (const trace of traces) {
      const article = document.createElement("article");
      article.className = "mini-list__item";

      const titleRow = document.createElement("div");
      titleRow.className = "mini-list__row";

      const title = document.createElement("strong");
      title.textContent = trace.intent || "pending";

      const badge = document.createElement("span");
      badge.className = statusBadgeClass(trace.status);
      badge.textContent = trace.status || "pending";

      const preview = document.createElement("p");
      preview.textContent = trace.user_input || "-";

      const meta = document.createElement("p");
      meta.className = "hint";
      meta.textContent = `${formatDateTime(trace.created_at)} · ${trace.total_latency_ms ?? "-"} ms`;

      const link = document.createElement("a");
      link.className = "text-link";
      link.href = `/admin/traces?request_id=${encodeURIComponent(trace.request_id)}`;
      link.textContent = "查看详情";

      titleRow.append(title, badge);
      article.append(titleRow, preview, meta, link);
      recentTraces.appendChild(article);
    }
  }

  async function loadDashboard() {
    try {
      const [health, ragConfig, traces, documents] = await Promise.all([
        fetchJson("/api/v1/health"),
        fetchJson("/api/v1/admin/rag/config"),
        fetchJson("/api/v1/traces?page=1&page_size=5"),
        fetchJson("/api/v1/knowledge/documents"),
      ]);

      overviewStatus.textContent = health.status || "-";
      overviewQueue.textContent = String(health.queued_requests ?? 0);
      overviewDocuments.textContent = String(documents.length);
      overviewTopK.textContent = String(ragConfig.top_k);

      setDefinitionList(runtimeDetails, [
        ["服务状态", health.status || "-"],
        ["运行模式", health.runtime_mode || "-"],
        ["排队请求", String(health.queued_requests ?? 0)],
        ["最近刷新", formatDateTime(new Date().toISOString())],
      ]);

      setDefinitionList(ragDetails, [
        ["Top K", String(ragConfig.top_k)],
        ["Threshold", String(ragConfig.score_threshold)],
        ["Candidate Multiplier", String(ragConfig.candidate_multiplier)],
        ["重排序", ragConfig.reranker_enabled ? "已启用" : "未启用"],
        ["Chunk Size", String(ragConfig.chunk_size)],
        ["Chunk Overlap", String(ragConfig.chunk_overlap)],
      ]);

      renderDocuments(documents);
      renderRecentTraces(traces.items || []);
    } catch (error) {
      recentTraces.innerHTML = `<div class="empty-state">概览加载失败：${error.message}</div>`;
    }
  }

  refreshButton.addEventListener("click", loadDashboard);
  loadDashboard();
})();
