(() => {
  const requestInput = document.getElementById("performance-request-id");
  const loadButton = document.getElementById("performance-load");
  const refreshButton = document.getElementById("performance-refresh");
  const meta = document.getElementById("performance-meta");
  const tableBody = document.getElementById("performance-table-body");

  const rows = [
    ["意图判断", "intent_decision"],
    ["embedding", "embedding"],
    ["Qdrant 检索", "qdrant_search"],
    ["重排", "rerank"],
    ["首 token 生成", "first_token_generation"],
    ["总耗时", "total"],
  ];

  function renderRows(values) {
    tableBody.innerHTML = "";
    rows.forEach(([label, key]) => {
      const row = document.createElement("tr");
      const nameCell = document.createElement("td");
      const valueCell = document.createElement("td");
      nameCell.textContent = label;
      valueCell.textContent = values[key] == null ? "-" : String(values[key]);
      row.append(nameCell, valueCell);
      tableBody.appendChild(row);
    });
  }

  async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${response.status}`);
    }
    return response.json();
  }

  async function getLatestRequestId() {
    const data = await fetchJson("/api/v1/traces?page=1&page_size=1&status=completed");
    return data.items?.[0]?.request_id || "";
  }

  function extractValues(detail) {
    const trace = detail.trace || detail;
    const byType = {};
    (trace.steps || []).forEach((step) => {
      byType[step.step_type] = step.latency_ms;
    });
    return {
      intent_decision: byType.intent_decision,
      embedding: byType.embedding,
      qdrant_search: byType.qdrant_search,
      rerank: byType.rerank,
      first_token_generation: byType.first_token_generation,
      total: trace.total_latency_ms,
    };
  }

  async function loadPerformance(requestId = "") {
    meta.textContent = "加载中...";
    renderRows({});
    const targetRequestId = requestId || (await getLatestRequestId());
    if (!targetRequestId) {
      meta.textContent = "暂无已完成请求。";
      return;
    }
    requestInput.value = targetRequestId;
    const detail = await fetchJson(
      `/api/v1/traces/${encodeURIComponent(targetRequestId)}?view_all=true&step_limit=100`,
    );
    const trace = detail.trace || detail;
    renderRows(extractValues(detail));
    meta.textContent = `Request ${targetRequestId}，状态 ${trace.status || "-"}，输入：${trace.user_input || "-"}`;
  }

  loadButton.addEventListener("click", () => {
    loadPerformance(requestInput.value.trim()).catch((error) => {
      meta.textContent = `加载失败：${error.message}`;
      renderRows({});
    });
  });

  refreshButton.addEventListener("click", () => {
    loadPerformance(requestInput.value.trim()).catch((error) => {
      meta.textContent = `加载失败：${error.message}`;
      renderRows({});
    });
  });

  loadPerformance().catch((error) => {
    meta.textContent = `加载失败：${error.message}`;
  });
})();
