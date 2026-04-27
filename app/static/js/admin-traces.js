(() => {
  const { fetchJson, formatDateTime, statusBadgeClass } = window.AdminCommon;

  const traceList = document.getElementById("trace-list");
  const traceSummary = document.getElementById("trace-summary");
  const traceTimeline = document.getElementById("trace-timeline");
  const traceSnapshot = document.getElementById("trace-snapshot");
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

  const STEP_LABELS = {
    intent_decision: {
      title: "意图识别",
      description: "判断是否需要走知识库、是否改写用户问题。",
    },
    retrieval: {
      title: "知识库检索",
      description: "根据改写后的问题召回、融合、过滤和整理来源片段。",
    },
    embedding: {
      title: "Embedding 向量化",
      description: "把检索 query 转成向量，用于向量库检索。",
    },
    qdrant_search: {
      title: "Qdrant 向量检索",
      description: "在向量库中按语义相似度召回候选 chunk。",
    },
    rerank: {
      title: "重排序",
      description: "用 reranker 对候选 chunk 重新打分排序。",
    },
    llm_generation: {
      title: "LLM 生成",
      description: "把用户问题和最终来源片段交给模型生成回答。",
    },
    first_token_generation: {
      title: "首 Token 生成",
      description: "从进入生成阶段到第一个 token 返回的耗时。",
    },
  };

  const STAGE_LABELS = {
    dense_candidates: {
      title: "向量召回候选",
      description: "Qdrant 按语义相似度召回的原始候选，适合判断“语义检索有没有找回来”。",
    },
    bm25_candidates: {
      title: "关键词召回候选",
      description: "BM25 按关键词匹配召回的原始候选，适合判断“关键词是否命中正确文档”。",
    },
    rrf_fused: {
      title: "RRF 融合排序",
      description: "把向量召回和关键词召回按排名融合后的排序结果。",
    },
    coarse_deduped: {
      title: "前置粗去重后",
      description: "重排前先去掉明显重复和同文档过多候选，减少 reranker 输入量。",
    },
    score_filtered: {
      title: "低分过滤后",
      description: "关闭重排时按阈值过滤低分候选，避免弱相关内容进入 Prompt。",
    },
    rerank_input: {
      title: "重排输入",
      description: "真正送入 reranker 的候选列表，可看输入数量是否过大。",
    },
    rerank_output: {
      title: "重排输出",
      description: "reranker 重新打分后的排序，可和重排输入对比排名变化。",
    },
    rerank_score_filtered: {
      title: "重排低分过滤后",
      description: "按 reranker 分数阈值过滤后的结果。",
    },
    postprocess_input: {
      title: "后处理输入",
      description: "进入向量相似度去重和相邻 chunk 合并前的来源列表。",
    },
    vector_deduped: {
      title: "同文档相似去重后",
      description: "同一文档内余弦相似度过高的 chunk 只保留一个。",
    },
    adjacent_merged: {
      title: "相邻 Chunk 合并后",
      description: "同文档相邻片段合并成更完整段落，减少上下文割裂。",
    },
    citation_mapped: {
      title: "引用编号映射后",
      description: "给最终来源分配【1】【2】这样的引用编号。",
    },
    final_sources: {
      title: "最终进入 Prompt 的来源",
      description: "最终会交给 LLM 的知识库片段，生成质量主要看这里。",
    },
  };

  const INTENT_LABELS = {
    chat: "普通对话",
    knowledge_qa: "知识库问答",
    task: "任务执行",
    follow_up: "多轮追问",
    reject: "拒答",
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
      title.textContent = labelIntent(trace.intent) || "待处理";

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

    renderSummary(trace);
    renderTimeline(trace.steps || []);
    renderSnapshot(wrapper.snapshot);

    traceOutput.textContent = trace.final_output || trace.generation_record?.llm_output || "-";
    renderLazyJson(traceRaw, trace, "点击“查看完整 JSON”后展示完整 trace 原始数据。");
  }

  function renderSummary(trace) {
    traceSummary.innerHTML = "";
    const summaryCard = document.createElement("div");
    summaryCard.className = "trace-summary";
    [
      ["用户输入", trace.user_input || "-"],
      ["意图", labelIntent(trace.intent) || "-"],
      ["是否走知识库", trace.need_rag ? "是" : "否"],
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
  }

  function renderTimeline(steps) {
    traceTimeline.innerHTML = "";
    if (!steps.length) {
      traceTimeline.innerHTML = '<div class="empty-state">当前请求还没有步骤明细。</div>';
      return;
    }

    for (const step of steps) {
      const label = STEP_LABELS[step.step_type] || {
        title: step.step_type || "未知步骤",
        description: "当前步骤暂无专门说明。",
      };
      const article = document.createElement("article");
      article.className = "timeline-item";

      const titleRow = document.createElement("div");
      titleRow.className = "timeline-item__title-row";

      const title = document.createElement("strong");
      title.textContent = label.title;

      const badge = document.createElement("span");
      badge.className = statusBadgeClass(step.status);
      badge.textContent = step.status || "pending";

      const description = document.createElement("p");
      description.className = "timeline-item__description";
      description.textContent = label.description;

      const meta = document.createElement("div");
      meta.className = "snapshot-header snapshot-header--compact";
      meta.append(
        createMetric("顺序", String(step.step_order ?? "-")),
        createMetric("耗时", `${step.latency_ms ?? "-"} ms`),
        createMetric("关联数据", labelRecordRef(step.record_ref_type)),
      );

      const jsonButton = document.createElement("button");
      jsonButton.className = "button button--ghost button--small";
      jsonButton.type = "button";
      jsonButton.textContent = "查看步骤 JSON";
      const code = document.createElement("pre");
      code.className = "code-block is-hidden";
      jsonButton.addEventListener("click", () => toggleLazyJson(jsonButton, code, step, "查看步骤 JSON"));

      titleRow.append(title, badge);
      article.append(titleRow, description, meta, jsonButton, code);
      traceTimeline.appendChild(article);
    }
  }

  function renderSnapshot(snapshot) {
    traceSnapshot.innerHTML = "";
    if (!snapshot) {
      traceSnapshot.textContent = "当前请求暂无内存快照，可能是服务重启前的历史日志。";
      return;
    }

    const header = document.createElement("div");
    header.className = "snapshot-header";
    header.append(
      createMetric("请求 ID", snapshot.request_id || "-"),
      createMetric("用户问题", snapshot.user_query || "-"),
      createMetric("检索 Query", snapshot.retrieval?.query || "-"),
    );
    traceSnapshot.appendChild(header);

    traceSnapshot.appendChild(createIntentCard(snapshot.intent));
    traceSnapshot.appendChild(createRetrievalSection(snapshot.retrieval?.stages || []));

    const generation = snapshot.generation || {};
    traceSnapshot.appendChild(
      createLazyTextDetails(
        "最终 Prompt",
        "真正发给模型的 messages。默认不渲染，避免大 Prompt 拖慢页面。",
        () => JSON.stringify(generation.prompt_messages || [], null, 2),
      ),
    );
    traceSnapshot.appendChild(
      createLazyTextDetails(
        "LLM 原始输出",
        "模型未经过引用补齐等后处理前的原始回答。",
        () => generation.raw_llm_output || "-",
      ),
    );
    traceSnapshot.appendChild(
      createLazyTextDetails(
        "后处理后输出",
        "系统补齐引用编号等后处理后的最终回答。",
        () => generation.final_output || "-",
      ),
    );
    traceSnapshot.appendChild(
      createLazyTextDetails(
        "完整快照 JSON",
        "调试需要时再展开全量快照 JSON。",
        () => JSON.stringify(snapshot, null, 2),
      ),
    );
  }

  function createIntentCard(intent) {
    const card = document.createElement("section");
    card.className = "snapshot-section snapshot-section--plain";
    const title = document.createElement("h4");
    title.textContent = "意图识别结果";
    const grid = document.createElement("div");
    grid.className = "snapshot-header snapshot-header--compact";
    grid.append(
      createMetric("意图", labelIntent(intent?.intent) || "-"),
      createMetric("是否检索", intent?.need_rag ? "是" : "否"),
      createMetric("改写查询", intent?.rewrite_query || "-"),
    );
    const reason = document.createElement("p");
    reason.className = "snapshot-note";
    reason.textContent = `判断依据：${intent?.rationale || "-"}`;
    card.append(title, grid, reason);
    return card;
  }

  function createRetrievalSection(stages) {
    const section = document.createElement("section");
    section.className = "snapshot-section snapshot-section--plain";

    const title = document.createElement("h4");
    title.textContent = `检索链路（${stages.length} 个阶段）`;
    const note = document.createElement("p");
    note.className = "snapshot-note";
    note.textContent = "默认只展示每个阶段的中文说明、数量和关键参数；点击阶段后再加载 chunk 列表。";
    section.append(title, note);

    for (const stage of stages) {
      section.appendChild(createStageBlock(stage));
    }
    return section;
  }

  function createMetric(label, value) {
    const item = document.createElement("div");
    item.className = "snapshot-metric";
    const strong = document.createElement("strong");
    strong.textContent = label;
    const span = document.createElement("span");
    span.textContent = value;
    item.append(strong, span);
    return item;
  }

  function createStageBlock(stage) {
    const label = STAGE_LABELS[stage.name] || {
      title: stage.name || "未知阶段",
      description: "当前阶段暂无专门说明。",
    };
    const details = document.createElement("details");
    details.className = "snapshot-stage";

    const summary = document.createElement("summary");
    const summaryTitle = document.createElement("span");
    summaryTitle.textContent = `${label.title} · ${stage.count || 0} 条`;
    const summaryHint = document.createElement("small");
    summaryHint.textContent = "点击展开候选片段";
    summary.append(summaryTitle, summaryHint);
    details.appendChild(summary);

    const description = document.createElement("p");
    description.className = "snapshot-note";
    description.textContent = label.description;
    details.appendChild(description);

    if (stage.metadata && Object.keys(stage.metadata).length) {
      const metaGrid = document.createElement("div");
      metaGrid.className = "snapshot-header snapshot-header--compact";
      for (const [key, value] of Object.entries(stage.metadata)) {
        metaGrid.appendChild(createMetric(labelMetadata(key), String(value)));
      }
      details.appendChild(metaGrid);
    }

    const container = document.createElement("div");
    container.className = "snapshot-lazy";
    container.dataset.loaded = "false";
    details.appendChild(container);

    details.addEventListener("toggle", () => {
      if (!details.open || container.dataset.loaded === "true") {
        return;
      }
      container.dataset.loaded = "true";
      renderStageChunks(container, stage);
    });

    return details;
  }

  function renderStageChunks(container, stage) {
    const chunks = stage.chunks || [];
    if (!chunks.length) {
      container.innerHTML = '<div class="empty-state">这个阶段没有可展示的候选片段。</div>';
      return;
    }

    const toolbar = document.createElement("div");
    toolbar.className = "snapshot-toolbar";
    const jsonButton = document.createElement("button");
    jsonButton.className = "button button--ghost button--small";
    jsonButton.type = "button";
    jsonButton.textContent = "查看阶段 JSON";
    const jsonCode = document.createElement("pre");
    jsonCode.className = "code-block is-hidden";
    jsonButton.addEventListener("click", () => toggleLazyJson(jsonButton, jsonCode, stage, "查看阶段 JSON"));
    toolbar.appendChild(jsonButton);

    const list = document.createElement("div");
    list.className = "snapshot-chunk-list";
    for (const chunk of chunks) {
      list.appendChild(createChunkCard(chunk));
    }
    container.append(toolbar, list, jsonCode);
  }

  function createChunkCard(chunk) {
    const card = document.createElement("article");
    card.className = "snapshot-chunk";

    const title = document.createElement("strong");
    title.textContent = `#${chunk.rank} · ${chunk.title || "-"} · 分数 ${formatScore(chunk.score)}`;

    const meta = document.createElement("p");
    meta.className = "snapshot-chunk__meta";
    meta.textContent = `文档 ID：${chunk.document_id || "-"} · Chunk：${chunk.chunk_id || "-"}`;

    const content = document.createElement("p");
    content.className = "snapshot-chunk__content";
    content.textContent = chunk.content || "-";

    const actions = document.createElement("div");
    actions.className = "snapshot-toolbar";

    const expandButton = document.createElement("button");
    expandButton.className = "button button--ghost button--small";
    expandButton.type = "button";
    expandButton.textContent = "展开全文";
    expandButton.addEventListener("click", () => {
      card.classList.toggle("is-expanded");
      expandButton.textContent = card.classList.contains("is-expanded") ? "收起全文" : "展开全文";
    });

    const jsonButton = document.createElement("button");
    jsonButton.className = "button button--ghost button--small";
    jsonButton.type = "button";
    jsonButton.textContent = "查看 Chunk JSON";
    const jsonCode = document.createElement("pre");
    jsonCode.className = "code-block is-hidden";
    jsonButton.addEventListener("click", () => toggleLazyJson(jsonButton, jsonCode, chunk, "查看 Chunk JSON"));

    actions.append(expandButton, jsonButton);
    card.append(title, meta, content, actions, jsonCode);
    return card;
  }

  function createLazyTextDetails(title, description, getText) {
    const details = document.createElement("details");
    details.className = "snapshot-section";
    const summary = document.createElement("summary");
    summary.textContent = title;
    const note = document.createElement("p");
    note.className = "snapshot-note";
    note.textContent = description;
    const code = document.createElement("pre");
    code.className = "code-block";
    code.textContent = "展开后加载内容。";
    details.append(summary, note, code);
    details.addEventListener("toggle", () => {
      if (details.open && code.textContent === "展开后加载内容。") {
        code.textContent = getText();
      }
    });
    return details;
  }

  function renderLazyJson(preElement, payload, placeholder) {
    preElement.classList.add("is-hidden");
    preElement.textContent = placeholder;
    delete preElement.dataset.loaded;
    const existingButton = document.getElementById("trace-raw-toggle");
    if (existingButton) {
      existingButton.textContent = "查看完整 JSON";
      existingButton.onclick = () => toggleLazyJson(existingButton, preElement, payload, "查看完整 JSON");
    }
  }

  function toggleLazyJson(button, preElement, payload, closedText) {
    const willShow = preElement.classList.contains("is-hidden");
    if (willShow && !preElement.dataset.loaded) {
      preElement.textContent = JSON.stringify(payload, null, 2);
      preElement.dataset.loaded = "true";
    }
    preElement.classList.toggle("is-hidden", !willShow);
    button.textContent = willShow ? "收起 JSON" : closedText;
  }

  function labelIntent(intent) {
    return INTENT_LABELS[intent] || intent || "";
  }

  function labelRecordRef(recordRefType) {
    const labels = {
      intent_record: "意图记录",
      retrieval_record: "检索记录",
      generation_record: "生成记录",
    };
    return labels[recordRefType] || "无";
  }

  function labelMetadata(key) {
    const labels = {
      limit: "召回上限",
      title_boost: "标题权重",
      rrf_k: "RRF K",
      max_chunks_per_document: "单文档保留数",
      min_score: "最低分",
      score_type: "分数类型",
      max_candidates: "重排候选上限",
      similarity_threshold: "相似度阈值",
    };
    return labels[key] || key;
  }

  function formatScore(score) {
    const value = Number(score);
    return Number.isFinite(value) ? value.toFixed(6) : "-";
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
      traceSnapshot.textContent = "当前没有可展示的内存快照。";
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
