(() => {
  const { fetchJson } = window.AdminCommon;

  const ragConfigForm = document.getElementById("rag-config-form");
  const ragConfigHint = document.getElementById("rag-config-hint");
  const evalRunButton = document.getElementById("rag-eval-run");
  const evalHint = document.getElementById("rag-eval-hint");
  const evalSummary = document.getElementById("rag-eval-summary");
  const evalCases = document.getElementById("rag-eval-cases");
  let currentTemperature = 0;
  let currentConfig = null;

  function fillRagConfigForm(config) {
    currentConfig = config;
    currentTemperature = Number(config.llm_temperature ?? 0);
    document.getElementById("rag-top-k").value = config.top_k;
    document.getElementById("rag-plan-top-k").value = config.plan_top_k ?? config.top_k;
    document.getElementById("rag-score-threshold").value = config.score_threshold;
    document.getElementById("rag-candidate-multiplier").value = config.candidate_multiplier;
    document.getElementById("rag-plan-candidate-multiplier").value =
      config.plan_candidate_multiplier ?? config.candidate_multiplier;
    document.getElementById("rag-rerank-candidate-limit").value = config.rerank_candidate_limit;
    document.getElementById("rag-plan-rerank-candidate-limit").value =
      config.plan_rerank_candidate_limit ?? config.rerank_candidate_limit;
    document.getElementById("rag-retrieval-mode").value = config.retrieval_mode;
    document.getElementById("rag-bm25-top-k").value = config.bm25_top_k;
    document.getElementById("rag-plan-bm25-top-k").value = config.plan_bm25_top_k ?? config.bm25_top_k;
    document.getElementById("rag-bm25-title-boost").value = config.bm25_title_boost;
    document.getElementById("rag-rrf-k").value = config.rrf_k;
    document.getElementById("rag-rrf-min-score").value = config.rrf_min_score;
    document.getElementById("rag-chunk-size").value = config.chunk_size;
    document.getElementById("rag-chunk-overlap").value = config.chunk_overlap;
    document.getElementById("rag-plan-max-retries").value = config.plan_max_retries ?? 1;
    document.getElementById("rag-plan-retry-multiplier").value = config.plan_retry_multiplier ?? 2;
    document.getElementById("rag-reranker-enabled").checked = Boolean(config.reranker_enabled);
    ragConfigHint.textContent =
      `当前配置：Top K=${config.top_k}/${config.plan_top_k ?? config.top_k}，Threshold=${config.score_threshold}，` +
      `Candidate Multiplier=${config.candidate_multiplier}/${config.plan_candidate_multiplier ?? config.candidate_multiplier}，` +
      `Rerank Limit=${config.rerank_candidate_limit}/${config.plan_rerank_candidate_limit ?? config.rerank_candidate_limit}，` +
      `Mode=${config.retrieval_mode}，BM25=${config.bm25_top_k}/${config.bm25_title_boost}，RRF=${config.rrf_k}/${config.rrf_min_score}，` +
      `Plan Retry=${config.plan_max_retries ?? 1} x${config.plan_retry_multiplier ?? 2}，` +
      `Chunk=${config.chunk_size}/${config.chunk_overlap}，` +
      `重排序=${config.reranker_enabled ? "开启" : "关闭"}`;
  }

  async function loadRagConfig() {
    const data = await fetchJson("/api/v1/admin/rag/config");
    fillRagConfigForm(data);
  }

  ragConfigForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const payload = {
        top_k: Number(document.getElementById("rag-top-k").value),
        plan_top_k: Number(document.getElementById("rag-plan-top-k").value),
        score_threshold: Number(document.getElementById("rag-score-threshold").value),
        candidate_multiplier: Number(document.getElementById("rag-candidate-multiplier").value),
        plan_candidate_multiplier: Number(document.getElementById("rag-plan-candidate-multiplier").value),
        rerank_candidate_limit: Number(document.getElementById("rag-rerank-candidate-limit").value),
        plan_rerank_candidate_limit: Number(document.getElementById("rag-plan-rerank-candidate-limit").value),
        plan_max_retries: Number(document.getElementById("rag-plan-max-retries").value),
        plan_retry_multiplier: Number(document.getElementById("rag-plan-retry-multiplier").value),
        retrieval_mode: document.getElementById("rag-retrieval-mode").value,
        bm25_top_k: Number(document.getElementById("rag-bm25-top-k").value),
        plan_bm25_top_k: Number(document.getElementById("rag-plan-bm25-top-k").value),
        bm25_title_boost: Number(document.getElementById("rag-bm25-title-boost").value),
        rrf_k: Number(document.getElementById("rag-rrf-k").value),
        rrf_min_score: Number(document.getElementById("rag-rrf-min-score").value),
        chunk_size: Number(document.getElementById("rag-chunk-size").value),
        chunk_overlap: Number(document.getElementById("rag-chunk-overlap").value),
        reranker_enabled: document.getElementById("rag-reranker-enabled").checked,
        llm_temperature: currentTemperature,
      };
      const data = await fetchJson("/api/v1/admin/rag/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      fillRagConfigForm(data);
      ragConfigHint.textContent =
        `已保存：Top K=${data.top_k}/${data.plan_top_k ?? data.top_k}，Threshold=${data.score_threshold}，` +
        `Candidate Multiplier=${data.candidate_multiplier}/${data.plan_candidate_multiplier ?? data.candidate_multiplier}，` +
        `Rerank Limit=${data.rerank_candidate_limit}/${data.plan_rerank_candidate_limit ?? data.rerank_candidate_limit}，` +
        `Mode=${data.retrieval_mode}，BM25=${data.bm25_top_k}/${data.bm25_title_boost}，RRF=${data.rrf_k}/${data.rrf_min_score}，` +
        `Plan Retry=${data.plan_max_retries ?? 1} x${data.plan_retry_multiplier ?? 2}，` +
        `Chunk=${data.chunk_size}/${data.chunk_overlap}，` +
        `重排序=${data.reranker_enabled ? "开启" : "关闭"}`;
    } catch (error) {
      ragConfigHint.textContent = `保存失败：${error.message}`;
    }
  });

  function buildEvaluationVariants(config) {
    return [
      {
        name: "当前配置",
        top_k: Number(config.top_k),
        score_threshold: Number(config.score_threshold),
        candidate_multiplier: Number(config.candidate_multiplier),
        rerank_candidate_limit: Number(config.rerank_candidate_limit),
        retrieval_mode: config.retrieval_mode,
        bm25_top_k: Number(config.bm25_top_k),
        bm25_title_boost: Number(config.bm25_title_boost),
        rrf_k: Number(config.rrf_k),
        rrf_min_score: Number(config.rrf_min_score),
        reranker_enabled: Boolean(config.reranker_enabled),
      },
      {
        name: "plan_execute 增强档",
        top_k: Number(config.plan_top_k ?? config.top_k),
        score_threshold: Number(config.score_threshold),
        candidate_multiplier: Number(config.plan_candidate_multiplier ?? config.candidate_multiplier),
        rerank_candidate_limit: Number(config.plan_rerank_candidate_limit ?? config.rerank_candidate_limit),
        retrieval_mode: config.retrieval_mode,
        bm25_top_k: Number(config.plan_bm25_top_k ?? config.bm25_top_k),
        bm25_title_boost: Number(config.bm25_title_boost),
        rrf_k: Number(config.rrf_k),
        rrf_min_score: Number(config.rrf_min_score),
        reranker_enabled: Boolean(config.reranker_enabled),
      },
    ];
  }

  function renderEvaluationSummary(response) {
    evalSummary.innerHTML = "";
    for (const summary of response.summaries || []) {
      const card = document.createElement("article");
      card.className = "metric-card";
      card.innerHTML = `
        <div class="detail-card__title-row">
          <h3>${summary.name}</h3>
          <span class="badge badge--neutral">${summary.total_cases} 题</span>
        </div>
        <div class="metric-list">
          <div class="metric-row"><span class="metric-row__label">来源命中率</span><strong>${(Number(summary.source_hit_rate || 0) * 100).toFixed(0)}%</strong></div>
          <div class="metric-row"><span class="metric-row__label">答案命中率</span><strong>${(Number(summary.answer_hit_rate || 0) * 100).toFixed(0)}%</strong></div>
          <div class="metric-row"><span class="metric-row__label">来源命中题数</span><strong>${summary.source_hit_cases} / ${summary.total_cases}</strong></div>
          <div class="metric-row"><span class="metric-row__label">平均返回来源</span><strong>${Number(summary.average_returned_sources || 0).toFixed(1)}</strong></div>
        </div>
      `;
      evalSummary.appendChild(card);
    }
  }

  function renderEvaluationCases(response) {
    evalCases.innerHTML = "";
    for (const caseResult of response.cases || []) {
      const item = document.createElement("article");
      item.className = "mini-list__item";
      const title = document.createElement("strong");
      title.textContent = caseResult.query;
      item.appendChild(title);

      for (const variant of caseResult.variants || []) {
        const row = document.createElement("div");
        row.className = "mini-list__row";
        const label = document.createElement("span");
        label.textContent = variant.name;
        const meta = document.createElement("div");
        meta.className = "mini-list__actions";
        meta.innerHTML = `
          <span class="badge ${variant.source_hit ? "badge--success" : "badge--neutral"}">来源 ${variant.source_hit ? "命中" : "未全中"}</span>
          <span class="badge ${variant.answer_hit ? "badge--success" : "badge--neutral"}">答案 ${variant.answer_hit ? "命中" : "未全中"}</span>
        `;
        row.append(label, meta);
        item.appendChild(row);
      }
      evalCases.appendChild(item);
    }
  }

  evalRunButton.addEventListener("click", async () => {
    try {
      if (!currentConfig) {
        throw new Error("当前配置尚未加载完成。");
      }
      evalRunButton.disabled = true;
      evalHint.textContent = "正在加载默认复杂题集并运行回归评测...";
      evalSummary.innerHTML = "";
      evalCases.innerHTML = '<div class="empty-state">评测运行中，请稍候。</div>';

      const preset = await fetchJson("/api/v1/admin/rag/evaluate/default-cases");
      const response = await fetchJson("/api/v1/admin/rag/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cases: preset.cases || [],
          variants: buildEvaluationVariants(currentConfig),
          generate_answer: true,
        }),
      });

      renderEvaluationSummary(response);
      renderEvaluationCases(response);
      evalHint.textContent = `已完成默认回归：${preset.name}，共 ${(preset.cases || []).length} 题。`;
    } catch (error) {
      evalSummary.innerHTML = "";
      evalCases.innerHTML = "";
      evalHint.textContent = `默认回归失败：${error.message}`;
    } finally {
      evalRunButton.disabled = false;
    }
  });

  loadRagConfig().catch((error) => {
    ragConfigHint.textContent = `加载配置失败：${error.message}`;
  });
})();
