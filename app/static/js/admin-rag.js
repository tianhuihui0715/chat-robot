(() => {
  const { fetchJson } = window.AdminCommon;

  const ragConfigForm = document.getElementById("rag-config-form");
  const ragConfigHint = document.getElementById("rag-config-hint");
  let currentTemperature = 0;

  function fillRagConfigForm(config) {
    currentTemperature = Number(config.llm_temperature ?? 0);
    document.getElementById("rag-top-k").value = config.top_k;
    document.getElementById("rag-score-threshold").value = config.score_threshold;
    document.getElementById("rag-candidate-multiplier").value = config.candidate_multiplier;
    document.getElementById("rag-rerank-candidate-limit").value = config.rerank_candidate_limit;
    document.getElementById("rag-retrieval-mode").value = config.retrieval_mode;
    document.getElementById("rag-bm25-top-k").value = config.bm25_top_k;
    document.getElementById("rag-bm25-title-boost").value = config.bm25_title_boost;
    document.getElementById("rag-rrf-k").value = config.rrf_k;
    document.getElementById("rag-rrf-min-score").value = config.rrf_min_score;
    document.getElementById("rag-chunk-size").value = config.chunk_size;
    document.getElementById("rag-chunk-overlap").value = config.chunk_overlap;
    document.getElementById("rag-reranker-enabled").checked = Boolean(config.reranker_enabled);
    ragConfigHint.textContent =
      `当前配置：Top K=${config.top_k}，Threshold=${config.score_threshold}，` +
      `Candidate Multiplier=${config.candidate_multiplier}，Rerank Limit=${config.rerank_candidate_limit}，` +
      `Mode=${config.retrieval_mode}，BM25=${config.bm25_top_k}/${config.bm25_title_boost}，RRF=${config.rrf_k}/${config.rrf_min_score}，` +
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
        score_threshold: Number(document.getElementById("rag-score-threshold").value),
        candidate_multiplier: Number(document.getElementById("rag-candidate-multiplier").value),
        rerank_candidate_limit: Number(document.getElementById("rag-rerank-candidate-limit").value),
        retrieval_mode: document.getElementById("rag-retrieval-mode").value,
        bm25_top_k: Number(document.getElementById("rag-bm25-top-k").value),
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
        `已保存：Top K=${data.top_k}，Threshold=${data.score_threshold}，` +
        `Candidate Multiplier=${data.candidate_multiplier}，Rerank Limit=${data.rerank_candidate_limit}，` +
        `Mode=${data.retrieval_mode}，BM25=${data.bm25_top_k}/${data.bm25_title_boost}，RRF=${data.rrf_k}/${data.rrf_min_score}，` +
        `Chunk=${data.chunk_size}/${data.chunk_overlap}，` +
        `重排序=${data.reranker_enabled ? "开启" : "关闭"}`;
    } catch (error) {
      ragConfigHint.textContent = `保存失败：${error.message}`;
    }
  });

  loadRagConfig().catch((error) => {
    ragConfigHint.textContent = `加载配置失败：${error.message}`;
  });
})();
