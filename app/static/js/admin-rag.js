(() => {
  const { fetchJson } = window.AdminCommon;

  const ragConfigForm = document.getElementById("rag-config-form");
  const ragConfigHint = document.getElementById("rag-config-hint");

  function fillRagConfigForm(config) {
    document.getElementById("rag-top-k").value = config.top_k;
    document.getElementById("rag-score-threshold").value = config.score_threshold;
    document.getElementById("rag-candidate-multiplier").value = config.candidate_multiplier;
    document.getElementById("rag-chunk-size").value = config.chunk_size;
    document.getElementById("rag-chunk-overlap").value = config.chunk_overlap;
    document.getElementById("rag-reranker-enabled").checked = Boolean(config.reranker_enabled);
    ragConfigHint.textContent =
      `当前配置：Top K=${config.top_k}，Threshold=${config.score_threshold}，` +
      `Candidate Multiplier=${config.candidate_multiplier}，` +
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
      ragConfigHint.textContent =
        `已保存：Top K=${data.top_k}，Threshold=${data.score_threshold}，` +
        `Candidate Multiplier=${data.candidate_multiplier}，` +
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
