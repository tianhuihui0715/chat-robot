(() => {
  const form = document.getElementById("rag-lab-form");
  const filesInput = document.getElementById("rag-lab-files");
  const questionsInput = document.getElementById("rag-lab-questions");
  const hint = document.getElementById("rag-lab-hint");
  const statusBadge = document.getElementById("rag-lab-status");
  const submitButton = document.getElementById("rag-lab-submit");
  const addVariantButton = document.getElementById("add-variant-button");
  const variantList = document.getElementById("variant-list");
  const variantTemplate = document.getElementById("variant-card-template");
  const actionsPanel = document.getElementById("rag-lab-actions");
  const sessionMeta = document.getElementById("rag-lab-session-meta");
  const exportExcelButton = document.getElementById("rag-lab-export-excel");
  const exportWordButton = document.getElementById("rag-lab-export-word");
  const resultsPanel = document.getElementById("rag-lab-results-panel");
  const summaryResults = document.getElementById("rag-lab-summary-results");
  const variantResults = document.getElementById("rag-lab-variant-results");
  const questionResults = document.getElementById("rag-lab-question-results");
  const viewButtons = Array.from(document.querySelectorAll("[data-view]"));
  const viewPanels = Array.from(document.querySelectorAll("[data-view-panel]"));

  const state = {
    sessionId: null,
    running: false,
    runningTimer: null,
    startedAt: 0,
    variantCounter: 0,
    variants: [],
  };

  const VARIANT_PRESETS = [
    {
      name: "方案 A",
      chunk_size: 1200,
      chunk_overlap_ratio: 0.1,
      retrieval_k: 6,
      retrieval_mode: "hybrid",
      bm25_top_k: 8,
      bm25_title_boost: 2.0,
      rrf_k: 60,
      rerank_k: 4,
      temperature: 0.0,
    },
    {
      name: "方案 B",
      chunk_size: 800,
      chunk_overlap_ratio: 0.2,
      retrieval_k: 8,
      retrieval_mode: "dense",
      bm25_top_k: 8,
      bm25_title_boost: 2.0,
      rrf_k: 60,
      rerank_k: 4,
      temperature: 0.4,
    },
    {
      name: "方案 C",
      chunk_size: 400,
      chunk_overlap_ratio: 0.25,
      retrieval_k: 10,
      retrieval_mode: "bm25",
      bm25_top_k: 12,
      bm25_title_boost: 2.4,
      rrf_k: 60,
      rerank_k: 0,
      temperature: 0.2,
    },
    {
      name: "方案 D",
      chunk_size: 1600,
      chunk_overlap_ratio: 0.08,
      retrieval_k: 5,
      retrieval_mode: "hybrid",
      bm25_top_k: 10,
      bm25_title_boost: 3.0,
      rrf_k: 80,
      rerank_k: 3,
      temperature: 0.0,
    },
  ];

  function setRunningState(isRunning) {
    state.running = isRunning;
    Array.from(form.querySelectorAll("input, textarea, button")).forEach((control) => {
      control.disabled = isRunning;
    });

    if (!isRunning) {
      if (state.runningTimer) {
        window.clearInterval(state.runningTimer);
        state.runningTimer = null;
      }
      submitButton.textContent = "开始实验";
      return;
    }

    state.startedAt = Date.now();
    submitButton.textContent = "实验中...";
    state.runningTimer = window.setInterval(() => {
      const elapsed = Math.max(1, Math.round((Date.now() - state.startedAt) / 1000));
      hint.textContent = `实验运行中，已耗时 ${elapsed} 秒。多格式文件解析、重排和答案生成都会增加总耗时。`;
    }, 1000);
  }

  function setStatus(text, type = "neutral") {
    statusBadge.textContent = text;
    statusBadge.className = `badge badge--${type}`;
  }

  function createHint(text) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = text;
    return p;
  }

  function createBadge(text, type = "neutral") {
    const badge = document.createElement("span");
    badge.className = `badge badge--${type}`;
    badge.textContent = text;
    return badge;
  }

  function shorten(text, limit = 160) {
    const compact = String(text || "").replace(/\s+/g, " ").trim();
    if (compact.length <= limit) {
      return compact;
    }
    return `${compact.slice(0, limit - 1)}…`;
  }

  function switchView(viewName) {
    viewButtons.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.view === viewName);
    });
    viewPanels.forEach((panel) => {
      panel.classList.toggle("is-active", panel.dataset.viewPanel === viewName);
    });
  }

  function addVariantCard(preset = null) {
    if (state.variants.length >= 4) {
      hint.textContent = "最多保留 4 套参数方案。";
      return;
    }

    state.variantCounter += 1;
    const fragment = variantTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".rag-lab-variant-card");
    const id = `variant-${state.variantCounter}`;
    card.dataset.variantId = id;

    const defaults = preset || VARIANT_PRESETS[Math.min(state.variants.length, VARIANT_PRESETS.length - 1)];
    const title = card.querySelector(".variant-title");
    const removeButton = card.querySelector(".variant-remove");
    const nameInput = card.querySelector(".variant-name");
    const chunkInput = card.querySelector(".variant-chunk-size");
    const overlapInput = card.querySelector(".variant-overlap-ratio");
    const retrievalInput = card.querySelector(".variant-retrieval-k");
    const retrievalModeInput = card.querySelector(".variant-retrieval-mode");
    const bm25TopKInput = card.querySelector(".variant-bm25-top-k");
    const bm25TitleBoostInput = card.querySelector(".variant-bm25-title-boost");
    const rrfKInput = card.querySelector(".variant-rrf-k");
    const rerankInput = card.querySelector(".variant-rerank-k");
    const temperatureInput = card.querySelector(".variant-temperature");

    title.textContent = defaults.name;
    nameInput.value = defaults.name;
    chunkInput.value = defaults.chunk_size;
    overlapInput.value = defaults.chunk_overlap_ratio;
    retrievalInput.value = defaults.retrieval_k;
    retrievalModeInput.value = defaults.retrieval_mode ?? "hybrid";
    bm25TopKInput.value = defaults.bm25_top_k ?? 8;
    bm25TitleBoostInput.value = defaults.bm25_title_boost ?? 2.0;
    rrfKInput.value = defaults.rrf_k ?? 60;
    rerankInput.value = defaults.rerank_k;
    temperatureInput.value = defaults.temperature;

    const updateTitle = () => {
      title.textContent = nameInput.value.trim() || "未命名方案";
    };
    nameInput.addEventListener("input", updateTitle);

    removeButton.addEventListener("click", () => {
      card.remove();
      state.variants = state.variants.filter((item) => item !== card);
      syncRemoveButtons();
    });

    variantList.appendChild(card);
    state.variants.push(card);
    syncRemoveButtons();
  }

  function syncRemoveButtons() {
    const removable = state.variants.length > 2;
    state.variants.forEach((card) => {
      card.querySelector(".variant-remove").disabled = !removable;
    });
  }

  function collectVariants() {
    const variants = state.variants.map((card, index) => {
      const name = card.querySelector(".variant-name").value.trim() || `方案 ${index + 1}`;
      const retrievalK = Number(card.querySelector(".variant-retrieval-k").value);
      const bm25TopK = Number(card.querySelector(".variant-bm25-top-k").value);
      const rerankK = Number(card.querySelector(".variant-rerank-k").value);
      if (rerankK > Math.max(retrievalK, bm25TopK)) {
        throw new Error(`${name} 的重排数量不能大于可用候选数量。`);
      }
      return {
        variant_id: card.dataset.variantId,
        name,
        chunk_size: Number(card.querySelector(".variant-chunk-size").value),
        chunk_overlap_ratio: Number(card.querySelector(".variant-overlap-ratio").value),
        retrieval_k: retrievalK,
        retrieval_mode: card.querySelector(".variant-retrieval-mode").value,
        bm25_top_k: bm25TopK,
        bm25_title_boost: Number(card.querySelector(".variant-bm25-title-boost").value),
        rrf_k: Number(card.querySelector(".variant-rrf-k").value),
        rerank_k: rerankK,
        temperature: Number(card.querySelector(".variant-temperature").value),
      };
    });

    if (!variants.length) {
      throw new Error("请至少保留一套参数方案。");
    }
    return variants;
  }

  function collectQuestions() {
    const questions = questionsInput.value
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (!questions.length) {
      throw new Error("请至少输入一个问题。");
    }
    return questions;
  }

  function createApplyButton(sessionId, variantId, variantName) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button button--ghost button--small";
    button.textContent = "应用为正式参数";
    button.addEventListener("click", async () => {
      button.disabled = true;
      button.textContent = "应用中...";
      try {
        const response = await fetch(
          `/api/v1/admin/rag/lab/sessions/${encodeURIComponent(sessionId)}/apply/${encodeURIComponent(variantId)}`,
          { method: "POST" },
        );
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${response.status}`);
        }
        const data = await response.json();
        hint.textContent = `已应用 ${variantName}。${data.note}`;
      } catch (error) {
        hint.textContent = `应用参数失败：${error.message}`;
      } finally {
        button.disabled = false;
        button.textContent = "应用为正式参数";
      }
    });
    return button;
  }

  function roundMetric(value, digits = 2) {
    if (!Number.isFinite(value)) {
      return "0";
    }
    return Number(value).toFixed(digits);
  }

  function computeVariantMetrics(variant) {
    const questionCount = variant.questions.length || 1;
    const totalSources = variant.questions.reduce((sum, question) => sum + (question.sources?.length || 0), 0);
    const topScores = variant.questions
      .map((question) => question.sources?.[0]?.score)
      .filter((score) => Number.isFinite(score));
    const answerLengths = variant.questions.map((question) => (question.answer_full || question.answer_preview || "").length);

    return {
      totalSources,
      avgSources: totalSources / questionCount,
      avgTopScore: topScores.length ? topScores.reduce((sum, score) => sum + score, 0) / topScores.length : 0,
      avgAnswerLength: answerLengths.length
        ? answerLengths.reduce((sum, length) => sum + length, 0) / answerLengths.length
        : 0,
      hitQuestions: variant.questions.filter((question) => (question.sources?.length || 0) > 0).length,
    };
  }

  function renderSummary(session) {
    summaryResults.innerHTML = "";
    const metricsByVariant = session.variants.map((variant) => ({
      variant,
      metrics: computeVariantMetrics(variant),
    }));

    const maxAvgSources = Math.max(...metricsByVariant.map((item) => item.metrics.avgSources), 0);
    const maxAvgTopScore = Math.max(...metricsByVariant.map((item) => item.metrics.avgTopScore), 0);
    const maxHitQuestions = Math.max(...metricsByVariant.map((item) => item.metrics.hitQuestions), 0);

    metricsByVariant.forEach(({ variant, metrics }) => {
      const card = document.createElement("article");
      card.className = "metric-card";

      const header = document.createElement("div");
      header.className = "detail-card__title-row";
      const title = document.createElement("div");
      const eyebrow = document.createElement("p");
      eyebrow.className = "panel__eyebrow";
      eyebrow.textContent = "Summary";
      const name = document.createElement("h3");
      name.textContent = variant.name;
      title.append(eyebrow, name);

      const badges = document.createElement("div");
      badges.className = "mini-list__actions";
      badges.append(createBadge(variant.retrieval_mode));
      if (metrics.avgSources === maxAvgSources && maxAvgSources > 0) {
        badges.append(createBadge("来源最多", "success"));
      }
      if (metrics.avgTopScore === maxAvgTopScore && maxAvgTopScore > 0) {
        badges.append(createBadge("首条分最高", "success"));
      }
      if (metrics.hitQuestions === maxHitQuestions && maxHitQuestions > 0) {
        badges.append(createBadge("命中题数最多", "success"));
      }
      header.append(title, badges);

      const list = document.createElement("div");
      list.className = "metric-list";
      [
        ["题目命中", `${metrics.hitQuestions} / ${session.question_count}`],
        ["总来源数", `${metrics.totalSources}`],
        ["平均来源数", roundMetric(metrics.avgSources, 1)],
        ["平均首条分", roundMetric(metrics.avgTopScore, 3)],
        ["平均回答长度", `${Math.round(metrics.avgAnswerLength)} 字`],
        ["参数", `chunk ${variant.chunk_size} / overlap ${variant.chunk_overlap_ratio} / dense ${variant.retrieval_k} / bm25 ${variant.bm25_top_k}`],
      ].forEach(([label, value]) => {
        const row = document.createElement("div");
        row.className = "metric-row";
        const labelElement = document.createElement("span");
        labelElement.className = "metric-row__label";
        labelElement.textContent = label;
        const valueElement = document.createElement("strong");
        valueElement.textContent = value;
        row.append(labelElement, valueElement);
        list.appendChild(row);
      });

      card.append(header, list, createApplyButton(session.session_id, variant.variant_id, variant.name));
      summaryResults.appendChild(card);
    });
  }

  function createQuestionCard(questionResult) {
    const item = document.createElement("article");
    item.className = "mini-list__item";

    const question = document.createElement("strong");
    question.textContent = questionResult.question;
    item.append(question, createHint(`回答摘要：${questionResult.answer_preview || "[空]"}`));

    if (questionResult.sources?.length) {
      const sourceList = document.createElement("div");
      sourceList.className = "mini-list";
      questionResult.sources.slice(0, 3).forEach((source) => {
        const sourceItem = document.createElement("article");
        sourceItem.className = "mini-list__item";
        const row = document.createElement("div");
        row.className = "mini-list__row";
        const name = document.createElement("strong");
        name.textContent = source.title;
        row.append(name, createBadge(`score ${Number(source.score).toFixed(3)}`));
        sourceItem.append(row, createHint(shorten(source.preview, 120)), createSourceMeta(source));
        sourceList.appendChild(sourceItem);
      });
      item.append(sourceList);
    } else {
      item.append(createHint("没有返回来源片段。"));
    }

    return item;
  }

  function renderByVariant(session) {
    variantResults.innerHTML = "";
    session.variants.forEach((variant) => {
      const panel = document.createElement("section");
      panel.className = "panel";

      const header = document.createElement("div");
      header.className = "panel__header";
      const left = document.createElement("div");
      const eyebrow = document.createElement("p");
      eyebrow.className = "panel__eyebrow";
      eyebrow.textContent = "Variant";
      const title = document.createElement("h2");
      title.textContent = variant.name;
      left.append(eyebrow, title);

      const right = document.createElement("div");
      right.className = "mini-list__actions";
      right.append(
        createBadge(`chunk ${variant.chunk_size}`),
        createBadge(`overlap ${variant.chunk_overlap_ratio}`),
        createBadge(`召回 ${variant.retrieval_k}`),
        createBadge(variant.retrieval_mode),
        createBadge(`BM25 ${variant.bm25_top_k}`),
        createBadge(`标题权重 ${variant.bm25_title_boost}`),
        createBadge(`RRF ${variant.rrf_k}`),
        createBadge(variant.rerank_k > 0 ? `重排 ${variant.rerank_k}` : "不重排"),
        createBadge(`temp ${variant.temperature}`),
        createApplyButton(session.session_id, variant.variant_id, variant.name),
      );

      header.append(left, right);
      panel.appendChild(header);

      const list = document.createElement("div");
      list.className = "mini-list";
      variant.questions.forEach((questionResult) => {
        list.appendChild(createQuestionCard(questionResult));
      });
      panel.appendChild(list);
      variantResults.appendChild(panel);
    });
  }

  function renderByQuestion(session) {
    questionResults.innerHTML = "";
    const questions = session.variants[0]?.questions?.map((item) => item.question) || [];
    questions.forEach((question, questionIndex) => {
      const variantSlice = session.variants.map((variant) => ({
        variant,
        result: variant.questions[questionIndex],
      }));
      const maxSourceCount = Math.max(...variantSlice.map((item) => item.result?.sources?.length || 0), 0);
      const maxTopScore = Math.max(
        ...variantSlice.map((item) => item.result?.sources?.[0]?.score ?? 0),
        0,
      );

      const panel = document.createElement("article");
      panel.className = "mini-list__item";

      const title = document.createElement("strong");
      title.textContent = question;
      panel.appendChild(title);

      const split = document.createElement("div");
      split.className = "detail-split";
      variantSlice.forEach(({ variant, result }) => {
        const card = document.createElement("section");
        card.className = "detail-card";
        const sourceCount = result?.sources?.length || 0;
        const topScore = result?.sources?.[0]?.score ?? 0;
        if ((sourceCount > 0 && sourceCount === maxSourceCount) || (topScore > 0 && topScore === maxTopScore)) {
          card.classList.add("detail-card--highlight");
        }

        const row = document.createElement("div");
        row.className = "detail-card__title-row";
        const name = document.createElement("h3");
        name.textContent = variant.name;
        const meta = document.createElement("div");
        meta.className = "mini-list__actions";
        meta.append(createBadge(sourceCount ? `${sourceCount} 条来源` : "无来源"));
        if (sourceCount > 0 && sourceCount === maxSourceCount) {
          meta.append(createBadge("来源最多", "success"));
        }
        if (topScore > 0 && topScore === maxTopScore) {
          meta.append(createBadge("首条分最高", "success"));
        }
        row.append(name, meta);

        card.append(
          row,
          createHint(`参数：chunk ${variant.chunk_size} / overlap ${variant.chunk_overlap_ratio} / mode ${variant.retrieval_mode} / dense ${variant.retrieval_k} / bm25 ${variant.bm25_top_k} / boost ${variant.bm25_title_boost} / rrf ${variant.rrf_k} / 重排 ${variant.rerank_k} / temp ${variant.temperature}`),
          createHint(`回答摘要：${result?.answer_preview || "[空]"}`),
        );

        if (result?.sources?.length) {
          card.append(createHint(`首条来源：${result.sources[0].title}`), createSourceMeta(result.sources[0]));
        }
        card.append(createApplyButton(session.session_id, variant.variant_id, variant.name));
        split.appendChild(card);
      });
      panel.appendChild(split);
      questionResults.appendChild(panel);
    });
  }

  function renderResults(session) {
    state.sessionId = session.session_id;
    actionsPanel.hidden = false;
    resultsPanel.hidden = false;
    sessionMeta.textContent = `会话 ${session.session_id.slice(0, 8)}，${session.document_count} 个文件，${session.question_count} 个问题。全量结果可下载为 Excel 或 Word。`;
    renderSummary(session);
    renderByVariant(session);
    renderByQuestion(session);
  }

  function createSourceMeta(source) {
    const meta = document.createElement("div");
    meta.className = "console-tags";
    const mode = source.retrieval_mode || source?.metadata?.retrieval_mode;
    const fusion = source.fusion_sources || source?.metadata?.fusion_sources;
    const chunkId = source.chunk_id || source?.metadata?.chunk_id;
    const chunkIds = source.chunk_ids || source?.metadata?.chunk_ids;
    const citationIndex = source.citation_index || source?.metadata?.citation_index;
    const mergedCount = Number(source.merged_chunk_count || source?.metadata?.merged_chunk_count || 1);
    if (citationIndex) {
      meta.appendChild(createConsoleTag(`来源 ${citationIndex}`));
    }
    if (mode) {
      meta.appendChild(createConsoleTag(mode));
    }
    if (fusion) {
      meta.appendChild(createConsoleTag(`融合 ${fusion}`));
    }
    if (mergedCount > 1) {
      meta.appendChild(createConsoleTag(`合并 ${mergedCount} 段`));
    }
    if (chunkId) {
      meta.appendChild(createConsoleTag(chunkId));
    }
    if (chunkIds && chunkIds !== chunkId) {
      meta.appendChild(createConsoleTag(chunkIds));
    }
    return meta;
  }

  function createConsoleTag(text) {
    const tag = document.createElement("span");
    tag.className = "console-tag";
    tag.textContent = text;
    return tag;
  }

  viewButtons.forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });

  addVariantButton.addEventListener("click", () => addVariantCard());

  exportExcelButton.addEventListener("click", () => {
    if (state.sessionId) {
      window.location.href = `/api/v1/admin/rag/lab/sessions/${encodeURIComponent(state.sessionId)}/export.xlsx`;
    }
  });

  exportWordButton.addEventListener("click", () => {
    if (state.sessionId) {
      window.location.href = `/api/v1/admin/rag/lab/sessions/${encodeURIComponent(state.sessionId)}/export.docx`;
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      if (!filesInput.files?.length) {
        throw new Error("请至少上传一个实验文件。");
      }

      const questions = collectQuestions();
      const variants = collectVariants();

      setRunningState(true);
      setStatus("运行中", "success");
      actionsPanel.hidden = true;
      resultsPanel.hidden = false;
      summaryResults.innerHTML = "";
      variantResults.innerHTML = "";
      questionResults.innerHTML = '<div class="empty-state">实验运行中，问题对比结果将在完成后展示。</div>';
      variantResults.innerHTML = `
        <section class="panel">
          <div class="panel__header">
            <div>
              <p class="panel__eyebrow">Running</p>
              <h2>正在执行实验</h2>
            </div>
          </div>
          <div class="detail-card">
            <p class="hint">系统正在解析上传文件、临时切分内容、生成 embedding、召回片段并完成答案生成，结果不会写入正式知识库。</p>
          </div>
        </section>
      `;

      hint.textContent = `已读取 ${filesInput.files.length} 个文件、${questions.length} 个问题，正在启动 ${variants.length} 套方案实验。`;

      const payload = new FormData();
      payload.append("questions", questions.join("\n"));
      payload.append("variants", JSON.stringify(variants));
      Array.from(filesInput.files).forEach((file) => payload.append("files", file));

      const response = await fetch("/api/v1/admin/rag/lab/run-upload", {
        method: "POST",
        body: payload,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${response.status}`);
      }

      const session = await response.json();
      renderResults(session);
      switchView("variant");
      const elapsed = Math.max(1, Math.round((Date.now() - state.startedAt) / 1000));
      hint.textContent = `实验完成：${session.document_count} 个文件，${session.question_count} 个问题，${session.variants.length} 套方案，耗时约 ${elapsed} 秒。`;
      setStatus("已完成", "success");
    } catch (error) {
      actionsPanel.hidden = true;
      resultsPanel.hidden = false;
      summaryResults.innerHTML = "";
      variantResults.innerHTML = "";
      questionResults.innerHTML = "";
      hint.textContent = `实验失败：${error.message}`;
      setStatus("失败", "error");
    } finally {
      setRunningState(false);
    }
  });

  questionsInput.value = [
    "在神雕侠侣中，比较激情的战斗有哪些场",
    "知识库导入进度里的 chunks 64/1742 表示什么",
  ].join("\n");

  addVariantCard(VARIANT_PRESETS[0]);
  addVariantCard(VARIANT_PRESETS[1]);
})();
