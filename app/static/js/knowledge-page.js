const state = {
  lastIngestedIds: new Set(),
  deletingIds: new Set(),
  activeJobId: null,
};

const STORAGE_KEY = "chat_robot_active_ingest_job_id";
const knowledgeForm = document.getElementById("knowledge-form");
const knowledgeFiles = document.getElementById("knowledge-files");
const knowledgeTitle = document.getElementById("knowledge-title");
const knowledgeContent = document.getElementById("knowledge-content");
const knowledgeSubmit = document.getElementById("knowledge-submit");
const knowledgeHint = document.getElementById("knowledge-hint");
const documentList = document.getElementById("document-list");
const refreshDocumentsButton = document.getElementById("refresh-documents");
const knowledgeCount = document.getElementById("knowledge-count");
const ingestProgress = document.getElementById("ingest-progress");
const ingestStage = document.getElementById("ingest-stage");
const ingestPercent = document.getElementById("ingest-percent");
const ingestFill = document.getElementById("ingest-fill");
const ingestMeta = document.getElementById("ingest-meta");
const ingestStatusInline = document.getElementById("ingest-status-inline");
const ingestPercentInline = document.getElementById("ingest-percent-inline");

const INGEST_POLL_INTERVAL_QUEUED_MS = 3000;
const INGEST_POLL_INTERVAL_RUNNING_MS = 5000;

function rememberActiveJob(jobId) {
  state.activeJobId = jobId;
  if (jobId) {
    window.localStorage.setItem(STORAGE_KEY, jobId);
  } else {
    window.localStorage.removeItem(STORAGE_KEY);
  }
}

function resetIngestProgress() {
  ingestProgress.hidden = true;
  ingestStage.textContent = "排队中";
  ingestPercent.textContent = "0%";
  ingestFill.style.width = "0%";
  ingestMeta.textContent = "等待任务开始。";
  ingestStatusInline.textContent = "待命";
  ingestPercentInline.textContent = "0%";
}

function calculateIngestPercent(data) {
  if (data.status === "completed") {
    return 100;
  }
  if (typeof data.total_chunks === "number" && data.total_chunks > 0) {
    return Math.min(99, Math.round(((data.processed_chunks ?? 0) / data.total_chunks) * 100));
  }
  if (typeof data.submitted_documents === "number" && data.submitted_documents > 0) {
    return Math.min(
      99,
      Math.round(((data.processed_documents ?? 0) / data.submitted_documents) * 100),
    );
  }
  return data.status === "running" ? 5 : 0;
}

function formatIngestStage(stage, status) {
  if (status === "queued") {
    return "排队中";
  }
  const labels = {
    preparing: "准备中",
    chunking: "切片中",
    processing: "处理中",
    embedding: "向量化中",
    upserting: "写入知识库中",
    completed: "已完成",
    failed: "失败",
  };
  return labels[stage] || "执行中";
}

function updateIngestProgress(data) {
  ingestProgress.hidden = false;

  const stageLabel = formatIngestStage(data.current_stage, data.status);
  const percentValue = calculateIngestPercent(data);
  ingestStage.textContent = stageLabel;
  ingestPercent.textContent = `${percentValue}%`;
  ingestFill.style.width = `${percentValue}%`;
  ingestStatusInline.textContent = stageLabel;
  ingestPercentInline.textContent = `${percentValue}%`;

  const metaParts = [];
  if (data.current_title) {
    metaParts.push(`当前文档：${data.current_title}`);
  }
  if (typeof data.total_chunks === "number" && data.total_chunks > 0) {
    metaParts.push(`chunks：${data.processed_chunks ?? 0}/${data.total_chunks}`);
  } else {
    metaParts.push(`文档：${data.processed_documents ?? 0}/${data.submitted_documents ?? 0}`);
  }
  ingestMeta.textContent = metaParts.join("，") || "等待任务开始。";
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error(`读取文件失败：${file.name}`));
    reader.readAsText(file, "utf-8");
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildDocumentItem(entry) {
  const item = document.createElement("article");
  item.className = "mini-list__item";
  if (state.lastIngestedIds.has(entry.document_id)) {
    item.style.borderColor = "rgba(25, 211, 255, 0.55)";
    item.style.boxShadow = "inset 0 0 0 1px rgba(25, 211, 255, 0.22)";
  }

  const row = document.createElement("div");
  row.className = "mini-list__row";

  const titleWrap = document.createElement("div");
  titleWrap.className = "mini-list__meta-group";

  const title = document.createElement("strong");
  title.textContent = entry.title;

  const id = document.createElement("span");
  id.className = "hint";
  id.textContent = entry.document_id;

  titleWrap.append(title, id);

  const actions = document.createElement("div");
  actions.className = "mini-list__actions";

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.className = "button button--ghost button--small";
  deleteButton.textContent = state.deletingIds.has(entry.document_id) ? "删除中..." : "删除";
  deleteButton.disabled = state.deletingIds.has(entry.document_id);
  deleteButton.addEventListener("click", async () => {
    const confirmed = window.confirm(`确定删除文档“${entry.title}”吗？`);
    if (!confirmed) {
      return;
    }
    await deleteDocument(entry.document_id, entry.title);
  });

  actions.appendChild(deleteButton);
  row.append(titleWrap, actions);
  item.appendChild(row);
  return item;
}

function renderDocuments(documents) {
  knowledgeCount.textContent = String(documents.length);
  if (!documents.length) {
    documentList.innerHTML = '<div class="empty-state">还没有已入库文档。</div>';
    return;
  }

  documentList.innerHTML = "";
  const ordered = [...documents].reverse();
  for (const entry of ordered) {
    documentList.appendChild(buildDocumentItem(entry));
  }
}

async function refreshDocuments() {
  try {
    const response = await fetch("/api/v1/knowledge/documents");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const documents = await response.json();
    renderDocuments(documents);
  } catch (error) {
    documentList.innerHTML = `<div class="empty-state">加载文档失败：${error.message}</div>`;
  }
}

async function deleteDocument(documentId, title) {
  state.deletingIds.add(documentId);
  knowledgeHint.textContent = `正在删除文档：${title}`;
  await refreshDocuments();

  try {
    const response = await fetch(`/api/v1/knowledge/documents/${encodeURIComponent(documentId)}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.lastIngestedIds.delete(documentId);
    knowledgeHint.textContent = `文档已删除：${title}`;
  } catch (error) {
    knowledgeHint.textContent = `删除失败：${error.message}`;
  } finally {
    state.deletingIds.delete(documentId);
    await refreshDocuments();
  }
}

async function fetchJobStatus(jobId) {
  const response = await fetch(`/api/v1/knowledge/ingest/${jobId}`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function waitForIngestJob(jobId) {
  rememberActiveJob(jobId);

  while (true) {
    const data = await fetchJobStatus(jobId);
    updateIngestProgress(data);

    if (data.status === "completed") {
      rememberActiveJob(null);
      return data;
    }
    if (data.status === "failed") {
      rememberActiveJob(null);
      return data;
    }

    knowledgeHint.textContent = `导入任务 ${jobId.slice(0, 8)} ${formatIngestStage(
      data.current_stage,
      data.status,
    )}`;

    await sleep(
      data.status === "running"
        ? INGEST_POLL_INTERVAL_RUNNING_MS
        : INGEST_POLL_INTERVAL_QUEUED_MS,
    );
  }
}

async function restoreActiveJob() {
  const rememberedJobId = window.localStorage.getItem(STORAGE_KEY);
  if (rememberedJobId) {
    try {
      const data = await fetchJobStatus(rememberedJobId);
      updateIngestProgress(data);
      if (data.status === "queued" || data.status === "running") {
        knowledgeHint.textContent = `已恢复导入任务 ${rememberedJobId.slice(0, 8)} 的进度显示。`;
        await waitForIngestJob(rememberedJobId);
        await refreshDocuments();
        return;
      }
      rememberActiveJob(null);
    } catch (error) {
      rememberActiveJob(null);
    }
  }

  try {
    const response = await fetch("/api/v1/knowledge/ingest/latest");
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    updateIngestProgress(data);
    if (data.status === "queued" || data.status === "running") {
      knowledgeHint.textContent = `已恢复最近未完成导入任务 ${data.job_id.slice(0, 8)} 的进度显示。`;
      await waitForIngestJob(data.job_id);
      await refreshDocuments();
    }
  } catch (error) {
    // Ignore restore errors; page can still be used normally.
  }
}

async function ingestKnowledge(event) {
  event.preventDefault();
  knowledgeSubmit.disabled = true;
  resetIngestProgress();
  knowledgeHint.textContent = "正在整理上传内容并创建导入任务，请稍等...";

  try {
    const documents = [];
    const files = Array.from(knowledgeFiles.files ?? []);

    for (const file of files) {
      const content = (await readFileAsText(file)).trim();
      if (!content) {
        continue;
      }
      documents.push({
        title: file.name,
        content,
        metadata: {
          source: "upload",
          file_name: file.name,
        },
      });
    }

    const manualTitle = knowledgeTitle.value.trim();
    const manualContent = knowledgeContent.value.trim();
    if (manualContent) {
      documents.push({
        title: manualTitle || `manual-${new Date().toISOString()}`,
        content: manualContent,
        metadata: {
          source: "manual",
        },
      });
    }

    if (!documents.length) {
      throw new Error("请先选择文件或输入知识内容。");
    }

    const totalChars = documents.reduce((sum, entry) => sum + entry.content.length, 0);
    knowledgeHint.textContent = `正在提交 ${documents.length} 篇内容，累计 ${totalChars} 个字符。`;

    const response = await fetch("/api/v1/knowledge/ingest", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ documents }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    rememberActiveJob(data.job_id);
    knowledgeHint.textContent = `导入任务 ${data.job_id.slice(0, 8)} 已创建，等待后台处理。`;
    updateIngestProgress(data);

    const finalStatus = await waitForIngestJob(data.job_id);
    if (finalStatus.status !== "completed") {
      throw new Error(finalStatus.error || "知识库导入失败。");
    }

    updateIngestProgress(finalStatus);
    state.lastIngestedIds = new Set(finalStatus.document_ids ?? []);
    await refreshDocuments();
    knowledgeHint.textContent = `导入成功：${finalStatus.ingested_count} 篇，当前共 ${finalStatus.total_documents ?? "-"} 篇文档。`;
    knowledgeFiles.value = "";
    knowledgeTitle.value = "";
    knowledgeContent.value = "";
  } catch (error) {
    rememberActiveJob(null);
    knowledgeHint.textContent = `导入失败：${error.message}`;
    ingestProgress.hidden = false;
    ingestStage.textContent = "失败";
    ingestPercent.textContent = "0%";
    ingestFill.style.width = "0%";
    ingestMeta.textContent = error.message;
    ingestStatusInline.textContent = "失败";
    ingestPercentInline.textContent = "0%";
  } finally {
    knowledgeSubmit.disabled = false;
  }
}

knowledgeForm.addEventListener("submit", ingestKnowledge);
refreshDocumentsButton.addEventListener("click", refreshDocuments);

resetIngestProgress();
refreshDocuments();
restoreActiveJob();
