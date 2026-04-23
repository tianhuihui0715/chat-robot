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

function statusBadgeClass(status) {
  if (status === "completed" || status === "ok") {
    return "badge badge--success";
  }
  if (status === "error" || status === "failed") {
    return "badge badge--error";
  }
  return "badge badge--neutral";
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

window.AdminCommon = {
  fetchJson,
  formatDateTime,
  statusBadgeClass,
};
