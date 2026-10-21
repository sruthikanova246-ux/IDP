"use strict";

// Score bars: widths come from data attributes (inline styles are blocked by the CSP).
document.querySelectorAll(".bar-fill[data-width]").forEach((el) => {
  el.style.width = Math.max(0, Math.min(100, parseFloat(el.dataset.width) || 0)) + "%";
});

// Confirmation before destructive form submissions.
document.querySelectorAll("form[data-confirm]").forEach((form) => {
  form.addEventListener("submit", (e) => {
    if (!window.confirm(form.dataset.confirm)) e.preventDefault();
  });
});

// Model status: poll while loading so the badge reflects reality.
(function modelStatus() {
  const badges = document.querySelectorAll("[data-model-badge]");
  if (!badges.length) return;
  const labels = { ready: "AI model ready", loading: "AI model loading…", not_loaded: "AI model not loaded yet",
                   failed: "AI model unavailable", disabled: "AI detection disabled" };
  let wasLoading = false;
  async function poll() {
    try {
      const res = await fetch("/api/model-status");
      const s = await res.json();
      badges.forEach((b) => { b.textContent = labels[s.state] || s.state; b.className = "badge badge-" + s.state; });
      if (s.state === "loading") { wasLoading = true; setTimeout(poll, 3000); }
      else if (wasLoading && document.querySelector("[data-model-detail]")) window.location.reload();
    } catch (_) { /* status is informational only */ }
  }
  poll();
  const retry = document.querySelector("[data-model-retry]");
  if (retry) retry.addEventListener("click", async () => {
    retry.disabled = true;
    await fetch(retry.dataset.url, { method: "POST", headers: { "X-CSRF-Token": retry.dataset.csrf } });
    poll();
  });
})();

// Upload page: client-side checks, preview, upload progress and analysis wait state.
(function upload() {
  const form = document.getElementById("upload-form");
  if (!form) return;
  const input = document.getElementById("evidence");
  const err = document.getElementById("file-error");
  const preview = document.getElementById("preview");
  const previewImg = document.getElementById("preview-img");
  const previewMeta = document.getElementById("preview-meta");
  const btn = document.getElementById("submit-btn");
  const progress = document.getElementById("progress");
  const fill = document.getElementById("progress-fill");
  const text = document.getElementById("progress-text");
  const maxBytes = parseInt(form.dataset.maxBytes, 10);
  const allowed = form.dataset.allowed.split(",");
  let objectUrl = null;

  function showError(msg) { err.textContent = msg; err.hidden = !msg; }

  function check(file) {
    if (!file) return "Choose an image file to analyse.";
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (!file.name.includes(".") || !allowed.includes(ext)) return "Unsupported file type. Allowed: " + allowed.join(", ") + ".";
    if (file.size === 0) return "The selected file is empty.";
    if (file.size > maxBytes) return "File is too large (" + (file.size / 1048576).toFixed(1) + " MB). Maximum is " + (maxBytes / 1048576) + " MB.";
    return "";
  }

  input.addEventListener("change", () => {
    const file = input.files[0];
    const problem = check(file);
    showError(problem);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    preview.hidden = true;
    if (!file || problem) return;
    objectUrl = URL.createObjectURL(file);
    previewImg.onerror = () => { previewMeta.textContent = file.name + " - the browser cannot preview this format; the server will still analyse it."; };
    previewImg.src = objectUrl;
    previewMeta.textContent = file.name + " - " + (file.size / 1024).toFixed(1) + " KB";
    preview.hidden = false;
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const file = input.files[0];
    const problem = check(file);
    showError(problem);
    if (problem) { input.focus(); return; }

    const xhr = new XMLHttpRequest();
    xhr.open("POST", form.action);
    xhr.setRequestHeader("X-Requested-With", "fetch");
    xhr.responseType = "json";
    btn.disabled = true;
    progress.hidden = false;
    fill.style.width = "0%";
    text.textContent = "Uploading…";

    xhr.upload.onprogress = (ev) => {
      if (!ev.lengthComputable) return;
      const p = Math.round((ev.loaded / ev.total) * 100);
      fill.style.width = p + "%";
      text.textContent = "Uploading… " + p + "%";
    };
    xhr.upload.onload = () => {
      fill.style.width = "100%";
      text.textContent = "Upload complete. Analysing: hashing, extracting metadata, running the AI model… " +
                         "(the first analysis can take longer while the model loads)";
    };
    xhr.onload = () => {
      const data = xhr.response || {};
      if (xhr.status === 200 && data.ok) { window.location.href = data.redirect; return; }
      btn.disabled = false;
      progress.hidden = true;
      showError(data.error || ("The server rejected the upload (HTTP " + xhr.status + ")."));
    };
    xhr.onerror = () => {
      btn.disabled = false;
      progress.hidden = true;
      showError("Network error: the server could not be reached.");
    };
    xhr.send(new FormData(form));
  });
})();
