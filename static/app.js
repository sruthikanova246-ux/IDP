"use strict";
/* Plain JS, no dependencies. Components: score bars, copy + toast, tabs, confirm dialog,
   model status, and the upload dropzone with preview and progress stepper. */

// Score bars: widths come from data attributes (inline styles are blocked by the CSP).
document.querySelectorAll(".bar-fill[data-width]").forEach((el) => {
  el.style.width = Math.max(0, Math.min(100, parseFloat(el.dataset.width) || 0)) + "%";
});

// Toast
const toast = document.getElementById("toast");
let toastTimer;
function showToast(text) {
  if (!toast) return;
  toast.textContent = text;
  toast.classList.add("is-on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("is-on"), 1800);
}

// Copy fields
document.querySelectorAll("[data-copy]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const input = document.getElementById(btn.dataset.copy);
    try {
      await navigator.clipboard.writeText(input.value);
    } catch (_) {
      input.select();
      document.execCommand("copy");
    }
    showToast("Copied to clipboard");
  });
});

// Tabs (WAI-ARIA pattern: arrow keys move between tabs)
document.querySelectorAll("[data-tabs]").forEach((root) => {
  const tabs = [...root.querySelectorAll("[role=tab]")];
  function select(tab, focus) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    if (focus) tab.focus();
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(tab, false));
    tab.addEventListener("keydown", (e) => {
      const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
      if (next === undefined) return;
      e.preventDefault();
      select(tabs[(next + tabs.length) % tabs.length], true);
    });
  });
});

// Confirm dialog for destructive forms
const dialog = document.getElementById("confirm-dialog");
document.querySelectorAll("form[data-confirm]").forEach((form) => {
  form.addEventListener("submit", (e) => {
    if (form.dataset.confirmed) return;
    e.preventDefault();
    if (!dialog || typeof dialog.showModal !== "function") {
      if (window.confirm("Delete this record and its stored files? This cannot be undone.")) form.submit();
      return;
    }
    dialog.showModal();
    const ok = dialog.querySelector("[data-dialog-ok]");
    const cancel = dialog.querySelector("[data-dialog-cancel]");
    const close = () => { dialog.close(); ok.onclick = cancel.onclick = null; };
    cancel.onclick = close;
    ok.onclick = () => { close(); form.dataset.confirmed = "1"; form.submit(); };
    cancel.focus();
  });
});

// Model status: refresh badges while the model is loading.
(function modelStatus() {
  const badges = document.querySelectorAll("[data-model-badge]");
  if (!badges.length) return;
  const labels = { ready: "Model ready", loading: "Model loading…", not_loaded: "Model not loaded yet",
                   failed: "Model unavailable", disabled: "AI detection off" };
  const tones = { ready: "is-ok", loading: "is-warn", not_loaded: "is-warn", failed: "is-bad", disabled: "" };
  async function poll() {
    try {
      const s = await (await fetch("/api/model-status")).json();
      badges.forEach((b) => { b.textContent = labels[s.state] || s.state; b.className = "badge " + (tones[s.state] || ""); });
      if (s.state === "loading") setTimeout(poll, 3000);
    } catch (_) { /* informational only */ }
  }
  poll();
  const retry = document.querySelector("[data-model-retry]");
  if (retry) retry.addEventListener("click", async () => {
    retry.disabled = true;
    await fetch(retry.dataset.url, { method: "POST", headers: { "X-CSRF-Token": retry.dataset.csrf } });
    poll();
  });
})();

// Upload: dropzone, client checks, preview, upload progress and analysis stepper.
(function upload() {
  const form = document.getElementById("upload-form");
  if (!form) return;
  const input = document.getElementById("evidence");
  const zone = document.getElementById("dropzone");
  const selected = document.getElementById("selected");
  const img = document.getElementById("preview-img");
  const name = document.getElementById("preview-name");
  const err = document.getElementById("file-error");
  const btn = document.getElementById("submit-btn");
  const clear = document.getElementById("clear-btn");
  const progress = document.getElementById("progress");
  const bar = document.getElementById("progress-bar");
  const fill = document.getElementById("progress-fill");
  const text = document.getElementById("progress-text");
  const steps = [...progress.querySelectorAll("[data-step]")];
  const maxBytes = parseInt(form.dataset.maxBytes, 10);
  const allowed = form.dataset.allowed.split(",");
  let url = null;

  const showError = (msg) => { err.textContent = msg; err.hidden = !msg; };
  const setStep = (key) => {
    let done = true;
    steps.forEach((li) => {
      const active = li.dataset.step === key;
      if (active) done = false;
      li.classList.toggle("is-active", active);
      li.classList.toggle("is-done", done && !active);
    });
  };
  const setProgress = (p, msg) => {
    fill.style.width = p + "%";
    bar.setAttribute("aria-valuenow", String(Math.round(p)));
    if (msg) text.textContent = msg;
  };

  function check(file) {
    if (!file) return "Choose an image file.";
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (!file.name.includes(".") || !allowed.includes(ext)) return "Unsupported file type. Allowed: " + allowed.join(", ") + ".";
    if (file.size === 0) return "The selected file is empty.";
    if (file.size > maxBytes) return "File is too large (" + (file.size / 1048576).toFixed(1) + " MB). Maximum is " + (maxBytes / 1048576) + " MB.";
    return "";
  }

  function show(file) {
    const problem = check(file);
    showError(problem);
    if (url) URL.revokeObjectURL(url);
    if (problem) { selected.hidden = true; zone.hidden = false; return; }
    url = URL.createObjectURL(file);
    img.onerror = () => { img.removeAttribute("src"); img.alt = "No browser preview for this format"; };
    img.src = url;
    name.textContent = file.name + " · " + (file.size / 1024).toFixed(0) + " KB";
    zone.hidden = true;
    selected.hidden = false;
    btn.focus();
  }

  input.addEventListener("change", () => show(input.files[0]));
  clear.addEventListener("click", () => { input.value = ""; selected.hidden = true; zone.hidden = false; showError(""); input.focus(); });

  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("is-over"); }));
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    show(file);
  });

  function reset(msg) {
    fill.classList.remove("is-indeterminate");
    bar.removeAttribute("aria-busy");
    btn.disabled = false; clear.disabled = false;
    progress.hidden = true;
    showError(msg);
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const problem = check(input.files[0]);
    if (problem) { showError(problem); return; }
    showError("");
    btn.disabled = true; clear.disabled = true;
    progress.hidden = false;
    setStep("upload"); setProgress(0, "Uploading…");

    const xhr = new XMLHttpRequest();
    xhr.open("POST", form.action);
    xhr.setRequestHeader("X-Requested-With", "fetch");
    xhr.responseType = "json";
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) setProgress((ev.loaded / ev.total) * 100, "Uploading… " + Math.round((ev.loaded / ev.total) * 100) + "%");
    };
    // The server analyses in a single request and does not report progress, so this step is
    // shown as indeterminate rather than with invented percentages.
    xhr.upload.onload = () => {
      setStep("analyse");
      bar.removeAttribute("aria-valuenow");
      bar.setAttribute("aria-busy", "true");
      fill.classList.add("is-indeterminate");
      text.textContent = "Analysing: hash, metadata, AI model and report. The first analysis can take longer while the model loads.";
    };
    xhr.onload = () => {
      const data = xhr.response || {};
      fill.classList.remove("is-indeterminate");
      bar.removeAttribute("aria-busy");
      if (xhr.status === 200 && data.ok) { setStep("done"); setProgress(100, "Done. Opening the result…"); window.location.href = data.redirect; return; }
      reset(data.error || ("The server rejected the upload (HTTP " + xhr.status + ")."));
    };
    xhr.onerror = () => reset("Network error: the server could not be reached.");
    xhr.send(new FormData(form));
  });
})();
