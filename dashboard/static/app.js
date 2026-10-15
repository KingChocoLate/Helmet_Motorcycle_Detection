"use strict";
const $ = (id) => document.getElementById(id);
let activeId = null, timer = null, stream = null, generation = 0, eventSignature = "", lastJob = null;
let modelChanging = false, submitting = false;
let showProcessingPreview = false;
let cameraId = null;
let modelReady = false, loadedRun = null, failedVideoId = null;

function notice(message = "") { $("notice").textContent = message; $("notice").hidden = !message; }
async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : `Request failed (${response.status}).`);
  }
  return response.json();
}
function fileUrl(id, filename) { return `/api/jobs/${id}/files/${encodeURIComponent(filename)}`; }
function options() { return {display_conf: Number($("display-conf").value), candidate_conf: Number($("candidate-conf").value)}; }
function clock(seconds) { return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2,"0")}`; }
function cell(tag, text, className) { const node = document.createElement(tag); node.textContent = text; if (className) node.className = className; return node; }

for (const name of ["display", "candidate"]) $(`${name}-conf`).addEventListener("input", () => {
  $(`${name}-value`).textContent = Number($(`${name}-conf`).value).toFixed(2);
});
$("mode").addEventListener("change", () => {
  const mode = $("mode").value;
  $("source").value = "";
  $("source").accept = mode === "image" ? "image/*" : "video/mp4,video/quicktime,video/x-msvideo";
  $("file-control").hidden = mode === "camera";
  $("camera-hint").hidden = mode !== "camera";
  $("file-hint").textContent = mode === "image" ? "JPG, PNG or WebP · up to 16 MB" : "Video · up to 250 MB and 15 minutes";
  $("run").textContent = mode === "camera" ? "Start camera ↗" : `Analyze ${mode} ↗`;
});

async function history() {
  const runs = await api("/api/jobs");
  $("history").replaceChildren();
  if (!runs.length) $("history").append(cell("p", "No saved runs yet.", "hint"));
  for (const run of runs) {
    const button = cell("button", run.filename);
    button.setAttribute("aria-current", String(run.id === activeId));
    button.append(cell("small", `${run.status} · ${new Date(run.created).toLocaleDateString()}`));
    button.addEventListener("click", async () => {
      if (submitting) return;
      try { const token = ++generation; await leaveCamera(); activeId = run.id; eventSignature = ""; await poll(token); }
      catch (error) { notice(error.message); }
    });
    $("history").append(button);
  }
}

async function leaveCamera() {
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
  const id = cameraId; cameraId = null;
  if (id) await api(`/api/jobs/${id}/stop`, {method:"POST"});
  clearTimeout(timer);
}

function renderEvents(job) {
  const signature = JSON.stringify([job.id, job.events, Boolean(job.browser_video)]);
  if (signature === eventSignature) return;
  eventSignature = signature;
  $("event-count").textContent = job.events.length;
  $("events").replaceChildren();
  const list = job.events;
  if (!list.length) $("events").append(cell("p", "No captures yet.", "empty-list"));
  for (const event of list) {
    const row = cell("article", "", "event capture-card"), thumb = cell("button", "", "thumb"), image = document.createElement("img");
    const isCrop = event.capture_mode === "nohelmet_crop";
    const title = ["motorcycle_nohelmet_crop", "tracked_motorcycle_nohelmet"].includes(event.capture_mode) ? "Motorcycle · NoHelmet" : isCrop ? "NoHelmet crop" : "Earlier capture";
    image.src = fileUrl(job.id, event.snapshot); image.alt = `${title} at ${clock(event.timestamp_seconds)}`; image.loading = "lazy"; thumb.append(image);
    thumb.addEventListener("click", () => {
      const dialog = $("snapshot-dialog");
      // A larger, centered inspector for evidence captured from moving footage.
      dialog.classList.toggle("video-capture", job.kind === "video" || job.kind === "camera");
      $("snapshot").src = image.src;
      $("snapshot-caption").textContent = ` ${title} · source frame ${event.frame_index} · ${clock(event.timestamp_seconds)} · confidence ${event.confidence.toFixed(2)}`.trim();
      dialog.showModal();
    });
    const detail = cell("div", "", "capture-detail");
    detail.append(cell("h3", `${title} · ${clock(event.timestamp_seconds)}`), cell("p", `Confidence ${event.confidence.toFixed(2)} · frame ${event.frame_index}${event.track_id != null ? " · Track #" + event.track_id : ""}${event.nohelmet_count ? " · " + event.nohelmet_count + " uncovered head(s)" : ""}`));
    if (job.kind === "video" && job.browser_video) {
      const jump = cell("button", "View this moment", "jump-link");
      jump.addEventListener("click", () => {
        const video = $("result-video");
        video.pause(); video.currentTime = event.timestamp_seconds;
        $("viewer-surface").scrollIntoView({behavior:"smooth", block:"center"});
      });
      detail.append(jump);
    }
    const save = cell("a", "Save image", "secondary capture-save");
    save.href = `${fileUrl(job.id, event.snapshot)}?enlarge=1`;
    save.download = `capture-${job.id.slice(0,8)}-${event.id}.jpg`;
    save.setAttribute("aria-label", `Save capture at ${clock(event.timestamp_seconds)}`);
    row.append(thumb, detail, save); $("events").append(row);
  }
}

function render(job) {
  lastJob = job;
  const video = $("result-video");
  if (loadedRun !== job.id) {
    video.pause(); video.removeAttribute("src"); video.load();
    loadedRun = job.id; failedVideoId = null;
    $("preview").removeAttribute("src");
    notice();
  }
  $("run-name").textContent = job.filename;
  $("run-context").hidden = false;
  const inferenceDevice = job.model?.device || job.inference_device;
  const deviceLabel = inferenceDevice ? `Inference: ${String(inferenceDevice).toUpperCase()}` : "Inference device unavailable";
  $("run-context").textContent = `${job.model?.checkpoint || "Unknown checkpoint"} · display ${job.display_conf.toFixed(2)} · capture ${job.candidate_conf.toFixed(2)} · ${deviceLabel}`;
  $("capture-rule").textContent = job.capture_rule
    ? job.capture_rule
    : "Earlier run · one-second capture interval";
  $("review-captures").hidden = !job.events.length;
  $("review-captures").textContent = `View ${job.events.length} captures ↓`;
  for (const id of ["mode", "source", "display-conf", "candidate-conf"]) $(id).disabled = ["queued", "running", "live"].includes(job.status) || submitting;
  $("run-state").textContent = job.phase === "encoding" ? "Preparing playback" : job.status;
  const active = ["queued", "running", "live"].includes(job.status);
  $("stop").disabled = !active;
  $("run").disabled = !modelReady || active || modelChanging || submitting;
  updateModelButtons(active);
  const processingVideo = job.kind === "video" && active;
  const previewHidden = processingVideo && !showProcessingPreview;
  $("toggle-preview").hidden = !processingVideo;
  $("toggle-preview").textContent = showProcessingPreview ? "Hide preview" : "Show preview";
  $("toggle-preview").setAttribute("aria-pressed", String(showProcessingPreview));
  $("processing-placeholder").hidden = !processingVideo || (showProcessingPreview && job.preview);
  $("processing-message").textContent = job.phase === "encoding" ? "Preparing playback…" : "Preview hidden · analysis continues";
  $("empty").hidden = Boolean(job.preview) || processingVideo;
  const playable = ["completed", "cancelled"].includes(job.status) && job.browser_video && failedVideoId !== job.id;
  $("preview").hidden = !job.preview || playable || previewHidden;
  video.hidden = !playable;
  $("expand-view").disabled = !job.preview || previewHidden;
  $("frame-summary").hidden = job.kind === "video";
  if (job.preview && !playable && !previewHidden) {
    const url = `${fileUrl(job.id, "preview.jpg")}?frame=${job.processed_frames}`;
    const previous = $("preview").getAttribute("src");
    // Refresh only when the server publishes another processed preview.
    if (previous !== url) $("preview").src = url;
  }
  if (playable) {
    const url = fileUrl(job.id, job.video);
    video.removeAttribute("poster");
    // Only set src when changing runs: progress updates must never reset playback.
    if (video.getAttribute("src") !== url) video.src = url;
  } else { video.pause(); }
  $("view-hint").textContent = job.kind === "video" && active
    ? (job.phase === "encoding" ? "Preparing playback…" : showProcessingPreview ? "Processing preview" : "Processing · preview hidden")
    : playable ? (job.status === "cancelled"
      ? "Stopped · partial video ready"
      : "Ready to play")
    : "Image preview";
  const remaining = active && Number.isFinite(job.estimated_remaining_seconds) ? ` · about ${clock(job.estimated_remaining_seconds)} left` : "";
  const elapsed = !active && job.service_seconds ? ` · finished in ${clock(job.service_seconds)}` : "";
  $("progress").textContent = `${job.processed_frames}${job.total_frames ? " / " + job.total_frames : ""} frames${job.processed_fps ? " · " + job.processed_fps + " processed fps" : ""}${remaining}${elapsed}`;
  $("progress-bar").hidden = !active || job.kind !== "video";
  if (job.total_frames) $("progress-bar").value = 100 * job.processed_frames / job.total_frames;
  else $("progress-bar").removeAttribute("value");
  $("download-video").hidden = !job.video;
  if (job.video) { $("download-video").href = fileUrl(job.id, job.video); $("download-video").download = "annotated.mp4"; }
  $("export").hidden = active; $("export").href = `/api/jobs/${job.id}/export`;
  const counts = [0,0,0];
  for (const detection of job.detections) if (detection.confidence >= job.display_conf) counts[detection.class_id]++;
  document.querySelectorAll("#counts strong").forEach((node,index) => node.textContent = counts[index]);
  renderEvents(job);
  if (job.error || job.warning) notice(job.error || job.warning);
}

async function poll(token) {
  clearTimeout(timer);
  try {
    const id = activeId, job = await api(`/api/jobs/${id}`);
    if (token !== generation || id !== activeId) return;
    render(job);
    if (["queued", "running"].includes(job.status)) timer = setTimeout(() => poll(token), 700);
    else if (job.status === "live") timer = setTimeout(() => poll(token), 1000);
    else await history();
  } catch (error) { if (token !== generation) return; notice(error.message); timer = setTimeout(() => poll(token), 2000); }
}

$("run").addEventListener("click", async () => {
  submitting = true; for (const id of ["mode", "source", "display-conf", "candidate-conf"]) $(id).disabled = true; notice(); $("run").disabled = true; $("result-video").pause();
  const token = ++generation;
  try {
    await leaveCamera();
    eventSignature = "";
    if ($("mode").value === "camera") { await startCamera(token); return; }
    const source = $("source").files[0];
    if (!source) throw new Error("Choose a file first.");
    const body = new FormData(); body.append("file", source); body.append("kind", $("mode").value);
    for (const [name,value] of Object.entries(options())) body.append(name, value);
    $("run-state").textContent = "Uploading…";
    const result = await api("/api/jobs", {method:"POST", body}); activeId = result.id;
    submitting = false; await poll(token);
  } catch (error) { notice(error.message); $("run-state").textContent = lastJob?.status || "Ready"; }
  finally { submitting = false; for (const id of ["mode", "source", "display-conf", "candidate-conf"]) $(id).disabled = ["queued", "running", "live"].includes(lastJob?.status); $("run").disabled = !modelReady || ["queued", "running", "live"].includes(lastJob?.status); }
});

$("stop").addEventListener("click", async () => {
  try { const id = activeId; ++generation; await leaveCamera(); if (id) await api(`/api/jobs/${id}/stop`, {method:"POST"}); await poll(generation); }
  catch (error) { notice(error.message); }
});

async function startCamera(token) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access requires localhost or HTTPS in a supported browser.");
  stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:"environment"},width:{ideal:1280},height:{ideal:720}},audio:false});
  try {
    const result = await api("/api/camera", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(options())});
    activeId = result.id; cameraId = result.id; submitting = false;
    if (token !== generation) { await leaveCamera(); return; }
    const camera = $("camera-source"); camera.srcObject = stream; await camera.play();
    const canvas = $("camera-canvas"), context = canvas.getContext("2d"); let sequence = 0;
    while (stream && token === generation) {
      canvas.width = camera.videoWidth; canvas.height = camera.videoHeight;
      context.drawImage(camera, 0, 0);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.85));
      if (!blob || token !== generation) break;
      const body = new FormData(); body.append("file", blob, "frame.jpg"); body.append("sequence", sequence++);
      const job = await api(`/api/camera/${activeId}/frame`, {method:"POST",body});
      if (token === generation) render(job);
      // Awaiting each response supplies backpressure; no overlapping frame requests.
    }
  } catch (error) { await leaveCamera(); if (activeId) await api(`/api/jobs/${activeId}/stop`, {method:"POST"}).catch(() => {}); throw error; }
}
$("close-dialog").addEventListener("click", () => $("snapshot-dialog").close());
$("result-video").addEventListener("error", () => {
  if (!video.getAttribute("src")) return;
  failedVideoId = lastJob?.id;
  $("result-video").hidden = true;
  if (lastJob?.preview) {
    $("preview").hidden = false; $("preview").src = fileUrl(lastJob.id, "preview.jpg");
  }
  notice("This browser cannot play the encoded video. The final frame and video download remain available.");
});
window.addEventListener("pagehide", () => {
  if (stream) { stream.getTracks().forEach(track => track.stop()); if (activeId) fetch(`/api/jobs/${activeId}/stop`, {method:"POST",keepalive:true}); }
});

const video = $("result-video"), surface = $("viewer-surface");

function expanded() { return document.fullscreenElement === surface || surface.classList.contains("expanded"); }
function updateExpandButton() {
  const open = expanded();
  $("expand-view").textContent = open ? "Close full display ×" : "Full display ⛶";
  $("expand-view").setAttribute("aria-pressed", String(open));
  $("expand-view").setAttribute("aria-label", open ? "Close full display" : "Enlarge footage");
}
async function toggleExpand() {
  if (!lastJob?.preview) return;
  if (document.fullscreenElement === surface) await document.exitFullscreen();
  else if (surface.classList.contains("expanded")) {
    surface.classList.remove("expanded"); document.body.classList.remove("viewer-expanded");
  } else {
    try {
      if (!surface.requestFullscreen) throw new Error("Fullscreen unavailable");
      await surface.requestFullscreen();
    } catch {
      surface.classList.add("expanded"); document.body.classList.add("viewer-expanded");
    }
  }
  updateExpandButton();
}
$("expand-view").addEventListener("click", toggleExpand);
$("preview").addEventListener("click", toggleExpand);
$("preview").addEventListener("keydown", event => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); toggleExpand(); }
});
video.addEventListener("dblclick", event => { event.preventDefault(); toggleExpand(); });
document.addEventListener("fullscreenchange", updateExpandButton);
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && surface.classList.contains("expanded")) toggleExpand();
});

function updateModelButtons(active = ["queued", "running", "live"].includes(lastJob?.status)) {
  $("weights-file").disabled = modelChanging || active;
  $("load-weights").disabled = modelChanging || active;
  $("default-weights").disabled = modelChanging || active;
}
function showModel(status) {
  modelReady = status.ready;
  $("model-state").textContent = modelReady ? "Model ready" : "Checkpoint needed";
  $("model-name").textContent = modelReady ? status.model.checkpoint : "No checkpoint loaded";
  $("model-details").textContent = modelReady
    ? `${status.model.device_name || status.model.device} · input ${status.model.imgsz}`
    : status.detail;
  $("run").disabled = !modelReady || modelChanging || ["queued", "running", "live"].includes(lastJob?.status);
}
async function changeModel(useDefault) {
  const file = $("weights-file").files[0];
  if (!useDefault && !file) { notice("Choose a .pt checkpoint first."); return; }
  if (!useDefault && (!file.name.toLowerCase().endsWith(".pt") || file.size > 250*1024*1024)) {
    notice("Choose a .pt checkpoint up to 250 MB."); return;
  }
  modelChanging = true; updateModelButtons(); $("run").disabled = true; notice();
  $("weights-message").textContent = "Loading checkpoint…";
  try {
    const body = new FormData();
    if (!useDefault) body.append("file", file);
    const status = await api(useDefault ? "/api/model/default" : "/api/model/upload", {
      method:"POST", headers:{"X-Helmet-Model":"1"}, ...(useDefault ? {} : {body})
    });
    showModel(status);
    $("weights-message").textContent = `Loaded ${status.model.checkpoint}`;
    $("weights-file").value = "";
  } catch (error) {
    $("weights-message").textContent = "Model unchanged."; notice(error.message);
  } finally {
    modelChanging = false; updateModelButtons();
    try { showModel(await api("/api/status")); } catch (error) { notice(error.message); }
  }
}
$("load-weights").addEventListener("click", () => changeModel(false));
$("default-weights").addEventListener("click", () => changeModel(true));
(async () => {
  try { showModel(await api("/api/status")); updateModelButtons(); await history(); }
  catch (error) { notice(error.message); }
})();
$("review-captures").addEventListener("click", () => $("capture-section").scrollIntoView({behavior:"smooth",block:"start"}));

$("toggle-preview").addEventListener("click", () => {
  showProcessingPreview = !showProcessingPreview;
  if (lastJob) render(lastJob);
});
