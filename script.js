(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const STAGE_ORDER = ["input", "script", "voice_scene", "image", "assemble", "done"];

  // ---------------- Top nav (Studio / Library) ----------------
  const navLinks = $$(".nav-link");
  const views = { studio: $("#view-studio"), library: $("#view-library") };

  navLinks.forEach((btn) => {
    btn.addEventListener("click", () => {
      navLinks.forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      const target = btn.dataset.view;
      Object.entries(views).forEach(([name, el]) => { el.hidden = name !== target; });
      if (target === "library") loadLibrary();
    });
  });

  // ---------------- Input tabs (Text / Image / Voice) ----------------
  let activeTab = "text";
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      activeTab = tab.dataset.tab;
      $$(".tab-panel").forEach((p) => p.classList.toggle("is-active", p.dataset.panel === activeTab));
    });
  });

  // ---------------- File pickers + drag/drop ----------------
  let selectedImageFile = null;
  let selectedVoiceFile = null;

  function wireDropzone(dropzoneId, inputId, chipId, onFile) {
    const dz = $(`#${dropzoneId}`);
    const input = $(`#${inputId}`);
    const chip = $(`#${chipId}`);

    const setFile = (file) => {
      if (!file) return;
      onFile(file);
      chip.hidden = false;
      chip.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
    };

    input.addEventListener("change", () => setFile(input.files[0]));

    ["dragover", "dragenter"].forEach((evt) =>
      dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.add("is-dragover"); })
    );
    ["dragleave", "drop"].forEach((evt) =>
      dz.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.remove("is-dragover"); })
    );
    dz.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files[0];
      if (file) { input.files = e.dataTransfer.files; setFile(file); }
    });
  }

  wireDropzone("image-dropzone", "image-input", "image-chip", (f) => (selectedImageFile = f));
  wireDropzone("voice-dropzone", "voice-input", "voice-chip", (f) => (selectedVoiceFile = f));

  // ---------------- Mic recording (optional, in-browser) ----------------
  const micBtn = $("#mic-btn");
  let mediaRecorder = null;
  let recordedChunks = [];
  let isRecording = false;

  micBtn.addEventListener("click", async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => e.data.size > 0 && recordedChunks.push(e.data);
        mediaRecorder.onstop = () => {
          const blob = new Blob(recordedChunks, { type: "audio/webm" });
          selectedVoiceFile = new File([blob], "recording.webm", { type: "audio/webm" });
          const chip = $("#voice-chip");
          chip.hidden = false;
          chip.textContent = `Recording captured · ${(blob.size / 1024).toFixed(0)} KB`;
          stream.getTracks().forEach((t) => t.stop());
        };
        mediaRecorder.start();
        isRecording = true;
        micBtn.classList.add("is-recording");
        micBtn.querySelector("span:last-child")?.remove();
        micBtn.append(" Stop recording");
      } catch (err) {
        alert("Mic access nahi mila — file upload use kar lo.");
      }
    } else {
      mediaRecorder.stop();
      isRecording = false;
      micBtn.classList.remove("is-recording");
      micBtn.lastChild.textContent = " Record instead";
    }
  });

  // ---------------- Generate form ----------------
  const form = $("#generate-form");
  const generateBtn = $("#generate-btn");

  const idleEl = $("#result-idle");
  const progressEl = $("#result-progress");
  const errorEl = $("#result-error");
  const doneEl = $("#result-done");

  function showResultState(state) {
    idleEl.hidden = state !== "idle";
    progressEl.hidden = state !== "progress";
    errorEl.hidden = state !== "error";
    doneEl.hidden = state !== "done";
  }

  function updateStages(currentStage) {
    const idx = STAGE_ORDER.indexOf(currentStage);
    $$(".stage-dot").forEach((dot) => {
      const dotIdx = STAGE_ORDER.indexOf(dot.dataset.stage);
      dot.classList.toggle("is-done", dotIdx < idx);
      dot.classList.toggle("is-active", dotIdx === idx);
    });
  }

  let pollTimer = null;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fd = new FormData();
    fd.append("input_type", activeTab);
    if (activeTab === "text") {
      const text = $("#text-input").value.trim();
      if (!text) { alert("Pehle kuch likho."); return; }
      fd.append("text", text);
    } else if (activeTab === "image") {
      if (!selectedImageFile) { alert("Pehle ek image choose karo."); return; }
      fd.append("file", selectedImageFile);
    } else {
      if (!selectedVoiceFile) { alert("Pehle ek voice note do — record karo ya upload karo."); return; }
      fd.append("file", selectedVoiceFile);
    }
    fd.append("duration_sec", $("#opt-duration").value);
    fd.append("language", $("#opt-language").value);
    fd.append("tone", $("#opt-tone").value);
    fd.append("aspect_ratio", $("#opt-aspect").value);

    generateBtn.disabled = true;
    showResultState("progress");
    updateStages("input");
    $("#progress-fill").style.width = "5%";
    $("#progress-message").textContent = "Job shuru ho rahi hai...";

    try {
      const res = await fetch("/api/generate", { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Kuch galat ho gaya" }));
        throw new Error(err.detail || "Request fail ho gayi");
      }
      const { job_id } = await res.json();
      pollStatus(job_id);
    } catch (err) {
      generateBtn.disabled = false;
      showResultState("error");
      $("#error-message").textContent = err.message;
    }
  });

  function pollStatus(jobId) {
    clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${jobId}`);
        if (!res.ok) throw new Error("Status check fail ho gaya");
        const job = await res.json();

        $("#progress-fill").style.width = `${job.progress || 5}%`;
        $("#progress-message").textContent = job.message || "";
        updateStages(job.stage);

        if (job.status === "done") {
          clearInterval(pollTimer);
          generateBtn.disabled = false;
          $("#result-video").src = job.video_url;
          $("#result-title").textContent = job.title || "Your video";
          $("#result-duration").textContent = job.duration_sec ? `${Math.round(job.duration_sec)}s` : "";
          $("#download-link").href = job.video_url;
          showResultState("done");
        } else if (job.status === "error") {
          clearInterval(pollTimer);
          generateBtn.disabled = false;
          showResultState("error");
          $("#error-message").textContent = job.error || "Video ban nahi payi.";
        }
      } catch (err) {
        clearInterval(pollTimer);
        generateBtn.disabled = false;
        showResultState("error");
        $("#error-message").textContent = "Server se connection toot gaya.";
      }
    }, 2000);
  }

  $("#retry-btn").addEventListener("click", () => showResultState("idle"));

  // ---------------- Library ----------------
  async function loadLibrary() {
    const grid = $("#library-grid");
    const empty = $("#library-empty");
    try {
      const res = await fetch("/api/videos");
      const items = await res.json();
      if (!items.length) {
        grid.innerHTML = "";
        empty.hidden = false;
        return;
      }
      empty.hidden = true;
      grid.innerHTML = items.map((v) => `
        <div class="glass library-card">
          <video src="${v.video_url}" controls preload="metadata"></video>
          <div class="library-card-body">
            <h4>${escapeHtml(v.title || "Untitled")}</h4>
            <span>${Math.round(v.duration_sec || 0)}s · ${v.n_scenes || 0} scenes</span>
          </div>
        </div>
      `).join("");
    } catch {
      grid.innerHTML = "";
      empty.hidden = false;
      empty.textContent = "Library load nahi ho payi — server chal raha hai?";
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
})();