/* VeriFake frontend: upload, prediction, Grad-CAM attribution, content credentials, performance tables. */
(() => {
  "use strict";

  const MAX_BYTES = 15 * 1024 * 1024;
  const ALLOWED_MIMES = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/x-ms-bmp",
    "image/tiff",
    "image/gif",
    "image/avif",
    "image/heic",
    "image/heif",
    "image/heic-sequence",
    "image/heif-sequence",
  ];
  const ALLOWED_EXT_RE = /\.(jpe?g|png|webp|bmp|tiff?|gif|avif|heic|heif)$/i;

  const $ = (id) => document.getElementById(id);
  const el = {
    status: $("status"),
    statusText: $("status-text"),
    tempPill: $("temp-pill"),
    dropzone: $("dropzone"),
    input: $("file-input"),
    dzEmpty: $("dz-empty"),
    dzPreview: $("dz-preview"),
    preview: $("preview-img"),
    dzFallback: $("dz-fallback"),
    replaceBar: $("replace-bar"),
    replaceBrowse: $("replace-browse-btn"),
    fileMeta: $("file-meta"),
    metaName: $("meta-name"),
    metaSize: $("meta-size"),
    metaDims: $("meta-dims"),
    reset: $("reset-btn"),
    retry: $("retry-btn"),
    nofaceRetry: $("noface-retry-btn"),
    progress: $("progress"),
    sEmpty: $("state-empty"),
    sScan: $("state-scanning"),
    sNoFace: $("state-no-face"),
    sErr: $("state-error"),
    sRes: $("state-result"),
    errMsg: $("error-msg"),
    nofaceTitle: $("noface-title"),
    nofaceProv: $("noface-prov"),
    verdictKicker: $("verdict-kicker"),
    confLabel: $("conf-label"),
    badge: $("verdict-badge"),
    gaugeVal: $("gauge-value"),
    desc: $("verdict-desc"),
    pRealVal: $("p-real-val"),
    pFakeVal: $("p-fake-val"),
    barReal: $("bar-real"),
    barFake: $("bar-fake"),
    smallFaceWarn: $("small-face-warn"),
    smallFaceTitle: $("small-face-title"),
    smallFaceDetail: $("small-face-detail"),
    pixOverrideNote: $("pix-override-note"),
    explain: $("attention-map"),
    compare: $("compare"),
    cmpOrig: $("cmp-original"),
    cmpCam: $("cmp-cam"),
    camLabel: $("cam-label"),
    rModel: $("r-model"),
    rLatency: $("r-latency"),
    rInput: $("r-input"),
    provRow: $("provenance"),
    provBadge: $("prov-badge"),
    provSummary: $("prov-summary"),
    provDetails: $("prov-details"),
    provKv: $("prov-kv"),
    diagBox: $("diag-box"),
    diagFaceStatus: $("diag-face-status"),
    diagKv: $("diag-kv"),
    diagViews: $("diag-views"),
  };

  let busy = false;
  let previewUrl = null;

  const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
  const fmtBytes = (b) => (b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(2)} MB`);
  const esc = (s) => String(s ?? "—").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));

  /* Active navigation sync */
  function setActiveNav(target) {
    document.querySelectorAll(".topnav a[data-nav]").forEach((a) => {
      const isActive = a.dataset.nav === target;
      a.classList.toggle("active", isActive);
      if (isActive) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
  }
  document.querySelectorAll("[data-nav]").forEach((a) => {
    a.addEventListener("click", () => setActiveNav(a.dataset.nav));
  });
  if ("IntersectionObserver" in window && $("analyze") && $("performance")) {
    const navObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveNav(entry.target.id === "performance" ? "performance" : "analyze");
          }
        }
      },
      { rootMargin: "-25% 0px -60% 0px" },
    );
    navObserver.observe($("analyze"));
    navObserver.observe($("performance"));
  }

  /* States */
  function showState(name) {
    for (const [k, node] of Object.entries({
      empty: el.sEmpty,
      scan: el.sScan,
      noface: el.sNoFace,
      err: el.sErr,
      res: el.sRes,
    })) {
      node.classList.toggle("hidden", k !== name);
    }
    el.progress.classList.toggle("hidden", name !== "scan");
    if (name !== "res") {
      el.provRow.classList.add("hidden");
      el.diagBox.classList.add("hidden");
      el.explain.classList.add("hidden");
    }
  }

  /* Health */
  async function loadHealth() {
    try {
      const r = await fetch("/api/health");
      const h = await r.json();
      if (h.model_loaded) {
        el.status.className = "status-pill online";
        el.statusText.textContent = "Model ready";
        el.status.title = `${h.model} · ${h.device}`;
        if (h.temperature != null && el.tempPill) {
          el.tempPill.textContent = `T = ${Number(h.temperature).toFixed(4)}`;
        }
      } else {
        el.status.className = "status-pill offline";
        el.statusText.textContent = "Model offline";
        el.status.title = h.error || "Model not loaded";
      }
    } catch {
      el.status.className = "status-pill offline";
      el.statusText.textContent = "Server unreachable";
    }
  }

  /* Validation & preview */
  function validate(file) {
    if (!file) return "No file selected.";
    if (file.size === 0) return "The selected file is empty.";
    if (file.size > MAX_BYTES) return `File is ${fmtBytes(file.size)}. The maximum allowed size is 15 MB.`;
    const mime = (file.type || "").toLowerCase();
    const extOk = ALLOWED_EXT_RE.test(file.name || "");
    if (mime && mime !== "application/octet-stream" && !ALLOWED_MIMES.includes(mime) && !extOk) {
      return `Unsupported file type (${file.type}). Supported: JPG, PNG, WEBP, BMP, TIFF, GIF, AVIF, HEIC/HEIF.`;
    }
    return null;
  }

  function setPreview(file) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    el.dzFallback.classList.add("hidden");
    el.preview.classList.remove("hidden");
    el.preview.onload = () => {
      el.preview.classList.remove("hidden");
      el.dzFallback.classList.add("hidden");
      if (el.preview.naturalWidth && el.preview.naturalHeight) {
        el.metaDims.textContent = `${el.preview.naturalWidth}×${el.preview.naturalHeight}`;
      }
    };
    el.preview.onerror = () => {
      el.preview.classList.add("hidden");
      el.dzFallback.classList.remove("hidden");
      el.metaDims.textContent = "Server decode";
    };
    el.preview.src = previewUrl;
    el.dzEmpty.classList.add("hidden");
    el.dzPreview.classList.remove("hidden");
    el.replaceBar.classList.remove("hidden");
    el.fileMeta.classList.remove("hidden");
    el.metaName.textContent = file.name || "Pasted image";
    el.metaName.title = file.name || "Pasted image";
    el.metaSize.textContent = fmtBytes(file.size);
    el.metaDims.textContent = "…";
    el.reset.classList.remove("hidden");
  }

  async function analyze(file) {
    if (busy) return;
    const problem = validate(file);
    if (problem) {
      showError(problem);
      return;
    }

    busy = true;
    setPreview(file);
    el.dropzone.classList.add("busy");
    showState("scan");

    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch("/predict", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Server returned ${res.status}`);
      renderResult(data);
    } catch (err) {
      showError(err.message === "Failed to fetch" ? "Could not reach the server. Is it still running?" : err.message);
    } finally {
      el.dropzone.classList.remove("busy");
      busy = false;
    }
  }

  function showError(msg) {
    el.errMsg.textContent = msg;
    showState("err");
  }

  function formatProvenancePresentText(prov) {
    const signerTxt = prov.signer || prov.claim_generator;
    const sigTxt = prov.signature_valid ? "Valid signature" : "Invalid signature";
    const aiTxt = prov.ai_verified_by_credentials ? "AI-generated assertion" : "No AI-generated assertion";
    return signerTxt ? `${sigTxt} · ${signerTxt} · ${aiTxt}` : `${sigTxt} · ${aiTxt}`;
  }

  function renderProvenance(prov) {
    if (!prov) {
      el.provRow.classList.add("hidden");
      return;
    }
    el.provRow.classList.remove("hidden");

    if (prov.present) {
      let badgeText = "Signed";
      let badgeCls = "signed";
      if (prov.ai_verified_by_credentials) {
        badgeText = "Declares AI";
        badgeCls = "ai";
      } else if (!prov.signature_valid) {
        badgeText = "Invalid signature";
        badgeCls = "invalid";
      }
      el.provBadge.textContent = badgeText;
      el.provBadge.className = `prov-badge ${badgeCls}`;
      el.provSummary.textContent = formatProvenancePresentText(prov);

      const rows = [
        ["Status", "Present"],
        ["Signature", prov.signature_valid ? `Valid (${prov.validation_state || "verified"})` : `Invalid (${prov.validation_state || "failed"})`],
        ["Signer", prov.signer || "—"],
        ["AI-generated flag", prov.ai_verified_by_credentials ? "Yes (AI-generated declared)" : "No"],
        ["Claim generator", prov.claim_generator || "—"],
        ["Source type", prov.digital_source_types && prov.digital_source_types.length ? prov.digital_source_types.join(", ") : "—"],
      ];
      el.provKv.innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd title="${esc(v)}">${esc(v)}</dd>`).join("");
      el.provDetails.open = false;
      el.provDetails.classList.remove("hidden");
    } else {
      el.provBadge.className = "prov-badge hidden";
      el.provSummary.textContent = "Content credentials not found. Missing credentials do not mean the image is real.";
      const hasExif = Boolean(prov.camera_exif && (prov.camera_exif.Make || prov.camera_exif.Model));
      if (hasExif) {
        const cam = [prov.camera_exif.Make, prov.camera_exif.Model].filter(Boolean).join(" ");
        el.provKv.innerHTML = `<dt>Camera EXIF (unverified)</dt><dd title="${esc(cam)}">${esc(cam)}</dd>`;
        el.provDetails.open = false;
        el.provDetails.classList.remove("hidden");
      } else {
        el.provKv.innerHTML = "";
        el.provDetails.classList.add("hidden");
      }
    }
  }

  function renderDiagnostics(diag) {
    if (!diag) {
      el.diagBox.classList.add("hidden");
      return;
    }
    el.diagBox.classList.remove("hidden");
    el.diagBox.open = false;
    if (diag.face_detected) {
      el.diagFaceStatus.textContent = diag.small_face
        ? `Small face (${diag.face_width_pct.toFixed(1)}% width)`
        : `Face detected · score ${diag.face_score.toFixed(3)}`;
      el.diagFaceStatus.className = `diag-status${diag.small_face ? " warn" : ""}`;
    } else {
      el.diagFaceStatus.textContent = "No face detected";
      el.diagFaceStatus.className = "diag-status warn";
    }

    const rawRes = diag.raw_crop_resolution ? diag.raw_crop_resolution.join("×") : "—";
    const finalRes = diag.crop_resolution ? diag.crop_resolution.join("×") : "—";
    const cropStr = diag.face_detected
      ? diag.upsampled
        ? `${rawRes} → ${finalRes} (Lanczos, m=${diag.crop_margin})`
        : `${finalRes} (margin ${diag.crop_margin})`
      : "—";
    const rows = [
      ["Face detected", diag.face_detected ? `Yes (score ${diag.face_score.toFixed(3)})` : "No"],
      ["Face box width", diag.face_detected ? `${diag.face_width_pct.toFixed(1)}% of image width${diag.small_face ? " (<15%)" : ""}` : "0%"],
      ["Crop resolution", cropStr],
    ];
    el.diagKv.innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd title="${esc(v)}">${esc(v)}</dd>`).join("");

    const pv = diag.per_view_probabilities;
    const tbody = el.diagViews.querySelector("tbody");
    if (pv && Object.keys(pv).length > 0) {
      const names = { crop: "Face crop", flipped: "Flipped crop", full_frame: "Full image" };
      tbody.innerHTML = Object.entries(pv)
        .map(
          ([k, v]) =>
            `<tr><th scope="row">${esc(names[k] || k)}</th><td class="num">${pct(v.FAKE, 1)}</td><td class="num">${pct(v.REAL, 1)}</td><td>${esc(v.label)}</td></tr>`,
        )
        .join("");
      el.diagViews.classList.remove("hidden");
    } else {
      tbody.innerHTML = "";
      el.diagViews.classList.add("hidden");
    }
  }

  function setView(view) {
    el.compare.dataset.view = view;
    document.querySelectorAll(".segmented button").forEach((btn) => {
      btn.setAttribute("aria-pressed", String(btn.dataset.view === view));
    });
  }
  document.querySelectorAll(".segmented button").forEach((btn) => {
    btn.addEventListener("click", () => setView(btn.dataset.view));
  });

  function renderResult(d) {
    if (d.preview) {
      el.preview.onload = null;
      el.preview.onerror = null;
      el.preview.src = d.preview;
      el.preview.classList.remove("hidden");
      el.dzFallback.classList.add("hidden");
    }
    if (d.image_size) {
      el.metaDims.textContent = d.image_size.join("×");
    }

    const prov = d.provenance || d.content_credentials;
    const diag = d.diagnostics;

    if (d.status === "no_face") {
      showState("noface");
      el.nofaceTitle.textContent = d.message || "No face detected. This tool analyzes faces only.";
      if (prov && prov.present) {
        el.nofaceProv.textContent = `Provenance: ${formatProvenancePresentText(prov)}.`;
        el.nofaceProv.classList.remove("hidden");
      } else {
        el.nofaceProv.classList.add("hidden");
      }
      return;
    }

    const override = Boolean(d.override_applied);
    const pix = d.pixel_model || {};
    const pixOk = Boolean(pix.probabilities && pix.status === "ok");
    const isUncertain = !override && (d.label === "UNCERTAIN" || d.uncertain);
    const isFake = override || (d.binary_label ? d.binary_label === "FAKE" : d.label === "FAKE");
    const cls = isUncertain ? "uncertain" : isFake ? "fake" : "real";
    showState("res");

    let verdictText = "Likely real";
    if (override) verdictText = "AI-generated (verified credentials)";
    else if (isUncertain) verdictText = "Uncertain";
    else if (isFake) verdictText = "Likely AI-generated";

    el.verdictKicker.textContent = override ? "Verified by credentials" : "Verdict";
    el.badge.textContent = verdictText;
    el.badge.dataset.label = d.label || "";
    el.badge.className = `verdict-heading ${cls}`;

    const conf = d.confidence || 0;
    el.confLabel.textContent = override ? "Provenance confidence" : "Confidence";
    el.gaugeVal.textContent = pct(conf, 1);

    if (override) {
      el.desc.textContent = "Signed Content Credentials declare AI generation.";
    } else if (isUncertain) {
      el.desc.textContent = "Calibrated score falls in the uncertain band — neither real nor AI-generated is conclusive.";
    } else if (isFake) {
      el.desc.textContent = "Calibrated 3-view analysis indicates this portrait is likely AI-generated.";
    } else {
      el.desc.textContent = prov && prov.signature_valid
        ? "Calibrated 3-view analysis finds no synthesis signs, and Content Credentials make no AI assertion."
        : "Calibrated 3-view analysis finds no synthesis signs.";
    }

    const probs = pixOk ? pix.probabilities : d.probabilities;
    if (probs) {
      const pReal = probs.REAL;
      const pFake = probs.FAKE;
      el.pRealVal.textContent = pct(pReal, 1);
      el.pFakeVal.textContent = pct(pFake, 1);
      el.barReal.style.width = `${(pReal * 100).toFixed(2)}%`;
      el.barFake.style.width = `${(pFake * 100).toFixed(2)}%`;
      if (isUncertain) {
        el.barReal.className = "split-seg seg-muted";
        el.barFake.className = "split-seg seg-uncertain";
      } else if (isFake) {
        el.barReal.className = "split-seg seg-muted";
        el.barFake.className = "split-seg seg-fake";
      } else {
        el.barReal.className = "split-seg seg-real";
        el.barFake.className = "split-seg seg-muted";
      }
    } else {
      el.pRealVal.textContent = override ? "0.0%" : "—";
      el.pFakeVal.textContent = override ? "100.0%" : "—";
      el.barReal.style.width = override ? "0%" : "50%";
      el.barFake.style.width = override ? "100%" : "50%";
      el.barReal.className = "split-seg seg-muted";
      el.barFake.className = override ? "split-seg seg-fake" : "split-seg seg-muted";
    }

    if (diag && diag.small_face) {
      el.smallFaceTitle.textContent = diag.warning || "Small face: low confidence";
      el.smallFaceDetail.textContent = `Face spans ${diag.face_width_pct.toFixed(1)}% of image width (<15%); crop ${diag.crop_resolution ? diag.crop_resolution.join("×") : ""} applied.`;
      el.smallFaceWarn.classList.remove("hidden");
    } else {
      el.smallFaceWarn.classList.add("hidden");
    }

    if (override && pixOk) {
      el.pixOverrideNote.textContent = `Pixel model alone predicted ${pix.label} (P(AI-generated) ${pct(pix.probabilities.FAKE, 1)}), overridden by signed Content Credentials.`;
      el.pixOverrideNote.classList.remove("hidden");
    } else if (override) {
      el.pixOverrideNote.textContent = "No face detected; verdict set strictly by verified Content Credentials.";
      el.pixOverrideNote.classList.remove("hidden");
    } else {
      el.pixOverrideNote.classList.add("hidden");
    }

    el.rModel.textContent = `${d.model} · T=${(d.temperature || 1).toFixed(2)}`;
    el.rLatency.textContent = `${Math.round(d.inference_ms)} ms`;
    el.rInput.textContent = d.detected_format ? `${d.image_size.join("×")} · ${d.detected_format}` : d.image_size.join("×");

    if (d.gradcam) {
      el.cmpOrig.src = d.preview || previewUrl;
      el.cmpCam.src = d.gradcam;
      const target = d.gradcam_target || d.label;
      el.camLabel.textContent = `Heatmap (${target})`;
      setView("split");
      el.explain.classList.remove("hidden");
    } else {
      el.explain.classList.add("hidden");
    }

    renderProvenance(prov);
    renderDiagnostics(diag);
  }

  function reset() {
    if (busy) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    el.input.value = "";
    el.preview.onload = null;
    el.preview.onerror = null;
    el.preview.removeAttribute("src");
    el.preview.classList.remove("hidden");
    el.dzFallback.classList.add("hidden");
    el.dzPreview.classList.add("hidden");
    el.dzEmpty.classList.remove("hidden");
    el.replaceBar.classList.add("hidden");
    el.fileMeta.classList.add("hidden");
    el.reset.classList.add("hidden");
    showState("empty");
  }

  /* Events */
  el.input.addEventListener("change", () => el.input.files[0] && analyze(el.input.files[0]));
  el.dropzone.addEventListener("click", (e) => {
    if (busy) {
      e.preventDefault();
      return;
    }
    if (e.target.closest("label, button, input, a")) return;
    if (!el.dzEmpty.classList.contains("hidden")) {
      el.input.click();
    }
  });
  const chooseBtn = $("choose-btn");
  if (chooseBtn) {
    chooseBtn.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === " ") && !busy) {
        e.preventDefault();
        el.input.click();
      }
    });
  }
  ["dragenter", "dragover"].forEach((t) =>
    el.dropzone.addEventListener(t, (e) => {
      e.preventDefault();
      if (!busy) el.dropzone.classList.add("drag");
    }),
  );
  ["dragleave", "dragend", "drop"].forEach((t) =>
    el.dropzone.addEventListener(t, (e) => {
      e.preventDefault();
      el.dropzone.classList.remove("drag");
    }),
  );
  el.dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) analyze(f);
  });
  ["dragover", "drop"].forEach((t) => window.addEventListener(t, (e) => e.preventDefault()));
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) analyze(item.getAsFile());
  });
  el.reset.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    reset();
  });
  if (el.replaceBrowse) {
    el.replaceBrowse.addEventListener("click", () => {
      if (!busy) el.input.click();
    });
  }
  el.retry.addEventListener("click", () => {
    reset();
    el.input.click();
  });
  el.nofaceRetry.addEventListener("click", () => {
    reset();
    el.input.click();
  });

  function inlineBarHtml(frac) {
    const p = `${(frac * 100).toFixed(1)}%`;
    return `<div class="inline-metric"><div class="inline-bar-track" aria-hidden="true"><div class="inline-bar-fill" style="width: ${p}"></div></div><span>${p}</span></div>`;
  }

  /* Performance metrics from /api/metrics (metrics.json) */
  async function loadMetrics() {
    let m;
    try {
      const r = await fetch("/api/metrics");
      if (!r.ok) throw new Error();
      m = await r.json();
    } catch {
      $("perf-empty").classList.remove("hidden");
      return;
    }

    const t = m.test;
    const tta = m.tta_comparison || {};
    const cal = m.calibration || {};
    const unc = t.uncertainty || {};
    const cm = t.confusion_matrix;
    const oos = m.out_of_scope_no_face || {};

    $("perf-body").classList.remove("hidden");
    $("perf-sub").textContent =
      `Measured on ${t.n.toLocaleString()} face-gated held-out test images (${m.backbone || "model"}, ${m.img_size}×${m.img_size}). ${oos.test_no_face_total ? `${oos.test_no_face_total.toLocaleString()} non-face images gated out of scope.` : ""}`;

    document.querySelectorAll("[data-metric]").forEach((node) => {
      const k = node.dataset.metric;
      node.textContent = k === "roc_auc" ? t[k].toFixed(4) : (t[k] * 100).toFixed(2);
    });

    const dtAcc = (tta.delta_accuracy || 0) * 100;
    const accNote = $("kpi-acc-note");
    accNote.textContent = `3-view TTA (${dtAcc >= 0 ? "+" : ""}${dtAcc.toFixed(2)}%)`;
    if (dtAcc >= 0) accNote.classList.add("good");

    const actualFake = cm[1][0] + cm[1][1];
    const actualReal = cm[0][0] + cm[0][1];
    $("kpi-recall-note").textContent = `${cm[1][1].toLocaleString()} / ${actualFake.toLocaleString()} AI-generated`;
    $("kpi-recall-note").classList.add("good");
    $("kpi-auc-note").textContent = `n = ${t.n.toLocaleString()}`;

    /* Confusion matrix */
    const cmLabels = t.confusion_matrix_labels || m.class_names || ["REAL", "FAKE"];
    $("cm-col-0").textContent = `Predicted ${cmLabels[0]}`;
    $("cm-col-1").textContent = `Predicted ${cmLabels[1]}`;
    $("cm-row-0").textContent = `Actual ${cmLabels[0]}`;
    $("cm-row-1").textContent = `Actual ${cmLabels[1]}`;
    $("cm-row-0-total").textContent = actualReal.toLocaleString();
    $("cm-row-1-total").textContent = actualFake.toLocaleString();
    $("cm-n").textContent = `n = ${t.n.toLocaleString()}`;

    const cellTags = [
      ["TN", "FP"],
      ["FN", "TP"],
    ];
    [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ].forEach(([i, j]) => {
      const rowSum = cm[i][0] + cm[i][1];
      const frac = rowSum ? cm[i][j] / rowSum : 0;
      $(`cm-${i}${j}`).innerHTML = `<b>${cm[i][j].toLocaleString()}</b><small>${(frac * 100).toFixed(2)}% (${cellTags[i][j]})</small>`;
    });
    $("cm-type1").textContent = `False positive rate: ${(actualReal ? cm[0][1] / actualReal : 0).toFixed(4)}`;
    $("cm-type2").textContent = `False negative rate: ${(actualFake ? cm[1][0] / actualFake : 0).toFixed(4)}`;

    /* ROC curve */
    $("roc-img").src = `/reports/roc_curve.png?v=${Date.now()}`;
    $("roc-auc-foot").textContent = `AUC ${t.roc_auc.toFixed(4)}`;

    /* Sorted per-source accuracy + highlighted unseen generator */
    const pg = m.per_generator || {};
    const ug = m.unseen_generator;
    const sources = Object.entries(pg).map(([name, info]) => ({
      name,
      cls: info.class,
      n: info.n,
      accuracy: info.accuracy,
      uncertain: info.uncertain_count ?? 0,
      unseen: false,
    }));
    if (ug && ug.n > 0) {
      sources.push({
        name: ug.generator,
        cls: ug.true_class || "FAKE",
        n: ug.n,
        accuracy: ug.accuracy,
        uncertain: ug.uncertain_count ?? 0,
        unseen: true,
      });
    }
    sources.sort((a, b) => b.accuracy - a.accuracy);

    $("generator-table").querySelector("tbody").innerHTML = sources
      .map((s) => {
        const clsHtml = s.unseen
          ? `<span class="cls-amber">${esc(s.cls)}</span>`
          : s.cls === "REAL"
            ? `<span class="cls-real">${esc(s.cls)}</span>`
            : `<span class="cls-fake">${esc(s.cls)}</span>`;
        const tag = s.unseen ? '<span class="unseen-pill">Unseen</span>' : "";
        return `
          <tr${s.unseen ? ' class="unseen-row"' : ""}>
            <th scope="row">${esc(s.name)}${tag}</th>
            <td>${clsHtml}</td>
            <td class="num">${s.n.toLocaleString()}</td>
            <td class="num bar-col">${inlineBarHtml(s.accuracy)}</td>
            <td class="num">${s.uncertain.toLocaleString()}</td>
          </tr>`;
      })
      .join("");

    if (ug && ug.n > 0) {
      $("unseen-tag").textContent = `Unseen ${ug.generator}: ${pct(ug.accuracy, 1)} (AUC ${ug.roc_auc_vs_test_real.toFixed(4)})`;
      $("gen-foot-left").textContent = `${Object.keys(pg).length} in-distribution sources + 1 held-out unseen generator`;
      $("gen-foot-right").textContent = `${ug.n.toLocaleString()} unseen ${ug.generator} faces evaluated`;
    } else {
      $("unseen-tag").textContent = `${sources.length} sources`;
      $("gen-foot-left").textContent = `${sources.length} evaluated sources`;
      $("gen-foot-right").textContent = "";
    }

    /* Compact robustness table */
    const robLabels = {
      original: "Original baseline",
      "resize_0.5x": "Resize 0.5×",
      resize_0_5x: "Resize 0.5×",
      "resize_0.25x": "Resize 0.25×",
      jpeg_q50: "JPEG q50 compression",
      blur: "Gaussian blur (σ = 1.5)",
      social_media_chain: "Social compression chain",
    };
    const rba = m.robustness_before_after;
    const robItems = rba
      ? Object.entries(rba).map(([k, v]) => ({
          label: robLabels[k] || k,
          n: v.n,
          beforeAcc: v.before_acc,
          afterAcc: v.after_acc,
          deltaPct: v.delta_acc * 100,
          auc: v.after_auc,
        }))
      : Object.entries(m.robustness || {}).map(([k, v]) => ({
          label: robLabels[k] || k,
          n: v.n,
          beforeAcc: null,
          afterAcc: v.accuracy,
          deltaPct: null,
          auc: v.roc_auc,
        }));

    $("rob-count").textContent = `${robItems.length} slices`;
    $("robust-table").querySelector("tbody").innerHTML = robItems
      .map((item) => {
        const beforeStr = item.beforeAcc != null ? pct(item.beforeAcc, 1) : "—";
        const deltaHtml =
          item.deltaPct == null
            ? "—"
            : `<span class="${item.deltaPct >= 0 ? "delta-up" : "delta-dn"}">${item.deltaPct >= 0 ? "+" : ""}${item.deltaPct.toFixed(2)}%</span>`;
        const aucStr = item.auc != null ? item.auc.toFixed(4) : "—";
        return `
          <tr>
            <th scope="row">${esc(item.label)}</th>
            <td class="num">${item.n ? item.n.toLocaleString() : "—"}</td>
            <td class="num">${beforeStr}</td>
            <td class="num bar-col">${inlineBarHtml(item.afterAcc)}</td>
            <td class="num">${deltaHtml}</td>
            <td class="num">${aucStr}</td>
          </tr>`;
      })
      .join("");

    if (robItems.length) {
      const worst = Math.min(...robItems.map((r) => r.afterAcc));
      const deltas = robItems.map((r) => r.deltaPct).filter((d) => d != null);
      $("rob-foot-left").textContent = `Lowest slice after upgrade: ${pct(worst, 1)}`;
      if (deltas.length) {
        const meanDelta = deltas.reduce((a, b) => a + b, 0) / deltas.length;
        $("rob-foot-right").textContent = `Mean delta: ${meanDelta >= 0 ? "+" : ""}${meanDelta.toFixed(2)}%`;
      } else {
        $("rob-foot-right").textContent = "—";
      }
    }

    /* Calibration table */
    const eceBefore = cal.test_ece_before;
    const eceAfter = cal.test_ece_after;
    const eceDrop = eceBefore ? ((eceAfter - eceBefore) / eceBefore) * 100 : 0;
    const valCal = cal.val_calibration || {};
    const band = unc.uncertain_band || cal.uncertain_band;
    const bandLabel = band && band.length === 2
      ? `Uncertain band [${Number(band[0]).toFixed(2)}, ${Number(band[1]).toFixed(2)}]`
      : "Uncertain band";

    if (cal.temperature != null) {
      el.tempPill.textContent = `T = ${cal.temperature.toFixed(4)}`;
    }
    const calRows = [
      ["Temperature scaling", cal.temperature != null ? `T = ${cal.temperature.toFixed(4)}` : "—"],
      [
        "Test ECE (before → after)",
        eceBefore != null
          ? `<span class="cal-old">${eceBefore.toFixed(4)}</span> → <span class="cal-new">${eceAfter.toFixed(4)}</span>`
          : "—",
      ],
      [
        "Validation ECE (before → after)",
        valCal.val_ece_before != null
          ? `<span class="cal-old">${valCal.val_ece_before.toFixed(4)}</span> → <span class="cal-new">${valCal.val_ece_after.toFixed(4)}</span>`
          : "—",
      ],
      ["Single-view test accuracy", tta.single_view_accuracy != null ? pct(tta.single_view_accuracy, 2) : "—"],
      [
        "3-view TTA test accuracy",
        tta.tta_3view_accuracy != null
          ? `${pct(tta.tta_3view_accuracy, 2)} <span class="${dtAcc >= 0 ? "delta-up" : "delta-dn"}">(${dtAcc >= 0 ? "+" : ""}${dtAcc.toFixed(2)}%)</span>`
          : "—",
      ],
      [
        bandLabel,
        unc.uncertain_count != null
          ? `<span class="cal-amber">${unc.uncertain_count} / ${t.n.toLocaleString()}</span> (${pct(unc.uncertain_rate, 2)})`
          : "—",
      ],
      [
        "Decisive accuracy (excl. uncertain)",
        unc.decisive_accuracy != null ? `<span class="cal-amber">${pct(unc.decisive_accuracy, 2)}</span>` : "—",
      ],
    ];
    $("cal-table").querySelector("tbody").innerHTML = calRows
      .map(([k, v]) => `<tr><th scope="row">${esc(k)}</th><td class="num">${v}</td></tr>`)
      .join("");
    $("cal-foot-right").textContent = eceBefore != null ? `Test ECE change: ${eceDrop.toFixed(1)}%` : "—";

    /* Leave-One-Generator-Out table */
    const logo = m.logo_generalization || {};
    const folds = logo.folds || [];
    const logoBody = $("logo-table").querySelector("tbody");
    const logoFootRow = $("logo-foot-row");
    if (folds.length) {
      $("logo-tag").textContent = `Mean ${pct(logo.mean_detection_rate, 1)} · AUC ${logo.mean_roc_auc_vs_real.toFixed(4)}`;
      logoBody.innerHTML = folds
        .map(
          (f) => `
          <tr>
            <th scope="row">${esc(f.generator)}</th>
            <td class="num">${f.held_out_n.toLocaleString()}</td>
            <td class="num bar-col">${inlineBarHtml(f.held_out_detection_rate)}</td>
            <td class="num">${f.held_out_roc_auc_vs_real.toFixed(4)}</td>
          </tr>`,
        )
        .join("");
      logoFootRow.innerHTML = `
        <th scope="row">Mean across ${folds.length} folds</th>
        <td class="num">—</td>
        <td class="num">${pct(logo.mean_detection_rate, 2)}</td>
        <td class="num">${logo.mean_roc_auc_vs_real.toFixed(4)}</td>`;
      $("logo-foot-right").textContent = `Mean AUC ${logo.mean_roc_auc_vs_real.toFixed(4)}`;
    } else {
      $("logo-tag").textContent = "Unavailable";
      logoBody.innerHTML = `<tr><td colspan="4">No Leave-One-Generator-Out results available.</td></tr>`;
      logoFootRow.innerHTML = "";
    }
  }

  loadHealth();
  loadMetrics();
})();
