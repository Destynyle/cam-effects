/*
 * Blob Art — version web
 * ──────────────────────
 * Portage de effects/blob_art.py : détection de mouvement → blobs,
 * couleur dominante + code hex, triangulation entre blobs.
 *
 * Modes (1-4 doigts maintenus 1.5s, boutons, ou touches 1-4) :
 *   Default · Loupe · Vitrail · Voronoï
 */

import { Delaunay } from "https://cdn.jsdelivr.net/npm/d3-delaunay@6/+esm";

const MP_VERSION = "0.10.14";
const MP_URL     = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL  =
  "https://storage.googleapis.com/mediapipe-models/" +
  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// ── Paramètres blobs (référence 640px de large, comme la version Python) ─────
const REF_W          = 640;
const MIN_BLOB_FRAC  = 400 / (640 * 480);   // MIN_BLOB_AREA relatif à l'image
const MAX_BLOBS      = 12;
const TRAIL_LEN      = 18;
const SAMPLE_RADIUS  = 14;
const LINE_DIST      = 260;
const DARKEN         = 0.30;                 // ≈ effet BG_ALPHA=0.55 du Python

// ── Détection de mouvement ───────────────────────────────────────────────────
const PROC_SIZE      = 192;    // plus grand côté de l'image de travail
const BG_RATE        = 0.03;   // vitesse d'apprentissage du fond
const BG_RATE_FAST   = 0.35;   // si la moitié de l'image bouge (auto-exposition…)
const DIFF_THRESH    = 60;     // |ΔR|+|ΔG|+|ΔB|
const WARMUP_FRAMES  = 15;

// ── Modes ────────────────────────────────────────────────────────────────────
const MODES            = ["default", "loupe", "vitrail", "voronoi"];
const MODE_LABELS      = ["Default", "Loupe", "Vitrail", "Voronoï"];
const GESTURE_HOLD_MS  = 1500;


// ══════════════════════════════════════════════════════════════════════════════
// Utilitaires couleur
// ══════════════════════════════════════════════════════════════════════════════

const hex2 = (v) => v.toString(16).padStart(2, "0").toUpperCase();
const toHex = ([r, g, b]) => `#${hex2(r)}${hex2(g)}${hex2(b)}`;
const rgb = ([r, g, b]) => `rgb(${r},${g},${b})`;
const luminance = ([r, g, b]) => 0.299 * r + 0.587 * g + 0.114 * b;
const textColor = (c) => (luminance(c) > 140 ? "#141414" : "#e6e6e6");
const mix = (...cs) => [0, 1, 2].map((k) => Math.round(cs.reduce((s, c) => s + c[k], 0) / cs.length));


// ══════════════════════════════════════════════════════════════════════════════
// Détection de mouvement → blobs
// ══════════════════════════════════════════════════════════════════════════════

class MotionDetector {
  constructor(w, h) {
    this.w = w;
    this.h = h;
    const n = w * h;
    this.bg     = new Float32Array(n * 3);
    this.mask   = new Uint8Array(n);
    this.tmp    = new Uint8Array(n);
    this.tmp2   = new Uint8Array(n);
    this.labels = new Int32Array(n);
    this.stack  = new Int32Array(n);
    this.frames = 0;
  }

  // morphologie 3×3 séparable : op = max (dilatation) ou min (érosion)
  _morph(src, dst, dilate) {
    const { w, h, tmp2: t } = this;
    const pick = dilate ? (a, b, c) => a | b | c : (a, b, c) => a & b & c;
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const l = x > 0 ? src[row + x - 1] : src[row + x];
        const r = x < w - 1 ? src[row + x + 1] : src[row + x];
        t[row + x] = pick(l, src[row + x], r);
      }
    }
    for (let y = 0; y < h; y++) {
      const up = y > 0 ? -w : 0;
      const dn = y < h - 1 ? w : 0;
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        dst[i] = pick(t[i + up], t[i], t[i + dn]);
      }
    }
  }

  // data : RGBA de l'image réduite. Retourne les blobs en coordonnées réduites.
  detect(data) {
    const { w, h, bg, mask } = this;
    const n = w * h;

    if (this.frames++ === 0) {
      for (let i = 0; i < n; i++) {
        bg[i * 3] = data[i * 4]; bg[i * 3 + 1] = data[i * 4 + 1]; bg[i * 3 + 2] = data[i * 4 + 2];
      }
      return [];
    }

    let moving = 0;
    for (let i = 0; i < n; i++) {
      const p = i * 4, q = i * 3;
      const d = Math.abs(data[p] - bg[q]) + Math.abs(data[p + 1] - bg[q + 1]) + Math.abs(data[p + 2] - bg[q + 2]);
      mask[i] = d > DIFF_THRESH ? 1 : 0;
      moving += mask[i];
    }

    const rate = moving > n * 0.5 ? BG_RATE_FAST : BG_RATE;
    for (let i = 0; i < n; i++) {
      const p = i * 4, q = i * 3;
      bg[q]     += rate * (data[p]     - bg[q]);
      bg[q + 1] += rate * (data[p + 1] - bg[q + 1]);
      bg[q + 2] += rate * (data[p + 2] - bg[q + 2]);
    }
    if (this.frames < WARMUP_FRAMES || rate === BG_RATE_FAST) return [];

    // close ×2 puis open ×1 (comme le Python)
    const a = mask, b = this.tmp;
    this._morph(a, b, true);  this._morph(b, a, true);
    this._morph(a, b, false); this._morph(b, a, false);
    this._morph(a, b, false); this._morph(b, a, true);

    return this._components(data);
  }

  // composantes connexes (4-voisinage) par remplissage
  _components(data) {
    const { w, h, mask, labels, stack } = this;
    labels.fill(0);
    const minArea = MIN_BLOB_FRAC * w * h;
    const blobs = [];
    let label = 0;

    for (let start = 0; start < w * h; start++) {
      if (!mask[start] || labels[start]) continue;
      label++;
      let sp = 0, area = 0, sx = 0, sy = 0;
      stack[sp++] = start;
      labels[start] = label;
      while (sp) {
        const i = stack[--sp];
        const x = i % w, y = (i / w) | 0;
        area++; sx += x; sy += y;
        if (x > 0     && mask[i - 1] && !labels[i - 1]) { labels[i - 1] = label; stack[sp++] = i - 1; }
        if (x < w - 1 && mask[i + 1] && !labels[i + 1]) { labels[i + 1] = label; stack[sp++] = i + 1; }
        if (y > 0     && mask[i - w] && !labels[i - w]) { labels[i - w] = label; stack[sp++] = i - w; }
        if (y < h - 1 && mask[i + w] && !labels[i + w]) { labels[i + w] = label; stack[sp++] = i + w; }
      }
      if (area >= minArea) blobs.push({ x: sx / area, y: sy / area, area });
    }

    blobs.sort((p, q) => q.area - p.area);
    return blobs.slice(0, MAX_BLOBS);
  }
}

function sampleColor(data, w, h, cx, cy, radius) {
  const x1 = Math.max(0, Math.round(cx - radius)), x2 = Math.min(w, Math.round(cx + radius) + 1);
  const y1 = Math.max(0, Math.round(cy - radius)), y2 = Math.min(h, Math.round(cy + radius) + 1);
  let r = 0, g = 0, b = 0, n = 0;
  for (let y = y1; y < y2; y++) {
    for (let x = x1; x < x2; x++) {
      const p = (y * w + x) * 4;
      r += data[p]; g += data[p + 1]; b += data[p + 2]; n++;
    }
  }
  return n ? [Math.round(r / n), Math.round(g / n), Math.round(b / n)] : [128, 128, 128];
}


// ══════════════════════════════════════════════════════════════════════════════
// Suivi des blobs (pour les traînées)
// ══════════════════════════════════════════════════════════════════════════════

class Tracker {
  constructor() { this.tracks = []; }

  update(blobs, maxDist) {
    const free = new Set(this.tracks);
    for (const b of blobs) {
      let best = null, bestD = maxDist;
      for (const t of free) {
        const d = Math.hypot(t.x - b.x, t.y - b.y);
        if (d < bestD) { best = t; bestD = d; }
      }
      if (best) free.delete(best);
      else this.tracks.push(best = { trail: [], missed: 0 });
      best.x = b.x; best.y = b.y; best.missed = 0;
      best.trail.push({ x: b.x, y: b.y, color: b.color });
      if (best.trail.length > TRAIL_LEN) best.trail.shift();
      b.trail = best.trail;
    }
    for (const t of free) t.missed++;
    this.tracks = this.tracks.filter((t) => t.missed < 5);
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// Modes de rendu
// ══════════════════════════════════════════════════════════════════════════════

// ── Mode 1 : Default ─────────────────────────────────────────────────────────

function renderDefault(ctx, blobs, u) {
  if (blobs.length >= 3) {
    const del = Delaunay.from(blobs, (b) => b.x, (b) => b.y);
    ctx.globalAlpha = 0.35;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = u;
    ctx.beginPath();
    del.render(ctx);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // connexions entre blobs proches
  const maxD = LINE_DIST * u;
  ctx.lineWidth = u;
  for (let i = 0; i < blobs.length; i++) {
    for (let j = i + 1; j < blobs.length; j++) {
      const a = blobs[i], b = blobs[j];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d >= maxD) continue;
      ctx.globalAlpha = (180 / 255) * (1 - d / maxD);
      ctx.strokeStyle = rgb(mix(a.color, b.color));
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;

  const font = Math.max(10, 10 * u);
  ctx.font = `${font}px ui-monospace, Menlo, Consolas, monospace`;
  ctx.textBaseline = "alphabetic";

  for (const b of blobs) {
    const { x, y, r, color } = b;

    // traînée
    const tr = b.trail || [];
    ctx.lineWidth = 2 * u;
    ctx.lineCap = "round";
    for (let i = 1; i < tr.length; i++) {
      ctx.globalAlpha = (60 / 255) * (i / tr.length) * 2.5;
      ctx.strokeStyle = rgb(tr[i].color);
      ctx.beginPath(); ctx.moveTo(tr[i - 1].x, tr[i - 1].y); ctx.lineTo(tr[i].x, tr[i].y); ctx.stroke();
    }

    // halo
    ctx.globalAlpha = 0.3;
    ctx.strokeStyle = rgb(color);
    ctx.lineWidth = u;
    ctx.beginPath(); ctx.arc(x, y, r + 10 * u, 0, Math.PI * 2); ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.lineWidth = 2 * u;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = rgb(color);
    ctx.beginPath(); ctx.arc(x, y, 4 * u, 0, Math.PI * 2); ctx.fill();

    // label hex
    const label = toHex(color);
    const tw = ctx.measureText(label).width;
    const lx = x + r + 6 * u, ly = y - 4 * u;
    ctx.fillRect(lx - 3 * u, ly - font, tw + 6 * u, font + 4 * u);
    ctx.fillStyle = textColor(color);
    ctx.fillText(label, lx, ly);
    ctx.fillStyle = "#b4b4b4";
    ctx.font = `${font * 0.85}px ui-monospace, Menlo, Consolas, monospace`;
    ctx.fillText(`${Math.round(x)},${Math.round(y)}`, lx, ly + font + 4 * u);
    ctx.font = `${font}px ui-monospace, Menlo, Consolas, monospace`;
  }
}

// ── Mode 2 : Loupe ───────────────────────────────────────────────────────────

function renderLoupe(ctx, source, blobs, u, zoom = 2.5) {
  for (const { x, y, r, color } of blobs) {
    if (r < 2) continue;
    const srcR = Math.max(r / zoom, 8 * u);

    ctx.save();
    ctx.beginPath(); ctx.arc(x, y, r - 1, 0, Math.PI * 2); ctx.clip();
    ctx.drawImage(source, x - srcR, y - srcR, srcR * 2, srcR * 2, x - r, y - r, r * 2, r * 2);
    ctx.restore();

    // bague
    ctx.strokeStyle = rgb(color);
    ctx.lineWidth = 2 * u;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = "#dcdcdc";
    ctx.lineWidth = u;
    ctx.beginPath(); ctx.arc(x, y, r + 3 * u, 0, Math.PI * 2); ctx.stroke();

    // reflet
    ctx.strokeStyle = "#fff";
    ctx.beginPath();
    ctx.ellipse(x - r / 3, y - r / 3, r / 5, r / 8, -Math.PI / 6, 0, Math.PI);
    ctx.stroke();
  }
}

// ── Mode 3 : Vitrail ─────────────────────────────────────────────────────────

function renderVitrail(ctx, blobs, u, alpha = 0.55) {
  if (blobs.length < 3) return;
  const { triangles } = Delaunay.from(blobs, (b) => b.x, (b) => b.y);

  ctx.globalAlpha = alpha;
  ctx.strokeStyle = "#1e1e1e";
  ctx.lineWidth = u;
  for (let t = 0; t < triangles.length; t += 3) {
    const a = blobs[triangles[t]], b = blobs[triangles[t + 1]], c = blobs[triangles[t + 2]];
    ctx.fillStyle = rgb(mix(a.color, b.color, c.color));
    ctx.beginPath();
    ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.lineTo(c.x, c.y); ctx.closePath();
    ctx.fill(); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  ctx.fillStyle = "#fff";
  for (const b of blobs) { ctx.beginPath(); ctx.arc(b.x, b.y, 4 * u, 0, Math.PI * 2); ctx.fill(); }
}

// ── Mode 4 : Voronoï ─────────────────────────────────────────────────────────

function renderVoronoi(ctx, blobs, u, w, h, alpha = 0.45) {
  if (blobs.length < 2) return;
  const vor = Delaunay.from(blobs, (b) => b.x, (b) => b.y).voronoi([0, 0, w, h]);

  ctx.globalAlpha = alpha;
  ctx.strokeStyle = "#c8c8c8";
  ctx.lineWidth = u;
  blobs.forEach((b, i) => {
    ctx.fillStyle = rgb(b.color);
    ctx.beginPath();
    vor.renderCell(i, ctx);
    ctx.fill(); ctx.stroke();
  });
  ctx.globalAlpha = 1;

  for (const b of blobs) {
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = rgb(b.color);
    ctx.lineWidth = 2 * u;
    ctx.beginPath(); ctx.arc(b.x, b.y, 5 * u, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// Gestes — MediaPipe Hand Landmarker
// ══════════════════════════════════════════════════════════════════════════════

async function makeDetector() {
  const { FilesetResolver, HandLandmarker } = await import(`${MP_URL}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${MP_URL}/wasm`);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch {
    return await HandLandmarker.createFromOptions(fileset, opts("CPU"));
  }
}

// doigts levés (index → auriculaire), 0-4
function countFingers(lm) {
  return [[8, 6], [12, 10], [16, 14], [20, 18]].filter(([tip, pip]) => lm[tip].y < lm[pip].y).length;
}

class GestureModeSwitcher {
  constructor() {
    this.modeIdx = 0;
    this._pending = -1;
    this._holdStart = 0;
  }

  // fingers : 1-4 ou -1. Retourne { target, progress }.
  update(fingers, now) {
    const target = fingers - 1;
    if (target < 0 || target >= MODES.length || target === this.modeIdx) {
      this._pending = -1;
      return { target: -1, progress: 0 };
    }
    if (fingers !== this._pending) {
      this._pending = fingers;
      this._holdStart = now;
    }
    const progress = Math.min((now - this._holdStart) / GESTURE_HOLD_MS, 1);
    if (progress >= 1) {
      this.modeIdx = target;
      this._pending = -1;
    }
    return { target, progress };
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// HUD
// ══════════════════════════════════════════════════════════════════════════════

function drawHud(ctx, w, h, u, modeIdx, nBlobs, gesture, hand) {
  const font = Math.max(12, 13 * u);
  ctx.font = `${font}px ui-monospace, Menlo, Consolas, monospace`;
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = "#b4b4b4";
  ctx.textAlign = "left";
  ctx.fillText(`Mode : ${MODE_LABELS[modeIdx]}`, 10 * u, h - 12 * u);
  ctx.fillStyle = "#787878";
  ctx.textAlign = "right";
  ctx.fillText(`blobs: ${nBlobs}`, w - 10 * u, h - 12 * u);
  ctx.textAlign = "left";

  if (gesture.progress > 0 && hand) {
    ctx.strokeStyle = ctx.fillStyle = "#50dc78";
    ctx.lineWidth = 3 * u;
    ctx.beginPath();
    ctx.arc(hand.x, hand.y, 28 * u, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * gesture.progress);
    ctx.stroke();
    ctx.fillText(MODE_LABELS[gesture.target], hand.x + 34 * u, hand.y + 5 * u);
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// Application
// ══════════════════════════════════════════════════════════════════════════════

const $ = (s) => document.querySelector(s);
const video  = $("#video");
const out    = $("#out");
const ctx    = out.getContext("2d");
const status = $("#status");

const frameCanvas = document.createElement("canvas");   // image propre (miroir), source de la loupe
const frameCtx    = frameCanvas.getContext("2d");
const procCanvas  = document.createElement("canvas");   // image réduite pour la détection
const procCtx     = procCanvas.getContext("2d", { willReadFrequently: true });

const state = {
  facing: "user",
  stream: null,
  motion: null,
  tracker: new Tracker(),
  switcher: new GestureModeSwitcher(),
  detector: null,
  gestures: true,
  lastTime: -1,
  recorder: null,
};

const initialMode = MODES.indexOf(new URLSearchParams(location.search).get("mode"));
if (initialMode >= 0) state.switcher.modeIdx = initialMode;

function setStatus(msg) { status.textContent = msg || ""; }

function syncButtons() {
  document.querySelectorAll("[data-mode]").forEach((b) =>
    b.classList.toggle("active", +b.dataset.mode === state.switcher.modeIdx));
  $("#btn-gesture").classList.toggle("active", state.gestures);
}

async function startCamera() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: state.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = state.stream;
  await video.play();

  const W = video.videoWidth, H = video.videoHeight;
  out.width = frameCanvas.width = W;
  out.height = frameCanvas.height = H;
  const s = PROC_SIZE / Math.max(W, H);
  procCanvas.width = Math.round(W * s);
  procCanvas.height = Math.round(H * s);
  state.motion = new MotionDetector(procCanvas.width, procCanvas.height);
  state.tracker = new Tracker();
  state.lastTime = -1;
}

function tick() {
  requestAnimationFrame(tick);
  if (!state.motion || video.readyState < 2 || video.currentTime === state.lastTime) return;
  state.lastTime = video.currentTime;

  const W = out.width, H = out.height;
  const u = Math.max(W, H) / REF_W;   // unité de dessin (1 = pixel en 640px)
  const mirror = state.facing === "user";

  // 1. image propre (miroir si caméra frontale)
  frameCtx.save();
  if (mirror) { frameCtx.translate(W, 0); frameCtx.scale(-1, 1); }
  frameCtx.drawImage(video, 0, 0, W, H);
  frameCtx.restore();

  // 2. détection de mouvement sur l'image réduite
  const pw = procCanvas.width, ph = procCanvas.height;
  procCtx.drawImage(frameCanvas, 0, 0, pw, ph);
  const data = procCtx.getImageData(0, 0, pw, ph).data;
  const scale = W / pw;
  const blobs = state.motion.detect(data).map((b) => ({
    x: (b.x + 0.5) * scale,
    y: (b.y + 0.5) * scale,
    r: Math.sqrt((b.area * scale * scale) / Math.PI),
    color: sampleColor(data, pw, ph, b.x, b.y, (SAMPLE_RADIUS * u) / scale),
  }));
  state.tracker.update(blobs, 80 * u);

  // 3. geste
  let hand = null, fingers = -1;
  if (state.gestures && state.detector) {
    const res = state.detector.detectForVideo(video, performance.now());
    const lm = res.landmarks?.[0];
    if (lm) {
      fingers = countFingers(lm);
      hand = { x: (mirror ? 1 - lm[9].x : lm[9].x) * W, y: lm[9].y * H };
    }
  }
  const before = state.switcher.modeIdx;
  const gesture = state.switcher.update(fingers, performance.now());
  if (state.switcher.modeIdx !== before) syncButtons();

  // 4. rendu
  ctx.drawImage(frameCanvas, 0, 0);
  ctx.fillStyle = `rgba(0,0,0,${DARKEN})`;
  ctx.fillRect(0, 0, W, H);

  switch (MODES[state.switcher.modeIdx]) {
    case "default": renderDefault(ctx, blobs, u); break;
    case "loupe":   renderLoupe(ctx, frameCanvas, blobs, u); break;
    case "vitrail": renderVitrail(ctx, blobs, u); break;
    case "voronoi": renderVoronoi(ctx, blobs, u, W, H); break;
  }
  drawHud(ctx, W, H, u, state.switcher.modeIdx, blobs.length, gesture, hand);
}


// ── Enregistrement ───────────────────────────────────────────────────────────

function toggleRecording() {
  const btn = $("#btn-rec");
  if (state.recorder) {
    state.recorder.stop();
    return;
  }
  const type = ["video/mp4", "video/webm;codecs=vp9", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
  const rec = new MediaRecorder(out.captureStream(30), type ? { mimeType: type } : undefined);
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    const blob = new Blob(chunks, { type: rec.mimeType });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `blob_art_${MODES[state.switcher.modeIdx]}_${Date.now()}.${rec.mimeType.includes("mp4") ? "mp4" : "webm"}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    state.recorder = null;
    btn.classList.remove("rec");
    btn.textContent = "●";
  };
  rec.start(1000);
  state.recorder = rec;
  btn.classList.add("rec");
  btn.textContent = "■";
}


// ── UI ───────────────────────────────────────────────────────────────────────

$("#btn-start").addEventListener("click", async () => {
  $("#btn-start").disabled = true;
  try {
    await startCamera();
  } catch (e) {
    $("#btn-start").disabled = false;
    $("#start p").textContent = `Caméra inaccessible : ${e.message}. La page doit être servie en HTTPS (ou localhost).`;
    return;
  }
  $("#start").remove();
  $("#bar").classList.remove("hidden");
  syncButtons();
  requestAnimationFrame(tick);

  setStatus("chargement du modèle main…");
  makeDetector()
    .then((d) => { state.detector = d; setStatus(""); })
    .catch((e) => { console.warn(e); state.gestures = false; syncButtons(); setStatus("gestes indisponibles — utilise les boutons"); });
});

document.querySelectorAll("[data-mode]").forEach((b) =>
  b.addEventListener("click", () => { state.switcher.modeIdx = +b.dataset.mode; syncButtons(); }));

$("#btn-gesture").addEventListener("click", () => {
  if (!state.detector) return;
  state.gestures = !state.gestures;
  syncButtons();
});

$("#btn-flip").addEventListener("click", async () => {
  state.facing = state.facing === "user" ? "environment" : "user";
  try { await startCamera(); } catch (e) { setStatus(`caméra : ${e.message}`); }
});

$("#btn-rec").addEventListener("click", toggleRecording);

// tap sur l'image (mobile) ou touche H : masquer / afficher la barre
out.addEventListener("click", () => $("#bar").classList.toggle("hidden"));

addEventListener("keydown", (e) => {
  const n = +e.key;
  if (n >= 1 && n <= MODES.length) { state.switcher.modeIdx = n - 1; syncButtons(); }
  if (e.key === "h") $("#bar").classList.toggle("hidden");
});
