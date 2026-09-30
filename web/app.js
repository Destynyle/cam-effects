/*
 * cam-effects — version web
 * ─────────────────────────
 * Noyau commun : caméra, boucle de rendu, modèles MediaPipe chargés à la
 * demande (mains, segmentation), enregistrement, sélection de l'effet.
 *
 * Chaque effet (effects/*.js) exporte :
 *   { id, label, help, needs: { hands?, segmenter? }, create(env) → instance }
 * instance :
 *   frame(f)          dessine une image (f = { ctx, source, W, H, u, now, hands, seg })
 *   controls          [{ id, label, title? }] — boutons propres à l'effet
 *   onControl(id)     clic sur un bouton
 *   isActive(id)      bouton surligné ?
 *   onKey(key)        touche clavier
 *   useHands          false = pas besoin des mains pour l'instant
 *   ready             promesse optionnelle : ressources chargées (attendue au rendu vidéo)
 * Optionnel sur l'effet : variants [{ id, label, params }] — versions proposées
 * au rendu vidéo (video.html) ; `params` arrive dans env.params de create().
 */

import { EFFECTS } from "./effects/index.js";
import { models, ensureModel, detectHands, segment } from "./mediapipe.js";

const REF_W = 640;   // les tailles des effets sont pensées pour 640px


// ══════════════════════════════════════════════════════════════════════════════
// État & DOM
// ══════════════════════════════════════════════════════════════════════════════

const $ = (s) => document.querySelector(s);
const video  = $("#video");
const out    = $("#out");
const ctx    = out.getContext("2d");

// image propre de la caméra (miroir en frontale) — source commune des effets
const frameCanvas = document.createElement("canvas");
const frameCtx    = frameCanvas.getContext("2d");

const params = new URLSearchParams(location.search);
const state = {
  facing: "user",
  stream: null,
  def: EFFECTS.find((e) => e.id === params.get("effect")) || EFFECTS[0],
  fx: null,
  lastTime: -1,
  seg: null,
  recorder: null,
  helpTimer: 0,
};

function setStatus(msg) { $("#status").textContent = msg || ""; }


// ══════════════════════════════════════════════════════════════════════════════
// Effets
// ══════════════════════════════════════════════════════════════════════════════

function env() {
  const W = out.width, H = out.height;
  return { W, H, u: Math.max(W, H) / REF_W, params, refreshControls: syncControls, setStatus };
}

function startEffect(def) {
  state.def = def;
  state.fx = def.create(env());
  state.seg = null;
  if (def.needs?.hands) ensureModel("hands", setStatus).catch(() => {});
  if (def.needs?.segmenter) ensureModel("segmenter", setStatus).catch(() => {});

  $("#effect").value = def.id;
  const url = new URL(location.href);
  url.searchParams.set("effect", def.id);
  if (params.get("effect") !== def.id) url.searchParams.delete("mode");
  history.replaceState(null, "", url);

  buildControls();
  showHelp(true, 7000);
}

function buildControls() {
  const box = $("#fx-controls");
  box.replaceChildren();
  for (const c of state.fx.controls || []) {
    const b = document.createElement("button");
    b.textContent = c.label;
    if (c.title) b.title = c.title;
    b.dataset.id = c.id;
    b.addEventListener("click", () => { state.fx.onControl?.(c.id); syncControls(); });
    box.append(b);
  }
  syncControls();
}

function syncControls() {
  const fx = state.fx;
  $("#fx-controls").querySelectorAll("button").forEach((b) =>
    b.classList.toggle("active", !!fx.isActive?.(b.dataset.id)));
  // les libellés peuvent changer (ex. nombre de clones)
  for (const c of fx.controls || []) {
    const b = $(`#fx-controls [data-id="${c.id}"]`);
    if (b && b.textContent !== c.label) b.textContent = c.label;
  }
}

function showHelp(show, autoHideMs) {
  const help = $("#help");
  clearTimeout(state.helpTimer);
  help.innerHTML = `<b>${state.def.label}</b><br>${state.def.help}`;
  help.classList.toggle("hidden", !show);
  if (show && autoHideMs) state.helpTimer = setTimeout(() => help.classList.add("hidden"), autoHideMs);
}


// ══════════════════════════════════════════════════════════════════════════════
// Caméra & boucle
// ══════════════════════════════════════════════════════════════════════════════

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
  state.lastTime = -1;
  startEffect(state.def);   // la résolution a pu changer : on recrée l'effet
}

function tick() {
  requestAnimationFrame(tick);
  if (!state.fx || video.readyState < 2 || video.currentTime === state.lastTime) return;
  state.lastTime = video.currentTime;

  const W = out.width, H = out.height;
  const now = performance.now();
  const def = state.def, fx = state.fx;

  frameCtx.save();
  if (state.facing === "user") { frameCtx.translate(W, 0); frameCtx.scale(-1, 1); }
  frameCtx.drawImage(video, 0, 0, W, H);
  frameCtx.restore();

  let hands = null;
  if (def.needs?.hands && fx.useHands !== false && models.hands) hands = detectHands(frameCanvas);
  if (def.needs?.segmenter && models.segmenter) state.seg = segment(frameCanvas, state.seg);

  ctx.save();
  fx.frame({ ctx, source: frameCanvas, W, H, u: Math.max(W, H) / REF_W, now, hands, seg: state.seg });
  ctx.restore();
}


// ══════════════════════════════════════════════════════════════════════════════
// Enregistrement
// ══════════════════════════════════════════════════════════════════════════════

function toggleRecording() {
  const btn = $("#btn-rec");
  if (state.recorder) { state.recorder.stop(); return; }

  const type = ["video/mp4", "video/webm;codecs=vp9", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
  const rec = new MediaRecorder(out.captureStream(30), type ? { mimeType: type } : undefined);
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    const blob = new Blob(chunks, { type: rec.mimeType });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${state.def.id}_${Date.now()}.${rec.mimeType.includes("mp4") ? "mp4" : "webm"}`;
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


// ══════════════════════════════════════════════════════════════════════════════
// UI
// ══════════════════════════════════════════════════════════════════════════════

const select = $("#effect");
for (const e of EFFECTS) select.add(new Option(e.label, e.id));
select.value = state.def.id;
select.addEventListener("change", () => startEffect(EFFECTS.find((e) => e.id === select.value)));

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
  requestAnimationFrame(tick);
});

$("#btn-flip").addEventListener("click", async () => {
  state.facing = state.facing === "user" ? "environment" : "user";
  try { await startCamera(); } catch (e) { setStatus(`caméra : ${e.message}`); }
});
$("#btn-rec").addEventListener("click", toggleRecording);
$("#btn-help").addEventListener("click", () => showHelp($("#help").classList.contains("hidden")));

// tap sur l'image (mobile) ou touche H : masquer / afficher la barre
out.addEventListener("click", () => $("#bar").classList.toggle("hidden"));

addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === "h") { $("#bar").classList.toggle("hidden"); return; }
  if (e.key === "?") { $("#btn-help").click(); return; }
  state.fx?.onKey?.(e.key);
  syncControls();
});

// script prêt : on peut démarrer
$("#btn-start").disabled = false;
$("#btn-start").textContent = "Démarrer la caméra";
