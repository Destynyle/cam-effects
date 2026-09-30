/*
 * cam-effects — rendu vidéo
 * ─────────────────────────
 * Applique les effets à un fichier vidéo, image par image, dans le navigateur.
 * Décodage / réencodage MP4 + son d'origine : Mediabunny (WebCodecs).
 * Chaque effet coché (et chaque variante) produit une vidéo téléchargeable.
 */

import {
  Input, Output, Conversion, BlobSource, BufferTarget, Mp4OutputFormat, ALL_FORMATS, QUALITY_HIGH,
} from "https://cdn.jsdelivr.net/npm/mediabunny@1.61.0/+esm";
import { EFFECTS } from "./effects/index.js";
import { ensureModel, detectHands, segment } from "./mediapipe.js";

const REF_W = 640;   // les tailles des effets sont pensées pour 640px

const $ = (s) => document.querySelector(s);
const state = { file: null, info: null, jobs: [], running: false, cancelled: false, current: null };


// ══════════════════════════════════════════════════════════════════════════════
// Choix des effets
// ══════════════════════════════════════════════════════════════════════════════

for (const def of EFFECTS) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<h3>${def.label}</h3><div class="chips"></div>`;
  const variants = def.variants ?? [{ id: "", label: "Normal", params: {} }];
  for (const v of variants) {
    const chip = document.createElement("label");
    chip.className = "chip";
    chip.innerHTML = `<input type="checkbox"><span>${v.label}</span>`;
    const box = chip.querySelector("input");
    box.value = JSON.stringify([def.id, v.id]);
    box.checked = def.id === "blob_art" && v.id === "default";
    box.addEventListener("change", updateButton);
    card.querySelector(".chips").append(chip);
  }
  $("#effects").append(card);
}

function selected() {
  return [...document.querySelectorAll("#effects input:checked")].map((b) => {
    const [defId, vId] = JSON.parse(b.value);
    const def = EFFECTS.find((e) => e.id === defId);
    const variant = (def.variants ?? []).find((v) => v.id === vId) ?? null;
    return { def, variant };
  });
}

function updateButton() {
  const n = selected().length;
  const btn = $("#btn-render");
  btn.disabled = state.running || !state.file || n === 0;
  btn.textContent = !state.file ? "Choisis une vidéo"
    : n === 0 ? "Coche au moins un effet"
    : `Lancer ${n} rendu${n > 1 ? "s" : ""}`;
}


// ══════════════════════════════════════════════════════════════════════════════
// Fichier
// ══════════════════════════════════════════════════════════════════════════════

const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const fmtSize = (b) => (b > 1e9 ? `${(b / 1e9).toFixed(1)} Go` : `${(b / 1e6).toFixed(1)} Mo`);

async function loadFile(file) {
  if (!file) return;
  state.file = null;
  $("#file-info").textContent = "lecture…";
  updateButton();
  try {
    const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
    const video = await input.getPrimaryVideoTrack();
    if (!video) throw new Error("aucune piste vidéo");
    const audio = await input.getPrimaryAudioTrack();
    const duration = await input.computeDuration();
    state.info = { width: video.displayWidth, height: video.displayHeight, duration, hasAudio: !!audio };
    state.file = file;
    $("#file-info").textContent =
      `${file.name} · ${video.displayWidth}×${video.displayHeight} · ${fmtTime(duration)} · ` +
      `${audio ? "avec son" : "sans son"} · ${fmtSize(file.size)}`;
  } catch (e) {
    console.error(e);
    $("#file-info").textContent = `Impossible de lire ce fichier : ${e.message}`;
  }
  updateButton();
}

$("#file").addEventListener("change", (e) => loadFile(e.target.files[0]));
const drop = $("#drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  loadFile(e.dataTransfer.files[0]);
});


// ══════════════════════════════════════════════════════════════════════════════
// Rendu
// ══════════════════════════════════════════════════════════════════════════════

// dimensions de sortie : petit côté = résolution choisie, valeurs paires
function outputSize() {
  const { width, height } = state.info;
  const target = +$("#res").value;
  const s = target && Math.min(width, height) > target ? target / Math.min(width, height) : 1;
  const even = (v) => Math.max(2, Math.round((v * s) / 2) * 2);
  return { W: even(width), H: even(height) };
}

const frameCanvas = document.createElement("canvas");   // image source, propre
const frameCtx    = frameCanvas.getContext("2d");
const outCanvas   = document.createElement("canvas");   // image avec l'effet (encodée + aperçu)
const outCtx      = outCanvas.getContext("2d");

async function renderJob(job, W, H, keepAudio) {
  const { def, variant } = job;
  const u = Math.max(W, H) / REF_W;
  const fx = def.create({
    W, H, u,
    params: new URLSearchParams(variant?.params ?? {}),
    refreshControls() {},
    setStatus(msg) { if (msg) setJob(job, msg); },
  });
  const needHands = !!def.needs?.hands && fx.useHands !== false;
  const needSeg = !!def.needs?.segmenter;
  const status = (msg) => msg && setJob(job, msg);
  if (needHands) await ensureModel("hands", status);
  if (needSeg) await ensureModel("segmenter", status);
  await fx.ready;

  frameCanvas.width = outCanvas.width = W;
  frameCanvas.height = outCanvas.height = H;
  let seg = null;

  const input = new Input({ source: new BlobSource(state.file), formats: ALL_FORMATS });
  const output = new Output({ format: new Mp4OutputFormat({ fastStart: "in-memory" }), target: new BufferTarget() });
  const conversion = await Conversion.init({
    input,
    output,
    video: {
      width: W, height: H, fit: "fill",
      allowTransformationMetadata: false,   // rotation appliquée aux images, pas en métadonnée
      forceTranscode: true,
      quality: QUALITY_HIGH,
      processedWidth: W, processedHeight: H,
      process(sample) {
        sample.draw(frameCtx, 0, 0, W, H);
        const hands = needHands ? detectHands(frameCanvas) : null;
        if (needSeg) seg = segment(frameCanvas, seg);
        outCtx.save();
        fx.frame({ ctx: outCtx, source: frameCanvas, W, H, u, now: sample.timestamp * 1000, hands, seg });
        outCtx.restore();
        return outCanvas;
      },
    },
    audio: keepAudio ? {} : { discard: true },
  });

  if (!conversion.isValid) {
    const why = conversion.discardedTracks.map((t) => `${t.track.type} : ${t.reason}`).join(", ");
    throw new Error(`conversion impossible (${why})`);
  }
  state.current = conversion;

  const t0 = performance.now();
  conversion.onProgress = (p) => {
    const elapsed = (performance.now() - t0) / 1000;
    const eta = p > 0.02 ? elapsed / p - elapsed : null;
    setJob(job, `${Math.round(p * 100)} %${eta !== null ? ` · reste ${fmtTime(eta)}` : ""}`, p);
  };
  await conversion.execute();
  state.current = null;

  return new Blob([output.target.buffer], { type: "video/mp4" });
}

function setJob(job, text, progress) {
  job.el.querySelector(".state").textContent = text;
  if (progress !== undefined) job.el.querySelector(".bar i").style.width = `${progress * 100}%`;
}

function fileName(job) {
  const base = state.file.name.replace(/\.[^.]+$/, "");
  return `${base}_${job.def.id}${job.variant ? `_${job.variant.id}` : ""}.mp4`;
}

async function renderAll() {
  const picks = selected();
  if (!state.file || !picks.length) return;
  const { W, H } = outputSize();
  const keepAudio = $("#audio").checked && state.info.hasAudio;

  state.running = true;
  state.cancelled = false;
  updateButton();
  $("#btn-cancel").disabled = false;
  $("#jobs-title").hidden = false;
  const preview = $("#preview");
  preview.style.display = "block";
  preview.replaceChildren(outCanvas);
  preview.scrollIntoView({ behavior: "smooth", block: "start" });

  const jobs = picks.map((p) => {
    const el = document.createElement("li");
    el.innerHTML = `<span class="name"></span><span class="state">en attente</span><span class="bar"><i></i></span>`;
    const job = { ...p, el };
    el.querySelector(".name").textContent = fileName(job);
    $("#jobs").prepend(el);
    return job;
  });

  for (const job of jobs) {
    if (state.cancelled) { setJob(job, "annulé"); continue; }
    try {
      setJob(job, "démarrage…", 0);
      const blob = await renderJob(job, W, H, keepAudio);
      const a = document.createElement("a");
      a.className = "dl";
      a.href = URL.createObjectURL(blob);
      a.download = fileName(job);
      a.textContent = `Télécharger · ${fmtSize(blob.size)}`;
      job.el.querySelector(".state").replaceChildren(a);
      job.el.classList.add("done");
    } catch (e) {
      console.error(e);
      state.current = null;
      job.el.classList.add("error");
      setJob(job, state.cancelled ? "annulé" : `erreur : ${e.message}`);
    }
  }

  state.running = false;
  $("#btn-cancel").disabled = true;
  preview.style.display = "none";
  updateButton();
}

$("#btn-render").addEventListener("click", renderAll);
addEventListener("beforeunload", (e) => { if (state.running) e.preventDefault(); });
$("#btn-cancel").addEventListener("click", async () => {
  state.cancelled = true;
  $("#btn-cancel").disabled = true;
  await state.current?.cancel();
});

// charge les modèles en avance pendant que l'utilisateur choisit
ensureModel("hands").catch(() => {});
