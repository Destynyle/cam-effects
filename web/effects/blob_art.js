/*
 * Blob Art — portage de effects/blob_art.py
 * ─────────────────────────────────────────
 * Détection de mouvement → blobs, couleur dominante + code hex,
 * triangulation entre blobs.
 *
 * Modes (1-4 doigts maintenus 1.5s, boutons, ou touches 1-4) :
 *   Default · Loupe · Vitrail · Voronoï
 */

import { Delaunay } from "https://cdn.jsdelivr.net/npm/d3-delaunay@6/+esm";
import { rgb, mono, circle } from "./common.js";

// ── Paramètres blobs (référence 640px de large, comme la version Python) ─────
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

  // morphologie 3×3 séparable : dilatation (OR) ou érosion (AND)
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

    return this._components();
  }

  // composantes connexes (4-voisinage) par remplissage
  _components() {
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
  ctx.font = mono(font);
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
    circle(ctx, x, y, r + 10 * u); ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.lineWidth = 2 * u;
    circle(ctx, x, y, r); ctx.stroke();
    ctx.fillStyle = rgb(color);
    circle(ctx, x, y, 4 * u); ctx.fill();

    // label hex
    const label = toHex(color);
    const tw = ctx.measureText(label).width;
    const lx = x + r + 6 * u, ly = y - 4 * u;
    ctx.fillRect(lx - 3 * u, ly - font, tw + 6 * u, font + 4 * u);
    ctx.fillStyle = textColor(color);
    ctx.fillText(label, lx, ly);
    ctx.fillStyle = "#b4b4b4";
    ctx.font = mono(font * 0.85);
    ctx.fillText(`${Math.round(x)},${Math.round(y)}`, lx, ly + font + 4 * u);
    ctx.font = mono(font);
  }
}

// ── Mode 2 : Loupe ───────────────────────────────────────────────────────────

function renderLoupe(ctx, source, blobs, u, zoom = 2.5) {
  for (const { x, y, r, color } of blobs) {
    if (r < 2) continue;
    const srcR = Math.max(r / zoom, 8 * u);

    ctx.save();
    circle(ctx, x, y, r - 1); ctx.clip();
    ctx.drawImage(source, x - srcR, y - srcR, srcR * 2, srcR * 2, x - r, y - r, r * 2, r * 2);
    ctx.restore();

    // bague
    ctx.strokeStyle = rgb(color);
    ctx.lineWidth = 2 * u;
    circle(ctx, x, y, r); ctx.stroke();
    ctx.strokeStyle = "#dcdcdc";
    ctx.lineWidth = u;
    circle(ctx, x, y, r + 3 * u); ctx.stroke();

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
  for (const b of blobs) { circle(ctx, b.x, b.y, 4 * u); ctx.fill(); }
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
    circle(ctx, b.x, b.y, 5 * u); ctx.fill(); ctx.stroke();
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// Gestes — changement de mode
// ══════════════════════════════════════════════════════════════════════════════

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

function drawHud(ctx, w, h, u, modeIdx, nBlobs, gesture, hand) {
  const font = Math.max(12, 13 * u);
  ctx.font = mono(font);
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
// Effet
// ══════════════════════════════════════════════════════════════════════════════

export default {
  id: "blob_art",
  label: "Blob Art",
  help: "Bouge devant la caméra : les zones en mouvement deviennent des blobs colorés.<br>" +
        "Lève 1 à 4 doigts et tiens 1,5 s pour changer de mode (ou touches 1-4).",
  needs: { hands: true },
  // rendu vidéo : mode fixe, pas de gestes (comme --mode en Python)
  variants: MODES.map((m, i) => ({ id: m, label: MODE_LABELS[i], params: { mode: m, gestures: "0" } })),

  create({ W, H, params, refreshControls }) {
    const s = PROC_SIZE / Math.max(W, H);
    const proc = document.createElement("canvas");
    proc.width = Math.round(W * s);
    proc.height = Math.round(H * s);
    const procCtx = proc.getContext("2d", { willReadFrequently: true });

    const motion = new MotionDetector(proc.width, proc.height);
    const tracker = new Tracker();
    const switcher = new GestureModeSwitcher();
    const initial = MODES.indexOf(params.get("mode"));
    if (initial >= 0) switcher.modeIdx = initial;

    const fx = {
      useHands: params.get("gestures") !== "0",

      controls: [
        ...MODE_LABELS.map((l, i) => ({ id: `mode${i}`, label: `${i + 1} ${l}` })),
        { id: "gestures", label: "✋", title: "Changer de mode avec les doigts" },
      ],
      isActive: (id) => (id === "gestures" ? fx.useHands : id === `mode${switcher.modeIdx}`),
      onControl(id) {
        if (id === "gestures") fx.useHands = !fx.useHands;
        else switcher.modeIdx = +id.slice(4);
      },
      onKey(key) {
        const n = +key;
        if (n >= 1 && n <= MODES.length) switcher.modeIdx = n - 1;
      },

      frame({ ctx, source, W, H, u, now, hands }) {
        // détection de mouvement sur l'image réduite
        const pw = proc.width, ph = proc.height;
        procCtx.drawImage(source, 0, 0, pw, ph);
        const data = procCtx.getImageData(0, 0, pw, ph).data;
        const scale = W / pw;
        const blobs = motion.detect(data).map((b) => ({
          x: (b.x + 0.5) * scale,
          y: (b.y + 0.5) * scale,
          r: Math.sqrt((b.area * scale * scale) / Math.PI),
          color: sampleColor(data, pw, ph, b.x, b.y, (SAMPLE_RADIUS * u) / scale),
        }));
        tracker.update(blobs, 80 * u);

        // geste
        let hand = null, fingers = -1;
        const h0 = hands?.[0];
        if (fx.useHands && h0) {
          fingers = countFingers(h0.lm);
          hand = h0.px[9];
        }
        const before = switcher.modeIdx;
        const gesture = switcher.update(fingers, now);
        if (switcher.modeIdx !== before) refreshControls();

        // rendu
        ctx.drawImage(source, 0, 0);
        ctx.fillStyle = `rgba(0,0,0,${DARKEN})`;
        ctx.fillRect(0, 0, W, H);

        switch (MODES[switcher.modeIdx]) {
          case "default": renderDefault(ctx, blobs, u); break;
          case "loupe":   renderLoupe(ctx, source, blobs, u); break;
          case "vitrail": renderVitrail(ctx, blobs, u); break;
          case "voronoi": renderVoronoi(ctx, blobs, u, W, H); break;
        }
        drawHud(ctx, W, H, u, switcher.modeIdx, blobs.length, gesture, hand);
      },
    };
    return fx;
  },
};
