/*
 * Ninjutsu — portage de effects/ninjutsu.py
 * ─────────────────────────────────────────
 * Reconnaissance des signes Naruto (KNN exporté par export_knn.py)
 * et déclenchement des jutsus sur les vraies séquences.
 */

import { HAND_CONNECTIONS, rgb, sans, circle, rand, randInt } from "./common.js";

const KNN_URL = new URL("../models/signs_knn", import.meta.url).href;

// ── Combos (vraies séquences Naruto) ─────────────────────────────────────────
const JUTSU_COMBOS = [
  [["Snake", "Ram", "Monkey", "Boar", "Horse", "Tiger"], "Katon : Goukakyuu no Jutsu"],
  [["Ram", "Boar", "Ox", "Dog", "Snake"],                "Kage Bunshin no Jutsu"],
  [["Ox", "Hare", "Monkey"],                             "Chidori"],
  [["Boar", "Dog", "Bird", "Monkey", "Ram"],             "Kuchiyose no Jutsu"],
  [["Dog", "Boar", "Ram"],                               "Doton : Doryuheki"],
];

const COLORS = {
  "Katon : Goukakyuu no Jutsu": [255, 60, 0],     // rouge feu
  "Kage Bunshin no Jutsu":      [255, 255, 0],    // jaune
  "Chidori":                    [0, 220, 255],    // bleu électrique
  "Kuchiyose no Jutsu":         [120, 180, 0],    // vert fumée
  "Doton : Doryuheki":          [160, 100, 30],   // marron terre
};

const COMBO_TIMEOUT    = 6.0;
const GESTURE_DEBOUNCE = 0.5;
const EFFECT_DURATION  = 3.0;
const VOTE_WINDOW      = 7;     // frames pour le vote temporel
const VOTE_THRESHOLD   = 4;     // confirmations requises


// ══════════════════════════════════════════════════════════════════════════════
// KNN
// ══════════════════════════════════════════════════════════════════════════════

let knnPromise = null;
function loadKnn() {
  knnPromise ??= (async () => {
    const meta = await (await fetch(`${KNN_URL}.json`)).json();
    const buf  = await (await fetch(`${KNN_URL}.bin`)).arrayBuffer();
    const X = new Int8Array(buf, 0, meta.count * meta.dim);
    const y = new Uint8Array(buf, meta.count * meta.dim, meta.count);
    return { ...meta, X, y };
  })();
  return knnPromise;
}

// équivalent de knn.predict_proba (poids uniformes)
function predict(knn, features) {
  const { X, y, dim, count, k, scale, labels } = knn;
  const q = features.map((v) => v * scale);
  const bestD = new Float64Array(k).fill(Infinity);
  const bestY = new Int32Array(k).fill(-1);

  for (let n = 0, off = 0; n < count; n++, off += dim) {
    let d = 0;
    for (let j = 0; j < dim; j++) {
      const t = X[off + j] - q[j];
      d += t * t;
      if (d >= bestD[k - 1]) break;
    }
    if (d >= bestD[k - 1]) continue;
    let i = k - 1;
    while (i > 0 && bestD[i - 1] > d) { bestD[i] = bestD[i - 1]; bestY[i] = bestY[i - 1]; i--; }
    bestD[i] = d; bestY[i] = y[n];
  }

  const votes = new Array(labels.length).fill(0);
  for (const v of bestY) if (v >= 0) votes[v]++;
  let best = 0;
  for (let i = 1; i < votes.length; i++) if (votes[i] > votes[best]) best = i;
  return { sign: labels[best], conf: votes[best] / k };
}


// ══════════════════════════════════════════════════════════════════════════════
// Normalisation des landmarks (même logique que le dataset)
// ══════════════════════════════════════════════════════════════════════════════

function normalizeHand(lm) {
  const w = lm[0], m = lm[9];
  let dist = Math.hypot(w.x - m.x, w.y - m.y, w.z - m.z);
  if (dist < 0.0001) dist = 1;
  const out = [];
  for (const p of lm) out.push((p.x - w.x) / dist, (p.y - w.y) / dist, (p.z - w.z) / dist);
  return out;
}

const palmCenterX = (lm) => [0, 5, 9, 13, 17].reduce((s, i) => s + lm[i].x, 0) / 5;

// vecteur 126D, mains triées gauche → droite, imputation des mains manquantes
function buildFeatures(hands, slots) {
  const det = hands.slice(0, 2).map((h) => ({ coords: normalizeHand(h.lm), cx: palmCenterX(h.lm) }));
  if (det.length === 2 && det[0].cx > det[1].cx) det.reverse();

  const features = [];
  for (let s = 0; s < 2; s++) {
    if (s < det.length) {
      slots[s] = det[s].coords;
      features.push(...det[s].coords);
    } else {
      features.push(...(slots[s] ?? new Array(63).fill(0)));
    }
  }
  return features;
}


// ══════════════════════════════════════════════════════════════════════════════
// Effets visuels
// ══════════════════════════════════════════════════════════════════════════════

function spawnFire(particles, cx, cy, u) {
  for (let i = 0; i < 15; i++) {
    const a = rand(0, 2 * Math.PI), s = rand(5, 18) * u;
    particles.push({
      x: cx, y: cy, vx: Math.cos(a) * s, vy: Math.sin(a) * s - 4 * u,
      color: [255, randInt(80, 200), randInt(0, 60)],
      life: randInt(25, 55), size: randInt(5, 14) * u,
    });
  }
}

function spawnLightning(particles, cx, cy, u) {
  for (let i = 0; i < 8; i++) {
    const a = rand(0, 2 * Math.PI), s = rand(8, 20) * u;
    particles.push({
      x: cx, y: cy, vx: Math.cos(a) * s, vy: Math.sin(a) * s,
      color: [randInt(0, 60), randInt(180, 255), 255],
      life: randInt(8, 18), size: randInt(2, 5) * u,
    });
  }
}

function updateParticles(ctx, particles, u) {
  ctx.globalCompositeOperation = "lighter";
  for (const p of particles) {
    p.max ??= p.life;
    p.x += p.vx; p.y += p.vy; p.vy += 0.25 * u; p.life--;
    if (p.life <= 0) continue;
    const t = p.life / p.max;
    ctx.fillStyle = rgb(p.color, t);
    circle(ctx, p.x, p.y, Math.max(1, p.size * t)); ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";
  return particles.filter((p) => p.life > 0);
}

function drawLightning(ctx, x1, y1, x2, y2, u, segs = 8) {
  const pts = [[x1, y1]];
  for (let i = 1; i < segs; i++) {
    const t = i / segs;
    pts.push([x1 + (x2 - x1) * t + rand(-25, 25) * u, y1 + (y2 - y1) * t + rand(-25, 25) * u]);
  }
  pts.push([x2, y2]);
  const path = () => { ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); };
  ctx.strokeStyle = "rgb(0,220,255)"; ctx.lineWidth = 3 * u; path(); ctx.stroke();
  ctx.strokeStyle = "#fff";           ctx.lineWidth = u;     path(); ctx.stroke();
}

function drawCloneAuras(ctx, cx, cy, tick, u) {
  for (const dx of [-130 * u, 130 * u]) {
    const r = (70 + 8 * Math.sin(tick * 0.15)) * u;
    ctx.strokeStyle = "rgb(255,255,0)"; ctx.lineWidth = 2 * u;
    circle(ctx, cx + dx, cy, r); ctx.stroke();
    ctx.strokeStyle = "rgb(180,200,0)"; ctx.lineWidth = u;
    circle(ctx, cx + dx, cy, r - 15 * u); ctx.stroke();
  }
  ctx.strokeStyle = "rgb(255,200,0)";
  ctx.beginPath(); ctx.moveTo(cx - 130 * u, cy); ctx.lineTo(cx + 130 * u, cy); ctx.stroke();
}

function drawSmokePuff(ctx, cx, cy, tick, u) {
  for (let i = 0; i < 5; i++) {
    const r = 30 + i * 18 + tick * 1.5;
    const alpha = Math.max(0, 1 - r / 200);
    ctx.fillStyle = rgb([80, 120, 60], alpha * 0.3);
    circle(ctx, cx + rand(-20, 20) * u, cy - i * 15 * u, r * u); ctx.fill();
  }
}

function drawEarthWall(ctx, w, h, tick, u) {
  const wallH = h * 0.6 * Math.min(1, tick / 30);
  const x = w / 2 - 80 * u, y = h - wallH, ww = 160 * u;
  ctx.fillStyle = rgb([140, 80, 30], 0.6);
  ctx.fillRect(x, y, ww, wallH);
  ctx.strokeStyle = "rgb(180,120,50)"; ctx.lineWidth = 2 * u;
  ctx.strokeRect(x, y, ww, wallH);
}

function drawJutsuName(ctx, jutsu, elapsed, w, h, u) {
  const t = elapsed / EFFECT_DURATION;
  const alpha = Math.min(1, t * 4) * Math.max(0, 1 - (t - 0.6) / 0.4);
  ctx.font = sans(30 * u, "700");
  ctx.textAlign = "center";
  ctx.fillStyle = rgb(COLORS[jutsu] ?? [255, 255, 255], Math.min(1, alpha));
  ctx.fillText(jutsu, w / 2, h / 4);
  ctx.textAlign = "left";
}

function drawHud(ctx, sign, conf, combo, w, h, u) {
  if (sign && sign !== "Idle") {
    ctx.font = sans(24 * u, "600");
    ctx.fillStyle = conf > 0.7 ? "rgb(100,255,100)" : "rgb(255,200,100)";
    ctx.fillText(`${sign}  ${Math.round(conf * 100)}%`, 20 * u, 45 * u);
  }
  if (combo.length) {
    ctx.font = sans(17 * u, "600");
    ctx.textAlign = "center";
    ctx.fillStyle = "#b4b4b4";
    ctx.fillText(combo.join(" → "), w / 2, h - 25 * u);
    ctx.textAlign = "left";
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// Effet
// ══════════════════════════════════════════════════════════════════════════════

export default {
  id: "ninjutsu",
  label: "Ninjutsu",
  help: "Enchaîne les signes avec les deux mains :<br>" +
        JUTSU_COMBOS.map(([seq, name]) => `<b>${name}</b> : ${seq.join(" → ")}`).join("<br>"),
  needs: { hands: true },

  create({ setStatus }) {
    let knn = null;
    setStatus("chargement des signes…");
    const ready = loadKnn()
      .then((k) => { knn = k; setStatus(""); })
      .catch((e) => { console.error(e); setStatus("modèle des signes indisponible"); });

    const st = {
      combo: [], lastSign: null, lastSignTime: 0,
      active: null, jutsuStart: 0, jutsuTick: 0,
      particles: [], votes: [], slots: [null, null],
    };

    function updateCombo(sign, now) {
      if (!sign || sign === "Idle" || sign === st.lastSign) return;
      if (now - st.lastSignTime > COMBO_TIMEOUT) st.combo = [];
      if (now - st.lastSignTime < GESTURE_DEBOUNCE) return;

      st.combo.push(sign);
      st.lastSign = sign;
      st.lastSignTime = now;

      // les combos les plus longs d'abord
      const sorted = [...JUTSU_COMBOS].sort((a, b) => b[0].length - a[0].length);
      for (const [seq, name] of sorted) {
        const tail = st.combo.slice(-seq.length);
        if (tail.length === seq.length && tail.every((s, i) => s === seq[i])) {
          st.active = name;
          st.jutsuStart = now;
          st.jutsuTick = 0;
          st.combo = [];
          st.votes = [];
          break;
        }
      }
    }

    return {
      ready,
      frame({ ctx, source, W, H, u, now, hands }) {
        ctx.drawImage(source, 0, 0);
        const t = now / 1000;

        // squelettes
        ctx.strokeStyle = "rgb(255,80,80)";
        ctx.lineWidth = u;
        for (const { px } of hands || []) {
          ctx.beginPath();
          for (const [s, e] of HAND_CONNECTIONS) { ctx.moveTo(px[s].x, px[s].y); ctx.lineTo(px[e].x, px[e].y); }
          ctx.stroke();
        }

        // prédiction + vote temporel
        let shown = null, conf = 0;
        if (knn && hands) {
          const pred = predict(knn, buildFeatures(hands, st.slots));
          conf = pred.conf;
          st.votes.push(pred.sign);
          if (st.votes.length > VOTE_WINDOW) st.votes.shift();
          const counts = {};
          for (const v of st.votes) counts[v] = (counts[v] || 0) + 1;
          const best = Object.keys(counts).reduce((a, b) => (counts[b] > counts[a] ? b : a));
          const stable = counts[best] >= VOTE_THRESHOLD ? best : null;
          updateCombo(stable, t);
          shown = stable || pred.sign;
        }

        // effets
        const cx = W / 2, cy = H / 2;
        const elapsed = t - st.jutsuStart;
        if (st.active && elapsed < EFFECT_DURATION) {
          const k = st.jutsuTick;
          switch (st.active) {
            case "Katon : Goukakyuu no Jutsu":
              if (k % 2 === 0) spawnFire(st.particles, cx, cy, u);
              break;
            case "Kage Bunshin no Jutsu":
              drawCloneAuras(ctx, cx, cy, k, u);
              break;
            case "Chidori":
              if (k % 3 === 0) spawnLightning(st.particles, cx, cy, u);
              for (let i = 0; i < 3; i++) drawLightning(ctx, cx, cy, rand(0, W), rand(0, H), u);
              break;
            case "Kuchiyose no Jutsu":
              drawSmokePuff(ctx, cx, cy, k, u);
              break;
            case "Doton : Doryuheki":
              drawEarthWall(ctx, W, H, k, u);
              break;
          }
          drawJutsuName(ctx, st.active, elapsed, W, H, u);
          st.jutsuTick++;
        } else if (st.active) {
          st.active = null;
        }

        st.particles = updateParticles(ctx, st.particles, u);
        drawHud(ctx, shown, conf, st.combo, W, H, u);
      },
    };
  },
};
