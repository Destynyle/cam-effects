/*
 * Hand Chakra — portage de effects/hand_chakra.py
 * ───────────────────────────────────────────────
 * Squelette lumineux sur les mains, toile d'énergie entre les doigts,
 * orbes aux extrémités, lien entre les deux poignets.
 */

import { HAND_CONNECTIONS, FINGERTIPS, rgb, circle } from "./common.js";

const PALETTES = [
  { id: "chakra",   name: "Chakra",   color: [0, 80, 255] },
  { id: "kyubi",    name: "Kyûbi",    color: [255, 110, 0] },
  { id: "senjutsu", name: "Senjutsu", color: [60, 255, 120] },
  { id: "susanoo",  name: "Susanoo",  color: [170, 60, 255] },
];

// trace un chemin en lueur additive : halo large → cœur blanc
function glowStroke(ctx, buildPath, color, width, u, intensity = 1) {
  ctx.globalCompositeOperation = "lighter";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const passes = [
    [width + 10 * u, 0.10],
    [width + 5 * u,  0.22],
    [width + 2 * u,  0.45],
  ];
  for (const [w, a] of passes) {
    ctx.lineWidth = w;
    ctx.strokeStyle = rgb(color, a * intensity);
    buildPath(); ctx.stroke();
  }
  ctx.lineWidth = Math.max(1, width * 0.6);
  ctx.strokeStyle = rgb([220, 235, 255], 0.9 * intensity);
  buildPath(); ctx.stroke();
  ctx.globalCompositeOperation = "source-over";
}

function skeletonPath(ctx, pts) {
  return () => {
    ctx.beginPath();
    for (const [s, e] of HAND_CONNECTIONS) {
      ctx.moveTo(pts[s].x, pts[s].y);
      ctx.lineTo(pts[e].x, pts[e].y);
    }
  };
}

function webPath(ctx, pts) {
  const tips = FINGERTIPS.map((i) => pts[i]);
  return () => {
    ctx.beginPath();
    for (let i = 0; i < tips.length; i++) {
      for (let j = i + 1; j < tips.length; j++) {
        ctx.moveTo(tips[i].x, tips[i].y);
        ctx.lineTo(tips[j].x, tips[j].y);
      }
    }
  };
}

function drawOrbs(ctx, pts, color, u, pulse) {
  ctx.globalCompositeOperation = "lighter";
  for (const i of FINGERTIPS) {
    const { x, y } = pts[i];
    const g = ctx.createRadialGradient(x, y, 0, x, y, 22 * u * pulse);
    g.addColorStop(0, rgb(color, 0.55));
    g.addColorStop(1, rgb(color, 0));
    ctx.fillStyle = g;
    circle(ctx, x, y, 22 * u * pulse); ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";
  for (const i of FINGERTIPS) {
    const { x, y } = pts[i];
    ctx.fillStyle = "#fff";
    circle(ctx, x, y, 7 * u); ctx.fill();
    ctx.strokeStyle = rgb(color);
    ctx.lineWidth = 2 * u;
    circle(ctx, x, y, 14 * u); ctx.stroke();
    ctx.strokeStyle = rgb(color, 0.35);
    ctx.lineWidth = u;
    circle(ctx, x, y, 20 * u); ctx.stroke();
  }
}

export default {
  id: "hand_chakra",
  label: "Hand Chakra",
  help: "Montre tes mains : squelette lumineux et toile d'énergie entre les doigts.<br>" +
        "Deux mains = lien de chakra entre les poignets. Bouton couleur (ou touche C).",
  needs: { hands: true },
  variants: PALETTES.map((p) => ({ id: p.id, label: p.name, params: { color: p.id } })),

  create({ params }) {
    let palette = Math.max(0, PALETTES.findIndex((p) => p.id === params.get("color")));
    const fx = {
      controls: [{ id: "color", label: PALETTES[palette].name, title: "Changer la couleur" }],
      onControl() {
        palette = (palette + 1) % PALETTES.length;
        fx.controls[0].label = PALETTES[palette].name;
      },
      onKey(key) { if (key === "c") fx.onControl("color"); },

      frame({ ctx, source, u, now, hands }) {
        ctx.drawImage(source, 0, 0);
        if (!hands?.length) return;

        const color = PALETTES[palette].color;
        const flicker = 0.75 + 0.25 * Math.sin(now / 90) * Math.sin(now / 37);
        const pulse = 1 + 0.12 * Math.sin(now / 160);

        for (const { px } of hands) {
          glowStroke(ctx, webPath(ctx, px), color, u, u, 0.55 * flicker);
          glowStroke(ctx, skeletonPath(ctx, px), color, 1.5 * u, u);
          drawOrbs(ctx, px, color, u, pulse);
        }

        if (hands.length === 2) {
          const a = hands[0].px[0], b = hands[1].px[0];
          glowStroke(ctx, () => { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); },
                     color, 3 * u, u, flicker);
        }
      },
    };
    return fx;
  },
};
