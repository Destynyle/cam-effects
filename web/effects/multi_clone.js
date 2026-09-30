/*
 * Multi-Clone Jutsu — portage de effects/multi_clone.py
 * ─────────────────────────────────────────────────────
 * La version Python calibrait un fond vide (sortir du cadre 3s) puis
 * soustrayait. Ici : segmentation de personne MediaPipe → aucune calibration.
 * La silhouette est dupliquée en cercle autour de toi.
 */

import { rgb, circle } from "./common.js";

const CLONE_COUNTS  = [2, 4, 6];
const RADIUS_FACTOR = 0.37;          // rayon du cercle (fraction du petit côté)
const CLONE_ALPHA   = 0.75;
const GLOW_COLOR    = [0, 80, 255];
const MASK_LO       = 0.35;          // seuil doux du masque de silhouette
const MASK_HI       = 0.65;

export default {
  id: "multi_clone",
  label: "Multi-Clone",
  help: "Mets-toi dans le cadre : ta silhouette est dupliquée en cercle autour de toi.<br>" +
        "Bouton pour changer le nombre de clones (ou touche C).",
  needs: { segmenter: true },
  variants: CLONE_COUNTS.map((n) => ({ id: `${n}clones`, label: `${n} clones`, params: { clones: String(n) } })),

  create({ W, H, params }) {
    const wanted = CLONE_COUNTS.indexOf(+params.get("clones"));
    let countIdx = wanted >= 0 ? wanted : 1;   // 4 clones par défaut
    const person = document.createElement("canvas");
    person.width = W; person.height = H;
    const personCtx = person.getContext("2d");
    const maskCanvas = document.createElement("canvas");
    const maskCtx = maskCanvas.getContext("2d");
    let maskImg = null;
    let tick = 0;

    // masque → canvas alpha + centre de la silhouette
    function buildMask(seg) {
      const { mask, w, h } = seg;
      if (maskCanvas.width !== w || maskCanvas.height !== h) {
        maskCanvas.width = w; maskCanvas.height = h;
        maskImg = maskCtx.createImageData(w, h);
      }
      const px = maskImg.data;
      let sum = 0, sx = 0, sy = 0;
      for (let i = 0; i < mask.length; i++) {
        const a = Math.min(1, Math.max(0, (mask[i] - MASK_LO) / (MASK_HI - MASK_LO)));
        px[i * 4 + 3] = a * 255;
        if (a > 0.5) { sum++; sx += i % w; sy += (i / w) | 0; }
      }
      maskCtx.putImageData(maskImg, 0, 0);
      if (sum < mask.length * 0.01) return null;
      return { x: (sx / sum / w) * W, y: (sy / sum / h) * H };
    }

    const fx = {
      controls: [{ id: "count", label: `${CLONE_COUNTS[countIdx]} clones`, title: "Nombre de clones" }],
      onControl() {
        countIdx = (countIdx + 1) % CLONE_COUNTS.length;
        fx.controls[0].label = `${CLONE_COUNTS[countIdx]} clones`;
      },
      onKey(key) { if (key === "c") fx.onControl("count"); },

      frame({ ctx, source, u, seg }) {
        ctx.drawImage(source, 0, 0);
        if (!seg) return;
        const center = buildMask(seg);
        if (!center) return;
        tick++;

        // silhouette découpée
        personCtx.globalCompositeOperation = "copy";
        personCtx.drawImage(source, 0, 0);
        personCtx.globalCompositeOperation = "destination-in";
        personCtx.drawImage(maskCanvas, 0, 0, W, H);

        const n = CLONE_COUNTS[countIdx];
        const radius = RADIUS_FACTOR * Math.min(W, H);
        const angles = Array.from({ length: n }, (_, i) => (2 * Math.PI / n) * i);

        // clones semi-transparents, puis l'original par-dessus
        ctx.globalAlpha = CLONE_ALPHA;
        for (const a of angles) ctx.drawImage(person, Math.cos(a) * radius, Math.sin(a) * radius);
        ctx.globalAlpha = 1;
        ctx.drawImage(person, 0, 0);

        // anneau d'énergie entre les clones
        const pulse = 6 * u * Math.sin(tick * 0.12);
        const nodes = angles.map((a) => ({
          x: center.x + Math.cos(a) * (radius + pulse),
          y: center.y + Math.sin(a) * (radius + pulse),
        }));
        ctx.strokeStyle = rgb(GLOW_COLOR);
        ctx.lineWidth = u;
        ctx.beginPath();
        nodes.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
        ctx.closePath();
        ctx.stroke();
        for (const p of nodes) {
          ctx.fillStyle = rgb(GLOW_COLOR);
          circle(ctx, p.x, p.y, 10 * u); ctx.fill();
          ctx.strokeStyle = rgb(GLOW_COLOR, 0.5);
          circle(ctx, p.x, p.y, 18 * u); ctx.stroke();
        }
      },
    };
    return fx;
  },
};
