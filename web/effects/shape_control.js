/*
 * Shape Control — portage de effects/shape_control.py
 * ───────────────────────────────────────────────────
 * Index tendu → curseur · Pinch (pouce+index) → attraper / lâcher
 * Deuxième main pendant un grab → redimensionner (écarter / rapprocher les index)
 */

import { rgb, sans, circle } from "./common.js";

const PINCH_ON       = 0.07;
const PINCH_OFF      = 0.11;
const PINCH_DEBOUNCE = 4;
const SMOOTH_MIN     = 0.08;   // lissage max quand immobile
const SMOOTH_MAX     = 0.40;   // lissage min quand rapide
const SMOOTH_SPEED   = 18.0;   // px/frame (à 640px) au-delà duquel on passe en mode rapide
const PAD            = 14;     // padding hitbox (à 640px)

const GRAB_COLOR      = [150, 255, 0];
const IDLE_COLOR      = [0, 80, 255];
const SHAPE_COLOR     = [255, 200, 200];
const HIGHLIGHT_COLOR = [255, 200, 0];
const ZONE_COLOR      = [255, 100, 100];


// ══════════════════════════════════════════════════════════════════════════════
// Géométrie 3D
// ══════════════════════════════════════════════════════════════════════════════

const SHAPES_3D = {
  cube: {
    verts: [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    edges: [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]],
  },
  pyramid: {
    verts: [[0, -1.5, 0], [-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]],
    edges: [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]],
  },
  octahedron: {
    verts: [[0, -1.5, 0], [0, 1.5, 0], [-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]],
    edges: [[0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [3, 4], [4, 5], [5, 2]],
  },
};

// rotation Ry · Rx · Rz puis projection perspective
function projectShape(verts, s, rx, ry, rz, cx, cy, fov) {
  const [sx, cxr] = [Math.sin(rx), Math.cos(rx)];
  const [sy, cyr] = [Math.sin(ry), Math.cos(ry)];
  const [sz, czr] = [Math.sin(rz), Math.cos(rz)];
  return verts.map(([x, y, z]) => {
    x *= s; y *= s; z *= s;
    [x, y] = [x * czr - y * sz, x * sz + y * czr];     // Rz
    [y, z] = [y * cxr - z * sx, y * sx + z * cxr];     // Rx
    [x, z] = [x * cyr + z * sy, -x * sy + z * cyr];    // Ry
    const zo = Math.max(1, z + fov);
    return { x: cx + (x * fov) / zo, y: cy + (y * fov) / zo };
  });
}

function pointInTriangle(p, [a, b, c]) {
  const sign = (p1, p2, p3) => (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
  const d1 = sign(p, a, b), d2 = sign(p, b, c), d3 = sign(p, c, a);
  return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
}


// ══════════════════════════════════════════════════════════════════════════════
// Formes
// ══════════════════════════════════════════════════════════════════════════════

class Shape {
  constructor(kind, x, y, size) {
    Object.assign(this, { kind, x, y, size, rx: 0.4, ry: 0.3, rz: 0, grabbed: false, proj: [] });
  }

  autoRotate() { this.ry += 0.02; this.rx += 0.008; }

  trianglePts(pad = 0) {
    const r = this.size, k = 1 + pad / Math.max(r, 1);
    const pts = [{ x: 0, y: -r }, { x: -r, y: r }, { x: r, y: r }];
    const cy = r / 3;   // centre de gravité du triangle
    return pts.map((p) => ({ x: this.x + p.x * k, y: this.y + cy + (p.y - cy) * k }));
  }

  draw(ctx, u, highlight, showZone) {
    const color = rgb(highlight ? HIGHLIGHT_COLOR : SHAPE_COLOR);
    const pad = PAD * u, { x, y, size: s } = this;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;

    if (this.kind in SHAPES_3D) {
      const def = SHAPES_3D[this.kind];
      this.proj = projectShape(def.verts, s, this.rx, this.ry, this.rz, x, y, 400 * u);
      ctx.lineWidth = 1.5 * u;
      ctx.beginPath();
      for (const [a, b] of def.edges) { ctx.moveTo(this.proj[a].x, this.proj[a].y); ctx.lineTo(this.proj[b].x, this.proj[b].y); }
      ctx.stroke();
      circle(ctx, x, y, 4 * u); ctx.fill();
      if (showZone) {
        const b = this._bounds();
        ctx.strokeStyle = rgb(ZONE_COLOR); ctx.lineWidth = u;
        ctx.strokeRect(b.x1 - pad, b.y1 - pad, b.x2 - b.x1 + 2 * pad, b.y2 - b.y1 + 2 * pad);
      }
      return;
    }

    ctx.lineWidth = 2 * u;
    if (this.kind === "circle") {
      circle(ctx, x, y, s); ctx.stroke();
      if (showZone) { ctx.strokeStyle = rgb(ZONE_COLOR); ctx.lineWidth = u; circle(ctx, x, y, s + pad); ctx.stroke(); }
    } else if (this.kind === "rectangle") {
      ctx.strokeRect(x - s, y - s / 2, 2 * s, s);
      if (showZone) { ctx.strokeStyle = rgb(ZONE_COLOR); ctx.lineWidth = u; ctx.strokeRect(x - s - pad, y - s / 2 - pad, 2 * s + 2 * pad, s + 2 * pad); }
    } else if (this.kind === "triangle") {
      const tri = (pts) => { ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.closePath(); };
      tri(this.trianglePts()); ctx.stroke();
      if (showZone) { ctx.strokeStyle = rgb(ZONE_COLOR); ctx.lineWidth = u; tri(this.trianglePts(pad)); ctx.stroke(); }
    }
  }

  _bounds() {
    const xs = this.proj.map((p) => p.x), ys = this.proj.map((p) => p.y);
    return { x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys) };
  }

  contains(p, u) {
    const pad = PAD * u, { x, y, size: s } = this;
    if (this.kind in SHAPES_3D) {
      if (!this.proj.length) return Math.hypot(p.x - x, p.y - y) < s + pad;
      const b = this._bounds();
      return p.x >= b.x1 - pad && p.x <= b.x2 + pad && p.y >= b.y1 - pad && p.y <= b.y2 + pad;
    }
    if (this.kind === "circle") return Math.hypot(p.x - x, p.y - y) < s + pad;
    if (this.kind === "rectangle") return Math.abs(p.x - x) <= s + pad && Math.abs(p.y - y) <= s / 2 + pad;
    if (this.kind === "triangle") return pointInTriangle(p, this.trianglePts(pad));
    return false;
  }
}

function makeShapes(w, h, u) {
  return [
    new Shape("cube",       w * 0.3, h * 0.4, 45 * u),
    new Shape("pyramid",    w * 0.6, h * 0.4, 45 * u),
    new Shape("circle",     w * 0.2, h * 0.7, 60 * u),
    new Shape("rectangle",  w * 0.5, h * 0.7, 60 * u),
    new Shape("triangle",   w * 0.8, h * 0.7, 60 * u),
    new Shape("octahedron", w * 0.8, h * 0.3, 45 * u),
  ];
}


// ══════════════════════════════════════════════════════════════════════════════
// Effet
// ══════════════════════════════════════════════════════════════════════════════

export default {
  id: "shape_control",
  label: "Shape Control",
  help: "Index tendu = curseur. Pinch pouce + index = attraper / lâcher une forme.<br>" +
        "Pendant un grab, écarte ou rapproche l'index de l'autre main pour redimensionner.",
  needs: { hands: true },

  create({ W, H, u }) {
    const st = {
      shapes: makeShapes(W, H, u),
      grabbed: null, grabOffset: { x: 0, y: 0 },
      prevDist: null, cursor: null,
      pinching: false, pinchCounter: 0, primary: null,
    };

    function smoothCursor(raw) {
      if (!st.cursor) { st.cursor = { ...raw }; return st.cursor; }
      const speed = Math.hypot(raw.x - st.cursor.x, raw.y - st.cursor.y);
      const a = SMOOTH_MIN + (SMOOTH_MAX - SMOOTH_MIN) * Math.min(1, speed / (SMOOTH_SPEED * u));
      st.cursor.x += (raw.x - st.cursor.x) * a;
      st.cursor.y += (raw.y - st.cursor.y) * a;
      return st.cursor;
    }

    const fx = {
      controls: [{ id: "reset", label: "Reset", title: "Remettre les formes en place" }],
      onControl() { st.shapes = makeShapes(W, H, u); st.grabbed = null; },
      onKey(key) { if (key === "r") fx.onControl("reset"); },

      frame({ ctx, source, u, hands }) {
        ctx.drawImage(source, 0, 0);
        hands ??= [];

        let cursor = null, pinchDist = 1;
        let h0 = null, h1 = null;

        if (hands.length) {
          // main primaire verrouillée par latéralité à la 1ʳᵉ détection
          st.primary ??= hands[0].handed;
          const pi = hands.findIndex((h) => h.handed === st.primary);
          if (pi >= 0) {
            h0 = hands[pi];
            h1 = hands.length > 1 ? hands[1 - pi] : null;
          }

          if (h0) {
            cursor = smoothCursor(h0.px[8]);
            pinchDist = Math.hypot(h0.lm[4].x - h0.lm[8].x, h0.lm[4].y - h0.lm[8].y);
            // hystérésis + debounce
            const raw = pinchDist < (st.pinching ? PINCH_OFF : PINCH_ON);
            if (raw === st.pinching) st.pinchCounter = 0;
            else if (++st.pinchCounter >= PINCH_DEBOUNCE) { st.pinching = raw; st.pinchCounter = 0; }
          }

          // resize avec la deuxième main
          if (h0 && h1) {
            const d = Math.hypot(h0.px[8].x - h1.px[8].x, h0.px[8].y - h1.px[8].y);
            if (st.prevDist !== null && st.grabbed) {
              st.grabbed.size = Math.max(20 * u, Math.min(300 * u, st.grabbed.size + (d - st.prevDist) * 0.5));
            }
            st.prevDist = d;
          } else {
            st.prevDist = null;
          }
        } else {
          st.pinching = false;
          st.primary = null;
        }

        // grab
        if (cursor) {
          if (st.pinching) {
            if (!st.grabbed) {
              const hit = [...st.shapes].reverse().find((s) => s.contains(cursor, u));
              if (hit) {
                st.grabbed = hit;
                hit.grabbed = true;
                st.grabOffset = { x: cursor.x - hit.x, y: cursor.y - hit.y };
              }
            } else {
              st.grabbed.x = cursor.x - st.grabOffset.x;
              st.grabbed.y = cursor.y - st.grabOffset.y;
            }
          } else {
            if (st.grabbed) st.grabbed.grabbed = false;
            st.grabbed = null;
          }
        }

        // dessin
        for (const s of st.shapes) {
          if (!s.grabbed) s.autoRotate();
          const zone = cursor && !st.grabbed && s.contains(cursor, u);
          s.draw(ctx, u, st.grabbed === s, zone);
        }

        // curseur + jauge de pinch
        if (cursor) {
          const color = rgb(st.pinching ? GRAB_COLOR : IDLE_COLOR);
          ctx.strokeStyle = ctx.fillStyle = color;
          ctx.lineWidth = 2 * u;
          circle(ctx, cursor.x, cursor.y, 14 * u); ctx.stroke();
          circle(ctx, cursor.x, cursor.y, 4 * u); ctx.fill();
          const ratio = Math.max(0, Math.min(1, 1 - (pinchDist - PINCH_ON) / (0.2 - PINCH_ON)));
          ctx.fillStyle = "rgb(60,60,60)";
          ctx.fillRect(cursor.x - 30 * u, cursor.y + 20 * u, 60 * u, 8 * u);
          ctx.fillStyle = color;
          ctx.fillRect(cursor.x - 30 * u, cursor.y + 20 * u, 60 * u * ratio, 8 * u);
        }

        // HUD
        ctx.font = sans(20 * u, "600");
        ctx.fillStyle = "rgb(200,200,200)";
        ctx.fillText(`Forme : ${st.grabbed ? st.grabbed.kind : "—"}`, 20 * u, 40 * u);
        if (hands.length === 2) {
          ctx.font = sans(17 * u, "600");
          ctx.fillStyle = "rgb(150,255,100)";
          ctx.fillText("2 mains : resize actif", 20 * u, 72 * u);
        }
      },
    };
    return fx;
  },
};
