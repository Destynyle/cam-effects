// Utilitaires partagés par les effets

export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
export const FINGERTIPS = [4, 8, 12, 16, 20];

export const rgb  = ([r, g, b], a = 1) => (a >= 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${a})`);
export const mono = (px, weight = "") => `${weight} ${px}px ui-monospace, Menlo, Consolas, monospace`.trim();
export const sans = (px, weight = "") => `${weight} ${px}px system-ui, -apple-system, "Segoe UI", sans-serif`.trim();

export function circle(ctx, x, y, r) {
  ctx.beginPath();
  ctx.arc(x, y, Math.max(0, r), 0, Math.PI * 2);
}

export function line(ctx, a, b) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
}

export const rand = (a, b) => a + Math.random() * (b - a);
export const randInt = (a, b) => Math.floor(rand(a, b + 1));
