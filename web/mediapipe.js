/*
 * MediaPipe — chargé une fois, à la demande, partagé par le live et le rendu vidéo
 */

const MP_VERSION = "0.10.14";
const MP_URL     = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/" +
  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const SEG_MODEL  =
  "https://storage.googleapis.com/mediapipe-models/" +
  "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite";

const SEG_SIZE = 256;   // plus grand côté de l'image envoyée au segmenteur

export const models = { hands: null, segmenter: null };
const loading = {};

let visionPromise = null;
function loadVision() {
  visionPromise ??= (async () => {
    const mod = await import(`${MP_URL}/vision_bundle.mjs`);
    const fileset = await mod.FilesetResolver.forVisionTasks(`${MP_URL}/wasm`);
    return { ...mod, fileset };
  })();
  return visionPromise;
}

// WebGL émulé en logiciel (SwiftShader, llvmpipe…) : le délégué GPU y est ~20× plus lent que le CPU
function softwareGL() {
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    if (!gl) return true;
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const name = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : "";
    return /swiftshader|llvmpipe|softpipe|software/i.test(name);
  } catch {
    return true;
  }
}

async function createWithFallback(Task, fileset, options) {
  const withDelegate = (delegate) => ({ ...options, baseOptions: { ...options.baseOptions, delegate } });
  const order = softwareGL() ? ["CPU", "GPU"] : ["GPU", "CPU"];
  try {
    return await Task.createFromOptions(fileset, withDelegate(order[0]));
  } catch {
    return await Task.createFromOptions(fileset, withDelegate(order[1]));
  }
}

// charge un modèle ("hands" | "segmenter") ; onStatus reçoit les messages d'état
export function ensureModel(kind, onStatus = () => {}) {
  if (models[kind]) return Promise.resolve(models[kind]);
  if (loading[kind]) return loading[kind];
  const label = kind === "hands" ? "modèle main" : "modèle silhouette";
  onStatus(`chargement du ${label}…`);
  loading[kind] = loadVision()
    .then(({ HandLandmarker, ImageSegmenter, fileset }) =>
      kind === "hands"
        ? createWithFallback(HandLandmarker, fileset, {
            baseOptions: { modelAssetPath: HAND_MODEL },
            runningMode: "VIDEO",
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minHandPresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          })
        : createWithFallback(ImageSegmenter, fileset, {
            baseOptions: { modelAssetPath: SEG_MODEL },
            runningMode: "VIDEO",
            outputConfidenceMasks: true,
            outputCategoryMask: false,
          }))
    .then((m) => { models[kind] = m; onStatus(""); return m; })
    .catch((e) => { console.error(e); onStatus(`${label} indisponible`); loading[kind] = null; throw e; });
  return loading[kind];
}

// MediaPipe (mode VIDEO) exige des timestamps strictement croissants,
// y compris quand on enchaîne plusieurs rendus vidéo : horloge commune.
let lastTs = 0;
function timestamp() {
  lastTs = Math.max(lastTs + 1, performance.now());
  return lastTs;
}

export function detectHands(canvas) {
  const res = models.hands.detectForVideo(canvas, timestamp());
  const W = canvas.width, H = canvas.height;
  const handed = res.handedness ?? res.handednesses ?? [];
  return (res.landmarks || []).map((lm, i) => ({
    lm,                                              // normalisé 0-1 (repère affiché)
    px: lm.map((p) => ({ x: p.x * W, y: p.y * H })), // pixels
    handed: handed[i]?.[0]?.categoryName ?? null,
  }));
}

// segmentation de personne → { mask: Float32Array, w, h } (réutilise `prev` si possible)
const segCanvas = document.createElement("canvas");
const segCtx    = segCanvas.getContext("2d");

export function segment(canvas, prev = null) {
  const s = SEG_SIZE / Math.max(canvas.width, canvas.height);
  const w = Math.round(canvas.width * s), h = Math.round(canvas.height * s);
  if (segCanvas.width !== w || segCanvas.height !== h) { segCanvas.width = w; segCanvas.height = h; }
  segCtx.drawImage(canvas, 0, 0, w, h);

  let out = prev;
  models.segmenter.segmentForVideo(segCanvas, timestamp(), (res) => {
    const m = res.confidenceMasks?.[0];
    if (!m) return;
    const data = m.getAsFloat32Array();
    if (!out || out.mask.length !== data.length) out = { mask: new Float32Array(data.length), w: m.width, h: m.height };
    out.mask.set(data);
  });
  return out;
}
