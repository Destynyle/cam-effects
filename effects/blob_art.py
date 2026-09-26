"""
Blob Art — Motion tracking + live color sampling
─────────────────────────────────────────────────
Détecte les zones de mouvement comme blobs,
échantillonne leur couleur dominante en direct,
affiche les codes hex et relie les blobs par
une triangulation géométrique.

Modes visuels — changer avec les doigts (maintenu 1.5s) :
  1 doigt  → Default  : cercles + hex + Delaunay filaire
  2 doigts → Loupe    : loupe magnifiante dans chaque blob
  3 doigts → Vitrail  : triangles Delaunay remplis (vitrail)
  4 doigts → Voronoï  : zones de territoire par blob

Appuie sur Q pour quitter.
"""

import cv2
import numpy as np
import math
import time
import urllib.request
import subprocess
import shutil
import mediapipe as mp
from collections import deque
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paths modèles ─────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Paramètres blobs ──────────────────────────────────────────────────────────
MIN_BLOB_AREA  = 400
MAX_BLOBS      = 12
TRAIL_LEN      = 18
SAMPLE_RADIUS  = 14
LINE_DIST      = 260
BG_ALPHA       = 0.55
FONT           = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE     = 0.42
FONT_THICKNESS = 1

# ── Modes ─────────────────────────────────────────────────────────────────────
MODES            = ["default", "loupe", "vitrail", "voronoi"]
MODE_LABELS      = ["Default", "Loupe", "Vitrail", "Voronoï"]
GESTURE_HOLD_SEC = 1.5


# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe — hand detection
# ══════════════════════════════════════════════════════════════════════════════

def ensure_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("Téléchargement du modèle hand_landmarker...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modèle téléchargé.")


def make_detector():
    ensure_model()
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def count_fingers(landmarks):
    """Compte les doigts levés (index→auriculaire). Retourne 0-4."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return sum(1 for tip, pip in zip(tips, pips) if landmarks[tip].y < landmarks[pip].y)


# ── Switcher de mode par geste ────────────────────────────────────────────────

class GestureModeSwitcher:
    def __init__(self):
        self.mode_idx    = 0
        self._pending    = -1
        self._hold_start = None

    @property
    def mode(self):
        return MODES[self.mode_idx]

    def update(self, fingers):
        """
        fingers : int 1-4 (doigts levés) ou -1 (pas de main détectée).
        Retourne (mode_idx, progress 0.0-1.0).
        """
        target = fingers - 1  # 1→0, 2→1, 3→2, 4→3
        if not (0 <= target < len(MODES)):
            self._pending    = -1
            self._hold_start = None
            return self.mode_idx, 0.0

        if fingers != self._pending:
            self._pending    = fingers
            self._hold_start = time.time()

        elapsed  = time.time() - self._hold_start
        progress = min(elapsed / GESTURE_HOLD_SEC, 1.0)

        if elapsed >= GESTURE_HOLD_SEC and target != self.mode_idx:
            self.mode_idx    = target
            self._hold_start = time.time()

        return self.mode_idx, progress


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires couleur
# ══════════════════════════════════════════════════════════════════════════════

def sample_color(frame, cx, cy, radius=SAMPLE_RADIUS):
    h, w = frame.shape[:2]
    x1 = max(0, cx - radius);  y1 = max(0, cy - radius)
    x2 = min(w, cx + radius);  y2 = min(h, cy + radius)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return (128, 128, 128)
    mean = cv2.mean(region)[:3]
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def bgr_to_hex(bgr):
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


def luminance(bgr):
    b, g, r = bgr
    return 0.299*r + 0.587*g + 0.114*b


def text_color(bg_bgr):
    return (20, 20, 20) if luminance(bg_bgr) > 140 else (230, 230, 230)


# ══════════════════════════════════════════════════════════════════════════════
# Détection de blobs
# ══════════════════════════════════════════════════════════════════════════════

def detect_blobs(fgmask, frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean  = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean  = cv2.morphologyEx(clean,  cv2.MORPH_OPEN,  kernel, iterations=1)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs   = []

    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:MAX_BLOBS]:
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx    = int(M["m10"] / M["m00"])
        cy    = int(M["m01"] / M["m00"])
        r     = int(math.sqrt(area / math.pi))
        color = sample_color(frame, cx, cy)
        blobs.append({"cx": cx, "cy": cy, "r": r, "area": area, "color": color})

    return blobs


# ══════════════════════════════════════════════════════════════════════════════
# Modes de rendu
# ══════════════════════════════════════════════════════════════════════════════

# ── Mode 1 : Default ──────────────────────────────────────────────────────────

def draw_delaunay_lines(frame, blobs, alpha=0.35):
    if len(blobs) < 3:
        return
    h, w = frame.shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for b in blobs:
        try:
            subdiv.insert((b["cx"], b["cy"]))
        except Exception:
            pass

    raw = subdiv.getTriangleList()
    if isinstance(raw, tuple):
        if not raw:
            return
        raw = raw[0]
    triangles = raw.astype(int)
    if len(triangles) == 0:
        return
    overlay   = frame.copy()
    for t in triangles:
        p1, p2, p3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
        if (0 <= p1[0] < w and 0 <= p1[1] < h and
            0 <= p2[0] < w and 0 <= p2[1] < h and
            0 <= p3[0] < w and 0 <= p3[1] < h):
            cv2.line(overlay, p1, p2, (255, 255, 255), 1)
            cv2.line(overlay, p2, p3, (255, 255, 255), 1)
            cv2.line(overlay, p3, p1, (255, 255, 255), 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_connections(frame, blobs):
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            a, b = blobs[i], blobs[j]
            d = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
            if d < LINE_DIST:
                alpha_line = int(180 * (1 - d / LINE_DIST))
                ca, cb = a["color"], b["color"]
                cm = tuple(int((ca[k]+cb[k])//2) for k in range(3))
                overlay = frame.copy()
                cv2.line(overlay, (a["cx"], a["cy"]), (b["cx"], b["cy"]), cm, 1)
                cv2.addWeighted(overlay, alpha_line/255, frame, 1-alpha_line/255, 0, frame)


def draw_blob_default(frame, blob, trails):
    cx, cy, r = blob["cx"], blob["cy"], blob["r"]
    color     = blob["color"]
    hex_code  = bgr_to_hex(color)

    # traînée
    key = (cx // 8, cy // 8)
    if key not in trails:
        trails[key] = deque(maxlen=TRAIL_LEN)
    trails[key].append((cx, cy, color))
    trail = trails[key]
    for i in range(1, len(trail)):
        p1, p2 = trail[i-1], trail[i]
        a  = int(60 * i / len(trail))
        ov = frame.copy()
        cv2.line(ov, (p1[0], p1[1]), (p2[0], p2[1]), p2[2], 2)
        cv2.addWeighted(ov, a/255, frame, 1-a/255, 0, frame)

    # halo
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), r + 10, color, 1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    cv2.circle(frame, (cx, cy), r, color, 2)
    cv2.circle(frame, (cx, cy), 4, color, -1)

    # label hex
    tc = text_color(color)
    (tw, th), _ = cv2.getTextSize(hex_code, FONT, FONT_SCALE, FONT_THICKNESS)
    lx, ly = cx + r + 6, cy - 4
    cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 4), color, -1)
    cv2.putText(frame, hex_code, (lx, ly), FONT, FONT_SCALE, tc, FONT_THICKNESS)
    coord = f"{cx},{cy}"
    cv2.putText(frame, coord, (lx, ly + th + 4), FONT, FONT_SCALE * 0.85,
                (180, 180, 180), FONT_THICKNESS)


def render_default(frame, blobs, trails):
    if len(blobs) >= 3:
        draw_delaunay_lines(frame, blobs)
    draw_connections(frame, blobs)
    for blob in blobs:
        draw_blob_default(frame, blob, trails)


# ── Mode 2 : Loupe ────────────────────────────────────────────────────────────

def draw_blob_loupe(frame, original, blob, zoom=2.5):
    cx, cy, r = blob["cx"], blob["cy"], blob["r"]
    h, w = frame.shape[:2]
    color = blob["color"]

    # région source (plus petite = effet zoom)
    src_r = max(int(r / zoom), 8)
    sx1 = max(0, cx - src_r);  sy1 = max(0, cy - src_r)
    sx2 = min(w, cx + src_r);  sy2 = min(h, cy + src_r)
    region = original[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return

    d = r * 2
    if d < 4:
        return
    zoomed = cv2.resize(region, (d, d), interpolation=cv2.INTER_LINEAR)

    # masque circulaire
    mask = np.zeros((d, d), dtype=np.uint8)
    cv2.circle(mask, (r, r), r - 1, 255, -1)

    # coordonnées destination sur le frame
    dx1, dy1 = cx - r, cy - r
    dx2, dy2 = cx + r, cy + r

    # recadrage si hors-limites
    ox1 = max(0, -dx1);  oy1 = max(0, -dy1)
    ox2 = d - max(0, dx2 - w);  oy2 = d - max(0, dy2 - h)
    fdx1 = max(0, dx1);  fdy1 = max(0, dy1)
    fdx2 = min(w, dx2);  fdy2 = min(h, dy2)
    if fdx2 <= fdx1 or fdy2 <= fdy1:
        return

    zoomed_crop = zoomed[oy1:oy2, ox1:ox2]
    mask_crop   = mask[oy1:oy2, ox1:ox2]
    mask_3ch    = cv2.merge([mask_crop, mask_crop, mask_crop])
    frame_reg   = frame[fdy1:fdy2, fdx1:fdx2]
    frame[fdy1:fdy2, fdx1:fdx2] = np.where(mask_3ch > 0, zoomed_crop, frame_reg)

    # bague
    cv2.circle(frame, (cx, cy), r,     color,       2)
    cv2.circle(frame, (cx, cy), r + 3, (220, 220, 220), 1)

    # reflet loupe
    hl = (cx - r//3, cy - r//3)
    cv2.ellipse(frame, hl, (r//5, r//8), -30, 0, 180, (255, 255, 255), 1)


def render_loupe(frame, original, blobs):
    for blob in blobs:
        draw_blob_loupe(frame, original, blob)


# ── Mode 3 : Vitrail ──────────────────────────────────────────────────────────

def render_vitrail(frame, blobs, alpha=0.55):
    if len(blobs) < 3:
        return
    h, w = frame.shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for b in blobs:
        try:
            subdiv.insert((b["cx"], b["cy"]))
        except Exception:
            pass

    raw = subdiv.getTriangleList()
    if isinstance(raw, tuple):
        if not raw:
            return
        raw = raw[0]
    triangles = raw.astype(int)
    if len(triangles) == 0:
        return
    overlay   = frame.copy()

    for t in triangles:
        p1, p2, p3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
        if not (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h and
                0 <= p3[0] < w and 0 <= p3[1] < h):
            continue

        def nearest(px, py):
            return min(blobs, key=lambda b: math.hypot(b["cx"]-px, b["cy"]-py))["color"]

        c1 = nearest(*p1);  c2 = nearest(*p2);  c3 = nearest(*p3)
        color = tuple(int((c1[k]+c2[k]+c3[k])//3) for k in range(3))
        pts_tri = np.array([p1, p2, p3], dtype=np.int32)
        cv2.fillConvexPoly(overlay, pts_tri, color)
        cv2.polylines(overlay, [pts_tri], True, (30, 30, 30), 1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # centres des blobs
    for b in blobs:
        cv2.circle(frame, (b["cx"], b["cy"]), 4, (255, 255, 255), -1)


# ── Mode 4 : Voronoï ──────────────────────────────────────────────────────────

def render_voronoi(frame, blobs, alpha=0.45):
    if len(blobs) < 2:
        return
    h, w = frame.shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for b in blobs:
        try:
            subdiv.insert((b["cx"], b["cy"]))
        except Exception:
            pass

    try:
        facets, centers = subdiv.getVoronoiFacetList([])
    except Exception:
        return

    overlay = frame.copy()
    for i, facet in enumerate(facets):
        pts = np.array(facet, dtype=np.int32)
        cx_f = int(centers[i][0]);  cy_f = int(centers[i][1])
        nearest = min(blobs, key=lambda b: math.hypot(b["cx"]-cx_f, b["cy"]-cy_f))
        color = nearest["color"]
        cv2.fillConvexPoly(overlay, pts, color)
        cv2.polylines(overlay, [pts], True, (200, 200, 200), 1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for b in blobs:
        cv2.circle(frame, (b["cx"], b["cy"]), 5, (255, 255, 255), -1)
        cv2.circle(frame, (b["cx"], b["cy"]), 5, b["color"], 2)


# ══════════════════════════════════════════════════════════════════════════════
# HUD — indicateur de mode
# ══════════════════════════════════════════════════════════════════════════════

def draw_mode_hud(frame, mode_idx, progress, hand_cx=None, hand_cy=None):
    h, w = frame.shape[:2]
    label = f"Mode : {MODE_LABELS[mode_idx]}"
    cv2.putText(frame, label, (10, h - 12), FONT, 0.9, (180, 180, 180), 1)

    # arc de progression autour de la main si geste en cours
    if progress > 0.0 and hand_cx is not None:
        angle = int(360 * progress)
        color_arc = (80, 220, 120)
        cv2.ellipse(frame, (hand_cx, hand_cy), (28, 28), -90, 0, angle, color_arc, 3)
        pending_idx = (mode_idx + 1) % len(MODES) if progress < 1.0 else mode_idx
        # affiche le futur mode
        fingers_shown = None
        # find which finger count matches the pending transition
        # (just show the target mode label near the hand)
        cv2.putText(frame, MODE_LABELS[pending_idx],
                    (hand_cx + 32, hand_cy + 6), FONT, 0.85, color_arc, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Boucle principale
# ══════════════════════════════════════════════════════════════════════════════

def process(frame, subtractor, trails, detector, switcher, w, h):
    original = frame.copy()

    # fond assombri
    dark   = (frame * (1 - BG_ALPHA)).astype(np.uint8)
    fgmask = subtractor.apply(frame)
    blobs  = detect_blobs(fgmask, frame)
    cv2.addWeighted(dark, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)

    # détection main + geste
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    result   = detector.detect(mp_img)
    fingers  = -1
    hand_cx  = hand_cy = None

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        fingers = count_fingers(lm)
        hand_cx = int(lm[9].x * w)
        hand_cy = int(lm[9].y * h)

    mode_idx, progress = switcher.update(fingers)
    mode = MODES[mode_idx]

    # rendu selon le mode
    if mode == "default":
        render_default(frame, blobs, trails)
    elif mode == "loupe":
        render_loupe(frame, original, blobs)
    elif mode == "vitrail":
        render_vitrail(frame, blobs)
    elif mode == "voronoi":
        render_voronoi(frame, blobs)

    # HUD
    draw_mode_hud(frame, mode_idx, progress, hand_cx, hand_cy)
    cv2.putText(frame, f"blobs: {len(blobs)}", (w - 80, h - 12),
                FONT, FONT_SCALE, (120, 120, 120), 1)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# Entrées / Sorties
# ══════════════════════════════════════════════════════════════════════════════

def run(input_path: str, output_path: str, mode: str = None, resize: int = None):
    cap        = cv2.VideoCapture(input_path)
    w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total_frames / fps

    # resize optionnel (ex: 720 → hauteur 720, largeur proportionnelle)
    if resize:
        scale = resize / h
        w, h  = int(w * scale), resize

    print(f"  {total_frames} frames  {duration:.1f}s  {w}x{h}  {fps:.0f}fps")

    out  = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    sub  = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    trails   = {}
    switcher = GestureModeSwitcher()
    if mode and mode in MODES:
        switcher.mode_idx = MODES.index(mode)

    t_start    = time.time()
    frame_idx  = 0
    report_every = max(1, int(fps * 5))  # log toutes les 5s de vidéo

    # pas de détection main en mode fichier — on fixe le mode et on traite en batch
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, (w, h))
        original = frame.copy()
        dark     = (frame * (1 - BG_ALPHA)).astype(np.uint8)
        fgmask   = sub.apply(frame)
        blobs    = detect_blobs(fgmask, frame)
        cv2.addWeighted(dark, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)

        m = switcher.mode
        if m == "default":
            render_default(frame, blobs, trails)
        elif m == "loupe":
            render_loupe(frame, original, blobs)
        elif m == "vitrail":
            render_vitrail(frame, blobs)
        elif m == "voronoi":
            render_voronoi(frame, blobs)

        draw_mode_hud(frame, switcher.mode_idx, 0.0)
        cv2.putText(frame, f"blobs: {len(blobs)}", (w - 80, h - 12),
                    FONT, FONT_SCALE, (120, 120, 120), 1)
        out.write(frame)

        frame_idx += 1
        if frame_idx % report_every == 0 and total_frames > 0:
            elapsed  = time.time() - t_start
            progress = frame_idx / total_frames
            eta      = elapsed / progress - elapsed
            print(f"  [{progress*100:5.1f}%]  {frame_idx}/{total_frames} frames"
                  f"  écoulé {elapsed:.0f}s  ETA {eta:.0f}s")

    cap.release()
    out.release()

    elapsed = time.time() - t_start
    ratio   = elapsed / duration if duration > 0 else 0
    print(f"  Rendu terminé en {elapsed:.1f}s  ({ratio:.1f}x la durée réelle)")

    # réinjecter l'audio original via ffmpeg
    if shutil.which("ffmpeg"):
        tmp = output_path + ".tmp.mp4"
        shutil.move(output_path, tmp)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", tmp,
            "-i", input_path,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", output_path
        ], check=False)
        Path(tmp).unlink(missing_ok=True)
    else:
        print("[!] ffmpeg non trouvé — audio non inclus dans la sortie.")


def run_live(cam_index: int = 0, virtual_dev: str = None, mode: str = None):
    cap  = cv2.VideoCapture(cam_index)
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sub  = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    trails   = {}
    detector = make_detector()
    switcher = GestureModeSwitcher()
    if mode and mode in MODES:
        switcher.mode_idx = MODES.index(mode)

    virtual_out = None
    if virtual_dev:
        try:
            import pyfakewebcam
            virtual_out = pyfakewebcam.FakeWebcam(virtual_dev, w, h)
        except ImportError:
            print("pyfakewebcam non installé. Affichage fenêtre uniquement.")

    cv2.namedWindow("Blob Art", cv2.WINDOW_NORMAL)
    print("Modes : 1 doigt=Default  2=Loupe  3=Vitrail  4=Voronoï  (maintenu 1.5s)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        result = process(frame, sub, trails, detector, switcher, w, h)

        if virtual_out:
            virtual_out.schedule_frame(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            cv2.imshow("Blob Art", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
