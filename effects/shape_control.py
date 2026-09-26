"""
Shape Control — Manipulation de formes 3D/2D à la main
───────────────────────────────────────────────────────
Gestes :
  Index tendu          → curseur
  Pinch (pouce+index)  → attraper / lâcher une forme
  Deux mains pinch     → redimensionner
  Molette souris       → changer de forme active (debug)

Formes : Cube, Pyramide, Sphère, Tore (wireframe 3D)
         Cercle, Rectangle, Triangle (2D)

Appuie sur Q pour quitter, TAB pour changer de forme.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"

PINCH_ON        = 0.07
PINCH_OFF       = 0.11
PINCH_DEBOUNCE  = 4
GRAB_RADIUS     = 0.35   # hitbox serrée (35% du rayon visuel)
SMOOTH_MIN      = 0.08   # lissage max quand immobile
SMOOTH_MAX      = 0.40   # lissage min quand rapide
SMOOTH_SPEED    = 18.0   # px/frame au-delà duquel on passe en mode rapide
GRAB_COLOR      = (0, 255, 150)
IDLE_COLOR      = (255, 80, 0)
SHAPE_COLOR     = (200, 200, 255)
HIGHLIGHT_COLOR = (0, 200, 255)


# ══════════════════════════════════════════════════════════════════════════════
# Géométrie 3D
# ══════════════════════════════════════════════════════════════════════════════

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def project(pts3d, cx, cy, fov=400):
    """Projection perspective simple."""
    out = []
    for x, y, z in pts3d:
        z_off = z + fov
        if z_off < 1:
            z_off = 1
        px = int(cx + x * fov / z_off)
        py = int(cy + y * fov / z_off)
        out.append((px, py))
    return out


SHAPES_3D = {
    "cube": {
        "verts": np.array([
            [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
            [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
        ], dtype=float),
        "edges": [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ],
    },
    "pyramid": {
        "verts": np.array([
            [0, -1.5, 0],
            [-1, 1, -1],[1, 1, -1],[1, 1, 1],[-1, 1, 1],
        ], dtype=float),
        "edges": [
            (0,1),(0,2),(0,3),(0,4),
            (1,2),(2,3),(3,4),(4,1),
        ],
    },
    "octahedron": {
        "verts": np.array([
            [0,-1.5,0],[0,1.5,0],
            [-1,0,-1],[1,0,-1],[1,0,1],[-1,0,1],
        ], dtype=float),
        "edges": [
            (0,2),(0,3),(0,4),(0,5),
            (1,2),(1,3),(1,4),(1,5),
            (2,3),(3,4),(4,5),(5,2),
        ],
    },
}

SHAPES_2D = {
    "circle":    lambda cx, cy, r: ("circle",    cx, cy, r),
    "rectangle": lambda cx, cy, r: ("rectangle", cx, cy, r),
    "triangle":  lambda cx, cy, r: ("triangle",  cx, cy, r),
}

ALL_SHAPES = list(SHAPES_3D.keys()) + list(SHAPES_2D.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Classe Shape
# ══════════════════════════════════════════════════════════════════════════════

PAD = 14   # padding hitbox en pixels


class Shape:
    def __init__(self, kind, x, y, size=80):
        self.kind    = kind
        self.x       = float(x)
        self.y       = float(y)
        self.size    = float(size)
        self.rx      = 0.4
        self.ry      = 0.3
        self.rz      = 0.0
        self.grabbed = False
        self._proj   = []   # derniers points projetés (3D uniquement)

    def auto_rotate(self):
        self.ry += 0.02
        self.rx += 0.008

    def _get_triangle_pts(self):
        cx, cy, r = int(self.x), int(self.y), int(self.size)
        return np.array([[cx, cy-r], [cx-r, cy+r], [cx+r, cy+r]], dtype=np.float32)

    def draw(self, frame, highlight=False, show_zone=False):
        color  = HIGHLIGHT_COLOR if highlight else SHAPE_COLOR
        cx, cy = int(self.x), int(self.y)
        s      = self.size

        if self.kind in SHAPES_3D:
            defn    = SHAPES_3D[self.kind]
            verts   = defn["verts"] * (s / 80)
            R       = rot_y(self.ry) @ rot_x(self.rx) @ rot_z(self.rz)
            rotated = (R @ verts.T).T
            self._proj = project(rotated, cx, cy)
            for a, b in defn["edges"]:
                cv2.line(frame, self._proj[a], self._proj[b], color, 1)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            if show_zone and self._proj:
                xs = [p[0] for p in self._proj]
                ys = [p[1] for p in self._proj]
                tl = (min(xs) - PAD, min(ys) - PAD)
                br = (max(xs) + PAD, max(ys) + PAD)
                cv2.rectangle(frame, tl, br, (100, 100, 255), 1)

        elif self.kind == "circle":
            cv2.circle(frame, (cx, cy), int(s), color, 2)
            if show_zone:
                cv2.circle(frame, (cx, cy), int(s) + PAD, (100, 100, 255), 1)

        elif self.kind == "rectangle":
            r = int(s)
            cv2.rectangle(frame, (cx-r, cy-r//2), (cx+r, cy+r//2), color, 2)
            if show_zone:
                cv2.rectangle(frame, (cx-r-PAD, cy-r//2-PAD),
                              (cx+r+PAD, cy+r//2+PAD), (100, 100, 255), 1)

        elif self.kind == "triangle":
            pts = self._get_triangle_pts().astype(int)
            cv2.polylines(frame, [pts], True, color, 2)
            if show_zone:
                big = self._get_triangle_pts()
                big_center = big.mean(axis=0)
                scaled = big_center + (big - big_center) * (1 + PAD / max(self.size, 1))
                cv2.polylines(frame, [scaled.astype(int)], True, (100, 100, 255), 1)

    def contains(self, px, py):
        cx, cy, s = self.x, self.y, self.size

        if self.kind in SHAPES_3D:
            if not self._proj:
                return math.hypot(px - cx, py - cy) < s + PAD
            xs = [p[0] for p in self._proj]
            ys = [p[1] for p in self._proj]
            return (min(xs) - PAD <= px <= max(xs) + PAD and
                    min(ys) - PAD <= py <= max(ys) + PAD)

        elif self.kind == "circle":
            return math.hypot(px - cx, py - cy) < s + PAD

        elif self.kind == "rectangle":
            r = s
            return (cx - r - PAD <= px <= cx + r + PAD and
                    cy - r//2 - PAD <= py <= cy + r//2 + PAD)

        elif self.kind == "triangle":
            pts   = self._get_triangle_pts()
            center = pts.mean(axis=0)
            scaled = center + (pts - center) * (1 + PAD / max(s, 1))
            result = cv2.pointPolygonTest(scaled.astype(np.float32),
                                          (float(px), float(py)), False)
            return result >= 0

        return False


# ══════════════════════════════════════════════════════════════════════════════
# Détection gestes
# ══════════════════════════════════════════════════════════════════════════════

def tip(lm, idx, w, h):
    p = lm[idx]
    return int(p.x * w), int(p.y * h)

def tip_norm(lm, idx):
    return lm[idx].x, lm[idx].y

def pinch_distance(lm):
    tx, ty = lm[4].x,  lm[4].y
    ix, iy = lm[8].x,  lm[8].y
    return math.hypot(tx - ix, ty - iy)

def index_tip(lm, w, h):
    return tip(lm, 8, w, h)


# ══════════════════════════════════════════════════════════════════════════════
# Boucle principale
# ══════════════════════════════════════════════════════════════════════════════

def make_state(w, h):
    shapes = [
        Shape("cube",      w * 0.3, h * 0.4, 80),
        Shape("pyramid",   w * 0.6, h * 0.4, 80),
        Shape("circle",    w * 0.2, h * 0.7, 60),
        Shape("rectangle", w * 0.5, h * 0.7, 60),
        Shape("triangle",  w * 0.8, h * 0.7, 60),
        Shape("octahedron",w * 0.8, h * 0.3, 80),
    ]
    return {
        "shapes":           shapes,
        "grabbed":          None,
        "grab_offset":      (0, 0),
        "prev_pinch_dist":  None,
        "cursor_smooth":    None,
        "pinching":         False,
        "pinch_counter":    0,
        "primary_hand_id":  None,   # "Left" ou "Right" — verrouillé à la première détection
    }


def smooth_cursor(state, raw):
    if state["cursor_smooth"] is None:
        state["cursor_smooth"] = [float(raw[0]), float(raw[1])]
        return raw
    sx, sy = state["cursor_smooth"]
    rx, ry = float(raw[0]), float(raw[1])
    # lissage adaptatif : plus rapide quand la main bouge vite
    speed  = math.hypot(rx - sx, ry - sy)
    alpha  = SMOOTH_MIN + (SMOOTH_MAX - SMOOTH_MIN) * min(1.0, speed / SMOOTH_SPEED)
    sx     = sx + (rx - sx) * alpha
    sy     = sy + (ry - sy) * alpha
    state["cursor_smooth"] = [sx, sy]
    return int(sx), int(sy)


def apply_effect(frame, detector, state, w, h):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results  = detector.detect(mp_image)
    lms      = results.hand_landmarks

    cursor     = None
    two_hands  = len(lms) == 2
    pinch_dist = 1.0

    # ── tracking par latéralité (Left/Right) ──────────────────────────────────
    def hand_id(i):
        try:
            return results.handedness[i][0].category_name  # "Left" ou "Right"
        except Exception:
            return None

    lm0 = None
    lm1 = None

    if lms:
        ids = [hand_id(i) for i in range(len(lms))]

        # verrouillage : dès la 1ère détection on note l'identité de la main primaire
        if state["primary_hand_id"] is None:
            state["primary_hand_id"] = ids[0]

        pid = state["primary_hand_id"]

        # cherche la main verrouillée parmi les détections
        if pid in ids:
            pi  = ids.index(pid)
            lm0 = lms[pi]
            lm1 = lms[1 - pi] if len(lms) > 1 else None
        else:
            # main primaire absente : on garde le dernier curseur, on ne switch pas
            lm0 = None

        if lm0 is not None:
            raw_cursor = index_tip(lm0, w, h)
            cursor     = smooth_cursor(state, raw_cursor)
            pinch_dist = pinch_distance(lm0)

            # hystérésis + debounce
            threshold = PINCH_OFF if state["pinching"] else PINCH_ON
            raw_pinch = pinch_dist < threshold
            if raw_pinch == state["pinching"]:
                state["pinch_counter"] = 0
            else:
                state["pinch_counter"] += 1
                if state["pinch_counter"] >= PINCH_DEBOUNCE:
                    state["pinching"]      = raw_pinch
                    state["pinch_counter"] = 0

        # resize avec la deuxième main
        if lm0 is not None and lm1 is not None:
            ix0, iy0 = index_tip(lm0, w, h)
            ix1, iy1 = index_tip(lm1, w, h)
            dist_now  = math.hypot(ix0-ix1, iy0-iy1)
            if state["prev_pinch_dist"] is not None and state["grabbed"]:
                delta = dist_now - state["prev_pinch_dist"]
                state["grabbed"].size = max(20, min(300, state["grabbed"].size + delta * 0.5))
            state["prev_pinch_dist"] = dist_now
        else:
            state["prev_pinch_dist"] = None
    else:
        state["pinching"]      = False
        state["primary_hand_id"] = None

    is_pinching = state["pinching"]

    # ── logique grab ─────────────────────────────────────────────────────────
    if cursor:
        cx, cy  = cursor
        grabbed = state["grabbed"]

        if is_pinching:
            if grabbed is None:
                for shape in reversed(state["shapes"]):
                    if shape.contains(cx, cy):
                        state["grabbed"]     = shape
                        state["grab_offset"] = (cx - shape.x, cy - shape.y)
                        shape.grabbed = True
                        break
            else:
                ox, oy = state["grab_offset"]
                grabbed.x = cx - ox
                grabbed.y = cy - oy
        else:
            if grabbed:
                grabbed.grabbed = False
            state["grabbed"] = None

    # ── dessin ───────────────────────────────────────────────────────────────
    for shape in state["shapes"]:
        if not shape.grabbed:
            shape.auto_rotate()
        highlight  = (state["grabbed"] == shape)
        show_zone  = (cursor is not None and shape.contains(*cursor)
                      and state["grabbed"] is None)
        shape.draw(frame, highlight, show_zone)

    # curseur + jauge pinch
    if cursor:
        color = GRAB_COLOR if is_pinching else IDLE_COLOR
        cv2.circle(frame, cursor, 14, color, 2)
        cv2.circle(frame, cursor, 4,  color, -1)
        # jauge visuelle de la distance pinch
        ratio   = max(0.0, min(1.0, 1.0 - (pinch_dist - PINCH_ON) / (0.20 - PINCH_ON)))
        gauge_w = int(60 * ratio)
        cx2, cy2 = cursor
        cv2.rectangle(frame, (cx2-30, cy2+20), (cx2+30, cy2+28), (60,60,60), -1)
        cv2.rectangle(frame, (cx2-30, cy2+20), (cx2-30+gauge_w*2, cy2+28), color, -1)

    # HUD
    grabbed_name = state["grabbed"].kind if state["grabbed"] else "—"
    cv2.putText(frame, f"Forme : {grabbed_name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    if two_hands:
        cv2.putText(frame, "2 mains : resize actif", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 150), 2)

    return frame


def make_detector():
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def run(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    state = make_state(w, h)

    with make_detector() as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(apply_effect(frame, detector, state, w, h))

    cap.release()
    out.release()


def run_live(cam_index: int = 0, virtual_dev: str = None):
    cap = cv2.VideoCapture(cam_index)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    virtual_out = None
    if virtual_dev:
        try:
            import pyfakewebcam
            virtual_out = pyfakewebcam.FakeWebcam(virtual_dev, w, h)
        except ImportError:
            print("pyfakewebcam non installé. Affichage fenêtre uniquement.")

    cv2.namedWindow("Shape Control", cv2.WINDOW_NORMAL)
    state = make_state(w, h)
    print(__doc__)

    with make_detector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            result = apply_effect(frame, detector, state, w, h)

            if virtual_out:
                virtual_out.schedule_frame(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                cv2.imshow("Shape Control", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
