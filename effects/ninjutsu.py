"""
Ninjutsu — Vraies séquences Naruto avec reconnaissance KNN
───────────────────────────────────────────────────────────
Enchaîne les vrais signes de mains pour déclencher un jutsu :

  Katon Goukakyuu  : Snake → Ram → Monkey → Boar → Horse → Tiger
  Shadow Clone     : Ram → Boar → Ox → Dog → Snake
  Chidori          : Ox → Hare → Monkey
  Invocation       : Boar → Dog → Bird → Monkey → Ram
  Doton Mur        : Dog → Boar → Ram

Appuie sur Q pour quitter.
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import math
import time
import random
from pathlib import Path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH  = Path(__file__).parent.parent / "models" / "hand_landmarker.task"
KNN_PATH    = Path(__file__).parent.parent / "models" / "signs_knn.pkl"
LABELS_PATH = Path(__file__).parent.parent / "models" / "signs_labels.pkl"

# ── Combos (vraies séquences Naruto) ─────────────────────────────────────────
JUTSU_COMBOS = {
    ("Snake", "Ram", "Monkey", "Boar", "Horse", "Tiger"): "Katon : Goukakyuu no Jutsu",
    ("Ram", "Boar", "Ox", "Dog", "Snake"):                "Kage Bunshin no Jutsu",
    ("Ox", "Hare", "Monkey"):                             "Chidori",
    ("Boar", "Dog", "Bird", "Monkey", "Ram"):             "Kuchiyose no Jutsu",
    ("Dog", "Boar", "Ram"):                               "Doton : Doryuheki",
}

# Couleurs (BGR)
COLORS = {
    "Katon : Goukakyuu no Jutsu": (0,   60,  255),   # rouge feu
    "Kage Bunshin no Jutsu":      (0,  255,  255),   # jaune/cyan
    "Chidori":                    (255, 220,   0),   # bleu électrique
    "Kuchiyose no Jutsu":         (0,  180,  120),   # vert fumée
    "Doton : Doryuheki":          (30, 100,  160),   # marron terre
}

COMBO_TIMEOUT   = 6.0
GESTURE_DEBOUNCE = 0.5
EFFECT_DURATION  = 3.0
VOTE_WINDOW      = 7     # frames pour le vote temporel
VOTE_THRESHOLD   = 4     # confirmations requises


# ══════════════════════════════════════════════════════════════════════════════
# Chargement modèle KNN
# ══════════════════════════════════════════════════════════════════════════════

def load_knn():
    if not KNN_PATH.exists():
        raise FileNotFoundError(f"Modèle KNN introuvable : {KNN_PATH}\nLance d'abord : python train.py")
    with open(KNN_PATH, "rb") as f:
        knn = pickle.load(f)
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)
    print(f"[+] Modèle KNN chargé — {len(labels)} signes : {labels}")
    return knn, labels


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation des landmarks (même logique que le repo)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_hand(landmarks):
    wrist  = landmarks[0]
    middle = landmarks[9]
    dist = math.sqrt((wrist.x - middle.x)**2 + (wrist.y - middle.y)**2 + (wrist.z - middle.z)**2)
    if dist < 0.0001:
        dist = 1.0
    coords = []
    for lm in landmarks:
        coords.extend([
            (lm.x - wrist.x) / dist,
            (lm.y - wrist.y) / dist,
            (lm.z - wrist.z) / dist,
        ])
    return coords


def palm_center(landmarks):
    idxs = [0, 5, 9, 13, 17]
    return (
        sum(landmarks[i].x for i in idxs) / 5,
        sum(landmarks[i].y for i in idxs) / 5,
    )


def build_features(hand_landmarks, slot_states):
    """Construit le vecteur 126D avec gestion des mains manquantes."""
    detections = []
    for lm in (hand_landmarks or [])[:2]:
        detections.append({"coords": normalize_hand(lm), "center": palm_center(lm)})

    # attribution par position horizontale (poignet gauche = slot 0)
    if len(detections) == 2:
        if detections[0]["center"][0] > detections[1]["center"][0]:
            detections[0], detections[1] = detections[1], detections[0]

    features = []
    for slot in range(2):
        if slot < len(detections):
            slot_states[slot] = detections[slot]["coords"]
            features.extend(detections[slot]["coords"])
        elif slot_states[slot] is not None:
            features.extend(slot_states[slot])   # imputation dernière position
        else:
            features.extend([0.0] * 63)

    return features


# ══════════════════════════════════════════════════════════════════════════════
# Vote temporel
# ══════════════════════════════════════════════════════════════════════════════

def voted_prediction(knn, features, vote_buffer):
    proba   = knn.predict_proba([features])[0]
    classes = knn.classes_
    pred    = classes[np.argmax(proba)]
    conf    = float(np.max(proba))

    vote_buffer.append(pred)
    if len(vote_buffer) > VOTE_WINDOW:
        vote_buffer.pop(0)

    counts  = {}
    for v in vote_buffer:
        counts[v] = counts.get(v, 0) + 1
    best    = max(counts, key=counts.get)
    stable  = counts[best] >= VOTE_THRESHOLD

    return best if stable else None, conf, pred


# ══════════════════════════════════════════════════════════════════════════════
# Effets visuels
# ══════════════════════════════════════════════════════════════════════════════

class Particle:
    def __init__(self, x, y, vx, vy, color, life, size=4):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = vx, vy
        self.color = color
        self.life  = life
        self.max_life = life
        self.size  = size

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.25
        self.life -= 1

    def draw(self, frame):
        if self.life <= 0:
            return
        alpha = self.life / self.max_life
        c = tuple(int(v * alpha) for v in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), max(1, int(self.size * alpha)), c, -1)


def spawn_fire(particles, cx, cy):
    for _ in range(15):
        a = random.uniform(0, 2 * np.pi)
        s = random.uniform(5, 18)
        particles.append(Particle(cx, cy, np.cos(a)*s, np.sin(a)*s - 4,
            (random.randint(0,60), random.randint(80,200), 255),
            life=random.randint(25, 55), size=random.randint(5,14)))


def spawn_lightning(particles, cx, cy):
    for _ in range(8):
        a = random.uniform(0, 2 * np.pi)
        s = random.uniform(8, 20)
        particles.append(Particle(cx, cy, np.cos(a)*s, np.sin(a)*s,
            (255, random.randint(180,255), random.randint(0,60)),
            life=random.randint(8, 18), size=random.randint(2,5)))


def draw_lightning(frame, x1, y1, x2, y2, color, segs=8):
    pts = [(x1, y1)]
    for i in range(1, segs):
        t = i / segs
        pts.append((int(x1 + (x2-x1)*t + random.randint(-25,25)),
                    int(y1 + (y2-y1)*t + random.randint(-25,25))))
    pts.append((x2, y2))
    for i in range(len(pts)-1):
        cv2.line(frame, pts[i], pts[i+1], color, 2)
        cv2.line(frame, pts[i], pts[i+1], (255,255,255), 1)


def draw_clone_auras(frame, cx, cy, tick):
    for dx in [-130, 130]:
        r = 70 + int(8 * np.sin(tick * 0.15))
        cv2.circle(frame, (cx+dx, cy), r,     (0, 255, 255), 2)
        cv2.circle(frame, (cx+dx, cy), r-15,  (0, 200, 180), 1)
    cv2.line(frame, (cx-130, cy), (cx+130, cy), (0, 200, 255), 1)


def draw_smoke_puff(frame, cx, cy, tick):
    for i in range(5):
        r  = 30 + i * 18 + int(tick * 1.5)
        alpha = max(0, 1.0 - r / 200)
        overlay = frame.copy()
        cv2.circle(overlay, (cx + random.randint(-20,20), cy - i*15), r,
                   (60, 120, 80), -1)
        cv2.addWeighted(overlay, alpha * 0.3, frame, 1 - alpha * 0.3, 0, frame)


def draw_earth_wall(frame, w, h, tick):
    progress = min(1.0, tick / 30)
    wall_h   = int(h * 0.6 * progress)
    overlay  = frame.copy()
    cv2.rectangle(overlay, (w//2 - 80, h - wall_h), (w//2 + 80, h), (30, 80, 140), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (w//2-80, h-wall_h), (w//2+80, h), (50, 120, 180), 2)


def draw_jutsu_name(frame, jutsu, elapsed, w, h):
    color = COLORS.get(jutsu, (255, 255, 255))
    t     = elapsed / EFFECT_DURATION
    alpha = min(1.0, t * 4) * max(0.0, 1.0 - (t - 0.6) / 0.4)
    size  = cv2.getTextSize(jutsu, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    x     = (w - size[0]) // 2
    overlay = frame.copy()
    cv2.putText(overlay, jutsu, (x, h//4), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)


def draw_combo_hud(frame, combo, w, h):
    if not combo:
        return
    text = " → ".join(combo)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    x    = (w - size[0]) // 2
    cv2.putText(frame, text, (x, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)


def draw_sign_name(frame, sign, conf):
    if sign and sign != "Idle":
        color = (100, 255, 100) if conf > 0.7 else (100, 200, 255)
        cv2.putText(frame, f"{sign}  {conf:.0%}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Boucle principale
# ══════════════════════════════════════════════════════════════════════════════

def make_state():
    return {
        "combo":             [],
        "last_sign":         None,
        "last_sign_time":    0,
        "active_jutsu":      None,
        "jutsu_start":       0,
        "jutsu_tick":        0,
        "particles":         [],
        "vote_buffer":       [],
        "slot_states":       [None, None],
    }


def update_combo(state, sign, now):
    if sign is None or sign == "Idle":
        return

    if sign == state["last_sign"]:
        return

    if now - state["last_sign_time"] > COMBO_TIMEOUT:
        state["combo"] = []

    if now - state["last_sign_time"] < GESTURE_DEBOUNCE:
        return

    state["combo"].append(sign)
    state["last_sign"] = sign
    state["last_sign_time"] = now

    # check tous les combos
    combo = tuple(state["combo"])
    for length in sorted({len(k) for k in JUTSU_COMBOS}, reverse=True):
        if combo[-length:] in JUTSU_COMBOS:
            jutsu = JUTSU_COMBOS[combo[-length:]]
            state["active_jutsu"] = jutsu
            state["jutsu_start"]  = now
            state["jutsu_tick"]   = 0
            state["combo"] = []
            state["vote_buffer"] = []
            print(f"[!] Jutsu déclenché : {jutsu}")
            break


def apply_effect(frame, detector, knn, state, w, h):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results  = detector.detect(mp_image)
    now      = time.time()

    # ── landmarks ─────────────────────────────────────────────────────────────
    if results.hand_landmarks:
        CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
                (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                (13,17),(0,17),(17,18),(18,19),(19,20)]
        for lm in results.hand_landmarks:
            pts = [(int(p.x*w), int(p.y*h)) for p in lm]
            for s, e in CONN:
                cv2.line(frame, pts[s], pts[e], (80, 80, 255), 1)

    # ── prédiction ────────────────────────────────────────────────────────────
    features = build_features(results.hand_landmarks, state["slot_states"])
    stable_sign, conf, raw_sign = voted_prediction(knn, features, state["vote_buffer"])

    update_combo(state, stable_sign, now)

    # ── effets ────────────────────────────────────────────────────────────────
    jutsu   = state["active_jutsu"]
    elapsed = now - state["jutsu_start"]
    cx, cy  = w // 2, h // 2

    if jutsu and elapsed < EFFECT_DURATION:
        t = state["jutsu_tick"]

        if jutsu == "Katon : Goukakyuu no Jutsu":
            if t % 2 == 0:
                spawn_fire(state["particles"], cx, cy)

        elif jutsu == "Kage Bunshin no Jutsu":
            draw_clone_auras(frame, cx, cy, t)

        elif jutsu == "Chidori":
            if t % 3 == 0:
                spawn_lightning(state["particles"], cx, cy)
            for _ in range(3):
                draw_lightning(frame, cx, cy,
                               random.randint(0,w), random.randint(0,h),
                               (255, 220, 0))

        elif jutsu == "Kuchiyose no Jutsu":
            draw_smoke_puff(frame, cx, cy, t)

        elif jutsu == "Doton : Doryuheki":
            draw_earth_wall(frame, w, h, t)

        draw_jutsu_name(frame, jutsu, elapsed, w, h)
        state["jutsu_tick"] += 1

    elif jutsu and elapsed >= EFFECT_DURATION:
        state["active_jutsu"] = None

    # particules
    for p in state["particles"]:
        p.update()
        p.draw(frame)
    state["particles"] = [p for p in state["particles"] if p.life > 0]

    # HUD
    draw_sign_name(frame, stable_sign or raw_sign, conf)
    draw_combo_hud(frame, state["combo"], w, h)

    return frame


def make_detector():
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def run(input_path: str, output_path: str):
    knn, _ = load_knn()
    cap    = cv2.VideoCapture(input_path)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    out    = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    state  = make_state()

    with make_detector() as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(apply_effect(frame, detector, knn, state, w, h))

    cap.release()
    out.release()


def run_live(cam_index: int = 0, virtual_dev: str = None):
    knn, _ = load_knn()
    cap    = cv2.VideoCapture(cam_index)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    virtual_out = None
    if virtual_dev:
        try:
            import pyfakewebcam
            virtual_out = pyfakewebcam.FakeWebcam(virtual_dev, w, h)
        except ImportError:
            print("pyfakewebcam non installé. Affichage fenêtre uniquement.")

    cv2.namedWindow("Ninjutsu", cv2.WINDOW_NORMAL)
    state = make_state()
    print(__doc__)

    with make_detector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            result = apply_effect(frame, detector, knn, state, w, h)

            if virtual_out:
                virtual_out.schedule_frame(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                cv2.imshow("Ninjutsu", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
