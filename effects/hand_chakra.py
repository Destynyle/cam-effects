import cv2
import mediapipe as mp
import numpy as np
import urllib.request
from pathlib import Path

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]
FINGERTIPS = [4, 8, 12, 16, 20]
CHAKRA_COLOR = (255, 80, 0)    # bleu chakra (BGR)
GLOW_COLOR   = (255, 255, 255)


def ensure_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Téléchargement du modèle hand_landmarker...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modèle téléchargé.")


def make_detector(mode):
    ensure_model()
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mode,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def get_landmarks_px(hand_landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]


def draw_glow(img, pt1, pt2, color, thickness=2, layers=5):
    for i in range(layers, 0, -1):
        c = tuple(int(c * i / layers) for c in color)
        cv2.line(img, pt1, pt2, c, thickness + i * 2)
    cv2.line(img, pt1, pt2, color, thickness)


def draw_energy_web(overlay, pts, color):
    tips = [pts[i] for i in FINGERTIPS]
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            draw_glow(overlay, tips[i], tips[j], color, thickness=1, layers=4)


def draw_hand_skeleton(overlay, pts, color):
    for s, e in CONNECTIONS:
        draw_glow(overlay, pts[s], pts[e], color, thickness=1, layers=3)


def draw_orbs(overlay, pts, color):
    for idx in FINGERTIPS:
        cv2.circle(overlay, pts[idx], 8, GLOW_COLOR, -1)
        cv2.circle(overlay, pts[idx], 14, color, 2)
        cv2.circle(overlay, pts[idx], 20, tuple(c // 3 for c in color), 1)


def apply_effect(frame, detector, w, h):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    overlay = np.zeros_like(frame)

    if results.hand_landmarks:
        all_pts = [get_landmarks_px(hand, w, h) for hand in results.hand_landmarks]
        for pts in all_pts:
            draw_hand_skeleton(overlay, pts, CHAKRA_COLOR)
            draw_energy_web(overlay, pts, CHAKRA_COLOR)
            draw_orbs(overlay, pts, CHAKRA_COLOR)

        if len(all_pts) == 2:
            draw_glow(overlay, all_pts[0][0], all_pts[1][0], CHAKRA_COLOR, thickness=2, layers=6)

    return cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)


def run(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    with make_detector(mp_vision.RunningMode.IMAGE) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(apply_effect(frame, detector, w, h))

    cap.release()
    out.release()


def run_live(cam_index: int = 0, virtual_dev: str = None):
    cap = cv2.VideoCapture(cam_index)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    virtual_out = None
    if virtual_dev:
        try:
            import pyfakewebcam
            virtual_out = pyfakewebcam.FakeWebcam(virtual_dev, w, h)
        except ImportError:
            print("pyfakewebcam non installé. Affichage fenêtre uniquement.")

    with make_detector(mp_vision.RunningMode.IMAGE) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = apply_effect(frame, detector, w, h)

            if virtual_out:
                virtual_out.schedule_frame(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                cv2.namedWindow("cam-effects: hand_chakra", cv2.WINDOW_NORMAL)
                cv2.imshow("cam-effects: hand_chakra", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
