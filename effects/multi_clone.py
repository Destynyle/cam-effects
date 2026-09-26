"""
Multi-Clone Jutsu
─────────────────
Au lancement : recule et reste immobile 3 secondes — le fond est capturé.
Ensuite reviens : ta silhouette sera dupliquée en cercle autour de toi.
"""

import cv2
import numpy as np

NUM_CLONES    = 4
RADIUS_FACTOR = 0.28   # rayon du cercle (fraction de la largeur)
CLONE_ALPHA   = 0.75
BG_FRAMES     = 60     # frames pour calibrer le fond (~2-3s)
GLOW_COLOR    = (255, 80, 0)


def capture_background(cap, w, h):
    print("Calibration du fond — reste immobile hors du cadre...")
    frames = []
    for i in range(BG_FRAMES, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame.astype(np.float32))

        # countdown à l'écran
        display = frame.copy()
        cv2.putText(display, f"Recule du cadre... {i // 20 + 1}s",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.namedWindow("cam-effects: multi_clone", cv2.WINDOW_NORMAL)
        cv2.imshow("cam-effects: multi_clone", display)
        cv2.waitKey(1)

    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    print("Fond capturé. Reviens dans le cadre !")
    return bg


def extract_person(frame, bg, threshold=25):
    """Soustraction de fond + nettoyage du masque."""
    diff = cv2.absdiff(frame, bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # nettoyage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    return mask  # 0-255 float-like


def blend_clone(canvas, person_bgr, mask, dx, dy, alpha):
    h, w = canvas.shape[:2]

    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(w, w - dx)
    src_y2 = min(h, h - dy)
    dst_x1 = max(0, dx)
    dst_y1 = max(0, dy)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return

    m = mask[src_y1:src_y2, src_x1:src_x2].astype(np.float32) / 255.0 * alpha
    m3 = np.stack([m] * 3, axis=-1)
    src = person_bgr[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
    dst = canvas[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = (dst * (1 - m3) + src * m3).astype(np.uint8)


def apply_effect(frame, bg, tick):
    h, w = frame.shape[:2]
    mask = extract_person(frame, bg)

    # fond propre (frame originale sans la personne)
    output = bg.copy()

    cx, cy = w // 2, h // 2
    radius = int(w * RADIUS_FACTOR)

    # clones en cercle (semi-transparents)
    for i in range(NUM_CLONES):
        angle = (2 * np.pi / NUM_CLONES) * i
        dx = int(np.cos(angle) * radius)
        dy = int(np.sin(angle) * radius)
        blend_clone(output, frame, mask, dx, dy, CLONE_ALPHA)

    # personne originale par-dessus (100% opaque)
    blend_clone(output, frame, mask, 0, 0, 1.0)

    # anneau d'énergie entre les clones
    pulse = int(6 * np.sin(tick * 0.12))
    for i in range(NUM_CLONES):
        a = (2 * np.pi / NUM_CLONES) * i
        px = cx + int(np.cos(a) * (radius + pulse))
        py = cy + int(np.sin(a) * (radius + pulse))
        cv2.circle(output, (px, py), 10, GLOW_COLOR, -1)
        cv2.circle(output, (px, py), 18, tuple(c // 2 for c in GLOW_COLOR), 1)
        # ligne vers le suivant
        a2 = (2 * np.pi / NUM_CLONES) * ((i + 1) % NUM_CLONES)
        p2x = cx + int(np.cos(a2) * (radius + pulse))
        p2y = cy + int(np.sin(a2) * (radius + pulse))
        cv2.line(output, (px, py), (p2x, p2y), GLOW_COLOR, 1)

    return output


def run(input_path: str, output_path: str):
    print("Mode fichier : place une frame de fond vide au début de ta vidéo.")
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # utilise les 60 premières frames comme fond
    bg_frames = []
    for _ in range(BG_FRAMES):
        ret, f = cap.read()
        if ret:
            bg_frames.append(f.astype(np.float32))
    bg = np.median(np.stack(bg_frames), axis=0).astype(np.uint8)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tick = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(apply_effect(frame, bg, tick))
        tick += 1

    cap.release()
    out.release()


def run_live(cam_index: int = 0, virtual_dev: str = None):
    cap = cv2.VideoCapture(cam_index)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg = capture_background(cap, w, h)

    virtual_out = None
    if virtual_dev:
        try:
            import pyfakewebcam
            virtual_out = pyfakewebcam.FakeWebcam(virtual_dev, w, h)
        except ImportError:
            print("pyfakewebcam non installé. Affichage fenêtre uniquement.")

    cv2.namedWindow("cam-effects: multi_clone", cv2.WINDOW_NORMAL)
    tick = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = apply_effect(frame, bg, tick)
        tick += 1

        if virtual_out:
            virtual_out.schedule_frame(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            cv2.imshow("cam-effects: multi_clone", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
