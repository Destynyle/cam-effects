# cam-effects

Effets visuels webcam en temps réel — Python, MediaPipe, OpenCV.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage général

```bash
python main.py --effect <nom> [options]
```

| Option | Description |
|---|---|
| `--effect` | Nom de l'effet (obligatoire) |
| `--input` | Fichier vidéo en entrée (sans = webcam live) |
| `--output` | Fichier de sortie (mode fichier) ou `/dev/videoX` (caméra virtuelle) |
| `--cam` | Index webcam (défaut : 0) |
| `--virtual` | Active la sortie vers une caméra virtuelle v4l2loopback |
| `--mode` | Mode visuel (blob_art uniquement) |

---

## Effets

### `blob_art` — Blob Art

Détecte les zones de mouvement comme des blobs, échantillonne leur couleur dominante, relie les blobs par triangulation géométrique.

**4 modes visuels** — switch en levant N doigts devant la caméra (maintenu 1.5s) :

| Doigts | Mode | Description |
|---|---|---|
| ☝️ 1 | `default` | Cercles colorés + codes hex + Delaunay filaire |
| ✌️ 2 | `loupe` | Loupe magnifiante (zoom ×2.5) dans chaque blob |
| 🤟 3 | `vitrail` | Triangles Delaunay remplis — effet vitrail |
| 🖖 4 | `voronoi` | Zones de territoire par blob |

```bash
# webcam live (mode default, switchable par geste)
python main.py --effect blob_art

# démarrer directement en mode vitrail
python main.py --effect blob_art --mode vitrail

# fichier vidéo en mode loupe
python main.py --effect blob_art --input input/video.mp4 --mode loupe
```

---

### `hand_chakra` — Chakra des mains

Squelette lumineux bleu sur les mains, avec halo et énergie entre les fingertips.

```bash
python main.py --effect hand_chakra
```

---

### `multi_clone` — Multi-Clone Jutsu

Duplique ta silhouette en cercle autour de toi par soustraction de fond.

**Au lancement** : recule hors du cadre et reste immobile ~3 secondes le temps de la calibration du fond. Reviens ensuite dans le cadre.

```bash
python main.py --effect multi_clone
```

---

### `ninjutsu` — Ninjutsu

Reconnaît les signes de mains Naruto via un KNN (99.8% précision, 28k samples) et déclenche des effets visuels sur les combos complets.

**Combos :**

| Jutsu | Séquence |
|---|---|
| Katon Goukakyuu | Snake → Ram → Monkey → Boar → Horse → Tiger |
| Kage Bunshin | Ram → Boar → Ox → Dog → Snake |
| Chidori | Ox → Hare → Monkey |
| Kuchiyose | Boar → Dog → Bird → Monkey → Ram |
| Doton Doryuheki | Dog → Boar → Ram |

```bash
# entraîner le modèle (une seule fois)
python train.py

# lancer
python main.py --effect ninjutsu
```

---

### `shape_control` — Shape Control

Manipule des formes 3D/2D à la main via MediaPipe.

**Gestes :**
- **Index tendu** → curseur
- **Pinch** (pouce + index) → attraper / lâcher une forme
- **Deux mains pinch** → redimensionner
- **TAB** → changer de forme active

**Formes disponibles :** Cube, Pyramide, Sphère, Tore (wireframe 3D) — Cercle, Rectangle, Triangle (2D)

```bash
python main.py --effect shape_control
```

---

## Caméra virtuelle (OBS / Webcamoid)

```bash
sudo modprobe v4l2loopback
python main.py --effect blob_art --virtual --output /dev/video2
```

## Notes
- Arch Linux : venv obligatoire (`externally-managed-environment`)
- MediaPipe 0.10+ : Tasks API uniquement (`mp.solutions` supprimé)
- Webcam : `/dev/video0` — caméra virtuelle : `/dev/video2`
