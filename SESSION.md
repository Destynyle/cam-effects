# Session — cam-effects

## Structure du projet
```
cam-effects/
├── main.py              # entry point : --effect <nom> [--input] [--cam] [--virtual] [--output] [--mode]
├── train.py             # entraîne le KNN sur le dataset naruto hand signs
├── requirements.txt     # mediapipe, opencv-python, numpy, scikit-learn, pandas, pyfakewebcam
├── venv/                # virtualenv Python (toujours activer avant : source venv/bin/activate)
├── models/
│   ├── hand_landmarker.task   # modèle MediaPipe mains (téléchargé auto au 1er lancement)
│   ├── signs_knn.pkl          # modèle KNN entraîné (12 signes naruto, 99.8% précision)
│   └── signs_labels.pkl
├── effects/
│   ├── hand_chakra.py         # squelette lumineux bleu sur les mains
│   ├── multi_clone.py         # multi-clonage par soustraction de fond (calibration 3s au départ)
│   ├── ninjutsu.py            # vrais combos naruto avec KNN
│   ├── shape_control.py       # manipulation formes 3D/2D à la main (pinch, resize 2 mains)
│   └── blob_art.py            # blob tracking mouvement + couleurs hex + Delaunay ← FAVORI
├── narutohandsigns/           # repo cloné avec dataset CSV (28k samples, 14 classes)
│   └── src/mediapipe_signs_db.csv
└── input/ output/
```

## Lancer un effet
```bash
source venv/bin/activate

# webcam live
python main.py --effect blob_art

# webcam live, démarrer en mode loupe
python main.py --effect blob_art --mode loupe

# fichier vidéo avec mode vitrail
python main.py --effect blob_art --input input/mavideo.mp4 --mode vitrail

# caméra virtuelle (Webcamoid/OBS)
sudo modprobe v4l2loopback
python main.py --effect blob_art --virtual --output /dev/video2
```

## Notes techniques
- MediaPipe 0.10+ : utilise Tasks API (pas mp.solutions, supprimé)
- `mp.Image` et `mp.ImageFormat` sont à la racine de mediapipe, pas dans `mp_vision`
- OpenCV récent : `getTriangleList()` retourne un tuple → `(raw[0] if isinstance(raw, tuple) else raw).astype(int)`
- Arch Linux : venv obligatoire (externally-managed-environment)
- Webcam : /dev/video0, caméra virtuelle : /dev/video2 (v4l2loopback)
- desty est dans le groupe `video`
- KNN entraîné sur 28k samples → 99.8% précision

## blob_art — modes visuels
Switch par geste : lever N doigts et maintenir 1.5s → arc de progression autour de la main

| Doigts | Mode | Description |
|---|---|---|
| 1 | **Default** | cercles + hex + Delaunay filaire |
| 2 | **Loupe** | loupe magnifiante dans chaque blob (zoom ×2.5) |
| 3 | **Vitrail** | triangles Delaunay remplis, couleurs interpolées |
| 4 | **Voronoï** | zones de territoire par blob |

`--mode` en CLI : `default | loupe | vitrail | voronoi`
En mode fichier vidéo, la détection main est désactivée (mode fixe, plus rapide).

## Combos ninjutsu
| Jutsu | Séquence |
|---|---|
| Katon Fireball | Snake → Ram → Monkey → Boar → Horse → Tiger |
| Shadow Clone | Ram → Boar → Ox → Dog → Snake |
| Chidori | Ox → Hare → Monkey |
| Invocation | Boar → Dog → Bird → Monkey → Ram |
| Doton Mur | Dog → Boar → Ram |

## Version web (`web/`) — https://destynyle.github.io/cam-effects/
- Repo : github.com/Destynyle/cam-effects (public, Pages sur `main` /), email commits : destysom01@gmail.com
- Statique, aucun build. `app.js` = noyau, `effects/*.js` = 1 module par effet (interface décrite en tête de app.js)
- Libs CDN : `@mediapipe/tasks-vision` 0.10.14 (HandLandmarker 2 mains + ImageSegmenter selfie), `d3-delaunay`
- blob_art : détection mouvement faite main (pas d'OpenCV.js)
- multi_clone : segmentation MediaPipe (plus de calibration fond)
- ninjutsu : `export_knn.py` → 600 ex/signe, int8 → web/models/signs_knn.bin (1 Mo, 99,6 %)
- shape_control : formes 3D ×size (la version Python les dessinait quasi ponctuelles : verts × size/80)
- Test headless : chromium --use-fake-device-for-media-stream --use-file-for-fake-video-capture=x.y4m + CDP
- Caméra exige HTTPS (ou localhost)

## Prochaines étapes
1. ~~Version web~~ → fait, les 5 effets (voir ci-dessus)
2. ~~Déploiement~~ → GitHub Pages
3. **TouchDesigner** — intégration OSC depuis Python vers le fixe Windows
4. **domain_expansion.py** — effet JJK pas encore codé
5. **Collecte de données perso** — améliorer le KNN ninjutsu avec ses propres mains

## Setup PC fixe (pour plus tard)
- GPU : Sapphire Pulse RX 7900 GRE (RDNA3, ROCm supporté)
- DaVinci Resolve à installer pour color grading + render
- ROCm pour accélération GPU AMD
