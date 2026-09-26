"""
export_knn.py — Exporte le dataset KNN pour la version web (ninjutsu)
──────────────────────────────────────────────────────────────────────
Sous-échantillonne le dataset (N exemples par signe) et quantifie en int8
→ web/models/signs_knn.bin + signs_knn.json (~1 Mo au lieu de 12 Mo).

Usage :
  python export_knn.py                 # 600 exemples par signe
  python export_knn.py --per 1000
  python export_knn.py --extra mon_dataset.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from train import REPO_CSV, load_csv, FEATURE_COLS

OUT_DIR = Path("web/models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   type=Path, default=REPO_CSV)
    parser.add_argument("--extra", type=Path, help="CSV additionnel à fusionner")
    parser.add_argument("--per",   type=int, default=600, help="Exemples gardés par signe")
    parser.add_argument("--k",     type=int, default=5)
    args = parser.parse_args()

    df = load_csv(args.csv)
    if args.extra and args.extra.exists():
        df = pd.concat([df, load_csv(args.extra)], ignore_index=True)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # ── sous-échantillonnage équilibré ────────────────────────────────────────
    labels = sorted(set(y))
    rng    = np.random.default_rng(0)
    idx    = np.concatenate([
        rng.permutation(np.where(y_train == lab)[0])[:args.per] for lab in labels
    ])
    Xs, ys = X_train[idx], y_train[idx]

    # ── quantification int8 ───────────────────────────────────────────────────
    scale = 127 / float(np.abs(Xs).max())
    Xq    = np.round(Xs * scale).astype(np.int8)
    yq    = np.array([labels.index(v) for v in ys], dtype=np.uint8)

    knn = KNeighborsClassifier(n_neighbors=args.k).fit(Xq.astype(np.float32) / scale, ys)
    acc = (knn.predict(X_test) == y_test).mean()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "signs_knn.bin").write_bytes(Xq.tobytes() + yq.tobytes())
    (OUT_DIR / "signs_knn.json").write_text(json.dumps({
        "labels": labels,
        "scale":  scale,
        "dim":    Xq.shape[1],
        "count":  len(Xq),
        "k":      args.k,
    }, indent=2))

    print(f"[+] {len(Xq)} exemples × {Xq.shape[1]} dims — précision test {acc:.2%}")
    print(f"[+] Exporté → {OUT_DIR}/signs_knn.bin ({Xq.nbytes + yq.nbytes} octets)")


if __name__ == "__main__":
    main()
