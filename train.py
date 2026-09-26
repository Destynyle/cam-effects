"""
train.py — Entraîne le KNN sur le dataset Naruto Hand Signs
────────────────────────────────────────────────────────────
Usage :
  python train.py                          # dataset repo par défaut
  python train.py --csv mon_dataset.csv   # dataset custom
  python train.py --extra mon_dataset.csv # fusion repo + tes données
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

REPO_CSV   = Path("narutohandsigns/src/mediapipe_signs_db.csv")
MODEL_OUT  = Path("models/signs_knn.pkl")
LABELS_OUT = Path("models/signs_labels.pkl")

FEATURE_COLS = [f"h{hand}_{i}_{ax}"
                for hand in (1, 2)
                for i in range(21)
                for ax in "xyz"]   # 126 colonnes


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # garde uniquement les colonnes utiles
    cols = ["label"] + FEATURE_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {missing[:5]}...")
    return df[cols].dropna()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   type=Path, help="CSV principal (remplace le repo)")
    parser.add_argument("--extra", type=Path, help="CSV additionnel à fusionner")
    parser.add_argument("--k",     type=int, default=5, help="Nombre de voisins KNN")
    args = parser.parse_args()

    # ── chargement ────────────────────────────────────────────────────────────
    csv_path = args.csv or REPO_CSV
    if not csv_path.exists():
        print(f"[!] CSV introuvable : {csv_path}")
        return

    print(f"[+] Chargement de {csv_path}...")
    df = load_csv(csv_path)

    if args.extra and args.extra.exists():
        print(f"[+] Fusion avec {args.extra}...")
        df_extra = load_csv(args.extra)
        df = pd.concat([df, df_extra], ignore_index=True)
        print(f"    → {len(df_extra)} samples ajoutés")

    print(f"[+] Total : {len(df)} samples")
    print(f"[+] Signes :")
    for label, count in df["label"].value_counts().items():
        print(f"    {label:15s} : {count}")

    # ── entraînement ─────────────────────────────────────────────────────────
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"\n[+] Entraînement KNN (k={args.k}) sur {len(X_train)} samples...")
    knn = KNeighborsClassifier(n_neighbors=args.k, metric="euclidean", n_jobs=-1)
    knn.fit(X_train, y_train)

    # ── évaluation ───────────────────────────────────────────────────────────
    y_pred = knn.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"[+] Précision sur le jeu de test : {acc:.1%}\n")
    print(classification_report(y_test, y_pred))

    # ── sauvegarde ───────────────────────────────────────────────────────────
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(knn, f)
    labels = sorted(set(y))
    with open(LABELS_OUT, "wb") as f:
        pickle.dump(labels, f)

    print(f"[+] Modèle sauvegardé → {MODEL_OUT}")
    print(f"[+] Labels sauvegardés → {LABELS_OUT}")


if __name__ == "__main__":
    main()
