#!/usr/bin/env python3
"""Step 2B: Classify BF-class from frame vs explanation.

Reads step2_output/dataset.jsonl.
Encodes frame_text and explanation with the same sentence encoder.
Reduces to 30 dims via PCA.
Runs 5-fold stratified CV with logistic regression (L2).
Reports macro-F1, accuracy, confusion matrix per representation.

Outputs:
  step2_output/classification_results.json
  step2_output/classification_summary.txt
  step2_output/confusion_frame.txt
  step2_output/confusion_explanation.txt

Usage:
  python3 step2b_classify.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "step2_output" / "dataset.jsonl"
OUT_DIR = HERE / "step2_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_COMPONENTS = 30
N_FOLDS = 5
SEED = 42


# ---- Load --------------------------------------------------------------------

def load_dataset() -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit(f"missing {DATA_PATH}")
    return [json.loads(l) for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---- Encoding ----------------------------------------------------------------

print(f"loading encoder: {EMBED_MODEL}")
_encoder = SentenceTransformer(EMBED_MODEL)


def encode(texts: list[str]) -> np.ndarray:
    return _encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)


# ---- CV ----------------------------------------------------------------------

def run_cv(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED) -> dict:
    """Stratified k-fold CV with a PCA + logistic regression pipeline."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    y_pred_all = np.zeros_like(y)
    per_fold = []

    for fold_idx, (tr, te) in enumerate(skf.split(X, y), start=1):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(N_COMPONENTS, X.shape[1], len(tr) - 1))),
            ("lr", LogisticRegression(
                max_iter=2000,
                C=1.0,
                penalty="l2",
                random_state=seed,
            )),
        ])
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        y_pred_all[te] = preds

        per_fold.append({
            "fold": fold_idx,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "accuracy": float(accuracy_score(y[te], preds)),
            "macro_f1": float(f1_score(y[te], preds, average="macro")),
        })

    classes = sorted(set(y.tolist()))
    cm = confusion_matrix(y, y_pred_all, labels=classes)

    return {
        "accuracy": float(accuracy_score(y, y_pred_all)),
        "macro_f1": float(f1_score(y, y_pred_all, average="macro")),
        "per_fold": per_fold,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "report": classification_report(
            y, y_pred_all, labels=classes, output_dict=True, zero_division=0
        ),
    }


# ---- Pretty print ------------------------------------------------------------

def format_confusion(cm: list[list[int]], classes: list[str]) -> str:
    lines = []
    lines.append("rows = true, cols = predicted")
    header = "        " + "".join(f"{c:>8s}" for c in classes)
    lines.append(header)
    for i, c in enumerate(classes):
        row = "".join(f"{v:>8d}" for v in cm[i])
        lines.append(f"{c:>8s}{row}")
    return "\n".join(lines)


# ---- Main --------------------------------------------------------------------

def main() -> None:
    rows = load_dataset()
    print(f"loaded rows: {len(rows)}")

    y = np.array([r["bf_class"] for r in rows])
    n_classes = len(set(y.tolist()))
    print(f"BF classes:  {sorted(set(y.tolist()))}")
    print(f"distribution: {dict(Counter(y.tolist()))}")
    print()

    frame_texts = [r["frame_text"] for r in rows]
    expl_texts = [r["explanation"] for r in rows]

    print("encoding frames...")
    X_frame = encode(frame_texts)
    print("encoding explanations...")
    X_expl = encode(expl_texts)

    print()
    print(f"frame embedding shape: {X_frame.shape}")
    print(f"expl  embedding shape: {X_expl.shape}")
    print()

    print(f"running {N_FOLDS}-fold CV on frames...")
    res_frame = run_cv(X_frame, y)

    print(f"running {N_FOLDS}-fold CV on explanations...")
    res_expl = run_cv(X_expl, y)

    random_baseline = 1.0 / n_classes

    # ---- Write outputs ----

    result = {
        "n_rows": len(rows),
        "n_classes": n_classes,
        "classes": sorted(set(y.tolist())),
        "class_distribution": dict(Counter(y.tolist())),
        "random_baseline_macro_f1": random_baseline,
        "frame": res_frame,
        "explanation": res_expl,
        "gap_frame_minus_explanation": res_frame["macro_f1"] - res_expl["macro_f1"],
    }
    (OUT_DIR / "classification_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (OUT_DIR / "confusion_frame.txt").write_text(
        format_confusion(res_frame["confusion_matrix"], res_frame["classes"]),
        encoding="utf-8",
    )
    (OUT_DIR / "confusion_explanation.txt").write_text(
        format_confusion(res_expl["confusion_matrix"], res_expl["classes"]),
        encoding="utf-8",
    )

    lines = []
    lines.append("=== Step 2B: classification results ===")
    lines.append("")
    lines.append(f"rows:            {len(rows)}")
    lines.append(f"BF classes:      {sorted(set(y.tolist()))}")
    lines.append(f"random baseline: {random_baseline:.4f} (1/{n_classes})")
    lines.append("")
    lines.append(f"{'representation':<16s}{'macro-F1':>12s}{'accuracy':>12s}")
    lines.append("-" * 40)
    lines.append(f"{'frame':<16s}{res_frame['macro_f1']:>12.4f}{res_frame['accuracy']:>12.4f}")
    lines.append(f"{'explanation':<16s}{res_expl['macro_f1']:>12.4f}{res_expl['accuracy']:>12.4f}")
    lines.append("")
    lines.append(f"gap (frame - explanation) macro-F1: {result['gap_frame_minus_explanation']:+.4f}")
    lines.append("")
    lines.append("per-fold macro-F1 (frame):")
    for f in res_frame["per_fold"]:
        lines.append(f"  fold {f['fold']}: {f['macro_f1']:.4f}")
    lines.append("")
    lines.append("per-fold macro-F1 (explanation):")
    for f in res_expl["per_fold"]:
        lines.append(f"  fold {f['fold']}: {f['macro_f1']:.4f}")
    lines.append("")
    lines.append("confusion matrix (frame):")
    lines.append(format_confusion(res_frame["confusion_matrix"], res_frame["classes"]))
    lines.append("")
    lines.append("confusion matrix (explanation):")
    lines.append(format_confusion(res_expl["confusion_matrix"], res_expl["classes"]))

    (OUT_DIR / "classification_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print()
    print("\n".join(lines))
    print()
    print(f"wrote {OUT_DIR}/classification_results.json")
    print(f"wrote {OUT_DIR}/classification_summary.txt")


if __name__ == "__main__":
    main()