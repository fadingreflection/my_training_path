#!/usr/bin/env python3
"""Step 2C: Classify BF-class from frame vs explanation vs code.

Reads step2_output/dataset.jsonl.
Encodes:
  - frame_text       with sentence-transformers/all-MiniLM-L6-v2
  - explanation      with sentence-transformers/all-MiniLM-L6-v2
  - code             with microsoft/codebert-base (first 512 tokens)
Runs 5-fold stratified CV with logistic regression (L2) on PCA(30).
Reports macro-F1, accuracy, confusion matrix per representation.

Outputs:
  step2_output/classification_results_3way.json
  step2_output/classification_summary_3way.txt
  step2_output/confusion_code.txt

Usage:
  python3 step2c_classify_with_code.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
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
PROJECT_ROOT = HERE.parent.parent
DATA_PATH = HERE / "step2_output" / "dataset.jsonl"
INPUTS_PATH = PROJECT_ROOT / "flow_harness" / "fixed_200_inputs.jsonl"
OUT_DIR = HERE / "step2_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CODE_MODEL = "microsoft/codebert-base"
CODE_MAX_LEN = 512
N_COMPONENTS = 30
N_FOLDS = 5
SEED = 42


# ---- Load --------------------------------------------------------------------

def load_dataset() -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit(f"missing {DATA_PATH}")
    return [json.loads(l) for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_code_by_input_id() -> dict[str, str]:
    """Map input_id -> code from fixed_200_inputs.jsonl."""
    out: dict[str, str] = {}
    for line in INPUTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["id"]] = r.get("code") or ""
    return out


# ---- Encoders ----------------------------------------------------------------

print(f"loading sentence encoder: {SENT_MODEL}")
_sent = SentenceTransformer(SENT_MODEL)


def encode_sentences(texts: list[str]) -> np.ndarray:
    return _sent.encode(texts, normalize_embeddings=True, show_progress_bar=True)


print(f"loading code encoder: {CODE_MODEL}")
_code_tok = AutoTokenizer.from_pretrained(CODE_MODEL)
_code_model = AutoModel.from_pretrained(CODE_MODEL)
_code_model.eval()

_device = "cuda" if torch.cuda.is_available() else "cpu"
_code_model.to(_device)
print(f"code encoder on device: {_device}")


def encode_code(texts: list[str], batch_size: int = 8) -> np.ndarray:
    """Mean-pooled last hidden state of CodeBERT for each code snippet."""
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = _code_tok(
                batch,
                padding=True,
                truncation=True,
                max_length=CODE_MAX_LEN,
                return_tensors="pt",
            ).to(_device)
            out = _code_model(**enc).last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            embeddings.append(pooled.cpu().numpy())
    arr = np.concatenate(embeddings, axis=0)
    # L2-normalize for parity with sentence embeddings
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-8, None)


# ---- CV ----------------------------------------------------------------------

def run_cv(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED) -> dict:
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


def format_confusion(cm: list[list[int]], classes: list[str]) -> str:
    lines = []
    lines.append("rows = true, cols = predicted")
    lines.append("        " + "".join(f"{c:>8s}" for c in classes))
    for i, c in enumerate(classes):
        row = "".join(f"{v:>8d}" for v in cm[i])
        lines.append(f"{c:>8s}{row}")
    return "\n".join(lines)


# ---- Main --------------------------------------------------------------------

def main() -> None:
    rows = load_dataset()
    code_map = load_code_by_input_id()
    print(f"loaded rows: {len(rows)}")

    # Attach code to each row
    missing_code = 0
    for r in rows:
        c = code_map.get(r["id"])
        if not c:
            missing_code += 1
        r["code"] = c or ""
    if missing_code:
        print(f"warning: {missing_code} rows have no code; they will be dropped")
        rows = [r for r in rows if r.get("code")]

    y = np.array([r["bf_class"] for r in rows])
    n_classes = len(set(y.tolist()))
    print(f"BF classes:   {sorted(set(y.tolist()))}")
    print(f"distribution: {dict(Counter(y.tolist()))}")
    print()

    frame_texts = [r["frame_text"] for r in rows]
    expl_texts = [r["explanation"] for r in rows]
    code_texts = [r["code"] for r in rows]

    print("encoding frames...")
    X_frame = encode_sentences(frame_texts)

    print("encoding explanations...")
    X_expl = encode_sentences(expl_texts)

    print("encoding code (CodeBERT)...")
    X_code = encode_code(code_texts)

    print()
    print(f"frame shape: {X_frame.shape}")
    print(f"expl  shape: {X_expl.shape}")
    print(f"code  shape: {X_code.shape}")
    print()

    print(f"running {N_FOLDS}-fold CV on frames...")
    res_frame = run_cv(X_frame, y)

    print(f"running {N_FOLDS}-fold CV on explanations...")
    res_expl = run_cv(X_expl, y)

    print(f"running {N_FOLDS}-fold CV on code...")
    res_code = run_cv(X_code, y)

    random_baseline = 1.0 / n_classes

    result = {
        "n_rows": len(rows),
        "n_classes": n_classes,
        "classes": sorted(set(y.tolist())),
        "class_distribution": dict(Counter(y.tolist())),
        "random_baseline_macro_f1": random_baseline,
        "frame": res_frame,
        "explanation": res_expl,
        "code": res_code,
        "gaps": {
            "frame_minus_explanation": res_frame["macro_f1"] - res_expl["macro_f1"],
            "frame_minus_code": res_frame["macro_f1"] - res_code["macro_f1"],
            "explanation_minus_code": res_expl["macro_f1"] - res_code["macro_f1"],
        },
    }

    (OUT_DIR / "classification_results_3way.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (OUT_DIR / "confusion_code.txt").write_text(
        format_confusion(res_code["confusion_matrix"], res_code["classes"]),
        encoding="utf-8",
    )

    lines = []
    lines.append("=== Step 2C: classification results (3-way) ===")
    lines.append("")
    lines.append(f"rows:            {len(rows)}")
    lines.append(f"BF classes:      {sorted(set(y.tolist()))}")
    lines.append(f"random baseline: {random_baseline:.4f} (1/{n_classes})")
    lines.append("")
    lines.append(f"{'representation':<16s}{'macro-F1':>12s}{'accuracy':>12s}")
    lines.append("-" * 40)
    lines.append(f"{'frame':<16s}{res_frame['macro_f1']:>12.4f}{res_frame['accuracy']:>12.4f}")
    lines.append(f"{'explanation':<16s}{res_expl['macro_f1']:>12.4f}{res_expl['accuracy']:>12.4f}")
    lines.append(f"{'code':<16s}{res_code['macro_f1']:>12.4f}{res_code['accuracy']:>12.4f}")
    lines.append("")
    lines.append("gaps (macro-F1):")
    for k, v in result["gaps"].items():
        lines.append(f"  {k:35s} {v:+.4f}")
    lines.append("")
    lines.append("per-fold macro-F1 (frame):")
    for f in res_frame["per_fold"]:
        lines.append(f"  fold {f['fold']}: {f['macro_f1']:.4f}")
    lines.append("")
    lines.append("per-fold macro-F1 (explanation):")
    for f in res_expl["per_fold"]:
        lines.append(f"  fold {f['fold']}: {f['macro_f1']:.4f}")
    lines.append("")
    lines.append("per-fold macro-F1 (code):")
    for f in res_code["per_fold"]:
        lines.append(f"  fold {f['fold']}: {f['macro_f1']:.4f}")
    lines.append("")
    lines.append("confusion matrix (frame):")
    lines.append(format_confusion(res_frame["confusion_matrix"], res_frame["classes"]))
    lines.append("")
    lines.append("confusion matrix (explanation):")
    lines.append(format_confusion(res_expl["confusion_matrix"], res_expl["classes"]))
    lines.append("")
    lines.append("confusion matrix (code):")
    lines.append(format_confusion(res_code["confusion_matrix"], res_code["classes"]))

    (OUT_DIR / "classification_summary_3way.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print()
    print("\n".join(lines))
    print()
    print(f"wrote {OUT_DIR}/classification_results_3way.json")
    print(f"wrote {OUT_DIR}/classification_summary_3way.txt")


if __name__ == "__main__":
    main()