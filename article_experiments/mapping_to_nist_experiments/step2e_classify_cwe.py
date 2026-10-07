#!/usr/bin/env python3
"""Step 2E: Direct CWE classification from frame vs explanation vs code.

Reads step2_output/dataset.jsonl (which contains 'cwe' field per row).
Encodes frame_text and explanation with MiniLM, code with CodeBERT.
Runs 5-fold stratified CV with logistic regression on PCA(30).
Reports macro-F1, accuracy, confusion matrix per representation,
plus the CWE class list and its distribution.

Outputs:
  step2_output/cwe_classification_results.json
  step2_output/cwe_classification_summary.txt
  step2_output/cwe_confusion_frame.txt
  step2_output/cwe_confusion_explanation.txt
  step2_output/cwe_confusion_code.txt

Usage:
  python3 step2e_classify_cwe.py
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
MIN_CLASS_SIZE = 3   # drop CWE classes with fewer examples


# ---- Load --------------------------------------------------------------------

def load_dataset() -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit(f"missing {DATA_PATH}")
    return [json.loads(l) for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_code_by_input_id() -> dict[str, str]:
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
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = _code_tok(
                batch, padding=True, truncation=True,
                max_length=CODE_MAX_LEN, return_tensors="pt",
            ).to(_device)
            out = _code_model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            embeddings.append(pooled.cpu().numpy())
    arr = np.concatenate(embeddings, axis=0)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-8, None)


# ---- CV ----------------------------------------------------------------------

def run_cv(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED) -> dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_pred_all = np.zeros(len(y), dtype=object)
    per_fold = []

    for fold_idx, (tr, te) in enumerate(skf.split(X, y), start=1):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(N_COMPONENTS, X.shape[1], len(tr) - 1))),
            ("lr", LogisticRegression(
                max_iter=3000,
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
    lines = ["rows = true, cols = predicted"]
    lines.append("            " + "".join(f"{c[-4:]:>8s}" for c in classes))
    for i, c in enumerate(classes):
        row = "".join(f"{v:>8d}" for v in cm[i])
        lines.append(f"{c[-8:]:>12s}{row}")
    return "\n".join(lines)


# ---- Main --------------------------------------------------------------------

def main() -> None:
    rows = load_dataset()
    code_map = load_code_by_input_id()
    for r in rows:
        r["code"] = code_map.get(r["id"], "")

    # Drop rows without code
    rows = [r for r in rows if r.get("code")]

    # Drop CWE classes with too few examples
    cwe_counts = Counter(r["cwe"] for r in rows)
    keep_cwes = {c for c, n in cwe_counts.items() if n >= MIN_CLASS_SIZE}
    dropped = [r for r in rows if r["cwe"] not in keep_cwes]
    rows = [r for r in rows if r["cwe"] in keep_cwes]

    print(f"kept rows:      {len(rows)}")
    print(f"dropped rows:   {len(dropped)} (CWE with < {MIN_CLASS_SIZE} examples)")
    print(f"CWE classes:    {len(keep_cwes)}")
    for c, n in sorted(cwe_counts.items(), key=lambda x: -x[1]):
        marker = "" if c in keep_cwes else "  (dropped)"
        print(f"  {c:12s} {n:4d}{marker}")
    print()

    y = np.array([r["cwe"] for r in rows])
    n_classes = len(set(y.tolist()))
    print(f"BF/CWE task: {n_classes} classes")
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

    print(f"running {N_FOLDS}-fold CV on frames (CWE)...")
    res_frame = run_cv(X_frame, y)

    print(f"running {N_FOLDS}-fold CV on explanations (CWE)...")
    res_expl = run_cv(X_expl, y)

    print(f"running {N_FOLDS}-fold CV on code (CWE)...")
    res_code = run_cv(X_code, y)

    random_baseline = 1.0 / n_classes

    result = {
        "n_rows": len(rows),
        "n_dropped": len(dropped),
        "n_classes": n_classes,
        "classes": sorted(set(y.tolist())),
        "class_distribution": dict(Counter(y.tolist())),
        "random_baseline_macro_f1": random_baseline,
        "frame": res_frame,
        "explanation": res_expl,
        "code": res_code,
    }

    (OUT_DIR / "cwe_classification_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for name, res in [("frame", res_frame), ("explanation", res_expl), ("code", res_code)]:
        (OUT_DIR / f"cwe_confusion_{name}.txt").write_text(
            format_confusion(res["confusion_matrix"], res["classes"]) + "\n",
            encoding="utf-8",
        )

    lines = []
    lines.append("=== Step 2E: direct CWE classification ===")
    lines.append("")
    lines.append(f"rows:            {len(rows)}")
    lines.append(f"CWE classes:     {n_classes}")
    lines.append(f"random baseline: {random_baseline:.4f} (1/{n_classes})")
    lines.append("")
    lines.append(f"{'representation':<16s}{'macro-F1':>12s}{'accuracy':>12s}")
    lines.append("-" * 40)
    for name, res in [("frame", res_frame), ("explanation", res_expl), ("code", res_code)]:
        lines.append(f"{name:<16s}{res['macro_f1']:>12.4f}{res['accuracy']:>12.4f}")
    lines.append("")
    lines.append("class distribution:")
    for c, n in sorted(Counter(y.tolist()).items(), key=lambda x: -x[1]):
        lines.append(f"  {c:12s} {n}")
    lines.append("")
    lines.append("confusion matrix (frame):")
    lines.append(format_confusion(res_frame["confusion_matrix"], res_frame["classes"]))
    lines.append("")
    lines.append("confusion matrix (explanation):")
    lines.append(format_confusion(res_expl["confusion_matrix"], res_expl["classes"]))
    lines.append("")
    lines.append("confusion matrix (code):")
    lines.append(format_confusion(res_code["confusion_matrix"], res_code["classes"]))

    (OUT_DIR / "cwe_classification_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print()
    print("\n".join(lines))
    print()
    print(f"wrote {OUT_DIR}/cwe_classification_summary.txt")


if __name__ == "__main__":
    main()