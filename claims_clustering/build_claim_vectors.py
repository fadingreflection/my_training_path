"""Собирает claim_vectors.npy из v0_outputs.jsonl."""
import json
import re
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from claim_extractors import extract_regex_features, FEATURE_ORDER
from tfidf_builder import build_tfidf

INPUT_PATH = Path("../flow_harness/v0_outputs.jsonl")
OUT_VECTORS = Path("claim_vectors.npy")
OUT_META = Path("claim_vectors_meta.json")

APPLY_SVD = False
SVD_COMPONENTS = 20


def load_flows(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def flow_uid(row):
    """Уникальный id для flow: input_id + температура."""
    return f"{row['input_id']}_t{row.get('temperature', 'na')}"


def tokenize_len(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def main():
    rows = load_flows(INPUT_PATH)
    print(f"Loaded {len(rows)} flows")

    texts = [r.get("flow_text", "") or "" for r in rows]
    uids = [flow_uid(r) for r in rows]

    # --- Regex features ---
    regex_feats = [extract_regex_features(t) for t in texts]
    regex_matrix = np.array(
        [[f[k] for k in FEATURE_ORDER] for f in regex_feats],
        dtype=np.float32,
    )
    print(f"Regex features: {regex_matrix.shape}")

    # --- TF-IDF ---
    tfidf_matrix, vec, terms = build_tfidf(texts, max_features=60)
    print(f"TF-IDF: {tfidf_matrix.shape}, top terms: {terms[:5]}")

    # --- Нормализация TF-IDF по длине ---
    lengths = np.array([tokenize_len(t) for t in texts], dtype=np.float32)
    lengths[lengths == 0] = 1.0
    # Деление sparse-матрицы на вектор — построчное
    from sklearn.preprocessing import normalize
    tfidf_normed = normalize(tfidf_matrix, norm="l2", axis=1)
    # Дополнительно делим на длину (в токенах)
    tfidf_scaled = tfidf_normed.multiply(1.0 / lengths[:, None]).tocsr()
    print(f"TF-IDF normalized by length (mean len={lengths.mean():.1f})")

    # --- Конкатенация ---
    regex_sparse = csr_matrix(regex_matrix)
    combined = hstack([regex_sparse, tfidf_scaled]).tocsr()
    print(f"Combined: {combined.shape}")

    # --- Z-score по столбцам (для dense части) ---
    # StandardScaler не работает со sparse, но у нас разреженная — делаем вручную
    # Convert to dense for small feature count (56 dims × 600 rows = OK)
    dense = combined.toarray()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dense)
    print(f"Scaled: {scaled.shape}")

    # --- Опционально SVD ---
    if APPLY_SVD and scaled.shape[1] > SVD_COMPONENTS:
        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
        scaled = svd.fit_transform(scaled)
        print(f"After SVD: {scaled.shape}, explained var: "
              f"{svd.explained_variance_ratio_.sum():.3f}")

    # --- Сохранение ---
    np.save(OUT_VECTORS, scaled.astype(np.float32))

    meta = {
        "n_rows": scaled.shape[0],
        "n_features": scaled.shape[1],
        "uids": uids,
        "regex_features": FEATURE_ORDER,
        "tfidf_terms": terms,
        "normalized_by_length": True,
        "zscore_applied": True,
        "svd_components": SVD_COMPONENTS if APPLY_SVD else None,
        "source": str(INPUT_PATH),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved: {OUT_VECTORS}, {OUT_META}")


if __name__ == "__main__":
    main()