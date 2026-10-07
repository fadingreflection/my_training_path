"""HDBSCAN с cosine (через L2-нормализацию + euclidean)."""
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
import hdbscan

X = np.load("claim_vectors.npy")

# L2-нормализация: теперь euclidean ≈ cosine
X_norm = normalize(X, norm="l2", axis=1)

for mcs in [5, 8, 10, 15]:
    for ms in [3, 5]:
        c = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",   # ← euclidean на нормализованных = cosine
            cluster_selection_method="eom",
        )
        labels = c.fit_predict(X_norm)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = int((labels == -1).sum())
        print(f"mcs={mcs:2d} ms={ms:2d}  K={n_clusters:2d}  "
              f"outliers={n_outliers:3d} ({n_outliers/len(labels):.1%})")