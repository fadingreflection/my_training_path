"""UMAP + HDBSCAN на VULNERABLE-only."""
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
import umap
import hdbscan

X = np.load("claim_vectors.npy")

is_vuln = []
for line in open("../flow_harness/v0_outputs.jsonl"):
    r = json.loads(line)
    is_vuln.append(r.get("verdict") == "VULNERABLE")
is_vuln = np.array(is_vuln)

X_vuln = normalize(X[is_vuln], norm="l2", axis=1)
print(f"VULNERABLE: {len(X_vuln)}")

# UMAP до 5 dim
print("Running UMAP...")
reducer = umap.UMAP(
    n_components=5,
    n_neighbors=15,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)
X_umap = reducer.fit_transform(X_vuln)
np.save("claim_vectors_umap_vuln.npy", X_umap)
print(f"UMAP shape: {X_umap.shape}")

print("\n=== HDBSCAN ===")
for mcs in [3, 5, 8, 10, 15]:
    for ms in [2, 3, 5]:
        c = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                            metric="euclidean", cluster_selection_method="eom")
        labels = c.fit_predict(X_umap)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = int((labels == -1).sum())
        print(f"mcs={mcs:2d} ms={ms:2d}  K={n_clusters:2d}  "
              f"outliers={n_outliers:3d} ({n_outliers/len(labels):.1%})")
