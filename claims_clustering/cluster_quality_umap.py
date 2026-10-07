import json
import numpy as np
from pathlib import Path
from sklearn.metrics import silhouette_score
import hdbscan

X_umap = np.load("claim_vectors_umap_vuln.npy")

c = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=3,
                    metric="euclidean", cluster_selection_method="eom")
labels = c.fit_predict(X_umap)

# Silhouette только на точках, попавших в кластеры
mask = labels != -1
sil = silhouette_score(X_umap[mask], labels[mask])
print(f"Silhouette: {sil:.3f}")
print(f"K: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Outliers: {(labels == -1).sum()} ({(labels == -1).mean():.1%})")

Path("cluster_quality_umap.json").write_text(json.dumps({
    "silhouette": float(sil),
    "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
    "n_outliers": int((labels == -1).sum()),
    "outlier_rate": float((labels == -1).mean()),
}, indent=2))
