"""HDBSCAN на claim-векторах с перебором параметров."""
import json
import numpy as np
from pathlib import Path
import hdbscan

X = np.load("claim_vectors.npy")
meta = json.loads(Path("claim_vectors_meta.json").read_text())
uids = meta["uids"]

results = []
for mcs in [5, 8, 10, 15, 20]:
    for ms in [3, 5, 10]:
        c = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = c.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = int((labels == -1).sum())
        outlier_rate = n_outliers / len(labels)

        results.append({
            "min_cluster_size": mcs,
            "min_samples": ms,
            "n_clusters": n_clusters,
            "n_outliers": n_outliers,
            "outlier_rate": float(outlier_rate),
            "labels": labels.tolist(),
        })
        print(f"mcs={mcs:2d} ms={ms:2d}  K={n_clusters:2d}  "
              f"outliers={n_outliers:3d} ({outlier_rate:.1%})")

Path("hdbscan_runs.json").write_text(json.dumps(results, ensure_ascii=False))
print("\nSaved: hdbscan_runs.json")
