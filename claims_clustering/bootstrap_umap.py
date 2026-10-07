"""Bootstrap stability для UMAP+HDBSCAN."""
import json
import numpy as np
from pathlib import Path
import umap
import hdbscan
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

X = np.load("claim_vectors.npy")
is_vuln = []
for line in open("../flow_harness/v0_outputs.jsonl"):
    r = json.loads(line)
    is_vuln.append(r.get("verdict") == "VULNERABLE")
is_vuln = np.array(is_vuln)
X_vuln = normalize(X[is_vuln], norm="l2", axis=1)

MCS, MS = 15, 3

def pipeline(X_in):
    reducer = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                        metric="cosine", random_state=42)
    X_u = reducer.fit_transform(X_in)
    c = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS,
                        metric="euclidean", cluster_selection_method="eom")
    return c.fit_predict(X_u)

base = pipeline(X_vuln)
print(f"Baseline: K={len(set(base)) - (1 if -1 in base else 0)}, "
      f"outliers={(base == -1).sum()}")

rng = np.random.default_rng(42)
aris = []
n = len(X_vuln)
for i in range(50):   # 50 вместо 100, UMAP медленный
    idx = rng.choice(n, size=n, replace=True)
    labels = pipeline(X_vuln[idx])
    mask = (base[idx] != -1) & (labels != -1)
    if mask.sum() < 10:
        continue
    ari = adjusted_rand_score(base[idx][mask], labels[mask])
    aris.append(ari)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/50 done, ARI so far = {np.mean(aris):.3f}")

report = {
    "mcs": MCS, "ms": MS,
    "n_bootstrap": len(aris),
    "ari_mean": float(np.mean(aris)),
    "ari_std": float(np.std(aris)),
    "ari_ci_low": float(np.percentile(aris, 2.5)),
    "ari_ci_high": float(np.percentile(aris, 97.5)),
}
print(json.dumps(report, indent=2))
Path("bootstrap_stability_umap.json").write_text(json.dumps(report, indent=2))
