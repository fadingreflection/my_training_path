import json
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_mutual_info_score
import hdbscan

X_umap = np.load("claim_vectors_umap_vuln.npy")

# Только VULNERABLE flow
verdicts, projects, lengths, cwes = [], [], [], []
for line in open("../flow_harness/v0_outputs.jsonl"):
    r = json.loads(line)
    if r.get("verdict") != "VULNERABLE":
        continue
    projects.append(r.get("project", "unknown"))
    lengths.append(len(r.get("flow_text", "")))
    cwes.append(r.get("cwe", "unknown"))

length_bins = np.digitize(lengths, bins=[500, 1000, 2000, 4000])

c = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=3,
                    metric="euclidean", cluster_selection_method="eom")
labels = c.fit_predict(X_umap)

report = {
    "ami_with_project": float(adjusted_mutual_info_score(labels, projects)),
    "ami_with_length": float(adjusted_mutual_info_score(labels, length_bins)),
    "ami_with_cwe": float(adjusted_mutual_info_score(labels, cwes)),
    "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
    "note": "AMI < 0.2 хорошо; > 0.4 кластеры по артефакту"
}
print(json.dumps(report, indent=2))
Path("ami_umap.json").write_text(json.dumps(report, indent=2))
