"""Сохранить финальные метки после UMAP + HDBSCAN."""
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
import hdbscan

MCS = 15
MS = 3

X_umap = np.load("claim_vectors_umap_vuln.npy")

# Индексы VULNERABLE flow
is_vuln = []
uids_vuln = []
meta = json.loads(Path("claim_vectors_meta.json").read_text())
uids_full = meta["uids"]
i = 0
for line in open("../flow_harness/v0_outputs.jsonl"):
    r = json.loads(line)
    if r.get("verdict") == "VULNERABLE":
        uids_vuln.append(uids_full[i])
    i += 1

c = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS,
                    metric="euclidean", cluster_selection_method="eom")
labels = c.fit_predict(X_umap)

# Сохранить
rows = [{"flow_id": uid, "cluster_id": int(lab)}
        for uid, lab in zip(uids_vuln, labels)]
Path("mechanism_labels_vuln.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
)
print(f"Saved: mechanism_labels_vuln.jsonl ({len(rows)} rows)")
print(f"K={len(set(labels)) - (1 if -1 in labels else 0)}, "
      f"outliers={(labels == -1).sum()}")

# Также сохранить reducer для будущего использования
import pickle
# (если нужно переиспользовать UMAP)
