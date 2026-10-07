"""Self-vs-cross similarity + sibling HitRate@K."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

X = np.load("claim_vectors.npy")
meta = json.loads(Path("claim_vectors_meta.json").read_text())
uids = meta["uids"]

# Группируем индексы по function_id
by_func = defaultdict(list)
for i, uid in enumerate(uids):
    func_id = uid.rsplit("_t", 1)[0]
    by_func[func_id].append(i)

# --- Self-similarity ---
self_sims = []
for func_id, idxs in by_func.items():
    if len(idxs) < 2:
        continue
    sim = cosine_similarity(X[idxs])
    iu = np.triu_indices(len(idxs), k=1)
    self_sims.append(sim[iu].mean())
self_sims = np.array(self_sims)

# --- Cross-similarity ---
rng = np.random.default_rng(42)
cross_sims = []
n = len(uids)
while len(cross_sims) < 5000:
    i, j = rng.choice(n, size=2, replace=False)
    if uids[i].rsplit("_t", 1)[0] == uids[j].rsplit("_t", 1)[0]:
        continue
    cross_sims.append(cosine_similarity(X[i:i+1], X[j:j+1])[0, 0])
cross_sims = np.array(cross_sims)

# --- Sibling HitRate@K ---
sim_full = cosine_similarity(X)
K = 5
hit_count = 0
functions_with_siblings = 0
per_func_hit = {}
for func_id, idxs in by_func.items():
    if len(idxs) < 2:
        continue
    functions_with_siblings += 1
    hit = False
    for i in idxs:
        sims = sim_full[i].copy()
        sims[i] = -1
        top_k = set(np.argsort(sims)[-K:])
        siblings = set(idxs) - {i}
        if siblings & top_k:
            hit = True
            break
    if hit:
        hit_count += 1
    per_func_hit[func_id] = hit

hit_rate = hit_count / functions_with_siblings if functions_with_siblings else 0.0

report = {
    "n_functions_with_siblings": int(functions_with_siblings),
    "self_mean": float(self_sims.mean()),
    "self_std": float(self_sims.std()),
    "self_median": float(np.median(self_sims)),
    "self_5th_percentile": float(np.percentile(self_sims, 5)),
    "cross_mean": float(cross_sims.mean()),
    "cross_std": float(cross_sims.std()),
    "cross_median": float(np.median(cross_sims)),
    "cross_95th_percentile": float(np.percentile(cross_sims, 95)),
    "gap": float(self_sims.mean() - cross_sims.mean()),
    "sibling_hitrate_at_5": float(hit_rate),
    "n_funcs_below_cross_mean": int((self_sims < cross_sims.mean()).sum()),
    "pct_funcs_below_cross_mean": float((self_sims < cross_sims.mean()).mean()),
}
print(json.dumps(report, indent=2))
Path("self_vs_cross.json").write_text(json.dumps(report, indent=2))

# --- Список функций, где siblings не попали в топ-5 (для разбора глазами) ---
miss = [fid for fid, hit in per_func_hit.items() if not hit]
Path("sibling_misses.json").write_text(json.dumps(miss, indent=2))
print(f"\nFunctions with no sibling in top-{K}: {len(miss)}")
print(f"Saved: sibling_misses.json ({len(miss)} ids)")
