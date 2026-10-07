"""
plot_hdbscan_umap_clusters.py

Визуализация кластеров UMAP + HDBSCAN на claim-векторах (VULNERABLE-only).

Идея:
  - кластеризация HDBSCAN делается в 5D UMAP-эмбеддинге (как в основном пайплайне);
  - визуализация — отдельный 2D UMAP-прогон (для глаз).

Использование как модуль:
    from plot_hdbscan_umap_clusters import plot_clusters
    plot_clusters(
        vectors_path="claim_vectors.npy",
        verdicts_path="../flow_harness/v0_outputs.jsonl",
        out_path="umap_hdbscan_clusters.png",
        min_cluster_size=5, min_samples=3,
    )

Использование как скрипт:
    python plot_hdbscan_umap_clusters.py \
        --vectors claim_vectors.npy \
        --verdicts ../flow_harness/v0_outputs.jsonl \
        --out umap_hdbscan_clusters.png \
        --mcs 5 --ms 3 \
        --func-ids func_ids.npy \
        --draw-sibling-edges
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

import umap
import hdbscan


# ----------------------------- загрузка данных -----------------------------

def load_vulnerable_vectors(
    vectors_path: str | Path,
    verdicts_path: str | Path,
    verdict_key: str = "verdict",
    vulnerable_value: str = "VULNERABLE",
) -> np.ndarray:
    """Возвращает L2-нормализованные векторы только для VULNERABLE-вердиктов."""
    X = np.load(vectors_path)
    is_vuln = np.array([
        json.loads(line).get(verdict_key) == vulnerable_value
        for line in open(verdicts_path)
    ])
    if len(is_vuln) != len(X):
        raise ValueError(
            f"Число вердиктов ({len(is_vuln)}) != числу векторов ({len(X)})"
        )
    X_vuln = normalize(X[is_vuln], norm="l2", axis=1)
    return X_vuln


# ------------------------------- UMAP + HDBSCAN ----------------------------

def reduce_umap(
    X: np.ndarray,
    n_components: int,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def cluster_hdbscan(
    X_umap: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(X_umap)
    return labels, clusterer


# ----------------------------------- plot ----------------------------------

def _scatter_clusters(
    ax,
    X_2d: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    point_size: int = 18,
    noise_size: int = 8,
    label_centroids: bool = True,
):
    mask_noise = labels == -1
    if mask_noise.any():
        ax.scatter(
            X_2d[mask_noise, 0], X_2d[mask_noise, 1],
            c="lightgray", s=noise_size, alpha=0.4,
            label=f"noise ({int(mask_noise.sum())})",
        )

    unique = sorted(set(labels.tolist()) - {-1})
    cmap = plt.get_cmap("tab20", max(len(unique), 1))

    for i, cl in enumerate(unique):
        m = labels == cl
        alpha = 0.85
        if probabilities is not None:
            # альфа пропорциональна уверенности кластера
            alpha_vals = 0.15 + 0.75 * probabilities[m]
        else:
            alpha_vals = alpha
        ax.scatter(
            X_2d[m, 0], X_2d[m, 1],
            c=[cmap(i)], s=point_size, alpha=alpha_vals,
            label=f"cluster {cl} (n={int(m.sum())})",
        )

        if label_centroids and m.sum() > 0:
            cx, cy = X_2d[m, 0].mean(), X_2d[m, 1].mean()
            ax.text(
                cx, cy, str(cl),
                fontsize=11, weight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="black", lw=0.5),
            )

    return unique


def _draw_sibling_edges(ax, X_2d: np.ndarray, func_ids: np.ndarray,
                        color: str = "black", lw: float = 0.3,
                        alpha: float = 0.15):
    """Соединяет точки одного func_id (siblings по температуре)."""
    for fid in np.unique(func_ids):
        idx = np.where(func_ids == fid)[0]
        if len(idx) < 2:
            continue
        pts = X_2d[idx]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                ax.plot(
                    [pts[i, 0], pts[j, 0]],
                    [pts[i, 1], pts[j, 1]],
                    c=color, lw=lw, alpha=alpha,
                )


def _cluster_mechanism_table(labels: np.ndarray,
                             mechanism_ids: np.ndarray) -> np.ndarray:
    """Матрица cluster × mechanism (число точек)."""
    clusters = sorted(set(labels.tolist()) - {-1})
    mechs = sorted(set(mechanism_ids.tolist()))
    table = np.zeros((len(clusters), len(mechs)), dtype=int)
    c_idx = {c: i for i, c in enumerate(clusters)}
    m_idx = {m: j for j, m in enumerate(mechs)}
    for lab, mech in zip(labels, mechanism_ids):
        if lab == -1:
            continue
        table[c_idx[lab], m_idx[mech]] += 1
    return table


# ------------------------------- main entry --------------------------------

def plot_clusters(
    vectors_path: str | Path = "claim_vectors.npy",
    verdicts_path: str | Path = "../flow_harness/v0_outputs.jsonl",
    out_path: str | Path = "umap_hdbscan_clusters.png",
    min_cluster_size: int = 5,
    min_samples: int = 3,
    n_neighbors: int = 15,
    random_state: int = 42,
    func_ids_path: Optional[str | Path] = None,
    mechanism_ids_path: Optional[str | Path] = None,
    draw_sibling_edges: bool = False,
    use_probabilities: bool = True,
    dpi: int = 200,
    show: bool = False,
    verbose: bool = True,
):
    """
    Полный пайплайн: загрузка → UMAP 5D → HDBSCAN → UMAP 2D → рисунок.

    Возвращает dict с ключами:
        labels, probabilities, X_2d, X_umap5, cluster_mechanism_table (если есть)
    """
    # 1. Данные
    X_vuln = load_vulnerable_vectors(vectors_path, verdicts_path)
    if verbose:
        print(f"VULNERABLE: {len(X_vuln)}")

    # 2. UMAP 5D для кластеризации
    if verbose:
        print("Running UMAP 5D (for clustering)...")
    X_umap5 = reduce_umap(
        X_vuln, n_components=5, n_neighbors=n_neighbors,
        min_dist=0.0, metric="cosine", random_state=random_state,
    )

    # 3. HDBSCAN в 5D
    if verbose:
        print(f"Running HDBSCAN (mcs={min_cluster_size}, ms={min_samples})...")
    labels, clusterer = cluster_hdbscan(
        X_umap5,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels.tolist()) - {-1})
    n_noise = int((labels == -1).sum())
    if verbose:
        print(f"K={n_clusters}, noise={n_noise} "
              f"({n_noise / len(labels):.1%})")

    # 4. UMAP 2D для визуализации
    if verbose:
        print("Running UMAP 2D (for visualization)...")
    X_2d = reduce_umap(
        X_vuln, n_components=2, n_neighbors=n_neighbors,
        min_dist=0.1, metric="cosine", random_state=random_state,
    )

    # 5. Рисунок
    fig, ax = plt.subplots(figsize=(10, 8))

    if draw_sibling_edges:
        if func_ids_path is None:
            raise ValueError("draw_sibling_edges=True, но func_ids_path не задан")
        func_ids = np.load(func_ids_path)
        # учитываем, что func_ids должны быть выровнены по VULNERABLE-подмножеству
        if len(func_ids) != len(X_vuln):
            raise ValueError(
                f"len(func_ids)={len(func_ids)} != len(X_vuln)={len(X_vuln)}. "
                "Убедитесь, что func_ids сохранены ТОЛЬКО для VULNERABLE-flow."
            )
        _draw_sibling_edges(ax, X_2d, func_ids)

    _scatter_clusters(
        ax, X_2d, labels,
        probabilities=probabilities if use_probabilities else None,
    )

    ax.set_title(
        f"UMAP 2D | HDBSCAN K={n_clusters}, "
        f"noise={n_noise} ({n_noise / len(labels):.1%})"
    )
    ax.legend(loc="best", fontsize=8, ncol=2, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    if verbose:
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)

    result = {
        "labels": labels,
        "probabilities": probabilities,
        "X_2d": X_2d,
        "X_umap5": X_umap5,
    }

    # 6. Опционально: таблица cluster × mechanism
    if mechanism_ids_path is not None:
        mech = np.load(mechanism_ids_path)
        if len(mech) != len(X_vuln):
            raise ValueError(
                f"len(mechanism_ids)={len(mech)} != len(X_vuln)={len(X_vuln)}"
            )
        table = _cluster_mechanism_table(labels, mech)
        result["cluster_mechanism_table"] = table
        if verbose:
            print("\ncluster × mechanism (rows=cluster, cols=mechanism):")
            print(table)

    return result


# ----------------------------------- CLI -----------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UMAP + HDBSCAN визуализация кластеров claim-векторов."
    )
    p.add_argument("--vectors", default="claim_vectors.npy")
    p.add_argument("--verdicts", default="../flow_harness/v0_outputs.jsonl")
    p.add_argument("--out", default="umap_hdbscan_clusters.png")
    p.add_argument("--mcs", type=int, default=15,
                   help="HDBSCAN min_cluster_size")
    p.add_argument("--ms", type=int, default=3,
                   help="HDBSCAN min_samples")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--func-ids", default=None,
                   help=".npy с func_id для каждого VULNERABLE-flow "
                        "(для рёбер siblings)")
    p.add_argument("--mechanism-ids", default=None,
                   help=".npy с mechanism_id для таблицы cluster × mechanism")
    p.add_argument("--draw-sibling-edges", action="store_true")
    p.add_argument("--no-probabilities", action="store_true",
                   help="не использовать probability как alpha")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_clusters(
        vectors_path=args.vectors,
        verdicts_path=args.verdicts,
        out_path=args.out,
        min_cluster_size=args.mcs,
        min_samples=args.ms,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
        func_ids_path=args.func_ids,
        mechanism_ids_path=args.mechanism_ids,
        draw_sibling_edges=args.draw_sibling_edges,
        use_probabilities=not args.no_probabilities,
        dpi=args.dpi,
        show=args.show,
    )