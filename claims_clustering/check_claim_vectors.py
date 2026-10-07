"""Sanity check для claim_vectors.npy."""
import json
from pathlib import Path

import numpy as np

VECTORS = Path("claim_vectors.npy")
META = Path("claim_vectors_meta.json")
OUT = Path("claim_vectors_sanity.json")


def main():
    X = np.load(VECTORS)
    meta = json.loads(META.read_text())

    report = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum()),
        "constant_columns": [],
        "non_constant_count": 0,
        "warnings": [],
    }

    # Константные столбцы
    stds = X.std(axis=0)
    constant_cols = np.where(stds < 1e-8)[0].tolist()
    report["constant_columns"] = constant_cols
    report["non_constant_count"] = X.shape[1] - len(constant_cols)

    # Корреляция с длиной
    uids = meta["uids"]
    # Длину flow не восстановим из векторов, поэтому берём первую компоненту
    # как прокси и предупреждаем, если она имеет экстремальный разброс
    col_variances = X.var(axis=0)
    if col_variances.max() > 1e6:
        report["warnings"].append(
            f"Max column variance {col_variances.max():.1e} — possibly scaling issue"
        )

    # Проверки
    if report["nan_count"] > 0:
        report["warnings"].append("NaN present — inspect build pipeline")
    if report["inf_count"] > 0:
        report["warnings"].append("Inf present — inspect normalization")
    if report["non_constant_count"] < 10:
        report["warnings"].append(
            f"Only {report['non_constant_count']} non-constant columns — "
            f"features may not discriminate"
        )
    if len(constant_cols) > 0:
        report["warnings"].append(
            f"{len(constant_cols)} constant columns — should be dropped"
        )

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"Rows: {report['n_rows']}")
    print(f"Features: {report['n_features']}")
    print(f"NaN: {report['nan_count']}")
    print(f"Inf: {report['inf_count']}")
    print(f"Constant columns: {len(constant_cols)}")
    print(f"Non-constant: {report['non_constant_count']}")
    if report["warnings"]:
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  - {w}")
    else:
        print("\nNo warnings.")


if __name__ == "__main__":
    main()