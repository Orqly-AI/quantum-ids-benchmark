"""Smoke tests for the unified data.load() pipeline.

Run inside the qml env:
    wsl -d Ubuntu-22.04 -- bash -lc 'bash ~/run.sh test_data.py'

Datasets whose files are not present are SKIPPED (not failed), so this works on a
machine that only has NSL-KDD downloaded. Exit code is non-zero only on a real
assertion failure.
"""
from __future__ import annotations
import sys
import numpy as np
import data


def _check(name, n_features, reduction, binary):
    Xtr, ytr, Xte, yte, meta = data.load_meta(
        name, n_features=n_features, reduction=reduction, binary=binary)

    # Shape contract
    assert Xtr.ndim == 2 and Xte.ndim == 2, f"{name}: X must be 2-D"
    if reduction == "none":
        # reduction='none' keeps ALL encoded dims (n_features is ignored).
        assert Xtr.shape[1] == meta["raw_dim_after_encoding"], \
            f"{name}/none: width {Xtr.shape[1]} != raw {meta['raw_dim_after_encoding']}"
    else:
        assert Xtr.shape[1] == n_features, \
            f"{name}: expected {n_features} features, got {Xtr.shape[1]}"
    assert Xte.shape[1] == Xtr.shape[1], f"{name}: train/test feature width mismatch"
    assert Xtr.shape[0] == ytr.shape[0], f"{name}: train X/y length mismatch"
    assert Xte.shape[0] == yte.shape[0], f"{name}: test X/y length mismatch"

    # dtype contract
    assert Xtr.dtype == np.float32 and Xte.dtype == np.float32, f"{name}: X dtype"
    assert ytr.dtype == np.int64 and yte.dtype == np.int64, f"{name}: y dtype"

    # Value sanity: finite, and (minmax) within [0, pi] band
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all(), f"{name}: non-finite X"
    assert Xtr.min() >= -1e-6 and Xtr.max() <= np.pi + 1e-4, \
        f"{name}: minmax range [{Xtr.min():.3f}, {Xtr.max():.3f}] outside [0, pi]"

    # Label contract
    if binary:
        assert set(np.unique(ytr)).issubset({0, 1}), f"{name}: non-binary labels"
        assert meta["n_classes"] == 2
    else:
        assert meta["n_classes"] >= 2, f"{name}: multiclass needs >=2 classes"
    assert ytr.min() >= 0, f"{name}: negative label"

    ev = meta["explained_variance"]
    ev_s = f"{ev:.3f}" if isinstance(ev, float) and ev == ev else "n/a"
    print(f"  PASS {name:8s} | {reduction:11s} | binary={binary} "
          f"| train={Xtr.shape} test={Xte.shape} | classes={meta['n_classes']} "
          f"| evr={ev_s}")
    return meta


def main():
    failures = []
    skipped = []
    # (dataset, n_features, reduction, binary)
    cases = [
        ("nslkdd", 8, "pca", True),
        ("nslkdd", 8, "autoencoder", True),
        ("nslkdd", 6, "pca", False),       # multiclass
        ("nslkdd", 8, "none", True),       # no reduction (keeps full dim)
        ("unsw", 8, "pca", True),
        ("unsw", 10, "autoencoder", False),
        ("cicids", 8, "pca", True),
        ("cicids", 8, "pca", False),
        ("toniot", 8, "pca", True),
        ("toniot", 8, "pca", False),
        ("toniot", 8, "autoencoder", True),
    ]
    for name, nf, red, binary in cases:
        try:
            _check(name, nf, red, binary)
        except FileNotFoundError as e:
            skipped.append((name, str(e).split('.')[0]))
            print(f"  SKIP {name:8s} | {red:11s} | data file not present")
        except AssertionError as e:
            failures.append(f"{name}/{red}/binary={binary}: {e}")
            print(f"  FAIL {name:8s} | {red:11s} | {e}")

    print("\n==== test_data summary ====")
    if skipped:
        seen = sorted({s[0] for s in skipped})
        print(f"skipped (no data): {seen}")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("All present-dataset checks passed.")


if __name__ == "__main__":
    main()
