"""Tuned classical baselines for the quantum-vs-classical IDS comparison (task #5).

Goal: make the classical side strong enough that a Q1 reviewer cannot dismiss the
comparison as "the quantum model only wins because the baselines are untuned".
Every baseline here is hyperparameter-searched (not default) with stratified
cross-validation on the *training* set only, then refit and scored once on the
held-out official NSL-KDD test set.

Models
------
  * random_forest  : sklearn RandomForestClassifier
  * xgboost        : XGBClassifier (hist)
  * svm_rbf        : SVC(kernel='rbf')
  * mlp            : TunedMLP (torch) - wider/deeper than the param-matched MLP
  * cnn1d          : CNN1D (torch) - 1D-CNN over the feature vector

Class imbalance
---------------
NSL-KDD's training set is only mildly imbalanced (~46.5% attack), and the test
set is *more* attack-heavy (~56.9%) with a large R2L shift. We therefore prefer
cost-sensitive learning (class weights / scale_pos_weight) over SMOTE as the
default, and expose SMOTE as an ablation. See `paper/methodology.md` for the
justification; `make_class_weight` / `resample_smote` implement both.

These live in their own module (not quantum_model.py, owned by quantum-dev) and
plug into the harness via `register_baselines(model_factory)` and the torch
builders `build_torch_baseline`. The tuned hyperparameters discovered offline can
be frozen into configs/ for the reproducible sweep (task #6/#7).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

# Parallelism budget for sklearn estimators / searches. The project shares ONE
# laptop (12 cores) and a single 4GB GPU with the quantum jobs, so n_jobs=-1
# oversubscribes the box and thrashes everything. Default to a modest 4 workers;
# override with env QIDS_N_JOBS when running solo. Used everywhere instead of -1.
N_JOBS = int(os.environ.get("QIDS_N_JOBS", "4"))


# --------------------------------------------------------------------------- #
# Torch deep baselines (kept here so quantum_model.py stays quantum-dev's file)
# --------------------------------------------------------------------------- #
class TunedMLP(nn.Module):
    """Strong classical MLP baseline: configurable width/depth/dropout + BN.

    Distinct from quantum_model.ClassicalMLP, which is intentionally tiny and
    parameter-matched to the HybridQNN for the like-for-like comparison. This one
    is allowed to be as good as it can on the full feature view.
    """

    def __init__(self, in_dim: int, hidden=(128, 64), dropout: float = 0.3,
                 n_classes: int = 2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h),
                       nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """1D-CNN baseline: treats the feature vector as a length-`in_dim` signal.

    Two Conv1d blocks (ReLU + BatchNorm + pooling) then a small classifier head.
    A common competitive deep baseline for tabular IDS features; included so the
    quantum model is benchmarked against a modern deep model, not only shallow ones.
    """

    def __init__(self, in_dim: int, channels=(32, 64), kernel: int = 3,
                 dropout: float = 0.3, n_classes: int = 2):
        super().__init__()
        k = max(1, min(kernel, in_dim))
        pad = k // 2
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=k, padding=pad), nn.ReLU(),
            nn.BatchNorm1d(c1), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=k, padding=pad), nn.ReLU(),
            nn.BatchNorm1d(c2), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(c2, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)            # (B, in_dim) -> (B, 1, in_dim)
        x = self.features(x)
        return self.classifier(x)


def build_torch_baseline(name: str, in_dim: int, hp: dict | None = None) -> nn.Module:
    """Construct a torch deep baseline ('mlp' or 'cnn1d') from a hyperparam dict."""
    hp = dict(hp or {})
    if name == "mlp":
        return TunedMLP(in_dim,
                        hidden=tuple(hp.get("hidden", (128, 64))),
                        dropout=hp.get("dropout", 0.3))
    if name == "cnn1d":
        return CNN1D(in_dim,
                     channels=tuple(hp.get("channels", (32, 64))),
                     kernel=hp.get("kernel", 3),
                     dropout=hp.get("dropout", 0.3))
    raise ValueError(f"unknown torch baseline {name!r} (use 'mlp' or 'cnn1d')")


# --------------------------------------------------------------------------- #
# Class-imbalance utilities (cost-sensitive learning vs SMOTE)
# --------------------------------------------------------------------------- #
def make_class_weight(y) -> dict[int, float]:
    """Balanced class weights {class: weight} (sklearn 'balanced' scheme)."""
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


def scale_pos_weight(y) -> float:
    """XGBoost scale_pos_weight = n_negative / n_positive."""
    y = np.asarray(y)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / pos if pos else 1.0


def resample_smote(X, y, seed: int = 42):
    """SMOTE oversampling of the minority class (ablation alternative to weights).

    Applied to the TRAINING set only -- never to validation/test -- to avoid
    leaking synthetic neighbours across the split.
    """
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=seed)
    Xr, yr = sm.fit_resample(X, y)
    return Xr, yr


# --------------------------------------------------------------------------- #
# Hyperparameter search spaces (sklearn estimators)
# --------------------------------------------------------------------------- #
def _rf_space():
    return {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [None, 10, 20, 30, 50],
        "max_features": ["sqrt", "log2", 0.5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }


def _xgb_space():
    return {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.3],
    }


def _svm_space():
    return {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.01, 0.1, 1.0],
    }


@dataclass
class TunedResult:
    """Outcome of a hyperparameter search for one sklearn baseline."""
    model_name: str
    best_params: dict[str, Any]
    cv_best_f1: float
    cv_results_summary: dict[str, Any] = field(default_factory=dict)
    search_time_s: float = 0.0


def _make_estimator(name: str, seed: int, class_weight, spw: float | None):
    """Base estimator with imbalance handling wired in, ready for the search."""
    if name == "random_forest":
        return RandomForestClassifier(
            random_state=seed, n_jobs=N_JOBS,
            class_weight=class_weight,   # 'balanced' or None
        )
    if name == "svm_rbf":
        return SVC(
            kernel="rbf", probability=True, random_state=seed,
            class_weight=class_weight,   # 'balanced' or None
        )
    if name == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost not installed in this env")
        return XGBClassifier(
            tree_method="hist", random_state=seed, n_jobs=N_JOBS,
            eval_metric="logloss",
            scale_pos_weight=(spw if spw is not None else 1.0),
        )
    raise ValueError(f"no search space for {name!r}")


def _space_for(name: str):
    return {"random_forest": _rf_space, "svm_rbf": _svm_space,
            "xgboost": _xgb_space}[name]()


def tune_sklearn_baseline(name: str, X, y, *, seed: int = 42, n_iter: int = 40,
                          cv_folds: int = 5, scoring: str = "f1",
                          imbalance: str = "class_weight",
                          svm_subsample: int = 12000,
                          search_subsample: int = 30000, verbose: int = 1) -> TunedResult:
    """Randomized hyperparameter search with stratified CV on the TRAINING set.

    imbalance:
      'class_weight' -> RF/SVM use class_weight='balanced'; XGB uses scale_pos_weight.
      'none'         -> no cost-sensitive reweighting (for the SMOTE ablation,
                        where X,y are already SMOTE-resampled by the caller).

    Returns the best params + CV score; does NOT touch the test set. The search
    runs on a stratified subsample of at most `search_subsample` rows
    (`svm_subsample` for the RBF-SVM, which is even more cost-sensitive): tuning on
    ~30k stratified rows gives stable hyperparameter rankings while keeping the
    whole grid affordable on the shared laptop. The chosen config is then refit on
    the FULL training set per seed by the runner (`fit_tuned_sklearn`), so the
    final models still see all the data.
    """
    X = np.asarray(X); y = np.asarray(y)
    use_weight = "balanced" if imbalance == "class_weight" else None
    spw = scale_pos_weight(y) if imbalance == "class_weight" else 1.0

    # Subsample the training data for the SEARCH only (tractability). SVM uses a
    # tighter cap than the tree/boosting models.
    from sklearn.model_selection import train_test_split
    cap = svm_subsample if name == "svm_rbf" else search_subsample
    if cap and len(X) > cap:
        Xfit, _, yfit, _ = train_test_split(
            X, y, train_size=cap, stratify=y, random_state=seed)
    else:
        Xfit, yfit = X, y

    est = _make_estimator(name, seed, use_weight, spw)
    space = _space_for(name)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    # Avoid NESTED parallelism: RF/XGB already parallelise across trees (n_jobs=
    # N_JOBS inside the estimator), so the outer search runs serially; SVM fits
    # single-threaded, so the search parallelises across folds/candidates instead.
    search_jobs = N_JOBS if name == "svm_rbf" else 1
    search = RandomizedSearchCV(
        est, space, n_iter=n_iter, scoring=scoring, cv=cv,
        random_state=seed, n_jobs=search_jobs, refit=False, verbose=verbose,
    )
    t0 = time.time()
    search.fit(Xfit, yfit)
    dt = time.time() - t0
    return TunedResult(
        model_name=name,
        best_params=search.best_params_,
        cv_best_f1=float(search.best_score_),
        cv_results_summary={
            "mean_test_f1_top": float(np.max(search.cv_results_["mean_test_score"])),
            "n_candidates": int(len(search.cv_results_["mean_test_score"])),
            "scoring": scoring, "cv_folds": cv_folds, "imbalance": imbalance,
            "search_subsample": int(len(Xfit)),
        },
        search_time_s=round(dt, 2),
    )


def fit_tuned_sklearn(name: str, best_params: dict, X, y, *, seed: int = 42,
                      imbalance: str = "class_weight", svm_max_train: int = 20000):
    """Refit a tuned sklearn estimator and return it.

    RF/XGB are fit on the full training set. The RBF-SVM has O(n^2..n^3) training
    cost and `probability=True` adds an internal 5-fold Platt-scaling CV, which is
    infeasible on the full ~126k-row NSL-KDD train set on the laptop; we therefore
    fit the SVM on a **stratified subsample** of at most `svm_max_train` rows
    (preserving the class ratio). This cap is recorded in the results metadata and
    stated in the methodology as a known, deliberate efficiency trade-off for the
    kernel SVM only.
    """
    use_weight = "balanced" if imbalance == "class_weight" else None
    spw = scale_pos_weight(y) if imbalance == "class_weight" else 1.0
    est = _make_estimator(name, seed, use_weight, spw)
    est.set_params(**best_params)
    X = np.asarray(X); y = np.asarray(y)
    if name == "svm_rbf" and svm_max_train and len(X) > svm_max_train:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, train_size=svm_max_train,
                                      stratify=y, random_state=seed)
    est.fit(X, y)
    return est


# --------------------------------------------------------------------------- #
# Torch deep-baseline hyperparameter grids (searched by run_baselines.py)
# --------------------------------------------------------------------------- #
def torch_grid(name: str) -> list[dict]:
    """Small explicit grid for the torch deep baselines (mlp / cnn1d).

    Kept small and explicit (not random) so a full grid is affordable on the
    laptop GPU while still being a genuine search rather than a single default.
    """
    if name == "mlp":
        return [
            {"hidden": (64, 32),  "dropout": 0.2, "lr": 1e-3},
            {"hidden": (128, 64), "dropout": 0.3, "lr": 1e-3},
            {"hidden": (256, 128), "dropout": 0.4, "lr": 5e-4},
            {"hidden": (128, 64, 32), "dropout": 0.3, "lr": 1e-3},
        ]
    if name == "cnn1d":
        return [
            {"channels": (16, 32), "kernel": 3, "dropout": 0.2, "lr": 1e-3},
            {"channels": (32, 64), "kernel": 3, "dropout": 0.3, "lr": 1e-3},
            {"channels": (32, 64), "kernel": 5, "dropout": 0.3, "lr": 5e-4},
            {"channels": (64, 128), "kernel": 3, "dropout": 0.4, "lr": 5e-4},
        ]
    raise ValueError(f"no torch grid for {name!r}")
