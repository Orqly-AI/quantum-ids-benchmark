"""Tuned classical baseline sweep + multi-seed evaluation (task #5).

Produces the classical half of the paper's comparison table:
  * Tunes RF / XGBoost / SVM-RBF via stratified-CV randomized search (once, on a
    reference seed) and runs a small grid search for the torch MLP / 1D-CNN.
  * Refits every model over >=5 seeds and scores on the fixed NSL-KDD test set.
  * Aggregates mean +/- std (+95% CI) per metric.
  * Saves predictions of a reference seed so the runner / a downstream step can
    run McNemar vs the quantum model.

Usage (inside the qml env, via run.sh):
  python run_baselines.py --dataset nslkdd --view full --seeds 42 43 44 45 46
  python run_baselines.py --view both --models random_forest xgboost svm_rbf mlp cnn1d
  python run_baselines.py --imbalance smote          # ablation
  python run_baselines.py --quick                    # tiny search for a smoke test

Views:
  full  -> reduction='none' (all encoded dims). The headline classical numbers.
  pca   -> reduction='pca', n_features = qubit budget. Same-budget comparison.
  both  -> run each model on both views.

Output: results/baselines_<dataset>_<view>_<imbalance>.json with per-seed metrics,
aggregates, tuned hyperparameters, reference-seed predictions, and provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import load_meta, load_full
from seeding import seed_everything
from evaluation import metrics, aggregate_to_json
from baselines import (tune_sklearn_baseline, fit_tuned_sklearn, build_torch_baseline,
                       torch_grid, make_class_weight, resample_smote)
try:
    from provenance import env_snapshot
except Exception:  # pragma: no cover
    def env_snapshot():
        return {}

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
# Force CPU for torch deep baselines: these are tiny nets (cnn1d, mlp) and the
# shared 4GB VRAM is easily OOM'd on large datasets (UNSW 82k test rows).
# CPU is equally fast at this scale and avoids the CUDA OOM entirely.
DEVICE = torch.device("cpu")

SKLEARN_MODELS = ("random_forest", "xgboost", "svm_rbf")
TORCH_MODELS = ("mlp", "cnn1d")


# --------------------------------------------------------------------------- #
# Torch training (class-weighted CE; deterministic per seed)
# --------------------------------------------------------------------------- #
def train_torch(model, Xtr, ytr, Xte, yte, *, epochs=40, bs=256, lr=1e-3,
                class_weight=None, name="model"):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if class_weight is not None:
        w = torch.tensor([class_weight.get(0, 1.0), class_weight.get(1, 1.0)],
                         dtype=torch.float32, device=DEVICE)
        lossf = nn.CrossEntropyLoss(weight=w)
    else:
        lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    n = len(Xtr_t)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
        logits = model(Xte_t)
        proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
    m = metrics(yte, pred, proba)
    m["train_time_s"] = round(train_time, 2)
    return m, pred.tolist(), proba


def _torch_score_proba(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
    return pred, proba


def _sklearn_score(est, X):
    pred = est.predict(X)
    if hasattr(est, "predict_proba"):
        score = est.predict_proba(X)[:, 1]
    elif hasattr(est, "decision_function"):
        score = est.decision_function(X)
    else:
        score = None
    return pred, score


# --------------------------------------------------------------------------- #
# Hyperparameter search (done once, on a reference seed)
# --------------------------------------------------------------------------- #
def tune_all(models, Xtr, ytr, *, ref_seed, imbalance, n_iter, cv_folds,
             quick=False):
    """Return {model: best_hp_dict}. sklearn models searched via CV; torch via grid."""
    tuned = {}
    for m in models:
        if m in SKLEARN_MODELS:
            ni = min(n_iter, 4) if quick else n_iter
            # In quick mode also shrink the search subsample for a fast smoke test.
            sub = 5000 if quick else 30000
            print(f"\n[tune] {m}: randomized CV search (n_iter={ni}, folds={cv_folds})")
            res = tune_sklearn_baseline(m, Xtr, ytr, seed=ref_seed, n_iter=ni,
                                        cv_folds=cv_folds, imbalance=imbalance,
                                        search_subsample=sub)
            print(f"[tune] {m}: best F1={res.cv_best_f1:.4f} params={res.best_params}")
            tuned[m] = {"best_params": res.best_params, "cv_best_f1": res.cv_best_f1,
                        "search_time_s": res.search_time_s,
                        "cv_summary": res.cv_results_summary}
        elif m in TORCH_MODELS:
            grid = torch_grid(m)
            if quick:
                grid = grid[:1]
            print(f"\n[tune] {m}: grid search over {len(grid)} configs "
                  f"(stratified val split)")
            best = _tune_torch(m, Xtr, ytr, grid, ref_seed, imbalance, quick)
            print(f"[tune] {m}: best val-F1={best['val_f1']:.4f} hp={best['hp']}")
            tuned[m] = {"best_params": best["hp"], "cv_best_f1": best["val_f1"]}
        else:
            raise ValueError(f"unknown model {m!r}")
    return tuned


def _tune_torch(name, Xtr, ytr, grid, seed, imbalance, quick):
    """Grid-search a torch baseline on a stratified train/val split of the train set."""
    from sklearn.model_selection import train_test_split
    seed_everything(seed)
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr,
                                      random_state=seed)
    cw = make_class_weight(yt) if imbalance == "class_weight" else None
    in_dim = Xtr.shape[1]
    epochs = 8 if quick else 30
    best = {"val_f1": -1.0, "hp": grid[0]}
    for hp in grid:
        seed_everything(seed)
        model = build_torch_baseline(name, in_dim, hp)
        m, _, _ = train_torch(model, Xt, yt, Xv, yv, epochs=epochs,
                              lr=hp.get("lr", 1e-3), class_weight=cw, name=name)
        if m["f1"] > best["val_f1"]:
            best = {"val_f1": m["f1"], "hp": hp}
    return best


# --------------------------------------------------------------------------- #
# Multi-seed evaluation of tuned models on the fixed test set
# --------------------------------------------------------------------------- #
def evaluate_seeds(models, tuned, Xtr, ytr, Xte, yte, *, seeds, imbalance):
    """Refit each tuned model on each seed; score on the fixed test set.

    Returns (per_seed[model] -> list[metric dict], ref[model] -> {pred, score}).
    The reference-seed predictions AND probability scores (first seed) are kept so
    a downstream compare step can run McNemar (on preds) and the paired bootstrap
    (on scores) vs the quantum model.
    """
    per_seed = {m: [] for m in models}
    ref = {}
    in_dim = Xtr.shape[1]

    for si, seed in enumerate(seeds):
        print(f"\n==== seed {seed} ({si+1}/{len(seeds)}) ====")
        seed_everything(seed)

        # Imbalance handling: build the training data once per seed.
        if imbalance == "smote":
            Xtr_s, ytr_s = resample_smote(Xtr, ytr, seed=seed)
            cw = None
        else:  # class_weight (default) or none
            Xtr_s, ytr_s = Xtr, ytr
            cw = make_class_weight(ytr) if imbalance == "class_weight" else None

        for m in models:
            t0 = time.time()
            if m in SKLEARN_MODELS:
                # XGB/RF/SVM imbalance via class_weight/scale_pos_weight is handled
                # inside fit_tuned_sklearn unless we've SMOTE-resampled already.
                imb = "none" if imbalance == "smote" else imbalance
                est = fit_tuned_sklearn(m, tuned[m]["best_params"], Xtr_s, ytr_s,
                                        seed=seed, imbalance=imb)
                pred, score = _sklearn_score(est, Xte)
                met = metrics(yte, pred, score)
                met["train_time_s"] = round(time.time() - t0, 2)
                preds = pred.tolist()
            else:  # torch
                hp = tuned[m]["best_params"]
                model = build_torch_baseline(m, in_dim, hp)
                met, preds, score = train_torch(model, Xtr_s, ytr_s, Xte, yte,
                                                epochs=40, lr=hp.get("lr", 1e-3),
                                                class_weight=cw, name=m)
            per_seed[m].append(met)
            print(f"  {m:<14} F1={met['f1']:.4f} DR={met['detection_rate']:.4f} "
                  f"FPR={met['false_positive_rate']:.4f} "
                  f"AUC={met.get('roc_auc', float('nan')):.4f} "
                  f"({met['train_time_s']}s)")
            if si == 0:
                ref[m] = {"pred": preds,
                          "score": (score.tolist() if hasattr(score, "tolist")
                                    else (list(score) if score is not None else None))}
    return per_seed, ref


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_view(dataset, view, models, seeds, imbalance, n_iter, cv_folds, quick):
    """Run tuning + multi-seed eval for one feature view; write a results JSON."""
    # Trees/SVM/NN want z-scored features; data.py supports scale='standard'.
    if view == "full":
        # load_full keeps every encoded dim (all 122 for NSL-KDD) — the headline
        # classical view (data-engineer's convenience wrapper, with_meta=True).
        Xtr, ytr, Xte, yte, meta = load_full(dataset, binary=True,
                                             scale="standard", with_meta=True)
    else:
        # Same-budget view: PCA to the quantum qubit count.
        Xtr, ytr, Xte, yte, meta = load_meta(dataset, n_features=8,
                                             reduction="pca", binary=True,
                                             scale="standard", seed=seeds[0])
    reduction = meta.get("reduction", "none")
    print(f"\n##### VIEW={view} ({reduction}) dims={Xtr.shape[1]} "
          f"train={Xtr.shape[0]} test={Xte.shape[0]} "
          f"attack%train={ytr.mean():.3f} attack%test={yte.mean():.3f} #####")

    ref_seed = seeds[0]
    tuned = tune_all(models, Xtr, ytr, ref_seed=ref_seed, imbalance=imbalance,
                     n_iter=n_iter, cv_folds=cv_folds, quick=quick)

    per_seed, ref = evaluate_seeds(models, tuned, Xtr, ytr, Xte, yte,
                                   seeds=seeds, imbalance=imbalance)

    aggregates = {m: aggregate_to_json(per_seed[m]) for m in models}

    # CV-vs-test generalisation gap per model: CV F1 (model selection, on train
    # folds) minus mean test F1 (held-out KDDTest+). On NSL-KDD this is large and
    # positive — it quantifies the train/test distribution shift and is a headline
    # number for the honest-broker angle (gap-analyst features it in Results).
    gap = {}
    for m in models:
        cv_f1 = tuned[m].get("cv_best_f1")
        test_f1 = aggregates[m].get("f1", {}).get("mean")
        if cv_f1 is not None and test_f1 is not None:
            gap[m] = {"cv_f1": float(cv_f1), "test_f1_mean": float(test_f1),
                      "cv_minus_test_f1": float(cv_f1 - test_f1)}

    out = {
        "meta": {
            "dataset": dataset, "view": view, "reduction": reduction,
            "n_features": int(Xtr.shape[1]), "seeds": list(seeds),
            "imbalance": imbalance, "n_iter_search": n_iter, "cv_folds": cv_folds,
            "data_meta": meta, "device": str(DEVICE),
            "attack_rate_train": float(ytr.mean()),
            "attack_rate_test": float(yte.mean()),
        },
        "tuned_hyperparameters": tuned,
        "per_seed_metrics": per_seed,
        "aggregates": aggregates,
        "cv_vs_test_gap": gap,
        "reference_seed": ref_seed,
        "reference_predictions": {
            "y_true": yte.tolist(),
            # per model: {"pred": [...hard...], "score": [...P(attack)...]}
            "by_model": ref,
        },
        "provenance": env_snapshot(),
    }
    fn = os.path.join(RESULTS, f"baselines_{dataset}_{view}_{imbalance}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {fn}")

    # Summary table (mean +/- std) — headline metrics for the paper.
    print(f"\n========= {dataset} / {view} / imbalance={imbalance} "
          f"(mean +/- std over {len(seeds)} seeds) =========")
    cols = [("F1", "f1"), ("AUPRC", "auprc"), ("TPR@1%", "tpr_at_1pct_fpr"),
            ("DR", "detection_rate"), ("FPR", "false_positive_rate"),
            ("AUC", "roc_auc")]
    hdr = f"{'Model':<14}" + "".join(f"{name:>16}" for name, _ in cols)
    print(hdr); print("-" * len(hdr))
    for m in models:
        a = aggregates[m]
        def cell(k):
            v = a.get(k)
            return f"{v['mean']:.3f}±{v['std']:.3f}" if v else "n/a"
        print(f"{m:<14}" + "".join(f"{cell(k):>16}" for _, k in cols))

    # Generalisation-gap line (CV vs KDDTest+).
    if gap:
        print(f"\n--- CV-vs-test F1 gap (distribution shift) ---")
        for m in models:
            g = gap.get(m)
            if g:
                print(f"  {m:<14} CV F1={g['cv_f1']:.3f}  test F1={g['test_f1_mean']:.3f}"
                      f"  gap={g['cv_minus_test_f1']:+.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nslkdd")
    ap.add_argument("--view", choices=["full", "pca", "both"], default="full")
    ap.add_argument("--models", nargs="+",
                    default=["random_forest", "xgboost", "svm_rbf", "mlp", "cnn1d"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--imbalance", choices=["class_weight", "smote", "none"],
                    default="class_weight")
    ap.add_argument("--n-iter", type=int, default=40, help="randomized search iters")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true",
                    help="tiny search + few epochs for a smoke test")
    a = ap.parse_args()

    views = ["full", "pca"] if a.view == "both" else [a.view]
    for v in views:
        run_view(a.dataset, v, a.models, a.seeds, a.imbalance,
                 a.n_iter, a.cv_folds, a.quick)


if __name__ == "__main__":
    main()
