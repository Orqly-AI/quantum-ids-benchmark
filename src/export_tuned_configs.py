"""Export tuned classical hyperparameters from baselines JSONs into a configs YAML.

After run_baselines.py produces results/baselines_<dataset>_<view>_<imbalance>.json,
this reads the discovered `tuned_hyperparameters` and emits a configs/ snippet that
sw-engineer's harness sweep (task #7) can consume: each classical model becomes an
ExperimentConfig with the tuned values dropped into `extra.best_params`, which the
factory applies via set_params (RF/XGB/SVM) or build_torch_baseline (MLP/CNN).

Usage:
    python export_tuned_configs.py                       # all baselines_*.json
    python export_tuned_configs.py --dataset nslkdd --view full
    python export_tuned_configs.py --out ../configs/classical_tuned.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
CONFIGS = os.path.join(HERE, "..", "configs")

# baselines model name -> harness model_type
_MODEL_TYPE = {
    "random_forest": "random_forest",
    "xgboost": "xgboost",
    "svm_rbf": "svm_rbf",
    "mlp": "tuned_mlp",
    "cnn1d": "cnn1d",
}


def export(paths, out_path):
    runs = []
    for path in paths:
        d = json.load(open(path))
        meta = d.get("meta", {})
        dataset = meta.get("dataset")
        view = meta.get("view")
        n_features = meta.get("n_features")
        reduction = meta.get("reduction", "none")
        imbalance = meta.get("imbalance", "class_weight")
        tuned = d.get("tuned_hyperparameters", {})
        for name, info in tuned.items():
            mt = _MODEL_TYPE.get(name)
            if mt is None:
                continue
            runs.append({
                "name": f"{dataset}_{view}_{name}_tuned",
                "dataset": dataset,
                "model_type": mt,
                "n_features": int(n_features) if n_features else 0,
                "reduction": reduction,
                "scale": "standard",
                "binary": True,
                "imbalance": imbalance,
                # The factory reads extra.best_params for sklearn set_params /
                # build_torch_baseline; cv_best_f1 kept for provenance only.
                "extra": {"best_params": info.get("best_params", {}),
                          "cv_best_f1": info.get("cv_best_f1")},
            })
    doc = {"defaults": {"seed": 42, "epochs": 40}, "runs": runs}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print(f"[export] {len(runs)} tuned config(s) from {len(paths)} file(s) -> {out_path}")
    for r in runs:
        print(f"   {r['name']:<32} {r['model_type']:<14} "
              f"params={r['extra']['best_params']}")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--dataset", default="*")
    ap.add_argument("--view", default="*")
    ap.add_argument("--imbalance", default="*")
    ap.add_argument("--out", default=os.path.join(CONFIGS, "classical_tuned.yaml"))
    a = ap.parse_args()
    pattern = os.path.join(a.results_dir,
                           f"baselines_{a.dataset}_{a.view}_{a.imbalance}.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no baselines JSON matching {pattern}")
    export(paths, a.out)


if __name__ == "__main__":
    main()
