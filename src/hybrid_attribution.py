"""Attribution controls for the 4-qubit hybrid's low-FPR result on NSL-KDD.

The hybrid (Linear(4,4)+Tanh -> 4-qubit angle-encoded circuit with three
StronglyEntangling layers -> Linear(4,2)) is the one quantum model whose result
survived FDR in the audit, but it was only ever compared with a random forest
on eight features. Here its quantum layer is replaced, inside the identical
pipeline (same four PCA features, same seeded 2,000-row subsample, same pre and
post layers, optimiser, loss, epochs, batch size and seeds), by:

  fourier   the circuit's exact classical function class. With RY angle encoding,
            each <Z_j> is a trigonometric polynomial in the four input angles with
            frequencies in {-1, 0, 1}: a linear combination of the 81 products
            prod_i f_i(a_i), f_i in {1, cos a_i, sin a_i} (Schuld et al., 2021).
            This layer is Linear(81, 4) on those features, a strict superset.
  mlp       a classical two-layer tanh block of the same width (40 parameters
            against the circuit's 36).
  linear    no middle layer at all (pre feeds post directly).
  hybrid    the quantum model itself, trained by the campaign's own code path.

Modes:
  train        one model and seed; writes <model>_s<seed>.json and scores .npz
  certificate  fit the 81-term expansion to a trained circuit and report how
               exactly it reproduces the circuit and the model's scores
  summary      per-seed paired bootstraps and per-attack-type detection at 1% FPR

Run from src/:  python hybrid_attribution.py train --model fourier --seed 42
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from kernel_analysis import score_metrics

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "hybrid_attribution")
SEEDS = (42, 43, 44, 45, 46)
MODELS = ("hybrid", "fourier", "mlp", "linear")


# --------------------------------------------------------------------------- #
# data and configuration: exactly the published run, seed varied
# --------------------------------------------------------------------------- #
def config(seed):
    from config import ExperimentConfig
    from hybrid_hardware import CFG
    d = json.loads(json.dumps(CFG))
    d["seed"] = int(seed)
    return ExperimentConfig.from_dict(d).validate()


def load(seed):
    """seed_everything -> load_data -> seeded subsample, as runner.run_config does."""
    from runner import load_data
    from seeding import seed_everything
    cfg = config(seed)
    seed_everything(cfg.seed)
    Xtr, ytr, Xte, yte = load_data(cfg)
    idx = np.random.RandomState(cfg.seed).permutation(len(Xtr))[:cfg.q_subsample]
    return cfg, Xtr[idx], ytr[idx], Xte, yte


def test_attack_names():
    from data import load_meta
    _, _, _, ymt, meta = load_meta("nslkdd", 4, "pca", False, "minmax", 42)
    return np.array(meta["class_names"])[ymt]


# --------------------------------------------------------------------------- #
# classical twins
# --------------------------------------------------------------------------- #
def trig_basis(a):
    """(B, n) angles -> (B, 3**n) products of {1, cos a_i, sin a_i}; torch or numpy."""
    try:
        import torch
        if isinstance(a, torch.Tensor):
            f = torch.stack([torch.ones_like(a), torch.cos(a), torch.sin(a)], -1)
            out = f[:, 0]
            for i in range(1, a.shape[1]):
                out = (out.unsqueeze(-1) * f[:, i].unsqueeze(1)).reshape(a.shape[0], -1)
            return out
    except ImportError:  # pragma: no cover
        pass
    a = np.asarray(a, float)
    f = np.stack([np.ones_like(a), np.cos(a), np.sin(a)], -1)
    out = f[:, 0]
    for i in range(1, a.shape[1]):
        out = (out[:, :, None] * f[:, i][:, None, :]).reshape(a.shape[0], -1)
    return out


def build_twin(kind, n=4):
    import torch.nn as nn

    class Trig(nn.Module):
        def forward(self, a):
            return trig_basis(a)

    class Twin(nn.Module):
        def __init__(self, middle):
            super().__init__()
            self.pre = nn.Sequential(nn.Linear(n, n), nn.Tanh())
            self.middle = middle
            self.post = nn.Linear(n, 2)

        def forward(self, x):
            return self.post(self.middle(self.pre(x)))

    middle = {"fourier": lambda: nn.Sequential(Trig(), nn.Linear(3 ** n, n)),
              "mlp": lambda: nn.Sequential(nn.Linear(n, n), nn.Tanh(), nn.Linear(n, n), nn.Tanh()),
              "linear": lambda: nn.Identity()}[kind]()
    return Twin(middle)


def train_twin(model, Xtr, ytr, Xte, cfg):
    """runner._train_torch's loop, on CPU, for a model without a quantum backend."""
    import torch
    import torch.nn as nn
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    classes, counts = np.unique(ytr, return_counts=True)
    w = counts.sum() / (len(classes) * counts)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)
    n = len(Xt)
    t0 = time.time()
    for _ in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.tensor(Xte, dtype=torch.float32)), 1)[:, 1].numpy()
    return proba, time.time() - t0


def n_params(model, trainable_middle_only=False):
    ps = [(k, p) for k, p in model.named_parameters()]
    if trainable_middle_only:
        ps = [(k, p) for k, p in ps if not (k.startswith("pre") or k.startswith("post"))]
    return int(sum(p.numel() for _, p in ps))


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def train(kind, seed):
    import logging
    from seeding import seed_everything
    cfg, Xtr, ytr, Xte, yte = load(seed)
    if kind == "hybrid":
        from model_factory import build_model
        from runner import _train_torch
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        model, _ = build_model(cfg)
        _, proba, train_time, _, _ = _train_torch(model, Xtr, ytr, Xte, yte, cfg, logging.getLogger("attr"))
    else:
        seed_everything(cfg.seed)            # same RNG state at construction as the hybrid path
        model = build_twin(kind)
        proba, train_time = train_twin(model, Xtr, ytr, Xte, cfg)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(os.path.join(RESULTS, f"{kind}_s{seed}.npz"), y_test=yte, score=proba, **sd)
    out = {"model": kind, "seed": seed, "train_time_s": train_time,
           "params_total": n_params(model), "params_middle": n_params(model, True),
           "metrics": score_metrics(yte, proba)}
    pub = sorted(glob.glob(os.path.join(os.path.dirname(RESULTS),
                                        f"nslkdd_hybrid_qnn_q4_l3_angle_s{seed}_*.json")))
    if kind == "hybrid" and pub:
        ps = np.asarray(json.load(open(pub[0], encoding="utf-8"))["predictions"]["y_score"], float)
        out["published_file"] = os.path.basename(pub[0])
        out["max_abs_diff_vs_published"] = float(np.max(np.abs(ps - proba)))
    json.dump(out, open(os.path.join(RESULTS, f"{kind}_s{seed}.json"), "w"), indent=2)
    return out


RF_VARIANTS = {"rf_f4_full": (4, False),    # the hybrid's four features, full training file
               "rf_f4_sub": (4, True),      # the hybrid's four features and seeded 2,000-row subsample
               "rf_f8_sub": (8, True)}      # the published forest's eight features, same subsample


def rf_control(variant, seed):
    """The published random forest (runner/model_factory path, balanced class weights) on the
    hybrid's input: separates the effect of the model class from that of the input it sees."""
    from config import ExperimentConfig
    from model_factory import build_model
    from runner import load_data
    from seeding import seed_everything
    n_feat, sub = RF_VARIANTS[variant]
    pub = sorted(glob.glob(os.path.join(os.path.dirname(RESULTS), "nslkdd_random_forest_q8_l3_angle_s42_rf_*.json")))[0]
    d = json.load(open(pub, encoding="utf-8"))["config"]
    d.update(seed=int(seed), n_features=n_feat, n_qubits=n_feat)
    cfg = ExperimentConfig.from_dict(d).validate()
    seed_everything(cfg.seed)
    Xtr, ytr, Xte, yte = load_data(cfg)
    if sub:   # identical rows to the hybrid's subsample: same permutation of the same file
        idx = np.random.RandomState(cfg.seed).permutation(len(Xtr))[:2000]
        Xtr, ytr = Xtr[idx], ytr[idx]
    model, kind = build_model(cfg)
    assert kind == "sklearn"
    t0 = time.time()
    model.fit(Xtr, ytr)
    score = model.predict_proba(Xte)[:, 1]
    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(os.path.join(RESULTS, f"{variant}_s{seed}.npz"), y_test=yte, score=score)
    out = {"model": variant, "seed": seed, "n_features": n_feat, "n_train": int(len(ytr)),
           "train_time_s": time.time() - t0, "metrics": score_metrics(yte, score)}
    json.dump(out, open(os.path.join(RESULTS, f"{variant}_s{seed}.json"), "w"), indent=2)
    return out


def certificate(seed=42, n_fit=4000, n_check=4000):
    """Least-squares fit of the 81-term expansion to the trained circuit's <Z_j>."""
    from hybrid_hardware import exact_expvals, head, pre
    d = np.load(os.path.join(RESULTS, f"hybrid_s{seed}.npz"))
    sd = {k: d[k] for k in d.files}
    W = sd["qlayer.weights"]
    rng = np.random.RandomState(0)
    A_fit = rng.uniform(-1, 1, (n_fit, 4))           # tanh keeps the angles in (-1, 1)
    A_chk = rng.uniform(-1, 1, (n_check, 4))
    Z_fit, Z_chk = exact_expvals(A_fit, W), exact_expvals(A_chk, W)
    Phi = trig_basis(A_fit)
    coef, *_ = np.linalg.lstsq(Phi, Z_fit, rcond=None)
    err_fit = float(np.max(np.abs(Phi @ coef - Z_fit)))
    err_chk = float(np.max(np.abs(trig_basis(A_chk) @ coef - Z_chk)))
    # the same fit without the 2**4 = 16 products in which all four factors are non-constant
    order = np.array([sum(1 for q in np.base_repr(i, 3).zfill(4) if q != "0") for i in range(Phi.shape[1])])
    keep = order < 4
    c3, *_ = np.linalg.lstsq(Phi[:, keep], Z_fit, rcond=None)
    err_no4 = float(np.max(np.abs(trig_basis(A_chk)[:, keep] @ c3 - Z_chk)))
    # the model's own test inputs: classical twin with the fitted coefficients vs the hybrid
    _, _, _, Xte, yte = load(seed)
    A_te = pre(Xte, sd)
    s_twin = head(trig_basis(A_te) @ coef, sd)
    err_score = float(np.max(np.abs(s_twin - sd["score"])))
    nz = int(np.sum(np.abs(coef) > 1e-9))
    out = {"seed": seed, "n_basis": int(Phi.shape[1]), "nonzero_coefficients": nz,
           "max_abs_err_fit": err_fit, "max_abs_err_heldout_angles": err_chk,
           "n_basis_without_fourway": int(keep.sum()), "max_abs_err_heldout_without_fourway": err_no4,
           "max_abs_score_diff_on_test_set": err_score,
           "metrics_twin": score_metrics(yte, s_twin)}
    json.dump(out, open(os.path.join(RESULTS, f"certificate_s{seed}.json"), "w"), indent=2)
    return out


def summary(n_boot=2000):
    from evaluation import paired_bootstrap
    names = test_attack_names()
    rf_files = {s: sorted(glob.glob(os.path.join(os.path.dirname(RESULTS),
                                                  f"nslkdd_random_forest_q8_l3_angle_s{s}_rf_*.json")))
                for s in SEEDS}
    rows, per_type = {}, {}
    y = None
    for kind in MODELS + tuple(RF_VARIANTS):
        for s in SEEDS:
            p = os.path.join(RESULTS, f"{kind}_s{s}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            y = d["y_test"] if y is None else y
            assert np.array_equal(y, d["y_test"])
            rows[(kind, s)] = d["score"]
    assert np.array_equal((names != "normal").astype(int), y), "attack names misaligned with labels"
    for s, fs in rf_files.items():
        if fs:
            rows[("rf", s)] = np.asarray(json.load(open(fs[0], encoding="utf-8"))["predictions"]["y_score"], float)
    from data import load_meta
    _, ym, _, _, meta = load_meta("nslkdd", 4, "pca", False, "minmax", 42)
    seen = set(np.array(meta["class_names"])[np.unique(ym)])
    novel = np.array([n not in seen for n in names])

    def tpr_types(score):
        """TPR@1%FPR on all attacks and on each attack group against all benign rows, with the
        ROC interpolation of evaluation.tpr_at_fpr (the benign rows fix the FPR axis, so the
        operating point is the same for every group, and ties are handled as in the main metric)."""
        from evaluation import tpr_at_fpr
        b = y == 0
        grp = lambda m: float(tpr_at_fpr(y[b | m], score[b | m], 0.01))
        return {"all": float(tpr_at_fpr(y, score, 0.01)),
                "seen_types": grp((y == 1) & ~novel), "novel_types": grp((y == 1) & novel)}

    out = {"n_test": int(len(y)), "n_attack_seen": int(((y == 1) & ~novel).sum()),
           "n_attack_novel": int(((y == 1) & novel).sum()), "models": {}, "paired": {}}
    for (kind, s), sc in rows.items():
        out["models"][f"{kind}_s{s}"] = {"metrics": score_metrics(y, sc), "tpr1_by_type": tpr_types(sc)}
    for s in SEEDS:
        if ("hybrid", s) not in rows:
            continue
        for other in ("fourier", "mlp", "linear", "rf") + tuple(RF_VARIANTS):
            if (other, s) not in rows:
                continue
            for mt in ("roc_auc", "tpr_at_1pct_fpr"):
                b = paired_bootstrap(y, rows[(other, s)], rows[("hybrid", s)], mt, n_boot=n_boot)
                out["paired"][f"hybrid_minus_{other}_s{s}_{mt}"] = {
                    "diff": float(b.observed_diff), "ci95": [float(b.ci95_low), float(b.ci95_high)],
                    "p": float(b.p_value)}
    agg = {}
    for kind in MODELS + ("rf",) + tuple(RF_VARIANTS):
        ms = [out["models"][f"{kind}_s{s}"] for s in SEEDS if f"{kind}_s{s}" in out["models"]]
        if ms:
            agg[kind] = {"n_seeds": len(ms)}
            for mt in ("roc_auc", "auprc", "tpr_at_1pct_fpr", "tpr_at_0.1pct_fpr"):
                v = [m["metrics"][mt] for m in ms]
                agg[kind][mt] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
            for part in ("seen_types", "novel_types"):
                v = [m["tpr1_by_type"][part] for m in ms]
                agg[kind][f"tpr1_{part}"] = {"mean": float(np.mean(v)),
                                             "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    out["aggregate"] = agg
    json.dump(out, open(os.path.join(RESULTS, "summary.json"), "w"), indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "rf", "certificate", "summary"])
    ap.add_argument("--model", choices=MODELS + tuple(RF_VARIANTS), default="fourier")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if a.mode == "train":
        out = train(a.model, a.seed)
    elif a.mode == "rf":
        out = rf_control(a.model, a.seed)
    elif a.mode == "certificate":
        out = certificate(a.seed)
    else:
        out = summary()
    print(json.dumps(out if a.mode != "summary" else out["aggregate"], indent=2))
