"""Quantum-attribution audit (task #9): is the 'quantum' advantage real?

This is the headline diagnostic of the paper's "honest broker" angle and the
direct answer to Bellante et al. (Computers & Security, 2025), who argue that
reported quantum-ML advantages on tabular security data are largely explained by
the *classical* preprocessing (dimensionality reduction, scaling) and by the
implicit regularization of small models -- not by quantum effects.

We quantify "how much is genuinely quantum" by pitting each quantum model against
*matched classical controls* and reporting the deltas with significance:

  1. matched-capacity MLP   -- a classical MLP whose trainable-parameter count is
                               matched to the HybridQNN (classical pre/post +
                               quantum weights), not merely shape-matched. If a
                               same-capacity classical net matches the QNN, the
                               "quantum" gain is a capacity/regularization effect.
  2. RBF-SVM                -- a strong classical kernel machine (reuses the tuned
                               baseline from baselines.py).
  3. random-feature kernel  -- sklearn RBFSampler / Nystroem feature map + linear
                               SVM: a CLASSICAL analogue of the quantum kernel
                               (QSVM). If random Fourier features match the
                               fidelity/projected quantum kernel, the kernel is not
                               doing anything classically inimitable.
  4. regularization sweep   -- vary classical regularization (MLP L2/dropout, RF
                               depth) to test whether matched-regularization
                               classical models close any remaining quantum gap.

Outputs a per-dataset table of (quantum metric) - (best matched control) deltas
with a paired test across seeds, plus a findings template at
paper/attribution_audit.md.

Reuses (does not reimplement): baselines._make_estimator / fit_tuned_sklearn for
RF/SVM, evaluation.compute_metrics + probability_metrics for the metric set, and
quantum_model for the quantum models. Verify in the qml env.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(HERE, "..", "paper")


# --------------------------------------------------------------------------- #
# Parameter counting & capacity matching
# --------------------------------------------------------------------------- #
def count_trainable_params(model) -> int:
    """Total trainable parameters of a torch model (0 for non-torch estimators)."""
    params = getattr(model, "parameters", None)
    if params is None:
        return 0
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def hidden_dim_for_param_budget(in_dim: int, budget: int, n_classes: int = 2,
                                n_hidden_layers: int = 2) -> int:
    """Smallest hidden width h s.t. a [in_dim]->(h)*L->[n_classes] MLP (with bias,
    Tanh) has >= `budget` trainable params. Used to capacity-match a classical MLP
    to a HybridQNN.

    Param count of such an MLP:
        first   : in_dim*h + h
        middle  : (L-1) * (h*h + h)
        out     : h*n_classes + n_classes
    Monotonic in h, so we just scan upward.
    """
    def nparams(h):
        first = in_dim * h + h
        middle = (n_hidden_layers - 1) * (h * h + h)
        out = h * n_classes + n_classes
        return first + middle + out

    h = 1
    while nparams(h) < budget:
        h += 1
        if h > 100000:  # safety
            break
    return h


def matched_mlp_for_qnn(qnn, in_dim: int, n_classes: int = 2,
                        n_hidden_layers: int = 2):
    """Build a classical MLP whose param count >= the given HybridQNN's.

    Returns (mlp_module, info_dict). The MLP reuses quantum_model.ClassicalMLP's
    architecture family (Linear+Tanh stack) but its hidden width is enlarged to
    reach capacity parity -- the like-for-like *capacity* control.
    """
    from quantum_model import ClassicalMLP

    budget = count_trainable_params(qnn)
    h = hidden_dim_for_param_budget(in_dim, budget, n_classes, n_hidden_layers)
    # ClassicalMLP is a fixed 2-hidden-layer Tanh net; build with the matched width.
    mlp = ClassicalMLP(in_dim, hidden=h, n_classes=n_classes)
    info = {
        "qnn_params": budget,
        "matched_mlp_params": count_trainable_params(mlp),
        "matched_hidden": h,
        "in_dim": in_dim,
    }
    return mlp, info


# --------------------------------------------------------------------------- #
# Random-feature kernel control (classical analogue of the quantum kernel)
# --------------------------------------------------------------------------- #
class RandomFeatureKernelSVM:
    """Classical random-feature kernel machine: a control for the QSVM.

    Approximates an RBF kernel with an explicit randomized feature map
    (sklearn RBFSampler = random Fourier features, or Nystroem = data-driven
    landmarks) and fits a linear SVM on the resulting features. This is the
    classical analogue of the quantum kernel: if it matches the QSVM, the quantum
    feature map provides no advantage a cheap classical map cannot.

    sklearn-style fit/predict/decision_function so it drops into the same eval path
    as QuantumKernelSVM and the runner's sklearn branch.
    """

    def __init__(self, method: str = "rbf_sampler", n_components: int = 256,
                 gamma: float = 1.0, C: float = 1.0, seed: int = 42):
        from sklearn.kernel_approximation import RBFSampler, Nystroem
        from sklearn.svm import LinearSVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if method == "rbf_sampler":
            fmap = RBFSampler(gamma=gamma, n_components=n_components,
                              random_state=seed)
        elif method == "nystroem":
            fmap = Nystroem(kernel="rbf", gamma=gamma, n_components=n_components,
                            random_state=seed)
        else:
            raise ValueError("method must be 'rbf_sampler' or 'nystroem'")
        self.method = method
        # Standardize the random features then a linear SVM (fast, convex).
        self.pipe = make_pipeline(
            fmap, StandardScaler(with_mean=True),
            LinearSVC(C=C, random_state=seed, dual="auto", max_iter=5000),
        )

    def fit(self, X, y):
        self.pipe.fit(np.asarray(X, dtype=np.float64), y)
        return self

    def predict(self, X):
        return self.pipe.predict(np.asarray(X, dtype=np.float64))

    def decision_function(self, X):
        return self.pipe.decision_function(np.asarray(X, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Classical regularization sweep (matched-regularization control)
# --------------------------------------------------------------------------- #
def regularization_grid():
    """Control grids that vary ONLY regularization strength, per model family.

    Tests Bellante's claim that matched regularization (not quantum effects)
    explains gaps. Returns {family: [param_dicts]}.
    """
    return {
        # RF: capacity via tree depth (shallower = more regularized).
        "random_forest": [{"max_depth": d} for d in (3, 5, 8, 12, None)],
        # SVM: inverse-regularization C.
        "svm_rbf": [{"C": c} for c in (0.1, 1.0, 10.0, 100.0)],
        # Matched MLP: L2 (weight decay) sweep -- consumed by the torch trainer.
        "matched_mlp": [{"weight_decay": wd} for wd in (0.0, 1e-4, 1e-3, 1e-2)],
    }


# --------------------------------------------------------------------------- #
# Delta computation & paired significance
# --------------------------------------------------------------------------- #
@dataclass
class AuditRow:
    """One quantum-vs-control comparison for a single metric on a dataset."""
    dataset: str
    metric: str
    quantum_model: str
    quantum_mean: float
    control_model: str
    control_mean: float
    delta: float                 # quantum - control (positive => quantum better*)
    p_value: float = float("nan")
    n_seeds: int = 0
    note: str = ""               # *for ECE/FPR lower is better; see direction


# Metrics where LOWER is better (delta sign should be flipped for "quantum better").
LOWER_IS_BETTER = ("ece", "brier", "false_positive_rate")


def paired_delta(quantum_scores, control_scores, metric: str):
    """Return (delta, p_value) for paired-by-seed quantum vs control scores.

    delta = mean(quantum) - mean(control). Significance is delegated to
    evaluation.paired_t_test (REUSE, not reimplement -- task #5's machinery, which
    gives the paired t-test + Wilcoxon + Cohen's d). We surface the t-test p here;
    callers wanting the full result can call evaluation.paired_t_test directly. For
    LOWER_IS_BETTER metrics the raw delta is returned as-is (interpretation handled
    by the reporter). NaN p when <2 paired seeds.
    """
    q = np.asarray(quantum_scores, dtype=float)
    c = np.asarray(control_scores, dtype=float)
    n = min(len(q), len(c))
    q, c = q[:n], c[:n]
    delta = float(np.mean(q) - np.mean(c)) if n else float("nan")
    p = float("nan")
    if n >= 2:
        try:
            from evaluation import paired_t_test
            # control = a, quantum = b -> mean_diff = quantum - control.
            res = paired_t_test(c.tolist(), q.tolist(), metric=metric)
            p = float(res.t_p_value)
        except Exception:
            from scipy.stats import ttest_rel
            _, p = ttest_rel(q, c)
            p = float(p)
    return delta, p


def quantum_better(metric: str, delta: float) -> bool:
    """Is the quantum model genuinely better given the metric's direction?"""
    if metric in LOWER_IS_BETTER:
        return delta < 0
    return delta > 0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def audit_table(rows: list[AuditRow]) -> str:
    """Render audit rows as a GitHub-flavoured markdown table."""
    hdr = ("| dataset | metric | quantum | q-mean | control | c-mean | "
           "delta (q-c) | quantum better? | p | comparison |\n"
           "|---|---|---|---:|---|---:|---:|:--:|---:|---|")
    lines = [hdr]
    for r in rows:
        better = "yes" if quantum_better(r.metric, r.delta) else "no"
        p = "n/a" if np.isnan(r.p_value) else f"{r.p_value:.3f}"
        lines.append(
            f"| {r.dataset} | {r.metric} | {r.quantum_model} | {r.quantum_mean:.4f} "
            f"| {r.control_model} | {r.control_mean:.4f} | {r.delta:+.4f} | {better} "
            f"| {p} | {r.note} |")
    return "\n".join(lines)


def write_report(rows: list[AuditRow], out_path: Optional[str] = None,
                 extra_notes: str = "") -> str:
    """Write the attribution-audit findings markdown. Returns the path."""
    out_path = out_path or os.path.join(PAPER_DIR, "attribution_audit.md")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    body = f"""# Quantum-Attribution Audit

*How much of the observed quantum performance is genuinely quantum, versus
classical preprocessing / capacity / regularization?* This directly addresses
Bellante et al. (Computers & Security, 2025).

## Method
Each quantum model (HybridQNN variants, QSVM) is compared against **matched
classical controls** on identical data splits and seeds:

1. **Capacity-matched MLP** -- classical MLP with trainable-parameter count
   matched to the HybridQNN (classical + quantum weights), not merely
   shape-matched.
2. **RBF-SVM** -- tuned classical kernel machine.
3. **Random-feature kernel** -- RBFSampler / Nystroem feature map + linear SVM,
   the classical analogue of the quantum kernel.
4. **Regularization sweep** -- classical regularization varied (MLP L2/dropout,
   RF depth) to test whether matched regularization closes any gap.

For each metric we report delta = (quantum) - (best matched control), with a
paired test across seeds. "Quantum better?" accounts for metric direction
(lower is better for {", ".join(LOWER_IS_BETTER)}).

## Results
{audit_table(rows) if rows else "_(populate by running the audit harness)_"}

## Interpretation
- A **small or negative** delta against the capacity-matched MLP indicates the
  advantage is a capacity/regularization effect, not a quantum one.
- A random-feature kernel that **matches** the QSVM indicates the quantum kernel
  is classically simulable in effect.
- Genuine quantum advantage requires a **positive, significant** delta against the
  *best* matched control across metrics.

{extra_notes}
"""
    with open(out_path, "w") as f:
        f.write(body)
    return out_path


# --------------------------------------------------------------------------- #
# End-to-end audit driver (reads harness result JSONs)
# --------------------------------------------------------------------------- #
# model_type -> role for the audit. Quantum models are the "treatment"; the rest
# are matched controls they are compared against.
QUANTUM_TYPES = ("hybrid_qnn", "qsvm")
CONTROL_TYPES = ("matched_mlp", "rf_kernel", "svm_rbf", "random_forest",
                 "classical_mlp", "xgboost")
AUDIT_METRICS = ("roc_auc", "auprc", "tpr_at_0.1pct_fpr", "tpr_at_1pct_fpr",
                 "f1", "ece")


def _records_by_key(records):
    """Group harness records -> {(dataset, model_name, model_type): {metric: [vals]}}.

    Each result JSON is one (config, metrics) run; multiple seeds of the same
    config accumulate into the metric lists.
    """
    out = {}
    for r in records:
        cfg = r.get("config", {})
        ds = cfg.get("dataset", "unknown")
        name = cfg.get("name") or cfg.get("model_type", "model")
        mtype = cfg.get("model_type", "")
        key = (ds, name, mtype)
        bucket = out.setdefault(key, {})
        for m, v in (r.get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                bucket.setdefault(m, []).append(float(v))
    return out


def run_attribution_audit(records, out_path=None):
    """Build the audit: per (dataset, quantum-model, metric), compare against the
    BEST matched control and emit AuditRows + the markdown report.

    `records` = list of harness result dicts (load via noise_plots.load_noise_records
    or json.load over results/*.json). Returns (rows, report_path).
    """
    grouped = _records_by_key(records)
    # index controls per dataset
    controls = {}  # dataset -> list[(name, mtype, {metric:[vals]})]
    quantum = {}   # dataset -> list[(name, mtype, {metric:[vals]})]
    for (ds, name, mtype), mdict in grouped.items():
        target = quantum if mtype in QUANTUM_TYPES else (
            controls if mtype in CONTROL_TYPES else None)
        if target is None:
            continue
        target.setdefault(ds, []).append((name, mtype, mdict))

    rows: list[AuditRow] = []
    for ds, qmodels in quantum.items():
        ds_controls = controls.get(ds, [])
        for qname, qtype, qmetrics in qmodels:
            for metric in AUDIT_METRICS:
                qvals = qmetrics.get(metric)
                if not qvals:
                    continue
                qmean = float(np.mean(qvals))
                # pick the BEST control on this metric (direction-aware).
                best = None  # (cname, cmean, cvals)
                for cname, ctype, cmetrics in ds_controls:
                    cvals = cmetrics.get(metric)
                    if not cvals:
                        continue
                    cmean = float(np.mean(cvals))
                    if best is None:
                        best = (cname, cmean, cvals)
                    else:
                        better = (cmean < best[1]) if metric in LOWER_IS_BETTER \
                            else (cmean > best[1])
                        if better:
                            best = (cname, cmean, cvals)
                if best is None:
                    continue
                cname, cmean, cvals = best
                delta, p = paired_delta(qvals, cvals, metric)
                rows.append(AuditRow(
                    dataset=ds, metric=metric, quantum_model=qname,
                    quantum_mean=qmean, control_model=cname, control_mean=cmean,
                    delta=delta, p_value=p, n_seeds=min(len(qvals), len(cvals)),
                    note="best-control",
                ))
                # Headline Bellante rebuttal: the QSVM's NATURAL classical analogue
                # is the random-feature kernel. Always surface that direct
                # comparison even when another control scores higher.
                if qtype == "qsvm" and cname != "rf_kernel":
                    for rfname, rftype, rfm in ds_controls:
                        if rftype != "rf_kernel":
                            continue
                        rfvals = rfm.get(metric)
                        if not rfvals:
                            continue
                        rfmean = float(np.mean(rfvals))
                        d2, p2 = paired_delta(qvals, rfvals, metric)
                        rows.append(AuditRow(
                            dataset=ds, metric=metric, quantum_model=qname,
                            quantum_mean=qmean, control_model=rfname,
                            control_mean=rfmean, delta=d2, p_value=p2,
                            n_seeds=min(len(qvals), len(rfvals)),
                            note="qkernel-vs-rf",
                        ))
    notes = ("\n> Auto-generated from results/*.json by attribution.run_attribution_audit. "
             "'best control' is chosen per-metric (direction-aware). Brier/ECE for the "
             "rf_kernel control may be NaN (LinearSVC has no calibrated probabilities) "
             "and are excluded from those rows.")
    path = write_report(rows, out_path=out_path, extra_notes=notes)
    return rows, path


def _cli_from_results(results_dir: str, out_path: Optional[str] = None):
    """Build the audit report from existing harness result JSONs."""
    import glob as _glob
    recs = []
    for path in sorted(_glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                recs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    rows, p = run_attribution_audit(recs, out_path=out_path)
    print(f"[attribution] {len(recs)} records -> {len(rows)} audit rows -> {p}")
    for r in rows:
        better = "Q" if quantum_better(r.metric, r.delta) else "C"
        print(f"  {r.dataset:<8} {r.metric:<18} {r.quantum_model:<20} "
              f"vs {r.control_model:<14} delta={r.delta:+.4f} [{better}] "
              f"p={r.p_value if not np.isnan(r.p_value) else float('nan'):.3f}")
    return rows, p


def _cli_from_config(config_path: str, results_dir: str, out_path: Optional[str] = None):
    """Run the audit's models through the harness, then build the report.

    For each ExperimentConfig in `config_path` (the attribution-audit sweep), run
    it via runner.run_config UNLESS a result JSON for that exact slug already
    exists in `results_dir` (idempotent: the overnight orchestrator may have
    produced some already). Then read ALL results in `results_dir` and build the
    quantum-vs-matched-control report.

    This is the entry point the overnight orchestrator calls:
        python attribution.py --config configs/attribution_audit.yaml --results-dir RES
    """
    import glob as _glob
    from config import load_configs
    from runner import run_config

    cfgs = load_configs(config_path)
    existing = {os.path.basename(p).rsplit("_", 2)[0]  # strip _<date>_<time>.json
                for p in _glob.glob(os.path.join(results_dir, "*.json"))}
    ran, skipped = 0, 0
    for cfg in cfgs:
        if cfg.slug() in existing:
            skipped += 1
            continue
        try:
            run_config(cfg, results_dir=results_dir)
            ran += 1
        except Exception as e:  # one bad run shouldn't sink the whole audit
            print(f"[attribution] WARN run {cfg.name} ({cfg.slug()}) failed: {e}")
    print(f"[attribution] ran {ran} audit configs, skipped {skipped} already-present")

    recs = []
    for path in sorted(_glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as f:
                recs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    rows, p = run_attribution_audit(recs, out_path=out_path)
    print(f"[attribution] {len(recs)} records -> {len(rows)} audit rows -> {p}")
    return rows, p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quantum-attribution audit (task #9)")
    ap.add_argument("--config", metavar="YAML",
                    help="run the audit sweep configs through the harness, then "
                         "build the report (used by the overnight orchestrator)")
    ap.add_argument("--from-results", metavar="DIR",
                    help="build the audit report from existing result JSONs in DIR")
    ap.add_argument("--results-dir", default=None,
                    help="results directory (with --config: where to read/write JSONs)")
    ap.add_argument("--out", default=None, help="output markdown path")
    args = ap.parse_args()
    if args.config:
        _rd = args.results_dir or os.path.join(HERE, "..", "results")
        _cli_from_config(args.config, _rd, out_path=args.out)
        raise SystemExit(0)
    if args.from_results:
        _cli_from_results(args.from_results, out_path=args.out)
        raise SystemExit(0)

    # Default (no args): self-test on synthetic data (no dataset / GPU needed).
    import torch  # noqa
    from quantum_model import build_model

    in_dim, q = 6, 6
    qnn = build_model("hybrid", in_dim, n_qubits=q, n_layers=2, encoding="angle")
    mlp, info = matched_mlp_for_qnn(qnn, in_dim)
    print("[param-match]", info)
    assert info["matched_mlp_params"] >= info["qnn_params"], "MLP must reach budget"

    # Random-feature kernel control on toy separable data.
    rng = np.random.RandomState(0)
    X = rng.rand(120, in_dim) * np.pi
    y = (X[:, 0] > np.pi / 2).astype(int)
    for method in ("rbf_sampler", "nystroem"):
        rf = RandomFeatureKernelSVM(method=method, n_components=128, gamma=1.0)
        rf.fit(X[:80], y[:80])
        acc = float((rf.predict(X[80:]) == y[80:]).mean())
        df = rf.decision_function(X[80:])
        print(f"[rf-kernel {method:<11}] acc={acc:.3f} df_shape={np.asarray(df).shape}")

    # Delta + report smoke test.
    rows = [
        AuditRow("nslkdd", "tpr_at_1pct_fpr", "Hybrid-Angle-SE", 0.91,
                 "matched_mlp", 0.90, *paired_delta([0.91, 0.92], [0.90, 0.89],
                                                    "tpr_at_1pct_fpr"), n_seeds=2),
        AuditRow("nslkdd", "ece", "Hybrid-Angle-SE", 0.08,
                 "matched_mlp", 0.06, *paired_delta([0.08, 0.07], [0.06, 0.05],
                                                    "ece"), n_seeds=2),
    ]
    p = write_report(rows, out_path=os.path.join(PAPER_DIR, "attribution_audit.md"))
    print("[report] wrote", p)

    # Driver self-test: synthetic harness records -> run_attribution_audit.
    def _rec(ds, name, mtype, seed, **mets):
        return {"config": {"dataset": ds, "name": name, "model_type": mtype,
                           "seed": seed}, "metrics": mets}
    recs = []
    for s in (42, 43):
        recs += [
            _rec("nslkdd", "Hybrid-Angle-SE", "hybrid_qnn", s,
                 roc_auc=0.95 + 0.001 * s % 2, auprc=0.94, f1=0.91, ece=0.07,
                 tpr_at_1pct_fpr=0.90, **{"tpr_at_0.1pct_fpr": 0.72}),
            _rec("nslkdd", "matched_mlp", "matched_mlp", s,
                 roc_auc=0.945, auprc=0.93, f1=0.905, ece=0.05,
                 tpr_at_1pct_fpr=0.905, **{"tpr_at_0.1pct_fpr": 0.74}),
            _rec("nslkdd", "QSVM-IQP", "qsvm", s,
                 roc_auc=0.93, auprc=0.92, f1=0.89, tpr_at_1pct_fpr=0.86,
                 **{"tpr_at_0.1pct_fpr": 0.66}),
            _rec("nslkdd", "rf_kernel", "rf_kernel", s,
                 roc_auc=0.928, auprc=0.918, f1=0.885, tpr_at_1pct_fpr=0.858,
                 **{"tpr_at_0.1pct_fpr": 0.655}),
        ]
    arows, ap_ = run_attribution_audit(recs, out_path=os.path.join(
        PAPER_DIR, "attribution_audit.md"))
    print(f"[driver] {len(arows)} audit rows from {len(recs)} synthetic records -> {ap_}")
    assert arows, "driver produced no rows"
    print("audit self-test complete")
