"""Evaluation protocol + statistical significance testing (task #5).

This module centralises the *methodology* that makes the quantum-vs-classical
comparison defensible for a Q1 review:

  1. metrics(...)             -- the agreed metric set (Acc, P, R/DR, F1, FPR, AUC).
  2. aggregate_seeds(...)     -- mean +/- std (and 95% CI) over >=5 seeds.
  3. mcnemar_test(...)        -- paired test on a SINGLE held-out test set:
                                 the right test when both models are evaluated on
                                 the same fixed NSL-KDD test set.
  4. paired_t_test(...)       -- paired t-test / Wilcoxon over per-seed scores:
                                 the right test for "mean F1 over seeds differs".
  5. compare_to_best_classical(...) -- convenience: pick the best classical model
                                 and run both tests vs the quantum model.

Choosing the test:
  * McNemar compares two classifiers on the SAME instances (their disagreement
    pattern), so it needs the per-instance predictions of both models on one test
    set. Use it for a single representative seed (or the majority-vote ensemble of
    seeds) -- it answers "do these two models make significantly different errors".
  * Paired t-test / Wilcoxon compares the DISTRIBUTION of a metric across seeds,
    answering "is the quantum model's mean F1 over seeds significantly higher".
  We report BOTH, since reviewers in this area ask for each.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             average_precision_score, brier_score_loss, roc_curve)


# Threshold-based + ranking metrics (always available with predictions/scores).
METRIC_KEYS = ("accuracy", "precision", "recall", "f1",
               "detection_rate", "false_positive_rate", "roc_auc",
               # calibration / imbalance-aware (task #8); need probabilities:
               "auprc", "tpr_at_0.1pct_fpr", "tpr_at_1pct_fpr",
               "brier", "ece")


def metrics(y_true, y_pred, y_score=None) -> dict:
    """Compute the full IDS metric set.

    Threshold metrics (Acc/P/R/F1/DR/FPR) come from `y_pred`. When `y_score`
    (predicted P(attack), from predict_proba / softmax) is supplied we also add
    the ranking + calibration/imbalance-aware metrics from `probability_metrics`:
    ROC-AUC, AUPRC, TPR at fixed low FPR operating points, Brier score, and ECE.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "detection_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }
    if y_score is not None:
        out.update(probability_metrics(y_true, y_score))
    return out


# --------------------------------------------------------------------------- #
# Calibration & imbalance-aware probability metrics (task #8)
# --------------------------------------------------------------------------- #
def tpr_at_fpr(y_true, y_score, target_fpr: float) -> float:
    """True-positive rate (detection rate) at a fixed maximum false-positive rate.

    Operationally the most important IDS metric: at a tolerable alarm budget
    (e.g. 0.1% or 1% of benign flows raising an alert) what fraction of attacks
    do we catch? We take the highest TPR achievable without exceeding
    `target_fpr` (interpolating the ROC curve at the operating point).
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # ROC fpr is sorted ascending; interpolate TPR at exactly target_fpr.
    if target_fpr <= fpr[0]:
        return float(tpr[0])
    if target_fpr >= fpr[-1]:
        return float(tpr[-1])
    return float(np.interp(target_fpr, fpr, tpr))


def expected_calibration_error(y_true, y_score, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE): weighted gap between confidence & accuracy.

    Bins predictions into `n_bins` equal-width confidence bins on [0,1]; within
    each bin compares mean predicted P(attack) to the empirical attack rate, and
    averages |gap| weighted by bin population. Lower is better-calibrated.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=False)
    n = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        m = int(mask.sum())
        if m == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y_true[mask].mean())   # empirical P(positive) in the bin
        ece += (m / n) * abs(conf - acc)
    return float(ece)


def probability_metrics(y_true, y_score, *, ece_bins: int = 15) -> dict:
    """Ranking + calibration/imbalance-aware metrics from predicted P(attack).

    Returns ROC-AUC, AUPRC (average precision), TPR@0.1% FPR, TPR@1% FPR, Brier
    score, and ECE. Robust to degenerate (single-class) inputs (-> NaN where a
    metric is undefined). These match/exceed the Meta-Quantum-Ensemble metric set
    and are the operating-point + calibration evidence the paper needs.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    out: dict[str, float] = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["auprc"] = float("nan")
    out["tpr_at_0.1pct_fpr"] = tpr_at_fpr(y_true, y_score, 0.001)
    out["tpr_at_1pct_fpr"] = tpr_at_fpr(y_true, y_score, 0.01)
    try:
        # Brier expects probabilities in [0,1]; clip defensively.
        out["brier"] = float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))
    except ValueError:
        out["brier"] = float("nan")
    out["ece"] = expected_calibration_error(y_true, y_score, n_bins=ece_bins)
    return out


# --------------------------------------------------------------------------- #
# Multi-seed aggregation: mean +/- std (+ 95% CI)
# --------------------------------------------------------------------------- #
@dataclass
class SeedAggregate:
    """Aggregated stats for one metric across seeds."""
    mean: float
    std: float
    ci95: float          # half-width of the 95% CI (t-based)
    n: int
    values: list[float]

    def as_str(self, prec: int = 4) -> str:
        return f"{self.mean:.{prec}f} ± {self.std:.{prec}f}"


def aggregate_seeds(per_seed_metrics: Sequence[dict]) -> dict[str, SeedAggregate]:
    """Aggregate a list of metric dicts (one per seed) into mean/std/CI per metric.

    Uses the sample std (ddof=1) and a t-distribution 95% CI half-width, which is
    the correct small-sample interval for ~5 seeds.
    """
    n = len(per_seed_metrics)
    keys = set().union(*[m.keys() for m in per_seed_metrics]) if n else set()
    out: dict[str, SeedAggregate] = {}
    for k in keys:
        vals = np.array([float(m[k]) for m in per_seed_metrics if k in m
                         and not (isinstance(m[k], float) and np.isnan(m[k]))],
                        dtype=float)
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        if vals.size > 1:
            tcrit = float(stats.t.ppf(0.975, df=vals.size - 1))
            ci95 = tcrit * std / np.sqrt(vals.size)
        else:
            ci95 = 0.0
        out[k] = SeedAggregate(mean=mean, std=std, ci95=float(ci95),
                               n=int(vals.size), values=vals.tolist())
    return out


def aggregate_to_json(per_seed_metrics: Sequence[dict]) -> dict:
    """JSON-friendly version of aggregate_seeds (plain dicts, no dataclasses)."""
    return {k: asdict(v) for k, v in aggregate_seeds(per_seed_metrics).items()}


# --------------------------------------------------------------------------- #
# McNemar's test (two classifiers, one fixed test set)
# --------------------------------------------------------------------------- #
@dataclass
class McNemarResult:
    n01: int            # model A wrong, model B right
    n10: int            # model A right, model B wrong
    statistic: float
    p_value: float
    corrected: bool     # continuity / exact correction applied
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


def mcnemar_test(y_true, pred_a, pred_b, exact_threshold: int = 25) -> McNemarResult:
    """McNemar's test on the disagreements between two models on the same test set.

    Builds the 2x2 contingency of (A correct?, B correct?) and tests the
    off-diagonal (the cases where exactly one model is right). Uses the exact
    binomial test when the discordant count is small (<= exact_threshold), else
    the chi-square statistic with continuity correction.

    `pred_a` is conventionally the classical (best baseline) and `pred_b` the
    quantum model, but the test is symmetric in its p-value.
    """
    y_true = np.asarray(y_true)
    a_correct = np.asarray(pred_a) == y_true
    b_correct = np.asarray(pred_b) == y_true
    n01 = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    n10 = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    n = n01 + n10
    if n == 0:
        return McNemarResult(n01, n10, 0.0, 1.0, corrected=False,
                             method="degenerate (no disagreements)")
    if n <= exact_threshold:
        # Exact binomial: under H0 each discordant pair is 50/50.
        k = min(n01, n10)
        p = float(min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5)))
        stat = float(k)
        return McNemarResult(n01, n10, stat, p, corrected=True, method="exact binomial")
    # Chi-square with continuity correction.
    stat = (abs(n01 - n10) - 1.0) ** 2 / n
    p = float(stats.chi2.sf(stat, df=1))
    return McNemarResult(n01, n10, float(stat), p, corrected=True,
                         method="chi-square (continuity corrected)")


# --------------------------------------------------------------------------- #
# Paired test over per-seed scores (distribution of a metric across seeds)
# --------------------------------------------------------------------------- #
@dataclass
class PairedTestResult:
    metric: str
    mean_a: float
    mean_b: float
    mean_diff: float    # b - a  (quantum - classical, if b is quantum)
    t_statistic: float
    t_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    n_seeds: int
    cohens_d: float     # paired effect size

    def to_dict(self) -> dict:
        return asdict(self)


def paired_t_test(scores_a: Sequence[float], scores_b: Sequence[float],
                  metric: str = "f1") -> PairedTestResult:
    """Paired t-test AND Wilcoxon signed-rank over per-seed scores (b - a).

    Both must be aligned by seed (scores_a[i], scores_b[i] are the same seed). We
    report the parametric paired t-test and the non-parametric Wilcoxon (robust to
    non-normality with few seeds) plus a paired Cohen's d effect size, so the
    reader can judge magnitude, not just significance.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must align by seed")
    diff = b - a
    n = a.size
    t_stat, t_p = stats.ttest_rel(b, a)
    # Wilcoxon needs >0 non-zero diffs; guard the degenerate all-equal case.
    if np.allclose(diff, 0):
        w_stat, w_p = 0.0, 1.0
    else:
        try:
            w_stat, w_p = stats.wilcoxon(b, a)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    cohens_d = float(diff.mean() / sd) if sd else 0.0
    return PairedTestResult(
        metric=metric,
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        mean_diff=float(diff.mean()),
        t_statistic=float(t_stat), t_p_value=float(t_p),
        wilcoxon_statistic=float(w_stat), wilcoxon_p_value=float(w_p),
        n_seeds=int(n), cohens_d=cohens_d,
    )


# --------------------------------------------------------------------------- #
# Paired bootstrap on a single test set (for AUPRC / TPR@FPR / AUC etc.)
# --------------------------------------------------------------------------- #
_SCORE_FNS = {
    "roc_auc": lambda yt, s: roc_auc_score(yt, s),
    "auprc": lambda yt, s: average_precision_score(yt, s),
    "tpr_at_0.1pct_fpr": lambda yt, s: tpr_at_fpr(yt, s, 0.001),
    "tpr_at_1pct_fpr": lambda yt, s: tpr_at_fpr(yt, s, 0.01),
    "brier": lambda yt, s: brier_score_loss(yt, np.clip(s, 0, 1)),
}


@dataclass
class BootstrapResult:
    metric: str
    score_a: float
    score_b: float
    observed_diff: float        # b - a on the full test set
    ci95_low: float
    ci95_high: float
    p_value: float              # two-sided, fraction of resamples crossing 0
    n_boot: int

    def to_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap(y_true, score_a, score_b, metric: str = "auprc", *,
                     n_boot: int = 2000, seed: int = 42) -> BootstrapResult:
    """Paired bootstrap test for a probability metric on ONE fixed test set.

    Resamples test instances with replacement (same indices applied to both
    models -> paired), recomputes metric(b) - metric(a) on each resample, and
    reports the 95% percentile CI and a two-sided p-value (the proportion of
    resampled differences on the opposite side of 0, doubled). This is the right
    significance test for AUPRC / TPR@FPR / AUC, where there is no per-seed list
    but a single held-out test set and two probability vectors.

    `score_a` = classical, `score_b` = quantum (diff = quantum - classical).
    """
    yt = np.asarray(y_true)
    sa = np.asarray(score_a, dtype=float)
    sb = np.asarray(score_b, dtype=float)
    fn = _SCORE_FNS.get(metric)
    if fn is None:
        raise ValueError(f"unsupported bootstrap metric {metric!r}; "
                         f"options: {sorted(_SCORE_FNS)}")
    obs_a, obs_b = float(fn(yt, sa)), float(fn(yt, sb))
    obs_diff = obs_b - obs_a
    rng = np.random.RandomState(seed)
    n = len(yt)
    diffs = np.empty(n_boot, dtype=float)
    valid = 0
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        ytr = yt[idx]
        if len(np.unique(ytr)) < 2:   # need both classes for these metrics
            continue
        diffs[valid] = fn(ytr, sb[idx]) - fn(ytr, sa[idx])
        valid += 1
    diffs = diffs[:valid]
    if valid == 0:
        return BootstrapResult(metric, obs_a, obs_b, obs_diff,
                               float("nan"), float("nan"), float("nan"), 0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p: proportion of resamples on the opposite side of 0, x2.
    prop = np.mean(diffs <= 0) if obs_diff > 0 else np.mean(diffs >= 0)
    p = float(min(1.0, 2.0 * prop))
    return BootstrapResult(metric, obs_a, obs_b, float(obs_diff),
                           float(lo), float(hi), p, int(valid))


# --------------------------------------------------------------------------- #
# Convenience: compare quantum to the best classical baseline
# --------------------------------------------------------------------------- #
def best_classical(per_model_agg: dict[str, dict], metric: str = "f1",
                   exclude: Sequence[str] = ()) -> str:
    """Return the name of the best classical model by mean `metric`.

    `per_model_agg` maps model_name -> {metric: SeedAggregate-or-dict}. `exclude`
    lists quantum model names to skip.
    """
    best, best_val = None, -np.inf
    for name, agg in per_model_agg.items():
        if name in exclude:
            continue
        m = agg.get(metric)
        val = m.mean if isinstance(m, SeedAggregate) else (m or {}).get("mean", -np.inf)
        if val is not None and val > best_val:
            best, best_val = name, val
    return best


def compare_models(quantum_name: str, classical_name: str, *,
                   per_seed: dict[str, list[dict]],
                   preds: dict[str, dict] | None = None,
                   scores: dict[str, dict] | None = None,
                   metric: str = "f1",
                   bootstrap_metrics: Sequence[str] = ("auprc", "tpr_at_1pct_fpr"),
                   ) -> dict:
    """Significance comparison of a quantum model vs a classical baseline.

    Runs (1) the paired t-test/Wilcoxon over per-seed `metric`, (2) McNemar on
    reference-seed hard predictions if `preds` given, and (3) a paired bootstrap
    on reference-seed probability scores for each metric in `bootstrap_metrics`
    if `scores` given.

    per_seed: model_name -> list of per-seed metric dicts (aligned by seed order).
    preds   : model_name -> {"y_true": [...], "y_pred": [...]} (reference seed).
    scores  : model_name -> {"y_true": [...], "y_score": [...]} (reference seed,
              P(attack)); used for the bootstrap on ranking/operating-point metrics.
    """
    out: dict = {"quantum": quantum_name, "classical": classical_name,
                 "metric": metric}
    a = [m[metric] for m in per_seed[classical_name]]
    b = [m[metric] for m in per_seed[quantum_name]]
    out["paired_test"] = paired_t_test(a, b, metric=metric).to_dict()
    if preds and quantum_name in preds and classical_name in preds:
        pc, pq = preds[classical_name], preds[quantum_name]
        out["mcnemar"] = mcnemar_test(pc["y_true"], pc["y_pred"],
                                      pq["y_pred"]).to_dict()
    if scores and quantum_name in scores and classical_name in scores:
        sc, sq = scores[classical_name], scores[quantum_name]
        out["bootstrap"] = {
            bm: paired_bootstrap(sc["y_true"], sc["y_score"], sq["y_score"],
                                 metric=bm).to_dict()
            for bm in bootstrap_metrics
        }
    return out


if __name__ == "__main__":
    # Self-test on synthetic data.
    rng = np.random.RandomState(0)
    yt = rng.randint(0, 2, 2000)
    # Probabilistic scores: quantum a bit better-ranked than classical.
    noise_c = rng.normal(0, 1.0, 2000)
    noise_q = rng.normal(0, 0.85, 2000)
    sc = 1 / (1 + np.exp(-(1.5 * yt - 0.75 + noise_c)))  # classical P(attack)
    sq = 1 / (1 + np.exp(-(1.8 * yt - 0.90 + noise_q)))  # quantum P(attack)
    pc = (sc > 0.5).astype(int)
    pq = (sq > 0.5).astype(int)
    print("metrics(classical):")
    for k, v in metrics(yt, pc, sc).items():
        print(f"   {k:<20} {v:.4f}")
    print("metrics(quantum):")
    for k, v in metrics(yt, pq, sq).items():
        print(f"   {k:<20} {v:.4f}")
    print("mcnemar:", mcnemar_test(yt, pc, pq).to_dict())
    print("bootstrap AUPRC:", paired_bootstrap(yt, sc, sq, "auprc",
                                                n_boot=500).to_dict())
    print("bootstrap TPR@1%FPR:", paired_bootstrap(yt, sc, sq, "tpr_at_1pct_fpr",
                                                    n_boot=500).to_dict())
    cseeds = [{"f1": 0.80 + 0.01 * i} for i in range(5)]
    qseeds = [{"f1": 0.83 + 0.012 * i + 0.002 * (i % 2)} for i in range(5)]
    print("agg classical f1:", aggregate_seeds(cseeds)["f1"].as_str())
    print("paired:", paired_t_test([m["f1"] for m in cseeds],
                                    [m["f1"] for m in qseeds]).to_dict())
