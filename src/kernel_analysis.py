"""Kernel-level analysis of the quantum models: fair surrogates, geometric
difference, concentration, and finite-shot robustness.

Engine: noiseless feature-map statevectors are simulated ONCE per sample with
PennyLane parameter broadcasting, reusing the exact feature maps from
quantum_model.py. Every kernel is then built from those statevectors:

  fidelity  K(x,x') = |<phi(x')|phi(x)>|^2          = |Psi_A Psi_B^H|^2
  projected K(x,x') = exp(-gamma ||r(x) - r(x')||^2) where r(x) stacks the
                      single-qubit <X>,<Y>,<Z> (Huang et al. 2021), taken from
                      each qubit's reduced density matrix.

This is O(N) circuit simulations instead of O(N^2), and the engine is checked
against QuantumKernelSVM before any result is used (see `validate`).
"""
import argparse
import json
import os
import time

import numpy as np
import pennylane as qml

from quantum_model import zz_feature_map, QuantumKernelSVM
from evaluation import tpr_at_fpr, paired_bootstrap
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "kernel_analysis")


# --------------------------------------------------------------------------- #
# statevector engine
# --------------------------------------------------------------------------- #
def statevectors(X, n_qubits, encoding="iqp", n_repeats=2, rotation="Y",
                 batch=1000):
    """Noiseless feature-map states, shape (N, 2**n). Wire 0 = most significant."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        if encoding == "iqp":
            zz_feature_map(x, n_qubits, n_repeats=n_repeats)
        elif encoding == "angle":
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation=rotation)
        else:
            raise ValueError(encoding)
        return qml.state()

    X = np.asarray(X, dtype=np.float64)
    out = np.empty((len(X), 2 ** n_qubits), dtype=np.complex128)
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        out[i:i + len(xb)] = np.asarray(circuit(xb))
    return out


def pauli_features(states, n_qubits):
    """Per-qubit (<X>,<Y>,<Z>) from reduced density matrices, ordered like
    QuantumKernelSVM's projected QNode: [X0,Y0,Z0, X1,Y1,Z1, ...]."""
    N = states.shape[0]
    psi = states.reshape((N,) + (2,) * n_qubits)
    feats = np.empty((N, 3 * n_qubits))
    for w in range(n_qubits):
        m = np.moveaxis(psi, w + 1, 1).reshape(N, 2, -1)
        rho = np.einsum("bik,bjk->bij", m, m.conj())
        r01 = rho[:, 0, 1]
        feats[:, 3 * w] = 2.0 * r01.real            # <X>
        feats[:, 3 * w + 1] = -2.0 * r01.imag       # <Y>
        feats[:, 3 * w + 2] = (rho[:, 0, 0] - rho[:, 1, 1]).real   # <Z>
    return feats


def rbf(FA, FB, gamma):
    sq = (np.sum(FA ** 2, 1)[:, None] + np.sum(FB ** 2, 1)[None, :] - 2.0 * FA @ FB.T)
    np.maximum(sq, 0.0, out=sq)
    return np.exp(-gamma * sq)


def fidelity(SA, SB, chunk=4000):
    K = np.empty((len(SA), len(SB)))
    for i in range(0, len(SA), chunk):
        K[i:i + chunk] = np.abs(SA[i:i + chunk] @ SB.conj().T) ** 2
    return K


# --------------------------------------------------------------------------- #
# metrics, identical to the paper's evaluation path
# --------------------------------------------------------------------------- #
def score_metrics(y, s):
    return {"roc_auc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "tpr_at_1pct_fpr": float(tpr_at_fpr(y, s, 0.01)),
            "tpr_at_0.1pct_fpr": float(tpr_at_fpr(y, s, 0.001))}


def load_nslkdd(n_features=8, seed=42):
    from data import load
    return load("nslkdd", n_features=n_features, reduction="pca", binary=True,
                scale="minmax", seed=seed)


# --------------------------------------------------------------------------- #
# validation: the engine must agree with QuantumKernelSVM
# --------------------------------------------------------------------------- #
def validate(n_qubits=8, n=12):
    Xtr, ytr, Xte, yte = load_nslkdd(n_qubits)
    X = np.asarray(Xtr[:n], dtype=np.float64)
    report = {}
    for enc in ("iqp", "angle"):
        S = statevectors(X, n_qubits, encoding=enc)
        # projected features vs the QNode the paper used
        q = QuantumKernelSVM(n_qubits=n_qubits, encoding=enc, kernel="projected")
        ref = np.array([np.asarray(q._proj_qnode(x), dtype=float).ravel() for x in X])
        mine = pauli_features(S, n_qubits)
        dp = float(np.max(np.abs(ref - mine)))
        # fidelity kernel vs the adjoint-circuit QNode the paper used
        qf = QuantumKernelSVM(n_qubits=n_qubits, encoding=enc, kernel="fidelity")
        Kref = qf._kernel_matrix(X[:6], X[:6])
        dk = float(np.max(np.abs(Kref - fidelity(S[:6], S[:6]))))
        report[enc] = {"max_abs_err_pauli": dp, "max_abs_err_fidelity": dk}
    return report


# --------------------------------------------------------------------------- #
# A. reproduce the published QSVM number
# --------------------------------------------------------------------------- #
def reproduce(n_qubits=8, n_train=2000):
    from sklearn.svm import SVC
    Xtr, ytr, Xte, yte = load_nslkdd(n_qubits)
    Xq, yq = Xtr[:n_train], ytr[:n_train]
    t0 = time.time()
    Str = statevectors(Xq, n_qubits, "iqp")
    Ste = statevectors(Xte, n_qubits, "iqp")
    t_sim = time.time() - t0
    Ftr, Fte = pauli_features(Str, n_qubits), pauli_features(Ste, n_qubits)
    svc = SVC(kernel="precomputed", C=1.0).fit(rbf(Ftr, Ftr, 1.0), yq)
    s = svc.decision_function(rbf(Fte, Ftr, 1.0))
    return {"n_train": n_train, "n_test": int(len(yte)), "sim_seconds": t_sim,
            "metrics": score_metrics(yte, s)}


# --------------------------------------------------------------------------- #
# A. fair-surrogate decomposition of the published QSVM-vs-rf-kernel gap
# --------------------------------------------------------------------------- #
GAMMAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
CS = [0.1, 1.0, 10.0, 100.0]


def _cv_tune(K_of_gamma, y, gammas, Cs, seed=42):
    """5-fold stratified CV on TRAINING rows only; returns (gamma, C, cv_auc).
    K_of_gamma(g) -> full train Gram matrix for bandwidth g (None = no bandwidth)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(y)), y))
    best = (None, None, -1.0)
    for g in gammas:
        K = K_of_gamma(g)
        for C in Cs:
            aucs = []
            for tr, va in folds:
                svc = SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr, tr)], y[tr])
                aucs.append(roc_auc_score(y[va], svc.decision_function(K[np.ix_(va, tr)])))
            m = float(np.mean(aucs))
            if m > best[2]:
                best = (g, C, m)
    return best


def _fit_score(Ktr, y, Kte, C):
    from sklearn.svm import SVC
    return SVC(kernel="precomputed", C=C).fit(Ktr, y).decision_function(Kte)


def fair(n_qubits=8, n_train=2000, n_boot=2000):
    from attribution import RandomFeatureKernelSVM
    Xtr, ytr, Xte, yte = load_nslkdd(n_qubits)
    Xq, yq = np.asarray(Xtr[:n_train], float), np.asarray(ytr[:n_train])
    Xte = np.asarray(Xte, float)
    yte = np.asarray(yte)

    Str, Ste = statevectors(Xq, n_qubits, "iqp"), statevectors(Xte, n_qubits, "iqp")
    Ftr, Fte = pauli_features(Str, n_qubits), pauli_features(Ste, n_qubits)

    scores, meta = {}, {}

    # 1. the published quantum model, exactly as reported (gamma=1, C=1)
    scores["Q proj (published: g=1,C=1)"] = _fit_score(rbf(Ftr, Ftr, 1.0), yq,
                                                      rbf(Fte, Ftr, 1.0), 1.0)
    # 2. quantum projected kernel, tuned with the SAME budget as the classical
    g, C, cv = _cv_tune(lambda g: rbf(Ftr, Ftr, g), yq, GAMMAS, CS)
    scores["Q proj (tuned)"] = _fit_score(rbf(Ftr, Ftr, g), yq, rbf(Fte, Ftr, g), C)
    meta["Q proj (tuned)"] = {"gamma": g, "C": C, "cv_auc": cv}
    # 3. quantum fidelity kernel (no bandwidth), C tuned
    Kf = fidelity(Str, Str)
    _, C, cv = _cv_tune(lambda g: Kf, yq, [None], CS)
    scores["Q fidelity (tuned C)"] = _fit_score(Kf, yq, fidelity(Ste, Str), C)
    meta["Q fidelity (tuned C)"] = {"C": C, "cv_auc": cv}

    # 4. the published classical surrogate: Nystroem-512 + LinearSVC, 1200 rows
    for sub, label in ((1200, "rf-kern (published: 1200 rows)"),
                       (n_train, "rf-kern (same 2000 rows)")):
        runs = []
        for sd in (42, 43, 44):
            rf = RandomFeatureKernelSVM(method="nystroem", n_components=512,
                                        gamma=1.0, C=1.0, seed=sd)
            rf.fit(Xtr[:sub], ytr[:sub])
            runs.append(rf.decision_function(Xte))
        scores[label] = runs[0]
        meta[label] = {"per_seed_roc_auc": [float(roc_auc_score(yte, r)) for r in runs]}

    # 5. exact classical RBF kernel on the raw PCA features
    scores["exact RBF (g=1,C=1)"] = _fit_score(rbf(Xq, Xq, 1.0), yq,
                                              rbf(Xte, Xq, 1.0), 1.0)
    g, C, cv = _cv_tune(lambda g: rbf(Xq, Xq, g), yq, GAMMAS, CS)
    scores["exact RBF (tuned)"] = _fit_score(rbf(Xq, Xq, g), yq, rbf(Xte, Xq, g), C)
    meta["exact RBF (tuned)"] = {"gamma": g, "C": C, "cv_auc": cv}

    ref = "Q proj (published: g=1,C=1)"
    table = {}
    for name, s in scores.items():
        row = {"metrics": score_metrics(yte, s), **meta.get(name, {})}
        if name != ref:
            row["vs_published_quantum"] = {}
            for mt in ("roc_auc", "auprc", "tpr_at_1pct_fpr"):
                b = paired_bootstrap(yte, s, scores[ref], mt, n_boot=n_boot)
                row["vs_published_quantum"][mt] = {
                    "delta_quantum_minus_this": b.observed_diff, "ci95": [b.ci95_low, b.ci95_high],
                    "p": b.p_value}
        table[name] = row
    return {"n_train": n_train, "n_test": int(len(yte)), "gamma_grid": GAMMAS,
            "C_grid": CS, "models": table}


# --------------------------------------------------------------------------- #
# B. geometric difference and model complexity (Huang et al. 2021)
# --------------------------------------------------------------------------- #
def _trace_normalise(K):
    return K * (len(K) / np.trace(K))


def _psd_sqrt(K):
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def geometric_difference(KC, KQ, lam):
    """g(K_C || K_Q) = sqrt(|| sqrt(KQ) sqrt(KC) (KC+lam I)^-2 sqrt(KC) sqrt(KQ) ||_inf),
    both trace-normalised to N. -> sqrt(||sqrt(KQ) KC^-1 sqrt(KQ)||) as lam -> 0.
    Small g: the classical kernel can reproduce anything the quantum kernel learns."""
    N = len(KC)
    KC, KQ = _trace_normalise(KC), _trace_normalise(KQ)
    sC, sQ = _psd_sqrt(KC), _psd_sqrt(KQ)
    inv = np.linalg.inv(KC + lam * np.eye(N))
    M = sQ @ sC @ inv @ inv @ sC @ sQ
    return float(np.sqrt(np.max(np.linalg.eigvalsh((M + M.T) / 2))))


def model_complexity(K, y, lam):
    """s_K = ||sqrt(K) (K+lam I)^-1 y||^2 with y in {-1,+1}, Tr(K)=N.
    -> y^T K^-1 y as lam -> 0. Lower = learns the *actual labels* from fewer data."""
    N = len(K)
    K = _trace_normalise(K)
    yy = np.where(np.asarray(y) > 0, 1.0, -1.0)
    a = np.linalg.solve(K + lam * np.eye(N), yy)
    return float(a @ K @ a)


def geometry(n_qubits=8, N=800, lams=(1e-3, 1e-2, 1e-1), seed=0):
    from data import load
    rng = np.random.RandomState(seed)
    out = {}
    for ds in ("nslkdd", "unsw", "cicids", "toniot"):
        Xtr, ytr, _, _ = load(ds, n_features=n_qubits, reduction="pca", binary=True,
                              scale="minmax", seed=42)
        idx = rng.choice(len(Xtr), N, replace=False)
        X, y = np.asarray(Xtr[idx], float), np.asarray(ytr[idx])
        S_iqp, S_ang = statevectors(X, n_qubits, "iqp"), statevectors(X, n_qubits, "angle")
        quantum = {"IQP projected (g=1)": rbf(pauli_features(S_iqp, n_qubits),
                                              pauli_features(S_iqp, n_qubits), 1.0),
                   "IQP fidelity": fidelity(S_iqp, S_iqp),
                   "angle fidelity": fidelity(S_ang, S_ang)}
        sq = np.sum(X ** 2, 1)[:, None] + np.sum(X ** 2, 1)[None, :] - 2 * X @ X.T
        med = float(np.median(sq[np.triu_indices(N, 1)]))
        classical = {f"RBF g={g:g}": rbf(X, X, g) for g in (0.03, 0.1, 0.3, 1.0)}
        classical["RBF median"] = rbf(X, X, 1.0 / med)
        classical["linear"] = X @ X.T + 1e-9 * np.eye(N)
        res = {"N": N, "sqrt_N": float(np.sqrt(N)), "positive_rate": float(np.mean(y)),
               "median_sq_dist": med, "kernels": {}}
        for qn, KQ in quantum.items():
            row = {}
            for lam in lams:
                gs = {cn: geometric_difference(KC, KQ, lam) for cn, KC in classical.items()}
                best = min(gs, key=gs.get)
                row[f"lam={lam:g}"] = {"g_min": gs[best], "best_classical": best, "g_all": gs,
                                      "s_quantum": model_complexity(KQ, y, lam),
                                      "s_best_classical": model_complexity(classical[best], y, lam)}
            res["kernels"][qn] = row
        out[ds] = res
        print(f"[geometry] {ds} done", flush=True)
    return out


# --------------------------------------------------------------------------- #
# C. kernel concentration vs qubit count
# --------------------------------------------------------------------------- #
def concentration(qubits=(2, 4, 6, 8, 10, 12), N=400, n_test=4000, seed=0):
    from sklearn.svm import SVC
    rng = np.random.RandomState(seed)
    out = {}
    for n in qubits:
        Xtr, ytr, Xte, yte = load_nslkdd(n)
        itr = rng.choice(len(Xtr), N, replace=False)
        ite = rng.choice(len(Xte), n_test, replace=False)
        X, y = np.asarray(Xtr[itr], float), np.asarray(ytr[itr])
        Xt, yt = np.asarray(Xte[ite], float), np.asarray(yte[ite])
        row = {}
        for enc in ("iqp", "angle"):
            S, St = statevectors(X, n, enc), statevectors(Xt, n, enc)
            F, Ft = pauli_features(S, n), pauli_features(St, n)
            for kname, K, Kt in ((f"{enc} fidelity", fidelity(S, S), fidelity(St, S)),
                                 (f"{enc} projected", rbf(F, F, 1.0), rbf(Ft, F, 1.0))):
                off = K[np.triu_indices(N, 1)]
                s = SVC(kernel="precomputed", C=1.0).fit(K, y).decision_function(Kt)
                row[kname] = {"offdiag_mean": float(off.mean()), "offdiag_var": float(off.var()),
                              "offdiag_std": float(off.std()),
                              "test_roc_auc": float(roc_auc_score(yt, s))}
        out[str(n)] = row
        print(f"[concentration] n={n} done", flush=True)
    return out


# --------------------------------------------------------------------------- #
# D. finite-shot robustness of the published projected QSVM
# --------------------------------------------------------------------------- #
def shots(n_qubits=8, n_train=2000, budgets=(16, 64, 256, 1024, 4096), reps=5):
    """Each Pauli expectation mu is replaced by the mean of M +-1 outcomes with
    P(+1)=(1+mu)/2 -- the estimate hardware would return from M shots per basis
    (all-X, all-Y and all-Z settings, so 3M shots per sample in total)."""
    from sklearn.svm import SVC
    Xtr, ytr, Xte, yte = load_nslkdd(n_qubits)
    Xq, yq = np.asarray(Xtr[:n_train], float), np.asarray(ytr[:n_train])
    yte = np.asarray(yte)
    Ftr = pauli_features(statevectors(Xq, n_qubits, "iqp"), n_qubits)
    Fte = pauli_features(statevectors(np.asarray(Xte, float), n_qubits, "iqp"), n_qubits)

    def run(A, B):
        s = SVC(kernel="precomputed", C=1.0).fit(rbf(A, A, 1.0), yq).decision_function(rbf(B, A, 1.0))
        return score_metrics(yte, s)

    out = {"exact": run(Ftr, Fte), "budgets": {}}
    for M in budgets:
        reps_m = []
        for r in range(reps):
            rng = np.random.RandomState(1000 * M + r)
            est = lambda F: (2.0 * rng.binomial(M, np.clip((1 + F) / 2, 0, 1)) - M) / M
            reps_m.append(run(est(Ftr), est(Fte)))
        out["budgets"][str(M)] = {
            k: {"mean": float(np.mean([m[k] for m in reps_m])),
                "std": float(np.std([m[k] for m in reps_m]))} for k in reps_m[0]}
        print(f"[shots] M={M} done", flush=True)
    return out



# --------------------------------------------------------------------------- #
# E. dequantisation: can a classical periodic feature map reproduce the
#    quantum kernel's robustness to distribution shift?
# --------------------------------------------------------------------------- #
def trig_features(X, pairwise=False):
    """phi(x) = [sin x_i, cos x_i] -- exactly the angle-encoded projected kernel's
    features. pairwise=True adds the IQP ZZ-angle terms for neighbouring
    features, sin/cos of 2 x_i and of 2 (pi - x_i)(pi - x_{i+1}): the same
    phases the ZZ feature map writes, but with no quantum state at all."""
    X = np.asarray(X, float)
    cols = [np.sin(X), np.cos(X)]
    if pairwise:
        zz = 2.0 * (np.pi - X[:, :-1]) * (np.pi - X[:, 1:])
        cols += [np.sin(2 * X), np.cos(2 * X), np.sin(zz), np.cos(zz)]
    return np.hstack(cols)


def dequant(n_qubits=8, n_train=2000, n_boot=2000):
    Xtr, ytr, Xte, yte = load_nslkdd(n_qubits)
    Xq, yq = np.asarray(Xtr[:n_train], float), np.asarray(ytr[:n_train])
    Xte, yte = np.asarray(Xte, float), np.asarray(yte)
    Ftr = pauli_features(statevectors(Xq, n_qubits, "iqp"), n_qubits)
    Fte = pauli_features(statevectors(Xte, n_qubits, "iqp"), n_qubits)

    scores, meta = {}, {}

    def tuned(name, A, B):
        g, C, cv = _cv_tune(lambda g: rbf(A, A, g), yq, GAMMAS, CS)
        scores[name] = _fit_score(rbf(A, A, g), yq, rbf(B, A, g), C)
        meta[name] = {"gamma": g, "C": C, "cv_auc": cv, "n_features": int(A.shape[1])}

    tuned("Q IQP projected (tuned)", Ftr, Fte)
    tuned("classical trig [sin,cos]", trig_features(Xq), trig_features(Xte))
    tuned("classical trig + IQP phases", trig_features(Xq, True), trig_features(Xte, True))
    tuned("classical RBF raw (tuned)", Xq, Xte)

    # ORACLE: best raw-feature RBF bandwidth chosen on the TEST set -- an upper
    # bound for what any bandwidth could do, not a legitimate model.
    oracle = {}
    for g in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        for C in CS:
            s = _fit_score(rbf(Xq, Xq, g), yq, rbf(Xte, Xq, g), C)
            oracle[f"g={g:g},C={C:g}"] = float(roc_auc_score(yte, s))
    best = max(oracle, key=oracle.get)

    ref = "Q IQP projected (tuned)"
    table = {}
    for name, s in scores.items():
        row = {"metrics": score_metrics(yte, s), **meta.get(name, {})}
        if name != ref:
            row["vs_quantum"] = {}
            for mt in ("roc_auc", "auprc", "tpr_at_1pct_fpr"):
                b = paired_bootstrap(yte, s, scores[ref], mt, n_boot=n_boot)
                row["vs_quantum"][mt] = {"delta_quantum_minus_this": b.observed_diff,
                                         "ci95": [b.ci95_low, b.ci95_high], "p": b.p_value}
        table[name] = row
    return {"n_train": n_train, "models": table,
            "oracle_raw_rbf": {"best": best, "best_test_auc": oracle[best], "grid": oracle}}



# --------------------------------------------------------------------------- #
# B'. geometric difference against the strongest classical family: kernels on
#     the circuit's own trigonometric phases, with no quantum state at all
# --------------------------------------------------------------------------- #
FULL_RANK_GAMMAS = (0.03, 0.1, 0.3, 1.0, 3.0)


def geometry_trig(n_qubits=8, N=800, lam=1e-2, seed=0):
    """min over bandwidths of g(K_C || K_Q) for three classical families; the
    rank-deficient linear kernel is excluded (its regularised inverse vanishes on
    the null space, which makes g spuriously small)."""
    from data import load
    rng = np.random.RandomState(seed)
    out = {}
    for ds in ("nslkdd", "unsw", "cicids", "toniot"):
        Xtr, ytr, _, _ = load(ds, n_features=n_qubits, reduction="pca", binary=True,
                              scale="minmax", seed=42)
        idx = rng.choice(len(Xtr), N, replace=False)
        X, y = np.asarray(Xtr[idx], float), np.asarray(ytr[idx])
        S = statevectors(X, n_qubits, "iqp")
        F = pauli_features(S, n_qubits)
        quantum = {"IQP projected": rbf(F, F, 1.0), "IQP fidelity": fidelity(S, S)}
        families = {"RBF raw": X, "trig [sin,cos]": trig_features(X),
                    "trig + IQP phases": trig_features(X, True)}
        res = {}
        for qn, KQ in quantum.items():
            row = {"s_quantum": model_complexity(KQ, y, lam)}
            for fn, Z in families.items():
                gs = {g: geometric_difference(rbf(Z, Z, g), KQ, lam) for g in FULL_RANK_GAMMAS}
                gb = min(gs, key=gs.get)
                row[fn] = {"g_min": gs[gb], "best_gamma": gb,
                           "s_classical": model_complexity(rbf(Z, Z, gb), y, lam)}
            res[qn] = row
        out[ds] = {"N": N, "sqrt_N": float(np.sqrt(N)), "lambda": lam, "kernels": res}
        print(f"[geometry_trig] {ds} done", flush=True)
    return out



# --------------------------------------------------------------------------- #
# F. hyperparameter landscape and shift-aware model selection
# --------------------------------------------------------------------------- #
WIDE_GAMMAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def _shift_folds(attack_names, k=5, seed=42):
    """Leave-attack-types-out CV: every attack type lives in exactly one fold, so
    each validation fold contains attack types its training fold never saw --
    the same condition as NSL-KDD's official test set. Benign rows are spread
    evenly over the folds."""
    from sklearn.model_selection import GroupKFold
    rng = np.random.RandomState(seed)
    groups = np.array([n if n != "normal" else f"normal_{rng.randint(k)}"
                       for n in attack_names])
    return list(GroupKFold(k).split(np.zeros(len(groups)), groups=groups))


def landscape(n_qubits=8, n_train=2000):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from data import load_meta
    Xtr, ytr, Xte, yte = load_nslkdd(n_qubits)
    _, ym, _, _, meta = load_meta("nslkdd", n_qubits, "pca", False, "minmax", 42)
    names = np.array(meta["class_names"])[np.asarray(ym[:n_train])]
    Xq, yq = np.asarray(Xtr[:n_train], float), np.asarray(ytr[:n_train])
    Xte, yte = np.asarray(Xte, float), np.asarray(yte)
    assert np.all((names != "normal").astype(int) == yq), "label misalignment"

    std_folds = list(StratifiedKFold(5, shuffle=True, random_state=42)
                     .split(np.zeros(n_train), yq))
    shift_folds = [(tr, va) for tr, va in _shift_folds(names)
                   if len(np.unique(yq[va])) == 2 and len(np.unique(yq[tr])) == 2]

    St = statevectors(Xq, n_qubits, "iqp")
    Se = statevectors(Xte, n_qubits, "iqp")
    fams = {"Q IQP projected": (pauli_features(St, n_qubits), pauli_features(Se, n_qubits)),
            "classical trig + IQP phases": (trig_features(Xq, True), trig_features(Xte, True)),
            "classical trig [sin,cos]": (trig_features(Xq), trig_features(Xte)),
            "classical RBF raw": (Xq, Xte)}

    def cv(K, folds, C):
        a = []
        for tr, va in folds:
            s = SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr, tr)], yq[tr])
            a.append(roc_auc_score(yq[va], s.decision_function(K[np.ix_(va, tr)])))
        return float(np.mean(a))

    def sweep(Ktr, Kte, C):
        s = SVC(kernel="precomputed", C=C).fit(Ktr, yq).decision_function(Kte)
        return {"std_cv": cv(Ktr, std_folds, C), "shift_cv": cv(Ktr, shift_folds, C),
                "test": score_metrics(yte, s)}

    out = {"n_shift_folds": len(shift_folds), "gammas": WIDE_GAMMAS, "Cs": CS, "families": {}}
    for fam, (A, B) in fams.items():
        grid = {}
        for g in WIDE_GAMMAS:
            Ktr, Kte = rbf(A, A, g), rbf(B, A, g)
            for C in CS:
                grid[f"{g:g}|{C:g}"] = sweep(Ktr, Kte, C)
        out["families"][fam] = grid
        print(f"[landscape] {fam} done", flush=True)
    Kf, Kfe = fidelity(St, St), fidelity(Se, St)
    out["families"]["Q IQP fidelity"] = {f"-|{C:g}": sweep(Kf, Kfe, C) for C in CS}
    print("[landscape] Q IQP fidelity done", flush=True)
    return out



# --------------------------------------------------------------------------- #
# G. paired tests on the configurations each selection rule actually picks
# --------------------------------------------------------------------------- #
def selected(n_qubits=8, n_train=2000, n_boot=2000):
    """Re-fit the std-CV and shift-CV winners from landscape.json and compare
    each to the quantum projected kernel's std-CV winner with a paired bootstrap."""
    from sklearn.svm import SVC
    land = json.load(open(os.path.join(RESULTS, "landscape.json")))
    Xtr, ytr, Xte, yte = load_nslkdd(n_qubits)
    Xq, yq = np.asarray(Xtr[:n_train], float), np.asarray(ytr[:n_train])
    Xte, yte = np.asarray(Xte, float), np.asarray(yte)
    St, Se = statevectors(Xq, n_qubits, "iqp"), statevectors(Xte, n_qubits, "iqp")
    fams = {"Q IQP projected": (pauli_features(St, n_qubits), pauli_features(Se, n_qubits)),
            "classical trig + IQP phases": (trig_features(Xq, True), trig_features(Xte, True)),
            "classical trig [sin,cos]": (trig_features(Xq), trig_features(Xte)),
            "classical RBF raw": (Xq, Xte)}

    def score(fam, key):
        g, C = key.split("|")
        C = float(C)
        if fam == "Q IQP fidelity":
            Ktr, Kte = fidelity(St, St), fidelity(Se, St)
        else:
            A, B = fams[fam]
            Ktr, Kte = rbf(A, A, float(g)), rbf(B, A, float(g))
        return SVC(kernel="precomputed", C=C).fit(Ktr, yq).decision_function(Kte)

    picks, scores = {}, {}
    for fam, grid in land["families"].items():
        for rule in ("std_cv", "shift_cv"):
            k = max(grid, key=lambda kk: grid[kk][rule])
            picks[(fam, rule)] = k
            scores[(fam, rule)] = score(fam, k)
    ref = ("Q IQP projected", "std_cv")
    out = {}
    for (fam, rule), s in scores.items():
        row = {"config": picks[(fam, rule)], "metrics": score_metrics(yte, s)}
        if (fam, rule) != ref:
            row["vs_quantum_stdcv"] = {}
            for mt in ("roc_auc", "auprc", "tpr_at_1pct_fpr"):
                b = paired_bootstrap(yte, s, scores[ref], mt, n_boot=n_boot)
                row["vs_quantum_stdcv"][mt] = {"delta_quantum_minus_this": b.observed_diff,
                                               "ci95": [b.ci95_low, b.ci95_high], "p": b.p_value}
        out[f"{fam} | {rule}"] = row
    return out



# --------------------------------------------------------------------------- #
# H. within-rule paired tests (quantum vs each classical family, same selector)
# --------------------------------------------------------------------------- #
def selected_within(n_qubits=8, n_train=2000, n_boot=2000):
    from sklearn.svm import SVC
    land = json.load(open(os.path.join(RESULTS, "landscape.json")))
    Xtr, ytr, Xte, yte = load_nslkdd(n_qubits)
    Xq, yq = np.asarray(Xtr[:n_train], float), np.asarray(ytr[:n_train])
    Xte, yte = np.asarray(Xte, float), np.asarray(yte)
    St, Se = statevectors(Xq, n_qubits, "iqp"), statevectors(Xte, n_qubits, "iqp")
    feats = {"Q IQP projected": (pauli_features(St, n_qubits), pauli_features(Se, n_qubits)),
             "classical trig + IQP phases": (trig_features(Xq, True), trig_features(Xte, True)),
             "classical trig [sin,cos]": (trig_features(Xq), trig_features(Xte)),
             "classical RBF raw": (Xq, Xte)}
    scores = {}
    for fam, grid in land["families"].items():
        for rule in ("std_cv", "shift_cv"):
            k = max(grid, key=lambda kk: grid[kk][rule])
            g, C = k.split("|")
            if fam == "Q IQP fidelity":
                Ktr, Kte = fidelity(St, St), fidelity(Se, St)
            else:
                A, B = feats[fam]
                Ktr, Kte = rbf(A, A, float(g)), rbf(B, A, float(g))
            scores[f"{fam}|{rule}"] = SVC(kernel="precomputed", C=float(C)).fit(Ktr, yq).decision_function(Kte)
    np.savez_compressed(os.path.join(RESULTS, "selected_scores.npz"), y_test=yte,
                        **{k.replace(" ", "_").replace("|", "__").replace("+", "plus")
                           .replace("[", "").replace("]", "").replace(",", "_"): v for k, v in scores.items()})
    out = {}
    for rule in ("std_cv", "shift_cv"):
        ref = scores[f"Q IQP projected|{rule}"]
        for fam in land["families"]:
            if fam == "Q IQP projected":
                continue
            row = {}
            for mt in ("roc_auc", "auprc", "tpr_at_1pct_fpr"):
                b = paired_bootstrap(yte, scores[f"{fam}|{rule}"], ref, mt, n_boot=n_boot)
                row[mt] = {"quantum": b.score_b, "this": b.score_a,
                           "delta_quantum_minus_this": b.observed_diff,
                           "ci95": [b.ci95_low, b.ci95_high], "p": b.p_value}
            out[f"{rule} | {fam}"] = row
    return out



# --------------------------------------------------------------------------- #
# I. the landscape on a second official split (UNSW-NB15), as a contrast case
# --------------------------------------------------------------------------- #
def landscape_ds(ds="unsw", n_qubits=8, n_train=2000, n_test=20000, seed=0):
    """Same protocol as `landscape`, for a dataset whose training file may be
    sorted by class: rows are drawn by a seeded stratified sample (identical for
    every kernel) instead of taking the first n_train."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.svm import SVC
    from data import load_meta
    X, ym, Xt, ymt, meta = load_meta(ds, n_qubits, "pca", False, "minmax", 42)
    names_all = np.array(meta["class_names"])
    benign = [n for n in names_all if str(n).lower() in ("normal", "benign")][0]
    yb, ybt = (names_all[ym] != benign).astype(int), (names_all[ymt] != benign).astype(int)
    itr, _ = train_test_split(np.arange(len(yb)), train_size=n_train, stratify=ym, random_state=seed)
    ite, _ = train_test_split(np.arange(len(ybt)), train_size=n_test, stratify=ymt, random_state=seed)
    Xq, yq = np.asarray(X[itr], float), yb[itr]
    Xe, ye = np.asarray(Xt[ite], float), ybt[ite]
    names = np.array([n if n != benign else "normal" for n in names_all[ym[itr]]])

    std_folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n_train), yq))
    shift_folds = [(tr, va) for tr, va in _shift_folds(names)
                   if len(np.unique(yq[va])) == 2 and len(np.unique(yq[tr])) == 2]
    St, Se = statevectors(Xq, n_qubits, "iqp"), statevectors(Xe, n_qubits, "iqp")
    fams = {"Q IQP projected": (pauli_features(St, n_qubits), pauli_features(Se, n_qubits)),
            "classical trig + IQP phases": (trig_features(Xq, True), trig_features(Xe, True)),
            "classical RBF raw": (Xq, Xe)}

    def cv(K, folds, C):
        a = []
        for tr, va in folds:
            s = SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr, tr)], yq[tr])
            a.append(roc_auc_score(yq[va], s.decision_function(K[np.ix_(va, tr)])))
        return float(np.mean(a))

    out = {"dataset": ds, "n_train": n_train, "n_test": n_test, "n_shift_folds": len(shift_folds),
           "unseen_test_types": sorted(set(names_all[ymt]) - set(names_all[ym])),
           "gammas": WIDE_GAMMAS, "Cs": CS, "families": {}}
    for fam, (A, B) in fams.items():
        grid = {}
        for g in WIDE_GAMMAS:
            Ktr, Kte = rbf(A, A, g), rbf(B, A, g)
            for C in CS:
                s = SVC(kernel="precomputed", C=C).fit(Ktr, yq).decision_function(Kte)
                grid[f"{g:g}|{C:g}"] = {"std_cv": cv(Ktr, std_folds, C),
                                        "shift_cv": cv(Ktr, shift_folds, C),
                                        "test": score_metrics(ye, s)}
        out["families"][fam] = grid
        print(f"[landscape_{ds}] {fam} done", flush=True)
    return out



# --------------------------------------------------------------------------- #
# J. paired tests for the second dataset, standard-CV picks, same sample
# --------------------------------------------------------------------------- #
def selected_within_ds(ds="unsw", n_qubits=8, n_train=2000, n_test=20000, seed=0, n_boot=2000):
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from data import load_meta
    land = json.load(open(os.path.join(RESULTS, f"landscape_{ds}.json")))
    X, ym, Xt, ymt, meta = load_meta(ds, n_qubits, "pca", False, "minmax", 42)
    names_all = np.array(meta["class_names"])
    benign = [n for n in names_all if str(n).lower() in ("normal", "benign")][0]
    yb, ybt = (names_all[ym] != benign).astype(int), (names_all[ymt] != benign).astype(int)
    itr, _ = train_test_split(np.arange(len(yb)), train_size=n_train, stratify=ym, random_state=seed)
    ite, _ = train_test_split(np.arange(len(ybt)), train_size=n_test, stratify=ymt, random_state=seed)
    Xq, yq = np.asarray(X[itr], float), yb[itr]
    Xe, ye = np.asarray(Xt[ite], float), ybt[ite]
    St, Se = statevectors(Xq, n_qubits, "iqp"), statevectors(Xe, n_qubits, "iqp")
    feats = {"Q IQP projected": (pauli_features(St, n_qubits), pauli_features(Se, n_qubits)),
             "classical trig + IQP phases": (trig_features(Xq, True), trig_features(Xe, True)),
             "classical RBF raw": (Xq, Xe)}
    scores = {}
    for fam, grid in land["families"].items():
        k = max(grid, key=lambda kk: grid[kk]["std_cv"])
        g, C = k.split("|")
        A, B = feats[fam]
        scores[fam] = SVC(kernel="precomputed", C=float(C)).fit(rbf(A, A, float(g)), yq) \
            .decision_function(rbf(B, A, float(g)))
    out = {}
    for fam in ("classical trig + IQP phases", "classical RBF raw"):
        row = {}
        for mt in ("roc_auc", "auprc", "tpr_at_1pct_fpr"):
            b = paired_bootstrap(ye, scores[fam], scores["Q IQP projected"], mt, n_boot=n_boot)
            row[mt] = {"quantum": b.score_b, "this": b.score_a,
                       "delta_quantum_minus_this": b.observed_diff,
                       "ci95": [b.ci95_low, b.ci95_high], "p": b.p_value}
        out[f"std_cv | {fam}"] = row
    return out



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["validate", "reproduce", "fair", "geometry", "concentration", "shots", "dequant", "geometry_trig", "landscape", "selected", "selected_within", "landscape_unsw", "selected_unsw"])
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    out = {"validate": validate, "reproduce": reproduce, "fair": fair, "geometry": geometry, "concentration": concentration, "shots": shots, "dequant": dequant, "geometry_trig": geometry_trig, "landscape": landscape, "selected": selected, "selected_within": selected_within, "landscape_unsw": lambda: landscape_ds("unsw"), "selected_unsw": lambda: selected_within_ds("unsw")}[a.part]()
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(RESULTS, f"{a.part}.json"), "w"), indent=2)
