"""Run the trained 4-qubit hybrid QNN's quantum layer on an IBM Quantum device.

The 4-qubit hybrid (angle encoding, three StronglyEntangling layers, classical
Linear(4,4)+Tanh in front and Linear(4,2) behind) is the one quantum model in
the paper whose low-FPR result on NSL-KDD survives FDR. Its published numbers
come from a noiseless statevector simulation; this script asks whether they
survive real hardware. Steps, one mode each:

  train     retrain the model with the campaign's exact recipe (same seed,
            subsample, optimiser, loss) and check the retrained model reproduces
            the published test scores; save the weights
  validate  rebuild the circuit in Qiskit and check its <Z_i> against PennyLane
  shots     predict the effect of finite shots alone (binomial resampling of the
            exact expectations) for several shot counts
  estimate  shot budget for a run
  dryrun    the same pipeline on a local noisy simulation of a Heron device
  run       the real device (refuses without --confirm); one job, weights bound
            as constants, the four input angles as circuit parameters

Only the quantum layer runs on the device. The measured <Z_i> are fed through
the trained classical head exactly as the simulated expectations are.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

from kernel_analysis import RESULTS, score_metrics

N_QUBITS, N_LAYERS = 4, 3
CFG = {"name": "ovn1_hybrid_angle_q4", "dataset": "nslkdd", "n_features": 4, "reduction": "pca",
       "binary": True, "scale": "minmax", "imbalance": "class_weight", "model_type": "hybrid_qnn",
       "n_qubits": N_QUBITS, "n_layers": N_LAYERS, "encoding": "angle", "device": "cpu",
       "seed": 42, "epochs": 20, "batch_size": 64, "lr": 0.005, "q_subsample": 2000,
       "extra": {"device": "cpu", "ansatz": "strongly_entangling"}}
STATE = os.path.join(RESULTS, "hybrid_q4_s42_state.npz")
PUBLISHED = os.path.join(RESULTS, "..", "nslkdd_hybrid_qnn_q4_l3_angle_s42_*.json")


# --------------------------------------------------------------------------- #
# the model, in numpy, from its saved weights
# --------------------------------------------------------------------------- #
def load_state():
    d = np.load(STATE)
    return {k: d[k] for k in d.files}


def pre(X, sd):
    return np.tanh(np.asarray(X, float) @ sd["pre.0.weight"].T + sd["pre.0.bias"])


def head(Z, sd):
    logits = np.asarray(Z, float) @ sd["post.weight"].T + sd["post.bias"]
    e = np.exp(logits - logits.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True))[:, 1]


def exact_expvals(A, weights):
    """<Z_i> of the variational circuit for each row of angles A, via PennyLane."""
    import pennylane as qml
    weights = np.asarray(weights, float)   # the saved weights are float32; compare in float64
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circ(a):
        qml.AngleEmbedding(a, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return np.array([np.asarray(circ(a), float) for a in np.asarray(A, float)])


# --------------------------------------------------------------------------- #
# the same circuit in Qiskit
# --------------------------------------------------------------------------- #
def hybrid_circuit(weights, n=N_QUBITS):
    """AngleEmbedding(RY) then StronglyEntanglingLayers with the trained weights
    bound. PennyLane's Rot(phi, theta, omega) is RZ(phi), RY(theta), RZ(omega) in
    circuit order; layer l entangles wire i with wire (i + r_l) mod n, where
    r_l = (l mod (n-1)) + 1, as in the template's default ranges."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    weights = np.asarray(weights, float)
    a = ParameterVector("a", n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(a[i], i)
    for l in range(weights.shape[0]):
        for i in range(n):
            phi, theta, omega = (float(v) for v in weights[l, i])
            qc.rz(phi, i)
            qc.ry(theta, i)
            qc.rz(omega, i)
        r = (l % (n - 1)) + 1
        for i in range(n):
            qc.cx(i, (i + r) % n)
    return qc, a


def observables(n=N_QUBITS):
    from qiskit.quantum_info import SparsePauliOp
    return [SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=n) for i in range(n)]


def run_estimator(A, backend, shots, weights, chunk=100, mode=None, log=print, max_seconds=None):
    """Return the (N, 4) matrix of measured <Z_i> for the rows of angles A."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2
    qc, _ = hybrid_circuit(weights)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa = pm.run(qc)
    obs = [o.apply_layout(isa.layout) for o in observables()]
    log(f"transpiled depth={isa.depth()} two-qubit gates="
        f"{sum(1 for inst in isa.data if inst.operation.num_qubits == 2)}")
    A = np.asarray(A, float)
    pubs = [(isa, obs, A[i:i + chunk].reshape(-1, 1, N_QUBITS)) for i in range(0, len(A), chunk)]
    est = EstimatorV2(mode=mode if mode is not None else backend)
    if max_seconds:
        est.options.max_execution_time = int(max_seconds)
    job = est.run(pubs, precision=1.0 / math.sqrt(shots))
    log(f"job {getattr(job, 'job_id', lambda: 'local')()} submitted: {len(pubs)} PUBs, "
        f"{len(A)} samples x {N_QUBITS} observables, {shots} shots each")
    res = job.result()
    Z = np.vstack([np.asarray(pr.data.evs) for pr in res])
    assert Z.shape == (len(A), N_QUBITS), Z.shape
    return Z, job


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def config():
    from config import ExperimentConfig
    return ExperimentConfig.from_dict(json.loads(json.dumps(CFG))).validate()


def test_set():
    from runner import load_data
    _, _, Xte, yte = load_data(config())
    return Xte, yte


def sample_test(n_test, subset=0):
    """Class-balanced subset `subset` of the official test set, in original order.

    Subsets are disjoint and deterministic: subset k is drawn with seed k from the
    rows that subsets 0..k-1 did not use, so subset 0 is the original draw and a
    later subset never repeats a row already measured on hardware."""
    Xte, yte = test_set()
    pool = np.ones(len(yte), bool)
    for k in range(subset + 1):
        rng = np.random.RandomState(k)
        idx = np.sort(np.concatenate([rng.choice(np.where((yte == c) & pool)[0], n_test // 2, replace=False)
                                      for c in (0, 1)]))
        pool[idx] = False
    return Xte[idx], yte[idx], idx


def published():
    files = sorted(glob.glob(PUBLISHED))
    return (files[0], json.load(open(files[0], encoding="utf-8"))) if files else (None, None)


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def train():
    """Exactly runner.run_config's path for the published config, keeping the weights."""
    import logging
    from model_factory import build_model
    from runner import load_data, _train_torch
    from seeding import seed_everything
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("hybrid_hardware")
    cfg = config()
    seed_everything(cfg.seed)
    Xtr, ytr, Xte, yte = load_data(cfg)
    idx = np.random.RandomState(cfg.seed).permutation(len(Xtr))[:cfg.q_subsample]
    Xtr, ytr = Xtr[idx], ytr[idx]
    model, kind = build_model(cfg)
    assert kind == "torch"
    _, proba, train_time, _, _ = _train_torch(model, Xtr, ytr, Xte, yte, cfg, log)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(STATE, test_score=proba, y_test=yte, **sd)
    out = {"state_file": os.path.basename(STATE), "state_keys": sorted(sd),
           "train_time_s": train_time, "metrics_full_test": score_metrics(yte, proba)}
    f, pub = published()
    if pub is not None:
        ps = np.asarray(pub["predictions"]["y_score"], float)
        out["published_file"] = os.path.basename(f)
        out["published_metrics"] = {k: pub["metrics"][k] for k in ("roc_auc", "auprc", "tpr_at_1pct_fpr")
                                    if k in pub["metrics"]}
        out["agreement_with_published"] = {
            "max_abs_score_diff": float(np.max(np.abs(ps - proba))),
            "pearson_r": float(np.corrcoef(ps, proba)[0, 1])}
    return out


def validate(n=64, seed=0):
    """(1) numpy pre/head + PennyLane circuit reproduce the trained model's scores;
    (2) the Qiskit circuit reproduces PennyLane's expectations."""
    from qiskit.quantum_info import Statevector
    sd = load_state()
    Xs, ys, idx = sample_test(n, seed)
    A = pre(Xs, sd)
    Z = exact_expvals(A, sd["qlayer.weights"])
    err_model = float(np.max(np.abs(head(Z, sd) - sd["test_score"][idx])))
    qc, a = hybrid_circuit(sd["qlayer.weights"])
    obs = observables()
    ref = np.array([[Statevector(qc.assign_parameters(dict(zip(a, row)))).expectation_value(o).real
                     for o in obs] for row in A])
    err_circ = float(np.max(np.abs(ref - Z)))
    out = {"n": n, "max_abs_err_numpy_model_vs_torch_scores": err_model,
           "max_abs_err_qiskit_vs_pennylane": err_circ}
    print(json.dumps(out, indent=2))
    assert err_model < 1e-5, "numpy reimplementation does not reproduce the trained model"
    assert err_circ < 1e-9, "Qiskit circuit does not match the PennyLane circuit"
    return out


def shots_sim(n_test, shots_list=(32, 64, 128, 256, 512), reps=20, seed=0):
    sd = load_state()
    Xs, ys, idx = sample_test(n_test, seed)
    Z = exact_expvals(pre(Xs, sd), sd["qlayer.weights"])
    base = score_metrics(ys, head(Z, sd))
    rng = np.random.RandomState(seed)
    out = {"n_test": n_test, "reps": reps, "exact": base}
    p = np.clip((1 + Z) / 2, 0, 1)
    for s in shots_list:
        ms = [score_metrics(ys, head(2 * rng.binomial(s, p) / s - 1, sd)) for _ in range(reps)]
        out[str(s)] = {k: {"mean": float(np.mean([m[k] for m in ms])),
                           "sd": float(np.std([m[k] for m in ms]))} for k in base}
    return out


def estimate(n_test, shots):
    return {"samples": n_test, "measurement_settings_per_sample": 1,
            "circuit_executions": n_test, "shots_per_setting": shots, "total_shots": n_test * shots,
            "rough_qpu_seconds_at_3k_shots_per_s": n_test * shots / 3000,
            "note": "the three Heron kernel jobs ran ~269k shots in 89 to 91 QPU seconds"}


def evaluate(Z_hw, Z_ex, ys, sd, idx, rf_file=None):
    from evaluation import paired_bootstrap
    corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    s_hw, s_ex = head(Z_hw, sd), head(Z_ex, sd)
    out = {"expectation_agreement": {
               "pearson_r": corr(Z_hw, Z_ex), "rmse": float(np.sqrt(np.mean((Z_hw - Z_ex) ** 2))),
               "pearson_r_per_qubit": [corr(Z_hw[:, i], Z_ex[:, i]) for i in range(N_QUBITS)]},
           "score_agreement": {"pearson_r": corr(s_hw, s_ex),
                               "max_abs_diff": float(np.max(np.abs(s_hw - s_ex)))},
           "exact": score_metrics(ys, s_ex), "hardware": score_metrics(ys, s_hw)}
    for metric in ("roc_auc", "auprc", "tpr_at_1pct_fpr"):
        b = paired_bootstrap(ys, s_ex, s_hw, metric)
        out[f"hardware_minus_exact_{metric}"] = {
            "diff": float(b.observed_diff), "ci95": [float(b.ci95_low), float(b.ci95_high)],
            "p": float(b.p_value)}
    if rf_file:
        rf = json.load(open(rf_file, encoding="utf-8"))
        s_rf = np.asarray(rf["predictions"]["y_score"], float)[idx]
        out["rf_file"] = os.path.basename(rf_file)
        out["rf_same_rows"] = score_metrics(ys, s_rf)
        for metric in ("roc_auc", "auprc", "tpr_at_1pct_fpr"):
            for name, s in (("hardware", s_hw), ("exact", s_ex)):
                b = paired_bootstrap(ys, s_rf, s, metric)
                out[f"{name}_minus_rf_{metric}"] = {
                    "diff": float(b.observed_diff), "ci95": [float(b.ci95_low), float(b.ci95_high)],
                    "p": float(b.p_value)}
    return out


def dryrun(n_test, shots, fake="FakeFez", seed=0, rf_file=None):
    import qiskit_ibm_runtime.fake_provider as fp
    backend = getattr(fp, fake)()
    sd = load_state()
    Xs, ys, idx = sample_test(n_test, seed)
    A = pre(Xs, sd)
    Z_ex = exact_expvals(A, sd["qlayer.weights"])
    t0 = time.time()
    Z_hw, _ = run_estimator(A, backend, shots, sd["qlayer.weights"])
    return {"backend": f"{fake} (local noisy simulation)", "n_test": n_test, "shots": shots,
            "subset_seed": seed, "wall_seconds": time.time() - t0,
            **evaluate(Z_hw, Z_ex, ys, sd, idx, rf_file)}


def run(backend_name, n_test, shots, confirm, account=None, max_seconds=150, seed=0, rf_file=None):
    if not confirm:
        raise SystemExit("refusing to submit to a real device without --confirm")
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(name=account) if account else QiskitRuntimeService()
    backend = (svc.backend(backend_name) if backend_name
               else svc.least_busy(simulator=False, operational=True, min_num_qubits=N_QUBITS))
    print("backend:", backend.name)
    sd = load_state()
    Xs, ys, idx = sample_test(n_test, seed)
    A = pre(Xs, sd)
    Z_ex = exact_expvals(A, sd["qlayer.weights"])
    t0 = time.time()
    Z_hw, job = run_estimator(A, backend, shots, sd["qlayer.weights"], max_seconds=max_seconds)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw = os.path.join(RESULTS, f"hybrid_hardware_{backend.name}_{stamp}.npz")
    np.savez_compressed(raw, Z_hardware=Z_hw, Z_exact=Z_ex, angles=A, X_test=Xs, y_test=ys,
                        test_index=idx, score_hardware=head(Z_hw, sd), score_exact=head(Z_ex, sd))
    out = {"backend": backend.name, "job_ids": [job.job_id()], "n_test": n_test, "shots": shots,
           "subset_seed": seed, "wall_seconds": time.time() - t0, "raw_file": os.path.basename(raw),
           **evaluate(Z_hw, Z_ex, ys, sd, idx, rf_file)}
    try:
        out["qpu_seconds"] = job.usage()
    except Exception:
        pass
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "validate", "shots", "estimate", "dryrun", "run"])
    ap.add_argument("--backend", default=None)
    ap.add_argument("--account", default=None)
    ap.add_argument("--fake", default="FakeFez")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--shots", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0,
                    help="index of the disjoint class-balanced test subset (0 = the original draw)")
    ap.add_argument("--rf", default=None, help="published RF result JSON for the same-rows comparison")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--max-seconds", type=int, default=150,
                    help="max_execution_time for a real job (QPU seconds)")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if a.mode == "train":
        out = train()
    elif a.mode == "validate":
        out = validate(seed=a.seed)
    elif a.mode == "shots":
        out = shots_sim(a.n_test, seed=a.seed)
    elif a.mode == "estimate":
        out = estimate(a.n_test, a.shots)
    elif a.mode == "dryrun":
        out = dryrun(a.n_test, a.shots, a.fake, a.seed, a.rf)
    else:
        out = run(a.backend, a.n_test, a.shots, a.confirm, a.account, a.max_seconds, a.seed, a.rf)
    print(json.dumps(out, indent=2, default=str))
    tag = a.mode if a.mode != "run" else f"run_{out['backend']}" + (f"_s{a.seed}" if a.seed else "")
    json.dump(out, open(os.path.join(RESULTS, f"hybrid_hardware_{tag}.json"), "w"), indent=2, default=str)
