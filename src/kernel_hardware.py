"""Run the IQP projected quantum kernel on IBM Quantum hardware.

The projected kernel needs only the single-qubit expectations <X_i>, <Y_i>, <Z_i>
of the feature-map state, so the whole experiment is ONE parameterised 8-qubit
circuit with 24 observables, executed once per sample. That is linear in the
number of samples and shallow enough for current devices.

Modes (in order of cost):
  validate  Aer statevector: the Qiskit circuit's expectations must equal the
            PennyLane pauli_features used in kernel_analysis.py (free).
  estimate  circuit / shot counts and a rough QPU-time figure; no jobs (free).
  dryrun    the full pipeline on a noisy fake backend, locally (free).
  run       submit to a real backend. Refuses without --confirm.

Examples:
  python kernel_hardware.py validate
  python kernel_hardware.py estimate --n-train 200 --n-test 500 --shots 256
  python kernel_hardware.py dryrun   --n-train 200 --n-test 500 --shots 256
  python kernel_hardware.py run --backend ibm_fez --n-train 200 --n-test 500 --shots 256 --confirm
"""
import argparse
import json
import math
import os
import time

import numpy as np

from kernel_analysis import (RESULTS, load_nslkdd, statevectors, pauli_features, rbf,
                             score_metrics)

N_QUBITS = 8
N_REPEATS = 2


# --------------------------------------------------------------------------- #
# circuit and observables
# --------------------------------------------------------------------------- #
def iqp_circuit(n=N_QUBITS, reps=N_REPEATS):
    """Same gate sequence as quantum_model.zz_feature_map: H, RZ(2x_i), then
    CNOT-RZ(2(pi-x_i)(pi-x_{i+1}))-CNOT on neighbouring wires, repeated."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for w in range(n):
            qc.h(w)
            qc.rz(2.0 * x[w], w)
        for w in range(n - 1):
            qc.cx(w, w + 1)
            qc.rz(2.0 * (math.pi - x[w]) * (math.pi - x[w + 1]), w + 1)
            qc.cx(w, w + 1)
    return qc, x


def observables(n=N_QUBITS):
    """[X0, Y0, Z0, X1, Y1, Z1, ...] -- the order pauli_features uses."""
    from qiskit.quantum_info import SparsePauliOp
    obs = []
    for w in range(n):
        for p in "XYZ":
            obs.append(SparsePauliOp.from_sparse_list([(p, [w], 1.0)], num_qubits=n))
    return obs


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def sample(n_train, n_test, seed=0):
    Xtr, ytr, Xte, yte = load_nslkdd(N_QUBITS)
    Xq, yq = np.asarray(Xtr[:n_train], float), np.asarray(ytr[:n_train])
    rng = np.random.RandomState(seed)
    pos, neg = np.where(yte == 1)[0], np.where(yte == 0)[0]
    k = n_test // 2
    ite = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, n_test - k, replace=False)])
    rng.shuffle(ite)
    return Xq, yq, np.asarray(Xte[ite], float), np.asarray(yte[ite]), ite


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def exact_features(X):
    return pauli_features(statevectors(X, N_QUBITS, "iqp"), N_QUBITS)


def run_estimator(X, backend, shots, chunk=100, mode=None, log=print, max_seconds=None):
    """Return the (N, 24) matrix of measured expectations for rows X."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2
    qc, _ = iqp_circuit()
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa = pm.run(qc)
    obs = [o.apply_layout(isa.layout) for o in observables()]
    log(f"transpiled depth={isa.depth()} two-qubit gates="
        f"{sum(1 for inst in isa.data if inst.operation.num_qubits == 2)}")
    X = np.asarray(X, float)
    pubs = [(isa, obs, X[i:i + chunk].reshape(-1, 1, N_QUBITS)) for i in range(0, len(X), chunk)]
    est = EstimatorV2(mode=mode if mode is not None else backend)
    if max_seconds:
        est.options.max_execution_time = int(max_seconds)   # hard cap on QPU spend per job
    job = est.run(pubs, precision=1.0 / math.sqrt(shots))
    log(f"job {getattr(job, 'job_id', lambda: 'local')()} submitted: {len(pubs)} PUBs, "
        f"{len(X)} samples x 24 observables, {shots} shots each")
    res = job.result()
    F = np.vstack([np.asarray(pr.data.evs) for pr in res])
    assert F.shape == (len(X), 3 * N_QUBITS), F.shape
    return F, job


def evaluate(Ftr, Fte, yq, yte, Ftr_exact, Fte_exact):
    from sklearn.svm import SVC
    out = {}
    corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    out["feature_agreement"] = {
        "pearson_r_train": corr(Ftr, Ftr_exact), "pearson_r_test": corr(Fte, Fte_exact),
        "rmse_train": float(np.sqrt(np.mean((Ftr - Ftr_exact) ** 2))),
        "rmse_test": float(np.sqrt(np.mean((Fte - Fte_exact) ** 2)))}
    Khw, Kex = rbf(Ftr, Ftr, 1.0), rbf(Ftr_exact, Ftr_exact, 1.0)
    iu = np.triu_indices(len(Ftr), 1)
    out["gram_agreement"] = {"pearson_r_offdiag": corr(Khw[iu], Kex[iu]),
                             "max_abs_err": float(np.max(np.abs(Khw - Kex)))}
    for name, A, B in (("hardware_train_hardware_test", Ftr, Fte),
                       ("exact_train_exact_test", Ftr_exact, Fte_exact),
                       ("exact_train_hardware_test", Ftr_exact, Fte)):
        s = SVC(kernel="precomputed", C=1.0).fit(rbf(A, A, 1.0), yq).decision_function(rbf(B, A, 1.0))
        out[name] = score_metrics(yte, s)
    return out


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def validate(n=16):
    from qiskit.quantum_info import Statevector
    Xq, _, _, _, _ = sample(n, 2)
    qc, x = iqp_circuit()
    obs = observables()
    mine = exact_features(Xq)
    ref = np.array([[Statevector(qc.assign_parameters(dict(zip(x, row)))).expectation_value(o).real
                     for o in obs] for row in Xq])
    err = float(np.max(np.abs(ref - mine)))
    print(json.dumps({"n": n, "max_abs_err_qiskit_vs_pennylane": err}, indent=2))
    assert err < 1e-9, "Qiskit circuit does not match the PennyLane feature map"
    return {"max_abs_err": err}


def estimate(n_train, n_test, shots):
    n = n_train + n_test
    groups = 3                      # X-, Y-, Z-basis settings commute within each group
    total_shots = n * groups * shots
    return {"samples": n, "measurement_settings_per_sample": groups,
            "circuit_executions": n * groups, "shots_per_setting": shots,
            "total_shots": total_shots,
            "rough_qpu_seconds_at_1k_shots_per_s": total_shots / 1000,
            "rough_qpu_seconds_at_4k_shots_per_s": total_shots / 4000,
            "note": "Estimator may batch further; treat as an upper-bound order of magnitude"}


def dryrun(n_train, n_test, shots, fake="FakeFez"):
    import qiskit_ibm_runtime.fake_provider as fp
    backend = getattr(fp, fake)()
    Xq, yq, Xte, yte, _ = sample(n_train, n_test)
    Ftr_ex, Fte_ex = exact_features(Xq), exact_features(Xte)
    t0 = time.time()
    F, _ = run_estimator(np.vstack([Xq, Xte]), backend, shots)
    Ftr, Fte = F[:len(Xq)], F[len(Xq):]
    out = {"backend": f"{fake} (local noisy simulation)", "n_train": n_train, "n_test": n_test,
           "shots": shots, "wall_seconds": time.time() - t0,
           **evaluate(Ftr, Fte, yq, yte, Ftr_ex, Fte_ex)}
    return out


def run(backend_name, n_train, n_test, shots, confirm, account=None, max_seconds=240):
    if not confirm:
        raise SystemExit("refusing to submit to a real device without --confirm")
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(name=account) if account else QiskitRuntimeService()
    backend = (svc.backend(backend_name) if backend_name
               else svc.least_busy(simulator=False, operational=True, min_num_qubits=N_QUBITS))
    print("backend:", backend.name)
    Xq, yq, Xte, yte, ite = sample(n_train, n_test)
    Ftr_ex, Fte_ex = exact_features(Xq), exact_features(Xte)
    t0 = time.time()
    F, job = run_estimator(np.vstack([Xq, Xte]), backend, shots,
                           max_seconds=max_seconds)               # one job: overhead paid once
    Ftr, Fte = F[:len(Xq)], F[len(Xq):]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw = os.path.join(RESULTS, f"hardware_{backend.name}_{stamp}.npz")
    np.savez_compressed(raw, F_train=Ftr, F_test=Fte, F_train_exact=Ftr_ex, F_test_exact=Fte_ex,
                        y_train=yq, y_test=yte, test_index=ite, X_train=Xq, X_test=Xte)
    out = {"backend": backend.name, "job_ids": [job.job_id()],
           "n_train": n_train, "n_test": n_test, "shots": shots,
           "wall_seconds": time.time() - t0, "raw_file": os.path.basename(raw),
           **evaluate(Ftr, Fte, yq, yte, Ftr_ex, Fte_ex)}
    try:
        out["qpu_seconds"] = job.usage()
    except Exception:
        pass
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "estimate", "dryrun", "run"])
    ap.add_argument("--backend", default=None)
    ap.add_argument("--account", default=None)
    ap.add_argument("--fake", default="FakeFez")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--max-seconds", type=int, default=240,
                    help="max_execution_time for a real job (QPU seconds)")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if a.mode == "validate":
        out = validate()
    elif a.mode == "estimate":
        out = estimate(a.n_train, a.n_test, a.shots)
    elif a.mode == "dryrun":
        out = dryrun(a.n_train, a.n_test, a.shots, a.fake)
    else:
        out = run(a.backend, a.n_train, a.n_test, a.shots, a.confirm, a.account, a.max_seconds)
    print(json.dumps(out, indent=2, default=str))
    tag = a.mode if a.mode != "run" else f"run_{out['backend']}"
    json.dump(out, open(os.path.join(RESULTS, f"hardware_{tag}.json"), "w"), indent=2, default=str)
