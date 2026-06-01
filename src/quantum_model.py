"""Hybrid quantum-classical classifiers (PennyLane + PyTorch) for intrusion detection.

A variational quantum circuit (VQC) is wrapped as a torch layer (qml.qnn.TorchLayer)
and sandwiched between small classical layers. Runs on lightning.gpu when available.

This module provides a configurable model library for ablation sweeps:
  * Encodings: angle, amplitude, IQP/ZZFeatureMap-style, and data re-uploading.
  * Ansaetze: StronglyEntanglingLayers vs BasicEntanglerLayers, configurable depth.
  * A Quantum Kernel SVM (QSVM) using a fidelity / projected quantum kernel.
  * A frozen `QuantumConfig` dataclass so the experiment runner can sweep variants.

Backward compatibility: `make_device`, `build_qnode`, `HybridQNN`, and
`ClassicalMLP` keep their original signatures; the existing run_experiment.py keeps
working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Device handling
# --------------------------------------------------------------------------- #
DEVICES = ("cpu", "gpu")  # user-facing device preference (maps to a backend)


def make_device(n_qubits: int, noise: Optional[dict] = None,
                prefer: str = "cpu"):
    """Select a simulation device by preference.

    prefer:
      'cpu' (DEFAULT) -> lightning.qubit (CPU pure-state). Fastest at our small
            qubit counts: GPU kernel-launch overhead and the lack of TorchLayer
            batching make lightning.gpu ~10x SLOWER for <=~10 qubits (benchmarked:
            6q 1.6s vs 15.6s/epoch), so CPU is the default for the sweeps.
      'gpu'           -> lightning.gpu (cuQuantum pure-state); worth it only at
            larger qubit counts. Falls back to lightning.qubit if unavailable.

    A non-None `noise` spec OVERRIDES `prefer` and forces 'default.mixed' (CPU
    density-matrix); lightning.* cannot represent mixed states. default.mixed is
    memory-bound at ~ (2**n)**2, hence the MAX_NOISY_QUBITS cap.

    Returns (device, backend_name) where backend_name is the concrete PennyLane
    device string ('lightning.qubit' | 'lightning.gpu' | 'default.mixed').
    """
    if normalize_noise(noise) is not None:
        return qml.device("default.mixed", wires=n_qubits), "default.mixed"
    prefer = (prefer or "cpu").lower()
    if prefer == "cpu":
        return qml.device("lightning.qubit", wires=n_qubits), "lightning.qubit"
    if prefer == "gpu":
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits)
            _ = dev.wires  # touch so a missing cuQuantum/driver fails here
            return dev, "lightning.gpu"
        except Exception:
            return qml.device("lightning.qubit", wires=n_qubits), "lightning.qubit"
    raise ValueError(f"prefer must be one of {DEVICES}, got {prefer!r}")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ENCODINGS = ("angle", "amplitude", "iqp", "reupload")
ANSATZE = ("strongly_entangling", "basic_entangler")

# NISQ noise channels (PennyLane density-matrix simulation on 'default.mixed').
#   depolarizing    -> qml.DepolarizingChannel(p)         (gate noise)
#   amplitude_damp  -> qml.AmplitudeDamping(p)            (T1 / energy relaxation)
#   phase_damp      -> qml.PhaseDamping(p)                (T2 / dephasing)
#   bit_flip        -> qml.BitFlip(p)                     (readout / X error)
NOISE_TYPES = ("depolarizing", "amplitude_damp", "phase_damp", "bit_flip")

# Practical qubit ceiling for noisy (density-matrix) simulation on a 4GB GPU.
# default.mixed stores a full density matrix of 2**n x 2**n complex128 entries, so
# memory grows as (2**n)**2 -- e.g. n=12 -> 2**24 entries * 16 bytes ~= 256 MB just
# for the state, before autodiff tape overhead. 10-12 qubits is the realistic
# ceiling; pure-state (lightning) simulation is unaffected and can go higher.
MAX_NOISY_QUBITS = 12


def normalize_noise(noise) -> Optional[dict]:
    """Normalize a noise spec to {'type': str, 'p': float} or None (noiseless).

    Accepts None, {}, or a dict like {'type': 'depolarizing', 'p': 0.01}. A
    p<=0 or type 'none' disables noise (returns None) so a sweep can include a
    noiseless baseline by setting p=0.
    """
    if not noise:
        return None
    ntype = str(noise.get("type", "depolarizing")).lower()
    if ntype in ("none", "", "noiseless"):
        return None
    p = float(noise.get("p", 0.0))
    if p <= 0.0:
        return None
    if ntype not in NOISE_TYPES:
        raise ValueError(f"noise type must be one of {NOISE_TYPES}, got {ntype!r}")
    if not (0.0 < p <= 1.0):
        raise ValueError(f"noise strength p must be in (0, 1], got {p}")
    return {"type": ntype, "p": p}


@dataclass(frozen=True)
class QuantumConfig:
    """Hyperparameters for a quantum model variant (used for ablation sweeps).

    in_dim       : classical input feature dimension (== n_features from data.load).
    n_qubits     : number of qubits (4..16 for the 4GB-VRAM GPU).
    n_layers     : ansatz depth (also number of re-upload blocks for 'reupload').
    encoding     : one of ENCODINGS.
    ansatz       : one of ANSATZE.
    rotation     : single-qubit rotation axis for angle/re-upload encoding.
    n_classes    : output classes (2 for binary IDS).
    diff_method  : autodiff method for the QNode ('adjoint' is fast on lightning;
                   automatically switched to 'backprop' when noise is enabled,
                   since default.mixed does not support adjoint differentiation).
    device       : simulation device preference, 'cpu' (DEFAULT) or 'gpu'. CPU
                   (lightning.qubit) is ~10x faster than GPU at <=~10 qubits.
                   A non-None `noise` forces the density-matrix device regardless.
    noise        : NISQ noise spec {'type': one of NOISE_TYPES, 'p': float in (0,1]}
                   or None for ideal (pure-state) simulation. When set, the model
                   runs on PennyLane 'default.mixed' (density matrix) and a noise
                   channel is applied to every wire after each variational layer.
    """

    in_dim: int
    n_qubits: int = 8
    n_layers: int = 3
    encoding: str = "angle"
    ansatz: str = "strongly_entangling"
    rotation: str = "Y"
    n_classes: int = 2
    diff_method: str = "adjoint"
    device: str = "cpu"
    noise: Optional[dict] = None

    def __post_init__(self):
        if self.encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}, got {self.encoding!r}")
        if self.ansatz not in ANSATZE:
            raise ValueError(f"ansatz must be one of {ANSATZE}, got {self.ansatz!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        if not (1 <= self.n_qubits <= 24):
            raise ValueError(f"n_qubits out of sane range: {self.n_qubits}")
        # Normalize the noise spec in-place (frozen dataclass -> object.__setattr__).
        norm = normalize_noise(self.noise)
        object.__setattr__(self, "noise", norm)
        if norm is not None and self.n_qubits > MAX_NOISY_QUBITS:
            raise ValueError(
                f"noisy density-matrix simulation is capped at {MAX_NOISY_QUBITS} "
                f"qubits on the 4GB GPU (got n_qubits={self.n_qubits}); memory ~ "
                f"(2**n)**2. Reduce n_qubits or disable noise.")

    @property
    def is_noisy(self) -> bool:
        return self.noise is not None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Encoding primitives
# --------------------------------------------------------------------------- #
def zz_feature_map(inputs, n_qubits: int, n_repeats: int = 2):
    """IQP / ZZFeatureMap-style encoding, built manually so it broadcasts over a
    batch dimension (PennyLane's qml.IQPEmbedding does NOT support parameter
    broadcasting and silently produces wrong-shaped output under qml.qnn.TorchLayer).

    Per repetition: Hadamard on every wire, single-qubit RZ(2*x_i), then pairwise
    ZZ rotations RZ(2*(pi - x_i)(pi - x_j)) on neighbouring wires (Havlicek et al.
    2019 feature map). `inputs[..., i]` indexing keeps it batch-safe.
    """
    pi = np.pi
    for _ in range(n_repeats):
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
            qml.RZ(2.0 * inputs[..., w], wires=w)
        for w in range(n_qubits - 1):
            qml.CNOT(wires=[w, w + 1])
            qml.RZ(2.0 * (pi - inputs[..., w]) * (pi - inputs[..., w + 1]), wires=w + 1)
            qml.CNOT(wires=[w, w + 1])


def _amplitude_norm_guard(inputs, eps: float = 1e-8):
    """Return a UNIT-norm amplitude vector with a numerically STABLE gradient.

    The bug this fixes is NOT just a zero forward value -- it's a backward (grad)
    explosion. AmplitudeEmbedding(normalize=True) differentiates x/||x||, whose
    Jacobian carries 1/||x|| terms; once training nudges a sample's vector toward a
    small/degenerate norm, the gradient w.r.t. the classical pre-net blows up to inf
    and Adam turns the next forward into a NaN state vector (observed: pre.0.weight
    -> NaN at step ~2 of an amplitude q8 run).

    Fix: normalize HERE with a denominator floored at `eps` (||x|| -> max(||x||,eps),
    a bounded-gradient operation -- the same trick as torch.nn.functional.normalize),
    then feed the already-unit vector to AmplitudeEmbedding(normalize=False) so
    PennyLane never runs its own unstable normalize. Batch-safe (last axis),
    backend-agnostic via qml.math.
    """
    sq = qml.math.sum(inputs * inputs, axis=-1, keepdims=True)
    denom = qml.math.sqrt(sq + eps * eps)          # >= eps, so grad is bounded
    denom = qml.math.clip(denom, eps, None)        # belt-and-braces floor
    return inputs / denom


def _apply_encoding(inputs, n_qubits: int, encoding: str, rotation: str = "Y"):
    """Apply a single data-encoding block on `n_qubits` wires.

    `inputs` is a 1-D tensor of length n_qubits (angle/iqp/reupload) or 2**n_qubits
    -- or anything, padded -- for amplitude.
    """
    wires = range(n_qubits)
    if encoding == "amplitude":
        # Stable unit-normalization with a floored denominator (bounded gradient),
        # then PennyLane re-normalizes (cheap, vector already ~unit). The REAL
        # NaN-killer for the amplitude variant is using diff_method='backprop'
        # rather than 'adjoint' (see build_vqc_qnode): adjoint differentiates the
        # Mottonen state-prep, whose angle gradients are singular when the 256
        # amplitudes (rank<=in_dim, so many near-equal) collide -> NaN. backprop
        # through the statevector is stable here.
        inputs = _amplitude_norm_guard(inputs)
        qml.AmplitudeEmbedding(inputs, wires=wires, normalize=True, pad_with=0.0)
    elif encoding == "iqp":
        zz_feature_map(inputs, n_qubits, n_repeats=2)
    elif encoding in ("angle", "reupload"):
        qml.AngleEmbedding(inputs, wires=wires, rotation=rotation)
    else:  # pragma: no cover - guarded by QuantumConfig
        raise ValueError(f"unknown encoding {encoding!r}")


def _apply_ansatz(weights, n_qubits: int, ansatz: str):
    """Apply one variational ansatz block. `weights` shape must match the template."""
    wires = range(n_qubits)
    if ansatz == "strongly_entangling":
        qml.StronglyEntanglingLayers(weights, wires=wires)
    elif ansatz == "basic_entangler":
        qml.BasicEntanglerLayers(weights, wires=wires)
    else:  # pragma: no cover - guarded by QuantumConfig
        raise ValueError(f"unknown ansatz {ansatz!r}")


def _ansatz_weight_shape(ansatz: str, n_layers: int, n_qubits: int):
    if ansatz == "strongly_entangling":
        return qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
    if ansatz == "basic_entangler":
        return qml.BasicEntanglerLayers.shape(n_layers, n_qubits)
    raise ValueError(f"unknown ansatz {ansatz!r}")  # pragma: no cover


def _apply_noise(noise: Optional[dict], n_qubits: int):
    """Apply one NISQ noise channel to every wire (no-op if noise is None).

    Inserted after each variational layer so deeper circuits accumulate more
    noise -- the realistic NISQ regime. Requires a density-matrix device
    (default.mixed); these channels are unsupported on lightning.* state devices.
    """
    if noise is None:
        return
    ntype, p = noise["type"], noise["p"]
    for w in range(n_qubits):
        if ntype == "depolarizing":
            qml.DepolarizingChannel(p, wires=w)
        elif ntype == "amplitude_damp":
            qml.AmplitudeDamping(p, wires=w)
        elif ntype == "phase_damp":
            qml.PhaseDamping(p, wires=w)
        elif ntype == "bit_flip":
            qml.BitFlip(p, wires=w)


# --------------------------------------------------------------------------- #
# QNode builders
# --------------------------------------------------------------------------- #
def build_qnode(n_qubits: int, n_layers: int, dev, encoding: str = "angle"):
    """Backward-compatible builder: encoding + StronglyEntanglingLayers -> <Z_i>.

    Kept so existing callers (run_experiment.py, HybridQNN default) are unchanged.
    Returns (qnode, weight_shapes).
    """
    cfg = QuantumConfig(
        in_dim=n_qubits, n_qubits=n_qubits, n_layers=n_layers,
        encoding=encoding, ansatz="strongly_entangling",
    )
    return build_vqc_qnode(cfg, dev)


def build_vqc_qnode(cfg: QuantumConfig, dev):
    """Build a parametrised VQC QNode from a QuantumConfig.

    For encoding='reupload' the data is re-encoded before every ansatz layer
    (data re-uploading, Perez-Salinas et al. 2020), giving each layer its own
    encoding+variational block. For other encodings the data is encoded once,
    then `n_layers` of the chosen ansatz are applied.

    Returns (qnode, weight_shapes) where weight_shapes is the TorchLayer dict.
    """
    n_qubits, n_layers = cfg.n_qubits, cfg.n_layers
    encoding, ansatz, rotation = cfg.encoding, cfg.ansatz, cfg.rotation
    noise = cfg.noise

    # default.mixed does not support adjoint differentiation; use backprop when
    # noisy. Amplitude encoding also needs backprop (Mottonen state-prep gradients
    # are singular under adjoint -> NaN); HybridQNN puts it on default.qubit.
    if noise is not None or encoding == "amplitude":
        diff_method = "backprop"
    else:
        diff_method = cfg.diff_method

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def qnode(inputs, weights):
        if encoding == "reupload":
            # weights packs n_layers single-layer ansatz blocks; re-encode each time.
            for layer in range(n_layers):
                _apply_encoding(inputs, n_qubits, "angle", rotation)
                _apply_ansatz(weights[layer], n_qubits, ansatz)
                _apply_noise(noise, n_qubits)  # per-layer NISQ noise (no-op if None)
        elif noise is not None:
            # Apply the ansatz layer-by-layer so noise accumulates per depth, the
            # realistic NISQ regime. weights[layer] is one ansatz layer's params.
            _apply_encoding(inputs, n_qubits, encoding, rotation)
            _apply_noise(noise, n_qubits)  # encoding/state-prep noise
            for layer in range(n_layers):
                _apply_ansatz(weights[layer:layer + 1], n_qubits, ansatz)
                _apply_noise(noise, n_qubits)
        else:
            _apply_encoding(inputs, n_qubits, encoding, rotation)
            _apply_ansatz(weights, n_qubits, ansatz)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    if encoding == "reupload":
        # n_layers independent single-layer ansatz blocks: shape (n_layers, *single).
        single = _ansatz_weight_shape(ansatz, 1, n_qubits)
        weight_shapes = {"weights": (n_layers,) + tuple(single)}
    else:
        # The full-depth weight shape is identical whether we apply the template
        # once (ideal) or slice it layer-by-layer (noisy): (n_layers, ...).
        weight_shapes = {"weights": _ansatz_weight_shape(ansatz, n_layers, n_qubits)}
    return qnode, weight_shapes


# --------------------------------------------------------------------------- #
# Hybrid torch models
# --------------------------------------------------------------------------- #
class HybridQNN(nn.Module):
    """classical(in->q) -> VQC(q qubits) -> classical(q->n_classes). Binary classifier.

    Backward compatible: the original positional/keyword signature still works.
    Pass a QuantumConfig via `config=` to use the full variant library (IQP,
    re-uploading, BasicEntangler, configurable depth, etc.).
    """

    def __init__(self, in_dim: int, n_qubits: int = 8, n_layers: int = 3,
                 encoding: str = "angle", *, config: Optional[QuantumConfig] = None,
                 ansatz: str = "strongly_entangling", rotation: str = "Y",
                 n_classes: int = 2, noise: Optional[dict] = None,
                 device: str = "cpu"):
        super().__init__()
        if config is None:
            config = QuantumConfig(
                in_dim=in_dim, n_qubits=n_qubits, n_layers=n_layers,
                encoding=encoding, ansatz=ansatz, rotation=rotation,
                n_classes=n_classes, noise=noise, device=device,
            )
        self.config = config
        self.n_qubits = config.n_qubits
        # Amplitude encoding differentiates Mottonen state-prep, which is singular
        # under adjoint on lightning.qubit (NaN grads once amplitudes collide). Run
        # the amplitude variant on default.qubit + backprop, which is stable. Noisy
        # runs already force default.mixed+backprop and take precedence.
        if config.noise is None and config.encoding == "amplitude":
            self.dev = qml.device("default.qubit", wires=config.n_qubits)
            self.backend = "default.qubit"
        else:
            self.dev, self.backend = make_device(config.n_qubits, config.noise,
                                                 config.device)
        qnode, weight_shapes = build_vqc_qnode(config, self.dev)

        # The quantum input dim differs by encoding:
        #   amplitude -> 2**n_qubits amplitudes; everything else -> n_qubits angles.
        q_in = (2 ** config.n_qubits) if config.encoding == "amplitude" else config.n_qubits
        self.pre = nn.Sequential(nn.Linear(config.in_dim, q_in), nn.Tanh())
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.post = nn.Linear(config.n_qubits, config.n_classes)

    def forward(self, x):
        x = self.pre(x)
        x = self.qlayer(x)
        return self.post(x)


class ClassicalMLP(nn.Module):
    """Parameter-matched classical baseline (same overall shape, no quantum layer)."""

    def __init__(self, in_dim: int, hidden: int = 16, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Parallel test-set evaluation (task #15)
# --------------------------------------------------------------------------- #
# Test-set inference of a HybridQNN is embarrassingly parallel: every sample's
# circuit is independent. The cost is the per-sample QNode loop inside
# qml.qnn.TorchLayer on a single core. THREADS do NOT help -- lightning.qubit's
# forward does not release the GIL through TorchLayer (measured: ~0.4x). PROCESSES
# do: each worker rebuilds the model from (config, state_dict) and evaluates a
# chunk. Measured ~7x at 8-10 workers, predictions byte-identical to serial.
def _qnn_predict_chunk(payload):
    """Process-pool worker: rebuild a HybridQNN from (config_dict, state_dict) and
    return (proba_pos, pred) for one chunk. Module-level so it is picklable.

    Evaluates in mini-batches of WORKER_MINIBATCH rows so peak RAM per worker is
    bounded regardless of test-set size (critical on large datasets like UNSW 82k).
    """
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch as _t
    _t.set_num_threads(1)
    cfg_dict, state_dict, X_chunk, mb_size = payload
    cfg = QuantumConfig(**cfg_dict)
    model = HybridQNN(in_dim=cfg.in_dim, config=cfg)
    model.load_state_dict(state_dict)
    model.eval()
    all_proba, all_pred = [], []
    with _t.no_grad():
        for start in range(0, len(X_chunk), mb_size):
            batch = _t.as_tensor(X_chunk[start:start + mb_size], dtype=_t.float32)
            logits = model(batch)
            sm = _t.softmax(logits, dim=1)
            if logits.shape[1] == 2:
                all_proba.append(sm[:, 1].numpy())
            all_pred.append(logits.argmax(1).numpy())
    import numpy as _np
    pred = _np.concatenate(all_pred)
    proba = _np.concatenate(all_proba) if all_proba else None
    return proba, pred


def parallel_predict(model: "HybridQNN", X, workers: Optional[int] = None,
                     minibatch: int = 2048):
    """Evaluate a trained HybridQNN on X across multiple processes.

    Chunks X into `workers` pieces, evaluates each in its own process (rebuilt from
    the model's QuantumConfig + state_dict), and recombines IN ORIGINAL ORDER so
    pred[i]/proba[i] still align with X[i] (McNemar / compare.py depend on this).

    Memory bounds:
      - workers  (default 6): limits the number of concurrent HybridQNN copies in RAM.
        Lowered from 8 to 6 to leave headroom for large test sets on the shared laptop.
      - minibatch (default 2048): each worker processes its chunk in mini-batches so
        peak activation RAM inside a worker is O(minibatch) rather than O(chunk_size),
        preventing OOM on large datasets (e.g. UNSW 82k-row test set).

    Returns (pred, proba_pos) where proba_pos is P(class=1) for binary models else
    None -- matching the serial runner path. Falls back to a single-process serial
    forward when workers<=1, X is small, or anything goes wrong (so the campaign
    never breaks on the parallel path).
    """
    import os
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    if workers is None:
        # Default 6 (not all cores): limits concurrent model copies in RAM and leaves
        # headroom on the 10-core/16-thread i7 when phases overlap classical jobs.
        workers = min(6, os.cpu_count() or 1)
    workers = max(1, int(workers))

    # Serial fallback: tiny inputs or a single worker (process spawn isn't worth it).
    if workers == 1 or n < 256 or not isinstance(model, HybridQNN):
        model.eval()
        with torch.no_grad():
            logits = model(torch.as_tensor(X, dtype=torch.float32))
            sm = torch.softmax(logits, dim=1)
            proba = sm[:, 1].cpu().numpy() if logits.shape[1] == 2 else None
            pred = logits.argmax(1).cpu().numpy()
        return pred, proba

    from concurrent.futures import ProcessPoolExecutor
    cfg_dict = model.config.to_dict()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # np.array_split keeps order; we concatenate results in the same chunk order.
    chunks = np.array_split(X, workers)
    payloads = [(cfg_dict, state_dict, c, minibatch) for c in chunks if len(c)]
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_qnn_predict_chunk, payloads))
    except Exception:
        # Robust fallback to serial on any multiprocessing failure.
        return parallel_predict(model, X, workers=1)

    preds = np.concatenate([r[1] for r in results])
    if results and results[0][0] is not None:
        proba = np.concatenate([r[0] for r in results])
    else:
        proba = None
    return preds, proba


# --------------------------------------------------------------------------- #
# Quantum Kernel SVM (QSVM)
# --------------------------------------------------------------------------- #
class QuantumKernelSVM:
    """Quantum Kernel SVM using a fidelity or projected quantum kernel.

    Builds a feature-map circuit (angle / IQP / amplitude encoding) and computes a
    Gram matrix that is fed to a classical sklearn SVC with kernel='precomputed'.

    fidelity kernel : K(x,x') = |<phi(x')|phi(x)>|^2 via the adjoint feature map
                      (qml.kernels-style; here implemented with a probs QNode that
                      returns P(0...0) of U(x')^dagger U(x), i.e. the fidelity).
    projected kernel: K(x,x') = exp(-gamma * || rho(x) - rho(x') ||^2) where rho is
                      the vector of single-qubit <X>,<Y>,<Z> expectations
                      (Huang et al. 2021). More robust to high qubit counts.

    Cost is O(N^2) circuit evaluations for the fidelity kernel, so keep the train
    subsample small (a few hundred to ~1500 rows) on the 4GB GPU.
    """

    def __init__(self, n_qubits: int = 8, encoding: str = "iqp",
                 kernel: str = "fidelity", n_repeats: int = 2, gamma: float = 1.0,
                 C: float = 1.0, rotation: str = "Y", noise: Optional[dict] = None,
                 device: str = "cpu"):
        from sklearn.svm import SVC  # local import: keeps torch-only paths light

        if kernel not in ("fidelity", "projected"):
            raise ValueError("kernel must be 'fidelity' or 'projected'")
        if encoding not in ("angle", "iqp", "amplitude"):
            raise ValueError("QSVM encoding must be 'angle', 'iqp' or 'amplitude'")
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.kernel = kernel
        self.n_repeats = n_repeats
        self.gamma = gamma
        self.rotation = rotation
        self.noise = normalize_noise(noise)
        if self.noise is not None and n_qubits > MAX_NOISY_QUBITS:
            raise ValueError(
                f"noisy QSVM capped at {MAX_NOISY_QUBITS} qubits (density matrix); "
                f"got n_qubits={n_qubits}")
        self.dev, self.backend = make_device(n_qubits, self.noise, device)
        self.svc = SVC(kernel="precomputed", C=C)
        self._X_train = None
        self._build_circuits()

    # --- feature map ------------------------------------------------------- #
    def _unitary_feature_map(self, x):
        """The *unitary* embedding only (no noise) -- safe to take qml.adjoint of."""
        wires = range(self.n_qubits)
        if self.encoding == "amplitude":
            qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)
        elif self.encoding == "iqp":
            zz_feature_map(x, self.n_qubits, n_repeats=self.n_repeats)
        else:  # angle
            qml.AngleEmbedding(x, wires=wires, rotation=self.rotation)

    def _feature_map(self, x):
        """Unitary embedding followed by NISQ noise (no-op if noiseless)."""
        self._unitary_feature_map(x)
        _apply_noise(self.noise, self.n_qubits)

    def _build_circuits(self):
        n = self.n_qubits

        if self.kernel == "fidelity":
            # Fidelity = probability of measuring |0...0> for U(x2)^dagger U(x1).
            # Only the UNITARY embedding may be adjointed (a noise channel is CPTP,
            # not invertible). Under noise we apply: U(x1), noise, U(x2)^dagger,
            # noise -> a noisy-fidelity kernel estimate (valid NISQ study quantity).
            adjoint_unitary = qml.adjoint(self._unitary_feature_map)

            @qml.qnode(self.dev, interface="autograd")
            def fidelity_qnode(x1, x2):
                self._unitary_feature_map(x1)
                _apply_noise(self.noise, n)
                adjoint_unitary(x2)
                _apply_noise(self.noise, n)
                return qml.probs(wires=range(n))

            self._fidelity_qnode = fidelity_qnode
            self._proj_qnode = None
        else:
            # Projected kernel: per-qubit Pauli expectations -> classical RBF.
            @qml.qnode(self.dev, interface="autograd")
            def proj_qnode(x):
                self._feature_map(x)
                obs = []
                for w in range(n):
                    obs += [qml.expval(qml.PauliX(w)),
                            qml.expval(qml.PauliY(w)),
                            qml.expval(qml.PauliZ(w))]
                return obs

            self._proj_qnode = proj_qnode
            self._fidelity_qnode = None

    # --- kernel matrices --------------------------------------------------- #
    def _fidelity(self, x1, x2) -> float:
        probs = self._fidelity_qnode(x1, x2)
        return float(probs[0])  # P(|0...0>) == |<phi(x2)|phi(x1)>|^2

    def _kernel_matrix(self, A, B) -> np.ndarray:
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if self.kernel == "fidelity":
            K = np.zeros((len(A), len(B)), dtype=np.float64)
            same = A is B or (A.shape == B.shape and np.array_equal(A, B))
            for i in range(len(A)):
                jstart = i if same else 0
                for j in range(jstart, len(B)):
                    k = self._fidelity(A[i], B[j])
                    K[i, j] = k
                    if same:
                        K[j, i] = k
            return K
        # projected kernel: feature vectors then RBF over them
        FA = np.array([np.asarray(self._proj_qnode(a), dtype=np.float64).ravel()
                       for a in A])
        FB = np.array([np.asarray(self._proj_qnode(b), dtype=np.float64).ravel()
                       for b in B])
        # squared euclidean distance via (a-b)^2 = a^2 + b^2 - 2ab
        sq = (np.sum(FA ** 2, axis=1)[:, None]
              + np.sum(FB ** 2, axis=1)[None, :]
              - 2.0 * FA @ FB.T)
        np.maximum(sq, 0.0, out=sq)
        return np.exp(-self.gamma * sq)

    # --- sklearn-style API ------------------------------------------------- #
    def fit(self, X_train, y_train):
        self._X_train = np.asarray(X_train, dtype=np.float64)
        K = self._kernel_matrix(self._X_train, self._X_train)
        self.svc.fit(K, y_train)
        return self

    def _check_fitted(self):
        if self._X_train is None:
            raise RuntimeError("QuantumKernelSVM must be fit before predict/decision.")

    def predict(self, X):
        self._check_fitted()
        K = self._kernel_matrix(np.asarray(X, dtype=np.float64), self._X_train)
        return self.svc.predict(K)

    def decision_function(self, X):
        self._check_fitted()
        K = self._kernel_matrix(np.asarray(X, dtype=np.float64), self._X_train)
        return self.svc.decision_function(K)


# --------------------------------------------------------------------------- #
# Factory: clean hook for the experiment runner to sweep variants
# --------------------------------------------------------------------------- #
def build_model(kind: str, in_dim: int, **kw):
    """Single entry point for the experiment runner / ablation sweeps.

    kind='hybrid'  -> HybridQNN(config=QuantumConfig(...))  (torch nn.Module)
    kind='mlp'     -> ClassicalMLP                          (torch nn.Module)
    kind='qsvm'    -> QuantumKernelSVM                      (sklearn-style)

    Extra kwargs are forwarded to the relevant constructor / QuantumConfig so a
    sweep can do e.g.:
        build_model('hybrid', in_dim=8, n_qubits=8, n_layers=4,
                    encoding='reupload', ansatz='basic_entangler')
        build_model('qsvm', in_dim=8, n_qubits=8, encoding='iqp', kernel='fidelity')
    """
    kind = kind.lower()
    if kind == "mlp":
        hidden = kw.pop("hidden", 16)
        n_classes = kw.pop("n_classes", 2)
        return ClassicalMLP(in_dim, hidden=hidden, n_classes=n_classes)
    if kind == "hybrid":
        cfg = QuantumConfig(
            in_dim=in_dim,
            n_qubits=kw.pop("n_qubits", 8),
            n_layers=kw.pop("n_layers", 3),
            encoding=kw.pop("encoding", "angle"),
            ansatz=kw.pop("ansatz", "strongly_entangling"),
            rotation=kw.pop("rotation", "Y"),
            n_classes=kw.pop("n_classes", 2),
            diff_method=kw.pop("diff_method", "adjoint"),
            noise=kw.pop("noise", None),
            device=kw.pop("device", "cpu"),
        )
        return HybridQNN(in_dim=in_dim, config=cfg)
    if kind == "qsvm":
        return QuantumKernelSVM(
            n_qubits=kw.pop("n_qubits", 8),
            encoding=kw.pop("encoding", "iqp"),
            kernel=kw.pop("kernel", "fidelity"),
            n_repeats=kw.pop("n_repeats", 2),
            gamma=kw.pop("gamma", 1.0),
            C=kw.pop("C", 1.0),
            rotation=kw.pop("rotation", "Y"),
            noise=kw.pop("noise", None),
            device=kw.pop("device", "cpu"),
        )
    raise ValueError(f"unknown model kind {kind!r} (use 'hybrid', 'mlp', or 'qsvm')")


# Catalogue of variants the runner can sweep for ablations. Each entry is a
# (kind, kwargs) pair consumable by build_model(kind, in_dim, **kwargs).
def variant_catalogue(n_qubits: int = 8, n_layers: int = 3):
    """Return the default ablation grid of model variants (name -> (kind, kwargs))."""
    q, L = n_qubits, n_layers
    return {
        "MLP":                  ("mlp",    {}),
        "Hybrid-Angle-SE":      ("hybrid", {"n_qubits": q, "n_layers": L,
                                            "encoding": "angle", "ansatz": "strongly_entangling"}),
        "Hybrid-Angle-BE":      ("hybrid", {"n_qubits": q, "n_layers": L,
                                            "encoding": "angle", "ansatz": "basic_entangler"}),
        "Hybrid-Amplitude-SE":  ("hybrid", {"n_qubits": q, "n_layers": L,
                                            "encoding": "amplitude", "ansatz": "strongly_entangling"}),
        "Hybrid-IQP-SE":        ("hybrid", {"n_qubits": q, "n_layers": L,
                                            "encoding": "iqp", "ansatz": "strongly_entangling"}),
        "Hybrid-Reupload-SE":   ("hybrid", {"n_qubits": q, "n_layers": L,
                                            "encoding": "reupload", "ansatz": "strongly_entangling"}),
        "Hybrid-Reupload-BE":   ("hybrid", {"n_qubits": q, "n_layers": L,
                                            "encoding": "reupload", "ansatz": "basic_entangler"}),
        "QSVM-IQP-Fidelity":    ("qsvm",   {"n_qubits": q, "encoding": "iqp", "kernel": "fidelity"}),
        "QSVM-Angle-Projected": ("qsvm",   {"n_qubits": q, "encoding": "angle", "kernel": "projected"}),
    }


# Default NISQ noise grid for the robustness study (task #10). A sweep takes the
# cross product of {channel} x {strength} and re-runs a base hybrid model under
# each, comparing TPR@low-FPR / ECE vs the p=0 (noiseless) baseline.
NOISE_STRENGTHS = (0.0, 0.001, 0.005, 0.01, 0.05)


def noise_sweep_specs(types=NOISE_TYPES, strengths=NOISE_STRENGTHS):
    """Yield (label, noise_spec_or_None) over the channel x strength grid.

    p=0.0 yields a single ('noiseless', None) baseline (deduplicated across types).
    Each noise_spec is the {'type','p'} dict to pass as build_model(..., noise=spec)
    or harness cfg.extra['noise'].
    """
    specs = [("noiseless", None)]
    for t in types:
        for p in strengths:
            if p <= 0.0:
                continue
            specs.append((f"{t}@{p}", {"type": t, "p": p}))
    return specs


if __name__ == "__main__":
    # Quick smoke test of every variant on synthetic data.
    torch.manual_seed(0)
    np.random.seed(0)
    q = 6
    in_dim = q
    Xt = torch.rand(8, in_dim)
    Xnp = np.random.rand(20, in_dim) * np.pi
    y = (np.random.rand(20) > 0.5).astype(int)

    cat = variant_catalogue(n_qubits=q, n_layers=2)
    for name, (kind, kw) in cat.items():
        m = build_model(kind, in_dim, **kw)
        if kind == "qsvm":
            m.fit(Xnp[:12], y[:12])
            pred = m.predict(Xnp[12:])
            print(f"[OK] {name:<22} backend={m.backend} pred_shape={pred.shape}")
        else:
            out = m(Xt)
            be = getattr(m, "backend", "n/a")
            print(f"[OK] {name:<22} backend={be} out={tuple(out.shape)}")

    # ---- NISQ noise smoke test (task #10): density-matrix forward/backward ---- #
    print("-- noise (default.mixed) --")
    qn = 4  # keep small: density matrix is (2**n)**2
    Xn = torch.rand(6, qn)
    yn = torch.randint(0, 2, (6,))
    lossf = nn.CrossEntropyLoss()
    for ch in NOISE_TYPES:
        m = build_model("hybrid", qn, n_qubits=qn, n_layers=2,
                        encoding="angle", noise={"type": ch, "p": 0.02})
        out = m(Xn)
        loss = lossf(out, yn); loss.backward()
        gnorm = sum(p.grad.abs().sum().item() for p in m.parameters()
                    if p.grad is not None)
        print(f"[OK] noisy hybrid {ch:<15} backend={m.backend} "
              f"out={tuple(out.shape)} grad_norm={gnorm:.3f}")
    # noisy QSVM (projected kernel is the natural density-matrix kernel)
    mq = build_model("qsvm", qn, n_qubits=qn, encoding="angle",
                     kernel="projected", noise={"type": "depolarizing", "p": 0.02})
    mq.fit(Xnp[:12, :qn], y[:12]); pr = mq.predict(Xnp[12:, :qn])
    print(f"[OK] noisy QSVM-projected backend={mq.backend} pred_shape={pr.shape}")
    print("smoke test complete")
