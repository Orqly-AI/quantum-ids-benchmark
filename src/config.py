"""Experiment configuration: a typed dataclass with YAML (de)serialization.

One ExperimentConfig fully specifies a single reproducible run: which dataset
and preprocessing, which model and its hyperparameters, the random seed, and the
training budget. Sweeps are just lists of these configs (see configs/ and
sweep.py). The runner (runner.py) records the resolved config verbatim into the
results JSON so every run is self-describing.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any

import yaml


# Models the harness knows how to build. The factory in model_factory.py maps
# each of these to a concrete estimator / nn.Module. Quantum-dev and ai-engineer
# extend the factory; the names here are the public contract used in YAML.
MODEL_TYPES = (
    "hybrid_qnn", "qsvm",                                    # quantum (quantum_model.py)
    "classical_mlp",                                         # param-matched MLP (quantum_model.py)
    "tuned_mlp", "cnn1d",                                    # deep baselines (baselines.py)
    "random_forest", "svm_rbf", "xgboost",                   # tuned sklearn baselines (baselines.py)
    "matched_mlp", "rf_kernel",                              # attribution-audit controls (attribution.py, #9)
)
# Dataset keys accepted by data.load() (data-engineer #3 aliases nslkdd/unsw/cicids).
DATASETS = ("nslkdd", "unsw", "cicids", "unsw_nb15", "cicids2017", "cic_iot2023",
            "toniot", "ton_iot", "nf-toniot", "iot")
# VQC encodings (quantum_model.ENCODINGS): angle/amplitude/iqp/reupload. 'reupload'
# = data re-uploading (re-encodes data before each ansatz layer). QSVM accepts
# angle/iqp/amplitude (passed via extra.encoding).
ENCODINGS = ("angle", "amplitude", "iqp", "reupload")
SCALES = ("minmax", "standard")
REDUCTIONS = ("pca", "autoencoder", "none")
# Class-imbalance handling: cost-sensitive weights, SMOTE oversampling, or nothing.
IMBALANCE = ("class_weight", "smote", "none")
# Quantum simulator device preference (quantum_model.build_model contract).
DEVICES = ("cpu", "gpu")


@dataclass
class ExperimentConfig:
    """A single experiment specification.

    `extra` is a free-form dict forwarded to the model factory for
    model-specific knobs (e.g. RF n_estimators, QSVM kernel params) so the
    schema stays stable as new models are added.
    """
    name: str = "experiment"
    # --- data / preprocessing ---
    dataset: str = "nslkdd"
    n_features: int = 8           # PCA/AE target dim; = n_qubits for quantum models
    reduction: str = "pca"        # dimensionality reduction: pca | autoencoder | none
    binary: bool = True           # normal(0) vs attack(1); else multiclass
    scale: str = "minmax"         # feature scaling before the model
    imbalance: str = "class_weight"  # class_weight | smote | none (cost-sensitive default)
    # --- model ---
    model_type: str = "hybrid_qnn"
    n_qubits: int = 8
    n_layers: int = 3
    encoding: str = "angle"
    # Quantum simulator device preference, forwarded to quantum_model.build_model.
    # 'cpu' (DEFAULT -> lightning.qubit) is ~10x faster than 'gpu' (lightning.gpu)
    # at our small qubit counts (measured). Ignored by classical models; noisy runs
    # are forced onto the density-matrix device by quantum_model regardless.
    device: str = "cpu"
    # --- training / reproducibility ---
    seed: int = 42
    epochs: int = 30
    batch_size: int = 64
    lr: float = 5e-3
    q_subsample: int = 8000       # train subsample for slow models (0 = use full set)
    # --- escape hatch for per-model hyperparameters ---
    extra: dict[str, Any] = field(default_factory=dict)

    # ---------- validation ----------
    def validate(self) -> "ExperimentConfig":
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset={self.dataset!r} not in {DATASETS}")
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type={self.model_type!r} not in {MODEL_TYPES}")
        if self.encoding not in ENCODINGS:
            raise ValueError(f"encoding={self.encoding!r} not in {ENCODINGS}")
        if self.scale not in SCALES:
            raise ValueError(f"scale={self.scale!r} not in {SCALES}")
        if self.reduction not in REDUCTIONS:
            raise ValueError(f"reduction={self.reduction!r} not in {REDUCTIONS}")
        if self.imbalance not in IMBALANCE:
            raise ValueError(f"imbalance={self.imbalance!r} not in {IMBALANCE}")
        if self.device not in DEVICES:
            raise ValueError(f"device={self.device!r} not in {DEVICES}")
        if self.n_features <= 0 or self.n_qubits <= 0 or self.n_layers <= 0:
            raise ValueError("n_features, n_qubits, n_layers must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        return self

    # ---------- (de)serialization ----------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        """Build from a dict, ignoring unknown keys (forward-compatible)."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            # Stash unknown top-level keys into `extra` rather than crashing, so
            # teammates can add fields incrementally without breaking old runs.
            extra = dict(d.get("extra", {}))
            for k in unknown:
                extra[k] = d[k]
            d = {k: v for k, v in d.items() if k in known}
            d["extra"] = extra
        return cls(**d).validate()

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def slug(self) -> str:
        """Filesystem-safe identifier used for result filenames.

        Includes the run `name` (when set to something other than the default) so
        that two runs sharing the same model_type/qubits/layers/encoding/seed but
        differing only in `extra` (e.g. the attribution audit's matched_mlp L2
        sweep, or rf_kernel feature-map variants) write to DISTINCT files instead
        of overwriting each other. aggregate.py groups by config.model_type from
        the JSON body, not the filename, so this is safe for aggregation.
        """
        base = (f"{self.dataset}_{self.model_type}_q{self.n_qubits}"
                f"_l{self.n_layers}_{self.encoding}_s{self.seed}")
        if self.name and self.name != "experiment":
            safe = "".join(c if (c.isalnum() or c in "-_.") else "-"
                           for c in self.name)
            base = f"{base}_{safe}"
        return base


def load_configs(path: str) -> list[ExperimentConfig]:
    """Load one config or a sweep from a YAML file.

    Accepts either a single mapping (one config) or a top-level list of
    mappings (a sweep). Also supports a ``{defaults: {...}, runs: [{...}]}``
    form where each run is merged onto the shared defaults.

    A run (or the defaults) may specify a ``seeds: [42, 43, ...]`` list instead of
    a single ``seed:``; it is expanded into one ExperimentConfig per seed (each
    gets a distinct results file via slug()). This is what makes multi-seed
    significance testing (paired t-test / Wilcoxon over seeds) a one-line config.
    """
    with open(path, "r") as f:
        doc = yaml.safe_load(f)
    if doc is None:
        return []

    def _expand(d: dict) -> list[ExperimentConfig]:
        # A 'seeds' list (in the run or merged from defaults) -> one config/seed.
        seeds = d.pop("seeds", None)
        if seeds is None:
            return [ExperimentConfig.from_dict(d)]
        return [ExperimentConfig.from_dict({**d, "seed": int(s)}) for s in seeds]

    if isinstance(doc, list):
        out = []
        for d in doc:
            out += _expand(dict(d))
        return out
    if isinstance(doc, dict) and "runs" in doc:
        defaults = doc.get("defaults", {}) or {}
        out = []
        for run in doc["runs"]:
            merged = {**defaults, **(run or {})}
            out += _expand(merged)
        return out
    return _expand(dict(doc))
