"""Model factory: maps an ExperimentConfig to a trainable/eval-able model.

This is the integration seam between the harness and the modeling code owned by
quantum-dev (#4, quantum_model.py) and ai-engineer (#5, baselines.py). The runner
stays model-agnostic by treating every model through one of two shapes:

  * a torch ``nn.Module``  -> runner trains it with its mini-batch loop
  * an sklearn-style estimator exposing ``fit`` / ``predict`` and optionally
    ``predict_proba`` (or ``decision_function``)

``build_model(cfg)`` returns ``(model, kind)`` where ``kind`` is ``"torch"`` or
``"sklearn"``. We delegate construction to the owners' own factories rather than
reimplement constructors, so new model variants land here automatically:

  * quantum / param-matched MLP -> ``quantum_model.build_model(kind, in_dim, **kw)``
    (kinds: ``hybrid``, ``qsvm``, ``mlp``)
  * tuned classical / deep baselines -> ``baselines.py``
    (``build_torch_baseline`` for ``tuned_mlp``/``cnn1d``; ``_make_estimator`` +
    ``cfg.extra['best_params']`` for tuned RF/SVM/XGB)

Per-model hyperparameters arrive via ``cfg.extra`` so adding a model never changes
the config schema. The runner forwards ``cfg.n_classes`` (derived from the data)
through ``cfg.extra['n_classes']`` for multiclass runs.
"""
from __future__ import annotations
import inspect
from typing import Any

from config import ExperimentConfig


def _n_classes(cfg: ExperimentConfig, extra: dict) -> int:
    # Runner injects the true class count into extra for multiclass datasets.
    return int(extra.get("n_classes", 2 if cfg.binary else extra.get("n_classes", 2)))


def _build_quantum(cfg: ExperimentConfig, extra: dict):
    """Delegate to quantum-dev's quantum_model.build_model (task #4)."""
    import quantum_model as qm
    n_classes = _n_classes(cfg, extra)
    # Device preference (team-lead finding: CPU lightning.qubit ~10x faster than
    # GPU at our small qubit counts). 'cpu' (DEFAULT) | 'gpu'. Prefer a first-class
    # cfg.device field; fall back to extra['device']. Noisy runs are forced onto
    # the density-matrix device inside quantum_model regardless of this.
    device = getattr(cfg, "device", None) or extra.get("device", "cpu")
    # Forward device / noise only if quantum_model.build_model accepts them, so
    # older builds without these kwargs keep working.
    params = inspect.signature(qm.build_model).parameters
    has_varkw = any(p.kind == p.VAR_KEYWORD for p in params.values())

    def _maybe(kw: dict, key: str, value):
        if value is not None and (has_varkw or key in params):
            kw[key] = value

    if cfg.model_type == "hybrid_qnn":
        kw = dict(
            n_qubits=cfg.n_qubits, n_layers=cfg.n_layers,
            encoding=cfg.encoding, n_classes=n_classes,
            ansatz=extra.get("ansatz", "strongly_entangling"),
            rotation=extra.get("rotation", "Y"),
            diff_method=extra.get("diff_method", "adjoint"),
        )
        _maybe(kw, "device", device)
        _maybe(kw, "noise", extra.get("noise"))
        model = qm.build_model("hybrid", in_dim=cfg.n_features, **kw)
        return model, "torch"
    if cfg.model_type == "qsvm":
        kw = dict(
            n_qubits=cfg.n_qubits,
            encoding=extra.get("encoding", cfg.encoding if cfg.encoding != "amplitude" else "iqp"),
            kernel=extra.get("kernel", "fidelity"),
            n_repeats=extra.get("n_repeats", 2),
            gamma=extra.get("gamma", 1.0), C=extra.get("C", 1.0),
            rotation=extra.get("rotation", "Y"),
        )
        _maybe(kw, "device", device)
        _maybe(kw, "noise", extra.get("noise"))
        model = qm.build_model("qsvm", in_dim=cfg.n_features, **kw)
        return model, "sklearn"
    raise ValueError(f"unhandled quantum model_type {cfg.model_type!r}")


def _build_classical(cfg: ExperimentConfig, extra: dict):
    """Classical baselines.

    Param-matched MLP comes from quantum_model (the like-for-like comparison
    partner of HybridQNN). Tuned/deep baselines come from ai-engineer's
    baselines.py. RF / SVM / XGB accept pre-tuned hyperparameters via
    ``cfg.extra['best_params']`` (produced by run_baselines.py's offline search).
    """
    n_classes = _n_classes(cfg, extra)

    if cfg.model_type == "classical_mlp":
        import quantum_model as qm
        return qm.build_model("mlp", in_dim=cfg.n_features,
                              hidden=extra.get("hidden", 16),
                              n_classes=n_classes), "torch"

    # ---- Attribution-audit controls (quantum-dev #9) ---- #
    if cfg.model_type == "matched_mlp":
        # Classical MLP capacity-matched (by trainable-param COUNT) to the
        # HybridQNN with the same quantum hyperparams. The reference QNN is built
        # only to measure its parameter budget, then sized away.
        import quantum_model as qm
        from attribution import matched_mlp_for_qnn
        ref_qnn = qm.build_model(
            "hybrid", in_dim=cfg.n_features, n_qubits=cfg.n_qubits,
            n_layers=cfg.n_layers, encoding=cfg.encoding, n_classes=n_classes,
            ansatz=extra.get("ansatz", "strongly_entangling"))
        mlp, info = matched_mlp_for_qnn(ref_qnn, cfg.n_features, n_classes=n_classes,
                                        n_hidden_layers=extra.get("n_hidden_layers", 2))
        # stash the match info so the runner can log it into provenance.
        extra["matched_info"] = info
        return mlp, "torch"

    if cfg.model_type == "rf_kernel":
        # Random-feature kernel control: classical analogue of the QSVM.
        from attribution import RandomFeatureKernelSVM
        return RandomFeatureKernelSVM(
            method=extra.get("rf_method", "rbf_sampler"),
            n_components=extra.get("n_components", 256),
            gamma=extra.get("gamma", 1.0), C=extra.get("C", 1.0),
            seed=cfg.seed), "sklearn"

    if cfg.model_type in ("tuned_mlp", "cnn1d"):
        from baselines import build_torch_baseline
        name = "mlp" if cfg.model_type == "tuned_mlp" else "cnn1d"
        return build_torch_baseline(name, cfg.n_features, extra), "torch"

    if cfg.model_type in ("random_forest", "svm_rbf", "xgboost"):
        from baselines import _make_estimator
        # Cost-sensitive weighting unless cfg.imbalance says otherwise. Under
        # 'smote' the runner resamples the data, so no estimator-side reweighting.
        class_weight = "balanced" if cfg.imbalance == "class_weight" else None
        spw = extra.get("scale_pos_weight")  # XGB; runner injects from train y
        est = _make_estimator(cfg.model_type, cfg.seed, class_weight, spw)
        # Apply offline-tuned hyperparameters if the sweep config provides them.
        best = extra.get("best_params") or {}
        if best:
            est.set_params(**best)
        return est, "sklearn"

    raise ValueError(f"unhandled classical model_type {cfg.model_type!r}")


def build_model(cfg: ExperimentConfig):
    """Return ``(model, kind)`` for the given config.

    ``kind`` is ``"torch"`` (an nn.Module trained by the runner's loop) or
    ``"sklearn"`` (an estimator the runner calls fit/predict on).
    """
    extra: dict[str, Any] = dict(cfg.extra or {})
    if cfg.model_type in ("hybrid_qnn", "qsvm"):
        return _build_quantum(cfg, extra)
    return _build_classical(cfg, extra)
