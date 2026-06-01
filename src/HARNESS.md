# Reproducibility harness (task #6)

Reproducible experiment sweeps + results aggregation for the quantum-IDS paper.
Every run is fully specified by one YAML config and produces a self-describing
JSON result (config + environment snapshot + timestamp + metrics) plus a log.

## Files
- `config.py` — `ExperimentConfig` dataclass + YAML (de)serialization + `load_configs()` (single config or sweep).
- `seeding.py` — `seed_everything(seed)` seeds python/numpy/torch/cuda (covers PennyLane).
- `provenance.py` — `env_snapshot()`: platform, Python, lib versions, GPU, source md5s.
- `logging_utils.py` — `get_logger(run_slug=...)`: console + per-run file under `results/logs/`.
- `model_factory.py` — `build_model(cfg) -> (model, kind)`; the integration seam for new models.
- `runner.py` — `run_config(cfg)` / `run_file(yaml)`: load data, build, train, eval, write JSON.
- `aggregate.py` — scan `results/*.json` -> `summary.csv` (per-seed) + `summary_agg.csv`
  (per-group mean/std/CI) + `summary.tex`. Metric-agnostic: auto-discovers metric keys.
- `figures.py` — publication figures (task #7) -> `figures/`: comparison bar charts,
  ROC curves, ablation line plots (metric vs qubits/depth), confusion matrices. Reads
  the `predictions` block for ROC/confusion. `--metric f1`. (Noise-degradation figures:
  `noise_plots.py`, task #10.)
- `run_experiment.py` — original NSL-KDD comparison, now a thin wrapper over the harness.
- `configs/` — single config (`nslkdd_qnn.yaml`); campaign sweeps (task #7) with
  quantum-dev's budgets baked in: `sweep_main.yaml` (NSL-KDD full model comparison),
  `sweep_cross.yaml` (4 datasets x best encodings + baselines), `sweep_qubits.yaml`
  (q in {4,8,12} x depth ablation), `sweep_noise.yaml` (NISQ robustness).

## Config schema
See `config.ExperimentConfig`. Fields: `name, dataset, reduction, n_features, binary,
scale, imbalance, model_type, n_qubits, n_layers, encoding, device, seed, epochs,
batch_size, lr, q_subsample, extra`.
- `model_type` in `config.MODEL_TYPES`: `hybrid_qnn`, `qsvm`, `classical_mlp`
  (param-matched), `tuned_mlp`, `cnn1d`, `random_forest`, `svm_rbf`, `xgboost`.
- `dataset` in `config.DATASETS` (`nslkdd`/`unsw`/`cicids` + aliases); `reduction`
  in `config.REDUCTIONS` (`pca`/`autoencoder`/`none`).
- `imbalance` in `config.IMBALANCE` (`class_weight`/`smote`/`none`): the runner applies
  balanced class weights (sklearn `class_weight`/XGB `scale_pos_weight`/weighted
  CrossEntropy for torch) or SMOTE-resamples the TRAIN set (test untouched).
- `device` in `config.DEVICES` (`cpu`/`gpu`, DEFAULT `cpu`): quantum simulator
  preference, forwarded to `quantum_model.build_model`. `cpu` -> lightning.qubit
  (~10x faster at our qubit counts); `gpu` -> lightning.gpu. Noisy runs are forced
  onto `default.mixed` by `quantum_model` regardless. Ignored by classical models.
- `extra` (dict) carries per-model hyperparameters, forwarded to the model factory —
  add a model knob without touching the schema. Recognised keys include:
  `extra.noise: {type, p}` (NISQ sweep, task #10 — forwarded to the quantum builder),
  `extra.weight_decay` (L2 for torch models, task #9 reg sweep), `extra.best_params`
  (offline-tuned sklearn hyperparams), `extra.ansatz`/`rotation`/`kernel`/`n_repeats`.

## Metrics recorded (per run)
Via `evaluation.metrics` (ai-engineer #5/#8): accuracy, precision, recall, f1,
detection_rate, false_positive_rate, and — when probabilities are available —
roc_auc, auprc, tpr_at_0.1pct_fpr, tpr_at_1pct_fpr, brier, ece, plus train/eval time.
The aggregator picks up any new metric key automatically; `summary.tex` shows
mean$\pm$std over seeds with a paired-t significance marker (quantum vs best classical).

## Per-instance predictions (for McNemar / paired bootstrap)
Each run also persists predictions (toggle off with `runner.py --no-predictions`):
`record["predictions"] = {y_true, y_pred, y_score, y_score_is_proba}`. `y_score` is
P(attack) in [0,1] when `y_score_is_proba` (predict_proba / softmax), else a raw
decision score (QSVM/LinearSVC — rank metrics valid, Brier/ECE skipped downstream).
Test order is deterministic across models, so arrays align by index per (dataset, seed).
For test sets >100k rows (e.g. CICIDS2017) the arrays go to a compressed sidecar
`<slug>.preds.npz` and the JSON carries `{sidecar, y_score_is_proba, has_score, n}`.
- Unknown top-level YAML keys are stashed into `extra` (forward-compatible).
- A sweep file is either a top-level list of configs, or `{defaults: {...}, runs: [{...}]}`
  where each run is merged onto `defaults`.

## How to launch (always via the qml env runner)
```bash
# single config
bash run.sh runner.py --config configs/nslkdd_qnn.yaml
# a sweep (each config -> its own results JSON; one failure won't stop the sweep)
bash run.sh runner.py --config configs/sweep_qubits.yaml
# aggregate everything in results/ -> CSV + LaTeX
bash run.sh aggregate.py
# legacy NSL-KDD comparison (unchanged CLI)
bash run.sh run_experiment.py --qubits 8 --layers 3 --epochs 30 --encoding angle
```
(From Windows: `wsl -d Ubuntu-22.04 -- bash -lc "bash '/mnt/c/Research work 2/quantum-ids/src/run.sh' runner.py --config configs/nslkdd_qnn.yaml"`.)

OPERATIONAL NOTE (single 4GB-GPU / shared CPU box): do NOT co-schedule quantum runs
with the classical hyperparameter searches. RF/XGB `RandomizedSearchCV(n_jobs=-1)` and
`SVC(probability=True)` Platt CV spawn ~15 CPU workers that starve the per-sample QNN
loop (lightning.qubit) — a q4 epoch that takes <1s alone can stall for minutes under
that load. Run the quantum sweep on its own, or pin the classical search to a few cores
(e.g. `QIDS_N_JOBS=4`). In isolation a 20-epoch/8q/2000-sample QNN run is ~5 min.

## Integration (already wired to teammates' code)
- **data-engineer (#3):** runner calls `data.load(dataset, n_features, reduction,
  binary, scale, seed) -> (Xtr, ytr, Xte, yte)` (falls back to legacy `load_nslkdd`).
- **quantum-dev (#4):** `model_factory` delegates to `quantum_model.build_model(kind,
  in_dim, **kw)` for `hybrid_qnn` (kind `hybrid`), `qsvm`, and `classical_mlp` (kind `mlp`).
  Extra knobs (`ansatz`, `rotation`, `kernel`, `n_repeats`, ...) pass through `cfg.extra`.
- **ai-engineer (#5):** `model_factory` delegates to `baselines.py` —
  `build_torch_baseline` for `tuned_mlp`/`cnn1d`; `_make_estimator` (cost-sensitive
  `class_weight='balanced'` by default) for `random_forest`/`svm_rbf`/`xgboost`.
  Offline-tuned hyperparameters from `run_baselines.py` go into a config via
  `extra.best_params: {...}` and are applied with `set_params`.

To add a model: register a new `model_type` in `config.MODEL_TYPES` and a branch in
`model_factory.build_model`, returning `(model, "torch")` for an `nn.Module` or
`(model, "sklearn")` for a `fit/predict/predict_proba` estimator.

## Environment
Pinned in `../requirements.lock.txt` (`python -m pip freeze` inside `qml`). torch is the
CUDA 12.1 build (`pip install torch --index-url https://download.pytorch.org/whl/cu121`).
