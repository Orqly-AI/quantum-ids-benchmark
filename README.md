# How Quantum Is the Advantage?

**A fair, calibration- and noise-aware benchmark and attribution audit of quantum machine learning for network intrusion detection.**

Reproducible code, configs, seeds, and splits for the paper of the same name. We
benchmark hybrid variational quantum circuits (VQC) and quantum-kernel SVMs (QSVM)
against five honestly-tuned classical baselines across four standard NIDS datasets
(NSL-KDD, UNSW-NB15, CICIDS2017, NF-ToN-IoT-v2), with an equal-budget fairness
design, operating-point/calibration metrics, statistical significance, a NISQ noise
sweep, and a **quantum-attribution audit** that decomposes any measured gain into
its classical and genuinely-quantum components.

**Target venue:** Computers & Security (Elsevier).

## TL;DR — headline result

Tuned classical models (Random Forest, XGBoost) **match or exceed the quantum models
on aggregate detection on every dataset**, and the attribution audit traces the
apparent quantum gain to classical preprocessing/regularisation rather than quantum
effects. The one robust exception is operationally meaningful: a small 4-qubit hybrid
circuit **significantly out-detects the best classical baseline at the 1% false-positive
operating point** on the distribution-shifted NSL-KDD task (p = 0.005).

| Dataset | Best classical F1 (same-budget) | Best quantum F1 | Verdict |
|---|---|---|---|
| NSL-KDD | 0.782 (XGBoost) | 0.730 | classical ahead; quantum wins at 1%-FPR |
| UNSW-NB15 | 0.891 (MLP) | 0.836 | classical ahead |
| CICIDS2017 | 1.000 (RF) | 0.968 | saturated (weak discriminator) |
| NF-ToN-IoT-v2 | 0.991 (RF) | 0.949 | saturated (weak discriminator) |

Two practitioner findings: random-CV evaluation overstates novel-attack detection by
~20 F1 points on NSL-KDD (use distribution-shift splits); and CPU quantum simulation
is **8–14× faster than GPU** at these qubit counts. *The contribution is the honest,
attributable methodology — it stands whether quantum wins, ties, or loses.*

## Reproduce

```bash
# inside the WSL `qml` conda env (see Environment below)
bash src/run.sh runner.py --config configs/overnight_q_phase1.yaml   # quantum ablation
python src/run_baselines.py --dataset nslkdd --view both --seeds 42 43 44 45 46
python src/aggregate.py        # -> results/summary.csv, summary_agg.csv, summary.tex
python src/figures.py --out figures/
python src/attribution.py --config configs/attribution_audit.yaml --results-dir results
```

## Hardware
- HP Victus 15 — Intel i7-12650H, 32 GB RAM
- NVIDIA RTX 3050 Ti Laptop GPU (4 GB VRAM, Ampere CC 8.6)
- Runs in WSL2 (Ubuntu 22.04) with CUDA GPU passthrough

## Environment
- Miniconda env `qml` (Python 3.10)
- PyTorch (CUDA 12.x) — classical layers + GPU acceleration
- PennyLane + `lightning.gpu` (cuQuantum/custatevec) — quantum circuit simulation
- scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn

## Datasets (open benchmarks)
- NSL-KDD (official KDDTrain+/KDDTest+ split — measures novel-attack generalisation)
- UNSW-NB15 (official train/test split)
- CICIDS2017 (3-day MachineLearningCVE subset; full corpus configurable)
- NF-ToN-IoT-v2 (NetFlow IoT; 200k-row stratified cap, configurable)

### Data API (`src/data.py`)
Single entry point for all datasets:
```python
from data import load, load_meta
Xtr, ytr, Xte, yte = load("nslkdd", n_features=8, reduction="pca", binary=True)
# load_meta(...) also returns a dict with explained_variance, class_names, shapes, etc.
```
- `dataset_name`: `nslkdd` | `unsw` | `cicids` | `toniot` (aliases accepted)
- `reduction`: `pca` | `autoencoder` (classical MLP bottleneck) | `none`
- `binary`: `True` -> normal(0)/attack(1); `False` -> multiclass int labels
- `scale`: `minmax` (final range `[0, pi]`, angle encoding) | `standard`
- Returns `np.float32` `X` of shape `(N, n_features)` and `np.int64` `y`.

For the classical baselines, use the full (un-reduced) view:
```python
from data import load_full
Xtr, ytr, Xte, yte = load_full("unsw", binary=True, scale="standard")  # all dims
```
`reduction="none"` ignores `n_features` and keeps every encoded dimension.

Download the non-NSL datasets first (run inside WSL `qml` env):
```bash
bash src/download_unsw.sh      # canonical 175,341/82,332 train/test split
bash src/download_cicids.sh    # CICIDS2017 MachineLearningCVE -> data/cicids2017.csv
                               # (CICIDS_DAYS="Monday-WorkingHours ..." for a subset)
bash src/download_toniot.sh    # NF-ToN-IoT-v2 (NetFlow IoT) -> data/nf_toniot.csv
```
Cleaning per dataset: NaN/inf -> 0, drop leaky/identifier columns (row ids, flow
ids, IPs/ports, timestamps), one-hot categoricals with train/test column
alignment, MinMax-to-[0,1] **before** reduction (preserves variance over the
sparse one-hot space — 8-component PCA retains ~84% var on NSL-KDD vs ~27% when
StandardScaled first). Legacy `load_nslkdd` / `load_nslkdd_full` are preserved.
Smoke test: `python src/test_data.py` (datasets without local files are skipped).

## Method
1. **Preprocessing:** clean, encode categoricals, scale, balance (SMOTE optional)
2. **Dimensionality reduction:** classical autoencoder / PCA -> n features = n qubits
3. **Quantum models** (`src/quantum_model.py`):
   - Hybrid VQC classifier (`HybridQNN`): encodings `angle | amplitude | iqp
     (ZZFeatureMap) | reupload (data re-uploading)`; ansatz
     `strongly_entangling | basic_entangler` with configurable depth.
   - Quantum Kernel SVM (`QuantumKernelSVM`): `fidelity` or `projected` kernel.
   - One entry point `build_model(kind, in_dim, **kw)` (`kind` = `hybrid|mlp|qsvm`)
     and a `variant_catalogue()` ablation grid; configured per-run through
     `model_factory` from `ExperimentConfig.extra`.
4. **Classical baselines:** Random Forest, XGBoost, MLP, SVM (RBF)
5. **Ablations:** #qubits, encoding, ansatz/entanglement, #layers, NISQ noise.

### Device selection (performance)
`make_device(n_qubits, noise, prefer)` / `QuantumConfig.device` (carried as
`extra.device`, values `'cpu' | 'gpu'`) selects the simulator and **defaults to
`'cpu'` (lightning.qubit)**: at our small qubit counts (<=~10) CPU is ~10x faster
than GPU because GPU kernel-launch overhead dominates tiny statevectors and
`TorchLayer` does not batch onto the GPU. Use `'gpu'` (lightning.gpu) only at
larger qubit counts (it falls back to CPU if cuQuantum is unavailable). A non-None
`noise` spec overrides this and forces the density-matrix device.

### NISQ noise robustness (`extra.noise = {type, p}`)
Set a noise spec to run on PennyLane `default.mixed` (density matrix): channels
`depolarizing | amplitude_damp | phase_damp | bit_flip`, strength `p in (0,1]`,
applied to every wire after each variational layer (and on the QSVM feature
state). `p=0` / unset = ideal pure-state. A channel is inserted per layer so
deeper circuits accumulate more noise (the realistic NISQ regime).

> **Memory ceiling:** `default.mixed` stores a full 2^n x 2^n density matrix, so
> memory grows as **(2^n)^2**. On the 4GB GPU box this caps noisy simulation at
> **~10-12 qubits** (enforced: `MAX_NOISY_QUBITS = 12`); ideal pure-state runs are
> unaffected. Noisy sims run on CPU (`default.mixed` has no GPU backend here).

Plot degradation curves (TPR@low-FPR, ECE, ROC-AUC vs `p`, one line per channel)
from the result JSONs with `python src/noise_plots.py --results-dir results`.

## Metrics
Threshold: Accuracy, Precision, Recall, F1, Detection Rate (DR), False-Positive Rate (FPR).
Ranking / calibration / imbalance-aware: ROC-AUC, AUPRC, TPR@0.1%FPR, TPR@1%FPR, Brier, ECE.
Significance: McNemar (shared test set), paired t / Wilcoxon + Cohen's d (over seeds),
paired bootstrap (operating-point metrics). Training/inference time logged throughout.

## Layout
- `data/`      raw + processed datasets
- `src/`       pipeline code (preprocess, models, train, eval, harness)
- `results/`   per-run JSON records (config + metrics + provenance), summary tables
- `figures/`   publication figures (bar charts, ROC, confusion, ablations)
- `paper/`     manuscript, attribution audit, literature review, highlights, cover letter

## Citation

If you use this benchmark or harness, please cite the paper (see `CITATION.cff`):

> *How Quantum Is the Advantage? A Fair, Calibration- and Noise-Aware Benchmark and
> Attribution Audit of Quantum Machine Learning for Network Intrusion Detection.*

## License
MIT — see `LICENSE`.
