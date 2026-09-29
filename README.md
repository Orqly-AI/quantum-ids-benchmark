# Significance Is Not Attribution

*The quantum machine learning advantages that survive correction in network intrusion detection are classical.*

[![arXiv](https://img.shields.io/badge/arXiv-2608.18155-b31b1b.svg)](https://arxiv.org/abs/2608.18155)

**A fair, calibration- and noise-aware benchmark and attribution audit of quantum machine learning for network intrusion detection.**

Reproducible code, configs, seeds, and splits for the paper of the same name. We
benchmark hybrid variational quantum circuits (VQC) and quantum-kernel SVMs (QSVM)
against five honestly-tuned classical baselines across four standard NIDS datasets
(NSL-KDD, UNSW-NB15, CICIDS2017, NF-ToN-IoT-v2), with an equal-budget fairness
design, operating-point/calibration metrics, statistical significance, a NISQ noise
sweep, and a **quantum-attribution audit** that decomposes any measured gain into
its classical and genuinely-quantum components.

**Preprint:** [arXiv:2608.18155](https://arxiv.org/abs/2608.18155) (v1; v2 with the kernel- and model-level
analysis of Section 7 and the hardware runs is being prepared, together with a journal submission).

## TL;DR — headline result

Tuned classical models (Random Forest, XGBoost) **match or exceed the quantum models
on aggregate detection on every dataset**. The attribution audit left two quantum results that
survived multiple-testing correction; **neither survives attribution** (paper Section 7).

**1. The quantum-kernel advantage is an artefact of classical tuning under shift.** The audit's
strongest positive, a quantum-kernel SVM that out-ranked its classical surrogate under both FDR and
Holm correction, is set by the classical bandwidth selection: on identical data with exact kernels
and the published training rows, a grid capped at gamma <= 3 yields a spurious ROC-AUC advantage of
0.167 and a grid to gamma <= 30 leaves 0.007, but on 10 further random training samples (1,000 to
4,000 rows) CV is anticorrelated with the RBF kernel's test AUC on every one and the wide grid also
yields spurious leads of up to 0.142; at each kernel's best setting the quantum kernel never leads by
more than 0.010. scikit-learn's default `SVC()` lands in the same dip (test AUC 0.807 against 0.948
for the tuned quantum kernel). Cross-validation cannot see NSL-KDD's train-to-test shift. Test-set controls show the mechanism:
on held-out training rows (no shift) CV ranks the grid correctly (r >= +0.94) and the artefact vanishes;
removing the 17 novel attack types shrinks it to 0.063, reweighting to the training attack mix to 0.012.
Holding attack families out of training on UNSW-NB15, ToN-IoT and CICIDS2017 (26 runs) recreates the
failure, including a spurious 0.17 quantum lead on CICIDS2017, while at its best setting the quantum
kernel never leads by more than 0.003; a label-free MMD between training and unlabelled test traffic
tracks the failure (Spearman rho = -0.61). Leave-attack-types-out CV repairs selection on NSL-KDD but
not under induced shift, so kernel comparisons under shift should be reported under several selection
rules with the oracle bound. A classical kernel on the circuit's own phases reproduces the IQP quantum
kernel (geometric difference 1.4 to 2.4) on all four datasets.

**2. The 4-qubit hybrid's low-FPR edge is classical.** The hybrid detects more attacks than the tuned
Random Forest at 1% FPR on NSL-KDD (p = 0.005, BH q = 0.030). Classical twins that replace only its
quantum layer (`src/hybrid_attribution.py`) show why: the trained circuit is exactly an 81-term
trigonometric polynomial (error 2e-15), and a tanh layer of the same width reproduces the result on all
five seeds (0.511 vs 0.515, never significantly different). About half of the margin over the forest
is the input (the forest on the hybrid's 4 features closes half the gap); the rest is the neural model
class, and the gain lies entirely in novel attack types (0.25 vs at most 0.15 for the forest).

**Real hardware.** Both models ran on three IBM Heron processors (`ibm_fez`, `ibm_kingston`,
`ibm_marrakesh`; 12 jobs, 1,047 QPU-seconds). The projected kernel (200 train + 500 test, 128 shots per
basis): Gram r = 0.97 to 0.98, classifier within 0.01 ROC-AUC of exact (`src/kernel_hardware.py`). The
hybrid's quantum layer (6,000 class-balanced test rows per device, three disjoint subsets): expectation
r = 0.97 to 0.98, 1%-FPR detection unchanged (0.505 to 0.517 vs 0.513 exact), margin over the forest
+0.13 to +0.14 (p <= 0.012), and no difference from the classical tanh twin on the same rows
(`src/hybrid_hardware.py`, `src/hybrid_hardware_pool.py`; raw expectations in
`results/kernel_analysis/*hardware_*.npz`).

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

# kernel-level analysis (paper Section 7): each part writes results/kernel_analysis/<part>.json
python src/kernel_analysis.py validate      # engine vs the QSVM circuits (agrees to 1e-15)
python src/kernel_analysis.py fair          # surrogate confounds removed, equal tuning
python src/kernel_analysis.py landscape     # bandwidth grid, standard vs shift-aware CV
python src/kernel_analysis.py landscape_nsl_seen      # test-set shift controls: novel types removed,
python src/kernel_analysis.py landscape_nsl_seen_rw   #   ... and attack mix reweighted,
python src/kernel_analysis.py landscape_nsl_iid       #   ... and held-out training rows (no shift)
python src/nsl_shift_stats.py              # shift statistics of the official split (Figure 6)
python src/kernel_analysis.py landscape_sample --n-train 2000 --seed 1   # random training samples
python src/default_svc.py                  # scikit-learn's default SVC() on the same protocol
python src/kernel_analysis.py induced --ds unsw --holdout Generic --seed 0   # one induced-shift run
python src/induced_summary.py               # all induced_<ds>_<holdout>_s<seed>.json -> induced_summary.json
python src/kernel_analysis.py landscape --folds balanced   # alternative shift-aware folds -> *_bal.json
python src/kernel_analysis.py geometry_trig # geometric difference, all four datasets
python src/kernel_analysis.py shots         # finite-shot robustness
python src/kernel_hardware.py dryrun        # noisy simulation of a Heron device (free)
python src/kernel_hardware.py run --confirm # real IBM Quantum device (needs an account; costs QPU time)

# the 4-qubit hybrid: classical twins (results/hybrid_attribution/) and hardware
python src/hybrid_attribution.py train --model fourier --seed 42   # also mlp, linear, hybrid; seeds 42-46
python src/hybrid_attribution.py rf --model rf_f4_full --seed 42   # the forest on the hybrid's input
python src/hybrid_attribution.py certificate --seed 42             # exact 81-term expansion of the circuit
python src/hybrid_attribution.py summary
python src/hybrid_hardware.py train         # retrain the published 4-qubit hybrid, keep its weights
python src/hybrid_hardware.py validate      # Qiskit circuit vs PennyLane (agrees to 1e-15)
python src/hybrid_hardware.py dryrun        # its quantum layer on a noisy Heron simulation (free)
python src/hybrid_hardware.py run --confirm --backend ibm_fez --seed 1 --rf results/<rf result>.json
python src/hybrid_hardware_pool.py          # pool the disjoint subsets per device
cd paper && bash _build_all.sh              # tables and figures from JSON, then every PDF and package
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

> **Availability.** We do not redistribute the source corpora: fetch them from their
> original providers with the `src/download_*.sh` scripts below, which together with
> the fixed seeds reproduce our exact splits. The **preprocessed datasets will be
> released on request** (open an issue or contact the authors). Per-run result records
> are already in `results/`.

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

If you use this benchmark or harness, please cite the paper
([arXiv:2608.18155](https://arxiv.org/abs/2608.18155); see also `CITATION.cff`):

```bibtex
@misc{gillani2026quantumadvantagefaircalibration,
      title={Significance Is Not Attribution: The Quantum Machine Learning Advantages That Survive Correction in Network Intrusion Detection Are Classical},
      author={Syeda Anshrah Gillani and Mirza Samad Ahmed Baig and Shahid Munir Shah and Abdul Akbar Khan and Muhammad Omer Khan and Asher Ali and Hamzah Siddiqui},
      year={2026},
      eprint={2608.18155},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2608.18155},
}
```

## License
Orqly-AI Restricted Attribution License (ORAL) v1.0 — see `LICENSE`.

Original Work by Mirza Samad Ahmed Baig and Syeda Anshrah Gillani (Orqly-AI),
Copyright (c) 2026. Source: https://github.com/Orqly-AI/quantum-ids-benchmark
Academic, non-commercial use is permitted with attribution and citation of the
associated research paper; commercial use and reproduction of the paper require
prior written permission from the authors.
