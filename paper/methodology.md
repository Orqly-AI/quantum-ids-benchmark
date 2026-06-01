# Methodology: Classical Baselines & Evaluation Protocol

Owner: ai-engineer (task #5). This documents the classical-ML rigor and the
evaluation protocol used to compare the hybrid quantum models against classical
baselines. The aim is a comparison a Q1 reviewer (Computers & Security /
Information Sciences / KBS / ESWA / EAAI) will accept as fair.

Source modules: `src/baselines.py` (tuned models + imbalance handling),
`src/evaluation.py` (metrics, multi-seed aggregation, significance tests),
`src/run_baselines.py` (tuning + multi-seed driver). Data via `src/data.py`
`load_meta(...)` (data-engineer, task #3); seeding via
`src/seeding.py::seed_everything` (sw-engineer, task #6).

---

## 1. Datasets and the train/test distribution-shift question

Primary benchmark: **NSL-KDD** with the *official* `KDDTrain+` / `KDDTest+`
split (125,973 train / 22,544 test rows). We deliberately do **not** re-shuffle
and re-split this dataset. The canonical split is the entire reason NSL-KDD
exists as a benchmark: the test set was constructed to contain attack behaviour
the training set under-represents, so honouring it measures *generalisation to
novel attacks* rather than memorisation.

Measured shift on the binary (normal vs attack) task:

| Quantity | Train | Test |
|---|---|---|
| Attack fraction | 0.4654 | 0.5692 |
| DoS | 0.3646 | 0.3309 |
| Probe | 0.0925 | 0.1074 |
| **R2L** | **0.0079** | **0.1280** |
| U2R | 0.0004 | 0.0030 |

The R2L (remote-to-local) class jumps from <1% of training rows to ~13% of test
rows, and **17 attack signatures appear only in the test set** (e.g. `apache2`,
`mscan`, `processtable`, `saint`, `snmpguess`, `worm`, `httptunnel`). This
covariate + prior shift is why absolute accuracies on `KDDTest+` are far below
the near-perfect numbers seen under random cross-validation, and why we report
this gap explicitly rather than hiding it behind a re-split.

**Implication for the methodology:** because the test distribution differs from
train, all model selection (hyperparameter search, early stopping, threshold and
imbalance decisions) is performed using cross-validation *within the training
set only*. The official test set is touched exactly once per model per seed, for
final scoring. No information from `KDDTest+` enters preprocessing, scaling,
feature reduction, or tuning.

Secondary datasets (UNSW-NB15, CICIDS2017) use the same protocol via the unified
`data.load_meta` API. UNSW-NB15 has an official train/test split (honoured);
CICIDS2017 has none, so we use a single stratified 70/30 split with a fixed seed.

## 2. Preprocessing and leakage control

Handled by `data.py` (data-engineer); the properties this methodology relies on:

- Categorical features (`protocol_type`, `service`, `flag` for NSL-KDD) are
  one-hot encoded; train/test columns are **aligned** so categories unseen in
  train are dropped and train-only categories are zero-filled in test. This
  yields a 122-dim encoded NSL-KDD feature space.
- All scalers / PCA / autoencoder reducers are **fit on train only** and applied
  to test — no leakage.
- Two feature views are evaluated (see §3).

## 3. Two feature views — why both are reported

A central fairness issue in QML-for-IDS papers is that the quantum model is fed
a heavily reduced feature set (one feature per qubit, 8 on our 4 GB GPU) while
"classical baselines" are silently given the same crippled input — or, worse,
the full feature set, with the asymmetry unstated. We report **both** views and
label them:

1. **Full view** (`reduction="none"`): classical baselines on all 122 encoded
   dimensions. These are the *headline* classical numbers — the strongest the
   classical models can do, the bar the quantum approach must be discussed
   against.
2. **Same-budget view** (`reduction="pca"`, `n_features = n_qubits = 8`):
   classical baselines on the *identical* 8-dimensional input the quantum models
   receive. This is the only apples-to-apples comparison for claims of a quantum
   advantage at a fixed feature/qubit budget.

Reporting both lets the paper make an honest claim: e.g. "at an equal 8-feature
budget the hybrid QNN is competitive with / superior to the classical models,
while full-dimensional classical models remain the practical upper bound."

## 4. Classical baselines (all tuned, none default)

Five baselines, each hyperparameter-searched — not run at library defaults:

| Model | Search | Space (abridged) |
|---|---|---|
| Random Forest | Randomized CV, 40 iters | n_estimators, max_depth, max_features, min_samples_{split,leaf} |
| XGBoost (hist) | Randomized CV, 40 iters | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma |
| SVM (RBF) | Randomized CV, 40 iters | C, gamma (search on a stratified ≤12k subsample; final fit capped at 20k rows) |
| MLP (torch) | Grid (4 configs) on a stratified val split | hidden sizes, dropout, lr |
| 1D-CNN (torch) | Grid (4 configs) on a stratified val split | conv channels, kernel size, dropout, lr |

Search uses **5-fold stratified** cross-validation on the training set, optimising
**F1** (the operative metric under class imbalance; accuracy is a poor selection
criterion here). The search is run once on a reference seed; the chosen
configuration is then **refit from scratch on every seed** for the final eval.
The 1D-CNN is included so the quantum model is benchmarked against a modern deep
model, not only shallow learners (a common reviewer complaint).

Two efficiency caps (stated for full disclosure; neither touches the test set):
the hyperparameter **search** runs on a stratified subsample of the training set
(≤30k rows for the tree/boosting models, ≤12k for the RBF-SVM) — sufficient for
stable hyperparameter rankings — while the **final** RF/XGBoost models are refit
on the *full* training set; only the RBF-SVM final fit is capped at a stratified
20k-row subsample, because its O(n²–n³) training with Platt-scaled probabilities is
infeasible on ~126k rows on the laptop. These caps are recorded in each results
JSON. To keep the shared machine usable while quantum jobs run concurrently,
estimator/search parallelism is bounded (env `QIDS_N_JOBS`, default 4) rather than
using all cores.

The tiny parameter-matched `ClassicalMLP` in `quantum_model.py` is a *separate*
control (same shape as the HybridQNN minus the quantum layer); the tuned `TunedMLP`
/ `CNN1D` in `baselines.py` are the genuine deep baselines. The quantum-attribution
audit (task #9) additionally builds an *exact-parameter-count*-matched MLP and a
random-feature-kernel control; that audit reuses `baselines._make_estimator`,
`fit_tuned_sklearn`, and `evaluation` directly so the metric/training code is shared.

## 5. Class-imbalance strategy: cost-sensitive learning (default), SMOTE (ablation)

NSL-KDD's *binary* training set is only mildly imbalanced (46.5% attack), so
aggressive resampling is neither necessary nor obviously beneficial — and the
test set is actually attack-majority (56.9%). We therefore **default to
cost-sensitive learning** and treat SMOTE as an ablation:

- RF / SVM: `class_weight="balanced"`.
- XGBoost: `scale_pos_weight = n_neg / n_pos`.
- Torch MLP / CNN: class-weighted cross-entropy (weights = sklearn "balanced").
- **SMOTE** (`imbalance="smote"`): synthetic minority oversampling applied to the
  **training set only**, never to validation or test, so synthetic neighbours
  cannot leak across the split.

Justification (stated in the paper): (a) the binary task is near-balanced, so
SMOTE's main benefit (minority recall) is small while it can distort the decision
boundary and inflate FPR; (b) cost-sensitive learning leaves the data
distribution intact and is the more defensible default for shifted test data;
(c) for the multiclass task (rare R2L/U2R), SMOTE matters more and the ablation
quantifies it. We report the default; the SMOTE ablation is available via a flag
and reported in the appendix.

## 6. Metrics

### 6.1 Threshold + ranking metrics
Per the project metric set (`evaluation.metrics`): Accuracy, Precision, Recall,
F1, **Detection Rate (DR = recall on the attack class = TP/(TP+FN))**,
**False Positive Rate (FPR = FP/(FP+TN))**, and ROC-AUC (from probability /
decision scores). DR and FPR are the operationally meaningful IDS metrics and are
always reported together. Training time is logged for the efficiency discussion
(quantum simulation is far slower — a fact the paper states plainly).

### 6.2 Calibration & imbalance-aware metrics (every model that emits probabilities)
Reviewers expect more than ROC-AUC for an imbalanced, safety-relevant task. From
the predicted P(attack) (sklearn `predict_proba` / torch `softmax`) we additionally
report (`evaluation.probability_metrics`):

- **AUPRC** (average precision) — area under the precision-recall curve; more
  informative than ROC-AUC under class imbalance and when the positive (attack)
  operating region is what matters.
- **TPR @ 0.1% FPR** and **TPR @ 1% FPR** — detection rate at fixed, *low*
  false-alarm budgets (interpolated on the ROC curve). The numbers an IDS operator
  actually cares about: at a tolerable alarm rate, what share of attacks is caught?
  A model can have high ROC-AUC yet poor TPR at a 0.1% FPR point, so we surface both.
- **Brier score** — MSE of the probability forecast (lower = better-calibrated + sharper).
- **ECE** (Expected Calibration Error, 15 equal-width bins) — mean gap between
  predicted confidence and empirical accuracy; tests whether the probabilities can
  be trusted as alarm scores.

These match/exceed the Meta-Quantum-Ensemble metric set and let the paper argue
about *calibration* and *low-FPR detection*, not only headline accuracy. All are
carried in every run's results JSON so the aggregator can emit them as columns.

## 7. Repetition: mean ± std over ≥ 5 seeds

Every final model is trained and evaluated over **5 seeds** (default
{42,43,44,45,46}; extendable). All RNGs are pinned per seed via
`seeding.seed_everything` (python / numpy / torch / CUDA, deterministic cuDNN).
We report **mean ± standard deviation** for each metric, plus a t-based **95%
confidence interval** (`evaluation.aggregate_seeds`, sample std with ddof=1 —
correct for the small-sample regime). Per-seed raw values are persisted in the
results JSON so any interval can be recomputed.

For the sklearn models the seed varies the bootstrap/feature subsampling and CV
folds; for the torch models it varies weight init and minibatch order. The same
seed list is shared across classical and quantum runs so per-seed scores are
**paired** for the significance test in §8.

**Generalisation-gap reporting.** For every model we record, per dataset, the
*cross-validation* F1 used for model selection (on training folds) alongside the
mean *test* F1 on the held-out official test set, and report the difference
(`cv_vs_test_gap` in the results JSON). On NSL-KDD this gap is large (tuned
RandomForest: CV F1 ≈ 0.98 vs `KDDTest+` F1 ≈ 0.77 — a ~0.20 drop). Surfacing this
gap explicitly, rather than reporting only a flattering CV/random-split number, is
the core "honest-broker" stance of the paper and the headline diagnostic of the
NSL-KDD distribution shift.

## 8. Statistical significance testing (quantum vs best classical)

We test the quantum model against the **best classical baseline** (highest mean
F1 on the relevant view) with two complementary tests — reviewers in this area
ask for each, and they answer different questions:

1. **McNemar's test** (`evaluation.mcnemar_test`) — compares the two models'
   *per-instance* predictions on the single fixed `KDDTest+` set. It tests the
   off-diagonal of the (A-correct?, B-correct?) contingency table, i.e. whether
   the two models make significantly *different* errors. We use the **exact
   binomial** form when the discordant count is small (≤25) and the
   continuity-corrected χ² otherwise. This is the appropriate paired test for two
   classifiers evaluated on one common test set.
2. **Paired t-test + Wilcoxon signed-rank** (`evaluation.paired_t_test`) — over
   the per-seed scores (F1 or any metric), paired by seed. The t-test is the
   parametric statement "mean over seeds differs"; the Wilcoxon signed-rank is the
   non-parametric companion. We also report a **paired Cohen's d** so magnitude,
   not just significance, is visible. Caveat we state explicitly: with 5 paired
   seeds the smallest two-sided p the Wilcoxon signed-rank can return is 0.0625,
   so we lead with the bootstrap/McNemar evidence and treat the per-seed Wilcoxon
   as corroborating rather than primary.
3. **Paired bootstrap** (`evaluation.paired_bootstrap`) — for the probability
   metrics that have no natural per-seed pairing story but a single fixed test set
   (AUPRC, TPR@1% FPR, ROC-AUC, Brier). We resample test instances with
   replacement (the *same* indices applied to both models, so the comparison stays
   paired), recompute metric(quantum) − metric(classical) per resample, and report
   the 95% percentile CI and a two-sided bootstrap p-value (2000 resamples). This
   is the most informative significance test for the low-FPR operating-point and
   calibration claims, since those are single-test-set quantities.

Significance is reported at α = 0.05, with exact / bootstrap p-values. We
explicitly avoid claiming significance from a single seed or from accuracy alone.
When multiple comparisons are made (across views / datasets / metrics), p-values
are interpreted with that multiplicity in mind (Holm correction noted where applied).

These tests are driven from the persisted reference-seed predictions/scores by
`compare.py`, which auto-selects the best classical baseline (by mean test F1) and
emits the McNemar + bootstrap + per-seed comparison for the quantum model. It
reads both the standalone-baseline JSONs and the unified-harness per-run JSONs
(whose `predictions` block carries `y_true`/`y_pred`/`y_score`), aligning the two
models on the identical fixed test set before pairing.

## 9. Reproducibility & provenance

Each run writes a self-describing JSON to `results/` containing: the resolved
config / CLI args, per-seed metrics, aggregates, the tuned hyperparameters, the
reference-seed predictions (for McNemar), and a full **environment snapshot**
(`provenance.env_snapshot`: platform, library versions, GPU, source-file md5s).
Determinism is best-effort on GPU (some CUDA kernels are nondeterministic); the
seed and environment are recorded so results are traceable even where bit-exact
reproduction is not guaranteed.

## 10. Threats to validity (to state in the paper)

- **Feature-budget asymmetry** — addressed by reporting both feature views (§3).
- **Distribution shift** — `KDDTest+` is harder than CV; we report the gap rather
  than mask it (§1).
- **Simulation vs hardware** — quantum results are noiseless-simulator numbers;
  the efficiency/runtime discussion notes simulation cost and the gap to NISQ
  hardware.
- **Few seeds** — 5 seeds is the practical floor; we use paired tests + Wilcoxon
  + effect size rather than over-claim from the parametric t-test alone.
- **Dataset age (NSL-KDD)** — mitigated by replicating the protocol on UNSW-NB15
  and CICIDS2017.
