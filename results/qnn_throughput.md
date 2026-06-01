# QNN Throughput Benchmark & Feasible Sweep Budget (task #14)

Gating reality-check for the experiment campaign (#7). Measured on the project
WSL `qml` env, **lightning.qubit (CPU)**, `diff_method=adjoint` (confirmed active),
median of 3 timings after warmup. Benchmark: `src/bench_qnn.py [--full]`.

## TL;DR
**The HybridQNN trains fast enough for a full campaign.** A worst-case config
(12 qubits, depth 3, 2000 train samples) is ~36 s/epoch; the typical config
(8 qubits, depth 2, 2000 samples) is ~15 s/epoch. The earlier "q4/64-sample epoch
didn't finish in 4 min" does **not reproduce** here — see "Discrepancy" below.
adjoint diff is active and is not the problem.

## Key findings
1. **adjoint is active.** `qnode.diff_method == 'adjoint'` on lightning.qubit. Good.
2. **TorchLayer loops per-sample (confirmed).** Bare QNode: batch=1 = 3.3 ms,
   batch=32 = 34.7 ms -> **~11x**, i.e. cost scales linearly with batch (no device
   broadcast). This is the known `qml.qnn.TorchLayer` behaviour, not a bug. It sets
   the throughput ceiling but is not catastrophic at our sizes.
3. **Throughput (forward+backward), samples/sec:**

   | qubits | depth | samp/s (fb) |
   |---:|---:|---:|
   | 4  | 2 | ~240 |
   | 4  | 3 | ~185 |
   | 8  | 2 | ~140 |
   | 8  | 3 | ~100 |
   | 12 | 2 | ~74  |
   | 12 | 3 | ~57  |

4. **Per-epoch wall-time (forward+backward):**

   | config | 1000 samples | 2000 samples |
   |---|---:|---:|
   | 4q / depth 2  | 4.3 s  | 8.7 s  |
   | 8q / depth 2  | 7.2 s  | 14.5 s |
   | 8q / depth 3  | 10.2 s | 20.4 s |
   | 12q / depth 2 | 13.9 s | 27.9 s |
   | 12q / depth 3 | 17.9 s | 35.7 s |

## Discrepancy with sw-engineer's harness observation
sw-engineer saw a q4/n_layers=1/64-sample/1-epoch hybrid_qnn smoke not finish in
~4 min through the harness. My direct measurement: q4/depth2/64-sample batch =
~0.27 s for one forward+backward, so a 64-sample epoch is well under 1 s. The QNN
forward/backward is NOT the cause. Likely culprits in that run (to check, not QNN):
- a co-scheduled `svm_rbf` with `probability=True` (internal 5-fold Platt CV) or an
  RF/XGB tuning job saturating the shared box at the same time;
- data `load()` cost (NSL-KDD load+PCA is a few seconds, paid once per run);
- a stuck/zombie process (we saw earlier WSL jobs starved under GPU/CPU contention).
Recommend re-timing a hybrid_qnn run **in isolation** (no concurrent baseline jobs)
to confirm; the numbers above were measured with the box otherwise idle.

## Recommended #7 budget
Defaults that keep the quantum campaign to a few hours on this one box:

- **Backend:** `device=cpu` (lightning.qubit) for all quantum runs (~10x faster
  than GPU here; matches the backend-finding).
- **Quantum train subsample:** **2000** stratified rows for HybridQNN (q<=8),
  **1500** for q=12. Test set stays full (22.5k) — eval is forward-only and cheap.
- **Epochs:** **20** (QNNs converge fast; 30 is fine for the headline configs but
  20 keeps the grid affordable). Early-stop optional.
- **Batch size:** 32-64 (per-sample loop makes batch size ~throughput-neutral;
  64 slightly better wall-clock).
- **QSVM (MEASURED, q=8):**
  - **Fidelity kernel = O(N^2) circuit evals, ~7.9 ms/pair.** 1000-train Gram
    (~500k pairs) = **~66 min**; 1200-train = **~95 min**; plus N_train x N_test
    for scoring. This is the campaign's single biggest cost. **Cap fidelity-QSVM
    train subsample at ~600 (NSL-KDD only)** and subsample test scoring to ~4000.
  - **Projected kernel = O(N) circuit evals, ~3.3 ms/sample.** 1000-train feature
    build = **~3.3 s** (then a classical RBF over the expectation vectors). This is
    **~1000x cheaper** than fidelity and scales to large N.
  - **RECOMMENDATION: make the projected kernel the DEFAULT QSVM** for the campaign
    (cheap, scales, robust). Run the fidelity kernel only as a small-N comparison
    point on NSL-KDD (<=600 train) so the paper still reports both kernels. For the
    attribution audit (#9) the random-feature-kernel-vs-QSVM comparison should use
    the projected kernel as the primary QSVM so the controls run at matched N.

### Cost projection for the campaign
Per hybrid_qnn run (20 epochs):
- 8q/depth2/2000: 20 x 14.5 s ~= **4.8 min**
- 12q/depth3/1500: 20 x ~27 s ~= **9 min**

For the headline grid on NSL-KDD (6 hybrid variants + QSVM) x 3 seeds:
- 6 hybrid x 3 seeds x ~5 min ~= **90 min**
- projected-QSVM x 3 seeds x ~1 min ~= **3 min** (negligible with the projected kernel)
- fidelity-QSVM (600-train, NSL-KDD only) x 3 seeds x ~24 min ~= **72 min**

**Recommendation:** run all 4 datasets for the BEST 1-2 hybrid encodings +
projected-QSVM at 3 seeds (feasible overnight); run the full 9-variant ablation on
**NSL-KDD only** at 3 seeds (~2 h incl. one small fidelity-QSVM point). The noise
sweep (#10): restrict to 1-2 encodings, q<=8, 1000-sample subsample, since
default.mixed is ~5-10x slower than lightning.qubit.

## If more speed is needed later (not required for #7)
- The per-sample TorchLayer loop is the only real lever. Options, in order:
  1. Lower train subsample / epochs (already recommended) — simplest, sufficient.
  2. A hand-rolled batched training loop calling the QNode with `qml.vmap` or a
     manually broadcasted parameter tensor (lightning.qubit supports parameter
     broadcasting for some ops, but NOT through TorchLayer today) — engineering
     effort, deferred.
  3. GPU only helps at q>=~18 (we don't reach that) — not worth it.
