# Gap Analysis & Novelty Positioning: Hybrid Quantum–Classical ML for Network Intrusion Detection

**Prepared by:** gap-analyst (quantum-ids team)
**Date:** 2026-05-30
**Input:** `literature_review.md` (lit-scout, Task #1), project + harness memory.
**Purpose:** Define 2–3 defensible novelty angles for a Q1 Elsevier submission, recommend a primary + backup, propose a paper framing/title, and give a 3-bullet experiment plan. All angles are constrained to be feasible on ONE laptop RTX 3050 Ti (4 GB VRAM), PennyLane `lightning.gpu`, ≤16 qubits, noiseless + noisy simulation, NO real QPU by default.

---

## 0. The Central Strategic Problem (read this first)

The literature review makes one thing unambiguous: **on tabular IDS data, well-tuned classical models (RF/XGBoost) are competitive with or superior to quantum models.** The three most rigorous works in the field all say so:

- **Bellante et al. (Computers & Security 2025, Elsevier Q1)** — quantum "advantage" in PCA-based IDS is *explainable by regularization, not quantum effects*.
- **Bhatnagar et al. (MQE 2026)** — best evaluation rigor in the field, and *admits RF/XGBoost remain superior*.
- **"Reality Check" (2025)** — quantum ≈ or < classical.

Bellante is published in our *exact target venue family* (Elsevier Q1). A reviewer **will** cite it against us. Therefore:

> **Any novelty whose headline claim is "our quantum model beats classical on accuracy/F1" is a losing bet.** It is likely false, and even if we get a lucky margin on one dataset, the reviewer has three Q1 papers to argue we got it from preprocessing/regularization, not quantum mechanics.

**Conclusion that drives everything below:** we must choose an angle that is **publishable even if quantum does not beat XGBoost on accuracy.** The contribution must be a *protocol, an analysis, a robustness/efficiency property, or a generalization property* — something true and useful regardless of whether the quantum model wins the accuracy horse race. Each candidate angle below is scored on exactly this criterion: **"Does the paper still stand if quantum loses to XGBoost?"**

---

## Candidate Angle 1 — The Honest-Broker Benchmark + the "Is It Even Quantum?" Audit (RECOMMENDED PRIMARY)

**(a) The precise gap.**
There is no standardized, fair, reproducible multi-dataset benchmark of QML-IDS against *honestly tuned* classical baselines under one protocol, *and* no systematic test of whether reported quantum gains survive an ablation that removes the quantum-specific component (i.e., is the gain quantum, or is it the classical PCA/encoder/regularization doing the work?). The field has the breadth paper (Abreu/QuantumNetSec) and the rigor paper (MQE) and the skeptic paper (Bellante) — but no single work that **unifies breadth + rigor + the regularization-audit** into one reproducible protocol with released code/seeds.

**(b) Why it's still open despite SOTA.**
- Abreu (QML-IDS/QuantumNetSec) has breadth (3–4 datasets, 3 methods) but **weak/under-tuned classical baselines** and no imbalance/calibration metrics, no regularization audit.
- MQE has the rigor (AUPRC, TPR@low-FPR, Brier, ECE, noise) but only 2 datasets, concedes classical superiority without *dissecting why*, and does not release a turnkey protocol.
- Bellante does the regularization audit but only for **fault-tolerant PCA-QML** in a narrow case study — not for the practical NISQ hybrid/QSVM models everyone actually uses, and not across the standard 3–4 dataset suite.
- Nobody has combined all three into a reproducible artifact. This is a methods/benchmark contribution, which Q1 Elsevier venues (Computers & Security, ESWA, KBS) explicitly value.

**(c) Our concrete contribution / claim.**
A reproducible benchmark and *diagnostic protocol* — call it the **"quantum-attribution audit"** — that for each (dataset × quantum method) answers: *how much of any observed gain is attributable to the quantum circuit vs. to classical preprocessing/regularization?* Mechanism:
1. **Matched-capacity controls.** For every hybrid VQC, build a *parameter-matched classical MLP* head on the identical PCA/encoder front-end (the harness already supports a param-matched `mlp` kind — see harness memory). For every QSVM (ZZ/fidelity kernel), build a classical RBF-SVM and a *random-feature* kernel approximation of the quantum kernel. If the classical matched control equals the quantum result, the "advantage" was not quantum.
2. **Regularization sweep.** Vary classical regularization (ridge on the readout, L2 on the MLP, RF depth) to test Bellante's hypothesis directly: does giving the classical baseline the same effective regularization the quantum circuit imposes erase the gap?
3. **One protocol, four datasets, honest tuning budget** (equal HPO budget for classical and quantum), fixed seeds/splits, statistical significance (paired bootstrap / Wilcoxon over seeds), full code release.

The **claim is not "quantum wins."** The claim is: *"Here is the first fair, reproducible, imbalance- and calibration-aware QML-IDS benchmark, and here is a rigorous attribution of where any quantum gains actually come from."* This is honest-broker science and is **publishable whether quantum wins, ties, or loses** — indeed, a clean negative/conditional result *is the contribution* and directly extends Bellante to the practical NISQ regime.

**(d) Experiments (feasible on our hardware).**
- 4 datasets (NSL-KDD, UNSW-NB15, CIC-IDS2017, CIC-IoT2023 or TON_IoT), PCA/feature-selection to ≤16 features → ≤16 qubits.
- Models: hybrid VQC, QSVM (ZZ kernel), + matched controls (param-matched MLP, RBF-SVM, random-feature kernel) + strong tuned classical (RF, XGBoost, tuned MLP, 1D-CNN). All already supported by the harness (`quantum_model`, `baselines`).
- Metrics: Acc, F1, **AUPRC, TPR@0.1%/1% FPR, Brier, ECE** (match MQE), + training/inference time + qubit/param budget.
- Significance over ≥5 seeds. Noiseless simulation suffices for the core table; Angle 2's noise study layers on top.
- All on ≤16 qubits, simulator — well within 4 GB / `lightning.gpu`.

**(e) Risk + honest framing.**
- *Risk:* "this is just a benchmark, not a method" — mitigate by making the **attribution audit** the methodological novelty (a reusable diagnostic), not the leaderboard.
- *The XGBoost risk is neutralized by design:* the paper's thesis explicitly anticipates that classical may win; the contribution is the *fair protocol + attribution*, framed positively as "what QML-IDS needs to demonstrate genuine advantage, and where (if anywhere) it currently does."
- *Risk:* overlap with Bellante — differentiate by scope (practical NISQ hybrid/QSVM, not FT-PCA), breadth (4 datasets), and operating-point/calibration rigor.

**(f) Differentiates from:** Abreu/QuantumNetSec (adds rigor + audit + honest baselines + reproducibility), MQE (adds breadth + attribution + released protocol), Bellante (extends the regularization critique from FT-PCA to practical NISQ models across a full benchmark, with a constructive framing).

---

## Candidate Angle 2 — Noise-Robustness & the Operating-Point Frontier (RECOMMENDED BACKUP / strong companion)

**(a) The precise gap.**
Almost all QML-IDS results are noiseless-simulator and report a single threshold (Acc/F1 at 0.5). Real IDS deployment cares about the **low-FPR operating point** (you cannot flood a SOC with false alarms) and about **NISQ noise**. No paper systematically maps how the *low-FPR detection frontier* (TPR@0.1%/1% FPR) of quantum vs. classical models **degrades under realistic noise channels** — i.e., whether any quantum advantage is *robust* or *evaporates under noise*.

**(b) Why it's still open.**
MQE tested 5 noise channels but reported aggregate metrics, not the low-FPR frontier under each noise level; Q-AGNN and wavelet-QSVM tested only depolarizing noise. The intersection — *operating-point robustness under a noise sweep* — is unoccupied.

**(c) Contribution / claim.**
A **noise–operating-point robustness study**: sweep depolarizing / amplitude-damping / phase-damping / readout error at several strengths; for each, report TPR@low-FPR and ECE for quantum vs. matched classical. Claim is about **robustness behavior**, not raw wins: e.g., "quantum kernels retain calibration better/worse under readout noise," or "the low-FPR frontier of hybrid VQCs collapses beyond noise level X — defining the hardware fidelity needed for deployable QML-IDS." Publishable regardless of who wins, because the *characterization* is the contribution.

**(d) Experiments.** Same model suite as Angle 1, run under PennyLane noisy-simulator (`default.mixed`) noise channels at multiple strengths; ≤16 qubits keeps `default.mixed` tractable on 4 GB (this is the heaviest compute — budget for it). Report frontier curves + ECE-vs-noise. No real QPU needed (we explicitly scope to *simulated* NISQ noise and state it as a limitation).

**(e) Risk + framing.** *Risk:* `default.mixed` density-matrix sim is expensive at 16 qubits on 4 GB — mitigate by capping noisy runs at ~10–12 qubits and fewer seeds. *XGBoost risk:* irrelevant here — classical models are noiseless by construction, so the honest framing is "quantum carries an extra noise tax; here is exactly how much, and the fidelity threshold at which it becomes competitive."

**(f) Differentiates from:** MQE (adds the operating-point frontier per noise level), Q-AGNN/wavelet-QSVM (adds channel breadth + calibration + classical operating-point comparison).

---

## Candidate Angle 3 — Cross-Dataset / Unseen-Attack Generalization of Quantum vs. Classical (viable, higher-risk)

**(a) The precise gap.** Generalization to *unseen attack families* and *cross-dataset shift* is tested in only ~3 works (Abreu, Wang et al., Q-AGNN). The hypothesis that **quantum feature maps generalize better under distribution shift** (e.g., because the kernel geometry is less prone to overfitting — the flip side of Bellante's regularization argument) is plausible but **untested as a primary claim**.

**(b) Why it's open.** Wang et al. did cross-base-station/unseen-attack on one dataset (5G-NIDD) with QCNNs; nobody has run a *quantum-vs-classical generalization gap* study across multiple datasets with a leave-one-attack-family-out protocol.

**(c) Contribution / claim.** Train-on-A / test-on-B (cross-dataset) and leave-one-attack-family-out protocols; measure the *generalization gap* (in-distribution minus out-of-distribution F1/AUPRC) for quantum vs. classical. **Reframes Bellante positively:** if quantum's implicit regularization is real, it should show up as *smaller generalization degradation under shift* — a property that XGBoost's raw accuracy advantage does not capture. The paper wins even if quantum loses in-distribution, as long as it degrades less out-of-distribution (a defensible, deployment-relevant story).

**(d) Experiments.** Same suite; add cross-dataset eval (e.g., train UNSW→test CIC-IDS on shared/aligned feature space) and leave-one-attack-out within multiclass datasets. ≤16 qubits, simulator.

**(e) Risk + framing.** *Highest scientific risk:* the hoped-for "quantum generalizes better" effect may simply not appear — then we have only a (still-publishable but weaker) generalization benchmark. *XGBoost risk:* if quantum neither wins in-distribution nor generalizes better, the story thins. This is why it is angle 3, not the primary.

**(f) Differentiates from:** Wang et al. (multi-dataset + leave-one-attack-out + quantum-vs-classical gap as the headline), Bellante (turns the regularization critique into a testable, positive generalization hypothesis).

---

## Recommendation

**Primary = Angle 1 (Honest-Broker Benchmark + Quantum-Attribution Audit), with Angle 2 (Noise–Operating-Point Robustness) folded in as a major section.** Together they form one coherent, rigor-forward paper that is **immune to the "XGBoost wins" risk** because its contributions are the *fair protocol, the attribution diagnostic, the calibration/low-FPR evaluation, the noise-robustness frontier, and the released reproducible artifact* — none of which depend on quantum winning the accuracy race. This directly out-rigors Abreu, matches/exceeds MQE, and constructively answers Bellante in our own target venue. **Backup = Angle 3** if reviewers want a sharper single "advantage" hypothesis; it can also be a follow-up paper.

**Why this is the safe-but-strong Q1 bet:** Q1 Elsevier reviewers reward methodological rigor, reproducibility, and honesty over yet another 99% accuracy claim they won't believe. We turn the field's biggest weakness (evaluation rigor + the unanswered "is it even quantum?" question) into our contribution, and our existing reproducible harness (config/runner/aggregate, fixed seeds, env snapshots) is *exactly* the infrastructure this angle needs — we are already tooled for it.

**Proposed framing / title (working):**
> **"How Quantum Is the Advantage? A Fair, Calibration- and Noise-Aware Benchmark and Attribution Audit of Quantum Machine Learning for Network Intrusion Detection."**

Framing paragraph: *Quantum machine learning for intrusion detection reports near-perfect accuracies, yet the field's most rigorous studies conclude that strong classical baselines remain competitive and that apparent quantum gains may stem from classical preprocessing and regularization rather than quantum effects. We resolve this ambiguity with the first unified, reproducible QML-IDS benchmark that (i) evaluates hybrid VQCs and quantum-kernel SVMs against honestly tuned classical baselines (RF, XGBoost, tuned MLP/CNN) across four standard NIDS datasets under one protocol with imbalance- and calibration-aware metrics (AUPRC, TPR at 0.1%/1% FPR, Brier, ECE) and statistical significance; (ii) introduces a quantum-attribution audit — parameter-matched classical controls and a regularization sweep — that quantifies how much of any observed gain is genuinely attributable to the quantum component; and (iii) maps the low-false-positive operating-point frontier under realistic NISQ noise channels to determine the hardware-fidelity regime, if any, in which QML-IDS becomes operationally competitive. We release full code, seeds, and splits.* The contribution stands whether quantum wins, ties, or loses — and gives the community the honest yardstick it currently lacks.

---

## 3-Bullet Experiment Plan (for team-lead)

- **Bullet 1 — Core honest-broker benchmark (noiseless):** 4 datasets (NSL-KDD, UNSW-NB15, CIC-IDS2017, CIC-IoT2023/TON_IoT) × {hybrid VQC, QSVM} vs honestly-tuned {RF, XGBoost, tuned-MLP, 1D-CNN} + **matched-capacity controls** (param-matched MLP, RBF-SVM, random-feature kernel). Metrics: Acc/F1 + **AUPRC, TPR@0.1%/1% FPR, Brier, ECE**, time, qubit/param budget. ≥5 seeds, paired-bootstrap/Wilcoxon significance. ≤16 features→≤16 qubits, simulator. *(Uses existing harness end-to-end.)*
- **Bullet 2 — Quantum-attribution audit:** regularization sweep (readout ridge, MLP L2, RF depth, QSVM kernel bandwidth) to test whether matched-regularization classical baselines close any quantum gap (directly answers Bellante in the NISQ regime); report gain decomposition per dataset/method.
- **Bullet 3 — Noise–operating-point robustness:** re-run the suite under PennyLane `default.mixed` with depolarizing / amplitude-damping / phase-damping / readout-error channels at multiple strengths; plot TPR@low-FPR and ECE vs noise level for quantum vs classical; report the fidelity threshold for operational competitiveness. Cap at ~10–12 qubits / fewer seeds to fit 4 GB density-matrix sim. *(Heaviest compute — schedule off-peak given shared-GPU contention.)*
