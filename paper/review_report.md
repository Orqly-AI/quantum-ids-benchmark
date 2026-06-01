# Referee Report — *Computers & Security*

**Manuscript:** "How Quantum Is the Advantage? A Fair, Calibration- and Noise-Aware Benchmark and Attribution Audit of Quantum Machine Learning for Network Intrusion Detection."

**Reviewer role:** Peer reviewer, *Computers & Security* (Q1).

---

## 1. Summary

The paper presents a unified, leakage-controlled benchmark that pits two families of quantum models — hybrid variational quantum circuits (VQCs) and quantum-kernel SVMs (QSVMs) — against five tuned classical baselines across four NIDS datasets (NSL-KDD, UNSW-NB15, CICIDS2017, NF-ToN-IoT-v2). It evaluates under imbalance- and calibration-aware metrics (AUPRC, TPR@0.1%/1% FPR, Brier, ECE) with significance testing, and adds a "quantum-attribution audit" (parameter-matched classical MLP controls, a random-feature kernel control, and a regularisation sweep) designed to attribute any observed gain to the quantum component versus classical preprocessing/regularisation. The headline finding is largely negative: honestly-tuned classical models (Random Forest, XGBoost) match or beat the quantum models on aggregate detection on all four datasets, and the audit attributes this to classical effects — confirming and extending Bellante et al. The single robust positive is a 4-qubit hybrid that significantly out-detects the best classical baseline at the 1% FPR operating point on NSL-KDD (p=0.005). All quantum results are simulated.

## 2. Overall recommendation

**Major Revision.** This is a well-motivated, honestly-framed, and unusually careful paper that addresses a real and well-documented credibility problem in the QML-IDS literature. The methodology (two feature views, leakage control, official splits, operating-point and calibration metrics, attribution audit, CV-to-test gap reporting) is exactly what the field has been missing, and the willingness to report a negative result is commendable and venue-appropriate. However, several issues currently prevent acceptance: (i) the entire quantum side is simulation-only, while the one positive claim is the kind of narrow, single-operating-point, single-dataset result most vulnerable to being a simulation artefact; (ii) statistical power is thin and, more seriously, the released aggregate file reveals an *unequal and undisclosed seed count* across models that undermines the "paired over 5 seeds" claim as written; (iii) the contribution is methodological rather than scientific-discovery, and two of the four datasets are near-saturated and arguably add little; (iv) the attribution audit, while a genuinely good idea, is reported almost entirely on one dataset and its central positive result rests on a single bootstrap p-value that is not corrected for the large multiple-comparison surface the audit creates. I judge this **borderline Q1**: the framing and rigor are Q1-tier, but the empirical contribution is thinner than the framing implies, and the headline positive needs hardening before it can carry the weight the abstract/discussion place on it.

## 3. Strengths

1. **The honest-broker framing is genuinely valuable and well-executed.** The paper correctly identifies that the field's deficit is methodological, not another accuracy record, and it commits to a contribution that "stands whether quantum wins, ties, or loses" (Abstract; Contribution 5). This is the right scientific posture for this topic and venue.

2. **Equal-budget fairness design (Sec. 4.2, Contribution 2).** Reporting both a *full* view and a *same-budget* view (every model on the identical PCA-reduced input the quantum model receives) directly removes the silent feature-budget asymmetry that confounds most prior QML-IDS work. This is a clean, defensible idea and is the comparison that actually licenses any "advantage" claim.

3. **Leakage control and respect for official splits (Sec. 4.1).** Fitting all preprocessing on train only, dropping leaky identifier columns (IP/port tuples, flow IDs, timestamps, row indices), CV-within-training-only model selection, and scoring the test set once per model per seed is the correct protocol and is described precisely.

4. **The CV-to-test gap finding is a real and generalisable contribution.** Documenting a consistent ~0.20 F1 drop on NSL-KDD from within-training CV (~0.98) to the official KDDTest+ split (~0.78) is an important, transferable diagnostic that explains why so much of the literature's near-perfect numbers are inflated. This alone is useful to the community.

5. **Operating-point and calibration metrics (Sec. 5.2, 6.3).** TPR@0.1%/1% FPR, Brier, and ECE are the operationally meaningful quantities for IDS, and reporting them with DR and FPR together is appropriate to the threat model (cost asymmetry).

6. **The backend-crossover efficiency result (Table 5)** — CPU `lightning.qubit` 8–14× faster than GPU `lightning.gpu` at small qubit counts — is a concrete, practically useful observation for the QML community, even if it is somewhat tangential to the main thesis.

7. **Reproducibility infrastructure.** Released code/seeds/splits, per-run JSON provenance with environment snapshots and source checksums, is above the norm for this literature.

## 4. Major weaknesses

**M1. Simulation-only, yet the headline positive is exactly the claim most fragile to simulation artefacts.** All quantum results are noiseless/noisy *simulation*; there is no QPU. The paper acknowledges this (Sec. 8, "Simulation only," and "we treat the simulated results as an optimistic bound"). The problem is the asymmetry between the honesty of the limitation and the weight placed on the single positive result: the 4-qubit-hybrid TPR@1%FPR edge on NSL-KDD is promoted to the Abstract, Discussion, and Conclusion as the paper's signature finding and the motivation for "real-hardware follow-up." But a single-operating-point advantage from a 4-qubit noiseless simulation is precisely the kind of effect that real-device noise, finite-shot sampling, transpilation, and connectivity constraints most plausibly erase. The authors even note (Sec. 6.4 / audit) that mild *simulated* noise slightly *raises* ROC-AUC — i.e., the "noise as regulariser" story is itself a simulation artefact and would not survive the correlated/non-Markovian noise they admit they cannot model. As written, the paper's one affirmative scientific claim is unfalsifiable on the evidence presented. The negative findings are robust to this (classical winning is only strengthened if quantum is given an optimistic simulated bound), but the positive is not.

**M2. Statistical power, and — more seriously — an apparent inconsistency between the stated protocol and the released data.** The methodology states every model is run over 5 seeds {42–46} with paired comparisons (Sec. 5.2; methodology §7). However, the released `results/summary_agg.csv` shows wildly unequal and non-5 seed counts: `nslkdd,hybrid_qnn` n_seeds=**39**, `matched_mlp`=**12**, `qsvm`=**12**, but `random_forest`=**4**, `svm_rbf`=**4**, `classical_mlp`=**4**; CICIDS `hybrid_qnn`=6, `qsvm`=3; ToN-IoT `qsvm`=3. This is a substantive problem: (a) it contradicts the paper's repeated "mean over ≥5 seeds, paired by seed" claim; (b) **paired** tests require matched per-seed runs across the two compared models — 39 quantum vs 4 RF seeds cannot be paired by seed as described in methodology §8; (c) the asymmetric counts (many more quantum runs than classical) raise the question of whether the reported quantum means/CIs and the classical means/CIs are even on comparable footing. The authors must reconcile the manuscript's protocol with the released aggregates and clarify exactly which seeds enter each significance test. With only 5 (or fewer) genuinely paired seeds, the authors correctly note the Wilcoxon floor is p=0.0625, so the entire significance story leans on the bootstrap — see M4.

**M3. Is a negative result with one narrow positive a sufficient contribution for Q1?** The negative aggregate result largely *replicates* Bhatnagar et al. (classical ensembles superior) and *extends* Bellante et al. (advantage explained by regularisation) — both of which the paper itself cites as having already established the core conclusion. The genuinely *new* scientific content reduces to: (i) the attribution audit machinery, (ii) the single 4-qubit low-FPR positive, (iii) the CV-to-test gap quantification, and (iv) the backend-crossover. Items (iii) and (iv) are useful but minor/tangential. Item (ii) is fragile (M1, M4). So the load-bearing novelty is essentially the audit *methodology* applied to NISQ models — a real contribution, but the paper should be honest that it is primarily a *benchmark/methods* paper that confirms prior critiques on a broader footing, rather than overselling the 4-qubit result as a discovery. The current Abstract/Discussion framing ("a small but real amount" of quantum advantage, "precisely where a genuine quantum advantage would first appear") risks over-claiming on one p-value.

**M4. The attribution audit is a good idea but under-delivered as evidence, and the central positive is not multiplicity-corrected.** The audit (Table 4, attribution_audit.md) is run essentially only on NSL-KDD — the other three datasets are not decomposed, so the paper's repeated claim that "the audit attributes this to classical effects … on a full multi-dataset benchmark" (Discussion, Conclusion) is not supported by the audit; it is supported only on one dataset, with the other three resting on the headline table alone. More critically: the released `attribution_audit.md` contains on the order of ~150 quantum-vs-control comparisons across variants/metrics/controls. The single headline positive (Hybrid-Angle q4, TPR@1%FPR, +0.050, p=0.005) is one cell in a very large grid, and the manuscript states multiplicity is only "noted where applied." Given the size of that comparison surface, an uncorrected p=0.005 is not convincing on its own; a Holm/BH correction across the audit family should be applied and reported, and the headline claim re-evaluated against it. The methodology text itself concedes the Wilcoxon floor issue and that the bootstrap is the primary evidence — so the entire positive finding rests on one bootstrap p-value over a single fixed test set, uncorrected. That is too thin to anchor the paper's affirmative narrative.

Additionally, the audit's most *quantum-favourable* numbers are quietly buried in the supplementary file but not in the paper: the IQP projected-kernel QSVM (`ovn1_qsvm_iqp_proj`) beats the random-feature kernels significantly on ROC-AUC (+0.13–0.15, p=0.002/0.008) and AUPRC — arguably a *stronger* "quantum kernel is not classically simulable in effect" signal than the 4-qubit VQC result that the paper chose to headline. Yet the same QSVM is catastrophically bad at TPR@1%FPR (0.047). The paper should explain this selective emphasis and present the QSVM-vs-random-feature evidence in the main text, since it bears directly on the audit's stated purpose (md interpretation: "a random-feature kernel that matches the QSVM indicates the quantum kernel is classically simulable in effect").

**M5. Two of four datasets are near-saturated and contribute little discriminative value (Sec. 6.1, Table 3, Fig. 3).** The paper itself concedes CICIDS2017 and NF-ToN-IoT-v2 are near-saturated (classical F1 ≥ 0.97, indeed 1.000 on CICIDS) and that high quantum scores there "should be read as 'this task is easy.'" If they have "weak discriminative power," then the "four standard datasets" breadth claim is effectively a two-dataset benchmark (NSL-KDD, UNSW-NB15) plus two confirmatory-but-uninformative datasets. This weakens the central "multi-dataset" selling point. Moreover, both are non-standard scopes: CICIDS is a 3-day MachineLearningCVE subset and ToN-IoT is capped at 200k of 16.9M rows. The capping/subsetting could itself affect saturation and the class structure; the authors should justify why these scopes are representative, or replace/augment with a genuinely discriminative modern dataset (e.g., a harder cross-dataset or temporal-split evaluation) rather than two saturated ones.

**M6. Novelty positioning vs. Bhatnagar / Bellante / Abreu needs sharper, more honest delineation (Sec. 2, Table 1).** The paper claims three differentiators: tuned baselines + equal-budget + significance (vs. Abreu), causal decomposition (vs. Bhatnagar), and NISQ extension of the regularisation argument (vs. Bellante). These are real but incremental. Against Bhatnagar in particular, the paper concedes Bhatnagar already set the evaluation standard (AUPRC, TPR@low-FPR, Brier, ECE, five noise channels) and already reached the "classical wins" conclusion; the marginal addition is the decomposition (M4 shows this is single-dataset). Against Bellante, the manuscript extends a fault-tolerant/PCA-specific argument to NISQ — but since all results are simulated (M1), the "NISQ" extension is simulated NISQ, narrowing the practical gap to Bellante. The Related Work should state plainly what a reader already knows from these three papers vs. what is genuinely first-established here.

**M7. Possible over-claim in the noise narrative (Sec. 6.4).** "Graceful degradation, and in the mild regime none at all," with noise *raising* ROC-AUC, is presented as evidence the 4-qubit signal is "credible rather than noise." But this is run on the **6-qubit** hybrid (Sec. 6.4 / Fig. 4), whereas the positive result is the **4-qubit** circuit — so the noise robustness shown does not directly defend the configuration that produced the headline result. The authors should run the noise sweep on the exact 4-qubit configuration that yields the positive, or stop using the noise result to bolster the 4-qubit claim.

## 5. Minor weaknesses

1. **Author affiliation "Fandaqah" (frontmatter)** appears to be a company, not an academic institution; combined with the released code under the `Orqly-AI` GitHub org, the competing-interest declaration ("no … financial interests or personal relationships") may warrant a closer look. Please confirm there is no commercial interest in the benchmark/tooling.

2. **The generative-AI disclosure** (used for "code scaffolding, experiment orchestration") is appropriately honest, but given the seed-count discrepancy (M2), the authors should double-check that auto-orchestrated runs were logged/aggregated consistently with the stated protocol.

3. **Qubit ceiling reported inconsistently:** Abstract/Sec. 5.3 say "ceiling of sixteen (noiseless)/twelve (noisy)"; Sec. 3.2 says noisy runs capped at twelve; the abstract earlier mentions "≤12 qubits." Standardise the wording.

4. **Table 2 (Brier/ECE for rf_kernel)** may be NaN (LinearSVC has no calibrated probabilities), per attribution_audit.md note — ensure no audit conclusion silently depends on excluded rows.

5. **"verify_sub" row** in attribution_audit.md (TPR@1%FPR 0.5507, no p-value) appears to be a verification/debug artefact left in the released audit table; clean it from any released supplementary material.

6. **SMOTE ablation** is promised "in the appendix" (methodology §5) but I see no appendix in the 18-page manuscript; either include it or remove the forward reference.

7. **Reduction method ambiguity:** Fig. 1 shows "PCA/AE" front-end and Sec. 3.2 mentions ablating an autoencoder, but all reported results use PCA. State whether the AE was actually run; if not, remove it from the figure to avoid implying unreported experiments.

8. **Effect sizes (Cohen's d)** are promised (Contribution 5, methodology §8) but not tabulated in the main results; report them alongside the p-values, especially given M2/M4.

9. **Per-dataset best-classical model identity** is given for the headline table but the audit controls' selection ("best control per-metric, direction-aware") creates a researcher-degrees-of-freedom concern — pre-register or at least clearly state that the control is chosen to be the *strongest* per metric (which makes any surviving quantum positive conservative, a point worth making explicitly in the paper's favour).

## 6. Specific questions to the authors (rebuttal)

1. **Reconcile the seed counts.** Why does `summary_agg.csv` show 39/12/12 seeds for quantum/matched-MLP/QSVM but only 4 for RF/SVM/classical-MLP on NSL-KDD? Exactly which seeds are paired in each McNemar/bootstrap/t-test? Does the "paired by seed" claim hold for the quantum-vs-RF comparisons that produce p=0.021 and p=0.005?

2. **Multiplicity.** How many total comparisons does the attribution audit perform, and does the headline TPR@1%FPR positive (p=0.005) survive Holm/BH correction across that family? Please report corrected p-values.

3. **Why headline the 4-qubit VQC and not the IQP projected-kernel QSVM?** The QSVM beats both random-feature controls on ROC-AUC/AUPRC with smaller p-values (0.002–0.008). Is the VQC result chosen because it is operationally framed (TPR@1%FPR), and if so, how do you justify de-emphasising the QSVM kernel-vs-random-feature evidence that speaks more directly to "classical simulability"?

4. **Noise on the 4-qubit circuit.** The noise sweep is on the 6-qubit hybrid. Does the 4-qubit configuration that produces the positive survive the same simulated channels, and at what shot budget? Were any results computed under finite-shot sampling, or all analytic/state-vector?

5. **Audit breadth.** Why is the attribution audit reported only for NSL-KDD? Can you provide the audit decomposition for UNSW-NB15 (the other discriminative dataset) to support the multi-dataset attribution claim?

6. **Dataset scope.** Justify the CICIDS 3-day subset and the 200k ToN-IoT cap. Does the saturation persist on the full corpora, or is it partly an artefact of subsetting? Given they are near-saturated, what do they add beyond confirming "easy tasks are easy"?

7. **What would falsify the positive?** State the specific real-hardware experiment (device, qubit count, shots, error rates) that would confirm or refute the low-FPR signal, and what effect size you would consider confirmatory given finite-shot noise.

## 7. Presentation / formatting issues

1. **Figures are vector/TikZ-rendered** (architecture Fig. 1, audit-concept Fig. 2, bar chart Fig. 3, qubit ablation Fig. 4) and read cleanly; no resolution problems. Good.
2. **Table 4 (audit)** uses bold to flag the single positive; ensure the caption defines that the Δ sign convention flips for ECE (it does state this — keep it).
3. **Fig. 4 caption** describes a dashed RF reference at 0.467 — the plotted RF line spans (4,0.467)–(12,0.467) as a constant; label it clearly as a constant reference, not a trend.
4. The abstract is long and dense (a single ~280-word paragraph). Consider tightening; the "$8$–$14\times$" backend finding and other secondary results crowd the central message.
5. Reference list is short (13 entries) for a benchmark paper claiming to survey a "fast-growing body of work"; several key QML-kernel/expressivity and IDS-benchmark references could strengthen Related Work. Verify all `\cite` keys resolve (the rendered pages show bracketed numbers correctly, but the bibliography ends at [13] while the text references appear consistent — double-check `bhatnagar2026mqe` and `bellante2025pca` years, as 2026 citations in a 2026 submission should be confirmed as published vs. preprint).
6. Minor: "ansatze" / "ansätze" spelling appears inconsistently; standardise.
7. The CRediT statement lists both authors as contributing equally to *all* roles verbatim — acceptable, but reviewers/editors sometimes flag identical all-role CRediT as uninformative.

---

## TOP 5 THINGS TO FIX FIRST (ordered by impact on acceptance)

1. **Reconcile and disclose the seed protocol (M2).** Explain the 39/12/4 seed discrepancy in the released aggregates, confirm the paired tests are actually paired, and re-report all significance with the genuine paired-seed sets. This currently undercuts the paper's core "statistical rigour" claim and must be resolved before any acceptance.

2. **Harden or down-weight the single positive result (M1 + M4).** Apply multiple-comparison correction across the audit family and report whether the 4-qubit TPR@1%FPR positive (p=0.005) survives. If it does not survive correction, reframe the paper honestly as a negative/methods contribution. If it does, still soften the Abstract/Discussion language to reflect that it is one operating point, one dataset, in noiseless simulation.

3. **Run the noise sweep (and, ideally, a finite-shot study) on the exact 4-qubit configuration that produces the positive (M7),** so the "credible rather than noise" argument actually defends the headline configuration, not a different (6-qubit) one.

4. **Extend the attribution audit beyond NSL-KDD, at minimum to UNSW-NB15 (M4/M6),** so the repeated "multi-dataset attribution" claim is supported by the audit itself rather than only by the headline F1 table.

5. **Justify or revise the dataset slate (M5).** Either defend the CICIDS/ToN-IoT subset scopes and explain what the two near-saturated datasets contribute, or substitute a genuinely discriminative modern/cross-dataset evaluation so the "four-dataset benchmark" claim carries real discriminative weight.
