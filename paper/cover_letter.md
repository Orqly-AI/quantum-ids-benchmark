# Cover Letter

**To:** The Editor-in-Chief, *Computers & Security* (Elsevier)
**Re:** Submission of original research manuscript

---

Dear Editor,

We are pleased to submit our manuscript, **"How Quantum Is the Advantage? A Fair, Calibration- and Noise-Aware Benchmark and Attribution Audit of Quantum Machine Learning for Network Intrusion Detection,"** for consideration as an original research article in *Computers & Security*.

Quantum machine learning (QML) for intrusion detection is a fast-growing area in which near-perfect accuracies (97–99.9%) are routinely reported. Yet the field's most rigorous studies — including Bellante et al. (*Computers & Security*, 2025), which we engage directly — caution that these gains may reflect classical dimensionality reduction and implicit regularisation rather than genuine quantum effects, and that honestly-tuned classical models remain competitive on tabular IDS. The community lacks a fair, reproducible methodology to settle the question. Our manuscript supplies exactly that.

**Contributions and fit for the journal.** We contribute (i) a single leakage-controlled, multi-dataset QML-IDS benchmark (NSL-KDD, UNSW-NB15, CICIDS2017, NF-ToN-IoT-v2) evaluating hybrid variational circuits and quantum-kernel SVMs against five honestly-tuned classical baselines, with an equal-budget fairness design and operationally meaningful, calibration- and imbalance-aware metrics; (ii) a **quantum-attribution audit** that decomposes any measured difference into its classical and genuinely-quantum components — a direct, constructive test of the Bellante et al. critique extended from their fault-tolerant case study to the practical NISQ models the community deploys; and (iii) a noise-robustness study plus a practical backend-efficiency finding. The work is squarely within the journal's scope: it concerns the rigorous, security-relevant evaluation of an emerging detection technology, and it answers a critique published in this venue.

**Principal findings, stated honestly.** Tuned classical models match or exceed the quantum models on aggregate detection on all four datasets, and the audit attributes this to classical preprocessing and regularisation rather than quantum effects. The one robust exception is operationally meaningful: a small four-qubit hybrid circuit significantly out-detects the best classical baseline at the strict 1% false-positive operating point on the distribution-shifted NSL-KDD task (p = 0.005). We also show that random-cross-validation evaluation overstates novel-attack detection by ~20 F1 points, and that CPU quantum simulation is 8–14× faster than GPU at these qubit counts. The contribution is the fair, attributable methodology and its honest result — which stands whether quantum wins, ties, or loses.

**Reproducibility.** All code, fixed seeds, dataset splits, and per-run provenance records are released in a public repository; every reported number is regenerable from the released harness.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and declare no competing financial interests. We believe the work will be of direct interest to the journal's readership working at the intersection of machine learning, emerging computing paradigms, and network security.

Thank you for your consideration. We look forward to the reviewers' feedback.

Sincerely,
Syeda Anshrah Gillani (corresponding author) — Hamdard University, Karachi, Pakistan · SyedaAnshrah16@gmail.com
Mirza Samad Ahmed Baig — Fandaqah, Al Khobar, Saudi Arabia · MirzaSamadcontact@gmail.com
