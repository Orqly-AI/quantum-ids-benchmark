# Cover Letter

**To:** The Editor-in-Chief, *Computers & Security* (Elsevier)
**Re:** Submission of original research manuscript

Dear Editor,

We are pleased to submit our manuscript, **"How Quantum Is the Advantage? A Fair, Calibration- and Noise-Aware Benchmark and Attribution Audit of Quantum Machine Learning for Network Intrusion Detection,"** for consideration as an original research article in *Computers & Security*.

Quantum machine learning (QML) for intrusion detection is a fast-growing area in which near-perfect accuracies (97 to 99.9%) are routinely reported. Yet the field's most rigorous studies, including Bellante et al. (*Computers & Security*, 2025), which we engage directly, caution that these gains may reflect classical dimensionality reduction and implicit regularisation rather than genuine quantum effects, and that honestly-tuned classical models remain competitive on tabular IDS. The community lacks a fair, reproducible methodology to settle the question. Our manuscript supplies exactly that.

**Contributions and fit for the journal.** We contribute (i) a single leakage-controlled, multi-dataset QML-IDS benchmark (NSL-KDD, UNSW-NB15, CICIDS2017, NF-ToN-IoT-v2) evaluating hybrid variational circuits and quantum-kernel SVMs against five honestly-tuned classical baselines, with an equal-budget fairness design and operationally meaningful, calibration- and imbalance-aware metrics; (ii) a **quantum-attribution audit** that decomposes any measured difference into its classical and genuinely-quantum components, a direct and constructive test of the Bellante et al. critique extended from their fault-tolerant case study to the practical NISQ models the community deploys; and (iii) a noise-robustness study plus a practical backend-efficiency finding. The work is squarely within the journal's scope: it concerns the rigorous, security-relevant evaluation of an emerging detection technology, and it answers a critique published in this venue.

**Principal findings, stated honestly.** Tuned classical models match or exceed the quantum models on aggregate detection on all four datasets, and the audit attributes this to classical preprocessing and regularisation rather than quantum effects. Two exceptions survive Benjamini-Hochberg false-discovery-rate control across the audit family. First, the quantum-kernel SVM significantly out-ranks its direct classical surrogate, a random-feature kernel, on AUPRC (q = 0.011) and ROC-AUC (q = 0.018); this is the study's most robust positive, and it is the comparison that speaks most directly to whether the quantum feature map is classically simulable in effect. Second, a small four-qubit hybrid out-detects the best classical baseline at the strict 1% false-positive operating point on the distribution-shifted NSL-KDD task (p = 0.005, q = 0.030). We report plainly that the latter survives FDR control but not the more conservative family-wise Holm correction. We also show that random-cross-validation evaluation overstates novel-attack detection by roughly 20 F1 points, and that CPU quantum simulation is 8 to 14 times faster than GPU at these qubit counts. The contribution is the fair, attributable methodology and its honest result, which stands whether quantum wins, ties, or loses.

**Scope and limitations we state up front.** All quantum results are simulated; we have no QPU access and treat the simulated numbers as an optimistic bound. Our seed budget is small and uneven (five seeds for the classical baselines on NSL-KDD, three elsewhere and for all quantum runs, two for the noise sweep), which we report explicitly, and the attribution audit is decomposed on NSL-KDD. We prefer to surface these limits in the manuscript rather than have them inferred.

**Reproducibility.** The code, configurations, fixed seeds, the download and preprocessing scripts that reconstruct our exact splits, and the per-run result records are released in a public repository; every reported number is regenerable from the released harness. We do not redistribute the source corpora, which are public benchmarks available from their original providers; preprocessed copies will be released on request.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and declare no competing financial interests. We believe the work will be of direct interest to the journal's readership working at the intersection of machine learning, emerging computing paradigms, and network security.

Thank you for your consideration. We look forward to the reviewers' feedback.

Sincerely,

Syeda Anshrah Gillani (corresponding author), Hamdard University, Karachi, Pakistan · SyedaAnshrah16@gmail.com
Mirza Samad Ahmed Baig, Fandaqah, Al Khobar, Saudi Arabia · MirzaSamadcontact@gmail.com
Shahid Munir Shah, SZABIST University, Pakistan · Shahidmunirshah@yahoo.com
Asher Ali, Fandaqah, Al Khobar, Saudi Arabia · CTO@Fandaqah.com
Hamzah Siddiqui, Fandaqah, Al Khobar, Saudi Arabia and Hamdard University, Karachi, Pakistan
