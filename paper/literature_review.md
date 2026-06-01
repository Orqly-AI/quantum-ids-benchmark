# Literature Review: Quantum & Hybrid Quantum–Classical Machine Learning for Network Intrusion Detection (2021–2026)

**Prepared by:** lit-scout (quantum-ids team)
**Date:** 2026-05-30
**Scope:** Peer-reviewed papers and high-visibility preprints applying quantum machine learning (QML) to network intrusion detection (NIDS), IoT security, and closely related anomaly/attack detection, 2021–2026.
**Purpose:** Feed the gap analysis (Task #2) and define the SOTA we must cite, contextualize, and beat for a Q1 Elsevier submission.

---

## 1. Executive Summary

The field is young, fast-growing, and methodologically narrow. Three observations dominate:

1. **Hybrid quantum–classical models are the de-facto standard.** Almost every credible result uses a classical feature extractor / dimensionality reducer feeding a small parameterized quantum circuit (VQC, QSVM kernel, or QCNN). Purely quantum, end-to-end models on raw network data essentially do not exist at scale because of NISQ qubit/depth limits.
2. **Reported headline accuracies are high but methodologically inconsistent.** Many papers report 97–99.9% accuracy, but on tiny PCA-reduced subsets, with binary tasks, single datasets, no class-imbalance-aware metrics (low-FPR TPR, AUPRC), and noiseless simulation. The few rigorous works (Bellante et al.; Bhatnagar et al.; "Reality Check") explicitly conclude **quantum offers little or only conditional advantage over strong classical baselines (RF/XGBoost)** on tabular IDS data.
3. **Evaluation rigor is the field's weakness and our opportunity.** Cross-dataset generalization, hardware-noise robustness, low-false-positive operating points, calibration, statistical significance, and honest classical baselines are almost universally missing or thin.

**SOTA "must cite and beat":** Abreu et al. (QML-IDS, 2024 / QuantumNetSec IJNM 2025), Kalinin & Krundyshev (2022, the foundational/most-cited), Gong et al. (VQCNN, J. Supercomputing 2024), the EPJ-QT HQCNN DDoS/malware paper (2025, real-QPU result), and the recent rigorous critiques (Bellante et al., Comp. & Security 2025; Bhatnagar et al. MQE 2026). See Section 5.

---

## 2. Comparison Table

Q1 status reflects Scopus/JCR quartile of the venue at time of writing (best-effort; to be re-verified for final manuscript). "sim" = simulator; "QPU" = run on real quantum hardware.

| # | Paper (short) | Year | Venue (Q?) | Method | Dataset(s) | Headline metric | Eval rigor notes |
|---|---|---|---|---|---|---|---|
| 1 | Kalinin & Krundyshev | 2022 | J. Computer Virology & Hacking Tech. (Springer, Q2/Q3) | QSVM, QCNN | Custom 10M-record stream dataset (from IoT/BoT-IoT) | ~98% acc; ~2× faster training | Big-data focus; custom data hurts comparability; sim |
| 2 | Said (PMC9226281 companion) / "Security IDS via QML" | 2022 | J. Computer Virology & Hacking Tech. (Q2/Q3) | QSVM, QCNN | IoT/BoT-IoT-derived | ~98% acc | Foundational; sim |
| 3 | Gong et al. — VQCNN | 2024 | The Journal of Supercomputing (Springer, **Q1**) | Variational QCNN (VQNN) | KDD99 / NSL-KDD | 97.21% precision | Single-family dataset; sim; classic-IDS comparison |
| 4 | Gong et al. — "Network attack detection via VQNN" | 2022 | The Journal of Supercomputing (**Q1**) | Variational QNN | KDD-type | High prec/recall/F1 | sim |
| 5 | Abreu, Rothenberg, Abelém — **QML-IDS** | 2024 | arXiv (precursor to IJNM) | VQC, QSVM, QCNN | UNSW-NB15, CIC-IDS2017, CIC-IoT2023 | F1: UNSW 88.1 / CICIDS 95.6 / CICIoT 82.6 | **Multi-dataset, multi-method, binary+multiclass — strongest comparative baseline**; sim |
| 6 | Abreu et al. — **QuantumNetSec** | 2025 | Int. J. Network Management (Wiley, **Q1/Q2**) | VQC/QSVM/QCNN + transpilation/circuit opt | 4 public NIDS datasets | Superior binary & multiclass (journal-extended QML-IDS) | NISQ-aware; sim + transpilation |
| 7 | Kumari, Pokhrel et al. — Wavelet QSVM (QWPT) | 2025 | arXiv (quant-ph/cs.LG) | QSVM + Quantum Haar Wavelet Packet Transform, fidelity kernel, SPSA | BoT-IoT, IoT-23 | BoT-IoT 96.67%, IoT-23 89.67% acc | Tested noiseless + depolarizing noise; +7pp vs QAE |
| 8 | Chaudhary, Rajasegarar, Pokhrel — **Q-AGNN** | 2026 | arXiv | Quantum-enhanced attentive **graph** NN; PQC + EfficientSU2 | BoT-IoT, UNSW-NB15, NF-BoT-IoT, NF-UNSW-NB15 | BoT-IoT 100% / UNSW 97.87% acc; F1 up to 1.00 | Honest: "moderate expressivity advantage over shallow MLPs"; noise-sensitive |
| 9 | Bhatnagar, Innan, Shafique et al. — **Meta-Quantum Ensemble (MQE)** | 2026 | arXiv | QSVM + QNN fused by Random Forest meta-learner | CIC-IDS2017, TON_IoT | CICIDS 97.14% acc/81.5% F1; TON 98.57%/99.05% | **Best evaluation rigor**: AUPRC, TPR@0.1%/1% FPR, Brier, ECE, 5 noise channels; admits RF/XGBoost still superior |
| 10 | Bellante, Carminati, Zanero, Luongo et al. — PCA-IDS case study | 2025 | Computers & Security (Elsevier, **Q1**) | Fault-tolerant QML for PCA-based IDS | Standard NIDS datasets | Conditional advantage only | **Critical/rigorous**; advantage "explainable by regularization, not quantum effects" |
| 11 | "Towards NIDS via QML: A Reality Check" | 2025 | (preprint/ResearchGate) | QSVM/VQC vs classical | NIDS benchmarks | Quantum ≈ or < classical | Skeptical reality-check; cite for balance |
| 12 | EPJ-QT — Unified HQCNN (DDoS + Android malware) | 2025 | EPJ Quantum Technology (Springer, **Q1**) | Hybrid QCNN, dressed quantum circuit | SDN DDoS + Android malware | DDoS: 99.86% acc, 100% recall, 99.88% F1 | **Run on AWS Braket real QPU** (rare); binary |
| 13 | EPJ-QT — Quantum-driven enhanced ML for IoT IDS | 2026 | EPJ Quantum Technology (**Q1**) | Hybrid QNN (HQNN) | 1.05M-sample IoT flow dataset, 76 features (Mirai/Gafgyt etc.) | High acc (IoT real-time framing) | Large dataset; sim |
| 14 | Wang, Fok, Thing (ST Engineering) — Hybrid Quan-Conv CNN | 2025 | arXiv | 5 hybrid QCNN architectures (angle/amplitude embed) | 5G-NIDD (1.2M sessions, 8 attacks) | Std split: 99.71% (vs CNN 99.61); unseen-attack 87.84% | **Cross-base-station + unseen-attack generalization tests** (rare, valuable); 4–16 qubits |
| 15 | QSVM-IGWO | 2024 | Cluster Computing (Springer, **Q1**) | QSVM + Improved Grey Wolf Optimizer feature selection | NSL-KDD | High acc/F1 (FPR reduction claimed) | Single dataset; sim |
| 16 | FedCQIDS — Federated + QCNN | 2025 | (J. of Computer Research & Development / crad.ict.ac.cn) | Federated learning + lightweight client-side QCNN | CICIDS-class | Handles imbalance/sparse features | Privacy-preserving angle; sim |
| 17 | Hybrid quantum-enhanced federated learning | 2024 | Scientific Reports (Nature, **Q1**) | Quantum-enhanced FL for cyberattack detection | Cyberattack/NIDS data | Strong acc (FL setting) | sim |
| 18 | Quantum deep-learning anomaly detection | 2024 | Quantum Machine Intelligence (Springer, Q2) | Quantum autoencoder + Q-OC-SVM / Q-RF / Q-kNN | Network anomaly data | Improved anomaly detection | Unsupervised/semi-sup; sim |
| 19 | Quantum IDS using outlier analysis | 2024 | Scientific Reports (Nature, **Q1**) | Quantum outlier/anomaly analysis | NIDS | Outlier-based detection | sim |
| 20 | Hybrid Quantum-Classical Autoencoders (unsupervised NIDS) | 2025 | arXiv | Hybrid quantum-classical autoencoders | UNSW-NB15, NSL-KDD, CIC-IDS2017 | Unsupervised reconstruction-based detection | **3-dataset comparison**; unsupervised; sim |
| 21 | Quantum-Driven Chaos-Informed DL framework | 2025 | Technologies (MDPI, Q2) | Quantum diffusion feature selection + quantum-attention graph classifier + chaos | NSL-KDD, UNSW-NB15 | Superior FS + classification | Heavy classical hybridization; sim |
| 22 | Quantum GAN + successive data injection (time series) | 2025 | arXiv | QGAN, VQC, data re-uploading, window shifting | Multivariate time-series network anomaly | Improved anomaly detection | Generative; sim |
| 23 | Quantum-aware secure blockchain IDS (IIoT) | 2025 | Scientific Reports (Nature, **Q1**) | Quantum-aware + blockchain IDS | Industrial IoT | Secure IIoT detection | Systems-oriented; sim |
| 24 | Quantum Genetic Algorithm self-supervised IDS (WSN/IoT) | 2025 | arXiv | Quantum genetic algorithm + self-supervised | WSN/IoT | Self-supervised detection | sim |
| 25 | QCNN-ID: Quantum-Classical Hybrid for IoT IDS | 2025 | Procedia CS / ScienceDirect (Elsevier) | Hybrid QCNN | IoT attack dataset(s) | QCNN > classical CNN (acc/prec/recall) | sim |
| 26 | Quantum-Enhanced ML for malicious URL detection | 2025 | Electronics (MDPI, Q2) | Quantum-enhanced classifiers | URL/phishing | Comparable/better detection | Adjacent (URL not flow); sim |
| 27 | High-efficiency VQC for high-dimensional data | 2024 | The Journal of Supercomputing (**Q1**) | Efficient VQC | High-dim (incl. security) | Efficiency-focused VQC | Method paper; relevant for encoding |
| 28 | VQNN with dynamic layerwise strategy | 2025 | The Journal of Supercomputing (**Q1**) | Adaptive-depth VQNN | (incl. classification benchmarks) | Improved trainability | Architecture/trainability angle |
| 29 | Privacy-aware hybrid quantum + DNN malware (indoor robots) | 2025 | arXiv | Hybrid quantum + DNN | Malware | Robust malware detection | Adjacent domain; sim |
| 30 | Hybrid classical-quantum NN for DDoS in SD vehicular nets | 2025 | Information (MDPI, Q2) | Hybrid classical-quantum NN | SDVN DDoS | DDoS detection | Domain-specific; sim |
| 31 | Federated quantum-inspired anomaly detection | 2025 | PMC / Sci. Reports-class | Quantum-*inspired* (not true quantum) FL anomaly | Network anomaly | Collaborative detection | "Quantum-inspired" caveat; sim |
| 32 | QML-IDS systematic mapping study | 2023 | Springer (book chapter) | Survey | — | Maps QML-IDS landscape | Useful for framing/coverage |
| 33 | Sai, Chamola, Buyya et al. — QML for Cybersecurity: Taxonomy | 2025 | arXiv | Survey/taxonomy | — | Taxonomy + future directions | Cite for taxonomy & gaps |
| 34 | Survey: Adapting Federated & QML for NIDS | 2025 | arXiv | Survey | — | FL+QML for NIDS roadmap | Cite for QFL gaps |
| 35 | Systematic review: anomaly detection in IoT toward QML | 2025 | EPJ Quantum Technology (**Q1**) | Systematic review | — | IoT anomaly → QML roadmap | Cite for IoT framing |
| 36 | QML algorithms for anomaly detection: a review | 2024 | arXiv | Survey | — | Reviews QAE/QSVM/QkNN etc. | Cite for method taxonomy |

---

## 3. Thematic Synthesis

### 3.1 Methods landscape
- **VQC / VQNN (parameterized quantum circuits).** The most common primitive. Used standalone (Gong et al., J. Supercomputing) or as the quantum head of a hybrid net. Trainability (barren plateaus) and encoding (angle vs amplitude) are recurring concerns; amplitude embedding repeatedly reported as unstable (Wang et al.).
- **QSVM (quantum-kernel SVM).** Second most common. Strength: principled kernel view, fidelity/ZZ feature maps. Often paired with classical feature selection/optimization (QSVM-IGWO; wavelet QWPT-QSVM). Scales poorly with sample count → almost always run on heavily subsampled/PCA-reduced data.
- **QCNN.** Popular for "image-like" reshaped flow features; consistently reported to slightly edge classical CNN on the *same reduced* data but to underperform on small data (Wang et al.). Real-QPU QCNN demonstrated for DDoS (EPJ-QT 2025).
- **Quantum graph NNs (Q-AGNN, 2026).** Newest direction: encode flow features into PQC embeddings + classical attention over flow graphs. Honest about only "moderate" advantage.
- **Quantum autoencoders / one-class / outlier methods.** The unsupervised/anomaly branch (Quantum DL anomaly detection 2024; hybrid QC autoencoders 2025; quantum outlier IDS 2024). Important because real IDS deployment is imbalanced/semi-supervised.
- **Quantum reservoir computing.** Notably **scarce** for NIDS specifically — an under-explored method (gap).
- **Ensembles / meta-learning (MQE 2026)** and **Quantum Federated Learning (QFL)** (Sci. Reports 2024; FedCQIDS 2025; surveys) are the emerging "systems" frontier.
- **Generative (QGAN, 2025)** for synthetic attack/time-series anomaly detection.

### 3.2 Datasets landscape
- **NSL-KDD / KDD99:** Most common, but dated and criticized; still the default for VQC/QSVM demos (Gong; QSVM-IGWO; chaos framework).
- **UNSW-NB15:** Strong second; used by QML-IDS, Q-AGNN, hybrid autoencoders, chaos framework.
- **CIC-IDS2017:** Heavily used (QML-IDS, MQE, autoencoders). MQE explicitly flags it contains "harder subsets" that stay difficult for quantum learners.
- **TON_IoT / BoT-IoT / IoT-23:** The IoT cluster, rising fast (MQE, Q-AGNN, wavelet-QSVM, Kalinin custom). NetFlow variants (NF-BoT-IoT, NF-UNSW-NB15) appearing (Q-AGNN).
- **CIC-IoT2023, 5G-NIDD:** Newest/largest; used by QML-IDS (CICIoT2023) and Wang et al. (5G-NIDD).
- **Pattern:** Most papers use **1–2 datasets**. Genuine **multi-dataset, cross-dataset** evaluation is rare (QML-IDS/QuantumNetSec, hybrid autoencoders, Q-AGNN, MQE are the main exceptions). This is a key differentiator for us.

### 3.3 Metrics & evaluation rigor
- Headline **accuracy/F1** dominate; reported numbers cluster 88–99.9%.
- **Class-imbalance-aware metrics** (AUPRC, TPR at fixed low FPR), **calibration** (Brier, ECE), and **statistical significance** are almost entirely absent — **MQE (2026) is the lone exemplar** and should be matched/exceeded.
- **Noise robustness:** Most results are noiseless simulator. Exceptions worth citing: MQE (5 noise channels), Q-AGNN (depolarizing), wavelet-QSVM (depolarizing), and the EPJ-QT DDoS paper + others that ran on **real QPUs** (rare and citable).
- **Honest classical baselines:** Frequently weak or absent. The rigorous papers (Bellante; MQE; Reality Check) all conclude classical RF/XGBoost remain competitive or superior on tabular IDS — a claim our paper must engage directly.

### 3.4 Stated limitations / future work (recurring across the corpus)
1. NISQ hardware: qubit count, depth, noise limit scalability.
2. Encoding bottleneck: amplitude embedding instability; classical preprocessing (PCA/feature selection) doing much of the heavy lifting (so is the "quantum advantage" real?).
3. Scalability to >10^5–10^6 samples (QSVM kernels especially).
4. Lack of cross-dataset generalization and unseen-attack evaluation.
5. Sim-to-hardware gap; few real-QPU results.
6. Integration with operational classical IDS pipelines; latency/throughput unaddressed.
7. Reproducibility: code/seeds/splits often unspecified.

---

## 4. Bibliography

1. Kalinin, M., Krundyshev, V. (2022). *Security intrusion detection using quantum machine learning techniques.* Journal of Computer Virology and Hacking Techniques, 19(1):125–136. https://link.springer.com/article/10.1007/s11416-022-00435-0 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC9226281/
2. Gong, C. et al. (2024). *Network intrusion detection based on variational quantum convolution neural network.* The Journal of Supercomputing. https://link.springer.com/article/10.1007/s11227-024-05919-y (ACM mirror: https://dl.acm.org/doi/10.1007/s11227-024-05919-y)
3. Gong, C. et al. (2022). *Network attack detection scheme based on variational quantum neural network.* The Journal of Supercomputing. https://link.springer.com/article/10.1007/s11227-022-04542-z
4. Abreu, D., Rothenberg, C. E., Abelém, A. (2024). *QML-IDS: Quantum Machine Learning Intrusion Detection System.* arXiv:2410.16308. https://arxiv.org/abs/2410.16308 ; https://arxiv.org/html/2410.16308v1
5. Abreu, D. et al. (2025). *QuantumNetSec: Quantum Machine Learning for Network Security.* International Journal of Network Management (Wiley). https://onlinelibrary.wiley.com/doi/10.1002/nem.70018
6. Kumari, S., Pokhrel, S. R. et al. (2025). *Modeling Wavelet Transformed Quantum Support Vector for Network Intrusion Detection.* arXiv:2512.01365. https://arxiv.org/abs/2512.01365
7. Chaudhary, D., Rajasegarar, S., Pokhrel, S. R. (2026). *Q-AGNN: Quantum-Enhanced Attentive Graph Neural Network for Intrusion Detection.* arXiv:2603.22365. https://arxiv.org/pdf/2603.22365
8. Bhatnagar, R., Innan, N., Jothi, A. A., Shafique, M. (2026). *Meta-Quantum Ensemble Framework for Robust Network Intrusion Detection.* arXiv:2605.28879. https://arxiv.org/html/2605.28879
9. Bellante, A., Fioravanti, T., Carminati, M., Zanero, S., Luongo, A. (2025). *Evaluating the Potential of Quantum Machine Learning in Cybersecurity: A Case-Study on PCA-based Intrusion Detection Systems.* Computers & Security (Elsevier). arXiv:2502.11173. https://arxiv.org/abs/2502.11173
10. *Towards Network Intrusion Detection via Quantum Machine Learning: A Reality Check.* (2025). https://www.researchgate.net/publication/389164564
11. *Unified hybrid quantum classical neural network framework for detecting DDoS and Android mobile malware attacks.* (2025). EPJ Quantum Technology. https://link.springer.com/article/10.1140/epjqt/s40507-025-00380-z
12. Bharathi, Sonai, S. (2026). *Quantum-driven enhanced machine learning algorithm for intrusion detection in IoT environment.* EPJ Quantum Technology, 13:20. https://link.springer.com/article/10.1140/epjqt/s40507-026-00463-5
13. Wang, Z., Fok, K.-W., Thing, V. L. L. (2025). *Network Attack Traffic Detection With Hybrid Quantum-Enhanced Convolution Neural Network.* arXiv:2504.20436. https://arxiv.org/html/2504.20436v1
14. *A novel intrusion detection system based on a hybrid quantum support vector machine and improved Grey Wolf optimizer (QSVM-IGWO).* (2024). Cluster Computing (Springer). https://link.springer.com/article/10.1007/s10586-024-04458-8
15. *An Intrusion Detection Model Integrating Federated Learning and Quantum Convolutional Neural Networks (FedCQIDS).* (2025). J. Computer Research & Development. https://crad.ict.ac.cn/en/article/doi/10.7544/issn1000-1239.202550413
16. *Hybrid quantum enhanced federated learning for cyber attack detection.* (2024). Scientific Reports (Nature). https://www.nature.com/articles/s41598-024-83682-z
17. *Quantum deep learning-based anomaly detection for enhanced network security.* (2024). Quantum Machine Intelligence (Springer). https://link.springer.com/article/10.1007/s42484-024-00163-2
18. *Quantum intrusion detection system using outlier analysis.* (2024). Scientific Reports (Nature). https://www.nature.com/articles/s41598-024-78389-0
19. *Hybrid Quantum-Classical Autoencoders for Unsupervised Network Intrusion Detection.* (2025). arXiv:2512.05069. https://arxiv.org/pdf/2512.05069
20. *Quantum-Driven Chaos-Informed Deep Learning Framework for Efficient Feature Selection and Intrusion Detection in IoT Networks.* (2025). Technologies (MDPI). https://doi.org/10.3390/technologies13100470
21. *Enhancing Network Anomaly Detection with Quantum GANs and Successive Data Injection for Multivariate Time Series.* (2025). arXiv:2505.11631. https://arxiv.org/pdf/2505.11631
22. *Quantum-aware secure blockchain intrusion detection system for industrial IoT networks.* (2025). Scientific Reports (Nature). https://www.nature.com/articles/s41598-025-31985-0
23. *A Quantum Genetic Algorithm-Enhanced Self-Supervised Intrusion Detection System for WSN in IoT.* (2025). arXiv:2509.03744. https://arxiv.org/pdf/2509.03744
24. *QCNN-ID: A Quantum-Classical Hybrid Model for IoT Intrusion Detection.* (2025). Procedia CS / ScienceDirect (Elsevier). https://www.sciencedirect.com/science/article/pii/S1877050925030133 ; https://hal.science/hal-05080861v1/document
25. *Quantum-Enhanced Machine Learning for Cybersecurity: Evaluating Malicious URL Detection.* (2025). Electronics (MDPI). https://www.mdpi.com/2079-9292/14/9/1827
26. *A high-efficiency variational quantum classifier for high-dimensional data.* (2024). The Journal of Supercomputing. https://link.springer.com/article/10.1007/s11227-024-06676-8
27. *Variational quantum neural networks based on dynamic layerwise strategy.* (2025). The Journal of Supercomputing. https://link.springer.com/article/10.1007/s11227-025-07394-5
28. *A case study for cyber-attack detection using quantum variational circuits.* (2025). Quantum Machine Intelligence (Springer). https://link.springer.com/article/10.1007/s42484-025-00277-1
29. *Privacy-Aware Framework of Robust Malware Detection in Indoor Robots: Hybrid Quantum Computing and DNN.* (2025). arXiv:2510.13136. https://arxiv.org/html/2510.13136v1
30. *A Hybrid Classical-Quantum Neural Network Model for DDoS Attack Detection in Software-Defined Vehicular Networks.* (2025). Information (MDPI). https://www.mdpi.com/2078-2489/16/9/722
31. *Federated quantum-inspired anomaly detection using collaborative neural clients.* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC12418598/
32. *Quantum Machine Learning in Intrusion Detection Systems: A Systematic Mapping Study.* (2023). Springer (book chapter). https://link.springer.com/chapter/10.1007/978-981-99-7886-1_9
33. Sai, S., Goyal, I., Sharma, S., Manuri, S. H., Chamola, V., Buyya, R. (2025). *Quantum Machine Learning for Cybersecurity: A Taxonomy and Future Directions.* arXiv:2512.15286. https://arxiv.org/pdf/2512.15286
34. *Towards Adapting Federated & Quantum Machine Learning for Network Intrusion Detection: A Survey.* (2025). arXiv:2509.21389. https://arxiv.org/pdf/2509.21389
35. *A systematic review of anomaly detection in IoT security: towards quantum machine learning approach.* (2025). EPJ Quantum Technology. https://link.springer.com/article/10.1140/epjqt/s40507-025-00414-6
36. *Quantum Machine Learning Algorithms for Anomaly Detection: a Review.* (2024). arXiv:2408.11047. https://arxiv.org/pdf/2408.11047

---

## 5. Shortlist: "Papers Our Novelty Must Differentiate From"

These are the works a Q1 reviewer will expect us to cite, benchmark against, and surpass. Grouped by why they matter.

### Tier A — Direct SOTA baselines (must reproduce/beat)
- **Abreu et al. — QML-IDS (2024) / QuantumNetSec (IJNM 2025)** [#4, #5]. The strongest *comparative* QML-IDS to date: 3+ datasets, three quantum methods, binary + multiclass, NISQ-aware. **This is the paper to beat on breadth.** We must report the same datasets (UNSW-NB15, CIC-IDS2017, CIC-IoT2023) and exceed their F1 (e.g., UNSW 88.1, CICIDS 95.6, CICIoT 82.6) with stronger evaluation.
- **Bhatnagar et al. — Meta-Quantum Ensemble (2026)** [#9]. The **gold standard for evaluation rigor** (AUPRC, TPR@0.1%/1% FPR, Brier, ECE, 5 noise channels). We must match or exceed this rigor, and ideally beat their CICIDS2017 (97.14% acc / 81.5% F1) and TON_IoT (98.6% / 99.05%) numbers — or beat their honesty by showing a real margin over RF/XGBoost, which they concede they do not.
- **Gong et al. — VQCNN, J. Supercomputing (2024)** [#2]. Most prominent Q1-journal VQC/QCNN result (97.21% precision on KDD/NSL-KDD). Must outperform on NSL-KDD with a fairer protocol.

### Tier B — High-credibility / real-hardware results (cite for legitimacy)
- **EPJ-QT Unified HQCNN (2025)** [#11]. One of the few **real-QPU** (AWS Braket) results; 99.86% DDoS accuracy. Differentiate by going beyond single-task binary DDoS to multi-dataset, multiclass, and low-FPR analysis.
- **Wang, Fok, Thing (2025)** [#13]. Rare **cross-base-station + unseen-attack** generalization study on 5G-NIDD. We should adopt a similar generalization protocol and show our model degrades less.
- **Kalinin & Krundyshev (2022)** [#1]. The **foundational / most-cited** QML-IDS paper. Mandatory citation; differentiate on standardized public datasets (they used a custom stream dataset).

### Tier C — Critical voices we must engage, not ignore
- **Bellante et al., Computers & Security (2025)** [#9 in table / ref 9]. Argues PCA-IDS quantum advantage is "explainable by regularization, not quantum effects." Since this is an **Elsevier Q1** venue (same target family), a reviewer will likely raise it. Our paper must directly address whether observed gains are genuinely quantum.
- **"Reality Check" (2025)** [ref 10] and surveys (Sai et al. 2025; FL+QML survey 2025) [refs 33–34]. Cite to frame the open problems and pre-empt the "is quantum even helping?" critique.

### Identified gaps our novelty can target (preview for gap-analyst)
1. **No standardized, fair, multi-dataset benchmark** with honest tuned classical baselines (RF/XGBoost/DL) across NSL-KDD + UNSW-NB15 + CIC-IDS2017 + CIC-IoT/TON_IoT under one protocol.
2. **Imbalance-aware + low-FPR operating-point evaluation** is almost absent (only MQE). Huge differentiator.
3. **Noise-robustness + sim-to-hardware** reporting is thin; combining systematic noise-channel study with (at least) one real-QPU confirmation would stand out.
4. **Cross-dataset / unseen-attack generalization** is rarely tested (only QML-IDS, Wang et al., Q-AGNN).
5. **Reproducibility** (public code, fixed seeds/splits, statistical significance) is routinely missing — easy, high-value win for a Q1 submission.
6. **Under-explored methods** for NIDS specifically: quantum reservoir computing, quantum GNNs at scale, and rigorous QFL — room for a novel architecture.
