# Quantum-Attribution Audit

*How much of the observed quantum performance is genuinely quantum, versus
classical preprocessing / capacity / regularization?* This directly addresses
Bellante et al. (Computers & Security, 2025).

## Method
Each quantum model (HybridQNN variants, QSVM) is compared against **matched
classical controls** on identical data splits and seeds:

1. **Capacity-matched MLP** -- classical MLP with trainable-parameter count
   matched to the HybridQNN (classical + quantum weights), not merely
   shape-matched.
2. **RBF-SVM** -- tuned classical kernel machine.
3. **Random-feature kernel** -- RBFSampler / Nystroem feature map + linear SVM,
   the classical analogue of the quantum kernel.
4. **Regularization sweep** -- classical regularization varied (MLP L2/dropout,
   RF depth) to test whether matched regularization closes any gap.

For each metric we report delta = (quantum) - (best matched control), with a
paired test across seeds. "Quantum better?" accounts for metric direction
(lower is better for ece, brier, false_positive_rate).

## Results
| dataset | metric | quantum | q-mean | control | c-mean | delta (q-c) | quantum better? | p | comparison |
|---|---|---|---:|---|---:|---:|:--:|---:|---|
| nslkdd | roc_auc | ovn1_hybrid_angle_q12 | 0.8701 | ClassicalMLP | 0.9373 | -0.0438 | no | n/a | best-control |
| nslkdd | auprc | ovn1_hybrid_angle_q12 | 0.9081 | ovn1_mlp_matched | 0.9439 | -0.0358 | no | 0.035 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovn1_hybrid_angle_q12 | 0.2660 | ovn1_mlp_matched | 0.2719 | -0.0059 | no | 0.966 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovn1_hybrid_angle_q12 | 0.3457 | rf | 0.4671 | -0.1214 | no | 0.026 | best-control |
| nslkdd | f1 | ovn1_hybrid_angle_q12 | 0.7244 | rf | 0.7636 | -0.0391 | no | 0.021 | best-control |
| nslkdd | ece | ovn1_hybrid_angle_q12 | 0.2222 | rf | 0.1928 | +0.0294 | no | 0.064 | best-control |
| nslkdd | roc_auc | ovn1_hybrid_angle_q4 | 0.8983 | ClassicalMLP | 0.9373 | -0.0486 | no | n/a | best-control |
| nslkdd | auprc | ovn1_hybrid_angle_q4 | 0.9285 | ovn1_mlp_matched | 0.9439 | -0.0154 | no | 0.219 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovn1_hybrid_angle_q4 | 0.4072 | ovn1_mlp_matched | 0.2719 | +0.1353 | yes | 0.428 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovn1_hybrid_angle_q4 | 0.5167 | rf | 0.4671 | +0.0496 | yes | 0.005 | best-control |
| nslkdd | f1 | ovn1_hybrid_angle_q4 | 0.7593 | rf | 0.7636 | -0.0043 | no | 0.776 | best-control |
| nslkdd | ece | ovn1_hybrid_angle_q4 | 0.1840 | rf | 0.1928 | -0.0088 | yes | 0.248 | best-control |
| nslkdd | roc_auc | ovnN_ampdamp_p01 | 0.8963 | ClassicalMLP | 0.9373 | -0.0446 | no | n/a | best-control |
| nslkdd | auprc | ovnN_ampdamp_p01 | 0.9226 | ovn1_mlp_matched | 0.9439 | -0.0178 | no | 0.455 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovnN_ampdamp_p01 | 0.0845 | ovn1_mlp_matched | 0.2719 | -0.1355 | no | 0.643 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovnN_ampdamp_p01 | 0.4296 | rf | 0.4671 | -0.0396 | no | 0.282 | best-control |
| nslkdd | f1 | ovnN_ampdamp_p01 | 0.7327 | rf | 0.7636 | -0.0303 | no | 0.423 | best-control |
| nslkdd | ece | ovnN_ampdamp_p01 | 0.1990 | rf | 0.1928 | +0.0050 | no | 0.660 | best-control |
| nslkdd | roc_auc | ovnN_ampdamp_p05 | 0.8885 | ClassicalMLP | 0.9373 | -0.0553 | no | n/a | best-control |
| nslkdd | auprc | ovnN_ampdamp_p05 | 0.9199 | ovn1_mlp_matched | 0.9439 | -0.0205 | no | 0.448 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovnN_ampdamp_p05 | 0.1083 | ovn1_mlp_matched | 0.2719 | -0.1117 | no | 0.729 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovnN_ampdamp_p05 | 0.4444 | rf | 0.4671 | -0.0248 | no | 0.591 | best-control |
| nslkdd | f1 | ovnN_ampdamp_p05 | 0.7352 | rf | 0.7636 | -0.0279 | no | 0.482 | best-control |
| nslkdd | ece | ovnN_ampdamp_p05 | 0.1900 | rf | 0.1928 | -0.0040 | yes | 0.843 | best-control |
| nslkdd | roc_auc | ovnN_bitflip_p05 | 0.8700 | ClassicalMLP | 0.9373 | -0.0730 | no | n/a | best-control |
| nslkdd | auprc | ovnN_bitflip_p05 | 0.9131 | ovn1_mlp_matched | 0.9439 | -0.0273 | no | 0.309 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovnN_bitflip_p05 | 0.0799 | ovn1_mlp_matched | 0.2719 | -0.1401 | no | 0.638 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovnN_bitflip_p05 | 0.3786 | rf | 0.4671 | -0.0907 | no | 0.060 | best-control |
| nslkdd | f1 | ovnN_bitflip_p05 | 0.7280 | rf | 0.7636 | -0.0350 | no | 0.133 | best-control |
| nslkdd | ece | ovnN_bitflip_p05 | 0.1995 | rf | 0.1928 | +0.0055 | no | 0.432 | best-control |
| nslkdd | roc_auc | ovnN_depol_p01 | 0.8781 | ClassicalMLP | 0.9373 | -0.0477 | no | n/a | best-control |
| nslkdd | auprc | ovnN_depol_p01 | 0.9153 | ovn1_mlp_matched | 0.9439 | -0.0251 | no | 0.229 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovnN_depol_p01 | 0.0816 | ovn1_mlp_matched | 0.2719 | -0.1384 | no | 0.640 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovnN_depol_p01 | 0.4249 | rf | 0.4671 | -0.0444 | no | 0.229 | best-control |
| nslkdd | f1 | ovnN_depol_p01 | 0.7348 | rf | 0.7636 | -0.0282 | no | 0.462 | best-control |
| nslkdd | ece | ovnN_depol_p01 | 0.1972 | rf | 0.1928 | +0.0032 | no | 0.767 | best-control |
| nslkdd | roc_auc | ovnN_depol_p05 | 0.8778 | ClassicalMLP | 0.9373 | -0.0581 | no | n/a | best-control |
| nslkdd | auprc | ovnN_depol_p05 | 0.9153 | ovn1_mlp_matched | 0.9439 | -0.0251 | no | 0.288 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovnN_depol_p05 | 0.0814 | ovn1_mlp_matched | 0.2719 | -0.1386 | no | 0.639 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovnN_depol_p05 | 0.4248 | rf | 0.4671 | -0.0445 | no | 0.092 | best-control |
| nslkdd | f1 | ovnN_depol_p05 | 0.7306 | rf | 0.7636 | -0.0325 | no | 0.433 | best-control |
| nslkdd | ece | ovnN_depol_p05 | 0.1999 | rf | 0.1928 | +0.0059 | no | 0.499 | best-control |
| nslkdd | roc_auc | ovnN_noiseless | 0.8455 | ClassicalMLP | 0.9373 | -0.1047 | no | n/a | best-control |
| nslkdd | auprc | ovnN_noiseless | 0.9043 | ovn1_mlp_matched | 0.9439 | -0.0361 | no | 0.324 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovnN_noiseless | 0.0829 | ovn1_mlp_matched | 0.2719 | -0.1371 | no | 0.657 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovnN_noiseless | 0.4225 | rf | 0.4671 | -0.0468 | no | 0.094 | best-control |
| nslkdd | f1 | ovnN_noiseless | 0.7329 | rf | 0.7636 | -0.0301 | no | 0.404 | best-control |
| nslkdd | ece | ovnN_noiseless | 0.1908 | rf | 0.1928 | -0.0032 | yes | 0.865 | best-control |
| nslkdd | roc_auc | ovnN_phase_p05 | 0.8970 | ClassicalMLP | 0.9373 | -0.0460 | no | n/a | best-control |
| nslkdd | auprc | ovnN_phase_p05 | 0.9225 | ovn1_mlp_matched | 0.9439 | -0.0179 | no | 0.462 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovnN_phase_p05 | 0.1419 | ovn1_mlp_matched | 0.2719 | -0.0781 | no | 0.753 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovnN_phase_p05 | 0.4206 | rf | 0.4671 | -0.0486 | no | 0.180 | best-control |
| nslkdd | f1 | ovnN_phase_p05 | 0.7325 | rf | 0.7636 | -0.0305 | no | 0.422 | best-control |
| nslkdd | ece | ovnN_phase_p05 | 0.1974 | rf | 0.1928 | +0.0034 | no | 0.812 | best-control |
| nslkdd | roc_auc | ovn1_hybrid_angle_be | 0.8477 | ClassicalMLP | 0.9373 | -0.0789 | no | n/a | best-control |
| nslkdd | auprc | ovn1_hybrid_angle_be | 0.9023 | ovn1_mlp_matched | 0.9439 | -0.0417 | no | 0.061 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovn1_hybrid_angle_be | 0.2837 | ovn1_mlp_matched | 0.2719 | +0.0118 | yes | 0.938 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovn1_hybrid_angle_be | 0.3280 | rf | 0.4671 | -0.1391 | no | 0.007 | best-control |
| nslkdd | f1 | ovn1_hybrid_angle_be | 0.7129 | rf | 0.7636 | -0.0507 | no | 0.004 | best-control |
| nslkdd | ece | ovn1_hybrid_angle_be | 0.2180 | rf | 0.1928 | +0.0252 | no | 0.009 | best-control |
| nslkdd | roc_auc | ovn1_hybrid_angle_se | 0.8253 | ClassicalMLP | 0.9373 | -0.1225 | no | n/a | best-control |
| nslkdd | auprc | ovn1_hybrid_angle_se | 0.8931 | ovn1_mlp_matched | 0.9439 | -0.0509 | no | 0.017 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovn1_hybrid_angle_se | 0.2756 | ovn1_mlp_matched | 0.2719 | +0.0038 | yes | 0.978 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovn1_hybrid_angle_se | 0.3663 | rf | 0.4671 | -0.1008 | no | 0.076 | best-control |
| nslkdd | f1 | ovn1_hybrid_angle_se | 0.7184 | rf | 0.7636 | -0.0452 | no | 0.025 | best-control |
| nslkdd | ece | ovn1_hybrid_angle_se | 0.2208 | rf | 0.1928 | +0.0279 | no | 0.071 | best-control |
| nslkdd | roc_auc | ovn2_nslkdd_hybrid_angle | 0.8253 | ClassicalMLP | 0.9373 | -0.1225 | no | n/a | best-control |
| nslkdd | auprc | ovn2_nslkdd_hybrid_angle | 0.8931 | ovn1_mlp_matched | 0.9439 | -0.0509 | no | 0.017 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovn2_nslkdd_hybrid_angle | 0.2756 | ovn1_mlp_matched | 0.2719 | +0.0038 | yes | 0.978 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovn2_nslkdd_hybrid_angle | 0.3663 | rf | 0.4671 | -0.1008 | no | 0.076 | best-control |
| nslkdd | f1 | ovn2_nslkdd_hybrid_angle | 0.7184 | rf | 0.7636 | -0.0452 | no | 0.025 | best-control |
| nslkdd | ece | ovn2_nslkdd_hybrid_angle | 0.2208 | rf | 0.1928 | +0.0279 | no | 0.071 | best-control |
| nslkdd | roc_auc | qnn_angle | 0.8253 | ClassicalMLP | 0.9373 | -0.1225 | no | n/a | best-control |
| nslkdd | auprc | qnn_angle | 0.8931 | ovn1_mlp_matched | 0.9439 | -0.0509 | no | 0.017 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | qnn_angle | 0.2756 | ovn1_mlp_matched | 0.2719 | +0.0038 | yes | 0.978 | best-control |
| nslkdd | tpr_at_1pct_fpr | qnn_angle | 0.3663 | rf | 0.4671 | -0.1008 | no | 0.076 | best-control |
| nslkdd | f1 | qnn_angle | 0.7184 | rf | 0.7636 | -0.0452 | no | 0.025 | best-control |
| nslkdd | ece | qnn_angle | 0.2208 | rf | 0.1928 | +0.0279 | no | 0.071 | best-control |
| nslkdd | roc_auc | verify_sub | 0.9221 | ClassicalMLP | 0.9373 | -0.0151 | no | n/a | best-control |
| nslkdd | auprc | verify_sub | 0.9442 | ovn1_mlp_matched | 0.9439 | -0.0088 | no | n/a | best-control |
| nslkdd | tpr_at_0.1pct_fpr | verify_sub | 0.2735 | ovn1_mlp_matched | 0.2719 | -0.1664 | no | n/a | best-control |
| nslkdd | tpr_at_1pct_fpr | verify_sub | 0.5507 | rf | 0.4671 | +0.0797 | yes | n/a | best-control |
| nslkdd | f1 | verify_sub | 0.7121 | rf | 0.7636 | -0.0524 | no | n/a | best-control |
| nslkdd | ece | verify_sub | 0.1935 | rf | 0.1928 | +0.0003 | no | n/a | best-control |
| nslkdd | roc_auc | ovn1_hybrid_iqp_se | 0.8693 | ClassicalMLP | 0.9373 | -0.0821 | no | n/a | best-control |
| nslkdd | auprc | ovn1_hybrid_iqp_se | 0.9159 | ovn1_mlp_matched | 0.9439 | -0.0281 | no | 0.099 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovn1_hybrid_iqp_se | 0.2316 | ovn1_mlp_matched | 0.2719 | -0.0403 | no | 0.872 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovn1_hybrid_iqp_se | 0.4515 | rf | 0.4671 | -0.0156 | no | 0.533 | best-control |
| nslkdd | f1 | ovn1_hybrid_iqp_se | 0.7429 | rf | 0.7636 | -0.0206 | no | 0.237 | best-control |
| nslkdd | ece | ovn1_hybrid_iqp_se | 0.1812 | rf | 0.1928 | -0.0116 | yes | 0.559 | best-control |
| nslkdd | roc_auc | ovn2_nslkdd_hybrid_iqp | 0.8693 | ClassicalMLP | 0.9373 | -0.0821 | no | n/a | best-control |
| nslkdd | auprc | ovn2_nslkdd_hybrid_iqp | 0.9159 | ovn1_mlp_matched | 0.9439 | -0.0281 | no | 0.099 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovn2_nslkdd_hybrid_iqp | 0.2316 | ovn1_mlp_matched | 0.2719 | -0.0403 | no | 0.872 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovn2_nslkdd_hybrid_iqp | 0.4515 | rf | 0.4671 | -0.0156 | no | 0.533 | best-control |
| nslkdd | f1 | ovn2_nslkdd_hybrid_iqp | 0.7429 | rf | 0.7636 | -0.0206 | no | 0.237 | best-control |
| nslkdd | ece | ovn2_nslkdd_hybrid_iqp | 0.1812 | rf | 0.1928 | -0.0116 | yes | 0.559 | best-control |
| nslkdd | roc_auc | ovn1_qsvm_angle_proj | 0.8548 | ClassicalMLP | 0.9373 | -0.0824 | no | n/a | best-control |
| nslkdd | roc_auc | ovn1_qsvm_angle_proj | 0.8548 | rf_kernel_nystroem | 0.8113 | +0.0435 | yes | 0.015 | qkernel-vs-rf |
| nslkdd | roc_auc | ovn1_qsvm_angle_proj | 0.8548 | rf_kernel_rbf | 0.7976 | +0.0573 | yes | 0.049 | qkernel-vs-rf |
| nslkdd | auprc | ovn1_qsvm_angle_proj | 0.8905 | ovn1_mlp_matched | 0.9439 | -0.0534 | no | 0.022 | best-control |
| nslkdd | auprc | ovn1_qsvm_angle_proj | 0.8905 | rf_kernel_nystroem | 0.8819 | +0.0086 | yes | 0.080 | qkernel-vs-rf |
| nslkdd | auprc | ovn1_qsvm_angle_proj | 0.8905 | rf_kernel_rbf | 0.8712 | +0.0193 | yes | 0.106 | qkernel-vs-rf |
| nslkdd | tpr_at_0.1pct_fpr | ovn1_qsvm_angle_proj | 0.0023 | ovn1_mlp_matched | 0.2719 | -0.2695 | no | 0.188 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovn1_qsvm_angle_proj | 0.0023 | rf_kernel_nystroem | 0.1378 | -0.1355 | no | 0.009 | qkernel-vs-rf |
| nslkdd | tpr_at_0.1pct_fpr | ovn1_qsvm_angle_proj | 0.0023 | rf_kernel_rbf | 0.1487 | -0.1464 | no | 0.058 | qkernel-vs-rf |
| nslkdd | tpr_at_1pct_fpr | ovn1_qsvm_angle_proj | 0.0472 | rf | 0.4671 | -0.4199 | no | 0.000 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovn1_qsvm_angle_proj | 0.0472 | rf_kernel_nystroem | 0.3180 | -0.2708 | no | 0.005 | qkernel-vs-rf |
| nslkdd | tpr_at_1pct_fpr | ovn1_qsvm_angle_proj | 0.0472 | rf_kernel_rbf | 0.3493 | -0.3021 | no | 0.029 | qkernel-vs-rf |
| nslkdd | f1 | ovn1_qsvm_angle_proj | 0.7074 | rf | 0.7636 | -0.0562 | no | 0.000 | best-control |
| nslkdd | f1 | ovn1_qsvm_angle_proj | 0.7074 | rf_kernel_nystroem | 0.7487 | -0.0413 | no | 0.004 | qkernel-vs-rf |
| nslkdd | f1 | ovn1_qsvm_angle_proj | 0.7074 | rf_kernel_rbf | 0.7454 | -0.0380 | no | 0.068 | qkernel-vs-rf |
| nslkdd | ece | ovn1_qsvm_angle_proj | 0.2733 | rf | 0.1928 | +0.0805 | no | 0.000 | best-control |
| nslkdd | ece | ovn1_qsvm_angle_proj | 0.2733 | rf_kernel_nystroem | 0.2366 | +0.0367 | no | 0.013 | qkernel-vs-rf |
| nslkdd | ece | ovn1_qsvm_angle_proj | 0.2733 | rf_kernel_rbf | 0.2541 | +0.0192 | no | 0.226 | qkernel-vs-rf |
| nslkdd | roc_auc | ovn1_qsvm_iqp_proj | 0.9453 | ClassicalMLP | 0.9373 | +0.0080 | yes | n/a | best-control |
| nslkdd | roc_auc | ovn1_qsvm_iqp_proj | 0.9453 | rf_kernel_nystroem | 0.8113 | +0.1340 | yes | 0.002 | qkernel-vs-rf |
| nslkdd | roc_auc | ovn1_qsvm_iqp_proj | 0.9453 | rf_kernel_rbf | 0.7976 | +0.1477 | yes | 0.008 | qkernel-vs-rf |
| nslkdd | auprc | ovn1_qsvm_iqp_proj | 0.9568 | ovn1_mlp_matched | 0.9439 | +0.0129 | yes | 0.253 | best-control |
| nslkdd | auprc | ovn1_qsvm_iqp_proj | 0.9568 | rf_kernel_nystroem | 0.8819 | +0.0749 | yes | 0.001 | qkernel-vs-rf |
| nslkdd | auprc | ovn1_qsvm_iqp_proj | 0.9568 | rf_kernel_rbf | 0.8712 | +0.0856 | yes | 0.006 | qkernel-vs-rf |
| nslkdd | tpr_at_0.1pct_fpr | ovn1_qsvm_iqp_proj | 0.0421 | ovn1_mlp_matched | 0.2719 | -0.2297 | no | 0.236 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovn1_qsvm_iqp_proj | 0.0421 | rf_kernel_nystroem | 0.1378 | -0.0957 | no | 0.018 | qkernel-vs-rf |
| nslkdd | tpr_at_0.1pct_fpr | ovn1_qsvm_iqp_proj | 0.0421 | rf_kernel_rbf | 0.1487 | -0.1066 | no | 0.102 | qkernel-vs-rf |
| nslkdd | tpr_at_1pct_fpr | ovn1_qsvm_iqp_proj | 0.4701 | rf | 0.4671 | +0.0030 | yes | 0.337 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovn1_qsvm_iqp_proj | 0.4701 | rf_kernel_nystroem | 0.3180 | +0.1521 | yes | 0.017 | qkernel-vs-rf |
| nslkdd | tpr_at_1pct_fpr | ovn1_qsvm_iqp_proj | 0.4701 | rf_kernel_rbf | 0.3493 | +0.1208 | yes | 0.148 | qkernel-vs-rf |
| nslkdd | f1 | ovn1_qsvm_iqp_proj | 0.7519 | rf | 0.7636 | -0.0116 | no | 0.007 | best-control |
| nslkdd | f1 | ovn1_qsvm_iqp_proj | 0.7519 | rf_kernel_nystroem | 0.7487 | +0.0033 | yes | 0.327 | qkernel-vs-rf |
| nslkdd | f1 | ovn1_qsvm_iqp_proj | 0.7519 | rf_kernel_rbf | 0.7454 | +0.0065 | yes | 0.597 | qkernel-vs-rf |
| nslkdd | ece | ovn1_qsvm_iqp_proj | 0.2832 | rf | 0.1928 | +0.0903 | no | 0.000 | best-control |
| nslkdd | ece | ovn1_qsvm_iqp_proj | 0.2832 | rf_kernel_nystroem | 0.2366 | +0.0466 | no | 0.008 | qkernel-vs-rf |
| nslkdd | ece | ovn1_qsvm_iqp_proj | 0.2832 | rf_kernel_rbf | 0.2541 | +0.0290 | no | 0.120 | qkernel-vs-rf |
| nslkdd | roc_auc | ovn2_nslkdd_qsvm | 0.9453 | ClassicalMLP | 0.9373 | +0.0080 | yes | n/a | best-control |
| nslkdd | roc_auc | ovn2_nslkdd_qsvm | 0.9453 | rf_kernel_nystroem | 0.8113 | +0.1340 | yes | 0.002 | qkernel-vs-rf |
| nslkdd | roc_auc | ovn2_nslkdd_qsvm | 0.9453 | rf_kernel_rbf | 0.7976 | +0.1477 | yes | 0.008 | qkernel-vs-rf |
| nslkdd | auprc | ovn2_nslkdd_qsvm | 0.9568 | ovn1_mlp_matched | 0.9439 | +0.0129 | yes | 0.253 | best-control |
| nslkdd | auprc | ovn2_nslkdd_qsvm | 0.9568 | rf_kernel_nystroem | 0.8819 | +0.0749 | yes | 0.001 | qkernel-vs-rf |
| nslkdd | auprc | ovn2_nslkdd_qsvm | 0.9568 | rf_kernel_rbf | 0.8712 | +0.0856 | yes | 0.006 | qkernel-vs-rf |
| nslkdd | tpr_at_0.1pct_fpr | ovn2_nslkdd_qsvm | 0.0421 | ovn1_mlp_matched | 0.2719 | -0.2297 | no | 0.236 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | ovn2_nslkdd_qsvm | 0.0421 | rf_kernel_nystroem | 0.1378 | -0.0957 | no | 0.019 | qkernel-vs-rf |
| nslkdd | tpr_at_0.1pct_fpr | ovn2_nslkdd_qsvm | 0.0421 | rf_kernel_rbf | 0.1487 | -0.1066 | no | 0.102 | qkernel-vs-rf |
| nslkdd | tpr_at_1pct_fpr | ovn2_nslkdd_qsvm | 0.4701 | rf | 0.4671 | +0.0030 | yes | 0.337 | best-control |
| nslkdd | tpr_at_1pct_fpr | ovn2_nslkdd_qsvm | 0.4701 | rf_kernel_nystroem | 0.3180 | +0.1521 | yes | 0.017 | qkernel-vs-rf |
| nslkdd | tpr_at_1pct_fpr | ovn2_nslkdd_qsvm | 0.4701 | rf_kernel_rbf | 0.3493 | +0.1208 | yes | 0.148 | qkernel-vs-rf |
| nslkdd | f1 | ovn2_nslkdd_qsvm | 0.7519 | rf | 0.7636 | -0.0116 | no | 0.007 | best-control |
| nslkdd | f1 | ovn2_nslkdd_qsvm | 0.7519 | rf_kernel_nystroem | 0.7487 | +0.0033 | yes | 0.327 | qkernel-vs-rf |
| nslkdd | f1 | ovn2_nslkdd_qsvm | 0.7519 | rf_kernel_rbf | 0.7454 | +0.0065 | yes | 0.597 | qkernel-vs-rf |
| nslkdd | ece | ovn2_nslkdd_qsvm | 0.2832 | rf | 0.1928 | +0.0903 | no | 0.000 | best-control |
| nslkdd | ece | ovn2_nslkdd_qsvm | 0.2832 | rf_kernel_nystroem | 0.2366 | +0.0466 | no | 0.008 | qkernel-vs-rf |
| nslkdd | ece | ovn2_nslkdd_qsvm | 0.2832 | rf_kernel_rbf | 0.2541 | +0.0290 | no | 0.120 | qkernel-vs-rf |
| nslkdd | roc_auc | qsvm_proj | 0.8548 | ClassicalMLP | 0.9373 | -0.0824 | no | n/a | best-control |
| nslkdd | roc_auc | qsvm_proj | 0.8548 | rf_kernel_nystroem | 0.8113 | +0.0435 | yes | 0.015 | qkernel-vs-rf |
| nslkdd | roc_auc | qsvm_proj | 0.8548 | rf_kernel_rbf | 0.7976 | +0.0572 | yes | 0.049 | qkernel-vs-rf |
| nslkdd | auprc | qsvm_proj | 0.8905 | ovn1_mlp_matched | 0.9439 | -0.0534 | no | 0.022 | best-control |
| nslkdd | auprc | qsvm_proj | 0.8905 | rf_kernel_nystroem | 0.8819 | +0.0086 | yes | 0.080 | qkernel-vs-rf |
| nslkdd | auprc | qsvm_proj | 0.8905 | rf_kernel_rbf | 0.8712 | +0.0193 | yes | 0.106 | qkernel-vs-rf |
| nslkdd | tpr_at_0.1pct_fpr | qsvm_proj | 0.0023 | ovn1_mlp_matched | 0.2719 | -0.2695 | no | 0.188 | best-control |
| nslkdd | tpr_at_0.1pct_fpr | qsvm_proj | 0.0023 | rf_kernel_nystroem | 0.1378 | -0.1355 | no | 0.009 | qkernel-vs-rf |
| nslkdd | tpr_at_0.1pct_fpr | qsvm_proj | 0.0023 | rf_kernel_rbf | 0.1487 | -0.1464 | no | 0.058 | qkernel-vs-rf |
| nslkdd | tpr_at_1pct_fpr | qsvm_proj | 0.0472 | rf | 0.4671 | -0.4199 | no | 0.000 | best-control |
| nslkdd | tpr_at_1pct_fpr | qsvm_proj | 0.0472 | rf_kernel_nystroem | 0.3180 | -0.2708 | no | 0.005 | qkernel-vs-rf |
| nslkdd | tpr_at_1pct_fpr | qsvm_proj | 0.0472 | rf_kernel_rbf | 0.3493 | -0.3021 | no | 0.029 | qkernel-vs-rf |
| nslkdd | f1 | qsvm_proj | 0.7074 | rf | 0.7636 | -0.0562 | no | 0.000 | best-control |
| nslkdd | f1 | qsvm_proj | 0.7074 | rf_kernel_nystroem | 0.7487 | -0.0413 | no | 0.004 | qkernel-vs-rf |
| nslkdd | f1 | qsvm_proj | 0.7074 | rf_kernel_rbf | 0.7454 | -0.0380 | no | 0.068 | qkernel-vs-rf |
| nslkdd | ece | qsvm_proj | 0.2733 | rf | 0.1928 | +0.0805 | no | 0.000 | best-control |
| nslkdd | ece | qsvm_proj | 0.2733 | rf_kernel_nystroem | 0.2366 | +0.0367 | no | 0.013 | qkernel-vs-rf |
| nslkdd | ece | qsvm_proj | 0.2733 | rf_kernel_rbf | 0.2541 | +0.0192 | no | 0.226 | qkernel-vs-rf |

## Interpretation
- A **small or negative** delta against the capacity-matched MLP indicates the
  advantage is a capacity/regularization effect, not a quantum one.
- A random-feature kernel that **matches** the QSVM indicates the quantum kernel
  is classically simulable in effect.
- Genuine quantum advantage requires a **positive, significant** delta against the
  *best* matched control across metrics.


> Auto-generated from results/*.json by attribution.run_attribution_audit. 'best control' is chosen per-metric (direction-aware). Brier/ECE for the rf_kernel control may be NaN (LinearSVC has no calibrated probabilities) and are excluded from those rows.
