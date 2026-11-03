# F014 layer 4 — L_phys-vs-error calibration

Per-window L_phys (SDOF self-consistency, amplitude-blind by design) vs
per-window RMSE, per fold-variant. Design promise: L_phys as uncertainty proxy.

## Correlations

| tag | n | pearson | spearman |
|---|---|---|---|
| pretrained_Off_set1_data_only | 89 | nan | nan |
| pretrained_Off_set1_both | 89 | -0.178 | -0.111 |
| pretrained_Off_set2_data_only | 90 | 0.191 | 0.112 |
| pretrained_Off_set2_both | 90 | 0.191 | 0.085 |
| pretrained_On_set3_data_only | 89 | -0.039 | 0.008 |
| pretrained_On_set3_both | 89 | nan | nan |
| ridge_Off_set1 | 89 | -0.402 | -0.629 |
| ridge_Off_set2 | 90 | -0.297 | -0.617 |
| ridge_Off_set3 | 82 | -0.496 | -0.838 |
| ridge_On_set1 | 89 | -0.232 | -0.709 |
| ridge_On_set2 | 89 | -0.394 | -0.675 |
| ridge_On_set3 | 89 | -0.493 | -0.736 |

Finding: 6/12 fold-variants have |pearson| < 0.3 —
Ridge subset anti-correlates (spearman -0.84..-0.62, all negative): the proxy points the wrong way there.
Pretrained rows are null (|r|<0.3 or constant L_phys, hence nan).
L_phys carries no usable per-window uncertainty signal (expected: it is
amplitude-blind by design, while RMSE is amplitude-dominated). Report as
null either way; do not tune downstream use on it. Needed for F014-2/3:
full ResNet 6-fold + Kaggle B/AMP/surface ckpts to test generality.
