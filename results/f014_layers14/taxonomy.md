# F014 layer 1 — failure-mode taxonomy (behavioral, inference-only)

Families: pretrained TimeSformer F012-2 (6 fold-variants) +
ridge 6-fold (6 folds). Full ResNet 6-fold set absent on this machine
(only F012-6 partial: Off_set1/Off_set2 x seeds 1-2) — excluded to keep the
comparison balanced; needed for F014-2/3 (see calibration.md).

Mean per-window Pearson r: pretrained 0.049, ridge 0.204.
Mean per-window RMSE: pretrained 1.204, ridge 0.463.

## Failure modes (honest: r~0.2 means shape unclaimed)

1. Shape unrecovered. Per-window Pearson r ~0.05 (pretrained) / ~0.20 (ridge) means
neither family recovers waveform shape/phase; only the coarse decay
envelope and (partially) the spectral peak survive. Any claim stronger than
'envelope + approximate fn' is not supported by these numbers.
2. Shift does not dominate error. Several folds show flat or negative
Delta_rmse (shift error <= matched error), so the dominant failure is
absolute accuracy, not sensing-condition shift — the robustness gap is
small because both sides are similarly poor, not because either is good.
Next experiments: (a) full ResNet 6-fold to test whether the gap is
encoder-specific; (b) B/AMP ablations (parked, need Kaggle ckpts);
(c) surface-vs-voxel representation comparison.
