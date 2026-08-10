# Multi-Mask Variance Experiment (Section 5.9 source)

10 independent random mask realizations per scenario (mask seeds 9000-9009), pooled across all 3 model seeds (42, 123, 456) for AmountRF_DLPIF and SAITS (Linear is deterministic, seed-independent).

| Method | Scenario | Mean F1 | Std F1 | N |
|---|---|---|---|---|
| AmountRF_DLPIF | 10pct | 0.7807 | 0.0247 | 30 |
| AmountRF_DLPIF | 20pct | 0.7886 | 0.0121 | 30 |
| Linear | 10pct | 0.6785 | 0.0209 | 10 |
| Linear | 20pct | 0.6769 | 0.0141 | 10 |
| SAITS | 10pct | 0.6566 | 0.0312 | 30 |
| SAITS | 20pct | 0.6424 | 0.0336 | 30 |
