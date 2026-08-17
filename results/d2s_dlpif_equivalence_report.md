# D2S-RF <-> DLPIF(AmountRF) Structural Equivalence Report

- Seeds: [42, 123, 456]
- Scenarios: ['10pct', '20pct', 'block7d', 'block30d']
- Dummy-backbone trials per (seed, scenario): 3
- Total comparisons: 36
- All exact matches: **True**
- Overall max |diff|: 0.00e+00 mm

DLPIF's AmountRF stage reconstructs masked PRECIP positions identically to D2S-RF regardless of backbone content, confirming that the WGAN-GP/SAITS backbone contributes nothing to the PRECIP reconstruction itself (Section 3.4 / 5.8 of the manuscript) -- verified here at the level of individual masked positions under adversarial (random-noise) backbone substitution, not just via aggregate metrics on two similar real backbones.

| Seed | Scenario | Max |diff| (mm, over trials) | Exact match |
|---|---|---|---|
| 42 | 10pct | 0.00e+00 | True |
| 42 | 20pct | 0.00e+00 | True |
| 42 | block30d | 0.00e+00 | True |
| 42 | block7d | 0.00e+00 | True |
| 123 | 10pct | 0.00e+00 | True |
| 123 | 20pct | 0.00e+00 | True |
| 123 | block30d | 0.00e+00 | True |
| 123 | block7d | 0.00e+00 | True |
| 456 | 10pct | 0.00e+00 | True |
| 456 | 20pct | 0.00e+00 | True |
| 456 | block30d | 0.00e+00 | True |
| 456 | block7d | 0.00e+00 | True |
