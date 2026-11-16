# MouldPilot v1.2 — Explainable Diagnosis

## Added
- Per-parameter local counterfactual explanation for each proposed adjustment.
- Combined prediction change, sum of isolated effects, and a non-additivity remainder.
- Held-out permutation importance, reported as increase in MAE after shuffling a feature.
- Training-data proximity check for current and proposed settings.

## Interpretation boundaries
- Counterfactual effects describe the fitted model, not physical cause.
- The non-additivity remainder is not proof of a physical interaction.
- Permutation importance is global, not part-specific diagnosis.
- Nearest-data distance is a heuristic support warning, not calibrated confidence.
- Demo mode uses synthetic data. No PLC or machine connection exists.
