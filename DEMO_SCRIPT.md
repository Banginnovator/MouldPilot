# 3-minute MouldPilot demo script

1. Open the app in **Demo simulator** mode. State that the training records are synthetic and the interface is demonstrating the proposed workflow.
2. Keep the prefilled settings and click **Analyse and propose a correction** after acknowledging the demo-range warning.
3. Show the current model estimate and the candidate settings. Explain that the optimiser searches nearby bounded combinations, not that the candidate is a proven physical fix.
4. Upload `sample_good_reference.png` and `sample_short_shot_sample.png` (if supplied) to show the pixel-difference heatmap. Explain that this comparison needs a fixed camera and lighting, and is not a trained defect detector.
5. Tick operator approval and record it. Point out that no real machine command is sent.
6. Enter a hypothetical next-sample measurement, record it and download the audit log.

## What to say honestly

- The process model is trained on synthetic illustrative data in this version.
- The reference-image module measures pixel differences; it does not classify defect types.
- Recommendations are candidates for operator review, not validated machine recipes.
- The next engineering milestone is a pilot using machine-specific labelled production data and controlled experiments.


## Explainable diagnosis (v1.2)

After generating a recommendation, open **Explainable diagnosis: what changed the model's prediction?**. Explain that each row changes one proposed setting from the same current baseline and asks the Random Forest for a new prediction. Negative isolated effects mean the model predicts a lower short-shot area for that single change; positive effects mean a higher area. The combined recommendation is evaluated separately. The non-additivity remainder is the difference between the combined change and the sum of isolated changes; it is a property of the fitted non-linear model, not proof of a physical interaction.

Open **Global model evidence** to show permutation importance calculated on the held-out split. Explain that it measures how much holdout MAE increases when an input is shuffled. It is not part-level cause attribution. The nearest-data distance is a heuristic support warning, not a calibrated confidence score.

All v1.2 explanation outputs are generated from the selected dataset. In Demo simulator mode, that dataset is synthetic. Do not present any values as factory evidence or production-ready settings.
