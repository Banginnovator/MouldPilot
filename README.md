# MouldPilot v1.2 · Explainable Diagnosis

A Streamlit prototype demonstrating a **supervised process-correction workflow** for injection-moulding short-shot defects.

## What it does

1. Loads either a built-in synthetic demo dataset or a user-uploaded labelled CSV.
2. Fits a Random Forest surrogate model to predict the `short_shots` response from six process parameters.
3. Searches a bounded set of candidate settings and ranks them by predicted short-shot area plus a penalty for large parameter movements.
4. Requires an operator approval click before recording the proposed adjustment.
5. Compares an uploaded sample photo with an uploaded known-good reference, highlights regions of visual deviation in red, and reports a deviation bounding box (not a trained defect classifier or a semantic short-shot mask).
6. Explains each proposed parameter change using one-setting-at-a-time counterfactual predictions from the current baseline. The combined-versus-isolated difference is shown as a non-additivity remainder (not physical interaction proof).
7. Calculates permutation importance on the held-out test split to show which inputs mattered most to predictive error in that dataset.
8. Adds a nearest-training-data support check for both current and proposed settings. This is a heuristic warning, not a calibrated confidence probability.
9. Accepts an optional measured baseline area and a next-sample measurement, then reports prediction error and measured change when both real measurements are supplied.
10. Stores an audit trail.

## Important limits

- The built-in dataset is **synthetic**. It is generated from an illustrative response surface and is not factory data.
- The image module highlights pixel-difference regions between a current part photo and a known-good reference. It does **not** identify missing plastic semantically, classify short shots or diagnose defect types. Fixed camera position, lighting, scale and part orientation are required; otherwise visual differences may be false alarms.
- Local counterfactual effects vary one process input at a time in the fitted model. They are model-behaviour explanations, not proof that a parameter physically caused a defect or that changing it will improve a real process. Because the model is non-linear, isolated effects need not add up to the combined prediction.
- Permutation importance measures the increase in holdout MAE when one input is shuffled. It is global predictive evidence, not a diagnosis of an individual defective part.
- The data-support label is based on standardised nearest-neighbour distance. It is not a calibrated confidence score or probability of correctness.
- Actual measured improvement is only calculated when the operator supplies both a measured current-part baseline and a measured next-part result. Model predictions are not measurements.
- The app does **not** connect to sensors, a PLC, an injection-moulding machine or any physical controller.
- The optimiser is a decision-support demonstration, not a causal model or validated process recipe.
- Never apply its demo output to a production machine. For real use, train and validate on machine-, mould- and material-specific data, define process windows with a qualified process engineer, and implement safety-rated controller integration separately.

## Run on macOS

1. Install Python 3.10 or newer.
2. Unzip the project folder.
3. Double-click `run_mac.command`. If macOS blocks it, open Terminal, navigate into the folder and run:

   ```bash
   chmod +x run_mac.command
   ./run_mac.command
   ```

4. When installation completes, Streamlit opens the app in your browser. If it does not, open the local URL printed in Terminal (usually `http://localhost:8501`).

## Run manually

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Upload your own data

Choose **Upload production data** in the sidebar and provide a CSV with at least 30 labelled rows. Required columns:

- `Tinj` — injection temperature
- `tinj` — injection time
- `Pinj` — injection pressure
- `Ph` — holding pressure
- `Bp` — back pressure
- `th` — holding time
- `short_shots` — measured short-shot area (consistent units across all rows)

The app also recognises some common descriptive column names. If your data has a `label_viability` column, rows marked `R` are excluded because they represent non-injectable process settings in the referenced research dataset.

For uploaded factory data, enter a machine-specific approved range for each setting and tick the confirmation box only after a qualified engineer has validated those limits. Observed data ranges are not safety limits.

## Technical design

- UI: Streamlit
- Data handling: pandas / NumPy
- Surrogate: scikit-learn RandomForestRegressor
- Search: bounded 3-level candidate search for each of six inputs (up to 729 combinations)
- Audit: downloadable CSV from the current session

## Data provenance note

An injection-moulding research dataset at the University of León / Zenodo was identified during development. However, the original image dataset and related process datasets are currently marked embargoed until **31 December 2027**. Therefore, this MVP does not use those research observations; its built-in sample data are synthetic. The research record describes one experimental machine, one polypropylene grade and one mould geometry, so validation would still be needed before transfer to other factories.

## Next development steps

1. Add a trained computer-vision segmentation/classification model using legally accessible labelled images.
2. Add machine-specific process data connectors (starting read-only).
3. Add causal/process-engineering constraints reviewed by a manufacturing expert.
4. Validate recommendations through designed trials at a partner factory.
5. Only after validation, investigate controller integration with machine-specific interlocks and safety review.
