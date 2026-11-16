# MouldPilot v1.2 — SPARK Night Round 1 POC

## What this is
A local Streamlit software demonstration of a supervised process-correction workflow for injection-moulding short shots.

## Demonstrated
- Random Forest surrogate trained on built-in synthetic data
- Bounded candidate search with a movement penalty
- One-setting-at-a-time counterfactual explanations
- Holdout permutation importance
- Nearest-training-data support heuristic
- Reference-image pixel-difference visualisation
- Operator approval workflow and session audit trail

## Verification performed for this package
- `python3 test_engine.py` — passed
- `python3 test_vision.py` — passed

The model test reports holdout metrics on a generated synthetic dataset only. These are software smoke-test results, not factory performance metrics.

## Critical limitations
- Built-in process data are synthetic, not factory observations.
- Image comparison is pixel-difference analysis, not semantic defect classification.
- The app has no physical sensor, PLC or machine connection.
- Candidate settings are not validated process recipes. Do not apply them to production equipment.
- No factory trial, customer deployment, measured savings or production-grade safety certification is claimed.

## Run
On macOS, install Python 3.10+ and double-click `run_mac.command`. Alternatively:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Demo recommendation
Use Demo simulator mode. Explain that all model outputs are based on synthetic illustrative records. Show the bounded recommendation, explainability panel, image pixel-difference comparison, operator approval and audit log. State clearly that no machine command is transmitted.
