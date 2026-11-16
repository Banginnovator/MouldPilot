from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from engine import (
    FEATURES, TARGET, DISPLAY, DEMO_RANGES, DEMO_BASELINE, DEFAULT_STEPS,
    make_demo_data, prepare_uploaded_data, fit_surrogate,
    recommend_adjustment, explain_adjustment, data_support_label, nearest_distance, build_audit_row,
)
from vision import compare_reference_images

st.set_page_config(page_title="MouldPilot v1.2 | Explainable process correction", page_icon="🧩", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1380px;}
  .hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(120,130,145,.25); border-radius: 18px; background: linear-gradient(115deg, rgba(35,94,82,.12), rgba(42,91,145,.07));}
  .hero h1 {margin: 0; letter-spacing: -.04em; font-size: 2.1rem;}
  .hero p {margin: .35rem 0 0; color: var(--text-color); opacity: .78;}
  .small-muted {font-size: .86rem; opacity: .72;}
  div[data-testid="stMetric"] {border: 1px solid rgba(120,130,145,.22); padding: 12px 14px; border-radius: 13px;}
  .notice {padding: .85rem 1rem; border-left: 4px solid #d49b32; background: rgba(212,155,50,.09); border-radius: 8px;}
  .success-note {padding: .85rem 1rem; border-left: 4px solid #2d936c; background: rgba(45,147,108,.09); border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "last_recommendation" not in st.session_state:
    st.session_state.last_recommendation = None
if "approved_recommendation" not in st.session_state:
    st.session_state.approved_recommendation = None

st.markdown("""
<div class="hero">
  <h1>🧩 MOULDPILOT <span style="font-size:.48em; opacity:.7; letter-spacing:0;">v1.2</span></h1>
  <p>Explainable, supervised process-correction prototype for legacy injection-moulding machines</p>
</div>
""", unsafe_allow_html=True)
st.write("")
st.markdown('<div class="notice"><b>Prototype boundary:</b> This is a software demonstration. It does not connect to or control a real injection-moulding machine. Recommendations are decision-support examples, not validated process instructions. Never use demo settings on production equipment.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Workspace")
    mode = st.radio("Data mode", ["Demo simulator", "Upload production data"], index=0)
    uploaded = None
    if mode == "Upload production data":
        uploaded = st.file_uploader("Upload a labelled CSV", type=["csv"], help="Required columns: Tinj, tinj, Pinj, Ph, Bp, th, short_shots. Each row should represent a labelled process configuration or production sample.")
    st.divider()
    st.caption("v1.2 build · explainable supervised workflow")
    st.caption("No machine write-access or PLC connection")

# Load and train data/model
try:
    if mode == "Demo simulator":
        data = make_demo_data()
        model_note = "Synthetic demonstration data generated from an illustrative response surface. Not factory measurements."
        ranges = DEMO_RANGES.copy()
        default_current = DEMO_BASELINE.copy()
        demo_mode = True
    else:
        if uploaded is None:
            st.info("Upload a labelled CSV to train the process-response model. A demo is available from the sidebar at any time.")
            st.stop()
        raw = pd.read_csv(uploaded)
        data, model_note = prepare_uploaded_data(raw)
        ranges = {f: (float(data[f].min()), float(data[f].max())) for f in FEATURES}
        default_current = {f: float(data[f].median()) for f in FEATURES}
        demo_mode = False
    bundle = fit_surrogate(data)
except Exception as exc:
    st.error(f"Could not prepare the model: {exc}")
    st.stop()

st.caption(model_note)
if not demo_mode:
    st.markdown('<div class="notice"><b>Machine-specific safety:</b> Data min/max values describe the observed dataset, not safe operating limits. Enter a process window approved for the specific machine, mould and material below before generating a recommendation.</div>', unsafe_allow_html=True)

# Top-level data/model metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Labelled records", f"{bundle.rows_used:,}")
m2.metric("Holdout MAE", f"{bundle.metrics['mae']:.2f} mm²")
r2 = bundle.metrics["r2"]
m3.metric("Holdout R²", f"{r2:.3f}" if np.isfinite(r2) else "N/A")
m4.metric("Model", "Random Forest")

st.divider()
left, right = st.columns([1.04, 0.96], gap="large")

with left:
    st.subheader("1 · Describe the current production cycle")
    st.write("Enter the process settings for the cycle you want to investigate.")
    current = {}
    input_cols = st.columns(2)
    for i, f in enumerate(FEATURES):
        col = input_cols[i % 2]
        lo, hi = ranges[f]
        if lo == hi:
            lo, hi = lo - 0.01, hi + 0.01
        default = float(np.clip(default_current[f], lo, hi))
        label, unit = DISPLAY[f]
        step = 0.1 if f in ["tinj", "Bp", "th"] else 1.0
        current[f] = col.number_input(
            f"{label} ({unit})", min_value=float(lo), max_value=float(hi),
            value=default, step=float(step), key=f"current_{f}"
        )
    st.caption("The model estimates short-shot area from these six process variables. It is a surrogate trained on the selected dataset.")

with right:
    st.subheader("2 · Product inspection record")
    ref_col, sample_col = st.columns(2)
    with ref_col:
        reference_image = st.file_uploader("Known-good reference part", type=["png", "jpg", "jpeg"], key="reference_image")
    with sample_col:
        product_image = st.file_uploader("Current production sample", type=["png", "jpg", "jpeg"], key="product_image")
    if reference_image is not None and product_image is not None:
        try:
            vision_result = compare_reference_images(reference_image.getvalue(), product_image.getvalue())
            img1, img2 = st.columns(2)
            with img1:
                st.image(reference_image, caption="Known-good reference", use_container_width=True)
            with img2:
                st.image(product_image, caption="Current sample", use_container_width=True)
            v1, v2 = st.columns(2)
            v1.metric("Image similarity", f"{vision_result['similarity']:.1f}%")
            v2.metric("Area with visual deviation", f"{vision_result['deviation_pct']:.1f}%")
            st.image(vision_result["deviation_overlay"], caption="Deviation regions highlighted in red (not a semantic short-shot mask)", use_container_width=True)
            with st.expander("Inspection details"):
                st.image(vision_result["heatmap"], caption="Pixel-difference heatmap", use_container_width=True)
                if vision_result["deviation_bbox"]:
                    st.write("Deviation bounding box (pixels):", vision_result["deviation_bbox"])
                else:
                    st.write("No pixels exceeded the demo deviation threshold.")
            if not vision_result["original_size_match"]:
                st.caption("The uploaded images had different dimensions and were resized to a common canvas.")
            st.warning("This highlights visual differences, not missing plastic specifically. Camera angle, lighting, rotation, shadows and part placement can create false positives. Use fixed imaging conditions and operator confirmation.")
        except Exception as exc:
            st.error(f"Could not compare the images: {exc}")
    elif product_image is not None:
        st.image(product_image, caption="Current sample", use_container_width=True)
        st.info("Upload a known-good reference image too to run the basic visual-difference check. Otherwise, record the defect manually below.")
    else:
        st.info("Optionally upload both a known-good reference and a current sample photographed under the same conditions.")
    observed_defect = st.selectbox("Operator's observed defect", ["Short shot", "No visible short shot", "Needs inspection"], index=0)
    operator_severity = st.slider("Operator-assessed short-shot severity", 0, 100, 35, help="A manual 0–100 assessment for the demo record. The optimiser itself uses the model's predicted area in mm².")
    baseline_observed = st.number_input("Measured defect area on current part (mm², optional)", min_value=0.0, value=0.0, step=0.5, help="Enter a measured area only if you have one. Leave at zero if unavailable; zero is treated as not supplied.")

st.divider()
st.subheader("3 · Set approved adjustment limits")
st.write("These limits must come from the machine manual or a qualified process engineer. In demo mode, the values below are illustrative search boundaries only.")
range_rows = []
step_values = {}
range_cols = st.columns(3)
for i, f in enumerate(FEATURES):
    label, unit = DISPLAY[f]
    lo_data, hi_data = ranges[f]
    with range_cols[i % 3]:
        st.markdown(f"**{label} ({unit})**")
        safe_lo = st.number_input(f"Approved minimum · {f}", value=float(lo_data), key=f"safe_lo_{f}")
        safe_hi = st.number_input(f"Approved maximum · {f}", value=float(hi_data), key=f"safe_hi_{f}")
        step_values[f] = st.number_input(f"Max change per recommendation · {f}", min_value=0.001, value=float(DEFAULT_STEPS[f] if demo_mode else max((hi_data-lo_data)*0.05, 0.001)), key=f"step_{f}")
        range_rows.append((f, safe_lo, safe_hi))
approved_ranges = {f: (lo, hi) for f, lo, hi in range_rows}
if not demo_mode:
    safe_confirmed = st.checkbox("I confirm these are approved process limits for this specific machine, mould and material.")
else:
    safe_confirmed = st.checkbox("I understand the demo ranges are illustrative only and are not safe production limits.", value=False)

st.divider()
st.subheader("4 · Generate a supervised correction")
penalty = st.slider("Preference for smaller adjustments", min_value=0.0, max_value=0.6, value=0.12, step=0.02, help="Higher values discourage large changes. This is a prototype tuning preference, not a factory safety setting.")
run = st.button("Analyse and propose a correction", type="primary", use_container_width=True, disabled=not safe_confirmed)

if run:
    try:
        rec = recommend_adjustment(bundle, current, approved_ranges, step_values, movement_penalty=float(penalty))
        rec["observed_defect"] = observed_defect
        rec["operator_severity"] = operator_severity
        rec["baseline_observed"] = float(baseline_observed) if baseline_observed > 0 else None
        rec["demo_mode"] = demo_mode
        st.session_state.last_recommendation = rec
        st.session_state.approved_recommendation = None
    except Exception as exc:
        st.error(str(exc))

rec = st.session_state.last_recommendation
if rec is not None:
    st.divider()
    st.subheader("5 · Recommendation for operator review")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted short-shot area · current", f"{rec['baseline_prediction']:.2f} mm²")
    col_b.metric("Predicted area · candidate", f"{rec['candidate_prediction']:.2f} mm²", delta=f"{rec['candidate_prediction']-rec['baseline_prediction']:.2f} mm²", delta_color="inverse")
    reduction = rec["predicted_reduction"]
    col_c.metric("Predicted reduction", f"{reduction:.2f} mm²", delta="Model estimate only")

    explanation = explain_adjustment(bundle, rec["baseline"], rec["candidate"])
    details = []
    for f in FEATURES:
        label, unit = DISPLAY[f]
        before = rec["baseline"][f]
        after = rec["candidate"][f]
        details.append({"Parameter": label, "Current": before, "Proposed": after, "Change": after-before, "Unit": unit})
    st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)

    # Model-support check: proximity to the training data is a warning signal,
    # not a calibrated probability or confidence score.
    current_distance = rec["distance_to_data"]
    candidate_distance = nearest_distance(bundle, rec["candidate"])
    support_label, support_text = data_support_label(candidate_distance)
    st.markdown("#### How much does the data support this recommendation?")
    support_col1, support_col2 = st.columns(2)
    support_col1.metric("Current setting · nearest-data distance", f"{current_distance:.2f}")
    support_col2.metric("Proposed setting · nearest-data distance", f"{candidate_distance:.2f}", delta=support_label, delta_color="off")
    st.caption(support_text + " Distance is standardised nearest-neighbour distance, not a probability of correctness.")

    st.markdown("#### Explainable diagnosis: what changed the model's prediction?")
    st.caption("For each setting, the app changes only that one value from the current baseline and asks the fitted model for a new prediction. This makes the model's local response inspectable. It does not prove a physical cause.")
    explain_df = pd.DataFrame(explanation["rows"])
    display_explain = explain_df[[
        "Parameter", "Current", "Proposed", "Change", "Unit",
        "Prediction if only this setting changed (mm²)",
        "Isolated model effect (mm²)", "Model reading"
    ]].copy()
    for col in ["Current", "Proposed", "Change", "Prediction if only this setting changed (mm²)", "Isolated model effect (mm²)"]:
        display_explain[col] = display_explain[col].map(lambda v: round(float(v), 3))
    st.dataframe(display_explain, use_container_width=True, hide_index=True)

    e1, e2, e3 = st.columns(3)
    e1.metric("Combined model change", f"{explanation['combined_delta']:+.2f} mm²", help="Combined candidate prediction minus current prediction. Negative means the model predicts a lower defect area.")
    e2.metric("Sum of isolated effects", f"{explanation['isolated_effect_sum']:+.2f} mm²", help="Sum of six one-setting-at-a-time comparisons. These effects are not necessarily additive.")
    e3.metric("Non-additivity remainder", f"{explanation['non_additivity']:+.2f} mm²", help="Difference between combined change and the sum of isolated changes. It captures model non-additivity, not a proven physical interaction.")
    st.info("Read the signs carefully: a negative isolated effect means the model predicts less short-shot area when that single setting is changed from the current baseline. A positive effect means it predicts more. The optimiser evaluates the full combination, so individual effects may not sum to the combined result.")

    with st.expander("Global model evidence · which inputs mattered on held-out data?"):
        perm_df = pd.DataFrame([
            {"Parameter": DISPLAY[f][0], "Increase in holdout MAE when shuffled (mm²)": bundle.permutation_importance[f]}
            for f in FEATURES
        ]).sort_values("Increase in holdout MAE when shuffled (mm²)", ascending=True)
        fig = px.bar(perm_df, x="Increase in holdout MAE when shuffled (mm²)", y="Parameter", orientation="h", title="Permutation importance on the held-out split")
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("A larger positive increase means shuffling that input made holdout predictions worse. Values near zero or negative mean this split provides little evidence of predictive contribution. This is global model evidence, not a cause diagnosis for this specific part.")

    if demo_mode:
        st.markdown('<div class="notice"><b>Demo-only result:</b> This recommendation is generated from synthetic data. It is not an engineering instruction and must not be applied to a real machine.</div>', unsafe_allow_html=True)

    confirm_label = "I approve this recommendation for the prototype record (no machine command will be sent)."
    approve = st.checkbox(confirm_label, key="operator_approval")
    if st.button("Record operator approval", disabled=not approve):
        st.session_state.approved_recommendation = rec
        st.session_state.audit_log.append(build_audit_row(
            rec["baseline"], rec["candidate"], rec["baseline_prediction"],
            rec["candidate_prediction"], True
        ))
        st.success("Approval recorded in this demo session. No machine settings were changed.")

    if st.session_state.approved_recommendation is not None:
        st.markdown('<div class="success-note"><b>Approval captured.</b> The next step is to inspect a subsequent part and record its measured short-shot area. The prototype will compare that measurement with the model estimate.</div>', unsafe_allow_html=True)
        observed_after = st.number_input("Measured short-shot area on next sample (mm²)", min_value=0.0, value=0.0, step=0.5, key="observed_after")
        note = st.text_input("Operator note (optional)", key="operator_note")
        if st.button("Record next-sample result"):
            approved_rec = st.session_state.approved_recommendation
            st.session_state.audit_log.append(build_audit_row(
                approved_rec["baseline"], approved_rec["candidate"],
                approved_rec["baseline_prediction"], approved_rec["candidate_prediction"],
                True, float(observed_after), note
            ))
            st.session_state.latest_verification = {
                "predicted": float(approved_rec["candidate_prediction"]),
                "measured": float(observed_after),
                "baseline_measured": approved_rec.get("baseline_observed"),
            }
            st.success("Next-sample result saved to the session audit log.")
        verification = st.session_state.get("latest_verification")
        if verification:
            st.markdown("#### Prediction versus measurement")
            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("Model prediction", f"{verification['predicted']:.2f} mm²")
            vc2.metric("Measured next sample", f"{verification['measured']:.2f} mm²")
            vc3.metric("Absolute prediction error", f"{abs(verification['measured'] - verification['predicted']):.2f} mm²")
            if verification["baseline_measured"] is not None:
                change = verification["baseline_measured"] - verification["measured"]
                st.metric("Measured change versus current part", f"{change:.2f} mm²", delta=f"{'reduction' if change >= 0 else 'increase'} of {abs(change):.2f} mm²", delta_color="normal" if change >= 0 else "inverse")
            else:
                st.info("To calculate actual measured improvement, enter a measured defect area for the current part in Section 2 before running the next test. Without that baseline measurement, the app can compare prediction with measurement but cannot prove the defect improved.")

# Explainability / model information
st.divider()
with st.expander("Model details and limitations"):
    imp = pd.DataFrame([{"Parameter": DISPLAY[f][0], "Holdout MAE increase": bundle.permutation_importance[f]} for f in FEATURES]).sort_values("Holdout MAE increase", ascending=True)
    fig = px.bar(imp, x="Holdout MAE increase", y="Parameter", orientation="h", title="Permutation importance on held-out data")
    fig.update_layout(height=310, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    - The model predicts the short-shot area column from six process parameters. It is not a causal diagnostic model.
    - Permutation importance is the change in held-out MAE after shuffling one feature. It is global predictive evidence, not a local or causal diagnosis.
    - The Explainable Diagnosis table performs one-setting-at-a-time counterfactual predictions from the current baseline. These effects may not add up because the model is non-linear.
    - The data-support label is based on nearest-neighbour distance in standardised feature space. It is a heuristic warning, not a calibrated confidence score.
    - Holdout MAE and R² are evaluation statistics on a random split of the selected dataset. They do not establish transfer to a different machine, polymer, mould or factory.
    - In demo mode, both training data and response values are synthetic.
    - The image module highlights pixel-difference regions against a known-good reference. It does not semantically identify missing plastic and is not a trained short-shot classifier; lighting, pose and camera changes can cause false deviations.
    - Local counterfactual comparisons are specific to the fitted model and baseline settings. They are not physical root-cause proof and do not validate a process change.
    - No PLC, controller, sensor or machine connection is included.
    """)

if st.session_state.audit_log:
    st.divider()
    st.subheader("Session audit log")
    log_df = pd.DataFrame(st.session_state.audit_log)
    st.dataframe(log_df, use_container_width=True, hide_index=True)
    st.download_button("Download audit log CSV", data=log_df.to_csv(index=False).encode("utf-8"), file_name="mouldpilot_audit_log.csv", mime="text/csv")

st.caption("MouldPilot v1.2 · Built for demonstration and technical validation · Not for live machine control")
