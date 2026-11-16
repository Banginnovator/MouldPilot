"""Quick smoke tests for MouldPilot's core modelling and optimiser functions."""
from engine import (
    FEATURES, DEMO_RANGES, DEMO_BASELINE, DEFAULT_STEPS,
    make_demo_data, fit_surrogate, recommend_adjustment, prepare_uploaded_data,
    explain_adjustment, data_support_label
)


def main():
    df = make_demo_data(n=500)
    bundle = fit_surrogate(df)
    assert bundle.rows_used == 500
    assert bundle.metrics["mae"] >= 0
    result = recommend_adjustment(bundle, DEMO_BASELINE, DEMO_RANGES, DEFAULT_STEPS)
    assert set(result["candidate"].keys()) == set(FEATURES)
    explanation = explain_adjustment(bundle, DEMO_BASELINE, result["candidate"])
    assert len(explanation["rows"]) == len(FEATURES)
    assert abs(explanation["combined_delta"] - result["predicted_reduction"] * -1) < 1e-6
    assert all(feature in bundle.permutation_importance for feature in FEATURES)
    assert data_support_label(result["distance_to_data"])[0] in {"Near observed data", "Moderate distance", "Far from observed data"}
    for feature in FEATURES:
        lo, hi = DEMO_RANGES[feature]
        assert lo <= result["candidate"][feature] <= hi
        assert abs(result["candidate"][feature] - DEMO_BASELINE[feature]) <= DEFAULT_STEPS[feature] + 1e-9
    usable, note = prepare_uploaded_data(df)
    assert len(usable) == 500
    print("PASS: synthetic data, model fit, constrained optimisation, explainable diagnosis and upload validation")
    print(f"Holdout MAE: {bundle.metrics['mae']:.3f}")
    print(f"Holdout R2: {bundle.metrics['r2']:.3f}")
    print("Candidate:", result["candidate"])
    print(f"Predicted reduction (synthetic units): {result['predicted_reduction']:.3f}")


if __name__ == "__main__":
    main()
