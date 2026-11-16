"""MouldPilot prototype engine.

The built-in demo data are synthetic and are only for illustrating the workflow.
They are not factory measurements and must never be used to set real machine limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

FEATURES = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]
TARGET = "short_shots"
DISPLAY = {
    "Tinj": ("Injection temperature", "°C"),
    "tinj": ("Injection time", "s"),
    "Pinj": ("Injection pressure", "bar"),
    "Ph": ("Holding pressure", "bar"),
    "Bp": ("Back pressure", "bar"),
    "th": ("Holding time", "s"),
}
# These ranges are only for the synthetic demo and correspond to one published
# experimental setup's design space. They are not universal or recommended
# operating limits for industrial machines.
DEMO_RANGES = {
    "Tinj": (180.0, 240.0),
    "tinj": (0.5, 2.0),
    "Pinj": (5.0, 35.0),
    "Ph": (5.0, 35.0),
    "Bp": (15.0, 30.0),
    "th": (4.0, 6.0),
}
DEMO_BASELINE = {
    "Tinj": 190.0,
    "tinj": 0.75,
    "Pinj": 12.0,
    "Ph": 12.0,
    "Bp": 20.0,
    "th": 4.0,
}
DEFAULT_STEPS = {
    "Tinj": 10.0,
    "tinj": 0.25,
    "Pinj": 5.0,
    "Ph": 5.0,
    "Bp": 2.5,
    "th": 1.0,
}

ALIASES = {
    "Tinj": ["Tinj", "injection_temperature", "injection temperature", "temperature"],
    "tinj": ["tinj", "injection_time", "injection time", "fill_time"],
    "Pinj": ["Pinj", "injection_pressure", "injection pressure", "fill_pressure"],
    "Ph": ["Ph", "holding_pressure", "holding pressure", "pack_pressure"],
    "Bp": ["Bp", "back_pressure", "back pressure"],
    "th": ["th", "holding_time", "holding time", "pack_time"],
    "short_shots": ["short_shots", "short shot area", "short_shot_area", "short shot", "short_shot"],
    "label_viability": ["label_viability", "viability", "label"],
}


@dataclass
class ModelBundle:
    model: RandomForestRegressor
    metrics: Dict[str, float]
    rows_used: int
    feature_importance: Dict[str, float]
    permutation_importance: Dict[str, float]
    x_reference: pd.DataFrame
    y_reference: pd.Series


def make_demo_data(n: int = 1800, seed: int = 17) -> pd.DataFrame:
    """Create transparent synthetic data for a click-through demonstration."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({
        "Tinj": rng.uniform(*DEMO_RANGES["Tinj"], n),
        "tinj": rng.uniform(*DEMO_RANGES["tinj"], n),
        "Pinj": rng.uniform(*DEMO_RANGES["Pinj"], n),
        "Ph": rng.uniform(*DEMO_RANGES["Ph"], n),
        "Bp": rng.uniform(*DEMO_RANGES["Bp"], n),
        "th": rng.uniform(*DEMO_RANGES["th"], n),
    })
    # Illustrative response surface, deliberately labelled synthetic.
    # It encodes plausible directions for a demo, not a validated physical model.
    temp_penalty = np.maximum(0, 215 - x["Tinj"].to_numpy()) * 0.34
    fill_time_penalty = np.maximum(0, 1.55 - x["tinj"].to_numpy()) * 22.0
    injection_penalty = np.maximum(0, 27 - x["Pinj"].to_numpy()) * 1.65
    holding_penalty = np.maximum(0, 25 - x["Ph"].to_numpy()) * 0.58
    holding_time_penalty = np.maximum(0, 5.5 - x["th"].to_numpy()) * 4.2
    back_pressure_penalty = np.maximum(0, 22 - x["Bp"].to_numpy()) * 0.35
    interaction = (np.maximum(0, 205 - x["Tinj"].to_numpy()) *
                   np.maximum(0, 24 - x["Pinj"].to_numpy())) * 0.015
    noise = rng.normal(0, 2.5, n)
    severity = (temp_penalty + fill_time_penalty + injection_penalty +
                holding_penalty + holding_time_penalty + back_pressure_penalty +
                interaction + noise)
    x[TARGET] = np.clip(severity, 0, None)
    x["source"] = "synthetic_demo"
    return x


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lookup = {str(c).strip().lower(): c for c in out.columns}
    rename = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            key = alias.strip().lower()
            if key in lookup:
                rename[lookup[key]] = canonical
                break
    return out.rename(columns=rename)


def prepare_uploaded_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Validate an uploaded dataset and return usable process/short-shot rows."""
    data = _normalise_columns(df)
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            ". Required fields are Tinj, tinj, Pinj, Ph, Bp, th and short_shots."
        )
    data = data.copy()
    if "label_viability" in data.columns:
        # 'R' means non-injectable in the referenced dataset; exclude it.
        data = data[data["label_viability"].astype(str).str.upper() != "R"]
    for col in FEATURES + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])
    data = data[(data[TARGET] >= 0)]
    if len(data) < 30:
        raise ValueError(f"Only {len(data)} usable rows found. Please provide at least 30 labelled production records.")
    data = data[FEATURES + [TARGET]].reset_index(drop=True)
    note = f"Loaded {len(data):,} usable labelled records. Non-injectable rows and incomplete records were excluded where identified."
    return data, note


def fit_surrogate(df: pd.DataFrame, seed: int = 42) -> ModelBundle:
    """Evaluate a model on a holdout split, then refit on all available rows."""
    x = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)
    if len(df) < 30:
        raise ValueError("At least 30 labelled rows are required to fit the prototype model.")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=seed
    )
    eval_model = RandomForestRegressor(
        n_estimators=180, max_depth=12, min_samples_leaf=2,
        random_state=seed, n_jobs=-1
    )
    eval_model.fit(x_train, y_train)
    pred = eval_model.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)) if y_test.nunique() > 1 else float("nan"),
    }
    final_model = RandomForestRegressor(
        n_estimators=180, max_depth=12, min_samples_leaf=2,
        random_state=seed, n_jobs=-1
    )
    final_model.fit(x, y)
    # Permutation importance is calculated on the held-out split, not the data
    # used to fit the final model. Values are the increase in holdout MAE when
    # a feature is shuffled; negative values can occur from sampling noise.
    perm = permutation_importance(
        eval_model, x_test, y_test, n_repeats=8, random_state=seed,
        scoring="neg_mean_absolute_error", n_jobs=-1
    )
    perm_importance = {
        f: float(v) for f, v in zip(FEATURES, perm.importances_mean)
    }
    importance = {f: float(v) for f, v in zip(FEATURES, final_model.feature_importances_)}
    return ModelBundle(
        final_model, metrics, len(df), importance, perm_importance, x.copy(), y.copy()
    )


def nearest_distance(bundle: ModelBundle, settings: Dict[str, float]) -> float:
    """Distance to nearest observed point, standardised by reference spread."""
    xref = bundle.x_reference.to_numpy(dtype=float)
    x = np.array([settings[f] for f in FEATURES], dtype=float)
    scale = np.nanstd(xref, axis=0)
    scale[scale == 0] = 1.0
    dist = np.sqrt(np.mean(((xref - x) / scale) ** 2, axis=1))
    return float(np.min(dist))


def recommend_adjustment(
    bundle: ModelBundle,
    current: Dict[str, float],
    approved_ranges: Dict[str, Tuple[float, float]],
    max_steps: Dict[str, float],
    movement_penalty: float = 0.10,
    grid_levels: Iterable[float] = (-1.0, 0.0, 1.0),
) -> Dict[str, object]:
    """Find a bounded candidate setting using a surrogate plus movement penalty.

    This is a decision-support demonstration. approved_ranges must come from an
    authorised process engineer / machine-specific validated process window.
    """
    for f in FEATURES:
        lo, hi = approved_ranges[f]
        if not lo < hi:
            raise ValueError(f"Invalid approved range for {f}: minimum must be below maximum.")
        if not (lo <= current[f] <= hi):
            raise ValueError(f"Current {f}={current[f]} is outside the approved range [{lo}, {hi}].")
        if max_steps[f] <= 0:
            raise ValueError(f"Maximum adjustment step for {f} must be positive.")

    baseline = pd.DataFrame([[current[f] for f in FEATURES]], columns=FEATURES)
    baseline_pred = float(bundle.model.predict(baseline)[0])
    levels = list(grid_levels)
    candidates = []
    for combo in product(levels, repeat=len(FEATURES)):
        row = {}
        for f, level in zip(FEATURES, combo):
            lo, hi = approved_ranges[f]
            proposed = current[f] + level * max_steps[f]
            row[f] = float(np.clip(proposed, lo, hi))
        candidates.append(row)
    cand = pd.DataFrame(candidates, columns=FEATURES).drop_duplicates().reset_index(drop=True)
    preds = bundle.model.predict(cand)
    scales = {f: max(approved_ranges[f][1] - approved_ranges[f][0], 1e-9) for f in FEATURES}
    move = np.zeros(len(cand))
    for f in FEATURES:
        move += np.abs(cand[f].to_numpy() - current[f]) / scales[f]
    # Penalise large moves so the optimiser prefers the smallest useful change.
    severity_scale = max(float(np.nanpercentile(bundle.y_reference, 90)), 1.0)
    objective = preds + movement_penalty * severity_scale * move
    best_i = int(np.argmin(objective))
    best = {f: float(cand.iloc[best_i][f]) for f in FEATURES}
    best_pred = float(preds[best_i])
    return {
        "baseline_prediction": baseline_pred,
        "candidate_prediction": best_pred,
        "predicted_reduction": baseline_pred - best_pred,
        "candidate": best,
        "baseline": dict(current),
        "movement_score": float(move[best_i]),
        "candidate_count": int(len(cand)),
        "distance_to_data": nearest_distance(bundle, current),
    }



def explain_adjustment(
    bundle: ModelBundle,
    current: Dict[str, float],
    candidate: Dict[str, float],
) -> Dict[str, object]:
    """Explain a recommendation with local, one-feature-at-a-time counterfactuals.

    The isolated effects are model comparisons, not causal effects. Because a
    Random Forest is non-linear, the isolated effects do not have to add up to
    the combined prediction. Their difference is reported as a non-additivity
    remainder rather than being attributed to a physical mechanism.
    """
    base_frame = pd.DataFrame([[current[f] for f in FEATURES]], columns=FEATURES)
    baseline_prediction = float(bundle.model.predict(base_frame)[0])
    candidate_frame = pd.DataFrame([[candidate[f] for f in FEATURES]], columns=FEATURES)
    combined_prediction = float(bundle.model.predict(candidate_frame)[0])

    rows = []
    isolated_deltas = []
    for f in FEATURES:
        before = float(current[f])
        after = float(candidate[f])
        isolated = dict(current)
        isolated[f] = after
        isolated_frame = pd.DataFrame([[isolated[k] for k in FEATURES]], columns=FEATURES)
        isolated_prediction = float(bundle.model.predict(isolated_frame)[0])
        effect = isolated_prediction - baseline_prediction
        isolated_deltas.append(effect)

        if np.isclose(before, after):
            reading = "Not changed by optimiser"
        elif effect < -0.25:
            reading = "Model predicts lower area when changed alone"
        elif effect > 0.25:
            reading = "Model predicts higher area when changed alone"
        else:
            reading = "Little isolated change in model prediction"

        rows.append({
            "Parameter": DISPLAY[f][0],
            "Current": before,
            "Proposed": after,
            "Change": after - before,
            "Unit": DISPLAY[f][1],
            "Prediction if only this setting changed (mm²)": isolated_prediction,
            "Isolated model effect (mm²)": effect,
            "Global holdout importance (MAE increase)": bundle.permutation_importance[f],
            "Model reading": reading,
        })

    combined_delta = combined_prediction - baseline_prediction
    isolated_sum = float(np.sum(isolated_deltas))
    return {
        "rows": rows,
        "baseline_prediction": baseline_prediction,
        "combined_prediction": combined_prediction,
        "combined_delta": combined_delta,
        "isolated_effect_sum": isolated_sum,
        "non_additivity": combined_delta - isolated_sum,
    }


def data_support_label(distance: float) -> Tuple[str, str]:
    """Heuristic proximity label; deliberately not called model confidence."""
    if distance <= 0.75:
        return "Near observed data", "The nearest training example is relatively close in standardised feature space."
    if distance <= 1.50:
        return "Moderate distance", "The nearest training example is not especially close; interpret with care."
    return "Far from observed data", "The setting is relatively far from all training examples; predictions are extrapolation-like and deserve extra scrutiny."


def build_audit_row(
    current: Dict[str, float],
    candidate: Dict[str, float],
    predicted_before: float,
    predicted_after: float,
    approved: bool,
    observed_after: float | None = None,
    note: str = "",
) -> Dict[str, object]:
    row = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "status": "approved_in_demo" if approved else "not_approved",
        "predicted_short_shot_area_before": predicted_before,
        "predicted_short_shot_area_after": predicted_after,
        "observed_short_shot_area_after": observed_after,
        "operator_note": note,
    }
    for f in FEATURES:
        row[f"before_{f}"] = current[f]
        row[f"proposed_{f}"] = candidate[f]
    return row
