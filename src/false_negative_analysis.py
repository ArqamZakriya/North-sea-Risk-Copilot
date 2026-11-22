
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
import labels


MODEL_DIR = ROOT / "models" / "extended"
OUTPUT_DIR = ROOT / "reports"

CLASS_NAMES = {
    0: "Normal",
    1: "Warning",
    2: "Critical",
}

WEATHER_TERMS = (
    "wind",
    "gust",
    "wave",
    "pressure",
    "temp",
    "steepness",
    "marine_stress",
)


def get_weather_columns(frame):
    """Find numeric weather-related features."""
    return [
        column
        for column in frame.columns
        if any(term in str(column).lower() for term in WEATHER_TERMS)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def main():
    print("Loading weather dataset...")

    raw = pd.read_csv(cfg.CSV_PATH, parse_dates=["time"])

    print("Building extended features and labels...")

    data = labels.build(
        raw,
        cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS],
        "extended",
    )

    train, validation, test = labels.time_split(data)

    print("Loading saved extended model...")

    model = joblib.load(MODEL_DIR / "best_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    features = joblib.load(MODEL_DIR / "features.joblib")

    print(f"Test rows: {len(test):,}")

    # Predict using the existing model. No retraining.
    X_test = test[features]
    X_scaled = scaler.transform(X_test)

    predictions = pd.Series(
        model.predict(X_scaled),
        index=test.index,
    )

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = pd.DataFrame(
            model.predict_proba(X_scaled),
            index=test.index,
            columns=[
                f"probability_{CLASS_NAMES[int(c)]}"
                for c in model.classes_
            ],
        )

    # Restrict comparison to actual Critical cases.
    critical = test.loc[test["y"] == 2].copy()
    critical["predicted"] = predictions.loc[critical.index]

    critical["prediction_group"] = critical["predicted"].map({
        2: "Correct_Critical",
        1: "Missed_as_Warning",
        0: "Missed_as_Normal",
    })

    # Add prediction confidence.
    if probabilities is not None:
        critical = critical.join(probabilities)

    weather_columns = get_weather_columns(test)

    comparison_columns = weather_columns.copy()

    if probabilities is not None:
        comparison_columns += list(probabilities.columns)

    groups = [
        "Correct_Critical",
        "Missed_as_Warning",
        "Missed_as_Normal",
    ]

    # Build overall and site-level comparisons.
    records = []

    for site_name in ["ALL_SITES"] + sorted(critical["site"].unique()):

        if site_name == "ALL_SITES":
            site_data = critical
        else:
            site_data = critical.loc[critical["site"] == site_name]

        for group_name in groups:

            group_data = site_data.loc[
                site_data["prediction_group"] == group_name
            ]

            for feature in comparison_columns:

                values = group_data[feature].dropna()

                records.append({
                    "site": site_name,
                    "prediction_group": group_name,
                    "feature": feature,
                    "count": len(values),
                    "mean": values.mean() if len(values) else None,
                    "median": values.median() if len(values) else None,
                    "std": values.std() if len(values) > 1 else None,
                    "min": values.min() if len(values) else None,
                    "max": values.max() if len(values) else None,
                })

    comparison = pd.DataFrame(records)

    # Summary of Critical prediction outcomes.
    outcome_summary = (
        critical.groupby(["site", "prediction_group"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for group in groups:
        if group not in outcome_summary.columns:
            outcome_summary[group] = 0

    outcome_summary["total_critical"] = outcome_summary[groups].sum(axis=1)

    outcome_summary["critical_recall"] = (
        outcome_summary["Correct_Critical"]
        / outcome_summary["total_critical"].replace(0, pd.NA)
    ).round(4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_path = OUTPUT_DIR / "critical_prediction_comparison.csv"
    outcome_path = OUTPUT_DIR / "critical_prediction_outcomes.csv"

    comparison.to_csv(comparison_path, index=False)
    outcome_summary.to_csv(outcome_path, index=False)

    print("\nCRITICAL PREDICTION OUTCOMES")
    print(outcome_summary.to_string(index=False))

    print("\nOVERALL PREDICTION GROUP COUNTS")
    print(
        critical["prediction_group"]
        .value_counts()
        .reindex(groups, fill_value=0)
        .to_string()
    )

    print("\nCOMPARISON REPORTS SAVED")
    print(comparison_path)
    print(outcome_path)

    print("\nAnalysis completed. The trained model was not changed.")


if __name__ == "__main__":
    main()
