
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
import labels


BASELINE_DIR = ROOT / "models" / "extended"
EXPERIMENT_DIR = ROOT / "models" / "experiment_v2"
REPORT_DIR = ROOT / "reports"


def get_predictions(model_dir, test, features):

    model = joblib.load(model_dir / "best_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    saved_features = joblib.load(model_dir / "features.joblib")

    if list(saved_features) != list(features):
        raise ValueError("Feature mismatch detected.")

    X_test = scaler.transform(test[features])

    return model.predict(X_test)


def main():

    print("=" * 65)
    print("COMPARISON OF ADDITIONAL FALSE CRITICAL ALERTS")
    print("=" * 65)

    print("\nLoading dataset...")

    raw = pd.read_csv(cfg.CSV_PATH, parse_dates=["time"])

    data = labels.build(
        raw,
        cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS],
        "extended",
    )

    train, validation, test = labels.time_split(data)
    features = labels.feature_columns(data)

    print(f"Test observations: {len(test):,}")

    print("\nGenerating predictions...")

    baseline_predictions = get_predictions(
        BASELINE_DIR, test, features
    )

    experiment_predictions = get_predictions(
        EXPERIMENT_DIR, test, features
    )

    comparison = test[["time", "site", "y"]].copy()

    comparison["baseline_prediction"] = baseline_predictions
    comparison["experiment_prediction"] = experiment_predictions

    class_names = {
        0: "Normal",
        1: "Warning",
        2: "Critical",
    }

    comparison["actual_class"] = comparison["y"].map(class_names)

    comparison["baseline_class"] = (
        comparison["baseline_prediction"].map(class_names)
    )

    comparison["experiment_class"] = (
        comparison["experiment_prediction"].map(class_names)
    )

    # Additional false Critical alerts introduced by Experiment V2.
    additional_false_positives = comparison[
        (comparison["baseline_prediction"] != 2)
        & (comparison["experiment_prediction"] == 2)
        & (comparison["y"] != 2)
    ].copy()

    # False Critical alerts corrected by Experiment V2.
    removed_false_positives = comparison[
        (comparison["baseline_prediction"] == 2)
        & (comparison["experiment_prediction"] != 2)
        & (comparison["y"] != 2)
    ].copy()

    # Additional false positives summary.
    print("\nADDITIONAL FALSE CRITICAL ALERTS")

    print(f"Total: {len(additional_false_positives):,}")

    print("\nBy actual class:")

    print(
        additional_false_positives["actual_class"]
        .value_counts()
        .to_string()
    )

    print("\nBaseline prediction before the extra alert:")

    print(
        additional_false_positives["baseline_class"]
        .value_counts()
        .to_string()
    )

    # False positives corrected summary.
    print("\nFALSE POSITIVES CORRECTED BY EXPERIMENT V2")

    print(f"Total: {len(removed_false_positives):,}")

    print("\nBy actual class:")

    print(
        removed_false_positives["actual_class"]
        .value_counts()
        .to_string()
    )

    print("\nBy site:")

    print(
        removed_false_positives["site"]
        .value_counts()
        .to_string()
    )

    print("\nBy month:")

    print(
        removed_false_positives["time"]
        .dt.month_name()
        .value_counts()
        .to_string()
    )

    # Site-level breakdown of additional false positives.
    site_summary = (
        additional_false_positives
        .groupby(["site", "actual_class"])
        .size()
        .unstack(fill_value=0)
    )

    site_summary["total_additional_alerts"] = (
        site_summary.sum(axis=1)
    )

    site_summary = site_summary.reset_index()

    # Month-level breakdown of additional false positives.
    additional_false_positives["month"] = (
        additional_false_positives["time"].dt.month
    )

    monthly_summary = (
        additional_false_positives
        .groupby(["month", "actual_class"])
        .size()
        .unstack(fill_value=0)
    )

    monthly_summary["total_additional_alerts"] = (
        monthly_summary.sum(axis=1)
    )

    monthly_summary = monthly_summary.reset_index()

    # Weather-feature comparison.
    weather_columns = [
        "wind_speed_100m",
        "wind_gusts_10m",
        "wave_height",
        "wave_period",
        "marine_stress",
    ]

    available_weather_columns = [
        column for column in weather_columns
        if column in test.columns
    ]

    if available_weather_columns:

        weather_data = test[
            available_weather_columns
        ].copy()

        weather_data["outcome"] = "Other"

        # Correctly predicted Critical.
        weather_data.loc[
            (comparison["y"] == 2)
            & (comparison["experiment_prediction"] == 2),
            "outcome",
        ] = "Correct Critical"

        # Additional false Critical alerts.
        weather_data.loc[
            additional_false_positives.index,
            "outcome",
        ] = "Additional False Critical"

        weather_summary = (
            weather_data
            .groupby("outcome")[available_weather_columns]
            .mean()
            .reset_index()
        )

    else:
        weather_summary = pd.DataFrame()

    # Create reports directory.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save additional false positive reports.
    additional_false_positives.to_csv(
        REPORT_DIR / "additional_false_positive_details.csv",
        index=False,
    )

    site_summary.to_csv(
        REPORT_DIR / "additional_false_positive_by_site.csv",
        index=False,
    )

    monthly_summary.to_csv(
        REPORT_DIR / "additional_false_positive_by_month.csv",
        index=False,
    )

    weather_summary.to_csv(
        REPORT_DIR / "additional_false_positive_weather.csv",
        index=False,
    )

    # Save corrected false positive cases.
    removed_false_positives.to_csv(
        REPORT_DIR / "removed_false_positive_details.csv",
        index=False,
    )

    # Display additional false positive breakdowns.
    print("\nADDITIONAL FALSE POSITIVES BY SITE")

    print(site_summary.to_string(index=False))

    print("\nADDITIONAL FALSE POSITIVES BY MONTH")

    print(monthly_summary.to_string(index=False))

    if not weather_summary.empty:

        print("\nAVERAGE WEATHER CONDITIONS")

        print(
            weather_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    print("\nREPORTS SAVED")

    print(REPORT_DIR / "additional_false_positive_details.csv")
    print(REPORT_DIR / "additional_false_positive_by_site.csv")
    print(REPORT_DIR / "additional_false_positive_by_month.csv")
    print(REPORT_DIR / "additional_false_positive_weather.csv")
    print(REPORT_DIR / "removed_false_positive_details.csv")

    print("\nFALSE POSITIVE RECONCILIATION")

    print(
        f"New false positives: {len(additional_false_positives):,}"
    )

    print(
        f"Corrected false positives: {len(removed_false_positives):,}"
    )

    print(
        "Net increase: "
        f"{len(additional_false_positives) - len(removed_false_positives):,}"
    )

    print("\nComparison complete.")
    print("Both trained models remain unchanged.")

    print("=" * 65)


if __name__ == "__main__":
    main()
