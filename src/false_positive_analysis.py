
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


def analyse_model(name, model_dir, test, features):

    print(f"\nEvaluating {name}...")

    model = joblib.load(model_dir / "best_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    saved_features = joblib.load(model_dir / "features.joblib")

    if list(saved_features) != list(features):
        raise ValueError(f"Feature mismatch: {name}")

    X_test = scaler.transform(test[features])
    y_true = test["y"].to_numpy()
    predictions = model.predict(X_test)

    details = test[["time", "site", "y"]].copy()
    details["model"] = name
    details["predicted"] = predictions

    details["actual_class"] = details["y"].map({
        0: "Normal",
        1: "Warning",
        2: "Critical",
    })

    details["predicted_class"] = details["predicted"].map({
        0: "Normal",
        1: "Warning",
        2: "Critical",
    })

    # Actual non-Critical observations.
    details["is_non_critical"] = details["y"] != 2

    # False Critical alerts.
    details["false_critical_alert"] = (
        (details["predicted"] == 2)
        & details["is_non_critical"]
    )

    fp = int(details["false_critical_alert"].sum())
    actual_non_critical = int(details["is_non_critical"].sum())

    summary = {
        "model": name,
        "false_critical_alerts": fp,
        "actual_normal_predicted_critical": int(
            ((y_true == 0) & (predictions == 2)).sum()
        ),
        "actual_warning_predicted_critical": int(
            ((y_true == 1) & (predictions == 2)).sum()
        ),
        "actual_non_critical": actual_non_critical,
        "false_positive_rate": (
            fp / actual_non_critical
            if actual_non_critical else 0
        ),
    }

    return summary, details


def calculate_group_rates(data, group_column):

    result = (
        data.groupby(["model", group_column], as_index=False)
        .agg(
            false_critical_alerts=("false_critical_alert", "sum"),
            actual_non_critical=("is_non_critical", "sum"),
        )
    )

    result["false_positive_rate"] = (
        result["false_critical_alerts"]
        / result["actual_non_critical"]
    )

    result["false_positive_rate_percent"] = (
        result["false_positive_rate"] * 100
    )

    return result


def main():

    print("=" * 60)
    print("FALSE POSITIVE ANALYSIS")
    print("=" * 60)

    print("\nLoading dataset...")

    raw = pd.read_csv(cfg.CSV_PATH, parse_dates=["time"])

    data = labels.build(
        raw,
        cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS],
        "extended",
    )

    train, validation, test = labels.time_split(data)
    features = labels.feature_columns(data)

    print(f"Test rows: {len(test):,}")

    baseline_summary, baseline_details = analyse_model(
        "Extended Baseline",
        BASELINE_DIR,
        test,
        features,
    )

    experiment_summary, experiment_details = analyse_model(
        "Experiment V2",
        EXPERIMENT_DIR,
        test,
        features,
    )

    # Overall comparison.
    summaries = pd.DataFrame([
        baseline_summary,
        experiment_summary,
    ])

    print("\nOVERALL FALSE-POSITIVE RESULTS")
    print(summaries.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))

    difference = (
        experiment_summary["false_critical_alerts"]
        - baseline_summary["false_critical_alerts"]
    )

    print("\nCHANGE IN FALSE CRITICAL ALERTS")
    print(f"Experiment V2 minus baseline: {difference:+,}")

    # Combine predictions from both models.
    all_scored = pd.concat(
        [baseline_details, experiment_details],
        ignore_index=True,
    )

    # Keep false-positive cases for detailed inspection.
    all_false_positives = all_scored[
        all_scored["false_critical_alert"]
    ].copy()

    # Site-level false-positive rates.
    site_summary = calculate_group_rates(
        all_scored,
        "site",
    )

    # Month-level false-positive rates.
    all_scored["month"] = all_scored["time"].dt.month
    all_scored["month_name"] = all_scored["time"].dt.month_name()

    monthly_summary = calculate_group_rates(
        all_scored,
        "month",
    )

    monthly_names = (
        all_scored[["month", "month_name"]]
        .drop_duplicates()
    )

    monthly_summary = monthly_summary.merge(
        monthly_names,
        on="month",
        how="left",
    )

    monthly_summary = monthly_summary.sort_values(
        ["model", "month"]
    )

    # Save reports.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_false_positives.to_csv(
        REPORT_DIR / "false_positive_details.csv",
        index=False,
    )

    summaries.to_csv(
        REPORT_DIR / "false_positive_summary.csv",
        index=False,
    )

    site_summary.to_csv(
        REPORT_DIR / "false_positive_by_site.csv",
        index=False,
    )

    monthly_summary.to_csv(
        REPORT_DIR / "false_positive_by_month.csv",
        index=False,
    )

    # Display site results.
    print("\nFALSE-POSITIVE RATES BY SITE")
    print(site_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))

    # Display monthly results.
    print("\nFALSE-POSITIVE RATES BY MONTH")
    print(monthly_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))

    print("\nREPORTS SAVED")
    print(REPORT_DIR / "false_positive_details.csv")
    print(REPORT_DIR / "false_positive_summary.csv")
    print(REPORT_DIR / "false_positive_by_site.csv")
    print(REPORT_DIR / "false_positive_by_month.csv")

    print("\nFalse-positive analysis complete.")
    print("Both trained models remain unchanged.")

    print("=" * 60)


if __name__ == "__main__":
    main()
