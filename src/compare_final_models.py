
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
import labels


BASELINE_DIR = ROOT / "models" / "extended"
EXPERIMENT_DIR = ROOT / "models" / "experiment_v2"
REPORT_DIR = ROOT / "reports"


def evaluate_model(name, model_dir, test, features):
    """Evaluate one saved model on the same test dataset."""

    print(f"\nEvaluating {name}...")

    model = joblib.load(model_dir / "best_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    saved_features = joblib.load(model_dir / "features.joblib")

    if list(saved_features) != list(features):
        raise ValueError(
            f"Feature mismatch for {name}. "
            "Check the saved feature list."
        )

    X_test = scaler.transform(test[features])
    y_test = test["y"].to_numpy()

    predictions = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test, predictions, average="macro"
        ),
        "critical_precision": precision_score(
            y_test, predictions, labels=[2],
            average=None, zero_division=0
        )[0],
        "critical_recall": recall_score(
            y_test, predictions, labels=[2],
            average=None, zero_division=0
        )[0],
        "critical_f1": f1_score(
            y_test, predictions, labels=[2],
            average=None, zero_division=0
        )[0],
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1, 2]
        ).tolist(),
    }

    # Number of actual Critical cases missed by the model.
    actual_critical = y_test == 2
    missed_critical = actual_critical & (predictions != 2)

    metrics["total_critical"] = int(actual_critical.sum())
    metrics["missed_critical"] = int(missed_critical.sum())
    metrics["missed_as_warning"] = int(
        ((y_test == 2) & (predictions == 1)).sum()
    )
    metrics["missed_as_normal"] = int(
        ((y_test == 2) & (predictions == 0)).sum()
    )

    return metrics, predictions


def calculate_site_metrics(test, predictions, model_name):
    """Calculate Critical recall separately for each site."""

    results = []

    for site, indices in test.groupby("site").groups.items():

        actual = test.loc[indices, "y"].to_numpy()
        predicted = predictions[test.index.get_indexer(indices)]

        critical_mask = actual == 2

        total_critical = int(critical_mask.sum())
        correctly_detected = int(
            ((actual == 2) & (predicted == 2)).sum()
        )

        missed = total_critical - correctly_detected

        results.append({
            "model": model_name,
            "site": site,
            "total_critical": total_critical,
            "detected_critical": correctly_detected,
            "missed_critical": missed,
            "critical_recall": (
                correctly_detected / total_critical
                if total_critical else np.nan
            ),
        })

    return results


def main():

    print("=" * 65)
    print("FINAL MODEL COMPARISON")
    print("The reserved 2026 test set is now being evaluated.")
    print("=" * 65)

    # Rebuild exactly the same dataset and chronological split.
    print("\nLoading weather dataset...")

    raw = pd.read_csv(
        cfg.CSV_PATH,
        parse_dates=["time"],
    )

    data = labels.build(
        raw,
        cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS],
        "extended",
    )

    train, validation, test = labels.time_split(data)

    features = labels.feature_columns(data)

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Number of features: {len(features)}")

    # Evaluate the original baseline.
    baseline_metrics, baseline_predictions = evaluate_model(
        "Extended Baseline",
        BASELINE_DIR,
        test,
        features,
    )

    # Evaluate Experiment V2.
    experiment_metrics, experiment_predictions = evaluate_model(
        "Experiment V2",
        EXPERIMENT_DIR,
        test,
        features,
    )

    # Display overall comparison.
    comparison = pd.DataFrame([
        {
            key: value
            for key, value in baseline_metrics.items()
            if key != "confusion_matrix"
        },
        {
            key: value
            for key, value in experiment_metrics.items()
            if key != "confusion_matrix"
        },
    ])

    print("\n" + "=" * 65)
    print("OVERALL TEST RESULTS")
    print("=" * 65)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # Calculate differences: Experiment V2 minus baseline.
    metric_names = [
        "accuracy",
        "macro_f1",
        "critical_precision",
        "critical_recall",
        "critical_f1",
    ]

    print("\n" + "=" * 65)
    print("CHANGE FROM BASELINE")
    print("=" * 65)

    for metric in metric_names:

        difference = (
            experiment_metrics[metric] - baseline_metrics[metric]
        ) * 100

        print(f"{metric}: {difference:+.2f} percentage points")

    print(
        "\nCritical cases missed by baseline:",
        baseline_metrics["missed_critical"],
    )

    print(
        "Critical cases missed by Experiment V2:",
        experiment_metrics["missed_critical"],
    )

    print(
        "Difference in missed Critical cases:",
        experiment_metrics["missed_critical"]
        - baseline_metrics["missed_critical"],
    )

    # Per-site results.
    baseline_sites = calculate_site_metrics(
        test,
        baseline_predictions,
        "Extended Baseline",
    )

    experiment_sites = calculate_site_metrics(
        test,
        experiment_predictions,
        "Experiment V2",
    )

    site_results = pd.DataFrame(
        baseline_sites + experiment_sites
    )

    site_pivot = site_results.pivot(
        index="site",
        columns="model",
        values="critical_recall",
    )

    site_pivot["change_percentage_points"] = (
        site_pivot["Experiment V2"]
        - site_pivot["Extended Baseline"]
    ) * 100

    print("\n" + "=" * 65)
    print("PER-SITE CRITICAL RECALL")
    print("=" * 65)

    print(
        site_pivot.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # Save reports.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        REPORT_DIR / "final_model_comparison.csv",
        index=False,
    )

    site_results.to_csv(
        REPORT_DIR / "final_site_comparison.csv",
        index=False,
    )

    with open(
        REPORT_DIR / "final_model_results.json",
        "w",
    ) as file:

        json.dump(
            {
                "test_period": "2026 test split",
                "test_set_evaluated": True,
                "baseline": baseline_metrics,
                "experiment_v2": experiment_metrics,
            },
            file,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 65)

    print("\nSaved reports:")
    print(REPORT_DIR / "final_model_comparison.csv")
    print(REPORT_DIR / "final_site_comparison.csv")
    print(REPORT_DIR / "final_model_results.json")

    print("\nBoth saved models remain unchanged.")


if __name__ == "__main__":
    main()