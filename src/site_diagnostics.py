import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Make project modules available when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
import labels


MODEL_DIR = ROOT / "models" / "extended"
OUTPUT_DIR = ROOT / "reports"


def class_distribution(frame):
    """Calculate the percentage of each risk class."""
    counts = (
        frame.groupby("site")["y"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )

    counts = counts.reindex(columns=[0, 1, 2], fill_value=0)
    counts.columns = ["Normal_pct", "Warning_pct", "Critical_pct"]

    return (counts * 100).round(2)


def evaluate_sites(test, predictions):
    """Calculate Critical-class metrics separately for each site."""
    results = []

    for site, group in test.groupby("site"):
        actual = group["y"].to_numpy()
        predicted = predictions.loc[group.index].to_numpy()

        critical_precision = precision_score(
            actual, predicted, labels=[2], average="macro", zero_division=0
        )

        critical_recall = recall_score(
            actual, predicted, labels=[2], average="macro", zero_division=0
        )

        missed_critical = int(((actual == 2) & (predicted != 2)).sum())

        predicted_normal = int(((actual == 2) & (predicted == 0)).sum())
        predicted_warning = int(((actual == 2) & (predicted == 1)).sum())

        results.append({
            "site": site,
            "test_hours": len(group),
            "critical_hours": int((actual == 2).sum()),
            "critical_precision": round(critical_precision, 4),
            "critical_recall": round(critical_recall, 4),
            "missed_critical": missed_critical,
            "missed_as_warning": predicted_warning,
            "missed_as_normal": predicted_normal,
        })

    return pd.DataFrame(results)


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

    print(f"Model features: {len(features)}")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")

    # Predict using the existing saved model. No retraining occurs.
    X_test = test[features]
    predictions = pd.Series(
        model.predict(scaler.transform(X_test)),
        index=test.index,
    )

    print("\nTRAIN CLASS DISTRIBUTION (%)")
    print(class_distribution(train).to_string())

    print("\nTEST CLASS DISTRIBUTION (%)")
    print(class_distribution(test).to_string())

    print("\nSITE-LEVEL CRITICAL PERFORMANCE")
    results = evaluate_sites(test, predictions)
    print(results.to_string(index=False))

    # Overall test performance.
    actual = test["y"].to_numpy()
    predicted = predictions.to_numpy()

    print("\nOVERALL TEST CONFUSION MATRIX")
    print("Rows = actual; columns = predicted")
    print("Class order: Normal (0), Warning (1), Critical (2)")
    print(confusion_matrix(actual, predicted, labels=[0, 1, 2]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "site_diagnostics.csv"
    results.to_csv(output_file, index=False)

    print(f"\nSite diagnostics saved to: {output_file}")


if __name__ == "__main__":
    main()
