
import sys
import json
import time
from pathlib import Path
from itertools import product

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as cfg
import labels
from train_models import macro_f1, crit_recall, grids


BASELINE_DIR = ROOT / "models" / "extended"
OUTPUT_DIR = ROOT / "models" / "experiment_v2"

# Preselected candidate based on the validation comparison.
SELECTED_WEIGHT = 1.5
SELECTED_MAX_DEPTH = 6
SELECTED_LEARNING_RATE = 0.1


def main():
    print("EXPERIMENT V2: CRITICAL-CLASS WEIGHTING")
    print("The 2026 test set will NOT be evaluated.")

    # Load existing baseline validation results.
    results_file = BASELINE_DIR / "results.json"

    with open(results_file, "r") as file:
        baseline_results = json.load(file)

    baseline_model = baseline_results["winner"]
    baseline_metrics = baseline_results["models"][baseline_model]

    baseline_recall = baseline_metrics["val_crit_recall"]
    baseline_f1 = baseline_metrics["val_macro_f1"]

    print(f"\nExisting baseline: {baseline_model}")
    print(f"Baseline validation Critical recall: {baseline_recall:.4f}")
    print(f"Baseline validation Macro-F1: {baseline_f1:.4f}")

    # Rebuild the same extended dataset and chronological split.
    print("\nLoading weather dataset...")

    raw = pd.read_csv(cfg.CSV_PATH, parse_dates=["time"])

    data = labels.build(
        raw,
        cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS],
        "extended",
    )

    train, validation, test = labels.time_split(data)

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print("Test rows: reserved and untouched")

    features = labels.feature_columns(data)

    # Fit scaler only on training data.
    scaler = StandardScaler().fit(train[features])

    X_train = scaler.transform(train[features])
    X_val = scaler.transform(validation[features])

    y_train = train["y"].to_numpy()
    y_val = validation["y"].to_numpy()

    # Original balanced sample weights.
    base_weights = compute_sample_weight("balanced", y_train)

    # Same XGBoost grid as the original training pipeline.
    xgb_factory, parameter_grid = grids(False)["XGBoost"]

    multipliers = [1.25, 1.5, 2.0]

    trials = []
    selected_model = None
    selected_result = None

    print("\nStarting experiments...")

    for multiplier, params in product(multipliers, parameter_grid):

        weights = base_weights.copy()
        weights[y_train == 2] *= multiplier

        print(
            f"\nTraining: Critical weight {multiplier}x | "
            f"Parameters: {params}"
        )

        model = xgb_factory(params)

        start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
            sample_weight=weights,
        )

        fit_seconds = time.perf_counter() - start

        predictions = model.predict(X_val)

        val_f1 = macro_f1(y_val, predictions)
        val_recall = crit_recall(y_val, predictions)

        val_precision = precision_score(
            y_val,
            predictions,
            labels=[2],
            average=None,
            zero_division=0,
        )[0]

        result = {
            "critical_weight_multiplier": multiplier,
            "parameters": params,
            "val_macro_f1": val_f1,
            "val_critical_recall": val_recall,
            "val_critical_precision": float(val_precision),
            "fit_seconds": round(fit_seconds, 2),
            "confusion_matrix": confusion_matrix(
                y_val,
                predictions,
                labels=[0, 1, 2],
            ).tolist(),
        }

        trials.append(result)

        print(
            f"Validation Macro-F1: {val_f1:.4f} | "
            f"Critical recall: {val_recall:.4f} | "
            f"Critical precision: {val_precision:.4f}"
        )

        # Save only the preselected validation candidate.
        if (
            multiplier == SELECTED_WEIGHT
            and params["max_depth"] == SELECTED_MAX_DEPTH
            and params["learning_rate"] == SELECTED_LEARNING_RATE
        ):
            selected_model = model
            selected_result = result

    if selected_model is None:
        raise RuntimeError(
            "Selected candidate was not found in the XGBoost parameter grid."
        )

    # Save experiment outputs separately from the baseline.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        selected_model,
        OUTPUT_DIR / "best_model.joblib",
        compress=3,
    )

    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")
    joblib.dump(features, OUTPUT_DIR / "features.joblib")

    trials_df = pd.DataFrame([
        {
            "critical_weight_multiplier": item["critical_weight_multiplier"],
            "parameters": str(item["parameters"]),
            "val_macro_f1": item["val_macro_f1"],
            "val_critical_recall": item["val_critical_recall"],
            "val_critical_precision": item["val_critical_precision"],
            "fit_seconds": item["fit_seconds"],
        }
        for item in trials
    ])

    trials_df = trials_df.sort_values(
        ["val_critical_recall", "val_macro_f1"],
        ascending=False,
    )

    trials_df.to_csv(
        OUTPUT_DIR / "validation_trials.csv",
        index=False,
    )

    experiment_results = {
        "experiment": "Critical-class sample-weight adjustment",
        "feature_set": "extended",
        "threshold_set": cfg.ACTIVE_THRESHOLDS,
        "baseline_model": baseline_model,
        "baseline_validation_macro_f1": baseline_f1,
        "baseline_validation_critical_recall": baseline_recall,
        "selection": {
            "critical_weight_multiplier": SELECTED_WEIGHT,
            "max_depth": SELECTED_MAX_DEPTH,
            "learning_rate": SELECTED_LEARNING_RATE,
            "selection_reason": (
                "Preselected validation candidate offering a balance "
                "between Critical recall and Macro-F1."
            ),
        },
        "selected_result": selected_result,
        "trials": trials,
        "test_set_evaluated": False,
    }

    with open(OUTPUT_DIR / "results.json", "w") as file:
        json.dump(experiment_results, file, indent=2)

    print("\nEXPERIMENT COMPLETE")

    print("\nBASELINE VALIDATION")
    print(f"Macro-F1: {baseline_f1:.4f}")
    print(f"Critical recall: {baseline_recall:.4f}")

    print("\nSELECTED EXPERIMENT VALIDATION")
    print(f"Critical weight: {SELECTED_WEIGHT}x")
    print(f"Parameters: {selected_result['parameters']}")
    print(f"Macro-F1: {selected_result['val_macro_f1']:.4f}")
    print(f"Critical recall: {selected_result['val_critical_recall']:.4f}")
    print(f"Critical precision: {selected_result['val_critical_precision']:.4f}")

    print("\nCHANGE FROM BASELINE")
    print(
        "Critical recall change: "
        f"{(selected_result['val_critical_recall'] - baseline_recall) * 100:+.2f} "
        "percentage points"
    )

    print(
        "Macro-F1 change: "
        f"{(selected_result['val_macro_f1'] - baseline_f1) * 100:+.2f} "
        "percentage points"
    )

    print("\nSaved experiment files:")
    print(OUTPUT_DIR / "best_model.joblib")
    print(OUTPUT_DIR / "scaler.joblib")
    print(OUTPUT_DIR / "features.joblib")
    print(OUTPUT_DIR / "validation_trials.csv")
    print(OUTPUT_DIR / "results.json")

    print("\nThe original extended model remains unchanged.")
    print("The 2026 test set was not evaluated.")


if __name__ == "__main__":
    main()
