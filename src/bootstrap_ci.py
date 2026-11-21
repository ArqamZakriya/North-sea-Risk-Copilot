"""Day-block bootstrap confidence intervals for model comparisons.

Hourly predictions are strongly correlated, so resampling single hours would give intervals that
are far too narrow. This script resamples whole days with replacement instead, using the same days
for every model so that differences are paired.

Each model is given as name=directory or name=directory:ModelName. The directory holds
best_model.joblib, scaler.joblib, features.joblib and results.json from train_models.py. With
:ModelName the file candidates/ModelName.joblib is used instead of the winner. The persistence
baseline is added automatically.

Use --split val to choose between models and --split test to report the final result.

Usage:
    python src/bootstrap_ci.py --split val base=models extended=models/extended
    python src/bootstrap_ci.py --split test base=models extended=models/extended xgb_base=models:XGBoost rf_base=models:RandomForest
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import labels

PERCENTILES = (2.5, 97.5)


def split_spec(spec):
    """Split name=dir[:Model] into (name, directory, candidate). A drive letter colon is not a separator."""
    name, rest = spec.split("=", 1)
    path, sep, candidate = rest.rpartition(":")
    if not sep or not candidate or "\\" in candidate or "/" in candidate:
        return name, rest, ""
    return name, path, candidate


def load_spec(spec):
    """Parse name=dir[:Model] and load the model, scaler, features and feature set."""
    name, path, candidate = split_spec(spec)
    d = Path(path)
    model_path = d / "candidates" / f"{candidate}.joblib" if candidate else d / "best_model.joblib"
    meta = json.load(open(d / "results.json"))
    return name, joblib.load(model_path), joblib.load(d / "scaler.joblib"), joblib.load(d / "features.joblib"), meta.get("feature_set", "base")


def metrics_from_cm(cm):
    """Macro-F1 and Critical recall from confusion matrices of shape (..., 3, 3), rows true, columns predicted."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    pred, true = cm.sum(axis=-2), cm.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred > 0, tp / pred, 0.0)
        rec = np.where(true > 0, tp / true, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return f1.mean(axis=-1), rec[..., 2]


def daily_cms(y, pred, day):
    """Confusion matrix for every day, shape (days, 3, 3)."""
    cm = np.zeros((day.max() + 1, 3, 3), dtype=np.int64)
    np.add.at(cm, (day, y, pred), 1)
    return cm


def compare(df, specs, split="val", boot=2000, seed=cfg.SEED):
    """Predict with every model on the chosen split and bootstrap macro-F1 and Critical recall by day."""
    th = cfg.THRESHOLD_SETS[cfg.ACTIVE_THRESHOLDS]
    built, frames = {}, []
    for spec in specs:
        name, model, scaler, feats, fs = load_spec(spec)
        if fs not in built:
            built[fs] = labels.build(df, th, fs)
        tr, va, te = labels.time_split(built[fs])
        part = va if split == "val" else te
        pred = model.predict(scaler.transform(part[feats]))
        frames.append(part[["site", "time", "y", "y_persist"]].assign(**{name: pred}).set_index(["site", "time"]))
    common = frames[0]
    for f in frames[1:]:
        common = common.join(f.drop(columns=["y", "y_persist"]), how="inner")
    names = [load_spec(s)[0] for s in specs] + ["persistence"]
    common["persistence"] = common["y_persist"]
    y = common["y"].to_numpy(dtype=int)
    day = pd.factorize(common.index.get_level_values("time").floor("D"))[0]
    cms = {n: daily_cms(y, common[n].to_numpy(dtype=int), day) for n in names}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(next(iter(cms.values()))), size=(boot, len(next(iter(cms.values())))))
    f1, rec = {}, {}
    for n, cm in cms.items():
        f1[n], rec[n] = metrics_from_cm(cm[idx].sum(axis=1))
    point = {n: metrics_from_cm(cm.sum(axis=0)) for n, cm in cms.items()}

    def ci(x):
        lo, hi = np.percentile(x, PERCENTILES)
        return float(lo), float(hi)

    out = dict(split=split, hours=int(len(common)), days=int(len(next(iter(cms.values())))), bootstrap_samples=boot, models={}, differences={})
    for n in names:
        out["models"][n] = dict(macro_f1=float(point[n][0]), macro_f1_ci=ci(f1[n]),
                                crit_recall=float(point[n][1]), crit_recall_ci=ci(rec[n]))
    for i, a in enumerate(names):
        for b in names[:i]:
            key = f"{a} minus {b}"
            out["differences"][key] = dict(
                macro_f1=float(point[a][0] - point[b][0]), macro_f1_ci=ci(f1[a] - f1[b]),
                crit_recall=float(point[a][1] - point[b][1]), crit_recall_ci=ci(rec[a] - rec[b]))
    return out


def print_report(res):
    """Print the intervals and mark differences whose interval excludes zero."""
    print(f"Split: {res['split']}, {res['hours']} hours in {res['days']} days, {res['bootstrap_samples']} bootstrap samples\n")
    for n, r in res["models"].items():
        print(f"{n:<14} macro-F1 {r['macro_f1']:.3f} [{r['macro_f1_ci'][0]:.3f}, {r['macro_f1_ci'][1]:.3f}]   "
              f"Critical recall {r['crit_recall']:.3f} [{r['crit_recall_ci'][0]:.3f}, {r['crit_recall_ci'][1]:.3f}]")
    print("\nPaired differences (95% interval, * means the interval excludes zero):")
    for key, r in res["differences"].items():
        flag = lambda lo_hi: "*" if lo_hi[0] > 0 or lo_hi[1] < 0 else " "
        print(f"{key:<28} macro-F1 {r['macro_f1']:+.3f} [{r['macro_f1_ci'][0]:+.3f}, {r['macro_f1_ci'][1]:+.3f}]{flag(r['macro_f1_ci'])}  "
              f"Critical recall {r['crit_recall']:+.3f} [{r['crit_recall_ci'][0]:+.3f}, {r['crit_recall_ci'][1]:+.3f}]{flag(r['crit_recall_ci'])}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("specs", nargs="+", help="name=dir or name=dir:ModelName")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--boot", type=int, default=2000)
    a = p.parse_args()
    if not cfg.CSV_PATH.exists():
        raise SystemExit(f"{cfg.CSV_PATH} not found.")
    result = compare(pd.read_csv(cfg.CSV_PATH, parse_dates=["time"]), a.specs, a.split, a.boot)
    print_report(result)
    out = cfg.MODELS_DIR / f"bootstrap_{a.split}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved", out)
