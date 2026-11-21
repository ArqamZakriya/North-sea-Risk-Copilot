"""Offline tests of the bootstrap comparison."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bootstrap_ci as bc
import train_models as tm
from test_labels import synthetic


def test_metrics_from_cm():
    cm = np.array([[8, 2, 0], [1, 6, 1], [0, 2, 8]])
    f1, rec = bc.metrics_from_cm(cm)
    assert abs(rec - 0.8) < 1e-9
    p0, r0 = 8 / 9, 8 / 10
    f_edge = 2 * p0 * r0 / (p0 + r0)
    assert abs(f1 - np.mean([f_edge, 2 * 0.6 * 0.75 / 1.35, f_edge])) < 1e-9
    f1b, _ = bc.metrics_from_cm(np.stack([cm, cm]))
    assert f1b.shape == (2,)


def test_compare_models(tmp_path):
    df = synthetic()
    tm.run(df, "cited", quick=True, out_dir=tmp_path / "base")
    tm.run(df, "cited", quick=True, out_dir=tmp_path / "ext", feature_set="extended")
    specs = [f"base={tmp_path / 'base'}", f"ext={tmp_path / 'ext'}", f"rf={tmp_path / 'base'}:RandomForest"]
    res = bc.compare(df, specs, split="val", boot=200)
    assert set(res["models"]) == {"base", "ext", "rf", "persistence"}
    lo, hi = res["models"]["base"]["macro_f1_ci"]
    assert lo <= res["models"]["base"]["macro_f1"] <= hi
    assert "ext minus base" in res["differences"] and "persistence minus rf" in res["differences"]
    bc.print_report(res)


def test_split_spec_handles_windows_paths():
    assert bc.split_spec("a=models") == ("a", "models", "")
    assert bc.split_spec("a=models:XGBoost") == ("a", "models", "XGBoost")
    assert bc.split_spec("a=C:\\tmp\\base") == ("a", "C:\\tmp\\base", "")
    assert bc.split_spec("a=C:\\tmp\\base:RandomForest") == ("a", "C:\\tmp\\base", "RandomForest")
    assert bc.split_spec("a=C:/tmp/base") == ("a", "C:/tmp/base", "")
