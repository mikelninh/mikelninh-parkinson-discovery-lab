import json
from pathlib import Path

import numpy as np
import pandas as pd

from parkinson_discovery.demo_data import make_demo_dataset
from parkinson_discovery.pipeline import run_pipeline
from parkinson_discovery.quantum import compare_rimay_result, create_rimay_pilot


def test_rimay_pilot_is_stratified_and_exact_size(tmp_path: Path):
    n = 600
    split = np.array(["train"] * 420 + ["validation"] * 90 + ["test"] * 90)
    label = np.arange(n) % 2
    df = pd.DataFrame({
        "molecule_id": [f"M{i:04d}" for i in range(n)],
        "active_label": label,
        "split": split,
        **{f"qf_{j:03d}": np.linspace(0, 1, n) + j for j in range(96)},
    })
    inp = tmp_path / "rimay_input.csv"
    df.to_csv(inp, index=False)
    manifest = create_rimay_pilot(inp, tmp_path / "pilot", sample_size=300, seed=42)
    pilot = pd.read_csv(tmp_path / "pilot" / "rimay_pilot.csv")
    assert len(pilot) == 300
    assert manifest["feature_count"] == 96
    assert set(pilot["split"]) == {"train", "validation", "test"}
    assert set(pilot["active_label"]) == {0, 1}


def test_rimay_prediction_comparison_aligns_by_molecule_id(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_pipeline(make_demo_dataset(), run_dir)
    prepared = pd.read_csv(run_dir / "dataset_prepared.csv")
    result = prepared[["molecule_id", "active_label", "split"]].copy()
    result["prediction"] = np.where(result["active_label"] == 1, 0.9, 0.1)
    result = result.sample(frac=1, random_state=7).reset_index(drop=True)
    result_path = tmp_path / "rimay_predictions.csv"
    result.to_csv(result_path, index=False)
    payload = compare_rimay_result(
        run_dir / "dataset_prepared.csv",
        result_path,
        run_dir / "metrics.json",
        run_dir / "quantum_comparison.json",
    )
    assert payload["mode"] == "prediction"
    assert payload["rimay_test"]["roc_auc"] == 1.0
    assert (run_dir / "quantum_comparison.json").exists()


def test_rimay_feature_comparison_accepts_export(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_pipeline(make_demo_dataset(), run_dir)
    payload = compare_rimay_result(
        run_dir / "dataset_prepared.csv",
        run_dir / "rimay_input.csv",
        run_dir / "metrics.json",
        run_dir / "quantum_comparison.json",
    )
    assert payload["mode"] == "features"
    assert payload["selected_rimay_model"].startswith("rimay_")
    saved = json.loads((run_dir / "quantum_comparison.json").read_text())
    assert "delta_rimay_minus_classical" in saved
