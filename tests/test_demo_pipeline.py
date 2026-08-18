from pathlib import Path

import pandas as pd

from parkinson_discovery.demo_data import make_demo_dataset
from parkinson_discovery.pipeline import run_pipeline


def test_demo_pipeline_end_to_end(tmp_path: Path):
    df = make_demo_dataset()
    assert len(df) >= 100
    assert df["active_label"].nunique() == 2

    result = run_pipeline(df, tmp_path)
    assert result["manifest"]["molecules"] >= 100
    assert result["manifest"]["version"] == "0.3.0"
    assert result["manifest"]["artifact_sha256"]
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "ranked_candidates.csv").exists()
    assert (tmp_path / "rimay_input.csv").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "assay_context_summary.json").exists()

    rimay = pd.read_csv(tmp_path / "rimay_input.csv")
    feature_cols = [c for c in rimay if c.startswith("qf_")]
    assert len(feature_cols) == 96
    assert set(rimay["split"]) == {"train", "validation", "test"}
