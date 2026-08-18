from pathlib import Path

from parkinson_discovery.demo_data import make_demo_dataset
from parkinson_discovery.repeats import run_repeated_scaffold_benchmark


def test_repeated_scaffold_benchmark(tmp_path: Path):
    payload = run_repeated_scaffold_benchmark(
        make_demo_dataset(),
        tmp_path,
        descriptor_count=32,
        seeds=(42, 71),
    )
    assert payload["successful_seeds"]
    assert "summary" in payload
    assert (tmp_path / "repeated_metrics.csv").exists()
    assert (tmp_path / "repeated_summary.json").exists()
