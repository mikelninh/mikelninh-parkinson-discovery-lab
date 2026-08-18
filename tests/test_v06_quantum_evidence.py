from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from parkinson_discovery.demo_data import make_demo_dataset
from parkinson_discovery.pipeline import run_pipeline
from parkinson_discovery.quantum_evidence import (
    paired_bootstrap_delta,
    run_quantum_trial,
    summarize_quantum_trials,
)


def test_paired_bootstrap_detects_clear_pr_auc_improvement():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    classical = np.array([0.45, 0.55, 0.60, 0.40, 0.52, 0.48, 0.51, 0.49])
    quantum = np.array([0.05, 0.95, 0.10, 0.90, 0.15, 0.85, 0.20, 0.80])
    result = paired_bootstrap_delta(y, classical, quantum, "pr_auc", n_bootstrap=400, seed=7)
    assert result["delta"] > 0
    assert result["probability_improvement"] > 0.9
    assert result["bootstrap_replicates_valid"] > 0


def test_quantum_trial_writes_molecule_level_disagreement_evidence(tmp_path: Path):
    source = make_demo_dataset()
    run_dir = tmp_path / "classical"
    run_pipeline(source, run_dir, descriptor_count=24)
    prepared = pd.read_csv(run_dir / "dataset_prepared.csv")

    # Deterministic returned probabilities, deliberately based on labels only for plumbing tests.
    # These synthetic values are not a scientific quantum result.
    rimay = prepared[["molecule_id", "active_label", "split"]].copy()
    rimay["prediction"] = np.where(rimay["active_label"].eq(1), 0.90, 0.10)
    rimay_path = tmp_path / "rimay.csv"
    rimay.to_csv(rimay_path, index=False)

    result = run_quantum_trial(
        run_dir / "dataset_prepared.csv",
        run_dir / "best_model.joblib",
        rimay_path,
        tmp_path / "trial",
        n_bootstrap=300,
        seed=11,
        backend_type="simulator",
        backend_name="synthetic-test",
        quantum_runtime_seconds=12.5,
        quantum_cost_eur=0.0,
    )
    assert result["test_molecules"] > 0
    assert result["compute"]["backend_type"] == "simulator"
    assert result["verdict"] in {"pass", "fail", "inconclusive"}
    evidence = pd.read_csv(tmp_path / "trial" / "paired_test_predictions.csv")
    assert {
        "classical_probability",
        "quantum_probability",
        "absolute_probability_disagreement",
        "disagreement_outcome",
    }.issubset(evidence.columns)


def _trial_payload(trial_id: str, delta: float, backend: str = "simulator") -> dict:
    return {
        "trial_id": trial_id,
        "verdict": "pass" if delta > 0 else "fail",
        "paired_bootstrap": {"pr_auc": {"delta": delta}},
        "compute": {
            "backend_type": backend,
            "quantum_runtime_seconds": 10.0,
            "quantum_cost_eur": 0.5,
        },
    }


def test_quantum_meta_requires_repeated_trials_and_aggregates_compute(tmp_path: Path):
    paths = []
    for i, delta in enumerate([0.04, 0.05, 0.06, 0.03, 0.05]):
        path = tmp_path / f"trial_{i}.json"
        path.write_text(json.dumps(_trial_payload(f"t{i}", delta)), encoding="utf-8")
        paths.append(path)
    result = summarize_quantum_trials(paths, tmp_path / "meta.json", n_bootstrap=1000, seed=3)
    assert result["trial_count"] == 5
    assert result["verdict"] == "pass"
    assert result["across_trial_bootstrap"]["ci95_low"] > 0
    assert result["compute"]["total_reported_quantum_runtime_seconds"] == 50.0
    assert result["compute"]["total_reported_quantum_cost_eur"] == 2.5


def test_quantum_meta_stays_inconclusive_with_only_two_trials(tmp_path: Path):
    paths = []
    for i in range(2):
        path = tmp_path / f"small_{i}.json"
        path.write_text(json.dumps(_trial_payload(f"s{i}", 0.10)), encoding="utf-8")
        paths.append(path)
    result = summarize_quantum_trials(paths, tmp_path / "small_meta.json")
    assert result["verdict"] == "inconclusive_insufficient_repeated_trials"
    assert result["across_trial_bootstrap"]["ci95_low"] is None
