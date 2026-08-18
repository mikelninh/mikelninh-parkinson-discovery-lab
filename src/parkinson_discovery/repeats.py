from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import SplitConfig
from .models import train_benchmarks
from .pipeline import prepare_dataset

DEFAULT_SEEDS = (11, 23, 42, 71, 101)


def _mean_ci(values: pd.Series, confidence: float = 0.95) -> dict[str, float]:
    arr = values.dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) <= 1:
        return {"mean": mean, "std": std, "ci_low": mean, "ci_high": mean}
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    half = z * std / np.sqrt(len(arr))
    return {"mean": mean, "std": std, "ci_low": float(mean - half), "ci_high": float(mean + half)}


def run_repeated_scaffold_benchmark(
    df: pd.DataFrame,
    out_dir: Path,
    descriptor_count: int = 96,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict:
    """Repeat the full baseline benchmark across deterministic scaffold split seeds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    failures: list[dict] = []
    for seed in seeds:
        try:
            prepared = prepare_dataset(df, split_config=SplitConfig(seed=seed))
            seed_dir = out_dir / f"seed_{seed}"
            metrics, _ = train_benchmarks(prepared, seed_dir, descriptor_count)
            for model, values in metrics["test"].items():
                records.append({"seed": seed, "model": model, **values})
        except ValueError as exc:
            failures.append({"seed": seed, "error": str(exc)})

    if not records:
        raise RuntimeError("All repeated scaffold benchmarks failed")
    results = pd.DataFrame(records)
    results.to_csv(out_dir / "repeated_metrics.csv", index=False)
    summary: dict[str, dict] = {}
    for model, group in results.groupby("model"):
        summary[model] = {
            metric: _mean_ci(group[metric])
            for metric in ["roc_auc", "pr_auc", "f1", "accuracy", "brier"]
        }
    best_model = max(summary, key=lambda m: summary[m]["pr_auc"]["mean"])
    payload = {
        "requested_seeds": list(seeds),
        "successful_seeds": sorted(results["seed"].unique().astype(int).tolist()),
        "failures": failures,
        "descriptor_count": descriptor_count,
        "best_model_by_mean_pr_auc": best_model,
        "summary": summary,
    }
    (out_dir / "repeated_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
