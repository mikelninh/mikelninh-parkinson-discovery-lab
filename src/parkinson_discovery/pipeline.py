from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .chemistry import canonical_smiles, scaffold_smiles
from .config import SplitConfig
from .models import train_benchmarks
from .quantum import export_rimay
from .ranking import rank_candidates
from .reporting import write_report
from .splits import scaffold_split

REQUIRED = {"molecule_id", "smiles", "active_label"}


def prepare_dataset(df: pd.DataFrame, split_config: SplitConfig = SplitConfig()) -> pd.DataFrame:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    out = df.copy()
    out["smiles"] = out["smiles"].map(canonical_smiles)
    out = out.dropna(subset=["smiles", "active_label"]).drop_duplicates("smiles").reset_index(drop=True)
    out["active_label"] = out["active_label"].astype(int)
    out = out[out["active_label"].isin([0, 1])].reset_index(drop=True)
    out["scaffold"] = out["smiles"].map(scaffold_smiles)
    out["split"] = scaffold_split(out, split_config)
    return out


def run_pipeline(
    df: pd.DataFrame,
    out_dir: Path,
    descriptor_count: int = 96,
    split_config: SplitConfig = SplitConfig(),
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_dataset(df, split_config=split_config)
    prepared.to_csv(out_dir / "dataset_prepared.csv", index=False)

    metrics, bundle = train_benchmarks(prepared, out_dir, descriptor_count)
    ranked = rank_candidates(prepared, bundle)
    ranked.to_csv(out_dir / "ranked_candidates.csv", index=False)
    export_rimay(prepared, out_dir / "rimay_input.csv", descriptor_count)
    report = write_report(prepared, metrics, ranked, out_dir)

    manifest = {
        "version": "0.2.0",
        "molecules": len(prepared),
        "unique_scaffolds": int(prepared["scaffold"].nunique()),
        "split_seed": split_config.seed,
        "split_counts": prepared["split"].value_counts().to_dict(),
        "active": int(prepared["active_label"].sum()),
        "inactive": int((1 - prepared["active_label"]).sum()),
        "best_model": metrics["best_model"],
        "rimay_status": "ready_for_pilot",
        "artifacts": [
            "dataset_prepared.csv", "metrics.json", "best_model.joblib",
            "ranked_candidates.csv", "rimay_input.csv", "rimay_input_feature_map.csv", "report.md",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics, "report": str(report)}
