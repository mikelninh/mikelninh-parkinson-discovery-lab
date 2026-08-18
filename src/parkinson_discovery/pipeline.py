from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .assay_context import write_assay_context_audit
from .chemistry import canonical_smiles, scaffold_smiles
from .config import ACTIVE_PCHEMBL, INACTIVE_PCHEMBL, SplitConfig
from .models import train_benchmarks
from .provenance import (
    artifact_hashes,
    canonical_json_sha256,
    environment_fingerprint,
    sha256_file,
)
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


def _source_snapshot(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_type") != "chembl_snapshot":
        raise ValueError(f"Source manifest is not a ChEMBL snapshot: {path}")
    return {
        "snapshot_id": payload.get("snapshot_id"),
        "manifest_sha256": payload.get("manifest_sha256"),
        "release": (payload.get("source") or {}).get("release"),
        "query_contract": payload.get("query_contract"),
        "manifest_path": str(path),
        "manifest_file_sha256": sha256_file(path),
    }


def run_pipeline(
    df: pd.DataFrame,
    out_dir: Path,
    descriptor_count: int = 96,
    split_config: SplitConfig = SplitConfig(),
    input_path: Path | None = None,
    source_manifest: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_dataset(df, split_config=split_config)
    prepared_path = out_dir / "dataset_prepared.csv"
    prepared.to_csv(prepared_path, index=False)

    assay_summary = write_assay_context_audit(prepared, out_dir / "assay_context_summary.json")
    metrics, bundle = train_benchmarks(prepared, out_dir, descriptor_count)
    ranked = rank_candidates(prepared, bundle)
    ranked.to_csv(out_dir / "ranked_candidates.csv", index=False)
    export_rimay(prepared, out_dir / "rimay_input.csv", descriptor_count)
    report = write_report(prepared, metrics, ranked, out_dir, assay_summary=assay_summary)

    config_contract = {
        "descriptor_count": descriptor_count,
        "split": {
            "train": split_config.train,
            "validation": split_config.validation,
            "test": split_config.test,
            "seed": split_config.seed,
            "strategy": "Bemis-Murcko scaffold split",
        },
        "labels": {
            "active_pchembl_gte": ACTIVE_PCHEMBL,
            "inactive_pchembl_lte": INACTIVE_PCHEMBL,
        },
        "selection": metrics["selection_rule"],
        "test_policy": "model family selected on validation; test touched once after refit",
    }
    prepared_hash = sha256_file(prepared_path)
    run_id = f"pdl-{canonical_json_sha256({'prepared': prepared_hash, 'config': config_contract})[:12]}"

    artifact_names = [
        "dataset_prepared.csv",
        "assay_context_summary.json",
        "metrics.json",
        "best_model.joblib",
        "ranked_candidates.csv",
        "rimay_input.csv",
        "rimay_input_feature_map.csv",
        "report.md",
    ]
    source = _source_snapshot(source_manifest)
    manifest = {
        "manifest_type": "pdl_run",
        "schema_version": 1,
        "version": "0.3.0",
        "run_id": run_id,
        "molecules": len(prepared),
        "unique_scaffolds": int(prepared["scaffold"].nunique()),
        "split_seed": split_config.seed,
        "split_counts": prepared["split"].value_counts().to_dict(),
        "active": int(prepared["active_label"].sum()),
        "inactive": int((1 - prepared["active_label"]).sum()),
        "best_model": metrics["best_model"],
        "rimay_status": "ready_for_pilot",
        "config_contract": config_contract,
        "config_sha256": canonical_json_sha256(config_contract),
        "environment": environment_fingerprint(),
        "input": {
            "path": str(input_path) if input_path else None,
            "sha256": sha256_file(input_path) if input_path and input_path.exists() else None,
        },
        "source_snapshot": source,
        "assay_context": assay_summary,
        "artifacts": artifact_names,
        "artifact_sha256": artifact_hashes(out_dir, artifact_names),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return {"manifest": manifest, "metrics": metrics, "report": str(report)}
