from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd
import requests

from .chembl import BASE_URL, ChEMBLClient, freeze_target_snapshot
from .models import predict_bundle, train_benchmarks
from .pipeline import prepare_dataset
from .provenance import canonical_json_sha256, sha256_file


# Literature-seeded examples, not a claim that every LRRK2 chemotype shares these off-targets.
DEFAULT_LRRK2_PANEL = (
    "LRRK1",
    "TTK",
    "STK10",
    "MAPK14",
    "JNK2",
    "CLK1",
    "JNK3",
    "DYRK2",
    "SLK",
    "DDR2",
    "STK17B",
)

PANEL_EVIDENCE = {
    "type_ii_2025": {
        "citation": "doi:10.1126/sciadv.ads3128",
        "note": (
            "RN341 kinome profiling/off-target validation included JNK2, STK10, MAPK14, TTK, "
            "CDKL1, CLK1, JNK3, DYRK2, SLK, DDR2 and STK17B; LRRK1 was also assessed."
        ),
    },
    "macrocycle_2026": {
        "citation": "doi:10.1021/acs.jmedchem.6c00238",
        "note": "Modern LRRK2 optimisation explicitly considered kinome selectivity alongside CNS/safety properties.",
    },
}


def _normalise_target_text(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _target_aliases(row: dict[str, Any]) -> set[str]:
    aliases = {_normalise_target_text(row.get("pref_name"))}
    for item in row.get("target_synonyms") or []:
        if isinstance(item, dict):
            aliases.add(_normalise_target_text(item.get("component_synonym") or item.get("synonym")))
        elif isinstance(item, str):
            aliases.add(_normalise_target_text(item))
    for component in row.get("target_components") or []:
        if not isinstance(component, dict):
            continue
        aliases.add(_normalise_target_text(component.get("accession")))
        for synonym in component.get("target_component_synonyms") or []:
            if isinstance(synonym, dict):
                aliases.add(_normalise_target_text(synonym.get("component_synonym")))
    aliases.discard("")
    return aliases


def resolve_human_single_protein_target(
    query: str,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Resolve a target name against live ChEMBL; fail on ambiguity instead of guessing."""
    session = session or requests.Session()
    response = session.get(
        f"{BASE_URL}/target/search.json",
        params={"q": query, "limit": 100},
        timeout=timeout,
    )
    response.raise_for_status()
    rows = response.json().get("targets", [])
    human = [
        row for row in rows
        if str(row.get("organism") or "").lower() == "homo sapiens"
        and str(row.get("target_type") or "").upper() == "SINGLE PROTEIN"
    ]
    normalized = _normalise_target_text(query)
    exact = [row for row in human if normalized in _target_aliases(row)]
    candidates = exact or human
    if not candidates:
        raise LookupError(f"No human single-protein ChEMBL target found for {query!r}")
    unique = {row.get("target_chembl_id"): row for row in candidates if row.get("target_chembl_id")}
    if len(unique) != 1:
        options = [
            {
                "target_chembl_id": row.get("target_chembl_id"),
                "pref_name": row.get("pref_name"),
                "organism": row.get("organism"),
            }
            for row in unique.values()
        ]
        raise LookupError(f"Ambiguous ChEMBL target {query!r}; candidates={options}")
    row = next(iter(unique.values()))
    return {
        "query": query,
        "target_chembl_id": row["target_chembl_id"],
        "pref_name": row.get("pref_name"),
        "organism": row.get("organism"),
        "target_type": row.get("target_type"),
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def freeze_selectivity_panel(
    targets: Iterable[str],
    out_dir: Path,
    standard_types: tuple[str, ...] = ("IC50", "Ki"),
    client: ChEMBLClient | None = None,
    resolver=resolve_human_single_protein_target,
) -> dict[str, Any]:
    """Freeze target-specific ChEMBL datasets; each off-target retains its own provenance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    client = client or ChEMBLClient()
    entries = []
    for target in targets:
        resolution = resolver(target)
        target_dir = out_dir / _safe_name(target)
        snapshot = freeze_target_snapshot(
            resolution["target_chembl_id"],
            target_dir,
            standard_types=standard_types,
            client=client,
        )
        entries.append({
            "name": target,
            "resolution": resolution,
            "snapshot_dir": str(target_dir),
            "snapshot_id": snapshot.get("snapshot_id"),
            "manifest_sha256": snapshot.get("manifest_sha256"),
        })

    panel = {
        "manifest_type": "pdl_selectivity_panel",
        "schema_version": 1,
        "version": "0.5.0",
        "standard_types": list(standard_types),
        "targets": entries,
        "evidence": PANEL_EVIDENCE,
        "warning": (
            "This is a configurable literature-seeded surveillance panel. Off-target liabilities are chemotype- "
            "and assay-dependent; broad experimental kinome profiling remains the reference standard."
        ),
    }
    panel["manifest_sha256"] = canonical_json_sha256(panel)
    (out_dir / "panel_manifest.json").write_text(json.dumps(panel, indent=2), encoding="utf-8")
    return panel


def train_selectivity_panel(
    snapshots_dir: Path,
    out_dir: Path,
    descriptor_count: int = 96,
) -> dict[str, Any]:
    panel_manifest_path = snapshots_dir / "panel_manifest.json"
    if not panel_manifest_path.exists():
        raise FileNotFoundError(f"Missing {panel_manifest_path}")
    panel = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in panel.get("targets", []):
        name = entry["name"]
        source_dir = Path(entry["snapshot_dir"])
        if not source_dir.is_absolute():
            source_dir = snapshots_dir / _safe_name(name)
        cleaned_path = source_dir / "cleaned_molecules.csv"
        if not cleaned_path.exists():
            results.append({"target": name, "status": "skipped", "reason": "missing_cleaned_snapshot"})
            continue
        raw = pd.read_csv(cleaned_path)
        if len(raw) < 30 or raw["active_label"].nunique() < 2:
            results.append({
                "target": name,
                "status": "skipped",
                "reason": "insufficient_binary_training_data",
                "molecules": len(raw),
            })
            continue
        target_out = out_dir / _safe_name(name)
        try:
            prepared = prepare_dataset(raw)
            metrics, _ = train_benchmarks(prepared, target_out, descriptor_count=descriptor_count)
        except ValueError as exc:
            results.append({"target": name, "status": "skipped", "reason": str(exc), "molecules": len(raw)})
            continue
        metadata = {
            "target": name,
            "target_chembl_id": (entry.get("resolution") or {}).get("target_chembl_id"),
            "source_snapshot_id": entry.get("snapshot_id"),
            "source_manifest_sha256": entry.get("manifest_sha256"),
            "molecules": len(prepared),
            "best_model": metrics["best_model"],
            "test": metrics["test"][metrics["best_model"]],
            "model_sha256": sha256_file(target_out / "best_model.joblib"),
            "interpretation": "Predicted off-target activity probability; computational surveillance only.",
        }
        (target_out / "selectivity_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        results.append({"target": name, "status": "trained", **metadata})

    summary = {
        "manifest_type": "pdl_selectivity_models",
        "schema_version": 1,
        "version": "0.5.0",
        "source_panel_sha256": sha256_file(panel_manifest_path),
        "descriptor_count": descriptor_count,
        "results": results,
    }
    summary["manifest_sha256"] = canonical_json_sha256(summary)
    (out_dir / "selectivity_models_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def annotate_selectivity(candidates: pd.DataFrame, models_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "smiles" not in candidates.columns:
        raise ValueError("Candidate data must contain a smiles column")
    out = candidates.copy()
    model_dirs = sorted(p for p in models_dir.iterdir() if p.is_dir() and (p / "best_model.joblib").exists())
    if not model_dirs:
        raise FileNotFoundError(f"No selectivity models found under {models_dir}")

    used = []
    probability_columns = []
    names_by_column = {}
    for model_dir in model_dirs:
        model_meta_path = model_dir / "selectivity_model.json"
        metadata = json.loads(model_meta_path.read_text(encoding="utf-8")) if model_meta_path.exists() else {}
        target_name = str(metadata.get("target") or model_dir.name.upper())
        bundle = joblib.load(model_dir / "best_model.joblib")
        probability = predict_bundle(bundle, out)
        column = f"offtarget_{_safe_name(target_name)}_probability"
        out[column] = probability
        probability_columns.append(column)
        names_by_column[column] = target_name
        used.append(target_name)

    matrix = out[probability_columns]
    out["max_offtarget_probability"] = matrix.max(axis=1)
    worst_columns = matrix.idxmax(axis=1)
    out["highest_predicted_offtarget"] = worst_columns.map(names_by_column)
    if "predicted_activity" in out.columns:
        out["lrrk2_vs_max_offtarget_margin_proxy"] = (
            pd.to_numeric(out["predicted_activity"], errors="coerce") - out["max_offtarget_probability"]
        )

    summary = {
        "models_used": used,
        "molecules": len(out),
        "policy": (
            "Off-target probabilities are model-based surveillance signals, not experimental selectivity measurements. "
            "The max-off-target field is for triage and must not replace kinome profiling."
        ),
    }
    return out, summary
