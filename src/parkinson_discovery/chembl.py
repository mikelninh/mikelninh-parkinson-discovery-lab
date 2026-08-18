from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .chemistry import canonical_smiles, scaffold_smiles
from .config import ACTIVE_PCHEMBL, INACTIVE_PCHEMBL, TARGETS
from .provenance import artifact_hashes, canonical_json_sha256

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


class ChEMBLClient:
    def __init__(self, timeout: int = 30, pause: float = 0.05):
        self.timeout = timeout
        self.pause = pause
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "parkinson-discovery-lab/0.3"})

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self.session.get(f"{BASE_URL}/{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def activities(self, target_chembl_id: str, limit: int = 1000) -> list[dict]:
        out: list[dict] = []
        offset = 0
        while True:
            payload = self._get(
                "activity.json",
                {"target_chembl_id": target_chembl_id, "limit": limit, "offset": offset},
            )
            batch = payload.get("activities", [])
            if not batch:
                break
            out.extend(batch)
            offset += len(batch)
            page_meta = payload.get("page_meta") or {}
            if not page_meta.get("next"):
                break
            time.sleep(self.pause)
        return out

    def molecule_smiles(self, ids: Iterable[str], chunk_size: int = 50) -> dict[str, str]:
        ids = list(dict.fromkeys(ids))
        out: dict[str, str] = {}
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i : i + chunk_size]
            try:
                payload = self._get(f"molecule/set/{','.join(chunk)}.json")
                molecules = payload.get("molecules", [])
            except requests.HTTPError:
                molecules = []
                for molecule_id in chunk:
                    try:
                        molecules.append(self._get(f"molecule/{molecule_id}.json"))
                    except requests.HTTPError:
                        continue
            for mol in molecules:
                structures = mol.get("molecule_structures") or {}
                smi = structures.get("canonical_smiles")
                if smi:
                    out[mol["molecule_chembl_id"]] = smi
            time.sleep(self.pause)
        return out

    def assays(self, ids: Iterable[str], chunk_size: int = 50) -> dict[str, dict]:
        ids = [x for x in dict.fromkeys(ids) if x]
        out: dict[str, dict] = {}
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i : i + chunk_size]
            try:
                payload = self._get(f"assay/set/{','.join(chunk)}.json")
                assays = payload.get("assays", [])
            except requests.HTTPError:
                assays = []
                for assay_id in chunk:
                    try:
                        assays.append(self._get(f"assay/{assay_id}.json"))
                    except requests.HTTPError:
                        continue
            for assay in assays:
                assay_id = assay.get("assay_chembl_id")
                if assay_id:
                    out[assay_id] = assay
            time.sleep(self.pause)
        return out

    def release_info(self) -> dict:
        payload = self._get("chembl_release.json", {"limit": 1000})
        rows: list[dict] = []
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                rows = value
                break
        if not rows:
            return {"raw": payload}

        def sort_key(row: dict) -> tuple[int, str]:
            release_id = row.get("chembl_release_id") or row.get("release_id") or 0
            try:
                release_id = int(release_id)
            except (TypeError, ValueError):
                release_id = 0
            return release_id, str(row.get("creation_date") or row.get("release_date") or "")

        return max(rows, key=sort_key)


def _assay_context(assay: dict | None) -> dict[str, object]:
    assay = assay or {}
    return {
        "assay_type": assay.get("assay_type"),
        "assay_category": assay.get("assay_category"),
        "assay_organism": assay.get("assay_organism"),
        "assay_tax_id": assay.get("assay_tax_id"),
        "assay_confidence_score": assay.get("confidence_score"),
        "assay_relationship_type": assay.get("relationship_type"),
        "bao_format": assay.get("bao_format"),
    }


def _label_for_value(value: float) -> int | None:
    if value >= ACTIVE_PCHEMBL:
        return 1
    if value <= INACTIVE_PCHEMBL:
        return 0
    return None


def _safe_iqr(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.quantile(0.75) - values.quantile(0.25))


def _clean_target_activities(
    activities: list[dict],
    target_id: str,
    standard_types: tuple[str, ...],
    client: ChEMBLClient | object,
    max_pchembl_iqr: float | None = None,
    min_label_agreement: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict], dict[str, int]]:
    stats = {
        "raw_activities": len(activities),
        "filtered_standard_type": 0,
        "filtered_relation": 0,
        "filtered_missing_pchembl": 0,
        "filtered_missing_molecule": 0,
        "canonical_measurements": 0,
        "ambiguous_structures": 0,
        "quality_filtered_structures": 0,
    }

    cleaned: list[dict] = []
    missing_smiles: set[str] = set()
    for a in activities:
        if a.get("standard_type") not in standard_types:
            stats["filtered_standard_type"] += 1
            continue
        if a.get("standard_relation") != "=":
            stats["filtered_relation"] += 1
            continue
        if a.get("pchembl_value") in (None, ""):
            stats["filtered_missing_pchembl"] += 1
            continue
        try:
            value = float(a["pchembl_value"])
        except (TypeError, ValueError):
            stats["filtered_missing_pchembl"] += 1
            continue
        if not math.isfinite(value):
            stats["filtered_missing_pchembl"] += 1
            continue
        molecule_id = a.get("molecule_chembl_id")
        if not molecule_id:
            stats["filtered_missing_molecule"] += 1
            continue
        smi = a.get("canonical_smiles")
        if not smi:
            missing_smiles.add(molecule_id)
        cleaned.append(
            {
                "activity_id": a.get("activity_id"),
                "molecule_id": molecule_id,
                "smiles": smi,
                "pchembl_value": value,
                "standard_type": a.get("standard_type"),
                "assay_id": a.get("assay_chembl_id"),
                "document_id": a.get("document_chembl_id"),
            }
        )

    if not cleaned:
        raise RuntimeError("No usable ChEMBL activities returned for the requested target/filters")

    smiles_lookup = client.molecule_smiles(missing_smiles) if missing_smiles else {}
    assay_ids = [row["assay_id"] for row in cleaned if row.get("assay_id")]
    assay_lookup = client.assays(assay_ids) if hasattr(client, "assays") else {}

    with_context: list[dict] = []
    for row in cleaned:
        if not row["smiles"]:
            row["smiles"] = smiles_lookup.get(row["molecule_id"])
        row["smiles"] = canonical_smiles(row["smiles"] or "")
        if not row["smiles"]:
            continue
        row.update(_assay_context(assay_lookup.get(row.get("assay_id"))))
        with_context.append(row)

    raw = pd.DataFrame(with_context)
    stats["canonical_measurements"] = len(raw)
    if raw.empty:
        raise RuntimeError("No ChEMBL activities had a usable molecular structure")

    grouped: list[dict] = []
    for smiles, g in raw.groupby("smiles", sort=False):
        values = g["pchembl_value"].astype(float)
        median = float(values.median())
        label = _label_for_value(median)
        if label is None:
            stats["ambiguous_structures"] += 1
            continue
        iqr = _safe_iqr(values)
        agreement = 0.0
        if label == 1:
            agreement = float((values >= ACTIVE_PCHEMBL).mean())
        else:
            agreement = float((values <= INACTIVE_PCHEMBL).mean())

        if max_pchembl_iqr is not None and iqr > max_pchembl_iqr:
            stats["quality_filtered_structures"] += 1
            continue
        if agreement < min_label_agreement:
            stats["quality_filtered_structures"] += 1
            continue

        standard_type_values = sorted(set(g["standard_type"].dropna().astype(str)))
        assay_type_values = sorted(set(g["assay_type"].dropna().astype(str)))
        organism_values = sorted(set(g["assay_organism"].dropna().astype(str)))
        confidence = pd.to_numeric(g["assay_confidence_score"], errors="coerce").dropna()
        heterogeneity = (
            len(standard_type_values) > 1
            or len(assay_type_values) > 1
            or len(organism_values) > 1
            or iqr >= 1.0
        )
        quality_score = float(max(0.0, min(1.0, agreement / (1.0 + iqr))))

        grouped.append(
            {
                "molecule_id": sorted(g["molecule_id"].unique())[0],
                "smiles": smiles,
                "pchembl_value": median,
                "pchembl_iqr": iqr,
                "pchembl_min": float(values.min()),
                "pchembl_max": float(values.max()),
                "active_label": label,
                "measurement_count": int(len(g)),
                "assay_count": int(g["assay_id"].dropna().nunique()),
                "document_count": int(g["document_id"].dropna().nunique()),
                "standard_type_count": len(standard_type_values),
                "standard_types": ";".join(standard_type_values),
                "assay_type_count": len(assay_type_values),
                "assay_types": ";".join(assay_type_values),
                "assay_organisms": ";".join(organism_values),
                "assay_confidence_min": float(confidence.min()) if not confidence.empty else None,
                "assay_confidence_max": float(confidence.max()) if not confidence.empty else None,
                "label_agreement": agreement,
                "context_quality_score": quality_score,
                "assay_heterogeneity_flag": bool(heterogeneity),
                "assay_ids": ";".join(sorted(set(g["assay_id"].dropna().astype(str)))),
                "document_ids": ";".join(sorted(set(g["document_id"].dropna().astype(str)))),
                "activity_ids": ";".join(sorted(set(g["activity_id"].dropna().astype(str)))),
                "source": f"ChEMBL target {target_id}",
                "scaffold": scaffold_smiles(smiles),
            }
        )

    result = pd.DataFrame(grouped)
    if result.empty:
        raise RuntimeError("All records became ambiguous or failed assay-context quality filters")
    stats["final_molecules"] = len(result)
    stats["active_molecules"] = int(result["active_label"].sum())
    stats["inactive_molecules"] = int((1 - result["active_label"]).sum())
    stats["heterogeneous_molecules"] = int(result["assay_heterogeneity_flag"].sum())
    return result.sort_values("molecule_id").reset_index(drop=True), raw, assay_lookup, stats


def build_target_dataset(
    target: str = "LRRK2",
    standard_types: tuple[str, ...] = ("IC50", "Ki"),
    client: ChEMBLClient | None = None,
    max_pchembl_iqr: float | None = None,
    min_label_agreement: float = 0.0,
) -> pd.DataFrame:
    """Pull, clean, deduplicate and aggregate quantitative ChEMBL target activity data.

    Assay context is preserved as metadata and label-quality diagnostics, but is not
    silently fed into the molecular ML feature matrix. This avoids teaching a model
    assay artefacts while still making heterogeneity visible and filterable.
    """
    target_id = TARGETS.get(target, target)
    client = client or ChEMBLClient()
    activities = client.activities(target_id)
    result, _, _, _ = _clean_target_activities(
        activities,
        target_id,
        standard_types,
        client,
        max_pchembl_iqr=max_pchembl_iqr,
        min_label_agreement=min_label_agreement,
    )
    return result


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def freeze_target_snapshot(
    target: str,
    out_dir: Path,
    standard_types: tuple[str, ...] = ("IC50", "Ki"),
    client: ChEMBLClient | None = None,
    max_pchembl_iqr: float | None = None,
    min_label_agreement: float = 0.0,
) -> dict:
    """Freeze source records, assay metadata, cleaned molecules and integrity hashes.

    The goal is to make a benchmark refer to a fixed ChEMBL-derived snapshot instead
    of an ever-changing live query. The snapshot can be verified later byte-for-byte.
    """
    target_id = TARGETS.get(target, target)
    client = client or ChEMBLClient()
    out_dir.mkdir(parents=True, exist_ok=True)

    activities = client.activities(target_id)
    cleaned, _, assay_lookup, stats = _clean_target_activities(
        activities,
        target_id,
        standard_types,
        client,
        max_pchembl_iqr=max_pchembl_iqr,
        min_label_agreement=min_label_agreement,
    )
    try:
        release = client.release_info() if hasattr(client, "release_info") else {"unknown": True}
    except (requests.RequestException, ValueError, TypeError):
        release = {"unknown": True}

    raw_path = out_dir / "raw_activities.jsonl"
    assays_path = out_dir / "raw_assays.jsonl"
    cleaned_path = out_dir / "cleaned_molecules.csv"
    _write_jsonl(raw_path, activities)
    _write_jsonl(assays_path, [assay_lookup[k] for k in sorted(assay_lookup)])
    cleaned.to_csv(cleaned_path, index=False)

    query_contract = {
        "target": target,
        "target_chembl_id": target_id,
        "standard_types": list(standard_types),
        "standard_relation": "=",
        "requires_pchembl_value": True,
        "active_pchembl_gte": ACTIVE_PCHEMBL,
        "inactive_pchembl_lte": INACTIVE_PCHEMBL,
        "max_pchembl_iqr": max_pchembl_iqr,
        "min_label_agreement": min_label_agreement,
        "aggregation": "median by canonical SMILES",
    }
    snapshot_seed = {
        "release": release,
        "query_contract": query_contract,
        "cleaned_sha256": artifact_hashes(out_dir, [cleaned_path.name])[cleaned_path.name],
    }
    snapshot_id = f"chembl-{target_id.lower()}-{canonical_json_sha256(snapshot_seed)[:12]}"
    hashed_files = [raw_path.name, assays_path.name, cleaned_path.name]
    manifest = {
        "manifest_type": "chembl_snapshot",
        "schema_version": 1,
        "pdl_version": "0.3.0",
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "ChEMBL",
            "base_url": BASE_URL,
            "release": release,
        },
        "query_contract": query_contract,
        "counts": stats,
        "artifact_sha256": artifact_hashes(out_dir, hashed_files),
        "artifacts": hashed_files,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (out_dir / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return manifest
