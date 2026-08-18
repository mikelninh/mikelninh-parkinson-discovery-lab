from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from .chemistry import canonical_smiles, mol_from_smiles
from .provenance import canonical_json_sha256, sha256_file


FILTER_CATALOGS = ("PAINS", "BRENK", "NIH", "ZINC")


@lru_cache(maxsize=None)
def _catalog(name: str) -> FilterCatalog:
    key = name.upper()
    if key not in FILTER_CATALOGS:
        raise ValueError(f"Unknown filter catalog {name!r}")
    enum_value = getattr(FilterCatalogParams.FilterCatalogs, key)
    params = FilterCatalogParams()
    params.AddCatalog(enum_value)
    return FilterCatalog(params)


def structural_alerts(smiles: str) -> dict[str, Any]:
    """Return RDKit catalog matches as evidence, not automatic rejection truth."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {
            "structural_alert_count": 0,
            "structural_alert_catalog_count": 0,
            "structural_alert_catalogs": "",
            "structural_alerts": "",
            "invalid_smiles": True,
        }
    matches: list[str] = []
    catalogs_hit: list[str] = []
    for name in FILTER_CATALOGS:
        entries = list(_catalog(name).GetMatches(mol))
        if entries:
            catalogs_hit.append(name)
        matches.extend(f"{name}:{entry.GetDescription()}" for entry in entries)
    unique_matches = sorted(set(matches))
    return {
        "structural_alert_count": len(unique_matches),
        "structural_alert_catalog_count": len(catalogs_hit),
        "structural_alert_catalogs": ";".join(catalogs_hit),
        "structural_alerts": ";".join(unique_matches),
        "invalid_smiles": False,
    }


def medicinal_properties(smiles: str) -> dict[str, Any]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {
            "qed": np.nan,
            "bertz_complexity": np.nan,
            "heavy_atom_count": np.nan,
            "ring_count": np.nan,
            "stereo_center_count": np.nan,
            "synthetic_complexity_proxy": np.nan,
        }
    bertz = float(Descriptors.BertzCT(mol))
    heavy = int(Descriptors.HeavyAtomCount(mol))
    rings = int(Descriptors.RingCount(mol))
    stereo = len(mol.GetSubstructMatches(mol)) * 0  # keep deterministic; replaced below
    try:
        from rdkit.Chem import rdMolDescriptors

        stereo = int(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
    except (AttributeError, TypeError):
        stereo = 0
    # Transparent bounded complexity heuristic; not a retrosynthesis/synthetic-accessibility model.
    proxy = min(1.0, max(0.0, 0.45 * min(bertz / 1200.0, 1.0) + 0.30 * min(heavy / 60.0, 1.0)
                         + 0.15 * min(rings / 8.0, 1.0) + 0.10 * min(stereo / 6.0, 1.0)))
    return {
        "qed": float(QED.qed(mol)),
        "bertz_complexity": bertz,
        "heavy_atom_count": heavy,
        "ring_count": rings,
        "stereo_center_count": stereo,
        "synthetic_complexity_proxy": float(proxy),
    }


def _fingerprints(smiles_values: list[str]):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    output = []
    for smiles in smiles_values:
        mol = mol_from_smiles(smiles)
        output.append(gen.GetFingerprint(mol) if mol is not None else None)
    return output


def nearest_known_chemistry(candidates: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    required = {"smiles"}
    if not required.issubset(candidates.columns) or not required.issubset(reference.columns):
        raise ValueError("Candidates and reference chemistry must contain a smiles column")
    ref = reference.copy()
    ref["smiles"] = ref["smiles"].map(canonical_smiles)
    ref = ref.dropna(subset=["smiles"]).drop_duplicates("smiles").reset_index(drop=True)
    if ref.empty:
        raise ValueError("Reference chemistry is empty after canonicalization")
    if "molecule_id" not in ref.columns:
        ref["molecule_id"] = [f"REF_{i:06d}" for i in range(len(ref))]

    query_smiles = [canonical_smiles(x) for x in candidates["smiles"]]
    if any(x is None for x in query_smiles):
        raise ValueError("Candidate data contains invalid SMILES")
    ref_fps = _fingerprints(ref["smiles"].tolist())
    ref_fps_valid = [fp for fp in ref_fps if fp is not None]
    ref_index = [i for i, fp in enumerate(ref_fps) if fp is not None]
    query_fps = _fingerprints([x for x in query_smiles if x is not None])

    rows = []
    for fp in query_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps_valid)
        best_pos = int(np.argmax(sims))
        best_idx = ref_index[best_pos]
        similarity = float(sims[best_pos])
        rows.append({
            "nearest_known_molecule_id": str(ref.iloc[best_idx]["molecule_id"]),
            "nearest_known_smiles": str(ref.iloc[best_idx]["smiles"]),
            "nearest_known_similarity": similarity,
            "novelty_proxy": float(1.0 - similarity),
        })
    return pd.DataFrame(rows, index=candidates.index)


def annotate_medchem(
    candidates: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "smiles" not in candidates.columns:
        raise ValueError("Candidate data must contain a smiles column")
    out = candidates.copy()
    out["smiles"] = out["smiles"].map(canonical_smiles)
    if out["smiles"].isna().any():
        raise ValueError("Candidate data contains invalid SMILES")

    properties = pd.DataFrame([medicinal_properties(s) for s in out["smiles"]], index=out.index)
    alerts = pd.DataFrame([structural_alerts(s) for s in out["smiles"]], index=out.index)
    out = pd.concat([out, properties, alerts], axis=1)
    if reference is not None:
        out = pd.concat([out, nearest_known_chemistry(out, reference)], axis=1)

    reasons: list[str] = []
    for _, row in out.iterrows():
        row_reasons = []
        if int(row["structural_alert_count"]) > 0:
            row_reasons.append("structural_alert_review")
        if "nearest_known_similarity" in out.columns and float(row["nearest_known_similarity"]) >= 0.90:
            row_reasons.append("very_close_to_reference_chemistry")
        if float(row["synthetic_complexity_proxy"]) >= 0.80:
            row_reasons.append("high_complexity_proxy")
        reasons.append(";".join(row_reasons))
    out["medchem_review_flags"] = reasons

    summary = {
        "molecules": len(out),
        "catalogs": list(FILTER_CATALOGS),
        "with_structural_alerts": int((out["structural_alert_count"] > 0).sum()),
        "with_reference": reference is not None,
        "policy": (
            "RDKit filter matches, QED, novelty and complexity are triage evidence. "
            "They do not establish toxicity, synthesizability or compound quality by themselves."
        ),
    }
    return out, summary


def write_medchem_artifact(
    candidates: pd.DataFrame,
    out_path: Path,
    reference: pd.DataFrame | None = None,
    reference_path: Path | None = None,
) -> dict[str, Any]:
    annotated, summary = annotate_medchem(candidates, reference)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(out_path, index=False)
    manifest = {
        "manifest_type": "pdl_medchem_evidence",
        "schema_version": 1,
        "version": "0.5.0",
        "output": out_path.name,
        "output_sha256": sha256_file(out_path),
        "reference": {
            "path": str(reference_path) if reference_path else None,
            "sha256": sha256_file(reference_path) if reference_path and reference_path.exists() else None,
        },
        "summary": summary,
        "contracts": {
            "structural_alert_catalogs": list(FILTER_CATALOGS),
            "novelty": "1 - max Morgan radius-2 2048-bit Tanimoto similarity to reference chemistry",
            "qed": "RDKit QED; descriptive drug-likeness metric",
            "synthetic_complexity_proxy": "transparent bounded descriptor heuristic; not synthetic accessibility",
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
