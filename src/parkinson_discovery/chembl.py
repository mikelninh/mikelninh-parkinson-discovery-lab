from __future__ import annotations

import time
from collections import defaultdict
from typing import Iterable

import pandas as pd
import requests

from .chemistry import canonical_smiles, scaffold_smiles
from .config import ACTIVE_PCHEMBL, INACTIVE_PCHEMBL, TARGETS

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


class ChEMBLClient:
    def __init__(self, timeout: int = 30, pause: float = 0.05):
        self.timeout = timeout
        self.pause = pause
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "parkinson-discovery-lab/0.2"})

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


def build_target_dataset(
    target: str = "LRRK2",
    standard_types: tuple[str, ...] = ("IC50", "Ki"),
    client: ChEMBLClient | None = None,
) -> pd.DataFrame:
    """Pull, clean, deduplicate and aggregate quantitative ChEMBL target activity data."""
    target_id = TARGETS.get(target, target)
    client = client or ChEMBLClient()
    activities = client.activities(target_id)

    cleaned: list[dict] = []
    missing_smiles: set[str] = set()
    for a in activities:
        if a.get("standard_type") not in standard_types:
            continue
        if a.get("standard_relation") != "=":
            continue
        if a.get("pchembl_value") in (None, ""):
            continue
        try:
            value = float(a["pchembl_value"])
        except (TypeError, ValueError):
            continue
        molecule_id = a.get("molecule_chembl_id")
        if not molecule_id:
            continue
        smi = a.get("canonical_smiles")
        if not smi:
            missing_smiles.add(molecule_id)
        cleaned.append({
            "molecule_id": molecule_id,
            "smiles": smi,
            "pchembl_value": value,
            "standard_type": a.get("standard_type"),
            "assay_id": a.get("assay_chembl_id"),
            "document_id": a.get("document_chembl_id"),
        })

    if not cleaned:
        raise RuntimeError("No usable ChEMBL activities returned for the requested target/filters")

    smiles_lookup = client.molecule_smiles(missing_smiles) if missing_smiles else {}
    for row in cleaned:
        if not row["smiles"]:
            row["smiles"] = smiles_lookup.get(row["molecule_id"])
        row["smiles"] = canonical_smiles(row["smiles"] or "")

    raw = pd.DataFrame(cleaned).dropna(subset=["smiles"])
    grouped = []
    for smiles, g in raw.groupby("smiles", sort=False):
        median = float(g["pchembl_value"].median())
        if median >= ACTIVE_PCHEMBL:
            label = 1
        elif median <= INACTIVE_PCHEMBL:
            label = 0
        else:
            continue
        grouped.append({
            "molecule_id": sorted(g["molecule_id"].unique())[0],
            "smiles": smiles,
            "pchembl_value": median,
            "active_label": label,
            "measurement_count": int(len(g)),
            "standard_types": ";".join(sorted(set(g["standard_type"].dropna()))),
            "assay_ids": ";".join(sorted(set(g["assay_id"].dropna()))),
            "document_ids": ";".join(sorted(set(g["document_id"].dropna()))),
            "source": f"ChEMBL target {target_id}",
            "scaffold": scaffold_smiles(smiles),
        })
    result = pd.DataFrame(grouped)
    if result.empty:
        raise RuntimeError("All records became ambiguous after pChEMBL labeling")
    return result.sort_values("molecule_id").reset_index(drop=True)
