from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors, Crippen

from .chemistry import mol_from_smiles, canonical_smiles, scaffold_smiles

BASES = [
    "c1ccccc1{}",
    "c1ccncc1{}",
    "c1ccc2ccccc2c1{}",
    "c1ccc2[nH]ccc2c1{}",
    "c1ccc2ncccc2c1{}",
    "c1ccc2occc2c1{}",
    "c1ccc2sccc2c1{}",
    "c1ccc2[nH]ncc2c1{}",
]
SUBSTITUENTS = [
    "F", "Cl", "Br", "C", "CC", "CCC", "O", "OC", "OCC", "N", "NC",
    "N(C)C", "C#N", "C(=O)N", "C(=O)O", "C(F)(F)F", "S(=O)(=O)N",
    "C1CCCCC1", "N1CCOCC1",
]


def _noise(text: str) -> float:
    digest = hashlib.sha256(text.encode()).digest()
    return (int.from_bytes(digest[:4], "big") / 2**32 - 0.5) * 0.35


def make_demo_dataset() -> pd.DataFrame:
    rows = []
    for base_i, base in enumerate(BASES):
        candidates = []
        for sub in SUBSTITUENTS:
            smiles = canonical_smiles(base.format(sub))
            mol = mol_from_smiles(smiles or "")
            if mol is None:
                continue
            score = (
                0.65 * Crippen.MolLogP(mol)
                - 0.010 * Descriptors.TPSA(mol)
                - 0.0015 * Descriptors.MolWt(mol)
                + 0.08 * Descriptors.NumHAcceptors(mol)
                + _noise(smiles)
            )
            candidates.append((smiles, score))
        candidates.sort(key=lambda x: x[1])
        mid = len(candidates) // 2
        for rank, (smiles, score) in enumerate(candidates):
            active = int(rank >= mid)
            pchembl = (6.25 + 0.25 * max(score, 0)) if active else (4.75 + 0.15 * min(score, 0))
            rows.append({
                "molecule_id": f"DEMO_{base_i:02d}_{rank:02d}",
                "smiles": smiles,
                "pchembl_value": float(np.clip(pchembl, 4.1, 7.4)),
                "active_label": active,
                "measurement_count": 1,
                "standard_types": "SYNTHETIC",
                "assay_ids": "demo",
                "document_ids": "demo",
                "source": "synthetic_demo_not_biological_data",
                "scaffold": scaffold_smiles(smiles),
            })
    df = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
    return df
