from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import DEFAULT_DESCRIPTOR_COUNT, DEFAULT_FP_BITS

PRIORITY_DESCRIPTORS = [
    "MolWt", "ExactMolWt", "HeavyAtomMolWt", "MolLogP", "MolMR", "TPSA",
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount",
    "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings", "FractionCSP3",
    "NHOHCount", "NOCount", "HeavyAtomCount", "LabuteASA", "BalabanJ", "BertzCT",
    "Chi0", "Chi1", "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n",
    "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3", "NumHeteroatoms",
    "NumAmideBonds", "NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters",
]


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def canonical_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold_smiles(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return "INVALID"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or Chem.MolToSmiles(mol, canonical=True)


@lru_cache(maxsize=8)
def descriptor_names(count: int = DEFAULT_DESCRIPTOR_COUNT) -> tuple[str, ...]:
    available = {name for name, _ in Descriptors._descList}
    ordered: list[str] = []
    for name in PRIORITY_DESCRIPTORS:
        if name in available and name not in ordered:
            ordered.append(name)
    for name in sorted(available):
        if name not in ordered and not name.startswith("BCUT2D_"):
            ordered.append(name)
    if count > len(ordered):
        raise ValueError(f"Requested {count} descriptors, only {len(ordered)} available")
    return tuple(ordered[:count])


def descriptors_for_smiles(smiles: str, names: tuple[str, ...] | None = None) -> dict[str, float]:
    names = names or descriptor_names()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {name: math.nan for name in names}
    funcs = dict(Descriptors._descList)
    out: dict[str, float] = {}
    for name in names:
        try:
            value = float(funcs[name](mol))
            out[name] = value if math.isfinite(value) else math.nan
        except Exception:
            out[name] = math.nan
    return out


def descriptor_frame(df: pd.DataFrame, count: int = DEFAULT_DESCRIPTOR_COUNT) -> tuple[pd.DataFrame, list[str]]:
    names = list(descriptor_names(count))
    rows = [descriptors_for_smiles(s, tuple(names)) for s in df["smiles"].tolist()]
    return pd.DataFrame(rows, index=df.index), names


def morgan_frame(df: pd.DataFrame, bits: int = DEFAULT_FP_BITS, radius: int = 2) -> pd.DataFrame:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    matrix = np.zeros((len(df), bits), dtype=np.uint8)
    for i, smiles in enumerate(df["smiles"]):
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        matrix[i] = arr
    return pd.DataFrame(matrix, index=df.index, columns=[f"morgan_{i}" for i in range(bits)])


def property_proxies(smiles: str) -> dict[str, float | bool]:
    """Transparent heuristics for ranking, not validated ADMET predictors."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {
            "mol_wt": math.nan, "logp": math.nan, "tpsa": math.nan,
            "hbd": math.nan, "hba": math.nan, "rotatable": math.nan,
            "lipinski_pass": False, "cns_likeness_proxy": 0.0,
            "drug_likeness_proxy": 0.0,
        }
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rot = int(Lipinski.NumRotatableBonds(mol))
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    lipinski = violations <= 1

    cns_checks = [mw <= 450, 1.0 <= logp <= 4.0, tpsa <= 90, hbd <= 3, rot <= 8]
    cns = sum(cns_checks) / len(cns_checks)
    drug_checks = [mw <= 500, -0.5 <= logp <= 5.0, tpsa <= 140, hbd <= 5, hba <= 10, rot <= 10]
    drug = sum(drug_checks) / len(drug_checks)
    return {
        "mol_wt": mw, "logp": logp, "tpsa": tpsa, "hbd": hbd, "hba": hba,
        "rotatable": rot, "lipinski_pass": bool(lipinski),
        "cns_likeness_proxy": float(cns), "drug_likeness_proxy": float(drug),
    }
