from __future__ import annotations

from collections import defaultdict
import random

import pandas as pd

from .chemistry import scaffold_smiles
from .config import SplitConfig


def scaffold_split(df: pd.DataFrame, config: SplitConfig = SplitConfig()) -> pd.Series:
    """Deterministic Bemis-Murcko scaffold split with whole scaffolds kept together."""
    if len(df) < 10:
        raise ValueError("Need at least 10 molecules for a meaningful scaffold split")
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, smiles in zip(df.index, df["smiles"], strict=True):
        groups[scaffold_smiles(smiles)].append(idx)

    rng = random.Random(config.seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    target = {
        "train": config.train * len(df),
        "validation": config.validation * len(df),
        "test": config.test * len(df),
    }
    assigned = {k: 0 for k in target}
    split = pd.Series(index=df.index, dtype="object")

    for _, indices in items:
        destination = max(target, key=lambda k: (target[k] - assigned[k]) / max(target[k], 1))
        split.loc[indices] = destination
        assigned[destination] += len(indices)

    if split.isna().any():
        raise RuntimeError("Scaffold split left unassigned rows")
    return split
