from __future__ import annotations

import pandas as pd

from .chemistry import property_proxies
from .models import predict_bundle


def rank_candidates(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    ranked = df.copy()
    ranked["predicted_activity"] = predict_bundle(bundle, ranked)
    props = pd.DataFrame([property_proxies(s) for s in ranked["smiles"]], index=ranked.index)
    ranked = pd.concat([ranked, props], axis=1)
    ranked["rank_score"] = (
        0.65 * ranked["predicted_activity"]
        + 0.20 * ranked["cns_likeness_proxy"]
        + 0.15 * ranked["drug_likeness_proxy"]
    )
    ranked = ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked
