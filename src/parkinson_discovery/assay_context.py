from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _quantiles(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"median": None, "p90": None, "max": None}
    return {
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
    }


def assay_context_summary(df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {"molecules": int(len(df))}
    if "standard_types" in df:
        counts: dict[str, int] = {}
        for value in df["standard_types"].fillna(""):
            for item in [x for x in str(value).split(";") if x]:
                counts[item] = counts.get(item, 0) + 1
        summary["molecules_by_standard_type"] = dict(sorted(counts.items()))
    if "assay_heterogeneity_flag" in df:
        flags = df["assay_heterogeneity_flag"].astype(str).str.lower().isin(["true", "1"])
        summary["heterogeneous_molecules"] = int(flags.sum())
        summary["heterogeneous_fraction"] = float(flags.mean()) if len(flags) else 0.0
    if "pchembl_iqr" in df:
        summary["pchembl_iqr"] = _quantiles(df["pchembl_iqr"])
    if "label_agreement" in df:
        values = pd.to_numeric(df["label_agreement"], errors="coerce").dropna()
        summary["label_agreement"] = {
            "median": float(values.median()) if not values.empty else None,
            "p10": float(values.quantile(0.10)) if not values.empty else None,
            "below_0_75": int((values < 0.75).sum()) if not values.empty else 0,
        }
    if "context_quality_score" in df:
        values = pd.to_numeric(df["context_quality_score"], errors="coerce").dropna()
        summary["context_quality_score"] = {
            "mean": float(values.mean()) if not values.empty else None,
            "median": float(values.median()) if not values.empty else None,
            "low_quality_below_0_5": int((values < 0.5).sum()) if not values.empty else 0,
        }
    if "measurement_count" in df:
        values = pd.to_numeric(df["measurement_count"], errors="coerce").dropna()
        summary["measurement_count"] = {
            "median": float(values.median()) if not values.empty else None,
            "max": int(values.max()) if not values.empty else None,
            "multi_measurement_molecules": int((values > 1).sum()) if not values.empty else 0,
        }
    return summary


def write_assay_context_audit(df: pd.DataFrame, out_path: Path) -> dict[str, object]:
    payload = assay_context_summary(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload
