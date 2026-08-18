from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .chemistry import descriptor_frame

RIMAY_MAX_FEATURES = 155


def export_rimay(df: pd.DataFrame, out_path: Path, feature_count: int = 96) -> Path:
    if feature_count > RIMAY_MAX_FEATURES:
        raise ValueError(f"Rimay export must stay below 156 features (got {feature_count})")
    features, names = descriptor_frame(df, feature_count)
    train_mask = df["split"] == "train"
    if not train_mask.any():
        raise ValueError("Rimay export requires a non-empty training split")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(features.loc[train_mask])
    scaler.fit(imputer.transform(features.loc[train_mask]))
    transformed = scaler.transform(imputer.transform(features))
    q = pd.DataFrame(transformed, columns=[f"qf_{i:03d}" for i in range(feature_count)], index=df.index)
    meta_cols = [c for c in ["molecule_id", "smiles", "active_label", "pchembl_value", "scaffold", "split"] if c in df]
    export = pd.concat([df[meta_cols].reset_index(drop=True), q.reset_index(drop=True)], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(out_path, index=False)
    pd.DataFrame({"rimay_feature": q.columns, "rdkit_descriptor": names}).to_csv(out_path.with_name(out_path.stem + "_feature_map.csv"), index=False)
    return out_path


def create_rimay_pilot(rimay_input: Path, out_dir: Path, sample_size: int = 300, seed: int = 42) -> dict:
    if not 200 <= sample_size <= 500:
        raise ValueError("Pilot sample_size must be between 200 and 500 molecules")
    df = pd.read_csv(rimay_input)
    required = {"molecule_id", "active_label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Rimay input missing required columns: {sorted(missing)}")
    if len(df) < sample_size:
        raise ValueError(f"Dataset has only {len(df)} rows; cannot create a {sample_size}-row pilot")
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_rand"] = rng.random(len(df))
    grouped = list(df.groupby(["split", "active_label"], sort=True))
    sizes = np.array([len(g) for _, g in grouped], dtype=int)
    raw = sizes / sizes.sum() * sample_size
    quotas = np.floor(raw).astype(int)
    if sample_size >= len(grouped):
        quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, sizes)
    while quotas.sum() < sample_size:
        capacity = sizes - quotas
        candidates = np.where(capacity > 0)[0]
        if len(candidates) == 0:
            break
        fractions = raw - np.floor(raw)
        idx = max(candidates, key=lambda i: (fractions[i], capacity[i]))
        quotas[idx] += 1
    while quotas.sum() > sample_size:
        candidates = np.where(quotas > 1)[0]
        if len(candidates) == 0:
            candidates = np.where(quotas > 0)[0]
        idx = min(candidates, key=lambda i: (raw[i] - np.floor(raw[i]), quotas[i]))
        quotas[idx] -= 1
    parts = [group.nsmallest(int(n), "_rand") for (_, group), n in zip(grouped, quotas, strict=True) if n > 0]
    pilot = pd.concat(parts, ignore_index=True).drop(columns="_rand").reset_index(drop=True)
    if len(pilot) != sample_size:
        raise RuntimeError(f"Pilot allocator produced {len(pilot)} rows instead of {sample_size}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(out_dir / "rimay_pilot.csv", index=False)
    feature_cols = [c for c in pilot.columns if c.startswith("qf_")]
    manifest = {
        "purpose": "Kipu Rimay simulator pilot handoff",
        "rows": int(len(pilot)), "feature_count": len(feature_cols), "label_column": "active_label",
        "id_column": "molecule_id", "split_column": "split",
        "split_counts": pilot["split"].value_counts().to_dict(),
        "class_counts": pilot["active_label"].value_counts().sort_index().to_dict(),
        "seed": seed,
        "scientific_boundary": "Integration/benchmark input only. It is not evidence of efficacy, safety, or quantum advantage.",
    }
    (out_dir / "rimay_pilot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text("# Rimay pilot bundle\n\nUpload `rimay_pilot.csv` to **Rimay – Quantum Feature Extraction – Simulator**.\n\nKeep `molecule_id`, `active_label`, and `split` intact. Do not create a new random split. Any preprocessing learned by the service must be learned from training rows only.\n\nWhen the run finishes, export either:\n\n1. a CSV containing `molecule_id` plus returned quantum feature columns, or\n2. a CSV containing `molecule_id` plus a probability column named `prediction`.\n\nThen run `pdl rimay-compare ...` to evaluate the result against the frozen classical benchmark.\n", encoding="utf-8")
    return manifest


def _metric_dict(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {"roc_auc": float(roc_auc_score(y_true, prob)) if y_true.nunique() == 2 else float("nan"), "pr_auc": float(average_precision_score(y_true, prob)), "f1": float(f1_score(y_true, pred, zero_division=0)), "accuracy": float(accuracy_score(y_true, pred)), "brier": float(brier_score_loss(y_true, prob))}


def _rimay_model_specs() -> dict[str, object]:
    return {
        "rimay_logreg": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "rimay_rf": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=350, random_state=42, class_weight="balanced_subsample", n_jobs=-1))]),
        "rimay_hgb": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=42))]),
    }


def compare_rimay_result(prepared_csv: Path, rimay_result_csv: Path, baseline_metrics_json: Path, out_path: Path) -> dict:
    prepared = pd.read_csv(prepared_csv)
    rimay = pd.read_csv(rimay_result_csv)
    if "molecule_id" not in rimay:
        raise ValueError("Rimay result must contain molecule_id for leakage-safe alignment")
    if rimay["molecule_id"].duplicated().any():
        raise ValueError("Rimay result contains duplicate molecule_id rows")
    keep = ["molecule_id", "active_label", "split"]
    merged = prepared[keep].merge(rimay, on="molecule_id", how="inner", validate="one_to_one")
    if len(merged) != len(prepared):
        raise ValueError(f"Rimay result does not cover the full frozen dataset ({len(prepared) - len(merged)} molecules missing)")
    if "active_label_x" in merged:
        merged = merged.rename(columns={"active_label_x": "active_label"})
        if "active_label_y" in merged:
            if not np.array_equal(merged["active_label"].to_numpy(), merged["active_label_y"].to_numpy()):
                raise ValueError("Rimay-returned labels disagree with frozen labels")
            merged = merged.drop(columns="active_label_y")
    if "split_x" in merged:
        merged = merged.rename(columns={"split_x": "split"})
        if "split_y" in merged:
            if not np.array_equal(merged["split"].to_numpy(), merged["split_y"].to_numpy()):
                raise ValueError("Rimay-returned splits disagree with frozen scaffold splits")
            merged = merged.drop(columns="split_y")
    baseline = json.loads(baseline_metrics_json.read_text(encoding="utf-8"))
    best_baseline = baseline["best_model"]
    baseline_test = baseline["test"][best_baseline]
    train_mask = merged["split"] == "train"
    val_mask = merged["split"] == "validation"
    test_mask = merged["split"] == "test"
    y = merged["active_label"].astype(int)
    if "prediction" in merged.columns:
        prob = pd.to_numeric(merged.loc[test_mask, "prediction"], errors="raise").to_numpy()
        if np.any((prob < 0) | (prob > 1)):
            raise ValueError("prediction must be a probability in [0, 1]")
        rimay_test = _metric_dict(y.loc[test_mask], prob)
        selected, validation, mode = "rimay_prediction", None, "prediction"
    else:
        excluded = {"molecule_id", "active_label", "split", "smiles", "scaffold", "pchembl_value"}
        feature_cols = [c for c in merged.columns if c not in excluded and pd.api.types.is_numeric_dtype(merged[c])]
        if not feature_cols:
            raise ValueError("No numeric Rimay feature columns found")
        X = merged[feature_cols]
        specs = _rimay_model_specs()
        validation = {}
        for name, prototype in specs.items():
            model = clone(prototype)
            model.fit(X.loc[train_mask], y.loc[train_mask])
            validation[name] = _metric_dict(y.loc[val_mask], model.predict_proba(X.loc[val_mask])[:, 1])
        selected = max(validation, key=lambda n: (validation[n]["pr_auc"], validation[n]["roc_auc"]))
        model = clone(specs[selected])
        fit_mask = train_mask | val_mask
        model.fit(X.loc[fit_mask], y.loc[fit_mask])
        rimay_test = _metric_dict(y.loc[test_mask], model.predict_proba(X.loc[test_mask])[:, 1])
        mode = "features"
    delta = {key: float(rimay_test[key] - baseline_test[key]) for key in ["roc_auc", "pr_auc", "f1", "accuracy", "brier"]}
    delta["brier_improvement"] = float(baseline_test["brier"] - rimay_test["brier"])
    verdict = "promising" if delta["pr_auc"] > 0 and delta["roc_auc"] > 0 else "no_clear_gain"
    payload = {"mode": mode, "selected_rimay_model": selected, "best_classical_baseline": best_baseline, "classical_test": baseline_test, "rimay_validation": validation, "rimay_test": rimay_test, "delta_rimay_minus_classical": delta, "verdict": verdict, "claim_boundary": "A single held-out comparison is not sufficient to claim quantum advantage. Repeat across scaffold seeds and report compute/runtime cost before making a claim."}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
