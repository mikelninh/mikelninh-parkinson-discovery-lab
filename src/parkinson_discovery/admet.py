from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .chemistry import canonical_smiles, descriptor_frame, mol_from_smiles, morgan_frame, scaffold_smiles
from .config import SplitConfig
from .provenance import artifact_hashes, canonical_json_sha256, environment_fingerprint, sha256_file
from .splits import scaffold_split


ADMET_DATASETS: dict[str, dict[str, Any]] = {
    "bbbp": {
        "task": "classification",
        "target_column": "p_np",
        "smiles_column": "smiles",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "label_semantics": "1 = blood-brain-barrier penetrant; 0 = non-penetrant",
        "benchmark": "MoleculeNet BBBP; curated from Martins et al. 2012",
        "citation": "MoleculeNet: doi:10.1039/C7SC02664A; BBBP: doi:10.1021/ci300124c",
    },
    "clintox": {
        "task": "classification",
        "target_column": "CT_TOX",
        "smiles_column": "smiles",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "label_semantics": "1 = ClinTox toxicity label; 0 = no ClinTox toxicity label",
        "benchmark": "MoleculeNet ClinTox",
        "citation": "MoleculeNet: doi:10.1039/C7SC02664A",
    },
    "esol": {
        "task": "regression",
        "target_column": "measured log solubility in mols per litre",
        "smiles_column": "smiles",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "label_semantics": "measured aqueous log solubility (mol/L)",
        "benchmark": "Delaney ESOL / MoleculeNet",
        "citation": "ESOL: doi:10.1021/ci034243x; MoleculeNet: doi:10.1039/C7SC02664A",
    },
}


def endpoint_spec(name: str) -> dict[str, Any]:
    key = name.lower().strip()
    if key not in ADMET_DATASETS:
        raise ValueError(f"Unknown ADMET dataset {name!r}; choose from {sorted(ADMET_DATASETS)}")
    return ADMET_DATASETS[key]


def fetch_endpoint_dataset(name: str, out_path: Path) -> dict[str, Any]:
    """Download a public benchmark file without silently transforming its contents."""
    spec = endpoint_spec(name)
    response = requests.get(spec["url"], timeout=120)
    response.raise_for_status()
    content = response.content
    if spec["url"].endswith(".gz") and out_path.suffix != ".gz":
        content = gzip.decompress(content)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content)
    manifest = {
        "dataset": name.lower(),
        "url": spec["url"],
        "benchmark": spec["benchmark"],
        "citation": spec["citation"],
        "sha256": sha256_file(out_path),
        "bytes": out_path.stat().st_size,
    }
    manifest_path = out_path.with_name(out_path.name + ".source.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_endpoint_dataset(name: str, raw: pd.DataFrame) -> pd.DataFrame:
    spec = endpoint_spec(name)
    smiles_col = spec["smiles_column"]
    target_col = spec["target_column"]
    missing = [c for c in (smiles_col, target_col) if c not in raw.columns]
    if missing:
        raise ValueError(f"{name} dataset missing required columns: {missing}")

    frame = raw[[smiles_col, target_col]].copy()
    frame.columns = ["smiles", "label"]
    frame["smiles"] = frame["smiles"].map(canonical_smiles)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["smiles", "label"]).reset_index(drop=True)

    if spec["task"] == "classification":
        frame = frame[frame["label"].isin([0, 1])].copy()
        grouped = frame.groupby("smiles", as_index=False).agg(
            label=("label", "mean"), duplicate_count=("label", "size")
        )
        # Conflicting duplicate labels are not converted into false certainty.
        grouped = grouped[grouped["label"].isin([0.0, 1.0])].copy()
        grouped["label"] = grouped["label"].astype(int)
    else:
        grouped = frame.groupby("smiles", as_index=False).agg(
            label=("label", "mean"), duplicate_count=("label", "size")
        )
        grouped["label"] = grouped["label"].astype(float)

    grouped = grouped.reset_index(drop=True)
    grouped.insert(0, "molecule_id", [f"{name.upper()}_{i:05d}" for i in range(len(grouped))])
    grouped["scaffold"] = grouped["smiles"].map(scaffold_smiles)
    return grouped


def _balanced_split(df: pd.DataFrame, task: str, seed: int) -> tuple[pd.Series, int]:
    if task != "classification":
        return scaffold_split(df, SplitConfig(seed=seed)), seed
    for candidate_seed in range(seed, seed + 50):
        split = scaffold_split(df, SplitConfig(seed=candidate_seed))
        ok = True
        for name in ("train", "validation", "test"):
            labels = df.loc[split == name, "label"]
            if len(labels) == 0 or labels.nunique() < 2:
                ok = False
                break
        if ok:
            return split, candidate_seed
    raise ValueError("Could not create class-balanced scaffold train/validation/test split in 50 seeds")


def _classification_metrics(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(prob, dtype=float), 1e-8, 1 - 1e-8)
    pred = (clipped >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, clipped)),
        "roc_auc": float(roc_auc_score(y_true, clipped)),
        "brier": float(brier_score_loss(y_true, clipped)),
        "ece_10bin": float(_expected_calibration_error(y_true.to_numpy(), clipped, bins=10)),
    }


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not mask.any():
            continue
        error += float(mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return error


def _regression_metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    mse = mean_squared_error(y_true, pred)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, pred)),
    }


def _classification_specs(seed: int) -> dict[str, tuple[object, str]]:
    return {
        "descriptor_logreg": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]),
            "descriptors",
        ),
        "descriptor_rf": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=180,
                    random_state=seed,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    min_samples_leaf=2,
                )),
            ]),
            "descriptors",
        ),
        "morgan_logreg": (
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
            "morgan",
        ),
    }


def _regression_specs(seed: int) -> dict[str, tuple[object, str]]:
    return {
        "descriptor_ridge": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=3.0)),
            ]),
            "descriptors",
        ),
        "descriptor_rf": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=180,
                    random_state=seed,
                    n_jobs=-1,
                    min_samples_leaf=2,
                )),
            ]),
            "descriptors",
        ),
        "morgan_ridge": (Ridge(alpha=8.0), "morgan"),
    }


def _platt_fit(prob: np.ndarray, y: pd.Series) -> LogisticRegression:
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(logits, y.astype(int))
    return calibrator


def _platt_apply(calibrator: LogisticRegression | None, prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    if calibrator is None:
        return p
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def train_endpoint_model(
    name: str,
    raw: pd.DataFrame,
    out_dir: Path,
    descriptor_count: int = 96,
    seed: int = 42,
    source_path: Path | None = None,
) -> dict[str, Any]:
    spec = endpoint_spec(name)
    prepared = prepare_endpoint_dataset(name, raw)
    if len(prepared) < 30:
        raise ValueError(f"Need at least 30 cleaned molecules for {name}; got {len(prepared)}")
    prepared["split"], actual_seed = _balanced_split(prepared, spec["task"], seed)

    desc, desc_names = descriptor_frame(prepared, descriptor_count)
    fp = morgan_frame(prepared)
    reps = {"descriptors": desc, "morgan": fp}
    train_mask = prepared["split"] == "train"
    val_mask = prepared["split"] == "validation"
    test_mask = prepared["split"] == "test"
    y = prepared["label"]

    if spec["task"] == "classification":
        specs = _classification_specs(actual_seed)
        validation: dict[str, dict[str, float]] = {}
        trained: dict[str, object] = {}
        for model_name, (prototype, representation) in specs.items():
            model = clone(prototype)
            model.fit(reps[representation].loc[train_mask], y.loc[train_mask].astype(int))
            prob = model.predict_proba(reps[representation].loc[val_mask])[:, 1]
            validation[model_name] = _classification_metrics(y.loc[val_mask].astype(int), prob)
            trained[model_name] = model
        best_name = max(validation, key=lambda n: (validation[n]["pr_auc"], validation[n]["roc_auc"]))
        representation = specs[best_name][1]
        best_model = trained[best_name]
        raw_val_prob = best_model.predict_proba(reps[representation].loc[val_mask])[:, 1]
        calibrator = _platt_fit(raw_val_prob, y.loc[val_mask].astype(int))
        raw_test_prob = best_model.predict_proba(reps[representation].loc[test_mask])[:, 1]
        test_prob = _platt_apply(calibrator, raw_test_prob)
        test_metrics = _classification_metrics(y.loc[test_mask].astype(int), test_prob)
        test_metrics["uncalibrated_brier"] = float(
            brier_score_loss(y.loc[test_mask].astype(int), np.clip(raw_test_prob, 1e-8, 1 - 1e-8))
        )
        uncertainty_contract = {
            "calibration": "Platt scaling fitted on validation scaffolds only",
            "predictive_uncertainty": "normalized binary entropy",
            "applicability_domain": "maximum Morgan/Tanimoto similarity to training chemistry",
        }
        interval_half_width = None
    else:
        specs = _regression_specs(actual_seed)
        validation = {}
        trained = {}
        for model_name, (prototype, representation) in specs.items():
            model = clone(prototype)
            model.fit(reps[representation].loc[train_mask], y.loc[train_mask].astype(float))
            pred = model.predict(reps[representation].loc[val_mask])
            validation[model_name] = _regression_metrics(y.loc[val_mask].astype(float), pred)
            trained[model_name] = model
        best_name = min(validation, key=lambda n: validation[n]["rmse"])
        representation = specs[best_name][1]
        best_model = trained[best_name]
        val_pred = best_model.predict(reps[representation].loc[val_mask])
        residuals = np.abs(y.loc[val_mask].to_numpy(dtype=float) - np.asarray(val_pred, dtype=float))
        interval_half_width = float(np.quantile(residuals, 0.90, method="higher"))
        test_pred = best_model.predict(reps[representation].loc[test_mask])
        test_metrics = _regression_metrics(y.loc[test_mask].astype(float), test_pred)
        lower = np.asarray(test_pred) - interval_half_width
        upper = np.asarray(test_pred) + interval_half_width
        yt = y.loc[test_mask].to_numpy(dtype=float)
        test_metrics["pi90_empirical_coverage"] = float(np.mean((yt >= lower) & (yt <= upper)))
        calibrator = None
        uncertainty_contract = {
            "prediction_interval": "symmetric 90th-percentile absolute residual interval calibrated on validation scaffolds",
            "applicability_domain": "maximum Morgan/Tanimoto similarity to training chemistry",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = out_dir / "endpoint_prepared.csv"
    metrics_path = out_dir / "metrics.json"
    model_path = out_dir / "model.joblib"
    prepared.to_csv(prepared_path, index=False)

    bundle = {
        "schema_version": 1,
        "pdl_version": "0.4.0",
        "endpoint": name.lower(),
        "task": spec["task"],
        "label_semantics": spec["label_semantics"],
        "model_name": best_name,
        "representation": representation,
        "model": best_model,
        "calibrator": calibrator,
        "descriptor_names": desc_names,
        "train_smiles": prepared.loc[train_mask, "smiles"].tolist(),
        "pi90_half_width": interval_half_width,
        "split_seed": actual_seed,
        "uncertainty_contract": uncertainty_contract,
    }
    joblib.dump(bundle, model_path)

    metrics = {
        "endpoint": name.lower(),
        "task": spec["task"],
        "best_model": best_name,
        "selection_split": "validation",
        "selection_rule": "max validation PR-AUC (ROC-AUC tie-break)"
        if spec["task"] == "classification"
        else "min validation RMSE",
        "split_seed_requested": seed,
        "split_seed_used": actual_seed,
        "split_counts": prepared["split"].value_counts().to_dict(),
        "validation": validation,
        "test": test_metrics,
        "uncertainty_contract": uncertainty_contract,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    artifact_names = ["endpoint_prepared.csv", "metrics.json", "model.joblib"]
    manifest = {
        "manifest_type": "pdl_admet_endpoint",
        "schema_version": 1,
        "version": "0.4.0",
        "endpoint": name.lower(),
        "dataset": spec,
        "molecules": len(prepared),
        "scaffolds": int(prepared["scaffold"].nunique()),
        "split_seed": actual_seed,
        "descriptor_count": descriptor_count,
        "model": best_name,
        "source": {
            "path": str(source_path) if source_path else None,
            "sha256": sha256_file(source_path) if source_path and source_path.exists() else None,
        },
        "environment": environment_fingerprint(),
        "artifacts": artifact_names,
        "artifact_sha256": artifact_hashes(out_dir, artifact_names),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {"metrics": metrics, "manifest": manifest, "model_path": str(model_path)}


def _representation_frame(bundle: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    if bundle["representation"] == "descriptors":
        names = list(bundle["descriptor_names"])
        frame, _ = descriptor_frame(df, len(names))
        return frame[names]
    if bundle["representation"] == "morgan":
        return morgan_frame(df)
    raise ValueError(f"Unknown representation {bundle['representation']!r}")


def _applicability_scores(query_smiles: list[str], train_smiles: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    train_fps = []
    for smiles in train_smiles:
        mol = mol_from_smiles(smiles)
        if mol is not None:
            train_fps.append(generator.GetFingerprint(mol))
    if not train_fps:
        return np.zeros(len(query_smiles), dtype=float)
    scores = []
    for smiles in query_smiles:
        mol = mol_from_smiles(smiles)
        if mol is None:
            scores.append(0.0)
            continue
        fp = generator.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        scores.append(float(max(sims)) if sims else 0.0)
    return np.asarray(scores, dtype=float)


def _ad_band(similarity: float) -> str:
    # Descriptive heuristic only; not a universal medicinal-chemistry threshold.
    if similarity >= 0.70:
        return "high_similarity"
    if similarity >= 0.40:
        return "moderate_similarity"
    return "low_similarity"


def _binary_entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), 1e-8, 1 - 1e-8)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def predict_endpoint(bundle: dict[str, Any], candidates: pd.DataFrame) -> pd.DataFrame:
    if "smiles" not in candidates.columns:
        raise ValueError("Candidate data must contain a smiles column")
    clean = candidates[["smiles"]].copy()
    clean["smiles"] = clean["smiles"].map(canonical_smiles)
    if clean["smiles"].isna().any():
        raise ValueError("Candidate data contains invalid SMILES")
    X = _representation_frame(bundle, clean)
    ad = _applicability_scores(clean["smiles"].tolist(), list(bundle["train_smiles"]))
    endpoint = bundle["endpoint"]

    if bundle["task"] == "classification":
        raw = bundle["model"].predict_proba(X)[:, 1]
        prob = _platt_apply(bundle.get("calibrator"), raw)
        result = pd.DataFrame({
            f"{endpoint}_probability": prob,
            f"{endpoint}_predictive_entropy": _binary_entropy(prob),
            f"{endpoint}_ad_similarity": ad,
            f"{endpoint}_ad_band": [_ad_band(x) for x in ad],
        }, index=candidates.index)
    else:
        pred = np.asarray(bundle["model"].predict(X), dtype=float)
        half = float(bundle.get("pi90_half_width") or 0.0)
        result = pd.DataFrame({
            f"{endpoint}_prediction": pred,
            f"{endpoint}_pi90_low": pred - half,
            f"{endpoint}_pi90_high": pred + half,
            f"{endpoint}_ad_similarity": ad,
            f"{endpoint}_ad_band": [_ad_band(x) for x in ad],
        }, index=candidates.index)
    return result


def annotate_candidates(candidates: pd.DataFrame, models_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = candidates.copy()
    used: list[str] = []
    for endpoint in ADMET_DATASETS:
        model_path = models_dir / endpoint / "model.joblib"
        if not model_path.exists():
            continue
        bundle = joblib.load(model_path)
        pred = predict_endpoint(bundle, out)
        out = pd.concat([out, pred], axis=1)
        used.append(endpoint)
    if not used:
        raise FileNotFoundError(
            f"No endpoint models found under {models_dir}; expected e.g. {models_dir / 'bbbp' / 'model.joblib'}"
        )
    ad_cols = [c for c in out.columns if c.endswith("_ad_band")]
    out["admet_low_similarity_endpoints"] = (out[ad_cols] == "low_similarity").sum(axis=1)
    summary = {
        "models_used": used,
        "molecules": len(out),
        "low_similarity_counts": {c: int((out[c] == "low_similarity").sum()) for c in ad_cols},
        "policy": "Endpoint predictions remain separate; V0.4 does not collapse them into a hidden composite drug score.",
    }
    return out, summary
