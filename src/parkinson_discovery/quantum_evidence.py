from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .models import predict_bundle
from .provenance import canonical_json_sha256, sha256_file
from .quantum import _metric_dict, _rimay_model_specs


PRIMARY_METRIC = "pr_auc"


def _safe_roc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _metric_functions() -> dict[str, tuple[Callable[[np.ndarray, np.ndarray], float], str]]:
    return {
        "pr_auc": (lambda y, p: float(average_precision_score(y, p)), "quantum_minus_classical"),
        "roc_auc": (_safe_roc, "quantum_minus_classical"),
        "brier": (lambda y, p: float(brier_score_loss(y, p)), "classical_minus_quantum"),
    }


def paired_bootstrap_delta(
    y: np.ndarray,
    classical_prob: np.ndarray,
    quantum_prob: np.ndarray,
    metric: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    if metric not in _metric_functions():
        raise ValueError(f"Unsupported bootstrap metric {metric!r}")
    fn, direction = _metric_functions()[metric]
    y = np.asarray(y, dtype=int)
    classical_prob = np.asarray(classical_prob, dtype=float)
    quantum_prob = np.asarray(quantum_prob, dtype=float)
    if not (len(y) == len(classical_prob) == len(quantum_prob)):
        raise ValueError("Paired bootstrap arrays must have equal lengths")
    if len(y) < 2:
        raise ValueError("Paired bootstrap requires at least two test molecules")

    classical_point = fn(y, classical_prob)
    quantum_point = fn(y, quantum_prob)
    if direction == "quantum_minus_classical":
        point_delta = quantum_point - classical_point
    else:
        point_delta = classical_point - quantum_point

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        c = classical_prob[idx]
        q = quantum_prob[idx]
        cv = fn(ys, c)
        qv = fn(ys, q)
        if not (np.isfinite(cv) and np.isfinite(qv)):
            continue
        delta = qv - cv if direction == "quantum_minus_classical" else cv - qv
        deltas.append(float(delta))
    if not deltas:
        raise ValueError(f"No valid bootstrap replicates for {metric}")
    arr = np.asarray(deltas, dtype=float)
    return {
        "metric": metric,
        "direction": direction,
        "classical": float(classical_point),
        "quantum": float(quantum_point),
        "delta": float(point_delta),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
        "probability_improvement": float(np.mean(arr > 0)),
        "bootstrap_replicates_requested": int(n_bootstrap),
        "bootstrap_replicates_valid": int(len(arr)),
    }


def _align_rimay(prepared: pd.DataFrame, rimay: pd.DataFrame) -> pd.DataFrame:
    if "molecule_id" not in rimay.columns:
        raise ValueError("Rimay result must contain molecule_id")
    if rimay["molecule_id"].duplicated().any():
        raise ValueError("Rimay result contains duplicate molecule_id rows")
    base = prepared[["molecule_id", "active_label", "split"]].copy()
    merged = base.merge(rimay, on="molecule_id", how="inner", validate="one_to_one")
    if len(merged) != len(prepared):
        raise ValueError(f"Rimay result missing {len(prepared) - len(merged)} frozen molecules")

    if "active_label_x" in merged.columns:
        frozen = merged["active_label_x"].astype(int)
        returned = merged.get("active_label_y")
        if returned is not None and not np.array_equal(frozen.to_numpy(), returned.astype(int).to_numpy()):
            raise ValueError("Rimay-returned labels disagree with frozen labels")
        merged = merged.rename(columns={"active_label_x": "active_label"}).drop(
            columns=[c for c in ["active_label_y"] if c in merged.columns]
        )
    if "split_x" in merged.columns:
        frozen_split = merged["split_x"].astype(str)
        returned_split = merged.get("split_y")
        if returned_split is not None and not np.array_equal(
            frozen_split.to_numpy(), returned_split.astype(str).to_numpy()
        ):
            raise ValueError("Rimay-returned splits disagree with frozen scaffold split")
        merged = merged.rename(columns={"split_x": "split"}).drop(
            columns=[c for c in ["split_y"] if c in merged.columns]
        )
    return merged


def _quantum_probabilities(merged: pd.DataFrame) -> tuple[np.ndarray, str, dict[str, Any] | None]:
    train_mask = merged["split"] == "train"
    val_mask = merged["split"] == "validation"
    test_mask = merged["split"] == "test"
    y = merged["active_label"].astype(int)

    if "prediction" in merged.columns:
        prob = pd.to_numeric(merged.loc[test_mask, "prediction"], errors="raise").to_numpy(dtype=float)
        if np.any((prob < 0) | (prob > 1)):
            raise ValueError("Rimay prediction must be a probability in [0, 1]")
        return prob, "returned_prediction", None

    excluded = {
        "molecule_id", "active_label", "split", "smiles", "scaffold", "pchembl_value",
    }
    feature_cols = [
        c for c in merged.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(merged[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric Rimay feature columns found")
    X = merged[feature_cols]
    specs = _rimay_model_specs()
    validation: dict[str, dict[str, float]] = {}
    for name, prototype in specs.items():
        model = clone(prototype)
        model.fit(X.loc[train_mask], y.loc[train_mask])
        validation[name] = _metric_dict(
            y.loc[val_mask], model.predict_proba(X.loc[val_mask])[:, 1]
        )
    selected = max(validation, key=lambda n: (validation[n]["pr_auc"], validation[n]["roc_auc"]))
    model = clone(specs[selected])
    fit_mask = train_mask | val_mask
    model.fit(X.loc[fit_mask], y.loc[fit_mask])
    prob = model.predict_proba(X.loc[test_mask])[:, 1]
    return np.asarray(prob, dtype=float), selected, validation


def _trial_verdict(intervals: dict[str, dict[str, Any]]) -> str:
    primary = intervals[PRIMARY_METRIC]
    roc = intervals["roc_auc"]
    if primary["ci95_low"] > 0 and roc["delta"] >= 0:
        return "pass"
    if primary["ci95_high"] <= 0:
        return "fail"
    return "inconclusive"


def run_quantum_trial(
    prepared_csv: Path,
    classical_model_joblib: Path,
    rimay_result_csv: Path,
    out_dir: Path,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    backend_type: str = "unknown",
    backend_name: str | None = None,
    quantum_runtime_seconds: float | None = None,
    quantum_cost_eur: float | None = None,
    provider: str = "Kipu Quantum",
) -> dict[str, Any]:
    prepared = pd.read_csv(prepared_csv)
    rimay = pd.read_csv(rimay_result_csv)
    merged = _align_rimay(prepared, rimay)
    test_mask = merged["split"] == "test"
    if not test_mask.any():
        raise ValueError("Frozen experiment has no test molecules")

    bundle = joblib.load(classical_model_joblib)
    prepared_by_id = prepared.set_index("molecule_id", drop=False)
    ordered_test_ids = merged.loc[test_mask, "molecule_id"].tolist()
    classical_test = prepared_by_id.loc[ordered_test_ids].reset_index(drop=True)
    classical_prob = np.asarray(predict_bundle(bundle, classical_test), dtype=float)
    quantum_prob, selected_quantum_model, quantum_validation = _quantum_probabilities(merged)
    y = merged.loc[test_mask, "active_label"].to_numpy(dtype=int)

    intervals = {
        metric: paired_bootstrap_delta(
            y, classical_prob, quantum_prob, metric, n_bootstrap=n_bootstrap, seed=seed + i * 1009
        )
        for i, metric in enumerate(("pr_auc", "roc_auc", "brier"))
    }
    verdict = _trial_verdict(intervals)

    evidence = merged.loc[test_mask, ["molecule_id", "active_label"]].copy().reset_index(drop=True)
    evidence["classical_probability"] = classical_prob
    evidence["quantum_probability"] = quantum_prob
    evidence["quantum_minus_classical_probability"] = quantum_prob - classical_prob
    evidence["absolute_probability_disagreement"] = np.abs(quantum_prob - classical_prob)
    evidence["classical_correct"] = ((classical_prob >= 0.5).astype(int) == y)
    evidence["quantum_correct"] = ((quantum_prob >= 0.5).astype(int) == y)
    evidence["disagreement_outcome"] = np.select(
        [
            evidence["quantum_correct"] & ~evidence["classical_correct"],
            evidence["classical_correct"] & ~evidence["quantum_correct"],
            evidence["classical_correct"] & evidence["quantum_correct"],
        ],
        ["quantum_only_correct", "classical_only_correct", "both_correct"],
        default="both_wrong",
    )
    evidence = evidence.sort_values("absolute_probability_disagreement", ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "paired_test_predictions.csv"
    evidence.to_csv(predictions_path, index=False)
    trial_seed = {
        "prepared_sha256": sha256_file(prepared_csv),
        "classical_model_sha256": sha256_file(classical_model_joblib),
        "rimay_result_sha256": sha256_file(rimay_result_csv),
        "bootstrap_seed": seed,
    }
    trial_id = f"qtrial-{canonical_json_sha256(trial_seed)[:12]}"
    payload = {
        "manifest_type": "pdl_quantum_trial",
        "schema_version": 1,
        "version": "0.6.0",
        "trial_id": trial_id,
        "test_molecules": int(len(evidence)),
        "classical_model": bundle.get("name"),
        "quantum_mode": "prediction" if selected_quantum_model == "returned_prediction" else "features",
        "selected_quantum_model": selected_quantum_model,
        "quantum_validation": quantum_validation,
        "paired_bootstrap": intervals,
        "verdict": verdict,
        "decision_rule": (
            "pass if PR-AUC paired-bootstrap 95% CI is entirely >0 and ROC-AUC point delta is non-negative; "
            "fail if PR-AUC CI is entirely <=0; otherwise inconclusive"
        ),
        "compute": {
            "provider": provider,
            "backend_type": backend_type,
            "backend_name": backend_name,
            "quantum_runtime_seconds": quantum_runtime_seconds,
            "quantum_cost_eur": quantum_cost_eur,
        },
        "artifacts": {
            "prepared_sha256": sha256_file(prepared_csv),
            "classical_model_sha256": sha256_file(classical_model_joblib),
            "rimay_result_sha256": sha256_file(rimay_result_csv),
            "paired_test_predictions": predictions_path.name,
            "paired_test_predictions_sha256": sha256_file(predictions_path),
        },
        "claim_boundary": (
            "This is one frozen scaffold trial. A V0.6 project-level quantum-value decision requires repeated "
            "independent scaffold trials and must report simulator/QPU status, runtime and cost."
        ),
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    (out_dir / "trial.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _bootstrap_trial_mean(values: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        means[i] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "probability_mean_improvement": float(np.mean(means > 0)),
    }


def summarize_quantum_trials(
    trial_jsons: Iterable[Path],
    out_path: Path,
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    trials = [json.loads(Path(path).read_text(encoding="utf-8")) for path in trial_jsons]
    if not trials:
        raise ValueError("No quantum trial manifests supplied")
    deltas = np.asarray(
        [t["paired_bootstrap"][PRIMARY_METRIC]["delta"] for t in trials], dtype=float
    )
    if len(trials) >= 3:
        across = _bootstrap_trial_mean(deltas, n_bootstrap, seed)
        if across["ci95_low"] > 0:
            verdict = "pass"
        elif across["ci95_high"] <= 0:
            verdict = "fail"
        else:
            verdict = "inconclusive"
    else:
        across = {
            "mean": float(np.mean(deltas)),
            "ci95_low": None,
            "ci95_high": None,
            "probability_mean_improvement": None,
        }
        verdict = "inconclusive_insufficient_repeated_trials"

    runtimes = [
        t.get("compute", {}).get("quantum_runtime_seconds") for t in trials
        if t.get("compute", {}).get("quantum_runtime_seconds") is not None
    ]
    costs = [
        t.get("compute", {}).get("quantum_cost_eur") for t in trials
        if t.get("compute", {}).get("quantum_cost_eur") is not None
    ]
    payload = {
        "manifest_type": "pdl_quantum_meta_benchmark",
        "schema_version": 1,
        "version": "0.6.0",
        "primary_metric": PRIMARY_METRIC,
        "trial_count": len(trials),
        "trial_ids": [t.get("trial_id") for t in trials],
        "trial_verdicts": [t.get("verdict") for t in trials],
        "primary_metric_delta_by_trial": deltas.tolist(),
        "across_trial_bootstrap": across,
        "verdict": verdict,
        "compute": {
            "backend_types": sorted(set(str(t.get("compute", {}).get("backend_type")) for t in trials)),
            "total_reported_quantum_runtime_seconds": float(sum(runtimes)) if runtimes else None,
            "total_reported_quantum_cost_eur": float(sum(costs)) if costs else None,
        },
        "decision_rule": (
            "At least three frozen scaffold trials required. Bootstrap the mean trial-level PR-AUC delta; "
            "pass if 95% CI >0, fail if 95% CI <=0, otherwise inconclusive."
        ),
        "claim_boundary": (
            "A statistical performance gain is not automatically a quantum-computational advantage. "
            "Backend, runtime, cost, reproducibility and classical simulation alternatives must also be reported."
        ),
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
