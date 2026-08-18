from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .chemistry import descriptor_frame, morgan_frame


def _metrics(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, prob)) if y_true.nunique() == 2 else float("nan")
    return out


def _model_specs():
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
                    n_estimators=350, random_state=42, class_weight="balanced_subsample", n_jobs=-1,
                )),
            ]),
            "descriptors",
        ),
        "descriptor_hgb": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=42)),
            ]),
            "descriptors",
        ),
        "morgan_logreg": (
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
            "morgan",
        ),
    }


def train_benchmarks(df: pd.DataFrame, out_dir: Path, descriptor_count: int = 96) -> tuple[dict, dict]:
    """Select model family on validation scaffolds; evaluate refit models on untouched test scaffolds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "validation"
    test_mask = df["split"] == "test"
    for split_name, mask in [("train", train_mask), ("validation", val_mask), ("test", test_mask)]:
        if mask.sum() == 0:
            raise ValueError(f"{split_name} scaffold split is empty")
        if df.loc[mask, "active_label"].nunique() < 2:
            raise ValueError(f"{split_name} scaffold split contains only one class")

    desc, desc_names = descriptor_frame(df, descriptor_count)
    fp = morgan_frame(df)
    representations = {"descriptors": desc, "morgan": fp}
    y = df["active_label"].astype(int)

    validation_metrics: dict[str, dict] = {}
    specs = _model_specs()
    for name, (prototype, representation) in specs.items():
        X = representations[representation]
        model = clone(prototype)
        model.fit(X.loc[train_mask], y.loc[train_mask])
        validation_metrics[name] = _metrics(y.loc[val_mask], model.predict_proba(X.loc[val_mask])[:, 1])

    best_name = max(
        validation_metrics,
        key=lambda n: (validation_metrics[n]["pr_auc"], validation_metrics[n].get("roc_auc", 0)),
    )

    fit_mask = train_mask | val_mask
    test_metrics: dict[str, dict] = {}
    final_models: dict[str, object] = {}
    for name, (prototype, representation) in specs.items():
        X = representations[representation]
        model = clone(prototype)
        model.fit(X.loc[fit_mask], y.loc[fit_mask])
        test_metrics[name] = _metrics(y.loc[test_mask], model.predict_proba(X.loc[test_mask])[:, 1])
        final_models[name] = model

    representation = specs[best_name][1]
    bundle = {
        "name": best_name,
        "representation": representation,
        "model": final_models[best_name],
        "descriptor_names": desc_names,
        "selection_rule": "highest validation PR-AUC, tie-break validation ROC-AUC",
    }
    joblib.dump(bundle, out_dir / "best_model.joblib")
    payload = {
        "best_model": best_name,
        "selection_split": "validation",
        "selection_rule": bundle["selection_rule"],
        "validation": validation_metrics,
        "test": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, bundle


def predict_bundle(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    if bundle["representation"] == "descriptors":
        X, _ = descriptor_frame(df, len(bundle["descriptor_names"]))
        X = X[bundle["descriptor_names"]]
    elif bundle["representation"] == "morgan":
        X = morgan_frame(df)
    else:
        raise ValueError(f"Unknown representation: {bundle['representation']}")
    return bundle["model"].predict_proba(X)[:, 1]
