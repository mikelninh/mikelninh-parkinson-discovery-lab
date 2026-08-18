from __future__ import annotations

import joblib
import pandas as pd

from parkinson_discovery.admet import (
    ADMET_DATASETS,
    annotate_candidates,
    predict_endpoint,
    prepare_endpoint_dataset,
    train_endpoint_model,
)
from parkinson_discovery.demo_data import make_demo_dataset


def test_admet_registry_has_v04_endpoints():
    assert set(ADMET_DATASETS) == {"bbbp", "clintox", "esol"}
    assert ADMET_DATASETS["bbbp"]["task"] == "classification"
    assert ADMET_DATASETS["esol"]["task"] == "regression"


def test_conflicting_classification_duplicates_are_removed():
    raw = pd.DataFrame({
        "smiles": ["CCO", "CCO", "CCN", "CCN"],
        "p_np": [0, 1, 1, 1],
    })
    clean = prepare_endpoint_dataset("bbbp", raw)
    assert len(clean) == 1
    assert clean.iloc[0]["smiles"] == "CCN"
    assert clean.iloc[0]["label"] == 1
    assert clean.iloc[0]["duplicate_count"] == 2


def test_bbbp_model_calibrates_and_exposes_applicability_domain(tmp_path):
    demo = make_demo_dataset()
    raw = pd.DataFrame({"smiles": demo["smiles"], "p_np": demo["active_label"]})
    result = train_endpoint_model("bbbp", raw, tmp_path / "bbbp", descriptor_count=24, seed=42)
    assert result["metrics"]["task"] == "classification"
    assert (tmp_path / "bbbp" / "manifest.json").exists()
    bundle = joblib.load(tmp_path / "bbbp" / "model.joblib")
    pred = predict_endpoint(bundle, demo.head(12)[["smiles"]])
    assert pred["bbbp_probability"].between(0, 1).all()
    assert pred["bbbp_predictive_entropy"].between(0, 1).all()
    assert pred["bbbp_ad_similarity"].between(0, 1).all()
    assert set(pred["bbbp_ad_band"]).issubset(
        {"high_similarity", "moderate_similarity", "low_similarity"}
    )


def test_esol_model_emits_validation_calibrated_prediction_interval(tmp_path):
    demo = make_demo_dataset()
    raw = pd.DataFrame({
        "smiles": demo["smiles"],
        "measured log solubility in mols per litre": -demo["pchembl_value"] / 2.0,
    })
    result = train_endpoint_model("esol", raw, tmp_path / "esol", descriptor_count=24, seed=42)
    assert result["metrics"]["task"] == "regression"
    assert "pi90_empirical_coverage" in result["metrics"]["test"]
    bundle = joblib.load(tmp_path / "esol" / "model.joblib")
    pred = predict_endpoint(bundle, demo.head(12)[["smiles"]])
    assert (pred["esol_pi90_low"] <= pred["esol_prediction"]).all()
    assert (pred["esol_prediction"] <= pred["esol_pi90_high"]).all()
    assert pred["esol_ad_similarity"].between(0, 1).all()


def test_candidate_annotation_keeps_endpoints_separate(tmp_path):
    demo = make_demo_dataset()
    bbbp_raw = pd.DataFrame({"smiles": demo["smiles"], "p_np": demo["active_label"]})
    esol_raw = pd.DataFrame({
        "smiles": demo["smiles"],
        "measured log solubility in mols per litre": -demo["pchembl_value"] / 2.0,
    })
    train_endpoint_model("bbbp", bbbp_raw, tmp_path / "models" / "bbbp", descriptor_count=16)
    train_endpoint_model("esol", esol_raw, tmp_path / "models" / "esol", descriptor_count=16)

    candidates = demo.head(8)[["molecule_id", "smiles"]].copy()
    annotated, summary = annotate_candidates(candidates, tmp_path / "models")
    assert summary["models_used"] == ["bbbp", "esol"]
    assert "bbbp_probability" in annotated.columns
    assert "esol_prediction" in annotated.columns
    assert "admet_low_similarity_endpoints" in annotated.columns
    assert "rank_score" not in annotated.columns
