from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from parkinson_discovery.demo_data import make_demo_dataset
from parkinson_discovery.medchem import annotate_medchem, medicinal_properties, structural_alerts
from parkinson_discovery.models import train_benchmarks
from parkinson_discovery.pipeline import prepare_dataset
from parkinson_discovery.selectivity import annotate_selectivity, resolve_human_single_protein_target


def test_medchem_properties_and_catalogs_return_evidence():
    props = medicinal_properties("CCO")
    assert 0 <= props["qed"] <= 1
    assert 0 <= props["synthetic_complexity_proxy"] <= 1
    alerts = structural_alerts("CCO")
    assert alerts["invalid_smiles"] is False
    assert alerts["structural_alert_count"] >= 0
    assert isinstance(alerts["structural_alerts"], str)


def test_nearest_known_chemistry_is_identity_aware():
    demo = make_demo_dataset()
    candidates = demo.head(5)[["molecule_id", "smiles"]]
    annotated, summary = annotate_medchem(candidates, reference=demo)
    assert summary["with_reference"] is True
    assert annotated["nearest_known_similarity"].eq(1.0).all()
    assert annotated["novelty_proxy"].eq(0.0).all()
    assert "medchem_review_flags" in annotated.columns


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeSession:
    def get(self, url, params=None, timeout=None):
        assert "target/search.json" in url
        assert params["q"] == "TTK"
        return _FakeResponse({
            "targets": [
                {
                    "target_chembl_id": "CHEMBL_FAKE_TTK",
                    "pref_name": "Dual specificity protein kinase TTK",
                    "organism": "Homo sapiens",
                    "target_type": "SINGLE PROTEIN",
                    "target_synonyms": [{"component_synonym": "TTK"}, {"component_synonym": "MPS1"}],
                },
                {
                    "target_chembl_id": "CHEMBL_MOUSE",
                    "pref_name": "Ttk",
                    "organism": "Mus musculus",
                    "target_type": "SINGLE PROTEIN",
                },
            ]
        })


def test_selectivity_target_resolver_requires_human_single_protein_exact_alias():
    resolved = resolve_human_single_protein_target("TTK", session=_FakeSession())
    assert resolved["target_chembl_id"] == "CHEMBL_FAKE_TTK"
    assert resolved["organism"] == "Homo sapiens"


def test_selectivity_annotation_keeps_individual_offtarget_models(tmp_path: Path):
    demo = make_demo_dataset()
    prepared = prepare_dataset(demo)
    target_dir = tmp_path / "models" / "ttk"
    metrics, bundle = train_benchmarks(prepared, target_dir, descriptor_count=24)
    assert metrics["best_model"] == bundle["name"]
    (target_dir / "selectivity_model.json").write_text(
        json.dumps({"target": "TTK", "best_model": bundle["name"]}), encoding="utf-8"
    )
    # train_benchmarks already wrote best_model.joblib; assert it really is loadable.
    assert joblib.load(target_dir / "best_model.joblib")["name"] == bundle["name"]

    candidates = demo.head(10)[["molecule_id", "smiles"]].copy()
    candidates["predicted_activity"] = 0.8
    annotated, summary = annotate_selectivity(candidates, tmp_path / "models")
    assert summary["models_used"] == ["TTK"]
    assert annotated["offtarget_ttk_probability"].between(0, 1).all()
    assert annotated["max_offtarget_probability"].between(0, 1).all()
    assert annotated["highest_predicted_offtarget"].eq("TTK").all()
    assert "lrrk2_vs_max_offtarget_margin_proxy" in annotated.columns
