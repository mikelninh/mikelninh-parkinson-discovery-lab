from pathlib import Path


from parkinson_discovery.chembl import build_target_dataset, freeze_target_snapshot
from parkinson_discovery.demo_data import make_demo_dataset
from parkinson_discovery.pipeline import run_pipeline
from parkinson_discovery.provenance import verify_manifest


class SnapshotFakeClient:
    def activities(self, target_id):
        return [
            {
                "activity_id": 101,
                "molecule_chembl_id": "CHEMBL1",
                "canonical_smiles": "c1ccccc1F",
                "pchembl_value": "6.4",
                "standard_type": "IC50",
                "standard_relation": "=",
                "assay_chembl_id": "A1",
                "document_chembl_id": "D1",
            },
            {
                "activity_id": 102,
                "molecule_chembl_id": "CHEMBL1",
                "canonical_smiles": "Fc1ccccc1",
                "pchembl_value": "6.2",
                "standard_type": "Ki",
                "standard_relation": "=",
                "assay_chembl_id": "A2",
                "document_chembl_id": "D2",
            },
            {
                "activity_id": 103,
                "molecule_chembl_id": "CHEMBL2",
                "canonical_smiles": "c1ccccc1Cl",
                "pchembl_value": "4.6",
                "standard_type": "IC50",
                "standard_relation": "=",
                "assay_chembl_id": "A3",
                "document_chembl_id": "D3",
            },
        ]

    def molecule_smiles(self, ids):
        return {}

    def assays(self, ids):
        return {
            "A1": {"assay_chembl_id": "A1", "assay_type": "B", "assay_organism": "Homo sapiens", "confidence_score": 9},
            "A2": {"assay_chembl_id": "A2", "assay_type": "B", "assay_organism": "Homo sapiens", "confidence_score": 9},
            "A3": {"assay_chembl_id": "A3", "assay_type": "B", "assay_organism": "Homo sapiens", "confidence_score": 9},
        }

    def release_info(self):
        return {"chembl_release_id": 37, "chembl_version": "ChEMBL_37"}


def test_v03_assay_context_is_preserved():
    df = build_target_dataset(client=SnapshotFakeClient())
    active = df[df["active_label"] == 1].iloc[0]
    assert active["measurement_count"] == 2
    assert active["standard_type_count"] == 2
    assert active["assay_count"] == 2
    assert active["label_agreement"] == 1.0
    assert bool(active["assay_heterogeneity_flag"])
    assert active["pchembl_iqr"] > 0


def test_freeze_snapshot_and_verify_detects_tamper(tmp_path: Path):
    manifest = freeze_target_snapshot("LRRK2", tmp_path, client=SnapshotFakeClient())
    assert manifest["pdl_version"] == "0.3.0"
    assert manifest["snapshot_id"].startswith("chembl-chembl1075104-")
    assert (tmp_path / "raw_activities.jsonl").exists()
    assert (tmp_path / "raw_assays.jsonl").exists()
    assert (tmp_path / "cleaned_molecules.csv").exists()
    result = verify_manifest(tmp_path / "snapshot_manifest.json")
    assert result["ok"]

    with (tmp_path / "cleaned_molecules.csv").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    result = verify_manifest(tmp_path / "snapshot_manifest.json")
    assert not result["ok"]
    assert result["mismatched"]


def test_run_manifest_is_hashed_and_verifiable(tmp_path: Path):
    run_dir = tmp_path / "run"
    result = run_pipeline(make_demo_dataset(), run_dir, descriptor_count=32)
    manifest = result["manifest"]
    assert manifest["version"] == "0.3.0"
    assert manifest["run_id"].startswith("pdl-")
    assert manifest["artifact_sha256"]
    assert "environment" in manifest
    assert (run_dir / "assay_context_summary.json").exists()
    verified = verify_manifest(run_dir / "manifest.json")
    assert verified["ok"]
