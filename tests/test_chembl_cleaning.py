from parkinson_discovery.chembl import build_target_dataset


class FakeClient:
    def activities(self, target_id):
        return [
            {"molecule_chembl_id": "CHEMBL1", "canonical_smiles": "c1ccccc1F", "pchembl_value": "6.4", "standard_type": "IC50", "standard_relation": "=", "assay_chembl_id": "A1", "document_chembl_id": "D1"},
            {"molecule_chembl_id": "CHEMBL1", "canonical_smiles": "Fc1ccccc1", "pchembl_value": "6.2", "standard_type": "IC50", "standard_relation": "=", "assay_chembl_id": "A2", "document_chembl_id": "D2"},
            {"molecule_chembl_id": "CHEMBL2", "canonical_smiles": "c1ccccc1Cl", "pchembl_value": "4.7", "standard_type": "Ki", "standard_relation": "=", "assay_chembl_id": "A3", "document_chembl_id": "D3"},
            {"molecule_chembl_id": "CHEMBL3", "canonical_smiles": "c1ccccc1Br", "pchembl_value": "5.5", "standard_type": "IC50", "standard_relation": "=", "assay_chembl_id": "A4", "document_chembl_id": "D4"},
            {"molecule_chembl_id": "CHEMBL4", "canonical_smiles": "c1ccncc1F", "pchembl_value": "7.0", "standard_type": "IC50", "standard_relation": ">", "assay_chembl_id": "A5", "document_chembl_id": "D5"},
        ]

    def molecule_smiles(self, ids):
        return {}


def test_cleaning_aggregates_and_excludes_ambiguous_relations():
    df = build_target_dataset(client=FakeClient())
    assert len(df) == 2
    assert set(df["active_label"]) == {0, 1}
    active = df[df["active_label"] == 1].iloc[0]
    assert active["measurement_count"] == 2
    assert abs(active["pchembl_value"] - 6.3) < 1e-9
