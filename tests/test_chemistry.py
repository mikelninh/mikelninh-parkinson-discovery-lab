from parkinson_discovery.chemistry import canonical_smiles, descriptor_names, property_proxies


def test_descriptor_budget_for_rimay():
    assert len(descriptor_names(96)) == 96


def test_basic_chemistry():
    smi = canonical_smiles("CCOc1ccccc1")
    assert smi
    props = property_proxies(smi)
    assert 0 <= props["cns_likeness_proxy"] <= 1
    assert 0 <= props["drug_likeness_proxy"] <= 1
