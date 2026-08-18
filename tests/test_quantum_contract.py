import pytest

from parkinson_discovery.demo_data import make_demo_dataset
from parkinson_discovery.pipeline import prepare_dataset
from parkinson_discovery.quantum import export_rimay


def test_rimay_feature_limit(tmp_path):
    df = prepare_dataset(make_demo_dataset())
    with pytest.raises(ValueError):
        export_rimay(df, tmp_path / "bad.csv", feature_count=156)
