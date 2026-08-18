from dataclasses import dataclass

TARGETS = {
    "LRRK2": "CHEMBL1075104",
}

ACTIVE_PCHEMBL = 6.0
INACTIVE_PCHEMBL = 5.0
DEFAULT_DESCRIPTOR_COUNT = 96
DEFAULT_FP_BITS = 1024


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15
    seed: int = 42
