# Parkinson Discovery Lab 🧠🧬⚛️

An evidence-first molecular ML benchmark for **Parkinson's disease drug-discovery research**, starting with human **LRRK2**.

The project asks a deliberately narrow question:

> **Can Kipu Rimay quantum-derived molecular features improve held-out LRRK2 activity prediction beyond strong classical baselines on chemically novel scaffolds?**

The project does **not** claim that target binding equals disease modification, that a predicted molecule is a drug, or that quantum advantage exists before it is measured.

## V0.3 — frozen provenance + assay context

V0.3 upgrades the project from a reproducible ML pipeline to a **reproducible scientific dataset/run contract**.

```text
live ChEMBL
    ↓
freeze raw activities + assay metadata + release/query contract
    ↓
SHA-256 verified source snapshot
    ↓
canonical molecules + assay-context diagnostics
    ↓
label quality / heterogeneity audit
    ↓
Bemis–Murcko scaffold split
    ↓
classical baselines + frozen run manifest
    ↓
96-feature Rimay export
    ↓
quantum-v-classical benchmark
```

### What V0.3 adds

- **Frozen ChEMBL snapshots** with raw activity JSONL, raw assay JSONL, cleaned molecule CSV and source manifest.
- **ChEMBL release metadata** captured at snapshot time when the API exposes it.
- **SHA-256 integrity hashes** for source snapshots and generated run artifacts.
- **Deterministic run IDs** derived from the prepared dataset and experiment contract.
- **Environment fingerprinting** for Python, RDKit, pandas, NumPy, scikit-learn and requests.
- **Assay-context audit**: standard type, assay type/organism, pChEMBL dispersion, label agreement and heterogeneity flags.
- **Quality filters** for excessive pChEMBL IQR and low measurement/label agreement.
- **No assay shortcut**: assay context is metadata/quality evidence, not silently inserted into the molecular predictor.
- **`pdl verify`** to detect missing or modified snapshot/run artifacts.
- Dashboard now surfaces snapshot/run identity and assay heterogeneity.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## 2. Freeze the real LRRK2 source dataset

This is the preferred V0.3 path:

```bash
pdl freeze-chembl \
  --target LRRK2 \
  --standard-types IC50,Ki \
  --out data/snapshots/lrrk2
```

It writes:

```text
data/snapshots/lrrk2/
├── raw_activities.jsonl
├── raw_assays.jsonl
├── cleaned_molecules.csv
└── snapshot_manifest.json
```

The manifest records the target, filtering contract, pChEMBL thresholds, aggregation rule, ChEMBL release metadata when available, row counts and SHA-256 hashes.

A stricter sensitivity snapshot can be created independently:

```bash
pdl freeze-chembl \
  --target LRRK2 \
  --standard-types IC50 \
  --max-pchembl-iqr 1.0 \
  --min-label-agreement 0.75 \
  --out data/snapshots/lrrk2_ic50_strict
```

## 3. Verify the frozen source

```bash
pdl verify --manifest data/snapshots/lrrk2/snapshot_manifest.json
```

A hash mismatch fails with exit code 2.

## 4. Run the classical benchmark from that exact snapshot

```bash
pdl run \
  --input data/snapshots/lrrk2/cleaned_molecules.csv \
  --source-manifest data/snapshots/lrrk2/snapshot_manifest.json \
  --out artifacts/lrrk2_v03 \
  --features 96 \
  --seed 42
```

The run now produces:

```text
artifacts/lrrk2_v03/
├── dataset_prepared.csv
├── assay_context_summary.json
├── metrics.json
├── best_model.joblib
├── ranked_candidates.csv
├── rimay_input.csv
├── rimay_input_feature_map.csv
├── report.md
└── manifest.json
```

`manifest.json` links the benchmark back to the frozen source snapshot and hashes every generated artifact.

## 5. Inspect assay context

```bash
pdl assay-audit \
  --input data/snapshots/lrrk2/cleaned_molecules.csv \
  --out artifacts/lrrk2_assay_audit.json
```

Per molecule, V0.3 can preserve:

- measurement count;
- assay and document count;
- standard types;
- assay types and assay organisms when supplied by ChEMBL;
- pChEMBL median/min/max/IQR;
- label agreement across repeated measurements;
- transparent context-quality score;
- assay-heterogeneity flag.

These fields are **not part of the 96 molecular features sent to Rimay**.

## 6. Baseline stability across scaffold splits

```bash
pdl repeat \
  --input data/snapshots/lrrk2/cleaned_molecules.csv \
  --out artifacts/lrrk2_repeated \
  --seeds 11,23,42,71,101
```

A single scaffold split can be unusually easy or hard; repeated seeds provide a better picture of stability.

## 7. Rimay pilot

```bash
pdl rimay-pilot \
  --input artifacts/lrrk2_v03/rimay_input.csv \
  --out artifacts/rimay_pilot \
  --size 300
```

Run the same molecules/split through **Rimay – Quantum Feature Extraction – Simulator** first.

Then import returned features or active-class probabilities:

```bash
pdl rimay-compare \
  --prepared artifacts/lrrk2_v03/dataset_prepared.csv \
  --rimay-result path/to/rimay_result.csv \
  --baseline artifacts/lrrk2_v03/metrics.json \
  --out artifacts/lrrk2_v03/quantum_comparison.json
```

The comparator aligns by `molecule_id`, never row order.

## Classical benchmark contract

Model-family selection happens on **validation scaffolds only**. Each model is then refit on train + validation and evaluated once on untouched test scaffolds.

Models:

- descriptor logistic regression;
- descriptor random forest;
- descriptor histogram gradient boosting;
- Morgan-fingerprint logistic regression.

Metrics:

- PR-AUC — primary model-selection metric;
- ROC-AUC;
- F1;
- accuracy;
- Brier score.

Representations:

- 96 selected RDKit numerical descriptors for the Rimay-comparable path;
- 1,024-bit Morgan fingerprints as a stronger classical-only representation baseline.

## ChEMBL cleaning contract

Defaults:

- human LRRK2: `CHEMBL1075104`;
- `IC50` + `Ki`;
- `standard_relation == "="`;
- valid pChEMBL values;
- canonical structure is the ML unit;
- repeated measurements aggregated by median;
- `pChEMBL >= 6` → active;
- `pChEMBL <= 5` → inactive;
- `5 < pChEMBL < 6` → ambiguous and excluded;
- source assay/document/activity IDs preserved;
- label dispersion and context heterogeneity recorded.

IC50 and Ki from heterogeneous assays are not mechanistically identical. V0.3 therefore makes the heterogeneity visible, supports single-type sensitivity snapshots, and avoids pretending the pooled label is perfect ground truth.

## Quantum win condition

We do **not** ask whether Rimay beats a weak baseline.

A credible positive result requires:

1. same molecules;
2. same frozen scaffold split;
3. identical label contract;
4. no preprocessing fitted on validation/test labels;
5. improvement over the strongest classical comparator on held-out data;
6. replication across multiple scaffold splits;
7. uncertainty/statistical analysis;
8. quantum compute/runtime/cost reported;
9. no extrapolation from target binding to clinical efficacy.

A negative result is also useful: if Rimay does not improve this benchmark, the repository should say so plainly.

## Scientific boundary

LRRK2 target activity addresses only the first part of the chain:

```text
binding
  → target engagement
  → cellular biology
  → brain exposure
  → selectivity / safety
  → disease mechanism
  → patient subgroup
  → clinical efficacy
```

Outputs are **computational hypotheses**. Do not synthesize or administer compounds based on this software.

## Roadmap to V1.0

### V0.3 — reproducible source + assay context ✅

Frozen ChEMBL snapshot, integrity hashes, release/query provenance, assay heterogeneity, quality diagnostics and verifiable run manifests.

### V0.4 — real multi-property evidence

Replace heuristic CNS/drug-likeness scores with independently validated predictors/datasets for:

- BBB permeability;
- solubility;
- basic ADME;
- toxicity endpoints;
- calibrated uncertainty and applicability domain.

The goal is **not** one magic score; preserve each endpoint and its uncertainty separately.

### V0.5 — selectivity + medicinal chemistry

Add:

- off-target/selectivity panel;
- structural-alert flags;
- novelty / nearest-neighbour evidence;
- synthesizability estimates;
- disagreement analysis across model families.

### V0.6 — real quantum benchmark

Run Rimay Simulator and, if justified, hardware on the frozen benchmark:

- repeated scaffold splits;
- quantum vs classical confidence intervals;
- compute time and cost;
- feature/disagreement analysis;
- explicit pass/fail decision on whether quantum adds value here.

### V0.7 — virtual screening

Screen a larger external compound library while enforcing:

- applicability-domain checks;
- uncertainty thresholds;
- deduplication against training chemistry;
- novelty/diversity selection;
- reproducible library provenance.

### V0.8 — structural evidence

Add orthogonal structure-based evidence where credible:

- protein-structure provenance;
- conformer generation;
- docking/pose scoring as supporting evidence, not truth;
- consensus with ligand-based models;
- failure/uncertainty flags.

### V0.9 — candidate evidence cards + expert review gate

Generate auditable candidate cards showing:

- source molecule identity;
- predicted LRRK2 activity;
- BBB/ADME/toxicity/selectivity estimates;
- uncertainty/applicability domain;
- nearest known chemistry;
- classical vs quantum rank;
- structural evidence;
- reasons to reject as well as reasons to investigate.

No candidate advances without a human scientific review gate.

### V1.0 — experimentally actionable shortlist

V1.0 is reached when the system can reproducibly take a frozen source/library snapshot and produce an **expert-reviewable top computational shortlist** with complete evidence provenance.

V1.0 success is **not** “we found a Parkinson's drug.” It is:

> **We built a reproducible system that narrows a large chemical search space into a small, evidence-rich set of hypotheses worth experimental testing — and can quantify whether quantum feature extraction improved that process.**

## Development

```bash
pytest
ruff check src tests
pdl demo --out artifacts/demo
pdl verify --manifest artifacts/demo/manifest.json
pdl serve --artifacts artifacts/demo
```

Open `http://127.0.0.1:8000`.

## License

MIT.
