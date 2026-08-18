# Parkinson Discovery Lab 🧠🧬⚛️

An evidence-first molecular ML research platform for **Parkinson's disease drug-discovery hypotheses**, starting with human **LRRK2**.

The project asks two separate questions:

1. **Can we build a reproducible, uncertainty-aware computational pipeline that narrows chemical space into better LRRK2 hypotheses?**
2. **Can Kipu Rimay quantum-derived molecular features improve held-out LRRK2 activity prediction beyond strong classical baselines on chemically novel scaffolds?**

The project does **not** claim that target binding equals disease modification, that a predicted molecule is a drug, that a molecular-property model establishes clinical safety, or that quantum advantage exists before it is measured.

## Current milestone: V0.4 — multi-property evidence + uncertainty ✅

V0.4 adds separately benchmarked **blood-brain-barrier penetration, clinical-toxicity and aqueous-solubility** models around the V0.3 LRRK2 activity benchmark.

```text
frozen LRRK2 ChEMBL snapshot
            ↓
      activity model
            ↓
  computational candidates
            ↓
 ┌──────────┼───────────┐
 ↓          ↓           ↓
BBBP      ClinTox      ESOL
 ↓          ↓           ↓
prob.      prob.       logS
entropy    entropy     PI90
AD sim.    AD sim.     AD sim.
 └──────────┼───────────┘
            ↓
 endpoint-by-endpoint evidence
            ↓
 no hidden “drug probability”
```

### V0.4 adds

- public benchmark registry/adapters for **BBBP**, **ClinTox (`CT_TOX`)** and **ESOL**;
- SHA-256 source sidecars for downloaded benchmark files;
- canonical-SMILES deduplication and rejection of conflicting duplicate classification labels;
- Bemis–Murcko scaffold train/validation/test splits;
- model-family selection on validation scaffolds only;
- validation-only **Platt calibration** for BBBP/ClinTox probabilities;
- validation-calibrated **90% residual intervals** for ESOL;
- Morgan/Tanimoto **applicability-domain similarity** for every prediction;
- endpoint manifests containing source hashes, environment, split and model contract;
- candidate annotation without collapsing endpoints into an arbitrary composite score.

See [`docs/V0.4_ADMET.md`](docs/V0.4_ADMET.md) for the full contract.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## 2. Freeze the real LRRK2 source dataset

```bash
pdl freeze-chembl \
  --target LRRK2 \
  --standard-types IC50,Ki \
  --out data/snapshots/lrrk2
```

This writes raw activities, assay context, cleaned molecules and `snapshot_manifest.json` with SHA-256 hashes and source/query provenance.

A stricter sensitivity snapshot can be generated separately:

```bash
pdl freeze-chembl \
  --target LRRK2 \
  --standard-types IC50 \
  --max-pchembl-iqr 1.0 \
  --min-label-agreement 0.75 \
  --out data/snapshots/lrrk2_ic50_strict
```

Verify the frozen source:

```bash
pdl verify --manifest data/snapshots/lrrk2/snapshot_manifest.json
```

## 3. Run the LRRK2 classical benchmark

```bash
pdl run \
  --input data/snapshots/lrrk2/cleaned_molecules.csv \
  --source-manifest data/snapshots/lrrk2/snapshot_manifest.json \
  --out artifacts/lrrk2 \
  --features 96 \
  --seed 42
```

The benchmark uses descriptor logistic regression, random forest, histogram gradient boosting and a 1,024-bit Morgan-fingerprint logistic baseline. Model family is selected on validation scaffolds; test scaffolds are evaluated after selection.

Repeat scaffold seeds before making stability claims:

```bash
pdl repeat \
  --input data/snapshots/lrrk2/cleaned_molecules.csv \
  --out artifacts/lrrk2_repeated \
  --seeds 11,23,42,71,101
```

## 4. Train the V0.4 ADMET evidence models

Download public benchmark files:

```bash
pdl admet-fetch --dataset bbbp
pdl admet-fetch --dataset clintox
pdl admet-fetch --dataset esol
```

Train each endpoint independently:

```bash
pdl admet-train --dataset bbbp \
  --input data/admet/bbbp.csv \
  --out artifacts/admet/bbbp

pdl admet-train --dataset clintox \
  --input data/admet/clintox.csv.gz \
  --out artifacts/admet/clintox

pdl admet-train --dataset esol \
  --input data/admet/esol.csv \
  --out artifacts/admet/esol
```

Then annotate the LRRK2 candidates:

```bash
pdl admet-annotate \
  --input artifacts/lrrk2/ranked_candidates.csv \
  --models artifacts/admet \
  --out artifacts/lrrk2/candidates_admet.csv
```

The output keeps each signal separate, for example:

```text
predicted_activity
bbbp_probability
bbbp_predictive_entropy
bbbp_ad_similarity
clintox_probability
clintox_predictive_entropy
clintox_ad_similarity
esol_prediction
esol_pi90_low
esol_pi90_high
esol_ad_similarity
```

A high BBBP prediction does not imply efficacy. A low ClinTox probability does not establish safety. ESOL is one developability signal, not a drug-quality verdict.

## 5. Rimay quantum benchmark

Create the frozen pilot handoff:

```bash
pdl rimay-pilot \
  --input artifacts/lrrk2/rimay_input.csv \
  --out artifacts/rimay_pilot \
  --size 300
```

Run the same molecules/split through **Rimay – Quantum Feature Extraction – Simulator** first, then import returned features or probabilities:

```bash
pdl rimay-compare \
  --prepared artifacts/lrrk2/dataset_prepared.csv \
  --rimay-result path/to/rimay_result.csv \
  --baseline artifacts/lrrk2/metrics.json \
  --out artifacts/lrrk2/quantum_comparison.json
```

A credible quantum result requires the same molecules, frozen scaffold split, same label contract, no validation/test leakage, repeated splits, uncertainty/statistical analysis, and reported compute time/cost. A negative result is also useful.

## Scientific boundary

The pipeline currently addresses only pieces of:

```text
binding
  → target engagement
  → cellular biology
  → brain exposure
  → ADME / safety
  → selectivity
  → disease mechanism
  → patient subgroup
  → clinical efficacy
```

Outputs are **computational hypotheses**. Do not synthesize or administer compounds based on this software.

## Roadmap to V1.0

### V0.3 — reproducible source + assay context ✅

Frozen ChEMBL source snapshots, integrity hashes, release/query provenance, assay heterogeneity, label-quality diagnostics and verifiable run manifests.

### V0.4 — real multi-property evidence + uncertainty ✅

BBBP, ClinTox and ESOL benchmark models; calibration/prediction intervals; applicability-domain evidence; endpoint-specific candidate annotation.

**Scientific completion still requires running the real public datasets and recording the held-out metrics. Synthetic CI tests validate software, not biology.**

### V0.5 — selectivity + medicinal chemistry

Next build:

- off-target/selectivity panel;
- PAINS/reactive/structural-alert evidence;
- novelty and nearest-neighbour chemistry;
- synthesizability estimates;
- family/model disagreement analysis;
- explicit rejection reasons.

### V0.6 — real quantum benchmark

- Rimay Simulator, then hardware only if justified;
- repeated scaffold splits;
- quantum-vs-classical confidence intervals;
- compute time and cost;
- disagreement/feature analysis;
- explicit pass/fail decision on quantum value for this task.

### V0.7 — large-library virtual screening

- 10K–100K+ external molecules;
- frozen library provenance;
- applicability-domain and uncertainty thresholds;
- deduplication against training chemistry;
- novelty/diversity selection.

### V0.8 — orthogonal structural evidence

- protein-structure provenance;
- conformer generation;
- docking/pose evidence as support, not truth;
- consensus/conflict with ligand-based models;
- uncertainty/failure flags.

### V0.9 — candidate evidence cards + expert gate

Every finalist should expose:

- identity/provenance;
- predicted LRRK2 activity and uncertainty;
- BBBP/ADMET/toxicity/selectivity evidence;
- applicability domain;
- nearest known chemistry and novelty;
- classical-vs-quantum disagreement;
- structural evidence;
- reasons to reject as well as investigate.

No candidate advances without a human scientific review gate.

### V1.0 — experimentally actionable shortlist

V1.0 is reached when a frozen source/library snapshot can reproducibly produce an **expert-reviewable top computational shortlist** with complete evidence provenance.

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

## References used by V0.4

- Wu et al., *MoleculeNet: a benchmark for molecular machine learning*, `doi:10.1039/C7SC02664A`.
- Martins et al., *A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling*, `doi:10.1021/ci300124c`.
- Delaney, *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure*, `doi:10.1021/ci034243x`.

## License

MIT.
