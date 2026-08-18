# Parkinson Discovery Lab 🧠🧬⚛️

An evidence-first molecular ML research platform for **Parkinson's disease drug-discovery hypotheses**, starting with human **LRRK2**.

The platform is deliberately designed to make claims harder, not easier: every layer should expose provenance, uncertainty, applicability limits and reasons a candidate might fail.

> **Goal:** reproducibly narrow a large chemical search space into a small, evidence-rich set of hypotheses worth experimental testing — and quantify whether quantum feature extraction improves that process.

## Current milestone: V0.5 — selectivity + medicinal chemistry ✅

```text
frozen LRRK2 ChEMBL evidence
          ↓
classical activity benchmark ─── Rimay-ready benchmark
          ↓
BBBP / ClinTox / ESOL
          ↓
medicinal-chemistry evidence
  • PAINS/BRENK/NIH/ZINC alerts
  • QED + complexity
  • nearest known chemistry
  • novelty proxy
          ↓
off-target surveillance
  • configurable kinase panel
  • target-specific ChEMBL snapshots
  • target-specific scaffold models
          ↓
reasons to investigate
+ reasons to reject
```

V0.5 does **not** claim that a predicted molecule is a drug, that low predicted toxicity establishes safety, that an alert automatically disqualifies chemistry, or that a computational off-target probability equals an experimental selectivity ratio.

## Milestones already built

### V0.3 — reproducible source + assay context ✅

- frozen ChEMBL snapshots;
- raw activity + assay provenance;
- SHA-256 integrity manifests;
- deterministic run IDs;
- assay heterogeneity and label-quality diagnostics;
- Bemis–Murcko scaffold benchmark contract.

### V0.4 — ADMET evidence + uncertainty ✅

- BBBP blood-brain-barrier classification;
- ClinTox toxicity classification;
- ESOL aqueous-solubility regression;
- validation-only calibration / prediction intervals;
- applicability-domain similarity;
- endpoint-specific outputs rather than a hidden “drug score”.

See [`docs/V0.4_ADMET.md`](docs/V0.4_ADMET.md).

### V0.5 — selectivity + medicinal chemistry ✅

- RDKit PAINS, BRENK, NIH and ZINC structural-alert evidence;
- QED and transparent molecular-complexity descriptors;
- nearest-known Morgan/Tanimoto chemistry + explicit novelty proxy;
- medchem review flags rather than automatic rejection;
- configurable LRRK2 off-target surveillance panel;
- live ChEMBL target resolution that fails on ambiguity instead of guessing;
- frozen per-target selectivity datasets;
- off-target models trained only where the data support a binary benchmark;
- individual predicted off-target probabilities and worst predicted off-target.

See [`docs/V0.5_SELECTIVITY_MEDCHEM.md`](docs/V0.5_SELECTIVITY_MEDCHEM.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

### 1. Freeze the real LRRK2 source

```bash
pdl freeze-chembl \
  --target LRRK2 \
  --standard-types IC50,Ki \
  --out data/snapshots/lrrk2
```

### 2. Establish the classical activity benchmark

```bash
pdl run \
  --input data/snapshots/lrrk2/cleaned_molecules.csv \
  --source-manifest data/snapshots/lrrk2/snapshot_manifest.json \
  --out artifacts/lrrk2 \
  --features 96

pdl repeat \
  --input data/snapshots/lrrk2/cleaned_molecules.csv \
  --out artifacts/lrrk2_repeated \
  --seeds 11,23,42,71,101
```

### 3. Train the V0.4 property models

```bash
pdl admet-fetch --dataset bbbp
pdl admet-fetch --dataset clintox
pdl admet-fetch --dataset esol

pdl admet-train --dataset bbbp --input data/admet/bbbp.csv --out artifacts/admet/bbbp
pdl admet-train --dataset clintox --input data/admet/clintox.csv.gz --out artifacts/admet/clintox
pdl admet-train --dataset esol --input data/admet/esol.csv --out artifacts/admet/esol

pdl admet-annotate \
  --input artifacts/lrrk2/ranked_candidates.csv \
  --models artifacts/admet \
  --out artifacts/lrrk2/candidates_admet.csv
```

### 4. Add medicinal-chemistry evidence

```bash
pdl medchem-annotate \
  --input artifacts/lrrk2/candidates_admet.csv \
  --reference data/snapshots/lrrk2/cleaned_molecules.csv \
  --out artifacts/lrrk2/candidates_medchem.csv
```

### 5. Build the selectivity surveillance panel

The default panel is literature-seeded and configurable. It is **not** presented as a universal LRRK2 off-target panel.

```bash
pdl selectivity-resolve --target TTK

pdl selectivity-freeze \
  --targets LRRK1,TTK,STK10,MAPK14,JNK2,CLK1,JNK3,DYRK2,SLK,DDR2,STK17B \
  --out data/selectivity

pdl selectivity-train \
  --input data/selectivity \
  --out artifacts/selectivity

pdl selectivity-annotate \
  --input artifacts/lrrk2/candidates_medchem.csv \
  --models artifacts/selectivity \
  --out artifacts/lrrk2/candidates_selectivity.csv
```

### 6. Prepare the Rimay experiment

```bash
pdl rimay-pilot \
  --input artifacts/lrrk2/rimay_input.csv \
  --out artifacts/rimay_pilot \
  --size 300
```

Run the exact frozen pilot through **Rimay – Quantum Feature Extraction – Simulator**, then import the returned features/probabilities:

```bash
pdl rimay-compare \
  --prepared artifacts/lrrk2/dataset_prepared.csv \
  --rimay-result path/to/rimay_result.csv \
  --baseline artifacts/lrrk2/metrics.json \
  --out artifacts/lrrk2/quantum_comparison.json
```

A negative quantum result is still a useful result. We do not optimise the experiment to make quantum look good.

## Scientific boundary

The project currently covers only pieces of:

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

All outputs are **computational hypotheses**. Do not synthesize or administer compounds based on this software.

## Roadmap to V1.0

### V0.6 — rigorous quantum experiment

- repeated classical-vs-Rimay scaffold comparisons;
- paired bootstrap/confidence intervals;
- compute runtime and cost accounting;
- prediction-disagreement analysis;
- explicit **pass / inconclusive / fail** decision on quantum value for this task.

### V0.7 — large-library virtual screening

- frozen 10K–100K+ molecule library snapshots;
- streaming/batched inference;
- duplicate and known-chemistry exclusion;
- applicability-domain and uncertainty gates;
- chemistry-diverse shortlist selection rather than 100 near-identical analogues.

### V0.8 — orthogonal structural evidence

- target-structure provenance;
- ligand conformers;
- docking/pose evidence as a separate supporting signal;
- model-vs-structure agreement/conflict;
- failure flags instead of pretending docking scores are binding free energies.

### V0.9 — candidate evidence cards + expert gate

Every finalist must expose:

- identity and full provenance;
- LRRK2 activity prediction + uncertainty;
- BBBP / ClinTox / ESOL evidence;
- selectivity surveillance;
- applicability domain;
- nearest known chemistry / novelty;
- structural alerts and complexity;
- classical-vs-quantum disagreement;
- structural evidence;
- explicit reasons to reject and investigate.

No candidate advances without a human scientific review gate.

### V1.0 — experimentally actionable computational shortlist

V1.0 is complete when a frozen source dataset and frozen screening library can reproducibly produce a small, expert-reviewable shortlist with all evidence and provenance attached.

**V1.0 is not “we found a Parkinson's drug”.** It is a research system capable of producing defensible hypotheses for laboratory testing.

## Development

```bash
pytest
ruff check src tests
pdl demo --out artifacts/demo
pdl verify --manifest artifacts/demo/manifest.json
pdl serve --artifacts artifacts/demo
```

## Key references

- Wu et al., *MoleculeNet*, `doi:10.1039/C7SC02664A`.
- Martins et al., BBBP benchmark source, `doi:10.1021/ci300124c`.
- Delaney, ESOL, `doi:10.1021/ci034243x`.
- *Type II kinase inhibitors that target Parkinson's disease-associated LRRK2*, Science Advances (2025), `doi:10.1126/sciadv.ads3128`.
- *Discovery of Potent, Selective, CNS-Penetrant Macrocyclic LRRK2 Inhibitors for the Treatment of Parkinson's Disease*, Journal of Medicinal Chemistry (2026), `doi:10.1021/acs.jmedchem.6c00238`.

## License

MIT.
