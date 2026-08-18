# Parkinson Discovery Lab 🧠🧬⚛️

An evidence-first molecular ML research platform for **Parkinson's disease drug-discovery hypotheses**, starting with human **LRRK2**.

The platform is deliberately designed to make claims harder, not easier: every layer exposes provenance, uncertainty, applicability limits and reasons a candidate might fail.

> **Goal:** reproducibly narrow a large chemical search space into a small, evidence-rich set of hypotheses worth experimental testing — and quantify whether quantum feature extraction improves that process.

## Current milestone: V0.6 — rigorous quantum evidence ✅

```text
frozen ChEMBL LRRK2
       ↓
classical activity ───── Rimay features/predictions
       │                         │
       └──── same test molecules ┘
                    ↓
           paired bootstrap
                    ↓
       PR-AUC / ROC-AUC / Brier
                    ↓
       disagreement molecules
                    ↓
     repeat across scaffold seeds
                    ↓
       PASS / FAIL / INCONCLUSIVE
                    ↓
    runtime + cost + backend context
```

A statistical predictive gain is **not automatically quantum computational advantage**. V0.6 keeps those claims separate.

## Built milestones

### V0.3 — reproducible source + assay context ✅
Frozen ChEMBL snapshots, raw activity/assay provenance, SHA-256 manifests, deterministic run IDs, label-quality diagnostics and Bemis–Murcko scaffold splits.

### V0.4 — ADMET evidence + uncertainty ✅
BBBP, ClinTox and ESOL benchmark models with validation-only calibration/prediction intervals and applicability-domain evidence. See [`docs/V0.4_ADMET.md`](docs/V0.4_ADMET.md).

### V0.5 — selectivity + medicinal chemistry ✅
RDKit structural-alert evidence, QED/complexity, nearest-known chemistry, novelty proxy, configurable ChEMBL-backed off-target surveillance and per-target models. See [`docs/V0.5_SELECTIVITY_MEDCHEM.md`](docs/V0.5_SELECTIVITY_MEDCHEM.md).

### V0.6 — paired quantum evidence ✅
- exact molecule/split alignment;
- frozen classical-model comparison;
- paired bootstrap confidence intervals;
- molecule-level disagreement analysis;
- simulator/QPU/backend/runtime/cost provenance;
- at least three repeated scaffold trials before a project-level decision;
- explicit pass/fail/inconclusive rules.

See [`docs/V0.6_QUANTUM_EVIDENCE.md`](docs/V0.6_QUANTUM_EVIDENCE.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Core research workflow

### 1. Freeze LRRK2 evidence

```bash
pdl freeze-chembl --target LRRK2 --standard-types IC50,Ki --out data/snapshots/lrrk2
```

### 2. Classical activity benchmark

```bash
pdl run \
  --input data/snapshots/lrrk2/cleaned_molecules.csv \
  --source-manifest data/snapshots/lrrk2/snapshot_manifest.json \
  --out artifacts/lrrk2 --features 96

pdl repeat \
  --input data/snapshots/lrrk2/cleaned_molecules.csv \
  --out artifacts/lrrk2_repeated \
  --seeds 11,23,42,71,101
```

### 3. BBBP / ClinTox / ESOL

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

### 4. Medicinal chemistry + selectivity

```bash
pdl medchem-annotate \
  --input artifacts/lrrk2/candidates_admet.csv \
  --reference data/snapshots/lrrk2/cleaned_molecules.csv \
  --out artifacts/lrrk2/candidates_medchem.csv

pdl selectivity-freeze \
  --targets LRRK1,TTK,STK10,MAPK14,JNK2,CLK1,JNK3,DYRK2,SLK,DDR2,STK17B \
  --out data/selectivity

pdl selectivity-train --input data/selectivity --out artifacts/selectivity

pdl selectivity-annotate \
  --input artifacts/lrrk2/candidates_medchem.csv \
  --models artifacts/selectivity \
  --out artifacts/lrrk2/candidates_selectivity.csv
```

The default panel is literature-seeded surveillance, not a universal selectivity panel. Experimental kinome profiling remains the serious-lead reference test.

### 5. Rimay pilot

```bash
pdl rimay-pilot \
  --input artifacts/lrrk2/rimay_input.csv \
  --out artifacts/rimay_pilot --size 300
```

Run the exact frozen input through Rimay, preserve `molecule_id`, labels and split, then evaluate one trial:

```bash
pdl quantum-trial \
  --prepared artifacts/lrrk2/dataset_prepared.csv \
  --classical-model artifacts/lrrk2/best_model.joblib \
  --rimay-result path/to/rimay_result.csv \
  --out artifacts/quantum/seed42 \
  --backend-type simulator \
  --backend-name "Rimay Simulator" \
  --quantum-runtime-seconds 420 \
  --quantum-cost-eur 0
```

Repeat on independent frozen scaffold trials, then:

```bash
pdl quantum-meta \
  --trials \
    artifacts/quantum/seed11/trial.json \
    artifacts/quantum/seed23/trial.json \
    artifacts/quantum/seed42/trial.json \
    artifacts/quantum/seed71/trial.json \
    artifacts/quantum/seed101/trial.json \
  --out artifacts/quantum/meta_benchmark.json
```

**Primary metric:** PR-AUC. Fewer than three trials cannot yield a project-level pass. A negative result remains useful.

## Scientific boundary

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

The project currently models only pieces of this chain. All outputs are **computational hypotheses**. Do not synthesize or administer compounds based on this software.

## Roadmap to V1.0

### V0.7 — large-library virtual screening
- frozen 10K–100K+ molecule library snapshots;
- batched inference and provenance;
- known-chemistry deduplication;
- applicability-domain + uncertainty gates;
- chemistry-diverse shortlist selection.

### V0.8 — orthogonal structural evidence
- target-structure provenance;
- conformer generation;
- docking/pose evidence as a separate signal;
- model-vs-structure agreement/conflict;
- failure flags rather than treating docking score as truth.

### V0.9 — candidate evidence cards + expert gate
Every finalist exposes identity/provenance, activity, uncertainty, BBBP/ADMET/toxicity, selectivity, applicability domain, novelty, alerts, classical-vs-quantum disagreement, structural evidence and explicit reasons to reject/investigate. No candidate advances without human scientific review.

### V1.0 — experimentally actionable computational shortlist
A frozen source dataset + frozen screening library reproducibly yields a small expert-reviewable shortlist with complete evidence provenance.

**V1.0 is not “we found a Parkinson's drug”.** It is a defensible hypothesis-generation system for deciding what is worth laboratory testing.

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
