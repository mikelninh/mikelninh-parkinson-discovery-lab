# Parkinson Discovery Lab 🧠🧬⚛️

An evidence-first molecular ML benchmark for **Parkinson's disease drug-discovery research**, starting with human **LRRK2**. The core question is deliberately narrow:

> **Can Kipu Rimay quantum-derived molecular features improve held-out LRRK2 activity prediction beyond strong classical baselines on chemically novel scaffolds?**

Kipu's Tinkuq assessment rated this setup **Suitable (85/100)** and recommended **Rimay – Quantum Feature Extraction** for the 1,000–4,000 molecule / 96–128 feature regime. That is a provider assessment, **not a measured performance result**. V0.2 turns the assessment into a falsifiable experiment.

## V0.2: from “quantum-ready” to an actual Rimay experiment

```text
ChEMBL LRRK2 activities
        ↓
strict cleaning + provenance
        ↓
canonical molecules + pChEMBL labels
        ↓
Bemis–Murcko scaffold split
        ↓
┌───────────────────────────┐
│ Classical baselines       │
│ descriptor LR / RF / HGB  │
│ Morgan-fingerprint LR     │
└─────────────┬─────────────┘
              ↓
      frozen benchmark
              ↓
96-feature Rimay export
              ↓
200–500 molecule pilot bundle
              ↓
Rimay Simulator / subscribed service
              ↓
returned quantum features or probabilities
              ↓
┌──────────────────────────────┐
│ same molecules + same split  │
│ ROC-AUC / PR-AUC / F1/Brier  │
└──────────────┬───────────────┘
               ↓
      quantum-v-classical delta
               ↓
 repeated scaffold seeds before any
       “quantum advantage” claim
```

## What changed from V0.1

- **Rimay pilot builder**: deterministic 200–500 molecule handoff bundle.
- **Rimay result importer/comparator**: accepts returned quantum features *or* probabilities and evaluates them on the frozen split.
- **Repeated scaffold benchmarks**: multi-seed classical baselines with mean/std/95% normal-approximation confidence intervals.
- **Kipu managed-service adapter**: optional official `qhub-service` SDK integration, without guessing Rimay's private/subscription-specific request schema.
- **Stricter feature contract**: at most 155 features; default remains 96.
- **Assay-type control**: `IC50,Ki` by default, with easy `--standard-types IC50` sensitivity runs.
- **Dashboard V0.2**: shows Rimay status and quantum-classical deltas once a result is imported.
- **No fake quantum score**: Kipu's forecast is recorded as a hypothesis to test, not as a result.

## 1. Run the software immediately

The demo uses **synthetic molecules and labels** only to prove the software path works.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pdl demo --out artifacts/demo
pdl serve --artifacts artifacts/demo
```

Open `http://127.0.0.1:8000`.

## 2. Real LRRK2 benchmark

```bash
pdl fetch-chembl --target LRRK2 --out data/lrrk2_chembl.csv
pdl run --input data/lrrk2_chembl.csv --out artifacts/lrrk2 --features 96 --seed 42
```

For an assay-type sensitivity check:

```bash
pdl fetch-chembl --target LRRK2 --standard-types IC50 --out data/lrrk2_ic50.csv
pdl run --input data/lrrk2_ic50.csv --out artifacts/lrrk2_ic50
```

The ChEMBL downloader requires network access. If the environment is offline, run these commands locally and keep the produced CSV snapshot under `data/` or an external data store.

## 3. Establish baseline stability across scaffold splits

```bash
pdl repeat \
  --input data/lrrk2_chembl.csv \
  --out artifacts/lrrk2_repeated \
  --seeds 11,23,42,71,101
```

Outputs:

- `repeated_metrics.csv` — one row per seed × model.
- `repeated_summary.json` — mean/std/CI by model.
- per-seed model artifacts for auditability.

This is important because a single scaffold split can flatter or punish a molecular model by chance.

## 4. Build the Rimay Simulator pilot

Kipu's assessment recommends sharing **200–500 molecules** first. V0.2 creates that handoff from the already-frozen experiment:

```bash
pdl rimay-pilot \
  --input artifacts/lrrk2/rimay_input.csv \
  --out artifacts/rimay_pilot \
  --size 300
```

The bundle contains:

- `rimay_pilot.csv`
- `rimay_pilot_manifest.json`
- `README.md`

Use **Rimay – Quantum Feature Extraction – Simulator** first. Keep the supplied `molecule_id`, `active_label`, and `split` unchanged.

## 5. Import the Rimay result

V0.2 supports either:

### A. Quantum features

A CSV with `molecule_id` plus numeric returned feature columns.

### B. Quantum probabilities

A CSV with `molecule_id,prediction`, where `prediction` is the active-class probability in `[0,1]`.

Then:

```bash
pdl rimay-compare \
  --prepared artifacts/lrrk2/dataset_prepared.csv \
  --rimay-result path/to/rimay_result.csv \
  --baseline artifacts/lrrk2/metrics.json \
  --out artifacts/lrrk2/quantum_comparison.json
```

The comparator refuses missing/duplicate molecule IDs and checks returned labels/splits if supplied. It aligns by molecule ID, **never by row order**.

## 6. Optional direct Kipu Hub API integration

Kipu's current Hub docs provide an official Python **Service SDK** for subscribed managed services. Install the optional integration:

```bash
pip install -e ".[kipu]"
```

After subscribing to the Rimay Simulator in a Kipu application, obtain the gateway endpoint and access keys from the application. Set:

```bash
export KIPU_ACCESS_KEY_ID="..."
export KIPU_SECRET_ACCESS_KEY="..."
```

Then provide a JSON request that matches **the actual OpenAPI/request schema shown for your subscribed Rimay service**:

```bash
pdl kipu-run \
  --endpoint "https://gateway.hub.kipu-quantum.com/..." \
  --request rimay_request.json \
  --out artifacts/lrrk2/kipu_execution.json
```

The repository intentionally does **not** invent a Rimay payload schema that we cannot verify publicly.

## Agentic Kipu integration

Kipu also exposes a hosted MCP server at:

```text
https://api.hub.kipu-quantum.com/mcp
```

MCP-capable clients can authenticate through OAuth and use the Hub tools, including a `run_subscribed_service` tool. See `docs/kipu.md` for a project-level configuration example.

## Classical benchmark contract

Model family selection happens on **validation scaffolds only**. After selection, each baseline is refit on train + validation and evaluated once on the test scaffolds.

Metrics:

- ROC-AUC
- PR-AUC (primary selection metric)
- F1
- accuracy
- Brier score / calibration

Representations:

- 96 selected RDKit descriptors
- 1,024-bit Morgan fingerprints (classical-only baseline)

## ChEMBL cleaning defaults

- human LRRK2 target `CHEMBL1075104`
- quantitative `IC50` and `Ki`
- valid `pChEMBL` values
- equality measurements only
- repeated values aggregated by median per canonical structure
- `pChEMBL >= 6` → active
- `pChEMBL <= 5` → inactive
- `5 < pChEMBL < 6` → excluded as ambiguous
- assay/document IDs preserved

**Caveat:** heterogeneous IC50/Ki assays are not perfectly interchangeable. V0.2 makes single-type sensitivity runs easy; later versions should explicitly model assay context.

## Quantum win condition

We do **not** ask whether Rimay beats a weak baseline. We ask whether it beats the **strongest classical pipeline** under the same chemistry split.

A credible win requires:

1. same molecules;
2. same frozen scaffold split;
3. no preprocessing fitted on validation/test labels;
4. improvement in held-out ROC-AUC/PR-AUC, not just training accuracy;
5. replication across scaffold seeds;
6. runtime and compute cost reported;
7. no extrapolation from target binding to clinical efficacy.

Kipu's Tinkuq report forecast **+5–12 percentage points ROC-AUC**. V0.2 treats that number as a **provider hypothesis to verify**, not a promised outcome.

## Scientific boundary

Target activity is only the first part of:

`binding → target engagement → cellular biology → brain exposure → safety → disease mechanism → patient subgroup → clinical efficacy`

Outputs are **computational hypotheses**. Do not synthesize or administer compounds based on this software. A serious programme requires validated BBB/ADMET/selectivity models, structural evidence, experimental assays, PK/PD, toxicology and clinical studies.

## Roadmap

- **V0.1:** reproducible data → scaffold split → classical baseline → Rimay-ready export.
- **V0.2 (this version):** Rimay pilot/import loop + repeated scaffold benchmark + optional Kipu SDK integration.
- **V0.3:** real frozen ChEMBL snapshot + assay-context modelling + data/version provenance hashes.
- **V0.4:** externally validated BBB/ADMET/toxicity models + uncertainty.
- **V0.5:** selectivity/off-target panel + medicinal-chemistry alerts.
- **V0.6:** real Rimay simulator/hardware repeated results + cost/latency benchmark.
- **V1:** large-library screening + novelty + structural/docking evidence + expert-reviewed experimental shortlist.

## License

MIT.
