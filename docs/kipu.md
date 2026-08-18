# Kipu / Rimay experiment card

## External assessment

Kipu Tinkuq assessed the proposed LRRK2 molecular-classification workflow as **Suitable (85/100)** and rated **Rimay – Quantum Feature Extraction 95/100** for the expected 1,000–4,000 samples and 96–128 features.

The same assessment recommends a **200–500 molecule pilot** before a full deployment. It also forecasts a +5–12pp ROC-AUC improvement. The repository treats that improvement as a **testable provider forecast**, not measured evidence.

## Research question

Does Rimay-derived quantum feature extraction improve LRRK2 activity classification on chemically novel scaffolds versus strong classical molecular representations?

## Input contract

- default: 96 numerical RDKit descriptors (strictly <156)
- binary activity label from pChEMBL thresholds
- fixed Bemis–Murcko scaffold split
- molecule ID + canonical SMILES for traceability
- preprocessing fitted on training molecules only

## Baselines

- descriptor logistic regression
- descriptor random forest
- descriptor histogram gradient boosting
- Morgan fingerprint logistic regression

## Primary metrics

- PR-AUC (model-family selection)
- ROC-AUC
- F1
- Brier score / calibration

## V0.2 handoff loop

```bash
pdl rimay-pilot --input artifacts/lrrk2/rimay_input.csv --out artifacts/rimay_pilot --size 300
```

Run the resulting CSV through **Rimay – Quantum Feature Extraction – Simulator** first. Export returned features or active-class probabilities, then:

```bash
pdl rimay-compare \
  --prepared artifacts/lrrk2/dataset_prepared.csv \
  --rimay-result rimay_result.csv \
  --baseline artifacts/lrrk2/metrics.json \
  --out artifacts/lrrk2/quantum_comparison.json
```

## Managed-service API

Kipu's Service SDK supports programmatic invocation of subscribed managed services using `HubServiceClient(service_endpoint, access_key_id, secret_access_key)` and `run(request=...)`.

We deliberately do not hard-code the Rimay request body because the exact marketplace service OpenAPI schema must be read from the subscribed service/application. Use:

```bash
pip install -e '.[kipu]'
export KIPU_ACCESS_KEY_ID='...'
export KIPU_SECRET_ACCESS_KEY='...'
pdl kipu-run --endpoint '...' --request rimay_request.json
```

## MCP option

Kipu's hosted remote MCP server:

```text
https://api.hub.kipu-quantum.com/mcp
```

Example project configuration for an MCP-capable editor/client:

```json
{
  "mcpServers": {
    "qhub-mcp": {
      "type": "http",
      "url": "https://api.hub.kipu-quantum.com/mcp"
    }
  }
}
```

Authentication is OAuth. The server exposes Hub/quantum tools and a top-level `run_subscribed_service` tool.

## Win condition

A useful result is a reproducible gain over the **strongest** classical baseline on the same held-out scaffold set, preferably across repeated scaffold seeds, with compute/runtime cost reported.

A single random split, a gain against a weak baseline, or a provider forecast is insufficient.

## What this experiment cannot show

It cannot establish disease modification, brain exposure, human safety, target validity, clinical efficacy, or quantum advantage outside this defined benchmark.
