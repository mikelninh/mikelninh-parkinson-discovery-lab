# Reproducibility contract — V0.3

## Why snapshots exist

A live ChEMBL query can change when the underlying database is updated. A reported benchmark therefore needs to identify the exact source records used, not only the query that produced them.

`pdl freeze-chembl` stores the raw activity response, fetched assay metadata, derived cleaned molecule table and a manifest containing the source/query contract and SHA-256 hashes.

## Snapshot identity

`snapshot_id` is derived from:

- ChEMBL release metadata available at fetch time;
- target/query contract;
- SHA-256 of the cleaned molecule table.

It is intended as a convenient stable label. Integrity is enforced by the full SHA-256 hashes in `snapshot_manifest.json`.

## Run identity

`run_id` is derived from:

- SHA-256 of `dataset_prepared.csv`;
- experiment configuration contract.

The run manifest also records the Python/package environment and hashes every generated artifact.

## Verification

```bash
pdl verify --manifest data/snapshots/lrrk2/snapshot_manifest.json
pdl verify --manifest artifacts/lrrk2_v03/manifest.json
```

Verification detects missing and byte-modified artifacts.

## What hashes do not prove

Hashes prove identity/integrity, not scientific correctness. A perfectly reproducible dataset may still contain assay bias, label noise, target-selection problems or biologically irrelevant endpoints.
