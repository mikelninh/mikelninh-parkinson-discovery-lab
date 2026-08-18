# Methodology notes

## Labels

pChEMBL places molar potency measures on a logarithmic scale. V0.3 keeps IC50/Ki by default, aggregates repeated values by median and excludes the intentionally ambiguous 5–6 pChEMBL zone.

Because IC50 and Ki measurements across heterogeneous assays are not interchangeable in a strict mechanistic sense, V0.3 preserves assay context and measurement dispersion. Single-type snapshots and optional label-quality filters make sensitivity analysis explicit.

## Assay-context quality

For each canonical structure V0.3 records, when available:

- measurement, assay and document counts;
- standard types, assay types and organisms;
- pChEMBL range and IQR;
- label agreement: fraction of measurements on the same activity side as the final median-derived label;
- a transparent context-quality diagnostic `agreement / (1 + IQR)`;
- a heterogeneity flag when standard/assay/organism context differs or pChEMBL IQR is large.

These fields are evidence about label quality. They are intentionally **not** inserted into the default molecular feature matrix because doing so could let the model exploit assay metadata rather than chemistry.

## Split

Random molecular splits can leak close analogues across train and test. V0.3 keeps Bemis–Murcko scaffolds intact across train/validation/test. Model family selection is performed on validation scaffolds; the test scaffolds are not used for selection.

## Repeated scaffold seeds

A single scaffold partition may be unusually easy or difficult. `pdl repeat` repeats the complete classical benchmark over deterministic scaffold seeds and reports per-model mean/std/95% normal-approximation confidence intervals. This is a stability check, not a substitute for external/prospective validation.

## Rimay feature preparation

RDKit descriptors are imputed and standardized using **training rows only**. The export includes molecule IDs and scaffold-split labels so returned quantum features can be aligned by identity, never by row order.

## Ranking

Current ranking remains:

`rank_score = 0.65 activity + 0.20 CNS-likeness proxy + 0.15 drug-likeness proxy`.

This is intentionally transparent and intentionally limited. V0.4 replaces these heuristic proxies with externally validated endpoint models and uncertainty.

## Interpretation

Even a successful LRRK2 activity classifier only addresses target-level molecular activity. It does not establish brain exposure, safety, disease modification, or clinical efficacy.
