# Methodology notes

## Labels

pChEMBL places molar potency measures on a logarithmic scale. V0.2 keeps IC50/Ki by default, aggregates repeated values by median and excludes the intentionally ambiguous 5–6 pChEMBL zone.

Because IC50 and Ki measurements across heterogeneous assays are not interchangeable in a strict mechanistic sense, V0.2 supports a single-type sensitivity run (`--standard-types IC50`). Future versions should incorporate assay context directly.

## Split

Random molecular splits can leak close analogues across train and test. V0.2 keeps Bemis–Murcko scaffolds intact across train/validation/test. Model family selection is performed on validation scaffolds; the test scaffolds are not used for selection.

## Repeated scaffold seeds

A single scaffold partition may be unusually easy or difficult. `pdl repeat` repeats the complete classical benchmark over deterministic scaffold seeds and reports per-model mean/std/95% normal-approximation confidence intervals. This is an engineering/statistical stability check, not a substitute for an external prospective dataset.

## Rimay feature preparation

RDKit descriptors are imputed and standardized using **training rows only**. The export includes molecule IDs and scaffold-split labels so returned quantum features can be aligned by identity, never by row order.

## Ranking

`rank_score = 0.65 activity + 0.20 CNS-likeness proxy + 0.15 drug-likeness proxy`.

This ranking is intentionally transparent and intentionally limited. CNS-likeness is a rule-based proxy over MW/logP/TPSA/HBD/rotatable bonds; it is not a BBB model. Replacing these proxies with externally validated predictors is a later roadmap item.

## Interpretation

Even a successful LRRK2 activity classifier only addresses target-level molecular activity. It does not establish brain exposure, safety, disease modification, or clinical efficacy.
