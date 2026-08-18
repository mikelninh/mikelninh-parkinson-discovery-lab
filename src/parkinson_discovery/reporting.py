from __future__ import annotations

from pathlib import Path

import pandas as pd


def _metric_rows(metrics: dict, section: str) -> str:
    rows = []
    for name, m in metrics[section].items():
        rows.append(
            f"| {name} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['f1']:.3f} | "
            f"{m['accuracy']:.3f} | {m['brier']:.3f} |"
        )
    return "\n".join(rows)


def _candidate_table(ranked: pd.DataFrame) -> str:
    lines = [
        "| rank | molecule | predicted activity | CNS proxy | rank score |",
        "|---:|---|---:|---:|---:|",
    ]
    for _, r in ranked.head(10).iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['molecule_id']} | {r['predicted_activity']:.3f} | "
            f"{r['cns_likeness_proxy']:.3f} | {r['rank_score']:.3f} |"
        )
    return "\n".join(lines)


def write_report(df: pd.DataFrame, metrics: dict, ranked: pd.DataFrame, out_dir: Path) -> Path:
    report = f"""# Parkinson Discovery Lab — run report

## Dataset

- Molecules: **{len(df)}**
- Active: **{int(df['active_label'].sum())}**
- Inactive: **{int((1-df['active_label']).sum())}**
- Unique scaffolds: **{df['scaffold'].nunique()}**
- Split: {df['split'].value_counts().to_dict()}

## Model selection — validation scaffolds only

Chosen model: **{metrics['best_model']}**

Selection rule: {metrics['selection_rule']}.

| model | ROC-AUC | PR-AUC | F1 | accuracy | Brier ↓ |
|---|---:|---:|---:|---:|---:|
{_metric_rows(metrics, 'validation')}

## Final untouched test benchmark

After model-family selection, each baseline is refit on train + validation and evaluated on the test scaffolds.

| model | ROC-AUC | PR-AUC | F1 | accuracy | Brier ↓ |
|---|---:|---:|---:|---:|---:|
{_metric_rows(metrics, 'test')}

## Top computational hypotheses

{_candidate_table(ranked)}

## Interpretation boundary

These ranks are **computational hypotheses, not drugs and not evidence of clinical efficacy**. The CNS and drug-likeness values are transparent heuristic proxies, not validated ADMET/BBB predictions. A real discovery programme requires orthogonal computational validation, experimental assays, selectivity, PK/PD, toxicology and clinical studies.

## Quantum comparison

`rimay_input.csv` contains the same molecules, labels and scaffold split plus 96 standardized numerical features. A quantum result is only meaningful if Kipu/Rimay follows the same no-leakage protocol and is evaluated on the untouched test scaffolds against the classical benchmark above. No quantum-advantage claim is made by this repository alone.
"""
    path = out_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    return path
