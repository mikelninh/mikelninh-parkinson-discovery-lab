from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import uvicorn

from .assay_context import write_assay_context_audit
from .chembl import build_target_dataset, freeze_target_snapshot
from .config import SplitConfig
from .demo_data import make_demo_dataset
from .kipu_service import run_managed_service
from .pipeline import run_pipeline
from .provenance import verify_manifest
from .quantum import compare_rimay_result, create_rimay_pilot
from .repeats import DEFAULT_SEEDS, run_repeated_scaffold_benchmark


def _seed_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from exc


def _standard_types(value: str) -> tuple[str, ...]:
    types = tuple(x.strip() for x in value.split(",") if x.strip())
    if not types:
        raise argparse.ArgumentTypeError("At least one ChEMBL standard type is required")
    return types


def main() -> None:
    parser = argparse.ArgumentParser(prog="pdl", description="Parkinson Discovery Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Generate synthetic chemistry demo and run full pipeline")
    demo.add_argument("--out", default="artifacts/demo")
    demo.add_argument("--seed", type=int, default=42)

    fetch = sub.add_parser("fetch-chembl", help="Pull and clean LRRK2 target data from ChEMBL")
    fetch.add_argument("--target", default="LRRK2")
    fetch.add_argument("--out", default="data/lrrk2_chembl.csv")
    fetch.add_argument(
        "--standard-types",
        type=_standard_types,
        default=("IC50", "Ki"),
        help="Comma-separated ChEMBL standard types; use IC50 for a stricter sensitivity run",
    )
    fetch.add_argument("--max-pchembl-iqr", type=float, default=None)
    fetch.add_argument("--min-label-agreement", type=float, default=0.0)

    freeze = sub.add_parser(
        "freeze-chembl",
        help="Create a versioned ChEMBL source snapshot with raw records, assay context and SHA-256 hashes",
    )
    freeze.add_argument("--target", default="LRRK2")
    freeze.add_argument("--out", default="data/snapshots/lrrk2")
    freeze.add_argument("--standard-types", type=_standard_types, default=("IC50", "Ki"))
    freeze.add_argument("--max-pchembl-iqr", type=float, default=None)
    freeze.add_argument("--min-label-agreement", type=float, default=0.0)

    audit = sub.add_parser("assay-audit", help="Summarize assay heterogeneity and label-quality metadata")
    audit.add_argument("--input", required=True)
    audit.add_argument("--out", default="artifacts/assay_context_summary.json")

    run = sub.add_parser("run", help="Run benchmark on a prepared labeled CSV")
    run.add_argument("--input", required=True)
    run.add_argument("--out", default="artifacts/run")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--features", type=int, default=96)
    run.add_argument(
        "--source-manifest",
        default=None,
        help="Optional snapshot_manifest.json produced by pdl freeze-chembl",
    )

    repeat = sub.add_parser("repeat", help="Repeat classical benchmark across scaffold split seeds")
    repeat.add_argument("--input", required=True)
    repeat.add_argument("--out", default="artifacts/repeated")
    repeat.add_argument("--features", type=int, default=96)
    repeat.add_argument("--seeds", type=_seed_tuple, default=DEFAULT_SEEDS)

    pilot = sub.add_parser("rimay-pilot", help="Build 200–500 molecule Rimay simulator handoff bundle")
    pilot.add_argument("--input", required=True, help="rimay_input.csv from a PDL run")
    pilot.add_argument("--out", default="artifacts/rimay_pilot")
    pilot.add_argument("--size", type=int, default=300)
    pilot.add_argument("--seed", type=int, default=42)

    compare = sub.add_parser("rimay-compare", help="Compare returned Rimay features/predictions to baseline")
    compare.add_argument("--prepared", required=True, help="dataset_prepared.csv")
    compare.add_argument("--rimay-result", required=True)
    compare.add_argument("--baseline", required=True, help="metrics.json")
    compare.add_argument("--out", default="artifacts/run/quantum_comparison.json")

    kipu = sub.add_parser("kipu-run", help="Invoke a subscribed Kipu managed service using official SDK")
    kipu.add_argument("--endpoint", required=True, help="Gateway endpoint from your Kipu application subscription")
    kipu.add_argument("--request", required=True, help="JSON request matching the subscribed service OpenAPI schema")
    kipu.add_argument("--out", default="artifacts/kipu_execution.json")
    kipu.add_argument("--timeout", type=int, default=900)

    verify = sub.add_parser("verify", help="Verify a frozen snapshot or run manifest byte-for-byte")
    verify.add_argument("--manifest", required=True)

    serve = sub.add_parser("serve", help="Serve dashboard/API for an artifact directory")
    serve.add_argument("--artifacts", default="artifacts/demo")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command == "demo":
        df = make_demo_dataset()
        result = run_pipeline(df, Path(args.out), split_config=SplitConfig(seed=args.seed))
        print(result["manifest"])
    elif args.command == "fetch-chembl":
        df = build_target_dataset(
            args.target,
            standard_types=args.standard_types,
            max_pchembl_iqr=args.max_pchembl_iqr,
            min_label_agreement=args.min_label_agreement,
        )
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Wrote {len(df)} cleaned molecules to {path}")
    elif args.command == "freeze-chembl":
        result = freeze_target_snapshot(
            args.target,
            Path(args.out),
            standard_types=args.standard_types,
            max_pchembl_iqr=args.max_pchembl_iqr,
            min_label_agreement=args.min_label_agreement,
        )
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "assay-audit":
        result = write_assay_context_audit(pd.read_csv(args.input), Path(args.out))
        print(json.dumps(result, indent=2))
    elif args.command == "run":
        input_path = Path(args.input)
        result = run_pipeline(
            pd.read_csv(input_path),
            Path(args.out),
            descriptor_count=args.features,
            split_config=SplitConfig(seed=args.seed),
            input_path=input_path,
            source_manifest=Path(args.source_manifest) if args.source_manifest else None,
        )
        print(result["manifest"])
    elif args.command == "repeat":
        result = run_repeated_scaffold_benchmark(
            pd.read_csv(args.input),
            Path(args.out),
            descriptor_count=args.features,
            seeds=args.seeds,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "rimay-pilot":
        result = create_rimay_pilot(Path(args.input), Path(args.out), args.size, args.seed)
        print(json.dumps(result, indent=2))
    elif args.command == "rimay-compare":
        result = compare_rimay_result(
            Path(args.prepared), Path(args.rimay_result), Path(args.baseline), Path(args.out)
        )
        print(json.dumps(result, indent=2))
    elif args.command == "kipu-run":
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = run_managed_service(args.endpoint, request, Path(args.out), args.timeout)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "verify":
        result = verify_manifest(Path(args.manifest))
        print(json.dumps(result, indent=2))
        if not result["ok"]:
            raise SystemExit(2)
    elif args.command == "serve":
        import os

        os.environ["PDL_ARTIFACT_DIR"] = args.artifacts
        uvicorn.run("parkinson_discovery.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
