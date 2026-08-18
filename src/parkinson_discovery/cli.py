from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import uvicorn

from .chembl import build_target_dataset
from .config import SplitConfig
from .demo_data import make_demo_dataset
from .kipu_service import run_managed_service
from .pipeline import run_pipeline
from .quantum import compare_rimay_result, create_rimay_pilot
from .repeats import DEFAULT_SEEDS, run_repeated_scaffold_benchmark


def _seed_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(prog="pdl", description="Parkinson Discovery Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Generate synthetic chemistry demo and run full pipeline")
    demo.add_argument("--out", default="artifacts/demo")
    demo.add_argument("--seed", type=int, default=42)

    fetch = sub.add_parser("fetch-chembl", help="Pull and clean LRRK2 target data from ChEMBL")
    fetch.add_argument("--target", default="LRRK2")
    fetch.add_argument("--out", default="data/lrrk2_chembl.csv")
    fetch.add_argument("--standard-types", default="IC50,Ki", help="Comma-separated ChEMBL standard types; use IC50 for a stricter single-assay-type run")

    run = sub.add_parser("run", help="Run benchmark on a prepared labeled CSV")
    run.add_argument("--input", required=True)
    run.add_argument("--out", default="artifacts/run")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--features", type=int, default=96)

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
        types = tuple(x.strip() for x in args.standard_types.split(",") if x.strip())
        df = build_target_dataset(args.target, standard_types=types)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Wrote {len(df)} cleaned molecules to {path}")
    elif args.command == "run":
        result = run_pipeline(pd.read_csv(args.input), Path(args.out), descriptor_count=args.features, split_config=SplitConfig(seed=args.seed))
        print(result["manifest"])
    elif args.command == "repeat":
        result = run_repeated_scaffold_benchmark(pd.read_csv(args.input), Path(args.out), descriptor_count=args.features, seeds=args.seeds)
        print(json.dumps(result, indent=2))
    elif args.command == "rimay-pilot":
        result = create_rimay_pilot(Path(args.input), Path(args.out), args.size, args.seed)
        print(json.dumps(result, indent=2))
    elif args.command == "rimay-compare":
        result = compare_rimay_result(Path(args.prepared), Path(args.rimay_result), Path(args.baseline), Path(args.out))
        print(json.dumps(result, indent=2))
    elif args.command == "kipu-run":
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = run_managed_service(args.endpoint, request, Path(args.out), args.timeout)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "serve":
        import os
        os.environ["PDL_ARTIFACT_DIR"] = args.artifacts
        uvicorn.run("parkinson_discovery.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
