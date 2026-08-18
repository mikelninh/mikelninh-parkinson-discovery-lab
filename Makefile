.PHONY: install test demo verify-demo serve repeat freeze-lrrk2
install:
	pip install -e '.[dev]'
test:
	pytest
demo:
	pdl demo
verify-demo:
	pdl verify --manifest artifacts/demo/manifest.json
serve:
	PDL_ARTIFACT_DIR=artifacts/demo uvicorn parkinson_discovery.api:app --reload
repeat:
	pdl repeat --input artifacts/demo/dataset_prepared.csv --out artifacts/repeated_demo
freeze-lrrk2:
	pdl freeze-chembl --target LRRK2 --out data/snapshots/lrrk2
