.PHONY: install test demo serve repeat
install:
	pip install -e '.[dev]'
test:
	pytest
demo:
	pdl demo
serve:
	PDL_ARTIFACT_DIR=artifacts/demo uvicorn parkinson_discovery.api:app --reload
repeat:
	pdl repeat --input artifacts/demo/dataset_prepared.csv --out artifacts/repeated_demo
