.PHONY: help setup download preprocess smoke train eval test style clean
.DEFAULT_GOAL := help

help: ## show targets
	@grep -E '^[a-z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'

setup: ## installs all dependencies (uv sync)
	uv sync --group dev

download: ## fetches XJTU-DV-Beam.zip, verifies md5, prints inventory
	bash scripts/download.sh

preprocess: ## builds voxels + window index (trial-level, no leakage)
	uv run python scripts/preprocess.py --config configs/base.yaml

smoke: ## shape/memory smoke test (encoder -> heads, logs max_memory)
	uv run python -c "import torch; from models.resnet3d import *; print(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)"

train: ## LOCO training, all O4 ablations
	bash scripts/run_folds.sh

eval: ## TEST-trial metrics, Delta/sigma estimators, O5 verdict
	uv run python -m evaluate --config configs/base.yaml

test: ## runs all unit tests
	uv run pytest -q

style: ## formats + lints all code
	uv run black src tests scripts
	uv run ruff check src tests scripts

clean: ## removes build artifacts, caches, __pycache__ etc.
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true
