.PHONY: help setup download preprocess smoke train eval ridge test style clean kernel-smoke kernel-train kernel-timing mini-e2e f014-layers3
.DEFAULT_GOAL := help

help: ## show targets
	@grep -E '^[a-z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'

setup: ## installs all dependencies (uv sync)
	command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
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

ridge: ## RidgeCV linear baseline, LOCO folds, same Delta/se/O5 table
	uv run python scripts/ridge_baseline.py

f014-layers3: ## F014-3 probes + latent stability (inference-only, batch-1 GPU)
	uv run python scripts/f014_layers3.py

mini-e2e: ## CPU miniature end-to-end smoke (synthetic, one fold, few epochs)
	uv run python scripts/mini_e2e.py

test: ## runs all unit tests
	uv run pytest -q

style: ## formats + lints all code
	uv run black src tests scripts kernel
	uv run ruff check src tests scripts kernel

kernel-smoke: ## push smoke kernel (env log + inventory + shape smoke)
	python3 scripts/stamp_kernel.py smoke && kaggle kernels push -p kernel

kernel-train: ## push training kernel (LOCO campaign)
	python3 scripts/stamp_kernel.py train && kaggle kernels push -p kernel

kernel-timing: ## push timing micro-kernel (15a ms/window)
	python3 scripts/stamp_kernel.py timing && kaggle kernels push -p kernel

clean: ## removes build artifacts, caches, __pycache__ etc.
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true
