#!/usr/bin/env bash
# LOCO folds x O4 ablations driver. See design sec 7-8, 10.
set -euo pipefail
uv run python -m structx.train --config configs/base.yaml "$@"
