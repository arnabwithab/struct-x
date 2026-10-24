#!/usr/bin/env bash
# Struct-X autonomous training campaign (F010). Resumable: each stage checks
# whether its outputs already exist before running.
# Stage 1: pull v4 pilot outputs. Stage 2: push v5 (5 folds), wait, pull.
# Stage 3: version checkpoints to dataset. Stage 4: verdict + tracker + push.
set -u
cd "$(dirname "$0")/.."

wait_for_kernel() { # $1 = slug
  for _ in $(seq 1 480); do
    st=$(kaggle kernels status "$1" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*' | tail -n 1)
    echo "$(date +%H:%M) $1 $st"
    case "$st" in
      *COMPLETE*) return 0 ;;
      *ERROR*|*FAILED*|*CANCELLED*) return 1 ;;
    esac
    sleep 60
  done
  return 2
}

kaggle quota 2>/dev/null | tail -n 3

mkdir -p outputs
if [ ! -f outputs/pilot_results.json ]; then
  wait_for_kernel eros483/struct-x-train || { echo "v4 not complete; aborting"; exit 1; }
  kaggle kernels output eros483/struct-x-train -p ./outputs
  find outputs -name "pilot_results.json" -o -name "ckpts" -maxdepth 2 | head
fi
ls outputs/

echo "=== pushing v5 (5 remaining folds) ==="
kaggle kernels push -p kernel-structx-train | tail -n 1
wait_for_kernel eros483/struct-x-train || { echo "v5 failed; see log"; exit 1; }
rm -rf outputs/v5 && mkdir -p outputs/v5
kaggle kernels output eros483/struct-x-train -p ./outputs/v5
ls outputs/v5/

echo "=== ALLDONE (manual: version ckpts, verdict, push) ==="
