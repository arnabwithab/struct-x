#!/usr/bin/env bash
# Overnight supervisor: ONE tick (parent loops it every 15 min under flock).
# Rules: advance the Kaggle queue only on COMPLETE + valid results JSON.
# On ERROR/CANCEL: log loudly, do nothing. Read-only on remote (no kills).
# State is derived from disk every tick (idempotent, restart-safe).
# Usage: scripts/nightly_watch.sh [--tick]   (default: one tick)
set -u
QDIR=/home/arnab/code/collegeCode/struct-x
cd "$QDIR" || exit 1
log() { echo "[$(date -u +%H:%M)] $*"; }

# ---------- KAGGLE ----------
ST=$(timeout 60 kaggle kernels status eros483/struct-x-train 2>&1 | grep -oE "COMPLETE|RUNNING|ERROR|CANCEL[A-Z_]*|QUEUED" | head -1)
log "kaggle=$ST"
NEXT=""
for t in b9 amp surface; do
  if [ ! -f "outputs/$t/results_On_set1.json" ]; then NEXT=$t; break; fi
done
log "next-needed=$NEXT"
if [ "$ST" = "COMPLETE" ] && [ -n "$NEXT" ]; then
  mkdir -p "outputs/$NEXT"
  OUT=$(timeout 500 kaggle kernels output "eros483/struct-x-train" -p "outputs/$NEXT" 2>&1 | grep -vE "^Warning" | head -3)
  log "pull: $OUT"
  if python3 -c "import json;d=json.load(open('outputs/$NEXT/results_On_set1.json'));assert d['results']['data_only'] and d['results']['both']" 2>/dev/null; then
    cp "outputs/$NEXT/results_On_set1.json" "results/${NEXT}_On_set1.json"
    git add "results/${NEXT}_On_set1.json" && git commit -qm "results: $NEXT On_set1 (data_only + both)" && git push -q origin main 2>&1 | tail -1
    log "$NEXT pulled+committed."
    if [ "$NEXT" = "surface" ]; then
      log "QUEUE COMPLETE (b9+amp+surface all in). F014 ckpt exfil is morning work."
    else
      case $NEXT in
        b9)      B=5; V="data_only,both"; A=1; R=voxel;      FOL=amp ;;
        amp)     B=5; V="data_only,both"; A=0; R=timesurface; FOL=surface ;;
      esac
      if [ -z "$(git status --porcelain | head -1)" ]; then
        python3 - "$B" "$V" "$A" "$R" <<'PYEOF'
import sys
b, v, a, r = sys.argv[1:5]
p = "kernel/train.py"
s = open(p).read()
s = s.replace('B = int(os.environ.get("STRUCTX_B", "5"))',
              'B = int(os.environ.get("STRUCTX_B", "%s"))' % b)
s = s.replace('"data_only,smoothness_only,sdof_only,both"', '"%s"' % v)
s = s.replace('USE_AMP = os.environ.get("STRUCTX_USE_AMP", "0")',
              'USE_AMP = os.environ.get("STRUCTX_USE_AMP", "%s")' % a)
s = s.replace('REPR = os.environ.get("STRUCTX_REPR", "voxel")',
              'REPR = os.environ.get("STRUCTX_REPR", "%s")' % r)
open(p, "w").write(s)
print("knobs set", b, v, a, r)
PYEOF
        python3 scripts/stamp_kernel.py train >/dev/null && kaggle kernels push -p kernel --accelerator gpu -t 43200 2>&1 | grep -oE "version [0-9]+ successfully pushed|error.*" | head -1
        git checkout -q -- kernel/train.py
        git status --porcelain | head -2
        log "fired $FOL."
      else
        log "TREE DIRTY - push skipped, needs human."
      fi
    fi
  else
    log "RESULTS JSON INVALID or pull failed - not advancing."
  fi
elif echo "$ST" | grep -qE "ERROR|CANCEL"; then
  log "ALERT: kernel $ST"
  timeout 90 kaggle kernels logs eros483/struct-x-train 2>&1 | tail -4 | cut -c1-160
fi

# ---------- REMOTE (read-only) ----------
RMT=$(timeout 60 ssh -o ConnectTimeout=15 -o BatchMode=yes snu@100.112.181.46 'ls ~/sx-queue/ | grep DONE; echo ---; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; echo ---; tail -1 ~/sx-queue/logs/JOB*.log 2>/dev/null | cut -c1-120' 2>&1 | head -8)
log "remote: $(echo "$RMT" | tr '\n' '|')"
