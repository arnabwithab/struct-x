#!/usr/bin/env bash
# Poll remote GPU queue: 60s ticks x45, then 30-min ticks. Stops on DONE_JOB8.
LOG=$HOME/sx-poll-remote.log
echo "$(date -u): remote poller start" >> "$LOG"
for i in $(seq 1 200); do
  OUT=$(timeout 60 ssh -o ConnectTimeout=15 -o BatchMode=yes snu@100.112.181.46 \
    'ls ~/sx-queue/ | grep -E "DONE" | tr "\n" " "; echo; tail -1 ~/sx-queue/logs/queue.log 2>/dev/null | cut -c1-120; echo; tail -1 ~/sx-queue/logs/JOB*.log 2>/dev/null | cut -c1-120; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null' 2>&1)
  echo "$(date -u) tick=$i :: $(echo "$OUT" | tr '\n' ' | ' | cut -c1-400)" >> "$LOG"
  case "$OUT" in
    *DONE_JOB8_F014rerun*) echo "$(date -u): JOB8 DONE, exiting" >> "$LOG"; break ;;
    *Traceback*|*OOM*|*FATAL*|*FAILED*) echo "$(date -u): FAILURE-SIGNATURE, continuing" >> "$LOG" ;;
  esac
  if [ "$i" -lt 45 ]; then sleep 60; else sleep 1800; fi
done
