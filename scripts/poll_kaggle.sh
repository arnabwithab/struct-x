#!/usr/bin/env bash
# Poll Kaggle amp-redo kernel: 60s ticks x45, then 30-min ticks. Stops on COMPLETE/ERROR.
LOG=$HOME/sx-poll-kaggle.log
SLUG=eros438/struct-x-train
echo "$(date -u): kaggle poller start ($SLUG)" >> "$LOG"
for i in $(seq 1 200); do
  ST=$(timeout 60 kaggle kernels status "$SLUG" 2>&1 | grep -oE "COMPLETE|RUNNING|ERROR|CANCEL[A-Z_]*|QUEUED" | head -1)
  echo "$(date -u) tick=$i status=${ST:-FETCH-FAIL}" >> "$LOG"
  case "$ST" in
    COMPLETE|ERROR|CANCEL*)
      timeout 90 kaggle kernels logs "$SLUG" 2>&1 | grep -v Warning | tail -15 | cut -c1-160 >> "$LOG"
      echo "$(date -u): TERMINAL $ST, exiting" >> "$LOG"
      break ;;
  esac
  if [ "$i" -lt 45 ]; then sleep 60; else sleep 1800; fi
done
