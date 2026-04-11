#!/bin/bash
# Sequential K=3200 P=6 LinearSVC sweep.
#
# Waits for the currently running run_06 (PID passed as $1) to finish, then
# runs C=0.003 and C=0.03 in sequence — each as a fresh python process so that
# all memory is fully released between fits (previous attempt OOM'd because of
# concurrent working sets).
#
# Usage:
#   nohup bash scripts/sequential_k3200.sh <CURRENT_PID> > logs/sequential_k3200.log 2>&1 &

set -u
CURRENT_PID=${1:?usage: sequential_k3200.sh <current_pid>}

cd "$(dirname "$0")/.."
source /home/yitongli/miniconda3/etc/profile.d/conda.sh
conda activate comp3314

echo "[$(date '+%F %T')] waiting for current run_06 PID $CURRENT_PID ..."
while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date '+%F %T')] PID $CURRENT_PID gone; starting sequential C sweep"

for C in 0.003 0.03; do
  LOG="logs/run_06_k3200_C${C}.stdout.log"
  echo "[$(date '+%F %T')] >>> START C=$C  log=$LOG"
  python -u runs/run_06_k3200.py --C "$C" > "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date '+%F %T')] <<< END   C=$C  rc=0"
  else
    echo "[$(date '+%F %T')] <<< FAIL  C=$C  rc=$rc  (continuing)"
  fi
done

touch logs/run_06_sequential_done.marker
echo "[$(date '+%F %T')] ALL SEQUENTIAL DONE"
