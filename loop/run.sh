#!/usr/bin/env bash
# loop/run.sh — THE DRIVER. Runs bounded agent iterations until the goal predicate passes.
#   nohup bash loop/run.sh > loop/loop.log 2>&1 &
#   # or via tmux/systemd (see README) for true 24/7.
set -u
REPO=/home/austin-whaley/Documents/wonderland
MODEL="${LOOP_MODEL:-openrouter/deepseek/deepseek-v4.1-flash}"   # provider/model
MAX_ITERS="${LOOP_MAX_ITERS:-500}"
MAX_HOURS="${LOOP_MAX_HOURS:-24}"
ITER=0
START=$(date +%s)
mkdir -p "$REPO/loop"

log(){ echo "[$(date -u +%FT%TZ)] $*"; }

log "driver start model=$MODEL max_iters=$MAX_ITERS max_hours=$MAX_HOURS"
while :; do
  ITER=$((ITER+1))
  ELAPSED_H=$(( ($(date +%s) - START) / 3600 ))
  if [ "$ITER" -gt "$MAX_ITERS" ] || [ "$ELAPSED_H" -ge "$MAX_HOURS" ]; then
    log "CAP reached iter=$ITER elapsed_h=$ELAPSED_H — stopping NOT_MET"; exit 2
  fi
  if bash "$REPO/loop/status.sh" > "$REPO/loop/status.out" 2>&1; then
    log "GOAL_MET after $ITER iterations"; cat "$REPO/loop/status.out"; exit 0
  fi
  log "iter=$ITER status=NOT_MET — launching agent run"
  # bounded, headless, auto-approved. Each run does ONE step and commits.
  opencode run --auto --dir "$REPO" -m "$MODEL" "$(cat "$REPO/loop/ITER.md")" \
    >> "$REPO/loop/agent.log" 2>&1 || log "agent run exited nonzero (continuing)"
  log "iter=$ITER agent run done"
  sleep 5
done
