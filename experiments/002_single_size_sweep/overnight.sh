#!/usr/bin/env bash
# Sequential unattended runner for experiment 002 (adapted from 001 overnight.sh, much simpler: one GPU job at a time).
#   bash overnight.sh --dry-run          plan + environment check, nothing executed
#   nohup bash overnight.sh > logs/overnight.log 2>&1 &
# Steps per run (conditions in $CONDITIONS, default = config conditions.run_order[0]; seeds from config train.seeds):
#   train(each seed) -> cache(once) -> q1(once) -> q2(each) -> q3(each) -> aggregate
# Every step is tried at most 3 times (CLAUDE.md); each attempt writes logs/overnight_<item>_<stage>_try<N>.log and the last line of the
# failure is stored in logs/overnight_status.json (stderr of the status helper goes to logs/overnight_status.err and is also echoed).
# A step that fails 3 times stops the whole chain (later steps depend on it) and the exit code is 1. Nothing is ever forced (no --force).
# Environment: assumes an activated conda env; if $CONDA_ENV is set and differs from CONDA_DEFAULT_ENV the script sources conda.sh and
# activates it (`conda run` was not chosen: it buffers output, so progress would not be visible in the logs).
# Variables: CONDA_ENV  CONDITIONS  RETRY_SLEEP(60)  DINO_WORK DATASET_DIR DINO_TRAIN EXP002_CACHE ACWM_PHYS_SRC (see README)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE" || exit 1
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
ts() { date +%Y-%m-%dT%H:%M:%S%z; }; say() { echo "[$(ts)] $*"; }
mkdir -p "$HERE/logs" "$HERE/results"
export DINO_WORK="${DINO_WORK:-$HOME/dinowm}"; export DATASET_DIR="${DATASET_DIR:-$DINO_WORK/data}"; export DINO_TRAIN="${DINO_TRAIN:-$DINO_WORK/train_runs}"
export EXP002_CACHE="${EXP002_CACHE:-$DINO_WORK/exp002_cache}"; export ACWM_PHYS_SRC="${ACWM_PHYS_SRC:-$DINO_WORK/acwm_phys_src}"
export WANDB_MODE="${WANDB_MODE:-disabled}"; export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
RETRY_SLEEP="${RETRY_SLEEP:-60}"; ATTEMPTS=3
if [ -n "${CONDA_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  if command -v conda >/dev/null 2>&1; then source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" || say "WARNING cannot activate conda env $CONDA_ENV"
  else say "WARNING conda not found; using $(command -v python)"; fi
fi
[ -z "${CONDA_DEFAULT_ENV:-}" ] && say "WARNING: CONDA_DEFAULT_ENV is empty (no conda env active). python=$(command -v python)"
say "python=$(command -v python) CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
PY=python; RUN="$PY run.py"
CONDITIONS="${CONDITIONS:-$($PY - <<'P'
import sys; sys.path.insert(0, "src")
from exp002.common import load_config
print(load_config()["conditions"]["run_order"][0])
P
)}"
SEEDS="$($PY - <<'P'
import sys; sys.path.insert(0, "src")
from exp002.common import load_config
print(" ".join(str(s) for s in load_config()["train"]["seeds"]))
P
)"
STATUS="$HERE/logs/overnight_status.json"
st() { $PY "$HERE/overnight_status.py" set --file "$STATUS" "$@" >/dev/null 2>>"$HERE/logs/overnight_status.err" || say "WARNING status update failed (see logs/overnight_status.err): $(tail -n 1 "$HERE/logs/overnight_status.err" 2>/dev/null)"; }
lastline() { grep -v '^[[:space:]]*$' "$1" 2>/dev/null | tail -n 1 | cut -c1-400; }
ITEMS=""; for c in $CONDITIONS; do for s in $SEEDS; do ITEMS="$ITEMS exp002_${c}_s${s}"; done; done; ITEMS="${ITEMS# } global"
say "conditions: $CONDITIONS ; seeds: $SEEDS ; items: $ITEMS"
if [ "$DRY" = 1 ]; then
  say "plan: for each condition/seed: train; then cache, q1 (global); then q2, q3 per run; then aggregate. Each step <= $ATTEMPTS attempts."
  $RUN check --dry-run || true; for c in $CONDITIONS; do $RUN train --condition "$c" --dry-run || true; done
  exit 0
fi
$PY "$HERE/overnight_status.py" init --file "$STATUS" --items "$ITEMS" --meta "conditions=$CONDITIONS" "start=$(ts)" 2>>"$HERE/logs/overnight_status.err" || say "WARNING status init failed: $(tail -n 1 "$HERE/logs/overnight_status.err")"

retry_step() {  # retry_step ITEM STAGE MARKER|- cmd...
  local item="$1" stage="$2" marker="$3"; shift 3
  if [ "$marker" != "-" ] && [ -e "$marker" ]; then st "$item" "$stage" done --note "already existed: $marker"; say "$item $stage: already done"; return 0; fi
  local a lg rc err=""
  for ((a = 1; a <= ATTEMPTS; a++)); do
    st "$item" "$stage" running --retries $((a - 1))
    lg="$HERE/logs/overnight_${item}_${stage}_try${a}.log"; say "$item $stage: attempt $a/$ATTEMPTS -> $lg"
    # Resume (확정 기록 10, 사용자 정정 2026-11-04): before EVERY train attempt (also retries) `prepare` keeps a loadable checkpoint (resume), repairs a broken
    # model_latest.pth from the newest loadable epoch file, and only when nothing loadable exists moves checkpoints to <DINO_TRAIN>/quarantine/ (fresh start).
    # If `prepare` fails the attempt fails (rc 97). The patched train.py (colab/train_resume.patch) is required for resume; trainrun.py refuses otherwise.
    ( if [ "$stage" = train ] && [ -f "$HERE/colab/colab_tools.py" ]; then "$PY" "$HERE/colab/colab_tools.py" prepare --train-root "$DINO_TRAIN" --run "$item" || exit 97; fi; "$@" ) > "$lg" 2>&1; rc=$?
    if [ "$rc" = 0 ]; then st "$item" "$stage" done --retries $((a - 1)); say "$item $stage: done"; return 0; fi
    err="rc=$rc: $(lastline "$lg")"; say "$item $stage: attempt $a failed ($err)"
    st "$item" "$stage" running --retries "$a" --error "$err"
    [ "$a" -lt "$ATTEMPTS" ] && sleep "$RETRY_SLEEP"
  done
  st "$item" "$stage" failed --retries "$ATTEMPTS" --error "$err"; return 1
}
fail() { say "STOP: $1 failed $ATTEMPTS times. Read logs/overnight_*_try*.log and report the cause (CLAUDE.md retry limit)."; exit 1; }

for c in $CONDITIONS; do for s in $SEEDS; do
  n="exp002_${c}_s${s}"
  retry_step "$n" train "$DINO_TRAIN/outputs/$n/DONE" $RUN train --condition "$c" --seeds "$s" || fail "$n train"
done; done
retry_step global cache "$EXP002_CACHE/q1_tokens/DONE" $RUN cache || fail "cache"
retry_step global q1 "$HERE/results/q1/q1_summary.json" $RUN q1 || fail "q1"
for c in $CONDITIONS; do for s in $SEEDS; do
  n="exp002_${c}_s${s}"
  retry_step "$n" q2 "$HERE/results/$n/q2_summary.json" $RUN q2 --condition "$c" --seeds "$s" || fail "$n q2"
  retry_step "$n" q3 "$HERE/results/$n/q3.json" $RUN q3 --condition "$c" --seeds "$s" || fail "$n q3"
done; done
retry_step global aggregate - $RUN aggregate || fail "aggregate"
$PY "$HERE/overnight_status.py" summary --file "$STATUS" > "$HERE/logs/overnight_summary.txt" 2>>"$HERE/logs/overnight_status.err"
say "finished. See $STATUS, $HERE/results.json"
