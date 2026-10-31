#!/usr/bin/env bash
# Unattended (overnight) run of experiment 001: trains baseline_single seeds 1 and 2 (seed 0 is expected to be training already),
# then runs repro_check -> extract -> diagnose -> probe -> intervene -> aggregate+sync for every seed, seed 0 first.
#
#   tmux:     bash overnight.sh
#   nohup:    nohup bash overnight.sh > logs/overnight.log 2>&1 &        (mkdir -p logs first)
#   plan only: bash overnight.sh --dry-run
#
# Status for the morning: logs/overnight_status.json (per seed/stage state, times, retries, last error), logs/overnight_summary.txt, results.json.
# Environment variables (all optional; defaults in brackets):
#   TRAIN_PARALLEL [1]  1 = up to 2 trainings on the GPU at the same time (an externally running seed-0 training counts as one), 0 = one at a time
#   WAIT_MAX_HOURS [16] max time to wait for an externally running training (seed 0) to produce DONE; then that seed is marked failed (the training is NOT touched)
#   POLL_SEC [120]      polling interval of the scheduler     GRACE_SEC [180]  an externally running training must be absent this long before we take over
#   HEARTBEAT_MIN [45]  a run directory with a file modified in the last N minutes counts as "still training" (second signal besides the process check)
#   RETRY_SLEEP [60]    seconds between attempts of a failed stage     (attempts per stage: 3, CLAUDE.md rule)
#   OVERNIGHT_SEEDS ["0 1 2"]   CONDA_ENV [dino_wm; a path such as /venv/dino_wm also works]   OVERNIGHT_SKIP_ACTIVATE=1 uses the current python
#   OVERNIGHT_PY [python]  python used for run.py.   DINO_WORK/DATASET_DIR/DINO_TRAIN/EXP001_CACHE/ACWM_PHYS_SRC as in run_all.sh (~/exp001_env.sh is sourced if present)
# Never passes --force, never writes into the run directory of a training it did not start, never kills a process it did not start.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1 ;;
  -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
  *) echo "unknown option $1"; exit 2 ;; esac; shift; done

ts() { date +%Y-%m-%dT%H:%M:%S%z; }
say() { echo "[$(ts)] $*"; }
WARN=0
warn() { WARN=$((WARN + 1)); say "WARNING: $*"; }

# ---------------------------------------------------------------- environment
if [ -f "$HOME/exp001_env.sh" ]; then
  set +u; # shellcheck disable=SC1091
  source "$HOME/exp001_env.sh"; set -u; say "sourced $HOME/exp001_env.sh"
else say "no $HOME/exp001_env.sh (using defaults/inherited variables)"; fi
SEEDS="${OVERNIGHT_SEEDS:-0 1 2}"
DIST="baseline_single"
TRAIN_PARALLEL="${TRAIN_PARALLEL:-1}"
WAIT_MAX_HOURS="${WAIT_MAX_HOURS:-16}"; POLL_SEC="${POLL_SEC:-120}"; GRACE_SEC="${GRACE_SEC:-180}"
WAIT_MAX_SEC="${WAIT_MAX_SEC:-$((WAIT_MAX_HOURS * 3600))}"   # WAIT_MAX_SEC overrides (used by the test)
HEARTBEAT_MIN="${HEARTBEAT_MIN:-45}"; RETRY_SLEEP="${RETRY_SLEEP:-60}"; ATTEMPTS=3
MAX_CONC=1; [ "$TRAIN_PARALLEL" = 1 ] && MAX_CONC=2
export DINO_WORK="${DINO_WORK:-$HOME/dinowm}"
export DATASET_DIR="${DATASET_DIR:-$DINO_WORK/data}"
export DINO_TRAIN="${DINO_TRAIN:-$DINO_WORK/train_runs}"
export EXP001_CACHE="${EXP001_CACHE:-$DINO_WORK/exp001_cache}"
export ACWM_PHYS_SRC="${ACWM_PHYS_SRC:-$DINO_WORK/acwm_phys_src}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export EXP001_DATASET="${EXP001_DATASET:-cloth_sim}"
export EXP001_APPROVED="$DIST"                       # only the baseline is approved here (candidate F distributions are not)
MODELS=""; for s in $SEEDS; do MODELS="$MODELS ${DIST}:${s}"; done; MODELS="${MODELS# }"
export EXP001_MODELS="$MODELS"
PY="${OVERNIGHT_PY:-python}"
RUN="$PY run.py"
STATUS="$HERE/logs/overnight_status.json"
SUMMARY="$HERE/logs/overnight_summary.txt"
LOCKDIR="$HERE/logs/.locks"
T0=$(date +%s)

if [ "$DRY" = 0 ] && [ "${OVERNIGHT_SKIP_ACTIVATE:-0}" != 1 ]; then
  CONDA_ENV="${CONDA_ENV:-dino_wm}"
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" || warn "cannot activate conda env $CONDA_ENV; using the current python ($(command -v python))"
  else warn "conda not found; using the current python ($(command -v python))"; fi
fi

# ---------------------------------------------------------------- checks
check_env() {
  say "== environment check =="
  local v bad=0
  for v in DATASET_DIR EXP001_CACHE DINO_TRAIN; do
    local val="${!v:-}"
    if [ -z "$val" ]; then warn "$v is not set"; bad=1; continue; fi
    if printf '%s' "$val" | LC_ALL=C grep -q '[^!-~]'; then warn "$v contains a space, control or non-ASCII character: '$val'"; bad=1; fi
    say "$v=$val"
  done
  [ -d "$ACWM_PHYS_SRC" ] || warn "ACWM_PHYS_SRC=$ACWM_PHYS_SRC does not exist"
  [ -d "$DATASET_DIR/cloth_sim" ] || warn "$DATASET_DIR/cloth_sim does not exist (data not downloaded?)"
  [ -e "$HERE/manifests/train_${DIST}.json" ] || say "manifests/train_${DIST}.json missing -> 'run.py listing' will be run before training"
  # disk: need (estimates, README) ~1.7 GB checkpoints per seed not yet trained + ~5 GB extraction cache per seed
  local ntrain=0 s
  for s in $SEEDS; do [ -e "$DINO_TRAIN/outputs/exp001_${DIST}_s${s}/DONE" ] || ntrain=$((ntrain + 1)); done
  local nseeds; nseeds=$(echo "$SEEDS" | wc -w)
  local need_ckpt=$(( (ntrain * 17 + 9) / 10 )) need_cache=$((nseeds * 5))
  say "disk need (estimate): checkpoints ~${need_ckpt} GB (${ntrain} seed(s) not DONE x 1.7 GB), extraction cache ~${need_cache} GB (${nseeds} x 5 GB)"
  local d1 d2 a1 a2
  d1="$DINO_TRAIN"; d2="$EXP001_CACHE"
  while [ -n "$d1" ] && [ ! -d "$d1" ] && [ "$d1" != / ]; do d1="$(dirname "$d1")"; done
  while [ -n "$d2" ] && [ ! -d "$d2" ] && [ "$d2" != / ]; do d2="$(dirname "$d2")"; done
  df -h "$d1" "$d2" 2>&1 | sed 's/^/    /'
  a1=$(df -BG --output=avail "$d1" 2>/dev/null | tail -n 1 | tr -dc '0-9'); a2=$(df -BG --output=avail "$d2" 2>/dev/null | tail -n 1 | tr -dc '0-9')
  if [ -n "$a1" ] && [ -n "$a2" ]; then
    if [ "$(stat -c %d "$d1" 2>/dev/null)" = "$(stat -c %d "$d2" 2>/dev/null)" ]; then
      [ "$a1" -lt $((need_ckpt + need_cache)) ] && warn "free ${a1} GB < estimated need $((need_ckpt + need_cache)) GB (same filesystem)"
    else
      [ "$a1" -lt "$need_ckpt" ] && warn "DINO_TRAIN filesystem: free ${a1} GB < estimated ${need_ckpt} GB"
      [ "$a2" -lt "$need_cache" ] && warn "EXP001_CACHE filesystem: free ${a2} GB < estimated ${need_cache} GB"
    fi
  else warn "could not read free disk space"; fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv 2>&1 | sed 's/^/    /'
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1 | sed 's/^/    /'
  else warn "nvidia-smi not found"; fi
  TORCH_INFO="$($PY -c 'import torch,sys
al=torch.cuda.get_arch_list()
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("arch_list", al)
print("sm_120_in_arch_list", "sm_120" in al)' 2>&1)"
  echo "$TORCH_INFO" | sed 's/^/    /'
  echo "$TORCH_INFO" | grep -q "sm_120_in_arch_list True" || warn "torch arch list does not contain sm_120 (RTX 5090 needs it) or torch is not importable"
  command -v flock >/dev/null 2>&1 || warn "flock not found: GPU stages of different seeds are not serialized"
  say "== check done: $WARN warning(s) =="
}

# ---------------------------------------------------------------- helpers
name_of() { echo "exp001_${DIST}_s$1"; }
run_dir_of() { echo "$DINO_TRAIN/outputs/$(name_of "$1")"; }
marker_of() {  # marker_of <seed> <stage>
  local n; n="$(name_of "$1")"
  case "$2" in
    train) echo "$DINO_TRAIN/outputs/$n/DONE" ;;
    repro_check) echo "$HERE/results/$n/repro_check.json" ;;
    extract) echo "$EXP001_CACHE/$n/DONE" ;;
    diagnose) echo "$HERE/results/$n/diagnose.json" ;;
    probe) echo "$HERE/results/$n/probe.json" ;;
    intervene) echo "$HERE/results/$n/intervene.json" ;;
  esac
}
PIPE_STAGES="repro_check extract diagnose probe intervene"
st() { $PY "$HERE/overnight_status.py" set --file "$STATUS" "$@" >/dev/null 2>&1 || true; }   # st SEED STAGE STATE [--retries N --error T --note T]
ext_running() {  # a training of this seed we did not start: process (hydra.run.dir=<...>/<run name>) or recent file activity in its run directory
  local s="$1" d; d="$(run_dir_of "$s")"
  pgrep -f "hydra.run.dir=[^ ]*$(name_of "$s")([[:space:]]|\$)" >/dev/null 2>&1 && return 0
  [ -d "$d" ] && [ -n "$(find "$d" -type f -mmin "-$HEARTBEAT_MIN" -print -quit 2>/dev/null)" ] && return 0
  return 1
}
other_pipeline_running() {  # a run_all.sh (e.g. the one that started seed 0) or a manual run.py stage for this seed is still active
  local s="$1"
  pgrep -f "^(/[a-z/]*)?(ba)?sh .*run_all\\.sh" >/dev/null 2>&1 && return 0
  pgrep -f "run\\.py (repro_check|extract|diagnose|probe|intervene) .*--model ${DIST}:${s}([[:space:]]|\$)" >/dev/null 2>&1 && return 0
  return 1
}
lastline() { grep -v '^[[:space:]]*$' "$1" 2>/dev/null | tail -n 1 | cut -c1-400; }
gpulock() { if command -v flock >/dev/null 2>&1; then flock "$LOCKDIR/gpu.lock" "$@"; else "$@"; fi; }

retry_step() {  # retry_step <seed> <stage> <marker|-> <cmd...>  : up to ATTEMPTS attempts of the same stage; returns 0 on success
  local seed="$1" stage="$2" marker="$3"; shift 3
  local n; n="$(name_of "$seed")"
  if [ "$marker" != "-" ] && [ -e "$marker" ]; then st "$seed" "$stage" done --note "result already existed: $marker"; say "seed $seed $stage: already done ($marker)"; return 0; fi
  local a lg rc err
  for ((a = 1; a <= ATTEMPTS; a++)); do
    st "$seed" "$stage" running --retries $((a - 1))
    lg="$HERE/logs/overnight_${n}_${stage}_try${a}.log"
    say "seed $seed $stage: attempt $a/$ATTEMPTS -> $lg"
    ( "$@" ) > "$lg" 2>&1; rc=$?
    if [ "$rc" = 0 ]; then st "$seed" "$stage" done --retries $((a - 1)); say "seed $seed $stage: done"; return 0; fi
    err="rc=$rc: $(lastline "$lg")"
    say "seed $seed $stage: attempt $a failed ($err)"
    st "$seed" "$stage" running --retries "$a" --error "$err"
    [ "$a" -lt "$ATTEMPTS" ] && sleep "$RETRY_SLEEP"
  done
  st "$seed" "$stage" failed --retries "$ATTEMPTS" --error "$err"
  return 1
}

skip_rest() {  # skip_rest <seed> <reason> <stages...>
  local seed="$1" why="$2"; shift 2; local x
  for x in "$@"; do st "$seed" "$x" skipped --note "$why"; done
}

agg_cmd() {  # aggregate (retried by caller), then sync whose failure never matters
  flock_agg() { if command -v flock >/dev/null 2>&1; then flock "$LOCKDIR/agg.lock" "$@"; else "$@"; fi; }
  flock_agg $RUN aggregate || return 1
  flock_agg $RUN sync || echo "sync returned non-zero (ignored)"
  return 0
}
aggregate_sync() {  # aggregate_sync <seed|final>
  local who="$1"
  retry_step "$who" aggregate_sync - agg_cmd
  local rc=$? ss=""
  [ -f "$HERE/results/sync_status.json" ] && ss="$(grep -o '"status": *"[a-z_]*"' "$HERE/results/sync_status.json" | head -n 1)"
  st "$who" aggregate_sync "$([ $rc = 0 ] && echo done || echo failed)" --note "sync: ${ss:-no results/sync_status.json} (a failed/skipped upload does not fail the experiment)"
  return 0
}

pipeline() {  # pipeline <seed>: all stages after training; runs as a background subshell
  local seed="$1" model="${DIST}:$1" stage failed=""
  for stage in $PIPE_STAGES; do
    if [ -n "$failed" ]; then st "$seed" "$stage" skipped --note "earlier stage '$failed' failed 3 times"; continue; fi
    case "$stage" in
      repro_check|extract|intervene) retry_step "$seed" "$stage" "$(marker_of "$seed" "$stage")" gpulock $RUN "$stage" --model "$model" || failed="$stage" ;;
      *) retry_step "$seed" "$stage" "$(marker_of "$seed" "$stage")" $RUN "$stage" --model "$model" || failed="$stage" ;;
    esac
  done
  aggregate_sync "$seed"
  say "seed $seed pipeline finished${failed:+ (stopped at $failed)}"
}

train_job() {  # train_job <seed>: background subshell started by us
  local seed="$1"
  retry_step "$seed" train - $RUN train --model "${DIST}:$seed"
}

killtree() { local p c; for c in $(pgrep -P "$1" 2>/dev/null); do killtree "$c"; done; kill "$1" 2>/dev/null; }

# ---------------------------------------------------------------- dry run
plan() {
  say "== plan (nothing is executed) =="
  say "models: $MODELS   TRAIN_PARALLEL=$TRAIN_PARALLEL (max $MAX_CONC concurrent trainings, an external training counts)  attempts/stage=$ATTEMPTS"
  say "wait for an external training up to ${WAIT_MAX_HOURS}h, poll ${POLL_SEC}s, takeover only after ${GRACE_SEC}s without process/file activity (heartbeat ${HEARTBEAT_MIN} min)"
  local s stg
  for s in $SEEDS; do
    if [ -e "$(marker_of "$s" train)" ]; then say "seed $s: train DONE exists -> skip training, start pipeline immediately"
    elif ext_running "$s"; then say "seed $s: training seems to be running already (process or recent files) -> WAIT for DONE, never start a second one; pipeline starts right after"
    else say "seed $s: no DONE, no running training -> start 'run.py train --model ${DIST}:$s' when a slot is free (resumes from model_latest.pth if the run directory exists)"; fi
    for stg in $PIPE_STAGES; do
      if [ -e "$(marker_of "$s" "$stg")" ]; then say "    $stg: result exists -> skipped"; else say "    $stg: $( [ "$stg" = diagnose ] || [ "$stg" = probe ] && echo CPU || echo "GPU (serialized across seeds by a lock)")"; fi
    done
    say "    aggregate_sync: run.py aggregate, then run.py sync (failure ignored)"
  done
  say "priority: scheduler handles seeds in order (0 first); a seed's pipeline starts as soon as its DONE exists, independent of other seeds' training"
  say "final: aggregate + sync, then logs/overnight_summary.txt"
}

check_env
if [ "$DRY" = 1 ]; then plan; say "dry run finished ($WARN warning(s))"; exit 0; fi

# ---------------------------------------------------------------- run
mkdir -p "$HERE/logs" "$LOCKDIR" "$HERE/results"
$PY "$HERE/overnight_status.py" init --file "$STATUS" --seeds "$SEEDS" --meta "train_parallel=$TRAIN_PARALLEL" "max_concurrent_trainings=$MAX_CONC" "repro_gate=off (recorded only, run continues)" "approved=$EXP001_APPROVED" "env_warnings=$WARN" "models=$MODELS" "script_start=$(ts)"
if [ ! -e "$HERE/manifests/train_${DIST}.json" ]; then say "running 'run.py listing' (manifest missing)"; $RUN listing > "$HERE/logs/overnight_listing.log" 2>&1 || warn "listing failed: $(lastline "$HERE/logs/overnight_listing.log")"; fi

declare -A TS_STATE TPID PPID_ EXT_FIRST NOACT_SINCE PIPEWAIT
for s in $SEEDS; do TS_STATE[$s]=waiting; done
CLEANED=0
finish() {
  [ "$CLEANED" = 1 ] && return; CLEANED=1
  local s
  for s in $SEEDS; do   # only processes started by this script
    [ -n "${TPID[$s]:-}" ] && kill -0 "${TPID[$s]}" 2>/dev/null && { say "stopping own training job of seed $s"; killtree "${TPID[$s]}"; }
    [ -n "${PPID_[$s]:-}" ] && kill -0 "${PPID_[$s]}" 2>/dev/null && { say "stopping own pipeline of seed $s"; killtree "${PPID_[$s]}"; }
  done
  { $PY "$HERE/overnight_status.py" summary --file "$STATUS"; echo; echo "total elapsed of this invocation: $(( $(date +%s) - T0 )) s ($(( ($(date +%s) - T0) / 60 )) min)"; } > "$SUMMARY" 2>&1
  say "summary written: $SUMMARY"
}
trap 'finish; exit 130' INT TERM
trap 'finish' EXIT

active_trainings() { local c=0 s; for s in $SEEDS; do case "${TS_STATE[$s]}" in ext|own|gone) c=$((c + 1)) ;; esac; done; echo "$c"; }
all_finished() {
  local s
  for s in $SEEDS; do case "${TS_STATE[$s]}" in failed|piped_done) ;; *) return 1 ;; esac; done
  return 0
}

say "== scheduler start: seeds $SEEDS =="
while :; do
  now=$(date +%s)
  for s in $SEEDS; do
    case "${TS_STATE[$s]}" in
      waiting)
        if [ -e "$(marker_of "$s" train)" ]; then
          st "$s" train done --note "DONE already existed"; TS_STATE[$s]=trained
        elif ext_running "$s"; then
          EXT_FIRST[$s]="$now"; TS_STATE[$s]=ext; st "$s" train running --note "training running outside this script; waiting for DONE (not touched)"
          say "seed $s: external training detected, waiting"
        elif [ "$(active_trainings)" -lt "$MAX_CONC" ]; then
          say "seed $s: starting own training"; TS_STATE[$s]=own
          ( train_job "$s" ) & TPID[$s]=$!
        fi ;;
      ext)
        if [ -e "$(marker_of "$s" train)" ]; then st "$s" train done --note "external training finished (DONE seen)"; say "seed $s: DONE appeared"; TS_STATE[$s]=trained
        elif ext_running "$s"; then NOACT_SINCE[$s]=""
          if [ $((now - EXT_FIRST[$s])) -gt "$WAIT_MAX_SEC" ]; then
            st "$s" train failed --error "waited ${WAIT_MAX_HOURS}h for an external training without DONE (it was left running)"; skip_rest "$s" "train wait timeout" $PIPE_STAGES aggregate_sync
            TS_STATE[$s]=failed; say "seed $s: wait timeout"
          fi
        else NOACT_SINCE[$s]="$now"; TS_STATE[$s]=gone; say "seed $s: external training no longer visible and no DONE; grace ${GRACE_SEC}s"; fi ;;
      gone)
        if [ -e "$(marker_of "$s" train)" ]; then st "$s" train done --note "external training finished (DONE seen)"; TS_STATE[$s]=trained
        elif ext_running "$s"; then TS_STATE[$s]=ext; say "seed $s: external training visible again"
        elif [ $((now - NOACT_SINCE[$s])) -ge "$GRACE_SEC" ]; then
          say "seed $s: external training is gone without DONE -> taking over (resume from model_latest.pth)"; TS_STATE[$s]=own
          ( train_job "$s" ) & TPID[$s]=$!
        fi ;;
      own)
        if ! kill -0 "${TPID[$s]}" 2>/dev/null; then
          wait "${TPID[$s]}" 2>/dev/null
          if [ -e "$(marker_of "$s" train)" ]; then TS_STATE[$s]=trained
          else TS_STATE[$s]=failed; skip_rest "$s" "train failed ${ATTEMPTS} times" $PIPE_STAGES aggregate_sync; say "seed $s: training failed; other seeds continue"; fi
        fi ;;
      trained)
        if other_pipeline_running "$s"; then
          [ -z "${PIPEWAIT[$s]:-}" ] && { PIPEWAIT[$s]="$now"; say "seed $s: a run_all.sh / run.py stage for this seed is running outside this script; waiting (it may create repro_check.json etc.; finished results are skipped afterwards)"; st "$s" repro_check pending --note "waiting for an external run_all.sh/run.py stage of this seed to exit"; }
          if [ $((now - PIPEWAIT[$s])) -le "$WAIT_MAX_SEC" ]; then continue; fi
          warn "seed $s: external stage still running after the wait limit; starting the pipeline anyway"
        fi
        say "seed $s: starting pipeline (repro_check, extract, diagnose, probe, intervene, aggregate+sync)"
        ( pipeline "$s" ) & PPID_[$s]=$!; TS_STATE[$s]=piped ;;
      piped)
        if ! kill -0 "${PPID_[$s]}" 2>/dev/null; then wait "${PPID_[$s]}" 2>/dev/null; TS_STATE[$s]=piped_done; fi ;;
    esac
  done
  # a free slot may now exist for a waiting seed: re-run the loop body without sleeping if something is startable
  all_finished && break
  sleep "$POLL_SEC"
done

say "== all seeds finished (done or failed); final aggregate + sync =="
aggregate_sync final
finish
say "overnight finished. Read: $STATUS  $SUMMARY  $HERE/results.json"
exit 0
