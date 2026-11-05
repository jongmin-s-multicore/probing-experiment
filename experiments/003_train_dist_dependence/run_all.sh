#!/usr/bin/env bash
# Run experiment 003 end to end (or part of it). Every stage skips work whose result exists (resume).
#   bash run_all.sh [--dry-run] [--force] [--from STAGE] [--only S1,S2] [--skip S1,S2]
# Stages (in order): check selftest download maskcheck listing train extract metrics baseline aggregate
#   `smoke` (12 GB memory/time measurement) is NOT in this chain: run it yourself first (`python run.py smoke`).
#   `train` refuses unless (1) env EXP003_APPROVED lists the distributions the user approved for GPU training and
#   (2) results/smoke.json exists with status finished (override only with ALLOW_NO_SMOKE=1). Neither was done by the author.
# Variables: CONDA_ENV (default dino_wm)  DINO_WORK / DATASET_DIR / DINO_TRAIN / EXP003_CACHE / ACWM_PHYS_SRC (paths, config.yaml paths)
#   MODELS  space separated "distribution:seed" (default: all 12 = 4 distributions x seeds 0,1,2 from config)
#   EXP003_APPROVED  e.g. "single_a026 single_a248 mix_adjacent mix_wide"   SYNC=1 uploads results to Google Drive at the end (opt-in)
# Retry limit: each python stage is attempted at most 3 times in a row (results/attempts.json, CLAUDE.md); this script does not loop.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
DRY=0; FORCE=0; FROM=""; ONLY=""; SKIP=""
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1 ;; --force) FORCE=1 ;;
  --from) FROM="$2"; shift ;; --only) ONLY="$2"; shift ;; --skip) SKIP="$2"; shift ;;
  *) echo "unknown option $1"; exit 2 ;; esac; shift; done

CONDA_ENV="${CONDA_ENV:-dino_wm}"
export DINO_WORK="${DINO_WORK:-$HOME/dinowm}"
export DATASET_DIR="${DATASET_DIR:-$DINO_WORK/data}"
export DINO_TRAIN="${DINO_TRAIN:-$DINO_WORK/train_runs}"
export EXP003_CACHE="${EXP003_CACHE:-$DINO_WORK/exp003_cache}"
export ACWM_PHYS_SRC="${ACWM_PHYS_SRC:-$DINO_WORK/acwm_phys_src}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p logs results
if [ "$DRY" = 0 ] && [ "$CONDA_ENV" != "none" ]; then   # CONDA_ENV=none: no conda (Colab); the caller already has the right python first in PATH
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" || { echo "cannot activate conda env $CONDA_ENV (run setup.sh)"; exit 1; }
fi
PY="python"
TS="$(date +%Y%m%d_%H%M%S)"
STAGES="check selftest download maskcheck listing train extract metrics baseline aggregate"
if [ -z "${MODELS:-}" ]; then MODELS="$($PY - <<'PYEOF'
import sys; sys.path.insert(0, "src")
from exp003.common import load_config, all_models
print(" ".join(all_models(load_config())))
PYEOF
)"; fi

selected() {
  local s="$1"
  [ -n "$ONLY" ] && { case ",$ONLY," in *",$s,"*) ;; *) return 1 ;; esac; }
  [ -n "$SKIP" ] && { case ",$SKIP," in *",$s,"*) return 1 ;; esac; }
  if [ -n "$FROM" ]; then
    local seen=0 t; for t in $STAGES; do [ "$t" = "$FROM" ] && seen=1; [ "$t" = "$s" ] && { [ $seen = 1 ] && return 0 || return 1; }; done
  fi
  return 0
}
run_stage() {  # run_stage <stage> <label> <marker-file or ""> <cmd...>
  local stage="$1" label="$2" marker="$3"; shift 3
  selected "$stage" || { echo "[skip:not selected] $label"; return 0; }
  if [ -n "$marker" ] && [ -e "$marker" ] && [ "$FORCE" = 0 ]; then echo "[skip:done] $label ($marker)"; return 0; fi
  local log="logs/${TS}_${label//[:\/ ]/_}.log"
  echo "[run] $label  -> $log"
  if [ "$DRY" = 1 ]; then echo "    $*"; return 0; fi
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then echo "[FAILED rc=$rc] $label (see $log). Finished stages are skipped on re-run. Max 3 consecutive attempts per stage (results/attempts.json)."; exit "$rc"; fi
}

F=""; [ "$FORCE" = 1 ] && F="--force"
DRYF=""; [ "$DRY" = 1 ] && DRYF="--dry-run"
run_stage check    check    ""                                   $PY run.py check
run_stage selftest selftest ""                                   $PY run.py selftest
run_stage download download "$HERE/results/download_check.json"  $PY run.py download
run_stage maskcheck maskcheck "$HERE/results/maskcheck.json"     $PY run.py maskcheck
run_stage listing  listing  "$HERE/manifests/listing.json"       $PY run.py listing
if [ "$DRY" = 0 ] && selected train && [ "${ALLOW_NO_SMOKE:-0}" != 1 ] && ! grep -q '"status": "finished"' results/smoke.json 2>/dev/null; then
  echo "[stop] results/smoke.json missing or not finished: run 'python run.py smoke' first (12 GB check), then re-run with --from train."; exit 3
fi
for m in $MODELS; do
  name="$($PY run.py run_name --model "$m" | tail -n 1)"
  run_stage train   "train_$name"   "$DINO_TRAIN/outputs/$name/DONE"     $PY run.py train --model "$m"
  run_stage extract "extract_$name" "$EXP003_CACHE/$name/DONE"           $PY run.py extract --model "$m" $F
  run_stage metrics "metrics_$name" "$HERE/results/$name/metrics.json"   $PY run.py metrics --model "$m" $F
done
run_stage baseline baseline "" $PY run.py baseline          # finds/copies the 001 results.json for H-H2 (fails clearly if missing)
run_stage aggregate aggregate "" $PY run.py aggregate
if [ "${SYNC:-0}" = 1 ] && [ "$DRY" = 0 ]; then $PY run.py sync 2>&1 | tee "logs/${TS}_sync.log" || true; fi
echo "done. results: $HERE/results.json"
