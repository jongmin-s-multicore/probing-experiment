#!/usr/bin/env bash
# Run experiment 001 end to end (or part of it). Every stage is skipped when its result file exists (resume); training resumes from
# model_latest.pth of its run directory when interrupted. Logs: logs/<time>_<stage>.log
#
#   bash run_all.sh [--dry-run] [--force] [--from STAGE] [--only S1,S2] [--skip S1,S2]
#
# Stages (in order): check selftest download gen listing train extract diagnose probe intervene aggregate
# Variables:
#   CONDA_ENV  conda env name (default dino_wm)         DINO_WORK / DATASET_DIR / DINO_TRAIN / EXP001_CACHE / ACWM_PHYS_SRC  paths (config.yaml paths)
#   MODELS     space separated "distribution:seed" to train+analyse (default "baseline_single:0")
#              e.g. MODELS="baseline_single:0 mix_ood3:0"   (mix_ood3 needs human approval of candidate F first)
#   GEN=1      also try the optional mid-size generator stage (needs ACWM_PHYS_GENERATOR_CMD, see README)
# Smoke tests are not part of this script (they use run names with a _smoke suffix): see README "스모크 테스트".
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
DRY=0; FORCE=0; FROM=""; ONLY=""; SKIP=""
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1 ;; --force) FORCE=1 ;;
  --from) FROM="$2"; shift ;; --only) ONLY="$2"; shift ;; --skip) SKIP="$2"; shift ;;
  *) echo "unknown option $1"; exit 2 ;; esac; shift; done

CONDA_ENV="${CONDA_ENV:-dino_wm}"
MODELS="${MODELS:-baseline_single:0}"
export EXP001_MODELS="$MODELS"
export DINO_WORK="${DINO_WORK:-$HOME/dinowm}"
export DATASET_DIR="${DATASET_DIR:-$DINO_WORK/data}"
export DINO_TRAIN="${DINO_TRAIN:-$DINO_WORK/train_runs}"
export EXP001_CACHE="${EXP001_CACHE:-$DINO_WORK/exp001_cache}"
export ACWM_PHYS_SRC="${ACWM_PHYS_SRC:-$DINO_WORK/acwm_phys_src}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
CLOTH="$DATASET_DIR/acwm_phys/deformable/clothmove"
mkdir -p logs

if [ "$DRY" = 0 ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" || { echo "cannot activate conda env $CONDA_ENV (run setup.sh)"; exit 1; }
fi
PY="python"
TS="$(date +%Y%m%d_%H%M%S)"
STAGES="check selftest download gen listing train extract diagnose probe intervene aggregate"

selected() {  # is stage $1 selected by --from/--only/--skip
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
  if [ "$rc" != 0 ]; then echo "[FAILED rc=$rc] $label (see $log). Fix and re-run: finished stages are skipped."; exit "$rc"; fi
}

F=""; [ "$FORCE" = 1 ] && F="--force"
DRYF=""; [ "$DRY" = 1 ] && DRYF="--dry-run"
run_stage check    check    ""                                        $PY run.py check
run_stage selftest selftest ""                                        $PY run.py selftest
run_stage download download "$CLOTH/ood_test/metadata.pt"             $PY run.py download
if [ "${GEN:-0}" = 1 ]; then run_stage gen gen "" $PY run.py gen; fi
run_stage listing  listing  "$HERE/manifests/listing.json"            $PY run.py listing
for m in $MODELS; do
  d="${m%%:*}"; s="${m##*:}"; name="exp001_${d}_s${s}"
  run_stage train     "train_$name"     "$DINO_TRAIN/outputs/$name/DONE"           $PY run.py train --model "$m"
  run_stage extract   "extract_$name"   "$EXP001_CACHE/$name/DONE"                 $PY run.py extract --model "$m" $F
  run_stage diagnose  "diagnose_$name"  "$HERE/results/$name/diagnose.json"        $PY run.py diagnose --model "$m" $F
  run_stage probe     "probe_$name"     "$HERE/results/$name/probe.json"           $PY run.py probe --model "$m" $F
  run_stage intervene "intervene_$name" "$HERE/results/$name/intervene.json"       $PY run.py intervene --model "$m" $F
done
run_stage aggregate aggregate "" $PY run.py aggregate
echo "done. results: $HERE/results.json"
