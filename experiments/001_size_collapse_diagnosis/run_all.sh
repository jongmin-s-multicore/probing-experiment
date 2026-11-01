#!/usr/bin/env bash
# Run experiment 001 end to end (or part of it). Every stage is skipped when its result file exists (resume); training resumes from
# model_latest.pth of its run directory when interrupted. Logs: logs/<time>_<stage>.log
#
#   bash run_all.sh [--dry-run] [--force] [--from STAGE] [--only S1,S2] [--skip S1,S2]
#
# Stages (in order): check selftest download maskcheck listing train repro_check [gate] extract diagnose probe intervene aggregate
#   repro_check: extract only a026/ind/a248 and write results/<run>/repro_check.json (area numbers). With REPRO_GATE=1 (default) the script
#   stops after it so that a human can look at the numbers; continue with `bash run_all.sh --from extract` (or REPRO_GATE=0).
# Variables:
#   CONDA_ENV  conda env name (default dino_wm)         DINO_WORK / DATASET_DIR / DINO_TRAIN / EXP001_CACHE / ACWM_PHYS_SRC  paths (config.yaml paths)
#   MODELS     space separated "distribution:seed" to train+analyse (default "baseline_single:0")
#              e.g. MODELS="baseline_single:0 mix_wide:0"   (control distributions need human approval of candidate F first)
#   EXP001_APPROVED  space separated distributions a human approved for training (train refuses others). Default: none -> set it explicitly.
#   EXP001_DATA_SOURCE drive (default, rclone from Google Drive) | hf.  EXP001_RCLONE_REMOTE, EXP001_DATA_REMOTE_PATH, EXP001_RCLONE_FLAGS: see README.
#   REPRO_GATE 1 (default): stop after the repro_check stage.   EXP001_DATASET  cloth_sim (default) | acwm_phys (legacy)
#   The old mid-size generation stage (gen / GEN=1) was removed: the 9 cloth sizes now come with the dataset (see README).
#   SYNC=1 (opt-in) upload results to Google Drive at the end; SYNC_CKPT=1 / SYNC_CACHE=1 also upload checkpoints / extraction cache.
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
export EXP001_DATASET="${EXP001_DATASET:-cloth_sim}"
REPRO_GATE="${REPRO_GATE:-1}"
# download skip marker: results/download_check.json (written after a finished download of the selected splits; per-split skipping is
# also done inside `run.py download`). With DOWNLOAD_DISTS set (extra train splits) the stage always runs so that the new splits are fetched.
if [ "$EXP001_DATASET" = "cloth_sim" ]; then DATA_ROOT="$DATASET_DIR/cloth_sim"; MARK_DL="$HERE/results/download_check.json"
  [ -n "${DOWNLOAD_DISTS:-}" ] && MARK_DL=""
else DATA_ROOT="$DATASET_DIR/acwm_phys/deformable/clothmove"; MARK_DL="$DATA_ROOT/ood_test/metadata.pt"; fi
mkdir -p logs results
if [ "$DRY" = 0 ]; then   # start times for results/run_meta.json (first_start is kept across resumed runs)
  NOW="$(date +%Y-%m-%dT%H:%M:%S%z)"
  FIRST="$NOW"; [ -f results/run_times.json ] && FIRST="$(sed -n 's/.*"first_start": *"\([^"]*\)".*/\1/p' results/run_times.json)"
  printf '{"first_start": "%s", "last_start": "%s"}\n' "${FIRST:-$NOW}" "$NOW" > results/run_times.json
fi

if [ "$DRY" = 0 ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" || { echo "cannot activate conda env $CONDA_ENV (run setup.sh)"; exit 1; }
fi
PY="python"
TS="$(date +%Y%m%d_%H%M%S)"
STAGES="check selftest download maskcheck listing train repro_check extract diagnose probe intervene aggregate"

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
# download: data.source (EXP001_DATA_SOURCE=drive|hf, default drive = rclone; see README). Default = ind_train + the 9 *_test splits. Training splits of control distributions: DOWNLOAD_DISTS="mix_wide" (see README)
run_stage download download "$MARK_DL"                                $PY run.py download --dists "${DOWNLOAD_DISTS:-}"
run_stage maskcheck maskcheck "$HERE/results/maskcheck.json"          $PY run.py maskcheck
run_stage listing  listing  "$HERE/manifests/listing.json"            $PY run.py listing
for m in $MODELS; do
  d="${m%%:*}"; s="${m##*:}"; name="$($PY run.py run_name --model "${d}:${s}" | tail -n 1)"
  run_stage train     "train_$name"     "$DINO_TRAIN/outputs/$name/DONE"           $PY run.py train --model "$m"
  had=0; [ -e "$HERE/results/$name/repro_check.json" ] && had=1
  run_stage repro_check "repro_$name" "$HERE/results/$name/repro_check.json" $PY run.py repro_check --model "$m" $F
  if [ "$REPRO_GATE" = 1 ] && [ "$had" = 0 ] && [ -e "$HERE/results/$name/repro_check.json" ] && [ "$DRY" = 0 ]; then
    echo "[gate] repro_check written: results/$name/repro_check.json. Look at the numbers, then: bash run_all.sh --from extract  (or REPRO_GATE=0)"; exit 0
  fi
  run_stage extract   "extract_$name"   "$EXP001_CACHE/$name/DONE"                 $PY run.py extract --model "$m" $F
  run_stage diagnose  "diagnose_$name"  "$HERE/results/$name/diagnose.json"        $PY run.py diagnose --model "$m" $F
  run_stage probe     "probe_$name"     "$HERE/results/$name/probe.json"           $PY run.py probe --model "$m" $F
  run_stage intervene "intervene_$name" "$HERE/results/$name/intervene.json"       $PY run.py intervene --model "$m" $F
done
run_stage aggregate aggregate "" $PY run.py aggregate
# Optional last step: upload to Google Drive (opt-in). Never fails the run. Needs config.yaml sync.drive.* (rclone remote) -- see README.
#   SYNC=1 bash run_all.sh ...     SYNC_CKPT=1 / SYNC_CACHE=1 add the large checkpoint / cache folders (default: excluded)
if [ "${SYNC:-0}" = 1 ]; then
  SF=""; [ "${SYNC_CKPT:-0}" = 1 ] && SF="$SF --ckpt"; [ "${SYNC_CACHE:-0}" = 1 ] && SF="$SF --cache"
  echo "[run] sync -> logs/${TS}_sync.log"
  if [ "$DRY" = 1 ]; then echo "    $PY run.py sync --dry-run$SF"
  else $PY run.py sync $SF 2>&1 | tee "logs/${TS}_sync.log" || true; fi
fi
echo "done. results: $HERE/results.json"
