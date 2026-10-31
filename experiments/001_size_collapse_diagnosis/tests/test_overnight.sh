#!/usr/bin/env bash
# Orchestration test of overnight.sh with a fake run.py in a temporary directory. No GPU/data/rclone; never touches real $DINO_TRAIN.
#   bash tests/test_overnight.sh [python]       (python needs only the stdlib; default python3)
set -uo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN="${1:-python3}"
T="$(mktemp -d "${TMPDIR:-/tmp}/ovn_test.XXXXXX")"
echo "temp dir: $T"
fail=0; ok() { echo "  PASS $1"; }; bad() { echo "  FAIL $1"; fail=1; }
setup() {  # setup <name>: fresh copy of the scripts + fake run.py
  W="$T/$1"; mkdir -p "$W/home" "$W/train" "$W/cache"
  cp "$SRC/overnight.sh" "$SRC/overnight_status.py" "$W/"; cp "$SRC/tests/fake_run.py" "$W/run.py"
  export HOME="$W/home" DINO_WORK="$W" DATASET_DIR="$W/data" DINO_TRAIN="$W/train" EXP001_CACHE="$W/cache" ACWM_PHYS_SRC="$W/src"
  export FAKE_ROOT="$W" OVERNIGHT_PY="$PYBIN" OVERNIGHT_SKIP_ACTIVATE=1 POLL_SEC=1 GRACE_SEC=2 RETRY_SLEEP=0 FAKE_SLEEP=1 WANDB_MODE=disabled
  mkdir -p "$W/data/cloth_sim" "$W/src"
}
q() { "$PYBIN" -c "import json,sys; d=json.load(open('$1')); print(eval(sys.argv[1]))" "$2"; }

echo "== scenario 1: seed 0 external training (then DONE), seed 1 extract always fails, seed 2 probe fails twice =="
setup s1
mkdir -p "$W/train/outputs/exp001_baseline_single_s0"
# fake external training: a process whose command line carries hydra.run.dir=<...>/exp001_baseline_single_s0
# (python, not bash -c: bash -c would exec the last command and hide the arguments from pgrep -f)
( "$PYBIN" -c 'import sys,time,pathlib; time.sleep(8); pathlib.Path(sys.argv[1].split("=",1)[1], "DONE").touch()' "hydra.run.dir=$W/train/outputs/exp001_baseline_single_s0" ) >/dev/null 2>&1 &
sleep 0.5
( cd "$W" && FAIL_SPEC="1:extract:99,2:probe:2" timeout 120 bash overnight.sh > "$W/out.log" 2>&1 ); rc=$?
[ $rc = 0 ] && ok "exit code 0" || bad "exit code $rc"
S="$W/logs/overnight_status.json"
[ "$(q "$S" "d['seeds']['0']['stages']['train']['state']")" = done ] && ok "seed0 train done (waited, did not start own)" || bad "seed0 train state"
grep -q "train seed0" "$W/events.log" && bad "seed 0 training was started by the script (duplicate!)" || ok "no duplicate seed-0 training"
[ "$(q "$S" "d['seeds']['0']['stages']['intervene']['state']")" = done ] && ok "seed0 pipeline complete" || bad "seed0 pipeline"
[ "$(q "$S" "d['seeds']['1']['stages']['extract']['state']")" = failed ] && ok "seed1 extract failed" || bad "seed1 extract state"
[ "$(q "$S" "d['seeds']['1']['stages']['extract']['retries']")" = 3 ] && ok "seed1 extract 3 attempts" || bad "seed1 extract attempts"
[ "$(grep -c 'extract seed1 FAIL' "$W/events.log")" = 3 ] && ok "exactly 3 extract attempts executed" || bad "attempt count in events"
q "$S" "d['seeds']['1']['stages']['extract']['last_error']" | grep -q "injected failure" && ok "last error line recorded" || bad "last error"
[ "$(q "$S" "d['seeds']['1']['stages']['diagnose']['state']")" = skipped ] && ok "seed1 later stages skipped" || bad "seed1 skip"
[ "$(q "$S" "d['seeds']['2']['stages']['probe']['state']")" = done ] && [ "$(q "$S" "d['seeds']['2']['stages']['probe']['retries']")" = 2 ] && ok "seed2 probe retried twice then done" || bad "seed2 probe"
[ "$(q "$S" "d['seeds']['2']['stages']['intervene']['state']")" = done ] && ok "seed2 unaffected by seed1 failure" || bad "seed2 isolation"
[ "$(q "$S" "d['seeds']['final']['stages']['aggregate_sync']['state']")" = done ] && ok "final aggregate+sync" || bad "final sync"
[ -f "$W/results.json" ] && [ -f "$W/logs/overnight_summary.txt" ] && ok "results.json + summary written" || bad "outputs"
# parallelism / priority from the event log
python3 - "$W/events.log" <<'PY' && ok "ordering: seed2 training waited for a free slot (external seed 0 counted)" || bad "ordering (see events.log)"
import sys
ev=[l.split(None,1) for l in open(sys.argv[1])]
t=lambda m: min(float(a) for a,b in ev if b.strip()==m)
# max 2 concurrent trainings, the external seed-0 training occupies one slot: seed 2 waits for seed 1's training to end
assert t("train seed2 start") >= t("train seed1 end")
# seed 0's pipeline did not wait for the other seeds
assert t("repro_check seed0 start") < t("probe seed2 end") + 100
PY
cat "$W/logs/overnight_summary.txt" | head -30

echo "== scenario 2: seed 0 external training disappears without DONE -> takeover after grace =="
setup s2
mkdir -p "$W/train/outputs/exp001_baseline_single_s0"; touch -d '3 hours ago' "$W/train/outputs/exp001_baseline_single_s0/model_latest.pth"
( cd "$W" && OVERNIGHT_SEEDS="0" timeout 120 bash overnight.sh > "$W/out.log" 2>&1 ); rc=$?
[ $rc = 0 ] && ok "exit 0" || bad "exit $rc"
grep -q "train seed0 start" "$W/events.log" && [ -f "$W/train/outputs/exp001_baseline_single_s0/DONE" ] && ok "stale run dir without process -> own training (resume) and pipeline" || bad "takeover"

echo "== scenario 3: wait timeout leaves external training untouched =="
setup s3
mkdir -p "$W/train/outputs/exp001_baseline_single_s0"
"$PYBIN" -c 'import time; time.sleep(15)' "hydra.run.dir=$W/train/outputs/exp001_baseline_single_s0" >/dev/null 2>&1 &
EP=$!
sleep 0.5
( cd "$W" && OVERNIGHT_SEEDS="0" WAIT_MAX_SEC=3 timeout 120 bash overnight.sh > "$W/out.log" 2>&1 ); rc=$?
[ "$(q "$W/logs/overnight_status.json" "d['seeds']['0']['stages']['train']['state']")" = failed ] && ok "timeout recorded as failed" || bad "timeout state"
kill -0 "$EP" 2>/dev/null && ok "external process still alive (not killed)" || bad "external process killed"
[ ! -e "$W/events.log" ] || ! grep -q "train seed0" "$W/events.log" && ok "no second training started" || bad "duplicate training"
wait 2>/dev/null

echo "== scenario 4: TRAIN_PARALLEL=0 (sequential) =="
setup s4
( cd "$W" && OVERNIGHT_SEEDS="1 2" TRAIN_PARALLEL=0 FAKE_SLEEP=2 timeout 120 bash overnight.sh > "$W/out.log" 2>&1 )
python3 - "$W/events.log" <<'PY' && ok "seed2 training starts after seed1 training ended" || bad "sequential ordering"
import sys
ev=[l.split(None,1) for l in open(sys.argv[1])]
t=lambda m: min(float(a) for a,b in ev if b.strip()==m)
assert t("train seed2 start") >= t("train seed1 end")
PY
echo "== scenario 5: an external run_all-style stage of seed 0 is still running after DONE -> pipeline waits =="
setup s5
mkdir -p "$W/train/outputs/exp001_baseline_single_s0"; touch "$W/train/outputs/exp001_baseline_single_s0/DONE"
"$PYBIN" -c 'import time; time.sleep(6)' run.py repro_check --model baseline_single:0 >/dev/null 2>&1 &
T5=$(date +%s.%N); sleep 0.3
( cd "$W" && OVERNIGHT_SEEDS="0" timeout 120 bash overnight.sh > "$W/out.log" 2>&1 )
FIRST=$(awk 'NR==1{print $1}' "$W/events.log" | head -1); FS=$(grep -m1 'repro_check seed0 start' "$W/events.log" | cut -d' ' -f1)
python3 -c "import sys; sys.exit(0 if float('$FS') - float('$T5') >= 5.5 else 1)" && ok "own repro_check started only after the external stage exited" || bad "pipeline did not wait (start offset $(python3 -c "print(float('$FS')-float('$T5'))"))"

echo; [ $fail = 0 ] && echo "ALL PASSED" || echo "SOME FAILED"; echo "(temp dir kept: $T)"
exit $fail
