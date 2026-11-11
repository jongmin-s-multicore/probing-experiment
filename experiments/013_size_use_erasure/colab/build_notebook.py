#!/usr/bin/env python3
"""Regenerate colab/exp013_colab.ipynb:   python3 colab/build_notebook.py     (run again after ANY code change)
The notebook has NO embedded code: cell A3 git-clones the public repo (REPO_URL, REPO_REF set in cell A2) and records HEAD.
The ref must contain 002 {src, run.py, config.yaml, colab/requirements_colab.txt}, 004 {src, run.py, config.yaml, colab/exp045_tools.py, colab/requirements_colab_extra.txt},
009 {src, config.yaml} and 013 {src, run.py, config.yaml}. Reads 002/results.json config_sha256 at build time. config.yaml values are NOT changed by the notebook.
Local paths (/content/mylab013, /content/venv013, /content/cache013, /content/pilot013) and the Drive output (MyDrive/exp013_work) do not overlap with 006/009/010/012
(/content/mylab012, /content/venv012, ...; exp012_work, ...) so 012 and 013 can run in separate Colab sessions at the same time EXCEPT the data location /content/dinowm/data/...
which is fixed by the absolute episode dirs inside the 002 train_manifest.json (identical read-only metadata.pt files) and the ACWM source checkout /content/dinowm/acwm_phys_src
(same pinned commit). Two notebooks in the SAME Colab runtime would share those two read-only locations; use separate sessions."""
import json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPS = HERE.parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


C = []
C.append(md(r'''
# Experiment 013 (does the predictor use size information for MOTION prediction? input-erasure test) on Google Colab
Order: **A** settings + setup (A1 GPU, A2 settings, A3 clone, A4 environment, A5 ACWM source, A6 env vars, A6b metadata.pt, A7 disk), **B** check (inputs, dry runs), **C** selftest (CPU toy), **D** approvals + optional pilot, **E** stage 0 gate and eraser fits (CPU), main `measure` (gate `APPROVED_MAIN_013`) and `readout`, **F** aggregate, **G** copy results out.
Every cell is idempotent; after a disconnect choose a GPU runtime, run A and B again and continue: `measure` and `readout` skip finished units (part files on Drive).

What is measured (config `experiments/013_size_use_erasure/config.yaml` and `notes/hypotheses.md` 013, user-confirmed 2026-11-11, **not changed by this notebook**):
stage 0 (CPU gate): per-level ridge regressors on real encoder latents; stage 1 passes only if other-level / same-level error ratio >= 1.1 in >= 4 of 6 directed pairs. **If the gate fails, `run.py measure` and `readout` refuse (exit code 5, no attempt consumed).**
stage 1: conditions (a) `single_ind`, (b) `multi_60_ind_160`, train seeds 0,1,2 (`model_latest`) plus random-init predictors (1234 + seed): one-step teacher-forced prediction with the size information erased from the predictor INPUT (scalar LEACE, level-mean-match), against 20 norm-matched random directions; motion error normalised by the persistence (Delta z = 0) error; episode bootstrap 2000 (seed 14000, 90% CI). Erasure check: linear size readout R2 at blocks 0..6. Stage 2 (no verdict): level-mean predicted vs real Delta z.
The only overrides used anywhere are existing CLI flags of `run.py` (pilot cell: `--conditions --train-seeds --episodes-per-level --results-dir`), which `results.json` records and marks `pilot: true`.

Inputs (read-only): the six 002 checkpoints `exp002_work/train_runs/outputs/exp002_{single_ind,multi_60_ind_160}_s{0,1,2}/checkpoints/model_latest.pth` (note the `exp002_` prefix), the 002 Q1 cache, and the per-split `metadata.pt` files (cell A6b). **No world model is trained.** `run.py measure` has a preflight (exit code 4, no attempt consumed) for checkpoints, cache, `metadata.pt` and the eraser fits.
Outputs (Drive): `MyDrive/exp013_work/{results,logs,results_pilot,timing_*.json,code_version.json}`; `results.json` is written to `exp013_work/results/` by aggregate.

Human approval (CLAUDE.md): GPU runs need you. The pilot needs `APPROVED_PILOT_013 = True`, the main run needs `APPROVED_MAIN_013 = True` (cell D0). Stage 0 gate and eraser fit are CPU-only and cheap (cell E0). All time numbers in this notebook are estimates (추정). Everything here is UNVERIFIED on Colab (written and checked on a CPU-only machine).
'''))

C.append(code(r'''
# A1. GPU / Python / disk
import sys, subprocess, os
def sh(cmd, cwd=None, env=None):
    env = dict(os.environ if env is None else env, MPLBACKEND='Agg')
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
    return p.wait()
sh('nvidia-smi')
print(sys.version)
sh('df -h /content | tail -1')
sh('free -g | head -2')
'''))

C.append(code(r'''
# A2. SETTINGS (all Drive paths and options live here). Mount Drive first, then check that the paths below are right for YOUR Drive.
import os
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive'
WORK002 = os.environ.get('EXP002_WORK_ROOT', DRIVE + '/exp002_work')     # READ-ONLY: train_runs/outputs/exp002_multi_60_ind_160_s{0,1,2}/ (checkpoints), exp002_cache/q1_tokens/, results/results.json
WORK013 = os.environ.get('EXP013_WORK_ROOT', DRIVE + '/exp013_work')     # OUTPUT (persistent): results/, logs/, results_pilot/, timing_*.json
DATA_SRC = os.environ.get('EXP002_DRIVE_MOUNT_SRC', DRIVE + '/dino-wm-videos/[data]/[cloth-sim] cloth_move')   # same as the 004/009 notebooks; only <split>/metadata.pt is read from here (cell A6b)
REPO_URL = 'https://github.com/jongmin-s-multicore/probing-experiment.git'     # PUBLIC repository (cloned in A3)
REPO_REF = 'master'            # branch name OR commit hash (or tag) to check out in A3; use a commit hash for an exactly reproducible run
GITHUB_TOKEN = None            # optional; not needed for the public repo (e.g. userdata.get('GITHUB_TOKEN') if you ever use a private fork)
LOCAL_ROOT = '/content/cache013'                                         # big temporary features (up to ~15 GB for the readout check, 추정) on the Colab local disk
os.makedirs(WORK013, exist_ok=True)
for k, v in (('WORK002', WORK002), ('WORK013', WORK013)):
    print('%-9s = %s  exists=%s' % (k, v, os.path.exists(v)))
for cond in ('single_ind', 'multi_60_ind_160'):
    for ts in (0, 1, 2):
        p = '%s/train_runs/outputs/exp002_%s_s%d/checkpoints/model_latest.pth' % (WORK002, cond, ts)
        print('checkpoint %s seed %d: %s exists=%s' % (cond, ts, p, os.path.exists(p)))
'''))

C.append(code(r'''
# A3. Code: git clone of the PUBLIC repo REPO_URL, checked out at REPO_REF (settings in cell A2). No embedded snapshot.
# The repo ref must contain 002, 004 (colab/exp045_tools.py etc.), 009 (src, config.yaml) and 013 files (commit and push them first).
EXPECTED_CFG002_SHA = '@@CFG002SHA@@'    # 002/results.json config_sha256 read at notebook build time (@@STAMP@@)
import json, os, subprocess
REPO = '/content/mylab013'
EXPS = REPO + '/experiments'
EXP002, EXP004, EXP009, EXP013 = EXPS + '/002_single_size_sweep', EXPS + '/004_depth_truncation', EXPS + '/009_positive_control_depth', EXPS + '/013_erasure_residual_info'
def git(*a, check=True):
    r = subprocess.run(['git', '-C', REPO] + list(a), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError('git %s failed: %s' % (' '.join(a[:2]), r.stderr.replace(GITHUB_TOKEN or '@@none@@', '***')))
    return r.stdout.strip()
clone_url = REPO_URL
if GITHUB_TOKEN:        # optional (public repo needs none)
    clone_url = REPO_URL.replace('https://', 'https://%s@' % GITHUB_TOKEN)
if not os.path.isdir(REPO + '/.git'):
    r = subprocess.run(['git', 'clone', clone_url, REPO], capture_output=True, text=True)
    assert r.returncode == 0, 'clone failed: ' + r.stderr.replace(GITHUB_TOKEN or '@@none@@', '***')
else:
    git('remote', 'set-url', 'origin', clone_url)
    git('fetch', '--tags', '--force', 'origin', '+refs/heads/*:refs/remotes/origin/*')
git('remote', 'set-url', 'origin', REPO_URL)          # do not keep the token in .git/config
if subprocess.run(['git', '-C', REPO, 'rev-parse', '--verify', '-q', 'origin/' + REPO_REF + '^{commit}'], capture_output=True).returncode == 0:
    target = 'origin/' + REPO_REF                       # branch name
else:
    target = REPO_REF                                   # commit hash or tag
git('checkout', '--force', '--detach', target)
HEAD_SHA = git('rev-parse', 'HEAD')
CODE_VERSION = {'repo_url': REPO_URL, 'requested_ref': REPO_REF, 'git_head': HEAD_SHA, 'head_subject': git('log', '-1', '--format=%s'), 'head_date': git('log', '-1', '--format=%cI')}
print('code at HEAD', HEAD_SHA, '(ref %s)' % REPO_REF)
print(git('log', '--oneline', '-1'))
os.makedirs(WORK013, exist_ok=True)
json.dump(CODE_VERSION, open(WORK013 + '/code_version.json', 'w'), indent=1)     # also in results.json (code_version, via env vars set in A6)
for p in (EXP002 + '/run.py', EXP004 + '/run.py', EXP013 + '/run.py', EXP013 + '/config.yaml', EXP013 + '/src/exp013/stages.py', EXP009 + '/src/exp009/erase.py', EXP009 + '/config.yaml', EXP013 + '/src/exp013/core.py', EXP013 + '/src/exp013/lib.py',
          EXP004 + '/colab/exp045_tools.py', EXP002 + '/colab/requirements_colab.txt', EXP004 + '/colab/requirements_colab_extra.txt'):
    assert os.path.isfile(p), 'missing in this ref: ' + p + ' (commit and push the code, or fix REPO_REF)'
import sys
sys.path.insert(0, EXP004 + '/colab')
import importlib, exp045_tools as T
importlib.reload(T)
print('exp045_tools loaded from', T.__file__)
'''))

C.append(code(r'''
# A4. Python environment: uv venv (3.9) with the 002 pins (+ matplotlib). Colab has no conda: documented exception to CLAUDE.md (same as 002/003/004/009).
# USE_SYSTEM_PYTHON=True uses Colab's own Python/torch; results are then NOT comparable with the 002 environment (record it). torch 2.0.x comes from the 002 pins.
USE_SYSTEM_PYTHON = False
VENV = '/content/venv013'
import hashlib, os, sys
REQ = [EXP002 + '/colab/requirements_colab.txt', EXP004 + '/colab/requirements_colab_extra.txt']
if USE_SYSTEM_PYTHON:
    PY = sys.executable
else:
    PY = VENV + '/bin/python'
    MARK = VENV + '/.exp013_ok_' + hashlib.sha256(b''.join(open(r, 'rb').read() for r in REQ)).hexdigest()[:12]
    if not os.path.exists(MARK):
        assert sh('pip install -q uv') == 0
        if not os.path.exists(PY):
            assert sh('uv venv --python 3.9 %s' % VENV) == 0
        assert sh('uv pip install --python %s --index-strategy unsafe-best-match -r %s -r %s' % (PY, REQ[0], REQ[1])) == 0
        open(MARK, 'w').write('ok\n')
CHK = "import sys,pkg_resources,torch,numpy,cv2,decord,hydra,yaml;print(sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available(),'numpy',numpy.__version__)"
assert sh('%s -c "%s"' % (PY, CHK)) == 0, 'environment check failed'
'''))

C.append(code(r'''
# A5. ACWM-Phys checkout at the pinned commit (== 002 config.yaml code.commit). Needed to unpickle the 002 checkpoints. NO resume patch (nothing is trained).
import re, subprocess, os
cfg002_txt = open(EXP002 + '/config.yaml').read()
COMMIT = re.search(r'commit:\s*"([0-9a-f]{40})"', cfg002_txt).group(1)
SRC = '/content/dinowm/acwm_phys_src'
if not os.path.isdir(SRC + '/.git'):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/jongmin-s-multicore/thread0.git', SRC], check=True)
    subprocess.run(['git', '-C', SRC, 'checkout', '--detach', COMMIT], check=True)
head = subprocess.run(['git', '-C', SRC, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
assert head == COMMIT, (head, COMMIT)
print('ACWM source at', head)
'''))

C.append(code(r'''
# A6. Environment variables, Drive-backed results/logs, stage helpers.
import os, json, shutil
for d in (LOCAL_ROOT + '/exp013_cache', '/content/pilot013/exp013_cache'):
    os.makedirs(d, exist_ok=True)
base_env = dict(os.environ)
base_env.update({
    'DINO_WORK': '/content/dinowm', 'DATASET_DIR': '/content/dinowm/data',
    'DINO_TRAIN': WORK002 + '/train_runs',                                          # READ-ONLY use: checkpoints (<DINO_TRAIN>/outputs/exp002_multi_60_ind_160_s<seed>/)
    'EXP002_CACHE': WORK002 + '/exp002_cache',                                      # READ-ONLY use: q1_tokens
    'ACWM_PHYS_SRC': SRC, 'WANDB_MODE': 'disabled', 'HYDRA_FULL_ERROR': '1', 'PYTHONUNBUFFERED': '1',
    'EXP013_CODE_REPO_URL': REPO_URL, 'EXP013_CODE_REF': REPO_REF,                  # recorded in results.json code_version (git_head is read from the clone)
    'MPLBACKEND': 'Agg',                                                            # also forced inside exp045_tools (the Colab kernel's inline backend is invalid in the venv)
    'PATH': os.path.dirname(PY) + os.pathsep + os.environ['PATH'],
})
# NOTE: PYTORCH_CUDA_ALLOC_CONF is deliberately not set (expandable_segments is unknown to torch 2.0.x; 003 commit 16edd98).
if not USE_SYSTEM_PYTHON:
    base_env['VIRTUAL_ENV'] = VENV
ENV_MAIN = dict(base_env, EXP013_CACHE=LOCAL_ROOT + '/exp013_cache')
ENV_PILOT = dict(base_env, EXP013_CACHE='/content/pilot013/exp013_cache')
print(json.dumps({k: ENV_MAIN[k] for k in ('DINO_TRAIN', 'EXP002_CACHE', 'EXP013_CACHE', 'ACWM_PHYS_SRC', 'MPLBACKEND')}, indent=1))
print('013:', T.link_dirs(WORK013, EXP013, ('results', 'logs')))
os.makedirs(WORK013 + '/results_pilot', exist_ok=True)

MODES = {      # pilot and main NEVER share a results folder: results.json of the pilot is written to results_pilot only and says pilot: true
    'pilot': dict(env=ENV_PILOT, results=WORK013 + '/results_pilot', timing=WORK013 + '/timing_pilot.json'),
    'main': dict(env=ENV_MAIN, results=WORK013 + '/results', timing=WORK013 + '/timing_main.json'),
}

def run013(stage, extra=(), mode='main', dry=False):
    m = MODES[mode]
    args = ['--results-dir', m['results']] + list(extra) + (['--dry-run'] if dry else [])
    rec = T.run_stage(PY, EXP013, stage, args, m['env'], timing_file=None if dry else m['timing'], label='013:%s%s' % (stage, '(dry)' if dry else ''))
    if rec['rc'] == 5 and not dry:
        raise SystemExit('GATE (exit code 5, NO attempt consumed): the stage-0 gate is missing / stale / NOT PASSED, so stage 1 refused to run. Read the "GATE" line above. Run the gate cell (E0) first; if it did not pass, STOP and report (do not override for the main run).')
    if rec['rc'] == 4 and not dry:
        raise SystemExit('PREFLIGHT FAILED (exit code 4, NO attempt consumed). Nothing was computed. Read the "PREFLIGHT ERROR" line above / log %s, fix the input (checkpoint folder, Q1 cache, metadata.pt, config), then run this cell again.' % rec['log'])
    T.require_ok(rec, m['results'])
    return rec

def save_results_json(mode):
    # aggregate writes results.json into the results folder (Drive); for the main run also <clone>/experiments/013.../results.json (lost with the session)
    src = (EXP013 + '/results.json') if mode == 'main' else (MODES[mode]['results'] + '/results.json')
    dst = MODES[mode]['results'] + '/results.json'
    if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
    print(('results.json present: ' + dst) if os.path.exists(dst) else ('no results.json at ' + dst))
'''))

C.append(code(r'''
# A6b. metadata.pt of the training manifests (needed by exp002 model_io.action_stats_for_run -> ManifestDataset: action mean/std). Only metadata.pt files, no mp4.
# Same cell as 009 A6b. Sources in order: (a) already local; (b) DATA_SRC/<split>/metadata.pt or DATA_SRC/cloth_sim/<split>/metadata.pt; (c) member of WORK002/cloth_sim_data.tar (unsafe members refused).
import json, os, shutil, tarfile
from pathlib import Path
need = {}                                   # local dir (as written in the manifest) -> train seeds
for cond in ('single_ind', 'multi_60_ind_160'):
  for ts in (0, 1, 2):
    mp = Path(WORK002) / 'train_runs' / 'outputs' / ('exp002_%s_s%d' % (cond, ts)) / 'train_manifest.json'
    assert mp.exists(), 'missing ' + str(mp)
    for ep in json.load(open(mp))['episodes']:
        need.setdefault(ep['dir'], set()).add('%s_s%d' % (cond, ts))
print('episode dirs in the manifests (both conditions):', {d: sorted(v) for d, v in need.items()})
def safe_member(name):
    return not name.startswith('/') and '..' not in name.split('/')
missing = []
tar_path = Path(WORK002) / 'cloth_sim_data.tar'
tf = None
for d in sorted(need):
    dest = Path(d) / 'metadata.pt'
    split = Path(d).name
    if dest.exists():
        print('local ok   ', dest); continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = None
    for cand in (Path(DATA_SRC) / split / 'metadata.pt', Path(DATA_SRC) / 'cloth_sim' / split / 'metadata.pt'):
        if cand.exists():
            shutil.copyfile(cand, dest); got = 'copied from ' + str(cand); break
    if got is None and tar_path.exists():
        if tf is None:
            tf = tarfile.open(tar_path, 'r:*')
        for m in tf:
            if not safe_member(m.name) or not m.isfile():
                continue
            parts = m.name.split('/')
            if parts[-1] == 'metadata.pt' and len(parts) >= 2 and parts[-2] == split:
                with tf.extractfile(m) as fi, open(dest, 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
                got = 'extracted %s from %s' % (m.name, tar_path); break
    if got is None:
        missing.append(d); print('!! NOT FOUND', dest)
    else:
        print(got, '->', dest)
if tf is not None:
    tf.close()
if missing:
    raise SystemExit('metadata.pt not found for splits %s. Fix DATA_SRC in cell A2 or put cloth_sim_data.tar into %s. Nothing was run.' % ([Path(d).name for d in missing], WORK002))
print('metadata.pt present for all %d episode dirs' % len(need))
'''))

C.append(code(r'''
# A7. Disk. Estimate (추정): the readout check writes the token features of 7 depths (about 2.2 GB each = 15 GB) per method and deletes them afterwards; `measure` writes nothing big.
T.disk_report(['/content', WORK013], need_gb={'013 tmp features of the readout check (7 depths x about 2.2 GB, one method at a time)': 16})
free_local = shutil.disk_usage('/content').free / 1e9
print('free local disk: %.0f GB' % free_local, ('  !! LOW (< 25 GB)' if free_local < 25 else ''))
'''))


C.append(md(r'''
# B. Check (read-only; nothing is trained)
'''))

C.append(code(r'''
# B1. Inputs on Drive: the 002 artefacts of BOTH conditions (checkpoints DONE, final checkpoint), Q1 cache, 002 config fingerprint. Stops with a clear message if something is missing. Writes nothing.
# T.check_inputs covers condition (b) multi_60_ind_160 (and the Q1 cache + config fingerprint); condition (a) single_ind is checked below.
ok, bad, lines = T.check_inputs(WORK002, [0, 1, 2], suffixes=('',), expected_epoch=15, expected_cfg_sha=EXPECTED_CFG002_SHA, cfg002_path=EXP002 + '/config.yaml', need_manifest=False)
print('\n'.join(lines))
for ts in (0, 1, 2):
    rd = '%s/train_runs/outputs/exp002_single_ind_s%d' % (WORK002, ts)
    for rel in ('DONE', 'hydra.yaml', 'train_manifest.json', 'checkpoints/model_latest.pth'):
        good = os.path.exists(rd + '/' + rel)
        print(('[ok]   ' if good else '[MISSING] ') + 'exp002_single_ind_s%d/%s' % (ts, rel))
        if not good:
            ok = False; bad.append('exp002_single_ind_s%d/%s' % (ts, rel))
if not ok:
    raise SystemExit('INPUTS MISSING - stopping (nothing was run):\n  - ' + '\n  - '.join(bad) + '\n  Fix the paths in cell A2 (WORK002).')
print('inputs OK')
'''))

C.append(code(r'''
# B2. 013 `check` (config consistency with 009/002, checkpoint folder structure of both conditions, Q1 cache, metadata.pt, gate and fit status) and the dry runs
# (they print the plan: expected "units 6 ... configs per unit 43"). No GPU work, no attempts recorded.
run013('check', mode='main')
run013('gate', mode='main', dry=True)
run013('fit', mode='main', dry=True)
run013('measure', mode='main', dry=True)
run013('readout', mode='main', dry=True)
run013('aggregate', mode='main', dry=True)
print(T.read_yaml_values(EXP013 + '/config.yaml', ['pilot', 'seed', 'base.conditions', 'predictor.train_seeds', 'predictor.random_init_seed', 'erasure.fit_seed', 'erasure.random_dir_seed', 'success.bootstrap']))
'''))

C.append(md(r'''
# C. Selftest (CPU, synthetic data + toy model; expected: failed=0)
'''))

C.append(code(r'''
# C1. selftest (does not touch Drive inputs; it starts subprocesses that must exit with code 5 and 4 on purpose).
rec = run013('selftest', mode='main')
print('selftest %.0f s, log %s' % (rec['seconds'], rec['log']))
'''))

C.append(md(r'''
# D. Approvals and OPTIONAL pilot (1 train seed, condition (b), first 10 test episodes per level) with the REAL checkpoint, results in `exp013_work/results_pilot`
Uses only existing CLI flags (`--conditions multi_60_ind_160 --train-seeds 0 --episodes-per-level 10 --results-dir`); the pilot results.json says `pilot: true` and lists the reasons. It measures time and VRAM and finds errors. Pilot and main never share a results folder or cache. The pilot runs its own stage-0 gate and eraser fit (CPU) first; if the gate does not pass, the pilot is refused as well (exit 5), unless `PILOT_OVERRIDE_GATE = True` (then results.json says so).
'''))

C.append(code(r'''
# D0. APPROVALS (CLAUDE.md: GPU/cost-heavy runs need human approval). Set by the human only.
APPROVED_PILOT_013 = False
APPROVED_MAIN_013 = False
ALLOW_MAIN_WITH_PILOT_TRUE = False      # True = run the main although config.yaml has pilot: true (results.json would then say pilot: true with that reason). Normally the human sets pilot: false in config.yaml first.
PILOT_OVERRIDE_GATE = False             # True = the pilot may run although the stage-0 gate did not pass (debugging the GPU path only; recorded in results.json). Never for the main run.
PILOT_CONDITIONS = 'multi_60_ind_160'   # CLI --conditions for the pilot only
PILOT_TRAIN_SEEDS = '0'                 # CLI --train-seeds for the pilot only
PILOT_EPISODES_PER_LEVEL = '10'         # CLI --episodes-per-level for the pilot only
print('APPROVED_PILOT_013 =', APPROVED_PILOT_013, '| APPROVED_MAIN_013 =', APPROVED_MAIN_013)
'''))

C.append(md(r'''
# E. Stage 0 gate and eraser fits (CPU only; Q1 cache from Drive, a few minutes, 추정), then stage 1
'''))

C.append(code(r'''
# E0. Stage 0 gate + eraser fits for the MAIN results folder (CPU; no approval flag needed because nothing runs on the GPU and nothing is trained).
# gate: writes results/stage0.json ("passed" true/false). fit: scalar LEACE per train seed + level means into the local cache. Both are idempotent.
rec = run013('gate', mode='main')
g = json.load(open(WORK013 + '/results/stage0.json'))
print('GATE passed =', g['passed'], '| pairs passing:', g['n_pairs_pass'], 'of', g['n_pairs'], '(threshold ratio >=', g['ratio_threshold'], 'in >=', g['min_pairs'], 'pairs)')
for p in g['pairs']:
    print('  regressor %-5s -> test %-5s  ratio %.3f  %s' % (p['regressor_level'], p['test_level'], p['ratio_other_over_same'], 'pass' if p['passes'] else ''))
if g['passed']:
    run013('fit', mode='main')
else:
    print('GATE NOT PASSED: stage 1 (measure / readout) will refuse to run (exit code 5). Report this to the human; do not override for the main run.')
'''))

C.append(code(r'''
# E1. Pilot (optional): gate + fit in the pilot folder, then measure (condition (b), seed 0, first 10 episodes per level) and aggregate. Time and sampled VRAM go to exp013_work/timing_pilot.json.
import time
if not APPROVED_PILOT_013:
    print('사람 승인 필요(CLAUDE.md): APPROVED_PILOT_013 = False -> nothing was run.')
else:
    t0 = time.time()
    ex = ['--conditions', PILOT_CONDITIONS, '--train-seeds', PILOT_TRAIN_SEEDS, '--episodes-per-level', PILOT_EPISODES_PER_LEVEL]
    run013('gate', ex[:4], mode='pilot')
    run013('fit', ex[:4], mode='pilot')
    for st in ('measure', 'aggregate'):
        extra = ex + (['--override-gate'] if (PILOT_OVERRIDE_GATE and st == 'measure') else [])
        rec = run013(st, extra, mode='pilot')
        print('>>> %s: %.0f s, sampled peak GPU memory %s MiB (base %s), log %s' % (st, rec['seconds'], rec['vram_peak_sampled_mib'], rec['vram_base_mib'], rec['log']))
    save_results_json('pilot')
    rp = json.load(open(WORK013 + '/results_pilot/results.json'))
    print('pilot results.json: pilot=%s reasons=%s' % (rp['pilot'], rp['pilot_reasons']))
    print('pilot total %.0f s' % (time.time() - t0))
'''))

C.append(code(r'''
# E2. Pilot timing and a (rough) re-estimate of the main run.
import json
tp = WORK013 + '/timing_pilot.json'
if os.path.exists(tp):
    for r in json.load(open(tp)):
        if r['rc'] == 0 and '(dry)' not in r['label']:
            print('%-24s %7.0f s  peak VRAM (sampled whole GPU) %s MiB  log %s' % (r['label'], r['seconds'], r['vram_peak_sampled_mib'], r['log']))
    print('The pilot unit evaluates 10 of 100 test episodes per level (90 of 900) with 43 configs x 2 predictors. The main run has 6 units x 900 episodes (10x the pilot episodes) plus the readout check. The log line "unit ... done in N s" gives the cost of one pilot unit.')
else:
    print('no pilot timing yet')
print('Before any measurement, README estimate (추정): measure about 1-1.5 h and readout about 1-1.5 h on an L4 (no measurement; derived from the predictor FLOPs and the 009 timing_main.json, see README).')
'''))

C.append(md(r'''
# Main stage 1 (needs `APPROVED_MAIN_013 = True` in cell D0 and a passed gate)
'''))

C.append(code(r'''
# E3. measure: 6 units (2 conditions x 3 train seeds), 900 test episodes each, 43 erasure configs x (trained + random-init predictor). Finished units (Drive part files) are skipped;
# rerun this cell after a disconnect (A and B first). Refuses with exit code 5 when the stage-0 gate did not pass.
if not APPROVED_MAIN_013:
    print('사람 승인 필요(CLAUDE.md): APPROVED_MAIN_013 = False -> nothing was run.')
else:
    v = T.read_yaml_values(EXP013 + '/config.yaml', ['pilot', 'seed', 'base.conditions', 'predictor.train_seeds', 'erasure.methods'])
    print('013 config:', v)
    if v['pilot'] is not False and not ALLOW_MAIN_WITH_PILOT_TRUE:
        raise SystemExit('013 config.yaml has pilot: %s. The human sets pilot: false (README) or ALLOW_MAIN_WITH_PILOT_TRUE=True. Nothing was run.' % v['pilot'])
    rec = run013('measure', mode='main')
    print('>>> measure done in %.0f s (sampled peak GPU %s MiB), log %s' % (rec['seconds'], rec['vram_peak_sampled_mib'], rec['log']))
'''))

C.append(code(r'''
# E4. readout (erasure check): 009-style linear size readout R2 at blocks 0..6 with the input erasure applied, 6 units x 3 methods (none, leace_scalar, level_mean_match).
# Needs about 16 GB of temporary local disk (one method at a time). Same gate rule and same approval flag as E3.
if not APPROVED_MAIN_013:
    print('사람 승인 필요(CLAUDE.md): APPROVED_MAIN_013 = False -> nothing was run.')
else:
    assert shutil.disk_usage('/content').free / 1e9 > 20, 'less than 20 GB free on /content (the readout check needs about 16 GB temporary, 추정)'
    rec = run013('readout', mode='main')
    print('>>> readout done in %.0f s (sampled peak GPU %s MiB), log %s' % (rec['seconds'], rec['vram_peak_sampled_mib'], rec['log']))
'''))

C.append(md(r'''
# F. Aggregate (CPU; Delta_use with bootstrap, per-method verdicts, stage 2, results.json)
'''))

C.append(code(r'''
# F1. aggregate -> results/results.json (Drive) and <clone>/experiments/013.../results.json (copied). pilot is true in results.json whenever config pilot is true, a CLI override was used, or the results folder differs.
if not APPROVED_MAIN_013:
    print('APPROVED_MAIN_013 = False -> nothing was run (aggregate follows the main measure).')
else:
    rec = run013('aggregate', mode='main')
    save_results_json('main')
    r = json.load(open(WORK013 + '/results/results.json'))
    print({k: r[k] for k in ('pilot', 'pilot_reasons', 'conditions', 'train_seeds')})
    print('gate:', r['gate']['message'])
    print({m: v['status'] for m, v in r['verdict'].items()})
    print('missing units:', r['missing_units'])
'''))

C.append(md(r'''
# G. Copy results out
'''))

C.append(code(r'''
# G1. What is on Drive now. Optional zip download (DOWNLOAD_BUNDLE).
DOWNLOAD_BUNDLE = False
import glob
for f in sorted(glob.glob(WORK013 + '/*') + glob.glob(WORK013 + '/results/*') + glob.glob(WORK013 + '/results_pilot/*')):
    print(f, os.path.getsize(f) if os.path.isfile(f) else '(dir)')
for f in ('results/attempts.json', 'results_pilot/attempts.json'):
    if os.path.exists(WORK013 + '/' + f):
        print(WORK013 + '/' + f, open(WORK013 + '/' + f).read().replace('\n', ' '))
if DOWNLOAD_BUNDLE:
    import zipfile
    zp = '/content/exp013_results_bundle.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for ext in ('json', 'log'):
            for f in glob.glob(WORK013 + '/**/*.' + ext, recursive=True):
                z.write(f, os.path.relpath(f, os.path.dirname(WORK013)))
    from google.colab import files
    files.download(zp)
'''))

C.append(code(r'''
# G2. Flush Drive writes before leaving.
from google.colab import drive
drive.flush_and_unmount()
'''))


def main():
    expected = json.loads((EXPS / "002_single_size_sweep" / "results.json").read_text())["config_sha256"]
    stamp = time.strftime("%F %T")
    for c in C:
        if c["cell_type"] == "code":
            c["source"] = [s.replace("@@STAMP@@", stamp).replace("@@CFG002SHA@@", expected) for s in c["source"]]
            if any("@@" in s and ("CFG002SHA" in s or "STAMP" in s) for s in c["source"]):
                raise SystemExit("placeholder left")
    nb = {"cells": C, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    out = HERE / "exp013_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", out, "cells", len(C))


if __name__ == "__main__":
    main()
