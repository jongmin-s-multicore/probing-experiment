#!/usr/bin/env python3
"""Regenerate colab/exp010_colab.ipynb:   python3 colab/build_notebook.py     (run again after ANY change to this file)
The notebook has NO embedded code: cell A3 git-clones the public repo (REPO_URL, REPO_REF set in cell A2) and records HEAD (exp010_work/code_version.json, results.json code_version).
The ref must contain 002 {src, config.yaml, results.json, colab/requirements_colab.txt}, 004 {src, run.py, config.yaml, results.json, colab/exp045_tools.py, colab/requirements_colab_extra.txt,
logs/colab_20261109/enc_cache_*.log}, 005 {src, run.py, config.yaml, results/plan.json, logs/colab_20261109/feat_*.log} and 010 {src, run.py, config.yaml}.
004/005 files are only run (run.py enc_cache / feat) or imported (exp045_tools); nothing there is modified. config.yaml values (Delta, task seeds, ...) are NOT changed by the notebook.
Local paths (/content/mylab010, /content/venv010, /content/cache010) and the Drive output (MyDrive/exp010_work) do not overlap with 006/009 (/content/mylab006, /content/mylab, ... exp006_work, exp009_work)
EXCEPT the data location /content/dinowm/data/... which is fixed by the absolute episode dirs inside the 002 train_manifest.json (identical read-only metadata.pt files)."""
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPS = HERE.parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


C = []
C.append(md(r'''
# Experiment 010 (source separation of the 005 "ours ~ random" result) on Google Colab
Order: **A** settings + setup, **B** inputs + dry runs + selftest, **R** (only if the 4 cache files are missing) regenerate the caches with the UNMODIFIED 004 `enc_cache` and 005 `feat`, **C** optional pilot, **D** main heads (approval gate), **E** report, **F** copy results out.
Every cell is idempotent; after a disconnect choose a GPU runtime, run A and B again and continue (heads skips finished prediction files; regenerated caches are restored from Drive).

What 010 reads (read-only): four feature files = 004 `tokens_k12.npy`, `rand_k12.npy` (encoder, k=12, trained / random-init) and 005 `feat_pred_out6.npy`, `feat_pred_out6_rand.npy` (predictor output of block 6, trained / random-init predictor), plus the 002 Q1 cache. **Those four files were only on a Colab session disk (004 README 6); they are NOT on Drive.** Section R regenerates them by running 004 `run.py enc_cache` (k_grid [0,2,...,12], random-init control, source train seed 0) and 005 `run.py feat` (train seed 0, random_init_seed 1234), both with their own config.yaml, and stores the four files on Drive (`exp010_work/cache004/enc`, `exp010_work/cache005`) so that later sessions only restore them.
Measured times of the original runs (Colab L4, `004_depth_truncation/results/colab/timing_main.json`): `enc_cache` 177 s, `feat` 137 s (excluding Drive reads/copies). The 010 heads estimate is the README estimate: about 3-3.5 h on L4 (추정, no measurement).

Human approval (CLAUDE.md): `APPROVED_REGEN_010` (cell R0, small GPU run, about 6 min plus copies), `APPROVED_PILOT_010` (C0) and `APPROVED_MAIN_010` (D0) are False by default and are set by you only. **The 010 hypothesis is still a draft (`notes/hypotheses.md` "010"): Delta (proposed 0.05), the task seeds (draft [2000, 2001, 2002]) and the GPU approval are your decisions; this notebook does not change `config.yaml`.** `config.yaml` has `pilot: true`; the main cell refuses while it is true (set it to false in the repo and push, or `ALLOW_MAIN_WITH_PILOT_TRUE`).
Outputs (Drive): `MyDrive/exp010_work/{results,results_pilot,logs,logs_regen,regen,cache004,cache005,timing_*.json,code_version.json}`.
Everything here is UNVERIFIED on Colab (written and checked on a CPU-only machine).
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
WORK002 = os.environ.get('EXP002_WORK_ROOT', DRIVE + '/exp002_work')     # READ-ONLY: train_runs/outputs/exp002_multi_60_ind_160_s0/ (checkpoint), exp002_cache/q1_tokens/
WORK010 = os.environ.get('EXP010_WORK_ROOT', DRIVE + '/exp010_work')     # OUTPUT (persistent): results, results_pilot, logs, regen, cache004, cache005, timing_*.json
DATA_SRC = os.environ.get('EXP002_DRIVE_MOUNT_SRC', DRIVE + '/dino-wm-videos/[data]/[cloth-sim] cloth_move')   # only <split>/metadata.pt is read from here (cell R1, regeneration only)
REPO_URL = 'https://github.com/jongmin-s-multicore/probing-experiment.git'     # PUBLIC repository (cloned in A3)
REPO_REF = 'master'            # branch name OR commit hash (or tag) to check out in A3; use a commit hash for an exactly reproducible run
GITHUB_TOKEN = None            # optional; not needed for the public repo
LOCAL = '/content/cache010'    # local copies of the four feature files (+ the full 004 enc_cache output when regenerating, about 30 GB, 추정 from the 004 log)
os.makedirs(WORK010, exist_ok=True)
for k, v in (('WORK002', WORK002), ('WORK010', WORK010)):
    print('%-8s = %s  exists=%s' % (k, v, os.path.exists(v)))
'''))

C.append(code(r'''
# A3. Code: git clone of the PUBLIC repo REPO_URL, checked out at REPO_REF (settings in cell A2). No embedded snapshot.
# The ref must contain 002, 004, 005 and 010 files (commit and push them first). HEAD is printed and written to exp010_work/code_version.json.
EXPECTED_CFG002_SHA = '@@CFG002SHA@@'    # 002/results.json config_sha256 read at notebook build time (@@STAMP@@)
import json, os, subprocess, sys
REPO = '/content/mylab010'
EXPS = REPO + '/experiments'
EXP002, EXP004, EXP005, EXP010 = EXPS + '/002_single_size_sweep', EXPS + '/004_depth_truncation', EXPS + '/005_inductive_bias_probe', EXPS + '/010_source_separation_ib'
def git(*a, check=True):
    r = subprocess.run(['git', '-C', REPO] + list(a), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError('git %s failed: %s' % (' '.join(a[:2]), r.stderr.replace(GITHUB_TOKEN or '@@none@@', '***')))
    return r.stdout.strip()
clone_url = REPO_URL
if GITHUB_TOKEN:
    clone_url = REPO_URL.replace('https://', 'https://%s@' % GITHUB_TOKEN)
if not os.path.isdir(REPO + '/.git'):
    r = subprocess.run(['git', 'clone', clone_url, REPO], capture_output=True, text=True)
    assert r.returncode == 0, 'clone failed: ' + r.stderr.replace(GITHUB_TOKEN or '@@none@@', '***')
else:
    git('remote', 'set-url', 'origin', clone_url)
    git('fetch', '--tags', '--force', 'origin', '+refs/heads/*:refs/remotes/origin/*')
git('remote', 'set-url', 'origin', REPO_URL)          # do not keep the token in .git/config
if subprocess.run(['git', '-C', REPO, 'rev-parse', '--verify', '-q', 'origin/' + REPO_REF + '^{commit}'], capture_output=True).returncode == 0:
    target = 'origin/' + REPO_REF
else:
    target = REPO_REF
git('checkout', '--force', '--detach', target)
HEAD_SHA = git('rev-parse', 'HEAD')
CODE_VERSION = {'repo_url': REPO_URL, 'requested_ref': REPO_REF, 'git_head': HEAD_SHA, 'head_subject': git('log', '-1', '--format=%s'), 'head_date': git('log', '-1', '--format=%cI')}
print('code at HEAD', HEAD_SHA, '(ref %s)' % REPO_REF)
print(git('log', '--oneline', '-1'))
os.makedirs(WORK010, exist_ok=True)
json.dump(CODE_VERSION, open(WORK010 + '/code_version.json', 'w'), indent=1)     # also in results.json: code_version (env vars set in A6)
need = [EXP002 + '/config.yaml', EXP002 + '/colab/requirements_colab.txt', EXP004 + '/run.py', EXP004 + '/config.yaml', EXP004 + '/colab/exp045_tools.py', EXP004 + '/colab/requirements_colab_extra.txt',
        EXP005 + '/run.py', EXP005 + '/config.yaml', EXP005 + '/results/plan.json', EXP010 + '/run.py', EXP010 + '/config.yaml', EXP010 + '/src/exp010/stages.py']
miss = [p for p in need if not os.path.isfile(p)]
assert not miss, 'files missing at this ref (commit and push them): %s' % miss
st = open(EXP010 + '/src/exp010/stages.py').read()
assert 'EXP010_CODE_REPO_URL' in st and 'def missing_inputs' in st, 'this ref has an older 010 code (no code_version / preflight): push the current 010 code'
sys.path.insert(0, EXP004 + '/colab')
import importlib, exp045_tools as T
importlib.reload(T)
print('exp045_tools loaded from', T.__file__)
'''))

C.append(code(r'''
# A4. Python environment: uv venv (3.9) with the 002 pins (+ matplotlib), the same requirement files as the 004/005/009 notebooks. Colab has no conda: documented exception to CLAUDE.md.
# USE_SYSTEM_PYTHON=True uses Colab's own Python/torch; results are then NOT comparable with the 002 environment (record it).
USE_SYSTEM_PYTHON = False
VENV = '/content/venv010'
import hashlib, os, sys
REQ = [EXP002 + '/colab/requirements_colab.txt', EXP004 + '/colab/requirements_colab_extra.txt']
if USE_SYSTEM_PYTHON:
    PY = sys.executable
else:
    PY = VENV + '/bin/python'
    MARK = VENV + '/.exp010_ok_' + hashlib.sha256(b''.join(open(r, 'rb').read() for r in REQ)).hexdigest()[:12]
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
# A5. ACWM-Phys checkout at the pinned commit (== 002 config.yaml code.commit). Needed to unpickle the 002 checkpoint (regeneration only). Nothing is trained.
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
# A6. Environment variables, Drive-backed logs, cache locations, stage helpers.
import os, json, shutil
LOC_ENC, LOC_FEAT = LOCAL + '/exp004_cache/enc', LOCAL + '/exp005_cache'              # EXP004_CACHE/enc/tokens_k12.npy ... ; EXP005_CACHE/feat_pred_out6*.npy
DRV_ENC, DRV_FEAT = WORK010 + '/cache004/enc', WORK010 + '/cache005'                  # persistent copies of the four files
CACHE_FILES = [(LOC_ENC + '/tokens_k12.npy', DRV_ENC + '/tokens_k12.npy'), (LOC_ENC + '/rand_k12.npy', DRV_ENC + '/rand_k12.npy'),
               (LOC_FEAT + '/feat_pred_out6.npy', DRV_FEAT + '/feat_pred_out6.npy'), (LOC_FEAT + '/feat_pred_out6_rand.npy', DRV_FEAT + '/feat_pred_out6_rand.npy')]
for d in (LOC_ENC, LOC_FEAT, DRV_ENC, DRV_FEAT):
    os.makedirs(d, exist_ok=True)
base_env = dict(os.environ)
base_env.update({
    'DINO_WORK': '/content/dinowm', 'DATASET_DIR': '/content/dinowm/data',          # same data location as the 002 session (manifest episode dirs are absolute paths)
    'DINO_TRAIN': WORK002 + '/train_runs',                                          # READ-ONLY use: checkpoint of seed 0 (regeneration only)
    'EXP002_CACHE': WORK002 + '/exp002_cache',                                      # READ-ONLY use: Q1 cache (frames, mask_feat, rows)
    'EXP004_CACHE': LOCAL + '/exp004_cache', 'EXP005_CACHE': LOCAL + '/exp005_cache',
    'ACWM_PHYS_SRC': SRC, 'WANDB_MODE': 'disabled', 'HYDRA_FULL_ERROR': '1', 'PYTHONUNBUFFERED': '1',
    'EXP010_CODE_REPO_URL': REPO_URL, 'EXP010_CODE_REF': REPO_REF,                  # recorded in results.json code_version (git_head is read from the clone)
    'MPLBACKEND': 'Agg',                                                            # the Colab kernel's inline backend is invalid in the venv (also forced inside exp045_tools)
    'PATH': os.path.dirname(PY) + os.pathsep + os.environ['PATH'],
})
# NOTE: PYTORCH_CUDA_ALLOC_CONF is deliberately not set (expandable_segments is unknown to torch 2.0.x; 003 commit 16edd98).
if not USE_SYSTEM_PYTHON:
    base_env['VIRTUAL_ENV'] = VENV
ENV = base_env
print(json.dumps({k: ENV[k] for k in ('DINO_TRAIN', 'EXP002_CACHE', 'EXP004_CACHE', 'EXP005_CACHE', 'ACWM_PHYS_SRC')}, indent=1))
print('010:', T.link_dirs(WORK010, EXP010, ('logs',)))
os.makedirs(WORK010 + '/results_pilot', exist_ok=True)
os.makedirs(WORK010 + '/logs_regen', exist_ok=True)
MODES = {'pilot': dict(results=WORK010 + '/results_pilot', timing=WORK010 + '/timing_pilot.json'),
         'main': dict(results=WORK010 + '/results', timing=WORK010 + '/timing_main.json')}
REGEN = {'004': dict(exp=EXP004, stage='enc_cache', results=WORK010 + '/regen/results_004'),
         '005': dict(exp=EXP005, stage='feat', results=WORK010 + '/regen/results_005')}

def run010(stage, extra=(), mode='main', dry=False, allow_rc=(0,)):
    m = MODES[mode]
    args = ['--results-dir', m['results']] + list(extra) + (['--dry-run'] if dry else [])
    rec = T.run_stage(PY, EXP010, stage, args, ENV, timing_file=None if dry else m['timing'], label='010:%s%s' % (stage, '(dry)' if dry else ''))
    if rec['rc'] == 4:
        raise SystemExit('PREFLIGHT STOP (rc 4, no attempt used): see the log %s . Missing input files: run section R (regeneration) or restore the caches (B2).' % rec['log'])
    if rec['rc'] not in allow_rc:
        T.require_ok(rec, m['results'])
    return rec

def run_regen(which, dry=False):
    r = REGEN[which]
    args = ['--results-dir', r['results']] + (['--dry-run'] if dry else [])
    rec = T.run_stage(PY, r['exp'], r['stage'], args, ENV, timing_file=None if dry else WORK010 + '/timing_regen.json', label='%s:%s%s' % (which, r['stage'], '(dry)' if dry else ''))
    if rec['log'] and os.path.exists(rec['log']):
        shutil.copy(rec['log'], WORK010 + '/logs_regen/')               # the clone's logs/ is lost with the session
    T.require_ok(rec, r['results'])
    return rec

def save_results_json(dest_dir):
    src = EXP010 + '/results.json'          # report writes <clone>/experiments/010.../results.json (lost with the session): copy to Drive right away
    if os.path.exists(src):
        shutil.copyfile(src, dest_dir + '/results.json')
        print('copied', src, '->', dest_dir + '/results.json')
    else:
        print('no', src)

def copy_atomic(src, dst):
    """Copy src -> dst via a .tmp file and rename; skip when dst exists with the same size."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return 'same size, kept'
    shutil.copyfile(src, dst + '.tmp')
    os.replace(dst + '.tmp', dst)
    return 'copied %.2f GB' % (os.path.getsize(dst) / 1e9)

REGEN_RAN = set()          # which regeneration stages ran in THIS session (cache_record.json is only (re)written then, or when it does not exist)

def cache_status():
    return [(l, d, os.path.exists(l), os.path.exists(d)) for l, d in CACHE_FILES]
'''))

C.append(code(r'''
# A7. Disk. Estimates (추정): regeneration needs about 30 GB local for the full 004 enc_cache output (k_grid 0..12 + random-init + pixel; "tokens_total_gb 30.3" in the 004 enc_cache log)
# plus the 005 pred_out6 features; Drive receives only the four files (tokens_k12 / rand_k12 about 2.17 GB each per the same log; the two feat files: size not measured here).
T.disk_report(['/content', WORK010], need_gb={'regeneration, local (only if section R runs)': 35})
'''))

C.append(md(r'''
# B. Inputs and dry runs (read-only; nothing is trained)
'''))

C.append(code(r'''
# B1. 002 inputs on Drive: Q1 cache and the seed-0 checkpoint folder. The folder names are printed next to what is really in Drive (006 failed 3 times on a missing `exp002_` prefix).
want_run = 'exp002_multi_60_ind_160_s0'
outputs = WORK002 + '/train_runs/outputs'
print('expected checkpoint:', outputs + '/' + want_run + '/checkpoints/model_latest.pth')
print('folders really in', outputs, ':', sorted(os.listdir(outputs)) if os.path.isdir(outputs) else '(missing)')
ok, bad, lines = T.check_inputs(WORK002, [0], suffixes=('',), expected_epoch=15, expected_cfg_sha=EXPECTED_CFG002_SHA, cfg002_path=EXP002 + '/config.yaml', need_manifest=False)
print('\n'.join(lines))
if not ok:
    raise SystemExit('INPUTS MISSING - stopping (nothing was run):\n  - ' + '\n  - '.join(bad) + '\n  Fix WORK002 in cell A2.')
print('002 inputs OK')
'''))

C.append(code(r'''
# B2. The four feature files: restore from Drive (exp010_work/cache004, cache005) to the local disk when they are there. Sizes and sha256 are checked against regen_record.json if it exists.
VERIFY_SHA = True
rec_path = WORK010 + '/cache_record.json'
record = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
for loc, drv, has_l, has_d in cache_status():
    if not has_l and has_d:
        print('restore', os.path.basename(drv), '<-', copy_atomic(drv, loc))
print('%-26s %-8s %-8s' % ('file', 'local', 'Drive'))
for loc, drv, has_l, has_d in cache_status():
    print('%-26s %-8s %-8s' % (os.path.basename(loc), has_l, has_d))
    if has_l and os.path.basename(loc) in record.get('files', {}):
        r = record['files'][os.path.basename(loc)]
        assert os.path.getsize(loc) == r['bytes'], '%s size differs from cache_record.json' % loc
        if VERIFY_SHA:
            assert T.sha256_file(loc) == r['sha256'], '%s sha256 differs from cache_record.json' % loc
if record:
    print('cache_record.json present (created %s); sizes%s verified for the local files' % (record.get('created_utc'), ' and sha256' if VERIFY_SHA else ''))
CACHES_PRESENT = all(x[2] for x in cache_status())
print('ALL FOUR FEATURE FILES PRESENT LOCALLY:', CACHES_PRESENT, '' if CACHES_PRESENT else '-> run section R (needs APPROVED_REGEN_010 = True)')
'''))

C.append(code(r'''
# B3. 010 dry runs, `check` (rc 1 while a cache file is missing; that is expected before section R and does not use an attempt) and the synthetic selftest (CPU).
# The real predictor cannot run on CPU (Attention.bias.to('cuda'), 004 README); the selftest uses synthetic data.
run010('check', dry=True, allow_rc=(0, 1))   # check ignores --dry-run and returns 1 while a file is missing
rc = run010('check', allow_rc=(0, 1))['rc']
print('check rc =', rc, '(0 = all files present)')
run010('heads', dry=True)       # prints "heads plan" with the head counts (README: representations 6000 + area 1500 + oracle 3000)
run010('report', dry=True)
rec = run010('selftest')
print('selftest %.0f s, log %s' % (rec['seconds'], rec['log']))
print(T.read_yaml_values(EXP010 + '/config.yaml', ['pilot', 'seed', 'seeds.head', 'seeds.task_a', 'seeds.eval', 'eval.n_inputs', 'tost.delta', 'tost.alpha', 'subexps']))
print('(seeds.task_a and tost.delta are DRAFT values awaiting your decision, notes/hypotheses.md 010; they are shown, not changed)')
'''))

C.append(md(r'''
# R. Regenerate the four cache files (only when B2 says they are missing; Drive does not have them)
Runs the UNMODIFIED `004 run.py enc_cache` and `005 run.py feat` with their own `config.yaml`. Own results folders (`exp010_work/regen/results_004`, `results_005`) so that no 004/005 attempts counter or result file is touched. Small GPU run: measured earlier on Colab L4 177 s + 137 s (`004_depth_truncation/results/colab/timing_main.json`), plus reading the Q1 frames from Drive and copying about 9 GB to Drive (not measured).
'''))

C.append(code(r'''
# R0. APPROVAL for the regeneration (CLAUDE.md: GPU use needs a human; small, but still gated). Set by the human only.
APPROVED_REGEN_010 = False
REGEN_FORCE = False       # True: run again even if the local files exist (004 --force is NOT used; delete the local files by hand if you really want an overwrite)
print('APPROVED_REGEN_010 =', APPROVED_REGEN_010)
'''))

C.append(code(r'''
# R1. PREFLIGHT for the regeneration (no GPU, no attempt used). Stops with a list when something is missing. Contains the metadata.pt placement (like 009 A6b: the 005 `feat` stage computes the
# action mean/std from the training manifest of seed 0, which reads <episode dir>/metadata.pt; the first 009 pilot failed twice without it).
import json, os, shutil, tarfile
from pathlib import Path
problems = []
rd = Path(outputs) / want_run
for f in ('DONE', 'hydra.yaml', 'train_manifest.json', 'checkpoints/model_latest.pth'):
    if not (rd / f).exists():
        problems.append('missing ' + str(rd / f))
# (1) config values of 004 / 005 that decide what the cache files contain (expected = the settings of the original 2026-11-09 runs: 004 enc_cache log, brief of this task)
v4 = T.read_yaml_values(EXP004 + '/config.yaml', ['encoder.k_grid', 'encoder.random_init_control', 'encoder.source_train_seed', 'encoder.batch_episodes', 'paths.cache_env'])
v5 = T.read_yaml_values(EXP005 + '/config.yaml', ['feat.train_seed', 'feat.random_init_seed', 'feat.batch_episodes', 'paths.cache_env', 'paths.cache004_env', 'exp4.cache_suffix', 'base.condition'])
v2 = T.read_yaml_values(EXP002 + '/config.yaml', ['probe_impl.random_init'])
print('004:', v4); print('005:', v5); print('002:', v2)
exp = {'004 encoder.k_grid': (v4['encoder.k_grid'], [0, 2, 4, 6, 8, 10, 12]), '004 encoder.random_init_control': (v4['encoder.random_init_control'], True),
       '004 encoder.source_train_seed': (v4['encoder.source_train_seed'], 0), '004 paths.cache_env': (v4['paths.cache_env'], 'EXP004_CACHE'),
       '005 feat.train_seed': (v5['feat.train_seed'], 0), '005 feat.random_init_seed': (v5['feat.random_init_seed'], 1234), '005 paths.cache_env': (v5['paths.cache_env'], 'EXP005_CACHE'),
       '005 exp4.cache_suffix': (v5['exp4.cache_suffix'], ''), '005 base.condition': (v5['base.condition'], 'multi_60_ind_160'),
       '002 random_init.seed': (v2['probe_impl.random_init']['seed'], 0)}
for k, (got, want) in exp.items():
    if got != want:
        problems.append('config differs from the original run: %s = %r (expected %r)' % (k, got, want))
# (2) config fingerprints (information): 004 results.json carries the config sha256 of the main run
for name, exp_dir in (('004', EXP004), ('005', EXP005)):
    rj = json.load(open(exp_dir + '/results.json')) if os.path.exists(exp_dir + '/results.json') else {}
    cur = T.sha256_file(exp_dir + '/config.yaml')
    print('%s config.yaml sha256 %s | results.json config_sha256 %s | %s' % (name, cur[:16], str(rj.get('config_sha256'))[:16], 'equal' if rj.get('config_sha256') == cur else 'DIFFERENT (cause NOT verified: 005 CHANGES.md records that n_inputs was changed 2000 -> 900 on Colab, but replacing it did not reproduce the sha; feat itself has no n_inputs; the feat values that matter are checked above)'))
# (3) 005 plan.json (the committed one of the 005 main run): predictor layers must be exactly pred_out6
plan = json.load(open(EXP005 + '/results/plan.json'))
lay = [(l['name'], l['module'], l['depth']) for l in plan['layers']]
if lay != [('pred_out6', 'predictor', 6)]:
    problems.append('005 results/plan.json layers are %s, expected [pred_out6 predictor 6]' % lay)
os.makedirs(REGEN['005']['results'], exist_ok=True)
shutil.copyfile(EXP005 + '/results/plan.json', REGEN['005']['results'] + '/plan.json')      # 005 `feat` reads plan.json from its results dir
# (4) metadata.pt of the seed-0 training manifest
need_dirs = set()
if (rd / 'train_manifest.json').exists():
    for ep in json.load(open(rd / 'train_manifest.json'))['episodes']:
        need_dirs.add(ep['dir'])
tar_path = Path(WORK002) / 'cloth_sim_data.tar'
tf = None
for d in sorted(need_dirs):
    dest, split = Path(d) / 'metadata.pt', Path(d).name
    if dest.exists():
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = None
    for cand in (Path(DATA_SRC) / split / 'metadata.pt', Path(DATA_SRC) / 'cloth_sim' / split / 'metadata.pt'):
        if cand.exists():
            shutil.copyfile(cand, dest); got = 'copied from ' + str(cand); break
    if got is None and tar_path.exists():
        tf = tf or tarfile.open(tar_path, 'r:*')
        for m in tf:
            parts = m.name.split('/')
            if m.isfile() and not m.name.startswith('/') and '..' not in parts and parts[-1] == 'metadata.pt' and len(parts) >= 2 and parts[-2] == split:
                with tf.extractfile(m) as fi, open(dest, 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
                got = 'extracted %s from %s' % (m.name, tar_path); break
    print(got or ('!! NOT FOUND ' + str(dest)))
    if got is None:
        problems.append('metadata.pt not found for split %s (DATA_SRC / cloth_sim_data.tar)' % split)
if tf is not None:
    tf.close()
print('metadata.pt dirs needed:', len(need_dirs))
if problems:
    raise SystemExit('PREFLIGHT FAILED (no GPU used, no attempt used):\n  - ' + '\n  - '.join(problems))
print('regeneration preflight OK')
'''))

C.append(code(r'''
# R2. 004 enc_cache -> tokens_k12.npy / rand_k12.npy. Dry run first (prints the plan). Skipped when the two local files exist. Writes the full k_grid to the LOCAL disk (about 30 GB, 추정).
# Afterwards ONLY the two needed files are copied to Drive.
if not APPROVED_REGEN_010:
    print('사람 승인 필요(CLAUDE.md): APPROVED_REGEN_010 = False -> nothing was run.')
else:
    run_regen('004', dry=True)
    have = [os.path.exists(l) for l, _ in CACHE_FILES[:2]]
    if all(have) and not REGEN_FORCE:
        print('tokens_k12 / rand_k12 already present locally -> skipped')
    else:
        assert shutil.disk_usage('/content').free / 1e9 > 33, 'less than 33 GB free on /content (enc_cache writes about 30 GB, 추정)'
        if os.path.exists(LOC_ENC + '/DONE') and not all(have):
            raise SystemExit('enc DONE exists but a k12 file is missing: delete %s by hand and rerun (004 --force is not used here)' % LOC_ENC)
        rec = run_regen('004')
        REGEN_RAN.add('004')
        print('>>> 004 enc_cache %.0f s (original run: 177 s), sampled peak GPU %s MiB, log %s' % (rec['seconds'], rec['vram_peak_sampled_mib'], rec['log']))
    for loc, drv in CACHE_FILES[:2]:
        print(os.path.basename(loc), '->', copy_atomic(loc, drv))
'''))

C.append(code(r'''
# R3. 005 feat -> feat_pred_out6.npy / feat_pred_out6_rand.npy (training seed 0, random_init_seed 1234 from 005 config.yaml; layers from the committed plan.json). Skipped when the two local files exist.
if not APPROVED_REGEN_010:
    print('사람 승인 필요(CLAUDE.md): APPROVED_REGEN_010 = False -> nothing was run.')
else:
    run_regen('005', dry=True)
    have = [os.path.exists(l) for l, _ in CACHE_FILES[2:]]
    if all(have) and not REGEN_FORCE:
        print('feat_pred_out6 files already present locally -> skipped')
    else:
        rec = run_regen('005')
        REGEN_RAN.add('005')
        print('>>> 005 feat %.0f s (original run: 137 s), sampled peak GPU %s MiB, log %s' % (rec['seconds'], rec['vram_peak_sampled_mib'], rec['log']))
    for loc, drv in CACHE_FILES[2:]:
        print(os.path.basename(loc), '->', copy_atomic(loc, drv))
'''))

C.append(code(r'''
# R4. Verification that the regenerated files come from the same settings as the 004/005 main runs, and cache_record.json (sizes + sha256 on Drive for later sessions).
# (a) the "enc_cache plan" / "feat plan" log lines and the consistency checks of the new logs are compared with the committed logs of the original runs (2026-11-09, Colab);
# (b) shapes/dtype of the four files; (c) sizes. Bitwise equality with the original files cannot be tested (they are gone); GPU nondeterminism may change the last digits (not tested).
import re, glob, time, hashlib
import numpy as np
def plan_fields(text):
    out = {}
    m = re.search(r'enc_cache plan: k_grid=(\[.*?\]) frames=(\[.*?\]) episodes=(\d+) random_init_control=(\w+)', text)
    if m: out['enc_cache plan'] = m.groups()
    m = re.search(r'feat plan: predictor layers (\[.*?\]) .*? encoder layers (\[.*?\])', text)
    if m: out['feat plan'] = m.groups()
    out['consistency'] = re.findall(r'consistency check (.*?): max\|diff\|/std = ([0-9.]+)', text)
    out['progress_end'] = re.findall(r'(?:enc_cache|feat extract_layers) (\d+)/(\d+)', text)[-1:]
    return out
def compare_logs(ref_text, new_text):
    a, b = plan_fields(ref_text), plan_fields(new_text)
    rows = []
    for k in ('enc_cache plan', 'feat plan', 'progress_end'):
        if k in a or k in b:
            rows.append((k, a.get(k), b.get(k), 'EQUAL' if a.get(k) == b.get(k) else 'DIFFERENT'))
    ca, cb = dict(a['consistency']), dict(b['consistency'])
    for name in sorted(set(ca) | set(cb)):
        rows.append(('consistency ' + name, ca.get(name), cb.get(name), 'info (max|diff|/std; compare magnitudes)'))
    return rows
REF = {'004': glob.glob(EXP004 + '/logs/colab_20261109/enc_cache_20261109-092707.log'), '005': glob.glob(EXP005 + '/logs/colab_20261109/feat_20261109-112051.log')}
new_logs = {w: sorted([f for f in glob.glob(WORK010 + '/logs_regen/%s_*.log' % ('enc_cache' if w == '004' else 'feat')) if '_dry_' not in os.path.basename(f)], key=os.path.getmtime) for w in ('004', '005')}
all_equal = True
for w in ('004', '005'):
    print('---', w, 'reference log:', REF[w][:1], '| newest regeneration log:', new_logs[w][-1:])
    if not REF[w] or not new_logs[w]:
        print('  (no reference or no new log: comparison skipped; the files may have been restored from Drive)')
        continue
    for k, ra, rb, st in compare_logs(open(REF[w][0], errors='replace').read(), open(new_logs[w][-1], errors='replace').read()):
        print('  %-34s ref=%s new=%s  %s' % (k, ra, rb, st))
        all_equal &= (st != 'DIFFERENT')
print('plan lines equal:', all_equal)
files = {}
shapes = {}
for loc, drv in CACHE_FILES:
    if not os.path.exists(loc):
        print('missing', loc); continue
    a = np.load(loc, mmap_mode='r')
    shapes[os.path.basename(loc)] = (list(a.shape), str(a.dtype))
    files[os.path.basename(loc)] = {'bytes': os.path.getsize(loc), 'sha256': T.sha256_file(loc), 'shape': list(a.shape), 'dtype': str(a.dtype)}
    print('%-26s shape=%s dtype=%s %.2f GB' % (os.path.basename(loc), a.shape, a.dtype, os.path.getsize(loc) / 1e9))
if len(files) == 4 and (REGEN_RAN or not os.path.exists(rec_path)):
    n_ep = {f['shape'][0] for f in files.values()}
    n_fr = {f['shape'][1] for f in files.values()}
    print('episodes per file', n_ep, '| frames per episode', n_fr, '(original: 1800 episodes, 8 frames [0,1,2,3,8,9,10,11] per the logs)')
    assert n_ep == {1800} and n_fr == {8}, 'unexpected episode/frame count'
    json.dump({'created_utc': time.strftime('%F %T', time.gmtime()), 'code_version': CODE_VERSION, 'files': files, 'plan_lines_equal_to_original_logs': all_equal,
               'config_values': {'004': v4, '005': v5}}, open(rec_path, 'w'), indent=1)
    print('wrote', rec_path)
'''))

C.append(md(r'''
# C. OPTIONAL pilot (needs the four files; results in `exp010_work/results_pilot`)
Runs `heads` for the FIRST configured task seed, head seed 0 and 3 tasks (`--task-seeds <first> --seeds 0 --max-tasks 3`; existing CLI flags only). It measures time and finds errors; the cost projection printed afterwards is a rough estimate (추정). Pilot predictions must never be mixed into the main results (separate folder).
'''))

C.append(code(r'''
# C0. APPROVAL for the pilot (small GPU run). Set by the human only.
APPROVED_PILOT_010 = False
PILOT_MAX_TASKS = '3'
print('APPROVED_PILOT_010 =', APPROVED_PILOT_010)
'''))

C.append(code(r'''
# C1. Pilot: one task seed (the first of config seeds.task_a), head seed 0, 3 tasks.
import time
if not APPROVED_PILOT_010:
    print('사람 승인 필요(CLAUDE.md): APPROVED_PILOT_010 = False -> nothing was run.')
else:
    assert all(x[2] for x in cache_status()), 'the four feature files are not all present locally (B2 / section R)'
    ts0 = T.read_yaml_values(EXP010 + '/config.yaml', ['seeds.task_a'])['seeds.task_a'][0]
    t0 = time.time()
    rec = run010('heads', ['--task-seeds', str(ts0), '--seeds', '0', '--max-tasks', PILOT_MAX_TASKS], mode='pilot')
    print('>>> pilot heads %.0f s, sampled peak GPU %s MiB, log %s' % (rec['seconds'], rec['vram_peak_sampled_mib'], rec['log']))
'''))

C.append(code(r'''
# C2. Rough projection of the main heads time from the pilot's cost.json (추정): pilot entries are for max_tasks tasks, one task seed and one head seed; the main run has n_tasks (100) tasks, all task seeds and head seeds.
cj = WORK010 + '/results_pilot/cost.json'
if os.path.exists(cj):
    cost = json.load(open(cj))
    cfgv = T.read_yaml_values(EXP010 + '/config.yaml', ['seeds.task_a', 'seeds.head'])
    nts, nhs = len(cfgv['seeds.task_a']), len(cfgv['seeds.head'])
    tot = sum(e['seconds'] for e in cost['entries'])
    for e in cost['entries']:
        print('  %-48s %6.0f s' % (e['what'], e['seconds']))
    print('pilot total %.0f s for %s tasks, 1 task seed, 1 head seed; gpu %s' % (tot, PILOT_MAX_TASKS, cost.get('gpu')))
    proj = tot / int(PILOT_MAX_TASKS) * 100 * nts * nhs
    print('naive projection to %d tasks x %d task seeds x %d head seeds: %s (추정; first entries include CUDA warm-up -> leans high). README estimate before any measurement: 3-3.5 h (추정).' % (100, nts, nhs, T.hms(proj)))
else:
    print('no pilot cost.json yet')
'''))

C.append(md(r'''
# D. Main heads (needs `APPROVED_MAIN_010 = True` in cell D0)
'''))

C.append(code(r'''
# D0. APPROVAL (CLAUDE.md: GPU/cost-heavy runs need human approval). Set by the human only.
# BEFORE you set it: the 010 hypothesis is a draft. Delta (0.05), the task seeds ([2000, 2001, 2002]) and this approval are YOUR decisions (notes/hypotheses.md "010").
APPROVED_MAIN_010 = False
ALLOW_MAIN_WITH_PILOT_TRUE = False      # True = run although config.yaml has pilot: true (results.json would then say pilot: true). Normally the human sets pilot: false in config.yaml and pushes first.
print('APPROVED_MAIN_010 =', APPROVED_MAIN_010)
'''))

C.append(code(r'''
# D1. heads: all configured task seeds x head seeds x (4 representations + area / oracle base models). No CLI overrides: seeds and counts come from config.yaml. Finished prediction files are skipped
# (rerun this cell after a disconnect: run A and B first). Expected GPU time: README estimate about 3-3.5 h on L4 (추정, no measurement; the pilot projection in C2 replaces it if you ran C).
if not APPROVED_MAIN_010:
    print('사람 승인 필요(CLAUDE.md): APPROVED_MAIN_010 = False -> nothing was run.')
else:
    v = T.read_yaml_values(EXP010 + '/config.yaml', ['pilot', 'seed', 'seeds.head', 'seeds.task_a', 'seeds.eval', 'eval.n_inputs', 'tost.delta', 'tost.alpha'])
    print('010 config:', v)
    if v['pilot'] is not False and not ALLOW_MAIN_WITH_PILOT_TRUE:
        raise SystemExit('010 config.yaml has pilot: %s. The human sets pilot: false (and pushes) or ALLOW_MAIN_WITH_PILOT_TRUE=True. Nothing was run.' % v['pilot'])
    assert all(x[2] for x in cache_status()), 'the four feature files are not all present locally (B2 / section R)'
    rec = run010('heads', mode='main')
    print('>>> heads done in %.0f s (sampled peak GPU %s MiB), log %s' % (rec['seconds'], rec['vram_peak_sampled_mib'], rec['log']))
'''))

C.append(md(r'''
# E. Report (CPU: 005 metrics per task seed, TOST with Delta from config.yaml, results.json, ib_scores.csv)
'''))

C.append(code(r'''
# E1. report -> <clone>/experiments/010.../results.json (copied to Drive immediately, with code_version) and exp010_work/results/ib_scores.csv.
if not APPROVED_MAIN_010:
    print('APPROVED_MAIN_010 = False -> nothing was run (report follows the main heads).')
else:
    run010('report', mode='main')
    save_results_json(WORK010 + '/results')
    r = json.load(open(EXP010 + '/results.json'))
    print({k: r[k] for k in ('pilot', 'task_seeds_used', 'task_seeds_missing_files', 'head_seeds', 'eval_n_inputs', 'tost_delta', 'code_version')})
    print({k: v['status'] for k, v in (r['analysis']['H_f_tost'].items() if isinstance(r['analysis']['H_f_tost'], dict) else enumerate(r['analysis']['H_f_tost']))})
'''))

C.append(md(r'''
# F. Copy results out
'''))

C.append(code(r'''
# F1. What is on Drive now, sha256 of the result files, attempts. Optional zip download (DOWNLOAD_BUNDLE). To bring back into the repo by hand: exp010_work/results/results.json -> experiments/010_source_separation_ib/results.json.
DOWNLOAD_BUNDLE = False
import glob
for f in sorted(glob.glob(WORK010 + '/*') + glob.glob(WORK010 + '/results/*') + glob.glob(WORK010 + '/results_pilot/*')):
    print(f, os.path.getsize(f) if os.path.isfile(f) else '(dir)')
T.print_sha_table([WORK010 + '/results/results.json', WORK010 + '/results/ib_scores.csv', WORK010 + '/results_pilot/cost.json', WORK010 + '/cache_record.json', WORK010 + '/code_version.json'])
for f in ('results/attempts.json', 'results_pilot/attempts.json', 'regen/results_004/attempts.json', 'regen/results_005/attempts.json'):
    if os.path.exists(WORK010 + '/' + f):
        print(WORK010 + '/' + f, open(WORK010 + '/' + f).read().replace('\n', ' '))
if DOWNLOAD_BUNDLE:
    import zipfile
    zp = '/content/exp010_results_bundle.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for ext in ('json', 'csv', 'log'):
            for f in glob.glob(WORK010 + '/**/*.' + ext, recursive=True):
                z.write(f, os.path.relpath(f, os.path.dirname(WORK010)))
    from google.colab import files
    files.download(zp)
'''))

C.append(code(r'''
# F2. Flush Drive writes before leaving.
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
    out = HERE / "exp010_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", out, "cells", len(C))


if __name__ == "__main__":
    main()
