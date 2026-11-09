#!/usr/bin/env python3
"""Regenerate colab/exp004_005_colab.ipynb. The notebook embeds a SNAPSHOT (tar.gz + base64) of the code it needs, because experiments 004/005 are not on the
GitHub branch that the 002 notebook clones. Run again after ANY code change:   python3 colab/build_notebook.py
Embedded: 002 {src, run.py, config.yaml, colab/colab_tools.py, colab/requirements_colab.txt} (copies, unmodified), 004 {src, run.py, config.yaml, README.md,
CHANGES.md, colab/exp045_tools.py, colab/requirements_colab_extra.txt}, 005 {src, run.py, config.yaml, README.md, CHANGES.md}. Not: __pycache__, logs, results.
Also reads 002/results.json config_sha256 (expected fingerprint of the 002 config) at build time.
"""
import base64, io, json, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPS = HERE.parents[1]
FILES = {
    "002_single_size_sweep": ["src", "run.py", "config.yaml", "colab/colab_tools.py", "colab/requirements_colab.txt"],
    "004_depth_truncation": ["src", "run.py", "config.yaml", "README.md", "CHANGES.md", "colab/exp045_tools.py", "colab/requirements_colab_extra.txt"],
    "005_inductive_bias_probe": ["src", "run.py", "config.yaml", "README.md", "CHANGES.md"],
}


def _filter(ti):
    if "__pycache__" in ti.name.split("/") or ti.name.endswith(".pyc"):
        return None
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mtime = 0
    return ti


def make_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
        for top, fs in FILES.items():
            for f in fs:
                tf.add(str(EXPS / top / f), arcname="%s/%s" % (top, f), filter=_filter)
    raw = buf.getvalue()
    b = base64.b64encode(raw).decode()
    return "\n".join(b[i:i + 100] for i in range(0, len(b), 100)), len(raw)


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


C = []
C.append(md(r'''
# Experiments 004 (depth truncation) and 005 (inductive-bias probe) on Google Colab
Run the cells top to bottom. Sections: **A** setup + input check, **B** 004 pilot, **C** 005 pilot, **D** 004 main run, **E** 005 main run, **F** copy results out.
Every cell is idempotent; after a disconnect choose a GPU runtime and run A again, then continue where you stopped (stages resume from their DONE markers / part files).

What is reused: the three 002 condition-(b) checkpoints (`multi_60_ind_160`, seeds 0,1,2) and the 002 Q1 cache. **No world model is trained here.** Nothing in `MyDrive/exp002_work` is written (inputs are read-only; the check cell only reads).
Where things live
- Drive (persists): `MyDrive/exp004_work/{results,logs,results_pilot}` and `MyDrive/exp005_work/{results,logs,results_pilot}` (the repo folders `results/` and `logs/` become symlinks to them; `results_pilot` is passed with `--results-dir`). `attempts.json` (3-failure guard) lives in each results folder, so it persists too.
- Local `/content` (lost with the session): `/content/main/{exp004_cache,exp005_cache}` and `/content/pilot/{exp002_cache,exp004_cache,exp005_cache}`, the venv, the code.
- **005 needs the 004 encoder-token cache (`EXP004_CACHE/enc`) from the main run.** If the session ends after section D, that cache is gone: re-run `enc_cache` (minutes of GPU, about 30 GB disk) or use the optional tar save/restore cells (default OFF) to keep it on Drive.
- Pilot and main never share a folder: pilot uses suffix `_smoke`, results folder `results_pilot`, caches under `/content/pilot`.

Human approval (CLAUDE.md): the pilot is a small GPU run; the main runs need `APPROVED_MAIN_004` / `APPROVED_MAIN_005` set to True **by you** at the top of section D. Seeds and pre-registered thresholds come from the config.yaml files and are not changed here. All times and sizes printed from the docs are estimates (추정) until the pilot measures them.
Everything in this notebook is UNVERIFIED on Colab as a whole (written and checked on a CPU-only machine).
'''))

C.append(code(r'''
# A1. GPU / Python / disk
import sys, subprocess, os
def sh(cmd, cwd=None, env=None):
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
# A2. Mount Drive, choose work roots and options
import os
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive'
WORK002 = os.environ.get('EXP002_WORK_ROOT', DRIVE + '/exp002_work')          # READ-ONLY inputs: train_runs, exp002_cache, manifests
WORK004 = os.environ.get('EXP004_WORK_ROOT', DRIVE + '/exp004_work')          # results, logs, results_pilot, cache_tar (persistent)
WORK005 = os.environ.get('EXP005_WORK_ROOT', DRIVE + '/exp005_work')
for d in (WORK004, WORK005):
    os.makedirs(d, exist_ok=True)
# Data folder, only needed to build the 002 `_smoke` pilot cache (cell C0). Same default as the 002 notebook (UNVERIFIED path).
DATA_SRC = os.environ.get('EXP002_DRIVE_MOUNT_SRC', DRIVE + '/dino-wm-videos/[data]/[cloth-sim] cloth_move')
# Big caches on the Colab LOCAL disk (default). Changing these to Drive paths works but is slow.
LOCAL_MAIN, LOCAL_PILOT = '/content/main', '/content/pilot'
SAVE_SMOKE_CACHE_TO_DRIVE = False     # True: the 002 `_smoke` pilot cache is built in WORK002/exp002_cache/q1_tokens_smoke (a NEW sibling folder, existing files untouched) and reused next session
print('WORK002 =', WORK002, 'exists', os.path.isdir(WORK002))
print('WORK004 =', WORK004, '| WORK005 =', WORK005)
print('DATA_SRC =', DATA_SRC, 'exists', os.path.isdir(DATA_SRC), '(only needed for the pilot cache)')
'''))

C.append(code(r'''
# A3. Code. CODE_SOURCE='embedded' (default): the snapshot below (002 + 004 + 005 code). 'git': clone REPO_URL@REPO_BRANCH (that branch must already contain 004/005 and the colab/ files: push first).
CODE_SOURCE = 'embedded'
REPO_URL = 'https://github.com/jongmin-s-multicore/probing-experiment.git'     # == git remote get-url origin; may be private -> GITHUB_TOKEN in Colab Secrets
REPO_BRANCH = 'master'
EXPECTED_CFG002_SHA = '@@CFG002SHA@@'    # 002/results.json config_sha256 read at notebook build time (@@STAMP@@)
import base64, io, os, subprocess, tarfile
REPO = '/content/mylab'
EXPS = REPO + '/experiments'
EXP002, EXP004, EXP005 = EXPS + '/002_single_size_sweep', EXPS + '/004_depth_truncation', EXPS + '/005_inductive_bias_probe'
EMBEDDED = """
@@B64@@
"""
if CODE_SOURCE == 'git':
    url = REPO_URL
    try:
        from google.colab import userdata
        tok = userdata.get('GITHUB_TOKEN')
        if tok:
            url = REPO_URL.replace('https://', 'https://%s@' % tok)
    except Exception:
        pass
    if not os.path.isdir(REPO + '/.git'):
        subprocess.run(['git', 'clone', '--branch', REPO_BRANCH, url, REPO], check=True)
    else:
        subprocess.run(['git', '-C', REPO, 'fetch', 'origin', REPO_BRANCH], check=True)
        subprocess.run(['git', '-C', REPO, 'checkout', REPO_BRANCH], check=True)
        subprocess.run(['git', '-C', REPO, 'pull', '--ff-only', 'origin', REPO_BRANCH], check=True)
    print(subprocess.run(['git', '-C', REPO, 'log', '--oneline', '-1'], capture_output=True, text=True).stdout)
else:
    raw = base64.b64decode(''.join(EMBEDDED.split()))
    os.makedirs(EXPS, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:*') as tf:
        for m in tf.getmembers():
            parts = m.name.split('/')
            assert not m.name.startswith('/') and '..' not in parts and (m.isfile() or m.isdir()), 'refused tar member: ' + m.name
        tf.extractall(EXPS)
    print('unpacked the embedded snapshot (%d bytes) into %s' % (len(raw), EXPS))
for p in (EXP002 + '/run.py', EXP004 + '/run.py', EXP005 + '/run.py', EXP004 + '/colab/exp045_tools.py', EXP002 + '/colab/requirements_colab.txt'):
    assert os.path.isfile(p), 'missing ' + p
import sys
sys.path.insert(0, EXP004 + '/colab')
import importlib, exp045_tools as T
importlib.reload(T)
print('exp045_tools loaded from', T.__file__)
'''))

C.append(code(r'''
# A4. Python environment: uv venv (3.9) with the 002 pins (+ matplotlib). Colab has no conda: documented exception to CLAUDE.md (same as 002/003).
# USE_SYSTEM_PYTHON=True uses Colab's own Python/torch; results are then NOT comparable with the 002 environment (record it).
USE_SYSTEM_PYTHON = False
VENV = '/content/venv'
import hashlib, os, sys
REQ = [EXP002 + '/colab/requirements_colab.txt', EXP004 + '/colab/requirements_colab_extra.txt']
if USE_SYSTEM_PYTHON:
    PY = sys.executable
else:
    PY = VENV + '/bin/python'
    MARK = VENV + '/.exp045_ok_' + hashlib.sha256(b''.join(open(r, 'rb').read() for r in REQ)).hexdigest()[:12]
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
# A5. ACWM-Phys checkout at the pinned commit (== 002 config.yaml code.commit). Needed to unpickle the 002 checkpoints and to build the 002 pilot cache.
# NO resume patch here (nothing is trained). The URL equals 002 config code.repo_url (public at the 2026-11-04 ls-remote test, per the 002 notebook).
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
import os, json
for d in (LOCAL_MAIN + '/exp004_cache', LOCAL_MAIN + '/exp005_cache', LOCAL_PILOT + '/exp004_cache', LOCAL_PILOT + '/exp005_cache'):
    os.makedirs(d, exist_ok=True)
# the pilot 002 cache: reuse an existing Drive q1_tokens_smoke (read-only), else build it locally (or on Drive when SAVE_SMOKE_CACHE_TO_DRIVE)
if os.path.exists(WORK002 + '/exp002_cache/q1_tokens_smoke/DONE') or SAVE_SMOKE_CACHE_TO_DRIVE:
    PILOT_CACHE002 = WORK002 + '/exp002_cache'
else:
    PILOT_CACHE002 = LOCAL_PILOT + '/exp002_cache'
    os.makedirs(PILOT_CACHE002, exist_ok=True)
base_env = dict(os.environ)
base_env.update({
    'DINO_WORK': '/content/dinowm', 'DATASET_DIR': '/content/dinowm/data',          # same as the 002 session (cache_episodes.json rows refer to this data location)
    'DINO_TRAIN': WORK002 + '/train_runs',                                          # READ-ONLY use: checkpoints
    'ACWM_PHYS_SRC': SRC, 'WANDB_MODE': 'disabled', 'HYDRA_FULL_ERROR': '1', 'PYTHONUNBUFFERED': '1',
    'EXP002_WORK_ROOT': WORK002, 'EXP002_DATA_MODE': 'mount', 'EXP002_DRIVE_MOUNT_SRC': DATA_SRC,
    'EXP002_MAKE_ARCHIVE': '0',                                                     # do not write a data archive into exp002_work
    'PATH': os.path.dirname(PY) + os.pathsep + os.environ['PATH'],
})
# NOTE: PYTORCH_CUDA_ALLOC_CONF is deliberately not set (expandable_segments is unknown to torch 2.0.x; 003 commit 16edd98). 004/005 do not set it either.
if not USE_SYSTEM_PYTHON:
    base_env['VIRTUAL_ENV'] = VENV
ENV_MAIN = dict(base_env, EXP002_CACHE=WORK002 + '/exp002_cache', EXP004_CACHE=LOCAL_MAIN + '/exp004_cache', EXP005_CACHE=LOCAL_MAIN + '/exp005_cache')
ENV_PILOT = dict(base_env, EXP002_CACHE=PILOT_CACHE002, EXP004_CACHE=LOCAL_PILOT + '/exp004_cache', EXP005_CACHE=LOCAL_PILOT + '/exp005_cache')
print(json.dumps({'ENV_MAIN': {k: ENV_MAIN[k] for k in ('EXP002_CACHE', 'EXP004_CACHE', 'EXP005_CACHE')}, 'ENV_PILOT': {k: ENV_PILOT[k] for k in ('EXP002_CACHE', 'EXP004_CACHE', 'EXP005_CACHE')}}, indent=1))
print('004:', T.link_dirs(WORK004, EXP004, ('results', 'logs')))
print('005:', T.link_dirs(WORK005, EXP005, ('results', 'logs')))
for d in ('results_pilot',):
    os.makedirs(WORK004 + '/' + d, exist_ok=True)
    os.makedirs(WORK005 + '/' + d, exist_ok=True)
# the 002 folders in this clone stay LOCAL (no symlink to Drive): a manifest copy is made only when the pilot cache is built (cell C0).

MODES = {
    'pilot': dict(env=ENV_PILOT, suffix=['--suffix', '_smoke'], r4=WORK004 + '/results_pilot', r5=WORK005 + '/results_pilot', timing=WORK004 + '/timing_pilot.json', results_json_name='results_pilot'),
    'main': dict(env=ENV_MAIN, suffix=[], r4=WORK004 + '/results', r5=WORK005 + '/results', timing=WORK004 + '/timing_main.json', results_json_name='results'),
}

def run004(stage, extra=(), mode='main', dry=False):
    m = MODES[mode]
    args = (m['suffix'] if stage in ('enc_cache', 'enc_probe', 'pred') else []) + ['--results-dir', m['r4']] + list(extra) + (['--dry-run'] if dry else [])
    rec = T.run_stage(PY, EXP004, stage, args, m['env'], timing_file=None if dry else m['timing'], label='004:%s%s' % (stage, '(dry)' if dry else ''))
    T.require_ok(rec, m['r4'])
    return rec

def run005(stage, extra=(), mode='main', dry=False):
    m = MODES[mode]
    args = (m['suffix'] if stage in ('tasks', 'feat', 'heads') else []) + ['--results-dir', m['r5']] + list(extra) + (['--dry-run'] if dry else [])
    rec = T.run_stage(PY, EXP005, stage, args, m['env'], timing_file=None if dry else m['timing'], label='005:%s%s' % (stage, '(dry)' if dry else ''))
    T.require_ok(rec, m['r5'])
    return rec

def save_results_json(exp_dir, dest_dir):
    # aggregate/report write <repo exp folder>/results.json (local clone, lost with the session): copy to Drive right away
    import shutil
    src = exp_dir + '/results.json'
    if os.path.exists(src):
        shutil.copyfile(src, dest_dir + '/results.json')
        print('copied', src, '->', dest_dir + '/results.json')
    else:
        print('no', src)
'''))

C.append(code(r'''
# A7. Disk and memory. Estimates (추정, 004 CHANGES.md 6절): enc_cache ~30 GB (pilot ~5 GB), pred temporary ~26 GB per training seed (deleted after each seed), 005 feat ~4.4 GB per layer.
# A GPU runtime's local disk size depends on the machine type; the main run needs roughly 30 + 26 + venv/data headroom.
T.disk_report(['/content', WORK004, WORK005], need_gb={'004 enc_cache (main)': 30, '004 pred tmp (1 seed at a time)': 26, '005 feat (per layer, real+random)': 4.4, 'pilot caches': 5 + 5})
import shutil
free_local = shutil.disk_usage('/content').free / 1e9
print('free local disk: %.0f GB' % free_local, '-> main 004 needs ~56 GB incl. pred tmp (추정)' + ('  !! LOW' if free_local < 60 else ''))
'''))

C.append(code(r'''
# A8. CHECK (read-only). Stops with a clear message if an input is missing. Writes nothing to exp002_work.
# 1) 002 checkpoints (DONE + final checkpoint) for seeds 0,1,2, q1_tokens/DONE, 002 config fingerprint
need_manifest = not os.path.exists(WORK002 + '/exp002_cache/q1_tokens_smoke/DONE')
ok, bad, lines = T.check_inputs(WORK002, [0, 1, 2], suffixes=('', '_smoke'), expected_epoch=15, expected_cfg_sha=EXPECTED_CFG002_SHA,
                                cfg002_path=EXP002 + '/config.yaml', need_manifest=need_manifest)
print('\n'.join(lines))
if not ok:
    raise SystemExit('INPUTS MISSING - stopping (nothing was run):\n  - ' + '\n  - '.join(bad) +
                     '\n  The 002 outputs must be on Drive at %s (train_runs/, exp002_cache/q1_tokens/). If the cache is missing, build it with 002 `run.py cache` (GPU; not done by this notebook).' % WORK002)
print('inputs OK')
'''))

C.append(code(r'''
# A9. 004 / 005 check --dry-run and check (these runs only print; no attempts are recorded). Main environment, so 'q1_tokens' (no suffix) is what is checked.
for mode in ('main',):
    run004('check', mode=mode, dry=True)
    run004('check', mode=mode)
    run004('enc_cache', mode=mode, dry=True)       # prints the exact disk estimate for the main cache
    run005('check', mode=mode, dry=True)
    run005('check', mode=mode)
# pass the next lines to a human: 'DONE=False' for the 002 runs means the inputs are not usable.
'''))

C.append(md(r'''
# B. 004 pilot (suffix `_smoke`, train seed 0, results in `exp004_work/results_pilot`, caches under /content/pilot)
Small GPU run; its purpose is to measure time and VRAM and to catch errors before the main run. Pilot numbers are never mixed with the main results (own folders, `pilot: true` is forced by the partial cache).
'''))

C.append(code(r'''
# C0. The pilot needs the 002 Q1 cache with suffix `_smoke` (25 episodes per level). Skipped when it exists (PILOT_CACHE002/q1_tokens_smoke/DONE).
# Building it = 002 `run.py download` (copies the data to /content: slow, time not measured) + `run.py cache --max-eps 25 --suffix _smoke` (GPU).
# Effects on 002: reads checkpoints (seed 0) and manifests/cache_episodes.json (copied into this CLONE); writes q1_tokens_smoke under PILOT_CACHE002 (local unless
# SAVE_SMOKE_CACHE_TO_DRIVE); 002's attempts.json is NOT touched (--results-dir points to exp004_work/pilot_002_results); 002 logs stay in the local clone and are copied to exp004_work/logs_002_pilot.
BUILD_SMOKE_CACHE = True
import shutil, glob
smoke_done = os.path.exists(PILOT_CACHE002 + '/q1_tokens_smoke/DONE')
print('002 _smoke cache present:', smoke_done, 'at', PILOT_CACHE002)
if not smoke_done:
    assert BUILD_SMOKE_CACHE, 'no _smoke cache and BUILD_SMOKE_CACHE=False: the pilot cannot run'
    os.makedirs(EXP002 + '/manifests', exist_ok=True)
    src_m = WORK002 + '/manifests/cache_episodes.json'
    assert os.path.exists(src_m), src_m + ' missing'
    shutil.copyfile(src_m, EXP002 + '/manifests/cache_episodes.json')
    r2 = WORK004 + '/pilot_002_results'
    os.makedirs(r2, exist_ok=True)
    t0 = __import__('time').time()
    for st, a in (('download', []), ('cache', ['--max-eps', '25', '--suffix', '_smoke'])):
        rec = T.run_stage(PY, EXP002, st, ['--results-dir', r2] + a, ENV_PILOT, timing_file=WORK004 + '/timing_pilot.json', label='002:%s(pilot cache)' % st)
        T.require_ok(rec, r2)
    os.makedirs(WORK004 + '/logs_002_pilot', exist_ok=True)
    for f in glob.glob(EXP002 + '/logs/*.log'):
        shutil.copy(f, WORK004 + '/logs_002_pilot/')
    print('pilot cache built in %.0f s' % (__import__('time').time() - t0))
print(open(PILOT_CACHE002 + '/q1_tokens_smoke/meta.json').read()[:400])
'''))

C.append(code(r'''
# B1. 004 pilot: all stages, training seed 0 (enc_cache -> enc_probe -> pred -> aggregate). Time, sampled peak VRAM and log path are stored in exp004_work/timing_pilot.json.
import time
t0 = time.time()
for st, ex in (('enc_cache', []), ('enc_probe', []), ('pred', ['--train-seeds', '0']), ('aggregate', [])):
    rec = run004(st, ex, mode='pilot')
    print('>>> %s: %.0f s, sampled peak GPU memory %s MiB (base %s), log %s' % (st, rec['seconds'], rec['vram_peak_sampled_mib'], rec['vram_base_mib'], rec['log']))
save_results_json(EXP004, WORK004 + '/results_pilot')
print('004 pilot total %.0f s' % (time.time() - t0))
'''))

C.append(code(r'''
# B2. 004 pilot measurements and re-estimate of the main run (estimates: linear in the number of episodes; fixed overheads make this an overestimate).
import json, re
tm = [r for r in json.load(open(WORK004 + '/timing_pilot.json')) if r['label'].startswith('004:') and r['rc'] == 0 and '(dry)' not in r['label']]
last = {}
for r in tm:
    last[r['stage']] = r            # latest successful run of each stage
for st, r in last.items():
    print('%-10s %7.0f s  peak VRAM (sampled whole GPU) %s MiB  log %s' % (st, r['seconds'], r['vram_peak_sampled_mib'], r['log']))
if 'enc_probe' in last and last['enc_probe']['log']:
    units = T.parse_enc_probe_units(open(last['enc_probe']['log'], errors='replace').read())
    if units:
        k, ctrl, sec = units[0]
        print('first enc_probe unit (k=%d control=%s): %.0f s for 5 attentive + 5 linear fits -> <= %.0f s per attentive fit (upper bound; includes the linear fits and data loading)' % (k, ctrl, sec, sec / 5))
n_pilot = json.load(open(PILOT_CACHE002 + '/q1_tokens_smoke/meta.json'))['n_episodes']
n_main = json.load(open(WORK002 + '/exp002_cache/q1_tokens/meta.json'))['n_episodes']
proj = T.project_004({s: last[s]['seconds'] for s in last}, n_pilot, n_main, n_train_seeds_main=3)
print('episodes pilot %d, main %d' % (n_pilot, n_main))
print('projected main run (추정):', {k: (T.hms(v) if k.endswith('_s') or k in ('enc_cache', 'enc_probe', 'pred', 'aggregate') else v) for k, v in proj.items()})
print('doc estimate (004 CHANGES.md 6절, 추정, before measurement): main 5-6 h, pilot ~1 h')
vr = [r['vram_peak_sampled_mib'] for r in last.values() if r['vram_peak_sampled_mib']]
print('max sampled GPU memory over stages: %s MiB' % (max(vr) if vr else 'n/a'))
'''))

C.append(md(r'''
# C. 005 pilot (needs the 004 pilot summary and the 004 pilot encoder cache)
'''))

C.append(code(r'''
# C1. 005 HUMAN DECISIONS: shows the current config values. This cell changes nothing.
# 확정 기록 12 (notes/hypotheses.md, 2026-11-09, user decision before any 004/005 result): branch.exp1_negative = false; normalization.noninf = {R: iid, D: const}.
def show_005_decisions():
    v = T.read_yaml_values(EXP005 + '/config.yaml', ['branch.exp1_negative', 'normalization.noninf', 'normalization.min_gap', 'verdict.h_e', 'verdict.h_g', 'pilot'])
    for k, x in v.items():
        print('  %-26s %s' % (k, x))
    e1, nz = v['branch.exp1_negative'], v['normalization.noninf']
    if e1 is None:
        print('!!! branch.exp1_negative is null (treated as false and recorded in plan.json): a HUMAN MUST DECIDE (true -> experiment 5 is on hold). The notebook does not change it.')
    elif e1 is False:
        print('decided (확정 기록 12, 2026-11-09): branch.exp1_negative = false (experiment 1 is not negative)')
    else:
        print('!!! branch.exp1_negative = %s differs from 확정 기록 12 (false). With true the 4.6 branch 5 puts experiment 5 on HOLD. Check that this is intended.' % e1)
    if nz == {'R': 'iid', 'D': 'const'}:
        print('decided (확정 기록 12, 2026-11-09): normalization.noninf = {R: iid, D: const} (reference where the denominator is defined for each metric; other-side score stays in score_alt)')
    elif nz == {'R': 'iid', 'D': 'iid'}:
        print('!!! normalization.noninf = {R: iid, D: iid}: the old, UNRESOLVED setting. With binary random labels the D-IB denominator is about 0 -> score NaN -> H-e not_evaluable. A HUMAN MUST DECIDE (README line ~86, CHANGES.md 8).')
    else:
        print('!!! normalization.noninf = %s differs from 확정 기록 12 ({R: iid, D: const}). Check that this is intended.' % nz)
show_005_decisions()
'''))

C.append(code(r'''
# C2. 005 pilot: plan (from the 004 PILOT summary) -> tasks -> feat -> heads (3 tasks, seed 0, first planned layer) with timing.
import json, time
sm_pilot = WORK004 + '/results_pilot/summary.json'
assert os.path.exists(sm_pilot), '004 pilot summary.json missing (run B1 first): ' + sm_pilot
t0 = time.time()
run005('plan', ['--summary', sm_pilot], mode='pilot')
plan = json.load(open(WORK005 + '/results_pilot/plan.json'))
layers = [l['name'] for l in plan.get('layers', [])]
print('pilot plan layers:', layers, '| hold:', plan.get('hold'))
assert layers and not plan.get('hold'), 'plan has no layers or is on hold: nothing to run (report this)'
PILOT_LAYER = layers[0]           # pilot measures the first layer only
run005('tasks', mode='pilot')
run005('feat', mode='pilot')
rec_h = run005('heads', ['--max-tasks', '3', '--seeds', '0', '--layers', PILOT_LAYER], mode='pilot')
print('>>> heads: %.0f s, sampled peak GPU memory %s MiB, log %s' % (rec_h['seconds'], rec_h['vram_peak_sampled_mib'], rec_h['log']))
rec_d = run005('heads', [], mode='pilot', dry=True)       # head counts of the full run for the pilot plan (default seeds, all tasks, all pilot layers)
print('005 pilot total %.0f s' % (time.time() - t0))
'''))

C.append(code(r'''
# C3. 005 first-head time and projection to the full heads stage (추정; the first head includes CUDA warm-up -> the projection leans high). Based on the pilot plan's layers; the main plan may differ.
txt = open(rec_h['log'], errors='replace').read()
fh = T.parse_first_heads(txt)
for x in fh:
    print(x)
pl = T.parse_heads_plan(open(rec_d['log'], errors='replace').read())
print('full heads stage for the pilot plan (attentive heads, base heads):', pl)
if fh and pl:
    p = T.project_005(fh[0]['first_head_s'], pl[0])
    print('projected attentive-head time (추정): %s  (first head %.2f s x %d heads)' % (T.hms(p['heads_total_s']), p['first_head_s'], p['n_attentive_heads']))
print('doc estimate (005 CHANGES.md 6절, 추정): 1-1.7 h per layer; base heads ~0.3 h')
import glob
cj = WORK005 + '/results_pilot/cost.json'
if os.path.exists(cj):
    print(open(cj).read()[:1500])
'''))

C.append(md(r'''
# D. 004 main run (human approval required)
Set the flags below. `False` -> the cells do nothing. The 3-attempt guard (`results/attempts.json`) is the one in run.py: three consecutive failures of a stage make it refuse; then stop and report the cause from `logs/`.
Config `pilot` must be false for a main run (README: the human changes it before the main run). The cell refuses while `pilot: true` unless you set `ALLOW_MAIN_WITH_PILOT_TRUE`.
'''))

C.append(code(r'''
# D0. APPROVALS (CLAUDE.md: GPU/cost-heavy runs need human approval). Set by the human only.
APPROVED_MAIN_004 = False
APPROVED_MAIN_005 = False
ALLOW_MAIN_WITH_PILOT_TRUE = False      # True = run although config.yaml has pilot: true (results would be labelled pilot). Normally set pilot: false in config.yaml first.
print('APPROVED_MAIN_004 =', APPROVED_MAIN_004, '| APPROVED_MAIN_005 =', APPROVED_MAIN_005)
'''))

C.append(code(r'''
# D1. 004 MAIN: enc_cache -> enc_probe -> pred (training seeds from config: 0,1,2; random-init control included) -> aggregate. Stages resume (DONE / part files).
# Expected (추정, replace with the pilot projection printed in B2): about 5-6 h. A foreground cell stops with the kernel: after a disconnect run A again, then this cell.
if not APPROVED_MAIN_004:
    print('사람 승인 필요(CLAUDE.md): APPROVED_MAIN_004 = False -> nothing was run.')
else:
    v = T.read_yaml_values(EXP004 + '/config.yaml', ['pilot', 'seed', 'probe.seeds', 'predictor.train_seeds', 'depth.drop_threshold'])
    print('004 config:', v)
    if v['pilot'] is not False and not ALLOW_MAIN_WITH_PILOT_TRUE:
        raise SystemExit('004 config.yaml has pilot: %s. Set pilot: false (human decision, README) or ALLOW_MAIN_WITH_PILOT_TRUE=True. Nothing was run.' % v['pilot'])
    free = shutil.disk_usage('/content').free / 1e9
    assert free > 35, 'only %.0f GB free on /content: enc_cache needs ~30 GB (추정)' % free
    for st in ('enc_cache', 'enc_probe', 'pred', 'aggregate'):
        rec = run004(st, mode='main')
        print('>>> %s done in %.0f s (sampled peak GPU %s MiB), log %s' % (st, rec['seconds'], rec['vram_peak_sampled_mib'], rec['log']))
    save_results_json(EXP004, WORK004 + '/results')
'''))

C.append(code(r'''
# D2. OPTIONAL (default OFF): keep the 004 main encoder cache on Drive so that 005 can run in a LATER session without recomputing enc_cache.
# The tar is uncompressed (~30 GB estimate, 추정) and needs that much free Drive space; the write is slow. Restore: cell E1.
SAVE_EXP004_CACHE_TAR = False
TAR004 = WORK004 + '/cache_tar/exp004_enc.tar'
if SAVE_EXP004_CACHE_TAR:
    print('saved', T.tar_save(LOCAL_MAIN + '/exp004_cache/enc', TAR004), 'bytes ->', TAR004)
else:
    print('skipped (SAVE_EXP004_CACHE_TAR=False). Without it, a new session must re-run 004 `enc_cache` before 005 (cell E1 does this for you if you set REBUILD_004_ENC_CACHE).')
'''))

C.append(md(r'''
# E. 005 main run (human approval required)
005 uses (1) `exp004_work/results/summary.json` (the 004 MAIN summary) and (2) `/content/main/exp004_cache/enc` (the 004 main encoder-token cache; lost when the session ends).
'''))

C.append(code(r'''
# E0. Decisions (read-only) and prerequisites for the 005 main run.
show_005_decisions()
sm_main = WORK004 + '/results/summary.json'
print('004 main summary.json present:', os.path.exists(sm_main))
print('004 main encoder cache present:', os.path.exists(LOCAL_MAIN + '/exp004_cache/enc/DONE'), '(tar on Drive: %s)' % os.path.exists(WORK004 + '/cache_tar/exp004_enc.tar'))
'''))

C.append(code(r'''
# E1. Make the 004 encoder cache available for 005: restore from the Drive tar (RESTORE_EXP004_CACHE_TAR) or recompute (REBUILD_004_ENC_CACHE). Both default OFF; needs APPROVED_MAIN_005.
RESTORE_EXP004_CACHE_TAR = False
REBUILD_004_ENC_CACHE = False
if not APPROVED_MAIN_005:
    print('사람 승인 필요(CLAUDE.md): APPROVED_MAIN_005 = False -> nothing was run.')
elif os.path.exists(LOCAL_MAIN + '/exp004_cache/enc/DONE'):
    print('004 encoder cache already present')
elif RESTORE_EXP004_CACHE_TAR:
    print(T.tar_restore(WORK004 + '/cache_tar/exp004_enc.tar', LOCAL_MAIN + '/exp004_cache'))
elif REBUILD_004_ENC_CACHE:
    run004('enc_cache', mode='main')
else:
    print('004 encoder cache is missing: set RESTORE_EXP004_CACHE_TAR or REBUILD_004_ENC_CACHE to True and run this cell again.')
'''))

C.append(code(r'''
# E2. 005 MAIN: plan -> tasks -> feat -> heads -> metrics -> report (seeds, tasks and thresholds from config.yaml). Expected (추정, replace with C3): heads 1-1.7 h per layer.
# Stages resume (predictions/*.npz exist -> skipped). plan.hold (experiment 1 negative) makes heads refuse by design.
if not APPROVED_MAIN_005:
    print('사람 승인 필요(CLAUDE.md): APPROVED_MAIN_005 = False -> nothing was run.')
else:
    show_005_decisions()
    v = T.read_yaml_values(EXP005 + '/config.yaml', ['pilot'])
    if v['pilot'] is not False and not ALLOW_MAIN_WITH_PILOT_TRUE:
        raise SystemExit('005 config.yaml has pilot: %s. Set pilot: false (human decision) or ALLOW_MAIN_WITH_PILOT_TRUE=True. Nothing was run.' % v['pilot'])
    assert os.path.exists(WORK004 + '/results/summary.json'), '004 main summary.json missing (section D)'
    assert os.path.exists(LOCAL_MAIN + '/exp004_cache/enc/DONE'), '004 main encoder cache missing (cell E1)'
    for st in ('plan', 'tasks', 'feat', 'heads', 'metrics', 'report'):
        rec = run005(st, mode='main')
        print('>>> %s done in %.0f s (sampled peak GPU %s MiB), log %s' % (st, rec['seconds'], rec['vram_peak_sampled_mib'], rec['log']))
    save_results_json(EXP005, WORK005 + '/results')
'''))

C.append(md(r'''
# F. Results
Files to bring back into the repo (by hand, on your PC): `exp004_work/results/results.json` -> `experiments/004_depth_truncation/results.json`, `exp004_work/results/summary.json` -> `experiments/004_depth_truncation/results/summary.json` (005 plan reads it there), `exp005_work/results/results.json` -> `experiments/005_inductive_bias_probe/results.json`. Compare the sha256 printed below after copying (`sha256sum`).
'''))

C.append(code(r'''
# F1. sha256 of the result files on Drive (main) and the pilot copies. Set DOWNLOAD_BUNDLE=True to also download a zip of the small result files (csv/json/md/png, no caches, no predictions).
DOWNLOAD_BUNDLE = False
T.print_sha_table([WORK004 + '/results/results.json', WORK004 + '/results/summary.json', WORK005 + '/results/results.json', WORK005 + '/results/REPORT.md',
                   WORK004 + '/results_pilot/results.json', WORK004 + '/results_pilot/summary.json'])
for w in (WORK004, WORK005):
    for f in ('results/attempts.json', 'results_pilot/attempts.json'):
        if os.path.exists(w + '/' + f):
            print(w + '/' + f, open(w + '/' + f).read().replace('\n', ' '))
if DOWNLOAD_BUNDLE:
    import zipfile, glob
    zp = '/content/exp004_005_results_bundle.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for w in (WORK004, WORK005):
            for ext in ('json', 'csv', 'md', 'png'):
                for f in glob.glob(w + '/results*/*.' + ext) + glob.glob(w + '/results*/parts/*.csv'):
                    z.write(f, os.path.relpath(f, os.path.dirname(w)))
    from google.colab import files
    files.download(zp)
'''))

C.append(code(r'''
# F2. Flush Drive writes before leaving.
from google.colab import drive
drive.flush_and_unmount()
'''))


def main():
    b64, nbytes = make_b64()
    expected = json.loads((EXPS / "002_single_size_sweep" / "results.json").read_text())["config_sha256"]
    stamp = time.strftime("%F %T")
    for c in C:
        if c["cell_type"] == "code":
            c["source"] = [s.replace("@@B64@@", b64).replace("@@STAMP@@", stamp).replace("@@CFG002SHA@@", expected) for s in c["source"]]
            if any("@@B64@@" in s for s in c["source"]):
                raise SystemExit("placeholder left")
    nb = {"cells": C, "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    out = HERE / "exp004_005_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", out, "snapshot %d bytes (tar.gz)" % nbytes, "cells", len(C))


if __name__ == "__main__":
    main()
