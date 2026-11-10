#!/usr/bin/env python3
"""Regenerate colab/exp009_colab.ipynb:   python3 colab/build_notebook.py     (run again after ANY code change)
The notebook has NO embedded code: cell A3 git-clones the public repo (REPO_URL, REPO_REF set in cell A2) and records HEAD.
The ref must contain 002 {src, run.py, config.yaml, colab/colab_tools.py, colab/requirements_colab.txt}, 004 {src, run.py, config.yaml,
colab/exp045_tools.py, colab/requirements_colab_extra.txt} and 009 {src, run.py, config.yaml}. Reads 002/results.json config_sha256 at build time. config.yaml values are NOT changed by the notebook."""
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
# Experiment 009 (positive control for the 004 depth-truncation pipeline) on Google Colab
Order: **A** settings + setup, **B** check (dry runs + real check), **C** selftest (CPU toy), **D** optional pilot (1 train seed, block 6), **E** intervene (main), **F** aggregate, **G** copy results out.
Every cell is idempotent; after a disconnect choose a GPU runtime, run A and B again and continue: `intervene` skips finished units (part files on Drive).

Config (user-confirmed 2026-11-10, in `experiments/009_positive_control_depth/config.yaml`, **not changed by this notebook**): method `leace_scalar`, strengths [0, 1.0], m_int [1..6], train seeds [0,1,2], probe seeds [0..4], success rule strength 1.0 / 2 seeds / slack 1. Planned units: 21 = 3 baselines + 6 blocks x 1 strength x 3 seeds.
The only overrides used anywhere are existing CLI flags of `run.py` (pilot cell: `--m-int 6 --train-seeds 0 --results-dir`), which `results.json` records (`m_int_used`, `m_int_overridden`, `train_seeds`).

Inputs (read-only): the three 002 condition-(b) checkpoints (`multi_60_ind_160`, seeds 0,1,2), the 002 Q1 cache, and 004 `results/depth_r2.csv` (random-init reference for auxiliary rule A and the baseline reproduction check). **No world model is trained.** Data: no mp4 download, but the per-split `metadata.pt` files (actions; needed for the action mean/std) must exist locally under `/content/dinowm/data/cloth_sim/<split>/` (cell A6b puts them there; first Colab pilot failed without them). Nothing in `exp002_work` / `exp004_work` is written except as noted in cell A6 (a read-only copy of depth_r2.csv goes into the LOCAL clone).
Outputs (Drive): `MyDrive/exp009_work/{results,logs,results_pilot,timing_*.json}`; `results.json` is copied to `exp009_work/results/` after aggregate.

Human approval (CLAUDE.md): GPU runs need you. The pilot needs `APPROVED_PILOT_009 = True`, the main intervene needs `APPROVED_MAIN_009 = True` (cell D0). All time numbers printed from docs are estimates (추정). Everything here is UNVERIFIED on Colab (written and checked on a CPU-only machine).
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
# A2. SETTINGS (all Drive paths and options live here). Mount Drive first, then check that the paths below are right for YOUR Drive.
import os
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive'
WORK002 = os.environ.get('EXP002_WORK_ROOT', DRIVE + '/exp002_work')     # READ-ONLY: train_runs/outputs/exp002_multi_60_ind_160_s{0,1,2}/ (checkpoints), exp002_cache/q1_tokens/, results/results.json
WORK004 = os.environ.get('EXP004_WORK_ROOT', DRIVE + '/exp004_work')     # READ-ONLY: results/depth_r2.csv (004 main run; same convention as the 004/005 notebook)
WORK009 = os.environ.get('EXP009_WORK_ROOT', DRIVE + '/exp009_work')     # OUTPUT (persistent): results/, logs/, results_pilot/, timing_*.json
DEPTH_R2_004 = WORK004 + '/results/depth_r2.csv'                         # 004 random-init reference (auxiliary rule A) + baseline reproduction check
DATA_SRC = os.environ.get('EXP002_DRIVE_MOUNT_SRC', DRIVE + '/dino-wm-videos/[data]/[cloth-sim] cloth_move')   # same as the 004 notebook; only <split>/metadata.pt is read from here (cell A6b)
REPO_URL = 'https://github.com/jongmin-s-multicore/probing-experiment.git'     # PUBLIC repository (cloned in A3)
REPO_REF = 'master'            # branch name OR commit hash (or tag) to check out in A3; use a commit hash for an exactly reproducible run
GITHUB_TOKEN = None            # optional; not needed for the public repo (e.g. userdata.get('GITHUB_TOKEN') if you ever use a private fork)
LOCAL_ROOT = '/content/main'                                             # big temporary features (up to ~15 GB per unit, 추정) on the Colab local disk
os.makedirs(WORK009, exist_ok=True)
for k, v in (('WORK002', WORK002), ('WORK004', WORK004), ('WORK009', WORK009), ('DEPTH_R2_004', DEPTH_R2_004)):
    print('%-13s = %s  exists=%s' % (k, v, os.path.exists(v)))
'''))

C.append(code(r'''
# A3. Code: git clone of the PUBLIC repo REPO_URL, checked out at REPO_REF (settings in cell A2). No embedded snapshot.
# The repo ref must contain 002, 004 (colab/exp045_tools.py etc.) and 009 files (commit and push them first).
EXPECTED_CFG002_SHA = '@@CFG002SHA@@'    # 002/results.json config_sha256 read at notebook build time (@@STAMP@@)
import json, os, subprocess
REPO = '/content/mylab'
EXPS = REPO + '/experiments'
EXP002, EXP004, EXP009 = EXPS + '/002_single_size_sweep', EXPS + '/004_depth_truncation', EXPS + '/009_positive_control_depth'
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
os.makedirs(WORK009, exist_ok=True)
json.dump(CODE_VERSION, open(WORK009 + '/code_version.json', 'w'), indent=1)     # environment record next to timing_*.json (also in results.json: code_version, via env vars set in A6)
for p in (EXP002 + '/run.py', EXP004 + '/run.py', EXP009 + '/run.py', EXP009 + '/config.yaml', EXP004 + '/colab/exp045_tools.py', EXP002 + '/colab/requirements_colab.txt'):
    assert os.path.isfile(p), 'missing ' + p
import sys
sys.path.insert(0, EXP004 + '/colab')
import importlib, exp045_tools as T
importlib.reload(T)
print('exp045_tools loaded from', T.__file__)
'''))

C.append(code(r'''
# A4. Python environment: uv venv (3.9) with the 002 pins (+ matplotlib). Colab has no conda: documented exception to CLAUDE.md (same as 002/003/004).
# USE_SYSTEM_PYTHON=True uses Colab's own Python/torch; results are then NOT comparable with the 002 environment (record it). torch 2.0.x comes from the 002 pins.
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
# A6. Environment variables, Drive-backed results/logs, 004 depth_r2.csv copy, stage helpers.
import os, json, shutil
for d in (LOCAL_ROOT + '/exp009_cache', '/content/pilot/exp009_cache'):
    os.makedirs(d, exist_ok=True)
base_env = dict(os.environ)
base_env.update({
    'DINO_WORK': '/content/dinowm', 'DATASET_DIR': '/content/dinowm/data',
    'DINO_TRAIN': WORK002 + '/train_runs',                                          # READ-ONLY use: checkpoints
    'EXP002_CACHE': WORK002 + '/exp002_cache',                                      # READ-ONLY use: q1_tokens
    'ACWM_PHYS_SRC': SRC, 'WANDB_MODE': 'disabled', 'HYDRA_FULL_ERROR': '1', 'PYTHONUNBUFFERED': '1',
    'EXP009_CODE_REPO_URL': REPO_URL, 'EXP009_CODE_REF': REPO_REF,                  # recorded in results.json code_version (git_head is read from the clone)
    'MPLBACKEND': 'Agg',                                                            # also forced inside exp045_tools (the Colab kernel's inline backend is invalid in the venv)
    'PATH': os.path.dirname(PY) + os.pathsep + os.environ['PATH'],
})
# NOTE: PYTORCH_CUDA_ALLOC_CONF is deliberately not set (expandable_segments is unknown to torch 2.0.x; 003 commit 16edd98).
base_env.pop('EXP009_M_INT', None)             # a stale env override must not leak in; the config value is used unless a CLI flag is given
if not USE_SYSTEM_PYTHON:
    base_env['VIRTUAL_ENV'] = VENV
ENV_MAIN = dict(base_env, EXP009_CACHE=LOCAL_ROOT + '/exp009_cache')
ENV_PILOT = dict(base_env, EXP009_CACHE='/content/pilot/exp009_cache')
print(json.dumps({k: ENV_MAIN[k] for k in ('DINO_TRAIN', 'EXP002_CACHE', 'EXP009_CACHE', 'ACWM_PHYS_SRC')}, indent=1))
print('009:', T.link_dirs(WORK009, EXP009, ('results', 'logs')))
os.makedirs(WORK009 + '/results_pilot', exist_ok=True)
# 009 reads 004 results/depth_r2.csv from <004 folder>/results (fixed path in the code): put a READ-ONLY COPY into the local clone. The Drive original is untouched.
os.makedirs(EXP004 + '/results', exist_ok=True)
if os.path.exists(DEPTH_R2_004):
    shutil.copyfile(DEPTH_R2_004, EXP004 + '/results/depth_r2.csv')
    print('004 depth_r2.csv copied; sha256', T.sha256_file(EXP004 + '/results/depth_r2.csv')[:16], 'rows', sum(1 for _ in open(EXP004 + '/results/depth_r2.csv')) - 1)
else:
    print('!! %s missing: aggregate will mark aux rule A as 계산 불가 and baseline_vs_004 as not_available' % DEPTH_R2_004)

MODES = {
    'pilot': dict(env=ENV_PILOT, results=WORK009 + '/results_pilot', timing=WORK009 + '/timing_pilot.json'),
    'main': dict(env=ENV_MAIN, results=WORK009 + '/results', timing=WORK009 + '/timing_main.json'),
}

def run009(stage, extra=(), mode='main', dry=False):
    m = MODES[mode]
    args = ['--results-dir', m['results']] + list(extra) + (['--dry-run'] if dry else [])
    rec = T.run_stage(PY, EXP009, stage, args, m['env'], timing_file=None if dry else m['timing'], label='009:%s%s' % (stage, '(dry)' if dry else ''))
    T.require_ok(rec, m['results'])
    return rec

def save_results_json(dest_dir):
    # aggregate writes <repo exp folder>/results.json (local clone, lost with the session): copy to Drive right away
    src = EXP009 + '/results.json'
    if os.path.exists(src):
        shutil.copyfile(src, dest_dir + '/results.json')
        print('copied', src, '->', dest_dir + '/results.json')
    else:
        print('no', src)
'''))

C.append(code(r'''
# A6b. metadata.pt of the training manifests (needed by exp002 model_io.action_stats_for_run -> ManifestDataset: action mean/std). Only metadata.pt files, no mp4.
# Sources in order: (a) already local; (b) DATA_SRC/<split>/metadata.pt or DATA_SRC/cloth_sim/<split>/metadata.pt; (c) member of WORK002/cloth_sim_data.tar (unsafe members refused).
import json, os, shutil, tarfile
from pathlib import Path
need = {}                                   # local dir (as written in the manifest) -> train seeds
for ts in (0, 1, 2):
    mp = Path(WORK002) / 'train_runs' / 'outputs' / ('exp002_multi_60_ind_160_s%d' % ts) / 'train_manifest.json'
    assert mp.exists(), 'missing ' + str(mp)
    for ep in json.load(open(mp))['episodes']:
        need.setdefault(ep['dir'], set()).add(ts)
print('episode dirs in the manifests:', {d: sorted(v) for d, v in need.items()})
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
# A7. Disk. Estimate (추정, 009 README): temporary features about 15 GB per unit at most, deleted at the end of each unit.
T.disk_report(['/content', WORK009], need_gb={'009 tmp features (1 unit at a time)': 15})
free_local = shutil.disk_usage('/content').free / 1e9
print('free local disk: %.0f GB' % free_local, ('  !! LOW (< 25 GB)' if free_local < 25 else ''))
'''))

C.append(md(r'''
# B. Check (read-only; nothing is trained)
'''))

C.append(code(r'''
# B1. Inputs on Drive (002 checkpoints DONE, final checkpoint, Q1 cache, 002 config fingerprint). Stops with a clear message if something is missing. Writes nothing.
ok, bad, lines = T.check_inputs(WORK002, [0, 1, 2], suffixes=('',), expected_epoch=15, expected_cfg_sha=EXPECTED_CFG002_SHA, cfg002_path=EXP002 + '/config.yaml', need_manifest=False)
print('\n'.join(lines))
if not ok:
    raise SystemExit('INPUTS MISSING - stopping (nothing was run):\n  - ' + '\n  - '.join(bad) + '\n  Fix the paths in cell A2 (WORK002 / WORK004).')
assert os.path.exists(DEPTH_R2_004), 'missing %s (004 results/depth_r2.csv)' % DEPTH_R2_004
print('inputs OK')
'''))

C.append(code(r'''
# B2. 009 `check` and the dry runs (they print the plan: expected "units total 21 (baseline 3 + intervention 18)"). No GPU work, no attempts recorded.
run009('check', mode='main')
run009('intervene', mode='main', dry=True)
run009('aggregate', mode='main', dry=True)
print(T.read_yaml_values(EXP009 + '/config.yaml', ['pilot', 'seed', 'intervention.method', 'intervention.strengths', 'intervention.m_int', 'predictor.train_seeds', 'probe.seeds', 'intervention.fit_seed', 'success']))
'''))

C.append(md(r'''
# C. Selftest (CPU, synthetic data + toy model; expected: passed=65 failed=0)
'''))

C.append(code(r'''
# C1. selftest (does not touch Drive inputs).
rec = run009('selftest', mode='main')
print('selftest %.0f s, log %s' % (rec['seconds'], rec['log']))
'''))

C.append(md(r'''
# D. OPTIONAL pilot: baseline + 1 unit (train seed 0, block 6, strength 1.0) with the REAL checkpoint, results in `exp009_work/results_pilot`
Uses only existing CLI flags (`--m-int 6 --train-seeds 0`; recorded in the pilot results.json). Block 6 is the cheapest block (only depth 6 is re-forwarded/probed) and has no recovery observation; it measures time, VRAM and finds errors. Pilot and main never share a folder.
'''))

C.append(code(r'''
# D0. APPROVALS (CLAUDE.md: GPU/cost-heavy runs need human approval). Set by the human only.
APPROVED_PILOT_009 = False
APPROVED_MAIN_009 = False
ALLOW_MAIN_WITH_PILOT_TRUE = False      # True = run the main although config.yaml has pilot: true (results.json would then say pilot: true). Normally the human sets pilot: false in config.yaml first.
PILOT_M_INT = '6'                       # CLI --m-int for the pilot only
PILOT_TRAIN_SEEDS = '0'                 # CLI --train-seeds for the pilot only
print('APPROVED_PILOT_009 =', APPROVED_PILOT_009, '| APPROVED_MAIN_009 =', APPROVED_MAIN_009)
'''))

C.append(code(r'''
# D1. Pilot: intervene (baseline for seed 0 + block 6 at strength 1.0) -> aggregate. Time and sampled VRAM go to exp009_work/timing_pilot.json.
import time
if not APPROVED_PILOT_009:
    print('사람 승인 필요(CLAUDE.md): APPROVED_PILOT_009 = False -> nothing was run.')
else:
    t0 = time.time()
    ex = ['--m-int', PILOT_M_INT, '--train-seeds', PILOT_TRAIN_SEEDS]
    for st in ('intervene', 'aggregate'):
        rec = run009(st, ex, mode='pilot')
        print('>>> %s: %.0f s, sampled peak GPU memory %s MiB (base %s), log %s' % (st, rec['seconds'], rec['vram_peak_sampled_mib'], rec['vram_base_mib'], rec['log']))
    save_results_json(WORK009 + '/results_pilot')
    print('pilot total %.0f s (baseline for 1 train seed + 1 intervention unit)' % (time.time() - t0))
'''))

C.append(code(r'''
# D2. Pilot timing and a (rough) re-estimate of the main run. The main has 3 baselines + 18 units; the pilot has 1 baseline + 1 unit of the cheapest block.
import json
tp = WORK009 + '/timing_pilot.json'
if os.path.exists(tp):
    for r in json.load(open(tp)):
        if r['rc'] == 0 and '(dry)' not in r['label']:
            print('%-22s %7.0f s  peak VRAM (sampled whole GPU) %s MiB  log %s' % (r['label'], r['seconds'], r['vram_peak_sampled_mib'], r['log']))
    print('The pilot unit is block 6 (one depth); blocks 1..5 re-forward and probe 6..2 depths and are slower (추정). README estimate before measurement: about 1.8-3.3 h for the whole main run (추정).')
else:
    print('no pilot timing yet')
'''))

C.append(md(r'''
# E. Main intervene (needs `APPROVED_MAIN_009 = True` in cell D0)
'''))

C.append(code(r'''
# E1. intervene: all units from config (m_int [1..6], strengths [0, 1.0], train seeds [0,1,2]) = 21 units. Finished units (Drive part files) are skipped; rerun this cell after a disconnect (A and B first).
# Expected GPU time: README estimate 1.8-3.3 h (추정, no measurement). A foreground cell stops with the kernel.
if not APPROVED_MAIN_009:
    print('사람 승인 필요(CLAUDE.md): APPROVED_MAIN_009 = False -> nothing was run.')
else:
    v = T.read_yaml_values(EXP009 + '/config.yaml', ['pilot', 'seed', 'intervention.method', 'intervention.strengths', 'intervention.m_int', 'predictor.train_seeds', 'probe.seeds'])
    print('009 config:', v)
    if v['pilot'] is not False and not ALLOW_MAIN_WITH_PILOT_TRUE:
        raise SystemExit('009 config.yaml has pilot: %s. The human sets pilot: false (README) or ALLOW_MAIN_WITH_PILOT_TRUE=True. Nothing was run.' % v['pilot'])
    assert shutil.disk_usage('/content').free / 1e9 > 20, 'less than 20 GB free on /content (units need up to ~15 GB temporary, 추정)'
    rec = run009('intervene', mode='main')
    print('>>> intervene done in %.0f s (sampled peak GPU %s MiB), log %s' % (rec['seconds'], rec['vram_peak_sampled_mib'], rec['log']))
'''))

C.append(md(r'''
# F. Aggregate (CPU; per-block verdicts, auxiliary rules A and B, results.json)
'''))

C.append(code(r'''
# F1. aggregate -> <clone>/results.json and results/depth_r2_009.csv (on Drive via the results symlink). results.json is copied to Drive immediately.
if not APPROVED_MAIN_009:
    print('APPROVED_MAIN_009 = False -> nothing was run (aggregate follows the main intervene).')
else:
    rec = run009('aggregate', mode='main')
    save_results_json(WORK009 + '/results')
    r = json.load(open(EXP009 + '/results.json'))
    print({k: r[k] for k in ('pilot', 'm_int_used', 'm_int_overridden', 'strengths', 'train_seeds', 'method')})
    print({m: b['verdict']['status'] for m, b in r['blocks'].items()})
'''))

C.append(md(r'''
# G. Copy results out
'''))

C.append(code(r'''
# G1. What is on Drive now. Optional zip download (DOWNLOAD_BUNDLE).
DOWNLOAD_BUNDLE = False
import glob
for f in sorted(glob.glob(WORK009 + '/*') + glob.glob(WORK009 + '/results/*') + glob.glob(WORK009 + '/results_pilot/*')):
    print(f, os.path.getsize(f) if os.path.isfile(f) else '(dir)')
for f in ('results/attempts.json', 'results_pilot/attempts.json'):
    if os.path.exists(WORK009 + '/' + f):
        print(WORK009 + '/' + f, open(WORK009 + '/' + f).read().replace('\n', ' '))
if DOWNLOAD_BUNDLE:
    import zipfile
    zp = '/content/exp009_results_bundle.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for ext in ('json', 'csv', 'log'):
            for f in glob.glob(WORK009 + '/**/*.' + ext, recursive=True):
                z.write(f, os.path.relpath(f, os.path.dirname(WORK009)))
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
    out = HERE / "exp009_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", out, "cells", len(C))


if __name__ == "__main__":
    main()
