#!/usr/bin/env python3
"""Regenerate colab/exp006_colab.ipynb:   python3 colab/build_notebook.py     (run again after ANY change to this file)
The notebook has NO embedded code: cell 3 git-clones the public repo (REPO_URL, REPO_REF set in cell 2) and records HEAD (code_version.json, results.json code_version).
The ref must contain 002 {src, config.yaml, results.json, colab/requirements_colab.txt} and 006 {src, run.py, config.yaml} (commit and push them first).
Reads 006 config.yaml base.config002_sha256 at build time. Nothing in 002/004/009 is imported or modified; the helpers (sh, bg, run_stage) are defined in the notebook.
Local paths (/content/mylab006, /content/venv006, /content/dinowm006) and the Drive output (MyDrive/exp006_work) do not overlap with 009 (/content/mylab, /content/venv,
/content/dinowm, MyDrive/exp009_work)."""
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


C0 = r'''
# Experiment 006: re-analysis of the 002 single-size sweep, analyses A-F (exploratory, post hoc) on Colab
Run the cells top to bottom. Every cell is idempotent (after a disconnect: choose the runtime, run cells 1-9 again, then continue). **This notebook only READS `MyDrive/exp002_work` and WRITES `MyDrive/exp006_work`.**
It can run next to the 009 notebook: separate Drive output folder (`exp006_work`), separate local folders (`/content/mylab006`, `/content/venv006`, `/content/dinowm006`), no shared files.

What it does (nothing is trained): loads the 6 finished 002 checkpoints (2 conditions x 3 training seeds), recomputes from the 002 cached frames (a) the open-loop rollout of the trained predictor, (b) rollouts of untrained predictors, (c) **new in F: the teacher-forced one-step predictions** (each step predicted from the REAL encoder latent of the previous frame; possible for steps 1-12, all of them), and computes analyses A-F of `notes/hypotheses.md` candidate J. No thresholds, no pass/fail, `interpretation` is null.

**Code comes from git** (cell 3: clone of the public repo, checkout of `REPO_REF`, HEAD hash recorded). The 006 changes (analysis F, this notebook) take effect on Colab only after they are committed and pushed to `REPO_REF`.

**GPU time (ESTIMATE, 추정, nothing measured): about 0.35-1.8 GPU hours for the 6 runs** (earlier A-E estimate 0.3-1.5 h plus about 18% for the teacher-forced pass: one more 12-step pass over all 1800 episodes per run, compared with 7200 episode-rollouts before). Reading the 3.5 GB frame cache from Drive may dominate. GPU memory about 3-6 GB (추정). Cell 9 (smoke) prints a measured estimate before the full extraction. The CPU stages (analysis F adds roughly 2x the ridge work of C): a few minutes to a few tens of minutes (추정). Selftest on Colab CPU: several minutes (추정).

Human approval (CLAUDE.md): the user approved running this on Colab together with 009 on 2026-11-10 (`notes/hypotheses.md`, candidate J addition). The full extraction is still OFF until you set `RUN_EXTRACT = True` in cell 10 after reading the smoke estimate.
Persistent output: `WORK_ROOT = MyDrive/exp006_work` (`extract/`, `cache/`, `results/` incl. `results.json`, `logs/`, `figures/`, `code_version.json`). `/content` is rebuilt every session. A stage that fails 3 times stops (CLAUDE.md retry limit).
Everything here is UNVERIFIED on Colab as a whole (written and checked on a CPU-only machine).
'''

C1 = r'''
# 1. GPU / Python check + helpers
import sys, subprocess, os, time
def sh(cmd, cwd=None):
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
    return p.wait()

def bg(cmd, log, cwd=None, env=None, every=20):
    # run cmd in its own session, output to `log`; tail it until it ends; returns the exit code (None if the cell was interrupted: the job keeps running)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    rcfile = log + '.rc'
    if os.path.exists(rcfile):
        os.remove(rcfile)
    full = '(%s) >> %s 2>&1; echo $? > %s' % (cmd, log, rcfile)
    env = dict(os.environ if env is None else env, MPLBACKEND='Agg')
    subprocess.Popen(['nohup', 'bash', '-c', full], cwd=cwd, env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shown = 0
    try:
        while not os.path.exists(rcfile):
            time.sleep(every)
            lines = open(log, errors='replace').read().splitlines() if os.path.exists(log) else []
            print('\n'.join(lines[shown:][-15:]), flush=True)
            shown = len(lines)
    except KeyboardInterrupt:
        print('cell interrupted; the background job keeps running (log: %s)' % log)
        return None
    lines = open(log, errors='replace').read().splitlines()
    print('\n'.join(lines[shown:][-40:]))
    return int(open(rcfile).read().strip() or 1)

sh('nvidia-smi')
print(sys.version)
sh('df -h /content | tail -1')
'''

C2 = r'''
# 2. SETTINGS. Mount Drive, then check that the paths are right for YOUR Drive. INPUT_ROOT is READ ONLY by construction (run.py refuses to write there); WORK_ROOT receives all outputs.
import os
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive'
INPUT_ROOT = os.environ.get('EXP006_INPUT_ROOT', DRIVE + '/exp002_work')   # READ-ONLY: train_runs/outputs/<run>/ (checkpoints), exp002_cache/q1_tokens/ (cache), results/ (002 q2_cells.npz, results.json), cloth_sim_data.tar (data archive: only metadata.pt is read)
WORK_ROOT = os.environ.get('EXP006_WORK_ROOT', DRIVE + '/exp006_work')     # OUTPUT (persistent). Own folder, separate from the 009 notebook's output folder
DATA_SRC = os.environ.get('EXP006_DATA_SRC', DRIVE + '/dino-wm-videos/[data]/[cloth-sim] cloth_move')   # optional source of <split>/metadata.pt if the archive is not usable (cell 7)
REPO_URL = 'https://github.com/jongmin-s-multicore/probing-experiment.git'     # PUBLIC repository (cloned in cell 3)
REPO_REF = 'master'          # branch name OR commit hash OR tag to check out in cell 3; use a commit hash for an exactly reproducible run
GITHUB_TOKEN = None          # optional; not needed for the public repo
REPO = '/content/mylab006'   # local clone (009 uses /content/mylab)
assert os.path.isdir(INPUT_ROOT), 'INPUT_ROOT %s not found: is the 002 Drive folder where you expect it?' % INPUT_ROOT
assert os.path.realpath(INPUT_ROOT) != os.path.realpath(WORK_ROOT) and not os.path.realpath(WORK_ROOT).startswith(os.path.realpath(INPUT_ROOT) + os.sep), 'WORK_ROOT must differ from INPUT_ROOT'
assert 'exp009' not in WORK_ROOT, 'WORK_ROOT must not be the 009 folder'
os.makedirs(WORK_ROOT, exist_ok=True)
probe = os.path.join(WORK_ROOT, '.write_test_notebook')
with open(probe, 'w') as f:                       # a Drive write failure must stop here, not later
    f.write('ok'); f.flush(); os.fsync(f.fileno())
assert open(probe).read() == 'ok', 'Drive write/read-back failed'
print('INPUT_ROOT (read only) =', INPUT_ROOT)
print('WORK_ROOT  (outputs)   =', WORK_ROOT, '| write test OK')
for n in ('train_runs/outputs', 'exp002_cache/q1_tokens', 'exp002_cache/q1_tokens/DONE', 'results', 'results/q1/cells.npz', 'cloth_sim_data.tar'):
    print('  %-34s exists: %s' % (n, os.path.exists(os.path.join(INPUT_ROOT, n))))
for r in ['exp002_%s_s%d' % (c, s) for c in ('multi_60_ind_160', 'single_ind') for s in (0, 1, 2)]:
    print('  %-34s checkpoint: %s' % (r, os.path.exists(os.path.join(INPUT_ROOT, 'train_runs/outputs', r, 'checkpoints/model_latest.pth'))))
'''

C3 = r'''
# 3. Code: git clone of the PUBLIC repo REPO_URL, checked out at REPO_REF (settings in cell 2). No embedded snapshot.
# The ref must contain 002 {src, config.yaml, results.json, colab/requirements_colab.txt} and 006 {src, run.py, config.yaml}: commit and push them first.
EXPECTED_CFG002_SHA = '@@CFG002SHA@@'    # 006 config.yaml base.config002_sha256, read at notebook build time (@@STAMP@@)
import hashlib, json, os, subprocess
EXPS = REPO + '/experiments'
EXP, E002 = EXPS + '/006_size_readout_reanalysis', EXPS + '/002_single_size_sweep'
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
    target = 'origin/' + REPO_REF                       # branch name
else:
    target = REPO_REF                                   # commit hash or tag
git('checkout', '--force', '--detach', target)
HEAD_SHA = git('rev-parse', 'HEAD')
CODE_VERSION = {'repo_url': REPO_URL, 'requested_ref': REPO_REF, 'git_head': HEAD_SHA, 'head_subject': git('log', '-1', '--format=%s'), 'head_date': git('log', '-1', '--format=%cI')}
print('code at HEAD', HEAD_SHA, '(ref %s)' % REPO_REF)
print(git('log', '--oneline', '-1'))
os.makedirs(WORK_ROOT, exist_ok=True)
json.dump(CODE_VERSION, open(WORK_ROOT + '/code_version.json', 'w'), indent=1)     # also in results.json: code_version (env vars EXP006_CODE_* set in cell 6)
need = [EXP + '/run.py', EXP + '/config.yaml', EXP + '/src/exp006/stages_f.py', E002 + '/config.yaml', E002 + '/results.json', E002 + '/src/exp002/model_io.py', E002 + '/colab/requirements_colab.txt']
miss = [p for p in need if not os.path.isfile(p)]
assert not miss, 'missing in the cloned ref %s (commit and push the 006 changes, or fix REPO_REF): %s' % (REPO_REF, miss)
sha = hashlib.sha256(open(E002 + '/config.yaml', 'rb').read()).hexdigest()
assert sha == EXPECTED_CFG002_SHA, '002 config.yaml sha256 differs from the pinned value: ' + sha
print('code in', EXP, '| 002 config sha256 OK')
'''

C4 = r'''
# 4. Environment: uv venv (Python 3.9) with the 002 requirements (colab/requirements_colab.txt of 002, unchanged) + matplotlib (figures only). Own venv (/content/venv006), not shared with 009.
import hashlib, os
VENV = '/content/venv006'
PY = VENV + '/bin/python'
REQ = E002 + '/colab/requirements_colab.txt'
MARK = VENV + '/.exp006_ok_' + hashlib.sha256(open(REQ, 'rb').read()).hexdigest()[:12]
if not os.path.exists(MARK):
    assert sh('pip install -q uv') == 0
    if not os.path.exists(PY):
        assert sh('uv venv --python 3.9 %s' % VENV) == 0
    assert sh('uv pip install --python %s --index-strategy unsafe-best-match -r %s' % (PY, REQ)) == 0
    sh('uv pip install --python %s matplotlib' % PY)          # optional: figures are skipped when missing
    open(MARK, 'w').write('ok\n')
CHK = "import sys,pkg_resources,torch,numpy,cv2,decord,hydra,scipy,yaml;print(sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available(),'numpy',numpy.__version__,'scipy',scipy.__version__)"
assert sh('%s -c "%s"' % (PY, CHK)) == 0, 'environment check failed'
'''

C5 = r'''
# 5. ACWM-Phys checkout at the pinned commit (== 002 config.yaml code.commit). NO resume patch: loading a checkpoint needs the model classes only (README section 4). Own folder (/content/dinowm006).
import subprocess, re
COMMIT = re.search(r'commit:\s*"([0-9a-f]{40})"', open(E002 + '/config.yaml').read()).group(1)
SRC = '/content/dinowm006/acwm_phys_src'
if not os.path.isdir(SRC + '/.git'):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/jongmin-s-multicore/thread0.git', SRC], check=True)
subprocess.run(['git', '-C', SRC, 'checkout', '--detach', COMMIT], check=True)
head = subprocess.run(['git', '-C', SRC, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
assert head == COMMIT, (head, COMMIT)
print('ACWM-Phys HEAD', head, '| (the 002 checkpoints were written by train.py + 002 train_resume.patch; not needed to load)')
'''

C6 = r'''
# 6. Environment variables, Drive-backed logs, stage runner
import os, shutil
DATASET_DIR = '/content/dinowm006/data'
os.environ.update({
    'EXP006_INPUT_ROOT': INPUT_ROOT, 'EXP006_WORK_ROOT': WORK_ROOT,
    'EXP006_EXP002_DIR': E002,                       # the 002 files of the clone, read only
    'ACWM_PHYS_SRC': SRC, 'DATASET_DIR': DATASET_DIR, 'WANDB_MODE': 'disabled',
    'EXP006_CODE_REPO_URL': REPO_URL, 'EXP006_CODE_REF': REPO_REF,        # recorded in results.json code_version (git_head is read from the clone)
    'PYTHONDONTWRITEBYTECODE': '1', 'MPLBACKEND': 'Agg',      # the kernel's MPLBACKEND (module://matplotlib_inline...) is invalid in the venv subprocesses (hit in 004)
    'PATH': os.path.dirname(PY) + os.pathsep + os.environ['PATH'], 'VIRTUAL_ENV': VENV,
})
tgt, lnk = WORK_ROOT + '/logs', EXP + '/logs'
os.makedirs(tgt, exist_ok=True)
if os.path.islink(lnk) and os.path.realpath(lnk) != os.path.realpath(tgt):
    os.remove(lnk)
if not os.path.islink(lnk):
    if os.path.isdir(lnk):
        shutil.copytree(lnk, tgt, dirs_exist_ok=True); shutil.rmtree(lnk)
    os.symlink(tgt, lnk)
print(lnk, '->', os.path.realpath(lnk))
def run_stage(stage, extra='', every=20):
    rc = bg('%s run.py %s %s' % (PY, stage, extra), EXP + '/logs/colab_%s.log' % stage, cwd=EXP, every=every)
    print('stage %s rc = %s' % (stage, rc))
    assert rc == 0, 'stage %s failed or was interrupted (rc=%s). A stage stops after 3 failed attempts (rc 3); rc 4 = safety stop / missing metadata.pt, not retried. Read the lines above.' % (stage, rc)
    return rc
'''

C7 = r'''
# 7. metadata.pt of the training splits (action normalisation statistics used by `extract`; no mp4 needed). The first 009 Colab pilot failed because these files were missing, so: place them, then verify.
# Sources in order: already local; DATA_SRC/<split>/metadata.pt (or DATA_SRC/cloth_sim/<split>/...); a member of INPUT_ROOT/cloth_sim_data.tar (read only, unsafe members refused).
import json, os, shutil, tarfile
from pathlib import Path
RUNS = ['exp002_%s_s%d' % (c, s) for c in ('multi_60_ind_160', 'single_ind') for s in (0, 1, 2)]
splits = set()
for r in RUNS:
    mp = Path(INPUT_ROOT) / 'train_runs' / 'outputs' / r / 'train_manifest.json'
    assert mp.exists(), 'missing ' + str(mp)
    splits |= {e['split'] for e in json.load(open(mp))['episodes']}
splits = sorted(splits)
print('splits needed by the 6 train manifests:', splits)
tar_path = Path(INPUT_ROOT) / 'cloth_sim_data.tar'
for sp in splits:
    dest = Path(DATASET_DIR) / 'cloth_sim' / sp / 'metadata.pt'
    if dest.exists():
        print('local ok   ', dest); continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    for cand in (Path(DATA_SRC) / sp / 'metadata.pt', Path(DATA_SRC) / 'cloth_sim' / sp / 'metadata.pt'):
        if cand.exists():
            shutil.copyfile(cand, dest); print('copied from', cand, '->', dest); break
    else:
        missing.append(sp)
if missing and tar_path.exists():                  # one streaming pass over the archive for everything still missing
    todo = set(missing)
    with tarfile.open(tar_path, 'r|*') as tf:
        for m in tf:
            parts = m.name.split('/')
            if m.isfile() and not m.name.startswith('/') and '..' not in parts and parts[-1] == 'metadata.pt' and len(parts) >= 2 and parts[-2] in todo:
                dest = Path(DATASET_DIR) / 'cloth_sim' / parts[-2] / 'metadata.pt'
                with tf.extractfile(m) as fi, open(dest, 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
                print('extracted', m.name, 'from', tar_path, '->', dest)
                todo.discard(parts[-2])
                if not todo:
                    break
    missing = sorted(todo)
if missing:
    raise SystemExit('metadata.pt not found for splits %s. Fix DATA_SRC in cell 2 or check %s. Nothing was run.' % (missing, tar_path))
run_stage('data_meta')        # records the metadata.pt field list (results/metadata_fields.json); nothing is read from the archive when the files are present
print('metadata.pt present for all %d splits' % len(splits))
'''

C8 = r'''
# 8. check (inputs, sizes, hashes of small files, metadata.pt preflight, model-load facts), --dry-run of every stage, and the synthetic selftest (CPU).
# CHECK_HASH=True also hashes checkpoints and big npy (slow, optional). The selftest takes several minutes on Colab CPU (추정) and prints SYNTHETIC numbers only; RUN_SELFTEST=False skips it.
CHECK_HASH = False
RUN_SELFTEST = True
STAGES_ALL = ['check', 'data_meta', 'extract', 'pool', 'labels', 'transfer', 'repro', 'analyze_a', 'analyze_b', 'analyze_c', 'analyze_d', 'analyze_e', 'analyze_f', 'aggregate', 'figures']
for st in STAGES_ALL:
    rc = bg('%s run.py %s --dry-run' % (PY, st), EXP + '/logs/colab_dry_%s.log' % st, cwd=EXP, every=5)
    assert rc == 0, 'dry run of %s failed' % st
print('--dry-run OK for', len(STAGES_ALL), 'stages')
run_stage('check', '--hash' if CHECK_HASH else '')
m = json.load(open(WORK_ROOT + '/results/input_manifest.json'))
print('missing input files:', m['missing'], '| metadata.pt ok:', m['metadata_pt_ok'])
assert not m['missing'] and m['metadata_pt_ok'], 'inputs incomplete: fix the Drive paths (cell 2) / metadata.pt (cell 7)'
if RUN_SELFTEST:
    run_stage('selftest', every=30)
'''

C9 = r'''
# 9. SMOKE FIRST: GPU memory and time estimate on one run (a few batches, includes the teacher-forced pass; writes only WORK_ROOT/results/smoke_extract.json). Read the printed ESTIMATE before cell 10.
RUN_SMOKE = True
if RUN_SMOKE:
    run_stage('extract', '--smoke', every=10)
    print(open(WORK_ROOT + '/results/smoke_extract.json').read()[:3000])
else:
    print('RUN_SMOKE is False: skipped')
'''

C10 = r'''
# 10. FULL EXTRACTION (GPU). OFF by default. Set RUN_EXTRACT = True after you have read the smoke estimate (README estimate: about 0.35-1.8 GPU hours for the 6 runs, 추정).
# Writes extract/<run>/{rollout_pooled, teacher_forced_pooled, untrained_*}.npy, meta.json, DONE. Resumable: finished runs are skipped
# (a DONE run without teacher_forced_pooled.npy is redone). A stage failing 3 times stops. Use --fresh in EXTRA to recompute everything.
RUN_EXTRACT = False
EXTRA = ''
if RUN_EXTRACT:
    run_stage('extract', EXTRA, every=30)
else:
    print('RUN_EXTRACT is False: skipped')
'''

C11 = r'''
# 11. CPU analysis. Stages that need NO extraction first (pool, A from the stored 002 predictions, E). After cell 10 finished set AFTER_EXTRACT = True for the rest (labels, transfer, repro, A reproduced, B, C, D, F).
AFTER_EXTRACT = False
for st in ['pool', 'analyze_a', 'analyze_e']:
    run_stage(st)
if AFTER_EXTRACT:
    for st in ['labels', 'transfer', 'repro', 'analyze_a', 'analyze_b', 'analyze_c', 'analyze_d', 'analyze_f']:
        run_stage(st)
else:
    print('AFTER_EXTRACT is False: transfer / repro / B / C / D / F skipped (they need the extracted latents)')
'''

C12 = r'''
# 12. results.json (experiment folder AND WORK_ROOT/results/results.json) + figures (analysis/*.png, needs matplotlib), copied to Drive.
run_stage('aggregate')
run_stage('figures')
os.makedirs(WORK_ROOT + '/figures', exist_ok=True)
import glob
for f in glob.glob(EXP + '/analysis/fig*.png'):
    shutil.copyfile(f, WORK_ROOT + '/figures/' + os.path.basename(f))
r = json.load(open(EXP + '/results.json'))
print({k: r[k] for k in ('stages_present', 'stages_missing', 'code_version')})
print('F present:', r.get('F') is not None, '| teacher forcing steps available:', (r.get('F') or {}).get('teacher_forcing_steps_available'))
print(sorted(os.listdir(WORK_ROOT + '/figures')))
'''

C13 = r'''
# 13. After everything is finished: flush Drive writes.
from google.colab import drive
drive.flush_and_unmount()
'''


def main():
    cfg_txt = (EXP / "config.yaml").read_text()
    expected = re.search(r'config002_sha256:\s*"([0-9a-f]{64})"', cfg_txt).group(1)
    stamp = time.strftime("%F %T")
    srcs = [C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13]
    cells = []
    for i, s in enumerate(srcs):
        s = s.replace("@@CFG002SHA@@", expected).replace("@@STAMP@@", stamp)
        assert "@@" not in s.replace("'@@none@@'", ""), "placeholder left"
        cells.append(md(s) if i == 0 else code(s))
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuClass": "premium", "provenance": []},
                                      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(cells):
        c["id"] = "exp006-%02d" % i
    out = HERE / "exp006_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out, "cells", len(cells), "bytes", out.stat().st_size)


if __name__ == "__main__":
    main()
