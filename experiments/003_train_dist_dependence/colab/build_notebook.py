#!/usr/bin/env python3
"""Regenerate colab/exp003_colab_resume.ipynb (2026-11-05; the older colab/exp003_colab.ipynb holds uncommitted run output and is NEVER written by this script). The notebook embeds a SNAPSHOT (tar+gzip+base64) of this experiment's code, because the 003 folder is not on GitHub.
Run again after ANY code change:   python3 colab/build_notebook.py
Embedded: src/ run.py config.yaml run_all.sh setup.sh README.md CHANGES.md colab/requirements_colab.txt   (not: __pycache__, logs, results, manifests, *.pyc)
"""
import base64, io, json, os, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
TOP = EXP.name                      # 003_train_dist_dependence
FILES = ["src", "run.py", "config.yaml", "run_all.sh", "setup.sh", "README.md", "CHANGES.md", "colab/requirements_colab.txt", "colab/train_resume.patch"]


def _filter(ti):
    n = ti.name
    if "__pycache__" in n.split("/") or n.endswith(".pyc"):
        return None
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mtime = 0                    # reproducible bytes
    return ti


def make_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
        for f in FILES:
            tf.add(str(EXP / f), arcname=TOP + "/" + f, filter=_filter)
    raw = buf.getvalue()
    b = base64.b64encode(raw).decode()
    return "\n".join(b[i:i + 100] for i in range(0, len(b), 100)), len(raw)


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


C0 = r'''
# Experiment 003, remaining 9 models (single_a026, single_a248, mix_adjacent x seeds 0,1,2) on Google Colab (resume version, 2026-11-05)
Run the cells top to bottom. Cells 1-9 are idempotent: after a disconnect choose a GPU runtime and run them again. **Do not touch the 002 runtime or notebook; open this notebook in its own runtime.**

**Resume:** cell 5 applies `colab/train_resume.patch` (copied unchanged from experiment 002) to the pinned `train.py`. With it `training.epochs=15` is the TOTAL target epoch: a run interrupted after epoch 5 continues with epochs 6..15 (encoders and optimizer states restored), a fresh run trains 1..15. `save_every_x_epoch` is 1 (operational value, not pre-registered), so at most one epoch is lost per disconnect.
Before every train start `run.py` checks `train.py`: if it is not patched it REFUSES to train (rc 3). Then it keeps a readable `model_latest.pth` (resume), replaces a broken one by the newest readable `model_<e>.pth`, or MOVES unusable checkpoints to `WORK_ROOT/train_runs/quarantine/` (never deleted) and starts from epoch 0. Each start is recorded in `train_runs/outputs/<run>/train_env.json` (training GPU, resume yes/no, patch sha256) and copied into `results.json` (`runs.<run>.train_env`).
The three finished `mix_wide` runs have `DONE` and are skipped; they are never retrained, re-extracted, pruned or moved (they were trained WITHOUT the patch, with save_every 5, on an A100; their train_env is reported as "unknown").
A resumed run is not bit-identical to an uninterrupted one (data order and RNG state are not restored; estimate, not verified on Colab).

**Human approval (CLAUDE.md):** the user approved GPU training of the 9 models on 2026-11-05 and the current Colab L4 runtime; cell 11 holds `EXP003_APPROVED` accordingly. The smoke gate uses the existing `results/smoke.json` (measured on an A100). **Memory on an L4 (24 GB) has not been measured**: cell 11 prints GPU memory and the time since start; watch the first epoch and stop if it runs out of memory. Do not change batch_size (pre-registered 24).

Session tips (Colab behaviour from memory, 추정, not checked today)
- Colab may reclaim an idle runtime and has a maximum session length. Keep the tab open.
- Everything that must survive lives on Drive in `MyDrive/exp003_work`: `train_runs/`, `exp003_cache/`, `results/` (including `results.json` copied by aggregate), `logs/`, `manifests/`. `/content` is rebuilt every session.
- Code delivery: the 003 folder is NOT on GitHub. Cell 3 unpacks the snapshot embedded in this notebook. A `MyDrive/exp003_work/exp003_code.tar` from an earlier session is IGNORED (it would hide the new code) unless you set `USE_DRIVE_TAR = True` in cell 3.
- Everything in this notebook is UNVERIFIED on Colab as a whole.
'''

C1 = r'''
# 1. GPU / Python check (+ helpers used by later cells)
import sys, subprocess, os, time
def sh(cmd, cwd=None):
    # run a shell command and stream its output into the cell
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='')
    return p.wait()

def bg(cmd, log, cwd=None, env=None, every=30):
    # run cmd with nohup in its own session, output to `log`; tail the log until it ends. Returns the exit code.
    # Interrupting the cell leaves the job running (re-run the cell: the commands used here skip finished work).
    os.makedirs(os.path.dirname(log), exist_ok=True)
    rcfile = log + '.rc'
    if os.path.exists(rcfile):
        os.remove(rcfile)
    full = '(%s) >> %s 2>&1; echo $? > %s' % (cmd, log, rcfile)
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
    print('\n'.join(lines[shown:][-30:]))
    return int(open(rcfile).read().strip() or 1)

sh('nvidia-smi')
print(sys.version)
print(subprocess.run("df -h /content | tail -1", shell=True, capture_output=True, text=True).stdout)
'''

C2 = r'''
# 2. Mount Drive and choose the persistent work root
import os
from google.colab import drive
drive.mount('/content/drive')
WORK_ROOT = os.environ.get('EXP003_WORK_ROOT', '/content/drive/MyDrive/exp003_work')
os.makedirs(WORK_ROOT, exist_ok=True)
# Data source folder (names contain brackets and a space: only passed through os.environ / pathlib, never through a shell command line).
DATA_SRC = os.environ.get('EXP003_DRIVE_MOUNT_SRC', '/content/drive/MyDrive/dino-wm-videos/[data]/[cloth-sim] cloth_move')
print('WORK_ROOT =', WORK_ROOT)
print('archive cloth_sim_data.tar on Drive:', os.path.exists(WORK_ROOT + '/cloth_sim_data.tar'))
print('mount source exists:', os.path.isdir(DATA_SRC))
'''

C3 = r'''
# 3. Unpack the 003 code to /content/mylab/experiments/003_train_dist_dependence
# The embedded snapshot is used. A WORK_ROOT/exp003_code.tar from an earlier session is IGNORED (it can be older than this notebook and would hide the resume code),
# unless USE_DRIVE_TAR = True below.
# The embedded snapshot was generated at notebook build time (@@STAMP@@); when the code changes, re-run `python3 colab/build_notebook.py` and re-upload the notebook.
# Only code files are overwritten; results/ logs/ manifests/ (Drive symlinks, cell 6) are untouched.
import base64, io, os, tarfile
TOP = '@@TOP@@'
REPO = '/content/mylab'
EXP = REPO + '/experiments/' + TOP
EMBEDDED = """
@@B64@@
"""
USE_DRIVE_TAR = False
ext_tar = WORK_ROOT + '/exp003_code.tar'
if os.path.exists(ext_tar) and not USE_DRIVE_TAR:
    print('WARNING: %s exists but is IGNORED (USE_DRIVE_TAR=False); the embedded snapshot is used.' % ext_tar)
if os.path.exists(ext_tar) and USE_DRIVE_TAR:
    src = open(ext_tar, 'rb').read()
    print('using', ext_tar, len(src), 'bytes (USE_DRIVE_TAR=True: make sure it contains src/exp003/resume.py and colab/train_resume.patch)')
else:
    src = base64.b64decode(''.join(EMBEDDED.split()))
    print('using the embedded snapshot', len(src), 'bytes')
os.makedirs(os.path.dirname(EXP), exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(src), mode='r:*') as tf:
    for m in tf.getmembers():
        parts = m.name.split('/')
        assert not m.name.startswith('/') and '..' not in parts and parts[0] == TOP and (m.isfile() or m.isdir()), 'refused tar member: ' + m.name
    tf.extractall(os.path.dirname(EXP))
os.chmod(EXP + '/run_all.sh', 0o755)
assert os.path.isfile(EXP + '/run.py') and os.path.isfile(EXP + '/colab/requirements_colab.txt')
assert os.path.isfile(EXP + '/src/exp003/resume.py') and os.path.isfile(EXP + '/colab/train_resume.patch'), 'code snapshot does not contain the resume code'
print('code in', EXP)
'''

C4 = r'''
# 4. Environment: uv venv with Python 3.9 (no conda on Colab; documented exception to CLAUDE.md). Rebuilt every new session (a few minutes).
# Verified in an earlier session: after install, import pkg_resources,torch,numpy,cv2,decord,hydra passes; torch 2.0.1+cu117, cuda True.
import hashlib, os
VENV = '/content/venv'
PY = VENV + '/bin/python'
MARK = VENV + '/.exp003_ok_' + hashlib.sha256(open(EXP + '/colab/requirements_colab.txt', 'rb').read()).hexdigest()[:12]
if not os.path.exists(MARK):
    assert sh('pip install -q uv') == 0
    if not os.path.exists(PY):
        assert sh('uv venv --python 3.9 %s' % VENV) == 0
    assert sh('uv pip install --python %s --index-strategy unsafe-best-match -r %s/colab/requirements_colab.txt' % (PY, EXP)) == 0
    open(MARK, 'w').write('ok\n')
CHK = "import sys,pkg_resources,torch,numpy,cv2,decord,hydra;print(sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available(),'numpy',numpy.__version__)"
assert sh('%s -c "%s"' % (PY, CHK)) == 0, 'environment check failed'
'''

C5 = r'''
# 5. ACWM-Phys checkout at the pinned commit (== config.yaml code.commit) + resume patch on train.py only (colab/train_resume.patch, copied unchanged from 002).
import hashlib, os, subprocess
COMMIT = '5a640bed66002eca7bdb31d10b2bde59ad944551'
SRC = '/content/dinowm/acwm_phys_src'
if not os.path.isdir(SRC + '/.git'):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/jongmin-s-multicore/thread0.git', SRC], check=True)
subprocess.run(['git', '-C', SRC, 'checkout', '--detach', COMMIT], check=False)      # no-op when already there; a patched train.py blocks a switch, which is fine
head = subprocess.run(['git', '-C', SRC, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
assert head == COMMIT, head
PATCH = EXP + '/colab/train_resume.patch'
if subprocess.run(['git', '-C', SRC, 'apply', '--reverse', '--check', PATCH], capture_output=True).returncode == 0:
    print('resume patch already applied')
else:
    subprocess.run(['git', '-C', SRC, 'apply', '--check', PATCH], check=True)
    subprocess.run(['git', '-C', SRC, 'apply', PATCH], check=True)
    print('resume patch applied')
dirty = subprocess.run(['git', '-C', SRC, 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True).stdout.splitlines()
assert [d.strip() for d in dirty if d.strip()] == ['M train.py'], dirty      # only train.py may differ from the pinned commit
text = open(SRC + '/train.py').read()
assert '[exp002 operational]' in text and 'range(init_epoch, self.total_epochs + 1)' in text, 'patch marker / total-epoch loop missing in train.py'
sha = hashlib.sha256(open(PATCH, 'rb').read()).hexdigest()
open(WORK_ROOT + '/acwm_resume_patch.sha256', 'w').write(sha + '\n')
print('commit', head, '| patch sha256', sha, '| marker present, loop = range(init_epoch, self.total_epochs + 1)')
'''

C6 = r'''
# 6. Environment variables + Drive-backed results/logs/manifests (symlinks into WORK_ROOT)
import os, shutil
os.environ.update({
    'DINO_WORK': '/content/dinowm',
    'DATASET_DIR': '/content/dinowm/data',                 # local copy (fast reads)
    'DINO_TRAIN': WORK_ROOT + '/train_runs',               # Drive: checkpoints, DONE markers
    'EXP003_CACHE': WORK_ROOT + '/exp003_cache',           # Drive
    'ACWM_PHYS_SRC': SRC,
    'WANDB_MODE': 'disabled',
    'EXP003_WORK_ROOT': WORK_ROOT,                         # archive lookup/creation: WORK_ROOT/cloth_sim_data.tar
    'EXP003_DATA_MODE': 'mount',
    'CONDA_ENV': 'none',                                   # run_all.sh: no conda on Colab
    'PATH': os.path.dirname(PY) + os.pathsep + os.environ['PATH'],
    'VIRTUAL_ENV': VENV,
})
# NOTE: PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set here. run.py adds expandable_segments only for torch >= 2.1 and respects a value set by you.
if DATA_SRC:
    os.environ['EXP003_DRIVE_MOUNT_SRC'] = DATA_SRC
for d in ('train_runs', 'exp003_cache'):
    os.makedirs(WORK_ROOT + '/' + d, exist_ok=True)
for d in ('results', 'logs', 'manifests'):
    tgt = WORK_ROOT + '/' + d
    lnk = EXP + '/' + d
    os.makedirs(tgt, exist_ok=True)
    if os.path.islink(lnk):
        if os.path.realpath(lnk) != os.path.realpath(tgt):
            os.remove(lnk)
    if not os.path.islink(lnk):
        if os.path.isdir(lnk):                              # real folder: merge into Drive (never overwrite), then replace by a link
            shutil.copytree(lnk, tgt, dirs_exist_ok=True)
            shutil.rmtree(lnk)
        os.symlink(tgt, lnk)
    print(lnk, '->', os.path.realpath(lnk))
'''

C7 = r'''
# 7. Data -> /content/dinowm/data (command: python run.py download). Idempotent: complete splits are skipped.
#    a) WORK_ROOT/cloth_sim_data.tar (1.45 GB, 16 splits) exists -> extract it locally.
#    b) otherwise parallel copy from the mounted Drive folder (about 10 min; 0 warnings in an earlier check), then the archive is written once to WORK_ROOT.
print(subprocess.run('df -h /content | tail -1', shell=True, capture_output=True, text=True).stdout)
rc = bg('%s run.py download' % PY, EXP + '/logs/colab_download.log', cwd=EXP)
assert rc == 0, 'download failed or cell interrupted (rc=%s); read the lines above, then re-run this cell' % rc
'''

C8 = r'''
# 8. check / selftest / maskcheck / listing (no GPU training; each stage is idempotent)
for stage in ('check', 'selftest', 'maskcheck', 'listing'):
    print('=== run.py %s' % stage)
    rc = bg('%s run.py %s' % (PY, stage), EXP + '/logs/colab_%s.log' % stage, cwd=EXP, every=15)
    assert rc == 0, 'run.py %s failed (rc=%s); see %s/logs' % (stage, rc, EXP)
'''

C9 = r'''
# 9b. SMOKE (GPU run: 12 GB memory/time measurement). Needs human approval (CLAUDE.md): set RUN_SMOKE = True only after approval.
RUN_SMOKE = False
if RUN_SMOKE:
    if os.path.exists(EXP + '/results/smoke.json'):
        print('results/smoke.json exists; showing it (delete it on Drive to repeat the smoke)')
        print(open(EXP + '/results/smoke.json').read()[:3000])
    else:
        rc = bg('%s run.py smoke' % PY, EXP + '/logs/colab_smoke.log', cwd=EXP, every=30)
        print('smoke rc =', rc)
else:
    print('RUN_SMOKE is False: skipped')
'''

C9B = r'''
# 9. 001 baseline results.json for the H-H2 verdict (12237220 bytes). The Colab snapshot has no experiments/001: find the file on Drive and copy it to
#    EXP/../001_size_collapse_diagnosis/results.json. FAILS (assert) if it is not found or does not contain the baseline runs. Do this BEFORE training.
#    If it is not under MyDrive, set os.environ['EXP003_001_SEARCH'] to a folder (or copy the file there by hand) and re-run.
rc = bg('%s run.py baseline' % PY, EXP + '/logs/colab_baseline.log', cwd=EXP, every=15)
print(open(EXP + '/results/baseline_001_install.json').read() if os.path.exists(EXP + '/results/baseline_001_install.json') else '(no install record)')
assert rc == 0, '001 results.json not found/readable (rc=%s): see above. Aggregate would give H-H2 not_evaluable.' % rc
'''

C10 = r'''
# 10. FULL TRAINING (GPU). Approved by the user on 2026-11-05 for single_a026 single_a248 mix_adjacent (mix_wide is listed but has DONE and is skipped).
# Resume behaviour: run_all.sh skips every run that has DONE. After a disconnect re-run cells 1-9 and this cell: a run without DONE continues from its last readable
# checkpoint with the patched train.py (epochs k+1..15), or restarts at epoch 0 if no readable checkpoint exists (old ones are moved to train_runs/quarantine).
# Stages after train (extract/metrics) run per model; baseline + aggregate at the end; aggregate copies results.json to WORK_ROOT/results/.
# Unmeasured on this GPU: time per epoch and memory on an L4. The loop below prints nvidia-smi memory and, per unfinished run, the epochs written so far and minutes since its start.
RUN_TRAIN = True
EXP003_APPROVED = 'single_a026 single_a248 mix_adjacent mix_wide'
if RUN_TRAIN:
    approved = EXP003_APPROVED.split()
    assert approved, 'EXP003_APPROVED is empty: write the approved distribution names yourself. Refusing to train.'
    assert os.path.exists(EXP + '/results/smoke.json') and '"status": "finished"' in open(EXP + '/results/smoke.json').read(), \
        'results/smoke.json missing or not finished: run the smoke (cell 9) first'
    import json, subprocess, glob
    env = dict(os.environ, EXP003_APPROVED=' '.join(approved))
    models = subprocess.run([PY, '-c', "import sys;sys.path.insert(0,'src');from exp003.common import load_config,all_models;print(' '.join(all_models(load_config())))"],
                            cwd=EXP, env=env, capture_output=True, text=True, check=True).stdout.split()
    env['MODELS'] = ' '.join(m for m in models if m.split(':')[0] in approved)         # only approved distributions (run_all.sh stops at a non-approved one)
    print('models to run:', env['MODELS'])
    assert env['MODELS'], 'none of the approved names match config distributions: ' + ' '.join(models)
    LOG = EXP + '/logs/train_all.log'
    def peek():
        gpu = subprocess.run('nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader', shell=True, capture_output=True, text=True).stdout.strip()
        rows = []
        for d in sorted(glob.glob(WORK_ROOT + '/train_runs/outputs/*')):
            if os.path.exists(d + '/DONE') or d.endswith('_smoke') or not os.path.exists(d + '/train_env.json'):
                continue
            try:
                a = json.load(open(d + '/train_env.json'))['attempts'][-1]
                import datetime
                mins = (datetime.datetime.now() - datetime.datetime.strptime(a['start_time'], '%Y-%m-%d %H:%M:%S')).total_seconds() / 60
                n = len(open(d + '/epoch_logs.jsonl').read().splitlines()) if os.path.exists(d + '/epoch_logs.jsonl') else 0
                rows.append('%s: action=%s epochs_logged_total=%d minutes_since_start=%.0f' % (os.path.basename(d), a['action'], n, mins))
            except Exception as e:
                rows.append('%s: (cannot read: %s)' % (os.path.basename(d), e))
        print('GPU[name, mem used, mem total, util]:', gpu, '\n  ' + '\n  '.join(rows), flush=True)
    running = subprocess.run('pgrep -f "[r]un_all.sh"', shell=True, capture_output=True).returncode == 0
    if running:
        print('run_all.sh is already running in this session; attaching to its log')
    else:
        p = subprocess.Popen('bash run_all.sh --from train >> %s 2>&1' % LOG, shell=True, cwd=EXP, env=env, start_new_session=True)
        print('started run_all.sh pid', p.pid)
    try:
        while subprocess.run('pgrep -f "[r]un_all.sh"', shell=True, capture_output=True).returncode == 0:
            print(subprocess.run(['tail', '-n', '3', LOG], capture_output=True, text=True).stdout, flush=True)
            peek()
            time.sleep(120)          # keeps the cell busy; interrupting the cell does not stop the background job
    except KeyboardInterrupt:
        print('cell interrupted; the background job keeps running')
    print(subprocess.run(['tail', '-n', '20', LOG], capture_output=True, text=True).stdout)
    print('results.json on Drive:', WORK_ROOT + '/results/results.json', os.path.exists(WORK_ROOT + '/results/results.json'))
else:
    print('RUN_TRAIN is False: skipped')
'''

C11 = r'''
# 11. After everything is finished (or before you leave): flush Drive writes.
from google.colab import drive
drive.flush_and_unmount()
'''

C12 = r'''
# Status monitor (last cell). One block every 3 min, also appended to Drive (exp003_work/heartbeat.log). Interrupting it does not stop training.
def monitor(every=180):
    import glob, json, os, subprocess, time
    mount = '/content/drive'
    work = mount + '/MyDrive/exp003_work'

    def out(cmd):
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    def alive(pattern):
        return 'ON' if out('pgrep -f "%s"' % pattern) else 'off'

    def runs():
        rows = []
        for d in sorted(glob.glob(work + '/train_runs/outputs/*')):
            name = os.path.basename(d)
            if os.path.exists(d + '/DONE'):
                rows.append(name + ' DONE')
                continue
            p = d + '/epoch_logs.jsonl'      # written by the pinned train.py (name as in 002; absent -> "no epoch yet")
            try:
                with open(p) as f:
                    last = json.loads(f.read().strip().splitlines()[-1])
                rows.append('%s e%s val_loss=%s (saved %d min ago)' % (name, last.get('epoch'), last.get('val_loss'), (time.time() - os.path.getmtime(p)) / 60))
            except (OSError, IndexError, ValueError):
                rows.append(name + ' started, no epoch finished yet')
        return ' | '.join(rows) or '(no run folder visible)'

    def last_line():
        try:
            lines = open(work + '/logs/train_all.log', errors='replace').read().splitlines()
            return lines[-1][:200] if lines else '(empty log)'
        except OSError as e:
            return '(cannot read train_all.log: %s)' % e.strerror

    def status():
        gpu = out('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader') or 'n/a'
        text = '[%s UTC] run_all=%s train=%s gpu=%s\n    %s\n    %s' % (
            time.strftime('%H:%M', time.gmtime()), alive('[r]un_all.sh'), alive('[t]rain.py --config-name'), gpu, runs(), last_line())
        print(text, flush=True)
        if not os.path.ismount(mount):
            print('    !!! DRIVE IS NOT MOUNTED: training cannot save. Do not wait, paste this output.', flush=True)
            return
        try:
            with open(work + '/heartbeat.log', 'a') as f:
                f.write(text + '\n')
        except OSError as e:
            print('    heartbeat not written: %s' % e.strerror, flush=True)

    try:
        while alive('[r]un_all.sh') == 'ON':
            status()
            time.sleep(every)
        status()
        print('run_all.sh is NOT running: finished or stopped. Read the last line above.')
    except KeyboardInterrupt:
        print('monitor stopped; training keeps running')

monitor()
'''


def main():
    b64, n = make_b64()
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    c3 = C3.replace("@@B64@@", b64).replace("@@TOP@@", TOP).replace("@@STAMP@@", stamp)
    cells = [md(C0), code(C1), code(C2), code(c3), code(C4), code(C5), code(C6), code(C7), code(C8), code(C9B), code(C9), code(C10), code(C11), code(C12)]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuClass": "premium", "provenance": []},
                                      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(cells):                      # nbformat >= 4.5 requires cell ids
        c["id"] = "exp003r-%02d" % i
    out = HERE / "exp003_colab_resume.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out, "| embedded tar.gz bytes:", n)


if __name__ == "__main__":
    main()
