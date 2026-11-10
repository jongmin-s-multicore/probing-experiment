#!/usr/bin/env python3
"""Regenerate colab/exp006_colab.ipynb. The notebook embeds a SNAPSHOT (tar+gzip+base64) of this experiment's code AND of the 002 files it imports
(002 src/, config.yaml, colab/requirements_colab.txt, results.json), because the 006 and 002 folders may not be on GitHub. Nothing of 002/003 is modified
(the files are only read here). Run again after ANY code change:   python3 colab/build_notebook.py
Embedded 006: src run.py config.yaml README.md      Embedded 002 (read only copy): src config.yaml colab/requirements_colab.txt results.json
(not: __pycache__, logs, results, *.pyc)."""
import base64, hashlib, io, json, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
E002 = EXP.parent / "002_single_size_sweep"
TOP6, TOP2 = EXP.name, E002.name
FILES6 = ["src", "run.py", "config.yaml", "README.md"]
FILES2 = ["src", "config.yaml", "colab/requirements_colab.txt", "results.json"]


def _filter(ti):
    n = ti.name
    if "__pycache__" in n.split("/") or n.endswith(".pyc"):
        return None
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mtime = 0
    return ti


def make_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
        for f in FILES6:
            tf.add(str(EXP / f), arcname=TOP6 + "/" + f, filter=_filter)
        for f in FILES2:
            tf.add(str(E002 / f), arcname=TOP2 + "/" + f, filter=_filter)
    raw = buf.getvalue()
    b = base64.b64encode(raw).decode()
    return "\n".join(b[i:i + 100] for i in range(0, len(b), 100)), len(raw)


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


C0 = r'''
# Experiment 006: re-analysis of the 002 single-size sweep (exploratory, post hoc) on Colab
Run the cells top to bottom. Cells 1-8 are idempotent. **This notebook only READS `MyDrive/exp002_work` and WRITES `MyDrive/exp006_work`.** Use its own runtime; do not touch the 002 or 003 runtimes/notebooks (003 trains in another runtime).

What it does (nothing is trained): loads the 6 finished 002 checkpoints (2 conditions x 3 training seeds), recomputes rollouts (trained and untrained predictors) from the 002 cached frames, and computes analyses A-E of `notes/hypotheses.md` candidate J. No thresholds, no pass/fail, no p values; `interpretation` in results.json is null. Numbers are interpreted later by /review-experiment.

**Runtime type is your choice** (GPU needed only for cells 9-10). Estimates (추정, from nothing measured yet) are printed by the smoke in cell 9. Cell 10 (full extraction) is OFF until you set `RUN_EXTRACT = True` after reading the smoke estimate (CLAUDE.md: confirm GPU use first).

Persistent output: `WORK_ROOT = MyDrive/exp006_work` (`extract/`, `cache/`, `results/` including `results.json`, `logs/`). `/content` is rebuilt every session. A stage that fails 3 times stops (CLAUDE.md retry limit); the cell shows the reason.
Code delivery: the snapshot embedded in cell 3 (006 code + the 002 source files it imports, read only copies). The 002 checkpoints were saved by the patched `train.py`, but loading needs only the pinned commit's model classes, not the patch (see README section 4; `run.py check` records the facts).
Everything here is UNVERIFIED on Colab as a whole; the CPU selftest ran on Python 3.9 with synthetic data.
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
'''

C2 = r'''
# 2. Mount Drive. INPUT_ROOT is READ ONLY by construction of the code (run.py refuses to write there); WORK_ROOT receives all outputs.
import os
from google.colab import drive
drive.mount('/content/drive')
INPUT_ROOT = os.environ.get('EXP006_INPUT_ROOT', '/content/drive/MyDrive/exp002_work')
WORK_ROOT = os.environ.get('EXP006_WORK_ROOT', '/content/drive/MyDrive/exp006_work')
assert os.path.isdir(INPUT_ROOT), 'INPUT_ROOT %s not found: is the 002 Drive folder where you expect it?' % INPUT_ROOT
assert os.path.realpath(INPUT_ROOT) != os.path.realpath(WORK_ROOT) and not os.path.realpath(WORK_ROOT).startswith(os.path.realpath(INPUT_ROOT) + os.sep), 'WORK_ROOT must differ from INPUT_ROOT'
os.makedirs(WORK_ROOT, exist_ok=True)
probe = os.path.join(WORK_ROOT, '.write_test_notebook')
with open(probe, 'w') as f:                       # a Drive write failure must stop here, not later
    f.write('ok'); f.flush(); os.fsync(f.fileno())
assert open(probe).read() == 'ok', 'Drive write/read-back failed'
print('INPUT_ROOT (read only) =', INPUT_ROOT)
print('WORK_ROOT  (outputs)   =', WORK_ROOT, '| write test OK')
for n in ('train_runs/outputs', 'exp002_cache/q1_tokens', 'results', 'cloth_sim_data.tar'):
    print('  %-28s exists: %s' % (n, os.path.exists(os.path.join(INPUT_ROOT, n))))
'''

C3 = r'''
# 3. Unpack the code snapshot to /content/mylab/experiments/ (006 code + read-only copies of the 002 files it imports). Not Drive. Re-run after a runtime reset.
import base64, hashlib, io, os, tarfile
TOP6, TOP2 = '@@TOP6@@', '@@TOP2@@'
REPO = '/content/mylab'
EXP = REPO + '/experiments/' + TOP6
E002 = REPO + '/experiments/' + TOP2
EMBEDDED = """
@@B64@@
"""
# snapshot built @@STAMP@@ ; re-run `python3 colab/build_notebook.py` after any code change and re-upload this notebook
src = base64.b64decode(''.join(EMBEDDED.split()))
print('embedded snapshot', len(src), 'bytes')
os.makedirs(REPO + '/experiments', exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(src), mode='r:*') as tf:
    for m in tf.getmembers():
        parts = m.name.split('/')
        assert not m.name.startswith('/') and '..' not in parts and parts[0] in (TOP6, TOP2) and (m.isfile() or m.isdir()), 'refused tar member: ' + m.name
    tf.extractall(REPO + '/experiments')
assert os.path.isfile(EXP + '/run.py') and os.path.isfile(E002 + '/src/exp002/model_io.py')
sha = hashlib.sha256(open(E002 + '/config.yaml', 'rb').read()).hexdigest()
assert sha == '@@CFG002_SHA@@', '002 config.yaml sha256 differs from the pinned value: ' + sha
print('code in', EXP, '| 002 config sha256 OK')
'''

C4 = r'''
# 4. Environment: uv venv, Python 3.9, the 002 requirements (colab/requirements_colab.txt of 002, unchanged) + matplotlib (figures only).
import hashlib, os
VENV = '/content/venv'
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
# 5. ACWM-Phys checkout at the pinned commit (== 002 config.yaml code.commit). NO resume patch: loading a checkpoint needs the model classes only (README section 4).
import subprocess
COMMIT = '5a640bed66002eca7bdb31d10b2bde59ad944551'
SRC = '/content/dinowm/acwm_phys_src'
if not os.path.isdir(SRC + '/.git'):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/jongmin-s-multicore/thread0.git', SRC], check=True)
subprocess.run(['git', '-C', SRC, 'checkout', '--detach', COMMIT], check=True)
head = subprocess.run(['git', '-C', SRC, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
assert head == COMMIT, head
print('ACWM-Phys HEAD', head, '| (the 002 checkpoints were written by train.py + 002 train_resume.patch sha256 52eb0052...; not needed to load)')
'''

C6 = r'''
# 6. Environment variables and Drive-backed logs
import os, shutil
os.environ.update({
    'EXP006_INPUT_ROOT': INPUT_ROOT, 'EXP006_WORK_ROOT': WORK_ROOT,
    'EXP006_EXP002_DIR': E002,                       # local read-only copy of the 002 source files
    'ACWM_PHYS_SRC': SRC, 'DATASET_DIR': '/content/dinowm/data', 'WANDB_MODE': 'disabled',
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
    assert rc == 0, 'stage %s failed or was interrupted (rc=%s). A stage stops after 3 failed attempts; read the lines above.' % (stage, rc)
    return rc
'''

C7 = r'''
# 7. metadata.pt of the training splits from the 002 archive (read only; action normalisation statistics). Streams the 1.45 GB tar once (a few minutes).
run_stage('data_meta')
'''

C8 = r'''
# 8. check (inputs, sizes, hashes of small files, model-load facts) and the synthetic selftest. CHECK_HASH=True also hashes checkpoints and big npy (slow, optional).
CHECK_HASH = False
run_stage('check', '--hash' if CHECK_HASH else '')
run_stage('selftest')
'''

C9 = r'''
# 9. SMOKE FIRST: GPU memory and time estimate on one run (a few batches; writes only WORK_ROOT/results/smoke_extract.json). Read the printed ESTIMATE before cell 10.
RUN_SMOKE = True
if RUN_SMOKE:
    run_stage('extract', '--smoke', every=10)
    print(open(WORK_ROOT + '/results/smoke_extract.json').read()[:3000])
else:
    print('RUN_SMOKE is False: skipped')
'''

C10 = r'''
# 10. FULL EXTRACTION (GPU). OFF by default. Set RUN_EXTRACT = True after you have read the smoke estimate and accepted the GPU use.
# Resumable: finished runs have extract/<run>/DONE and are skipped. A stage failing 3 times stops.
RUN_EXTRACT = False
if RUN_EXTRACT:
    run_stage('extract', every=30)
else:
    print('RUN_EXTRACT is False: skipped')
'''

C11 = r'''
# 11. CPU analysis. Stages that need NO extraction first (pool, A from stored 002 predictions, E). Then, after cell 10 finished, the rest.
AFTER_EXTRACT = False          # set True after the extraction finished
for st in ['pool', 'analyze_a', 'analyze_e']:
    run_stage(st)
if AFTER_EXTRACT:
    for st in ['labels', 'transfer', 'repro', 'analyze_a', 'analyze_b', 'analyze_c', 'analyze_d']:
        run_stage(st)
else:
    print('AFTER_EXTRACT is False: transfer / repro / B / C / D skipped (they need the extracted rollouts)')
'''

C12 = r'''
# 12. results.json (experiment folder AND WORK_ROOT/results/results.json) + figures (analysis/*.png, needs matplotlib)
run_stage('aggregate')
run_stage('figures')
print(open(EXP + '/results.json').read()[:1500])
'''

C13 = r'''
# 13. After everything is finished: flush Drive writes.
from google.colab import drive
drive.flush_and_unmount()
'''


def main():
    b64, n = make_b64()
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    cfg_sha = hashlib.sha256((E002 / "config.yaml").read_bytes()).hexdigest()
    c3 = C3.replace("@@B64@@", b64).replace("@@TOP6@@", TOP6).replace("@@TOP2@@", TOP2).replace("@@STAMP@@", stamp).replace("@@CFG002_SHA@@", cfg_sha)
    cells = [md(C0), code(C1), code(C2), code(c3), code(C4), code(C5), code(C6), code(C7), code(C8), code(C9), code(C10), code(C11), code(C12), code(C13)]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuClass": "premium", "provenance": []},
                                      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(cells):
        c["id"] = "exp006-%02d" % i
    out = HERE / "exp006_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out, "| embedded tar.gz bytes:", n)


if __name__ == "__main__":
    main()
