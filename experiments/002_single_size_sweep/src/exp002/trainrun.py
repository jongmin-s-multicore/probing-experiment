"""Training launcher: calls the branch's train.py WITHOUT modification (as 001 does). Only hydra overrides change the data loader
(exp002.data.load_manifest_slice_train_val) and the run directory. Batch size / num_hist / epochs / budget come from config.yaml and are
null until a human sets them after `python run.py smoke`; null -> ConfigNull (clear message), nothing is guessed."""
from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from .common import log, require, read_json, write_json, run_dir, run_name as mk_run_name

SRC = Path(__file__).resolve().parents[1]


def measure_steps_per_epoch(cfg, paths, cond):
    """steps_per_epoch of a condition from its manifest and the branch loader (metadata only). Needs torch + the branch checkout; UNVERIFIED here."""
    from . import budget as B
    from . import data as D
    if str(paths["acwm_src"]) not in sys.path:
        sys.path.insert(0, str(paths["acwm_src"]))
    H, bs = require(cfg, "train.num_hist"), require(cfg, "train.batch_size")
    man = paths["manifests"] / ("train_%s.json" % cond)
    if not man.exists():
        raise FileNotFoundError("%s missing - run `python run.py manifest`" % man)
    ti = cfg["train_impl"]
    ntr, nva = D.count_train_slices(man, int(H), int(ti["num_pred"]), cfg["data_impl"]["frameskip"], list(cfg["data_impl"]["action_dims"]),
                                    val_split_seed=cfg["data_impl"]["val_split_seed"])
    return {"n_train_slices": ntr, "n_valid_slices": nva, "steps_per_epoch": B.steps_per_epoch(ntr, int(bs), ti["steps_per_epoch_rounding"]),
            "steps_per_epoch_ceil": B.steps_per_epoch(ntr, int(bs), "ceil"), "steps_per_epoch_floor": B.steps_per_epoch(ntr, int(bs), "floor")}


def budget_for(cfg, paths, cond):
    """Epochs of `cond` so that every condition gets the same total number of optimizer steps as the reference condition. Returns a dict (json-able)."""
    from . import budget as B
    ep_ref = require(cfg, "train.epochs", "epochs of the budget reference condition %s" % cfg["conditions"]["budget_reference"])
    ref = cfg["conditions"]["budget_reference"]
    bsteps = cfg["train"]["budget"]
    meas = {c: measure_steps_per_epoch(cfg, paths, c) for c in sorted({ref, cond})}
    pl = B.plan({c: m["steps_per_epoch"] for c, m in meas.items()}, int(ep_ref), ref, bsteps)
    out = dict(pl[cond])
    out.update({"condition": cond, "reference_condition": ref, "reference_epochs": int(ep_ref), "train_budget_config": bsteps, "measured": meas,
                "batch_size": int(cfg["train"]["batch_size"]), "num_hist": int(cfg["train"]["num_hist"]), "num_pred_assumed": int(cfg["train_impl"]["num_pred"]),
                "steps_per_epoch_rounding": cfg["train_impl"]["steps_per_epoch_rounding"],
                "steps_note": "steps_computed = epochs x steps_per_epoch is a COMPUTED value (slice count / batch size), not read from the optimizer"})
    return out


def build_cmd(cfg, paths, manifest, seed, rd, smoke=None, epochs=None):
    """smoke: None or dict(num_hist, batch_size, n_rollout). epochs: epochs for this condition from budget_for (required when smoke is None)."""
    ti = cfg["train_impl"]
    if smoke is None:
        H = require(cfg, "train.num_hist")
        bs = require(cfg, "train.batch_size")
        require(cfg, "train.epochs")
        if ti["budget_mode"] != "steps_equalized":
            raise NotImplementedError("only train_impl.budget_mode == 'steps_equalized' is implemented")
        if epochs is None:
            raise ValueError("epochs of this condition must come from budget_for()")
        ov = {"training.epochs": int(epochs), "training.save_every_x_epoch": ti["save_every_x_epoch"]}
    else:
        H, bs = smoke["num_hist"], smoke["batch_size"]
        ov = {"training.epochs": 1, "training.save_every_x_epoch": 1, "env.dataset.n_rollout": smoke["n_rollout"]}
    ov.update({"env": ti["hydra_env"], "frameskip": cfg["data_impl"]["frameskip"], "num_hist": int(H), "training.batch_size": int(bs),
               "training.predictor_lr": ti["predictor_lr"], "training.seed": int(seed)})
    cmd = [sys.executable, "train.py", "--config-name", "train.yaml"] + ["%s=%s" % (k, v) for k, v in ov.items()]
    cmd += ["env.dataset._target_=exp002.data.load_manifest_slice_train_val", "+env.dataset.manifest=%s" % manifest,
            "ckpt_base_path=%s" % paths["train_base"], "hydra.run.dir=%s" % rd]
    env = dict(os.environ)
    env.update({"DATASET_DIR": str(paths["dataset_dir"]), "WANDB_MODE": env.get("WANDB_MODE", "disabled"), "HYDRA_FULL_ERROR": "1",
                "PYTHONPATH": str(SRC) + os.pathsep + env.get("PYTHONPATH", "")})
    return cmd, env


def stream(cmd, cwd, env, logfile=None, tail=300):
    """Run, echo every line (-> the stage log through Tee) and keep the last lines. Errors are never swallowed."""
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = deque(maxlen=tail)
    fh = open(logfile, "w") if logfile else None
    for line in p.stdout:
        sys.stdout.write(line)
        last.append(line.rstrip("\n"))
        if fh:
            fh.write(line)
    rc = p.wait()
    if fh:
        fh.close()
    return rc, list(last)


def observed_tqdm_totals(lines):
    """Best effort: distinct 'n/M' totals of tqdm bars in the last output lines (for comparing with steps_per_epoch). UNVERIFIED format."""
    import re
    return sorted({int(m.group(2)) for ln in lines for m in re.finditer(r"\b(\d+)/(\d+)\s*\[", ln)})


def train(cfg, paths, cond, seed, force=False, dry_run=False):
    name = mk_run_name(cond, seed)
    rd = run_dir(paths, name)
    man = paths["manifests"] / ("train_%s.json" % cond)
    for k in ("train.num_hist", "train.batch_size", "train.epochs"):
        require(cfg, k)
    try:
        bud = budget_for(cfg, paths, cond)
    except (FileNotFoundError, ImportError, ModuleNotFoundError) as e:
        if not dry_run:
            raise
        log("train %s: DRY-RUN could not compute steps_per_epoch (%s: %s). A real run computes it from the manifest + branch loader before launching." % (name, type(e).__name__, e))
        bud = None
    ep = bud["epochs"] if bud else int(cfg["train"]["epochs"])      # dry-run fallback: UNCONVERTED train.epochs (shown command is not final)
    cmd, env = build_cmd(cfg, paths, man, seed, rd, epochs=ep)
    log("train %s cwd=%s\n  %s" % (name, paths["acwm_src"], " ".join(cmd)))
    if bud:
        log("train %s budget: epochs=%d steps_per_epoch=%d steps_computed=%d (target %d, rel_diff %.4f)" % (name, bud["epochs"], bud["steps_per_epoch"], bud["steps_computed"], bud["steps_target"], bud["rel_diff"]))
    if dry_run:
        return 0
    if not man.exists():
        log("missing %s - run: python run.py manifest" % man)
        return 1
    if (rd / "DONE").exists() and not force:
        log("train %s: DONE exists, skip" % name)
        return 0
    rd.mkdir(parents=True, exist_ok=True)
    write_json(read_json(man), rd / "train_manifest.json")
    write_json({**bud, "seed": int(seed), "status": "started"}, rd / "train_budget.json")
    with open(rd / "run_info.txt", "a") as f:
        f.write("START exp002 commit=%s cond=%s seed=%s args=%s\n" % (cfg["code"]["commit"], cond, seed, " ".join(cmd)))
    rc, tail = stream(cmd, paths["acwm_src"], env)
    obs = observed_tqdm_totals(tail)
    write_json({**bud, "seed": int(seed), "status": "finished" if rc == 0 else "failed", "returncode": rc, "observed_tqdm_totals_in_last_300_lines": obs,
                "observed_note": "tqdm totals include train and validation bars; compare with steps_per_epoch by eye. Not a verified optimizer step count."}, rd / "train_budget.json")
    if rc == 0:
        (rd / "DONE").write_text("ok\n")
    return rc
