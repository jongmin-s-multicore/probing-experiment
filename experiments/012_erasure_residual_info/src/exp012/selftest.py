"""CPU self-test of experiment 012 on SYNTHETIC data and a toy world model (009's toy builders). Numbers printed here are NOT results.
Covers: verdict boundaries (0.5333333333, 0.2, 2 of 3 seeds, not_evaluable), probe/test/fit episode separation (asserts + poisoned-feature test), measure toys
(I2-a: class means differ but linear covariance with y erased -> (a) above chance; I2-b: nonlinear-only magnitude -> MLP R2 high / linear ~0; I2-c: size only in the
action coordinates -> (c) detects), logistic solver, unit bookkeeping/resume, config consistency with 009/002, checkpoint-structure preflight (exit code 4),
end-to-end toy run (extract -> measures -> aggregate)."""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from exp009 import erase as ER
from . import base as B
from . import measure as MS
from . import stages as S
from . import verdict as VD

REC = {"pass": [], "fail": [], "skipped": []}
WIN = {"early": [0, 1, 2, 3], "late": [4, 5, 6, 7]}


def check(name, cond, detail=""):
    (REC["pass"] if cond else REC["fail"]).append(name)
    print("  [%s] %s %s" % ("ok" if cond else "FAIL", name, detail))


def _raises(f, exc=AssertionError):
    try:
        f()
    except exc:
        return True
    return False


# ------------------------------------------------------------------------------------------------ A. verdict rules
def t_verdict(cfg):
    print("A  verdict rules (config success block)")
    sc = cfg["success"]
    ts = [0, 1, 2]
    thr = sc["a_accuracy_min"]
    check("config thresholds are the confirmed numbers (0.5333333333, 0.2, 0.2, 2 seeds)", (thr, sc["b_r2_min"], sc["c_r2_min"], sc["min_seeds"]) == (0.5333333333, 0.2, 0.2, 2))
    j = lambda vals, t, **kw: VD.judge_item(dict(zip(ts, vals)), t, 2, ts, **kw)
    check("(a) exactly 0.5333333333 counts (>=)", j([0.5333333333] * 3, thr)["status"] == VD.MET)
    check("(a) 0.5333333332 does not count", j([0.5333333332] * 3, thr)["status"] == VD.NOT_MET)
    check("(a) 8/15 = 0.53333.. counts", j([8 / 15] * 3, thr)["status"] == VD.MET)
    check("R2 exactly 0.2 counts, 0.19999 does not", j([0.2] * 3, 0.2)["status"] == VD.MET and j([0.19999] * 3, 0.2)["status"] == VD.NOT_MET)
    check("2 of 3 seeds -> met", j([0.9, 0.9, 0.0], 0.2)["status"] == VD.MET and j([0.9, 0.9, 0.0], 0.2)["n_seeds_met"] == 2)
    check("1 of 3 seeds -> not met", j([0.9, 0.0, 0.0], 0.2)["status"] == VD.NOT_MET)
    check("which seeds met is irrelevant (seeds 1 and 2)", j([0.0, 0.9, 0.9], 0.2)["status"] == VD.MET)
    check("a missing unit -> not_evaluable (no estimate substituted)", j([0.9, 0.9, None], 0.2)["status"] == VD.NOT_EVAL and j([0.9, 0.9, None], 0.2)["missing_train_seeds"] == [2])
    check("NaN value -> not_evaluable", j([0.9, 0.9, float("nan")], 0.2)["status"] == VD.NOT_EVAL)
    bv = lambda m_a: {ts_: {"mlp": m_a[i][0], "attentive": m_a[i][1]} for i, ts_ in enumerate(ts)}
    jb = lambda v: VD.judge_item(v, 0.2, 2, ts, any_of=True)
    check("(b) MLP OR attentive per seed: mlp>=0.2 on seeds 0,1 only -> met", jb(bv([(0.3, 0.0), (0.2, 0.0), (0.0, 0.0)]))["status"] == VD.MET)
    check("(b) mlp on seed 0, attentive on seed 1 -> met (any-of per seed)", jb(bv([(0.3, 0.0), (0.0, 0.5), (0.0, 0.0)]))["status"] == VD.MET)
    check("(b) both below on 2 seeds -> not met", jb(bv([(0.3, 0.0), (0.1, 0.1), (0.0, 0.0)]))["status"] == VD.NOT_MET)
    check("(b) attentive missing on one seed -> not_evaluable", jb(bv([(0.3, 0.3), (0.3, None), (0.3, 0.3)]))["status"] == VD.NOT_EVAL)
    v = VD.block_verdict(dict(zip(ts, [0.3] * 3)), bv([(0.0, 0.0)] * 3), dict(zip(ts, [0.0] * 3)), sc, ts)
    check("block: nothing met and all evaluable -> '이 측정들로는 잔여 정보 미검출'", v["overall"]["status"] == VD.UNDETECTED == "이 측정들로는 잔여 정보 미검출")
    v = VD.block_verdict(dict(zip(ts, [0.9] * 3)), bv([(0.0, 0.0)] * 3), dict(zip(ts, [0.0] * 3)), sc, ts)
    check("block: only (a) met -> status criterion_met with items_met [a_class_structure]", v["overall"]["status"] == VD.MET and v["overall"]["items_met"] == ["a_class_structure"])
    v = VD.block_verdict(dict(zip(ts, [0.3, 0.3, None])), bv([(0.0, 0.0)] * 3), dict(zip(ts, [0.0] * 3)), sc, ts)
    check("block: nothing met but (a) not evaluable -> not_evaluable (no 'undetected' claim)", v["overall"]["status"] == VD.NOT_EVAL)
    v = VD.block_verdict(dict(zip(ts, [None] * 3)), bv([(0.9, 0.0)] * 3), dict(zip(ts, [0.0] * 3)), sc, ts)
    check("block: (b) met while (a) not evaluable -> criterion_met listing the unevaluable item", v["overall"]["status"] == VD.MET and v["overall"]["items_not_evaluable"] == ["a_class_structure"])


# ------------------------------------------------------------------------------------------------ B. split assertions
def t_split():
    print("B  probe-train / direction-fit / test separation")
    Etr, E = 40, 70
    fit = np.arange(0, 40, 2)
    held, te = MS.probe_split(Etr, E, fit)
    check("probe_split: held = train episodes not in fit; test = test split", len(held) == 20 and not np.intersect1d(held, fit).size and te.min() == Etr and te.max() == E - 1)
    check("assert: probe-train overlapping fit episodes raises", _raises(lambda: MS.assert_probe_split(np.r_[held, fit[:1]], te, fit, Etr, E)))
    check("assert: probe-train containing a test episode raises", _raises(lambda: MS.assert_probe_split(np.r_[held, [Etr]], te, fit, Etr, E)))
    check("assert: fit episode in the test split raises", _raises(lambda: MS.assert_probe_split(held, te, np.r_[fit, [Etr + 1]], Etr, E)))
    check("assert: test index inside the train split raises", _raises(lambda: MS.assert_probe_split(held, np.r_[te, [3]], fit, Etr, E)))
    check("assert: empty probe-train set (fit = all) raises", _raises(lambda: MS.probe_split(Etr, E, np.arange(Etr))))
    check("row check: rows from a fit episode raise", _raises(lambda: MS.assert_rows_in(np.repeat(np.r_[held, fit[:1]], 3), held, fit, "t")))
    check("row check: clean rows pass", not _raises(lambda: MS.assert_rows_in(np.repeat(held, 3), held, fit, "t")))


# ------------------------------------------------------------------------------------------------ C. measure toys
def toy_cfgs(cfg, cfg2, n_train, n_test):
    from exp004 import selftest as T4
    c2 = T4.small_cfg2(cfg2, n_train=n_train, n_test=n_test)
    c2["probe_impl"]["device"] = "cpu"
    cfgt = copy.deepcopy(cfg)
    cfgt["measure"]["probe_seeds"] = [0, 1, 2]
    return cfgt, c2, MS.probe_cfg(cfgt, c2), T4.make_rows(c2)


def y_of(rows):
    return np.array([r["nominal_log_area"] for r in rows])


def apply_eraser(cfgt, pz, Yp, fit_ep):
    E = ER.fit_eraser(pz[fit_ep].reshape(-1, pz.shape[-1]), Yp[fit_ep].reshape(-1), cfgt["intervention"]["method"], float(cfgt["intervention"]["ridge_rel"]))
    return ER.apply_eraser_np(E, pz.reshape(-1, pz.shape[-1]), 1.0).reshape(pz.shape).astype(np.float32), E


def val(r, m, sp, w="late"):
    return r[m][w][sp]["mean"]


def t_measures(cfg, cfg2):
    print("C  measure toys (does the measurement catch what skeptic I2 says the scalar erasure can leave?)")
    cfgt, c2, c2p, rows = toy_cfgs(cfg, cfg2, 300, 40)
    E = len(rows)
    Etr = sum(r["part"] == "train" for r in rows)
    y = y_of(rows)
    nF, dz, da = 8, 12, 4
    Yp = np.repeat(y[:, None], nF, 1)
    rng = np.random.RandomState(0)
    fit_ep = S.fit_episodes(cfgt, 0, Etr)
    lv = np.array([r["level"] for r in rows])
    # ---- toy I2-a: class means differ in a 2-d span (linear + quadratic in y), unit noise
    u1, u2 = rng.randn(dz), rng.randn(dz)
    u1, u2 = u1 / np.linalg.norm(u1), u2 / np.linalg.norm(u2)
    q = y ** 2
    mean = 6.0 * y[:, None] * u1[None] + 14.0 * (q - q[:Etr].mean())[:, None] * u2[None]
    pz = (mean[:, None, :] + rng.randn(E, nF, dz)).astype(np.float32)
    pa = rng.randn(E, nF, da).astype(np.float32)
    pz0 = pz.copy()
    pz1, Er = apply_eraser(cfgt, pz, Yp, fit_ep)
    cov = lambda X: np.abs(np.cov(X[fit_ep].reshape(-1, dz).T, Yp[fit_ep].reshape(-1))[:-1, -1]).max()
    check("toy I2-a: after the scalar erasure the covariance of every coordinate with y is ~0 on the fit episodes", cov(pz1) < 1e-2 * cov(pz0), "(%.2g vs %.2g)" % (cov(pz1), cov(pz0)))
    wh = ["a_level_logistic", "b_mlp", "c_action_only_linear", "d_linear_pooled"]
    r0 = MS.measure_all(cfgt, c2p, pz0, pa, None, Yp, rows, Etr, fit_ep, WIN, [0, 1, 2], which=wh)
    r1 = MS.measure_all(cfgt, c2p, pz1, pa, None, Yp, rows, Etr, fit_ep, WIN, [0, 1, 2], which=wh)
    a0, a1 = val(r0, "a_level_logistic", "train"), val(r1, "a_level_logistic", "train")
    d0, d1 = val(r0, "d_linear_pooled", "train"), val(r1, "d_linear_pooled", "train")
    check("toy I2-a: linear R2 on the erased features is ~0 while the unerased value is high", abs(d1) < 0.1 and d0 > 0.5, "(unerased %.3f, erased %.3f)" % (d0, d1))
    check("toy I2-a: (a) accuracy on the erased features is ABOVE the 0.5333 threshold although linear R2 ~0 (the measurement catches the left-over class structure)", a1 >= 0.5333333333 and a1 > 1 / 3 + 0.2, "(unerased %.3f, erased %.3f, chance 0.333)" % (a0, a1))
    check("toy I2-a: (c) with no size information in the action coordinates is below 0.2", val(r1, "c_action_only_linear", "all") < 0.2, "(%.3f)" % val(r1, "c_action_only_linear", "all"))
    check("toy I2-a: MLP R2 is finite (the left-over structure here is a class-identity pattern, not an R2 gain: R2 is reported, not asserted)", val(r1, "b_mlp", "train") is not None, "(mlp %.3f linear %.3f)" % (val(r1, "b_mlp", "train"), d1))
    # negative control: size only LINEAR in y -> after erasure class structure is gone, (a) near chance
    mean_l = 6.0 * y[:, None] * u1[None]
    pzl = (mean_l[:, None, :] + rng.randn(E, nF, dz)).astype(np.float32)
    pzl1, _ = apply_eraser(cfgt, pzl, Yp, fit_ep)
    rl = MS.measure_all(cfgt, c2p, pzl1, pa, None, Yp, rows, Etr, fit_ep, WIN, [0, 1, 2], which=["a_level_logistic", "d_linear_pooled"])
    check("toy negative control (size purely linear in z): after the erasure (a) is near chance (< 0.5333) and linear R2 ~0", val(rl, "a_level_logistic", "train") < 0.5333 and abs(val(rl, "d_linear_pooled", "train")) < 0.1,
          "(acc %.3f, R2 %.3f)" % (val(rl, "a_level_logistic", "train"), val(rl, "d_linear_pooled", "train")))
    # ---- toy I2-b: magnitude stored only as |x1| (random sign): linear ~0, MLP high
    mag = 1.5 * np.exp(y)
    sign = rng.choice([-1.0, 1.0], size=(E, nF))
    pzn = (0.2 * rng.randn(E, nF, dz)).astype(np.float32)
    pzn[..., 0] = (sign * mag[:, None] + 0.1 * rng.randn(E, nF)).astype(np.float32)
    rn = MS.measure_all(cfgt, c2p, pzn, pa, None, Yp, rows, Etr, fit_ep, WIN, [0, 1, 2], which=wh)
    check("toy I2-b (nonlinear only): MLP R2 high, linear R2 ~0", val(rn, "b_mlp", "all") > 0.5 and abs(val(rn, "d_linear_pooled", "all")) < 0.1, "(mlp %.3f, linear %.3f)" % (val(rn, "b_mlp", "all"), val(rn, "d_linear_pooled", "all")))
    check("toy I2-b: a linear (a) classifier cannot read the levels from |x1| (accuracy < 0.5333)", val(rn, "a_level_logistic", "train") < 0.5333, "(%.3f)" % val(rn, "a_level_logistic", "train"))
    # ---- toy I2-c: size only in the action coordinates
    pza = rng.randn(E, nF, dz).astype(np.float32)
    paa = (y[:, None, None] * rng.randn(1, 1, da) * 2.0 + 0.5 * rng.randn(E, nF, da)).astype(np.float32)
    ra = MS.measure_all(cfgt, c2p, pza, paa, None, Yp, rows, Etr, fit_ep, WIN, [0, 1, 2], which=wh)
    check("toy I2-c (size only in action coordinates): (c) R2 high, (d) linear R2 on z ~0, (a) near chance", val(ra, "c_action_only_linear", "all") > 0.5 and abs(val(ra, "d_linear_pooled", "all")) < 0.1 and val(ra, "a_level_logistic", "train") < 0.5333,
          "(c %.3f, d %.3f, a %.3f)" % (val(ra, "c_action_only_linear", "all"), val(ra, "d_linear_pooled", "all"), val(ra, "a_level_logistic", "train")))
    check("results carry per-split values (train, interp, extrap, all) for R2 measures and 'train' only for (a)", set(ra["d_linear_pooled"]["late"]) == {"train", "interp", "extrap", "all"} and set(ra["a_level_logistic"]["late"]) == {"train"})
    check("per-probe-seed lists have one value per probe seed (R2) ", len(ra["b_mlp"]["late"]["all"]["per_seed"]) == 3 and ra["meta"]["train_row_episode_ids_disjoint_from_fit_and_test"] is True)
    # ---- poison test: the direction-fit episodes' features are garbage; nothing may change if they are never used for probe training
    pzp = pz1.copy()
    pap = pa.copy()
    pzp[fit_ep] = 1e3 * rng.randn(len(fit_ep), nF, dz)
    pap[fit_ep] = -1e3
    Yq = Yp.copy()
    Yq[fit_ep] = 77.0
    rp = MS.measure_all(cfgt, c2p, pzp, pap, None, Yq, rows, Etr, fit_ep, WIN, [0, 1, 2], which=wh)
    same = all(json.dumps(r1[k]["late"], sort_keys=True) == json.dumps(rp[k]["late"], sort_keys=True) for k in wh)
    check("poisoned features/labels of the direction-fit episodes change NOTHING (probes never train on them)", same)
    # test-episode labels are not used for training: change test labels -> predictions (and the classifier) are the same, only the scores move
    Yt = Yp.copy()
    Yt[Etr:] = Yt[Etr:] + 5.0
    rt = MS.measure_all(cfgt, c2p, pz1, pa, None, Yt, rows, Etr, fit_ep, WIN, [0, 1, 2], which=["a_level_logistic"])
    check("(a) does not depend on test labels other than the level names (test labels shifted: same accuracy)", json.dumps(rt["a_level_logistic"]["late"], sort_keys=True) == json.dumps(r1["a_level_logistic"]["late"], sort_keys=True))
    check("measure_all refuses a fit set covering the whole train split", _raises(lambda: MS.measure_all(cfgt, c2p, pz1, pa, None, Yp, rows, Etr, np.arange(Etr), WIN, [0], which=["d_linear_pooled"])))


def t_logistic():
    print("D  logistic solver")
    import torch
    rng = np.random.RandomState(1)
    n, d, K = 600, 5, 3
    yy = rng.randint(0, K, n)
    cen = rng.randn(K, d) * 1.5
    X = cen[yy] + rng.randn(n, d)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X1 = MS._prep(X, mu, sd, torch.device("cpu"))
    Y1 = torch.as_tensor(np.eye(K)[yy], dtype=torch.float64)
    pen = torch.cat([torch.ones(d), torch.full((1,), 1e-6)]).double()
    C = 0.5
    th = MS._newton(X1, Y1, C, pen, torch.zeros(K, d + 1, dtype=torch.float64))
    t = th.clone().requires_grad_(True)
    Z = X1 @ t.T
    f = C * (torch.logsumexp(Z, 1) - (Z * Y1).sum(1)).sum() + 0.5 * (pen * t * t).sum()
    f.backward()
    check("Newton solution is stationary (max |grad| < 1e-4)", float(t.grad.abs().max()) < 1e-4, "(%.2g)" % float(t.grad.abs().max()))
    cfg2 = {"probe_impl": {"inner_folds": 3, "device": "cpu"}}
    pred, info = MS.fit_logistic_cv(X, yy, np.arange(n), K, cfg2, 0)
    bayes_like = (pred(X) == yy).mean()
    check("fit_logistic_cv: separable-ish blobs are classified well above chance; C comes from the grid", bayes_like > 0.7 and info["C"] in MS.LOGISTIC_IMPL["C_grid"], "(train acc %.3f, C %g)" % (bayes_like, info["C"]))
    ylab = rng.randint(0, K, n)
    pred2, info2 = MS.fit_logistic_cv(X, ylab, np.arange(n), K, cfg2, 0)
    Xn = cen[rng.randint(0, K, 600)] + rng.randn(600, d)
    check("fit_logistic_cv: labels unrelated to X -> accuracy on fresh random labels near chance", abs((pred2(Xn) == rng.randint(0, K, 600)).mean() - 1 / 3) < 0.08)


# ------------------------------------------------------------------------------------------------ E. bookkeeping, config, preflight
def t_bookkeeping(cfg, cfg2, c9):
    print("E  units, resume, config consistency, checkpoint structure preflight")
    us = S.units([1, 2, 3, 4, 5, 6], [0, 1, 2])
    check("units: 3 baselines + 6 blocks x 3 seeds = 21", len(us) == 21 and sum(u[0] == "base" for u in us) == 3)
    check("needed baseline depths for m_int [1..6] = 1..6; for [6] = [6]; for [2,6] = [2,3,6]", S.needed_depths([1, 2, 3, 4, 5, 6], 6) == [1, 2, 3, 4, 5, 6] and S.needed_depths([6], 6) == [6] and S.needed_depths([2, 6], 6) == [2, 3, 6])
    tmp = Path(tempfile.mkdtemp())
    try:
        paths = {"results_dir": tmp}
        names = {S.part_path(paths, u[0], u[1], u[2]).name for u in us}
        check("part file names distinct for every unit", len(names) == 21)
        B.write_json({"depths": {"2": {}, "3": {}}}, S.part_path(paths, "base", 0))
        check("base unit_done only when it covers the needed depths", not S.unit_done(paths, ("base", 0, None), [2, 6]) and S.unit_done(paths, ("base", 0, None), [2]))
        B.write_json({}, S.part_path(paths, "unit", 0, 3))
        check("unit_done recognises finished erased units only", S.unit_done(paths, ("unit", 0, 3), [3]) and not S.unit_done(paths, ("unit", 0, 4), [4]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    v, o, src = S.resolve_m_int(cfg, None)
    check("default m_int [1..6] from config, not overridden", v == [1, 2, 3, 4, 5, 6] and not o and src == "config")
    v, o, src = S.resolve_m_int(cfg, "6,2")
    check("CLI override sorted and recorded as overridden", v == [2, 6] and o and src == "cli")
    res = B.consistency_checks(cfg, c9, cfg2)
    check("config consistency with 009 and 002 (%d checks)" % len(res), all(ok for _, ok, _ in res), str([n for n, ok, _ in res if not ok]))
    c_bad = copy.deepcopy(cfg)
    c_bad["intervention"]["fit_seed"] = 1
    c_bad["measure"]["b_attentive"]["d_model"] = 64
    bad = [n for n, ok, _ in B.consistency_checks(c_bad, c9, cfg2) if not ok]
    check("a changed fit_seed / attentive d_model is detected", any("fit_seed" in n for n in bad) and any("d_model" in n for n in bad), str(bad))
    check("config: seeds explicit (global, train seeds, probe seeds, fit seed) and pilot flag present", cfg["seed"] == 0 and cfg["predictor"]["train_seeds"] == [0, 1, 2] and cfg["measure"]["probe_seeds"] == [0, 1, 2, 3, 4] and cfg["intervention"]["fit_seed"] == 9000 and "pilot" in cfg)
    check("config: m_int [1..6], strengths [0, 1.0], method leace_scalar, depths m_int and m_int+1", cfg["intervention"]["m_int"] == [1, 2, 3, 4, 5, 6] and cfg["intervention"]["strengths"] == [0, 1.0] and cfg["intervention"]["method"] == "leace_scalar" and cfg["measure"]["depths"] == ["m_int", "m_int+1"])
    # checkpoint structure preflight (006 failed 3 times: run name lacked the exp002_ prefix)
    c4, _ = B.make_cfg004(cfg)
    tmp = Path(tempfile.mkdtemp())
    try:
        paths = {"train_base": tmp}
        out = tmp / "outputs"
        (out / "multi_60_ind_160_s0").mkdir(parents=True)
        bad = S.checkpoint_problems(c4, paths, [0])
        check("preflight: run dir without the exp002_ prefix is not accepted; message names the expected folder and the near miss", len(bad) == 1 and "exp002_multi_60_ind_160_s0" in bad[0] and "multi_60_ind_160_s0" in bad[0].split("directories there")[1], bad[0][:200])
        rd = out / "exp002_multi_60_ind_160_s0"
        (rd / "checkpoints").mkdir(parents=True)
        for f in ("DONE", "hydra.yaml", "train_manifest.json"):
            (rd / f).write_text("x")
        bad = S.checkpoint_problems(c4, paths, [0])
        check("preflight: model_latest.pth missing is reported", len(bad) == 1 and "model_latest.pth" in bad[0])
        (rd / "checkpoints" / "model_latest.pth").write_bytes(b"x")
        check("preflight: complete run dir (exp002_multi_60_ind_160_s0/checkpoints/model_latest.pth) -> no problem", S.checkpoint_problems(c4, paths, [0]) == [])
        check("preflight: seed 1 without a directory is reported", len(S.checkpoint_problems(c4, paths, [0, 1])) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # run.py: preflight exits 4 before any attempt is consumed
    src = (Path(S.__file__).parent.parent.parent / "run.py").read_text()
    check("run.py: preflight (exit 4) comes before attempts_check / attempts_record", src.index("input_preflight") < src.index("B.attempts_check") and "sys.exit(4)" in src)
    tmp = Path(tempfile.mkdtemp())
    try:
        env = dict(os.environ, DINO_TRAIN=str(tmp / "train_runs"), DINO_WORK=str(tmp / "work"), EXP012_CACHE=str(tmp / "cache"), MPLBACKEND="Agg")
        env.pop("EXP002_CACHE", None)
        p = subprocess.run([sys.executable, str(Path(S.__file__).parent.parent.parent / "run.py"), "measure", "--results-dir", str(tmp / "res")], capture_output=True, text=True, env=env, timeout=300)
        check("run.py measure with missing checkpoints/metadata exits 4 and writes no attempts.json", p.returncode == 4 and not (tmp / "res" / "attempts.json").exists() and "PREFLIGHT" in p.stderr, "(rc=%s)" % p.returncode)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ F. end to end on the toy world
def t_end_to_end(cfg, cfg2, c9):
    print("F  toy world model: extraction with erasure, measures, resume, aggregate")
    try:
        import torch
    except ImportError:
        REC["skipped"].append("F")
        print("  [skipped(no torch)]")
        return
    from exp009.selftest import toy_setup
    from exp002 import evalgrid as EG
    M, old, c2, rows, C, w, (nvis, nact, P, L, H), T4 = toy_setup(cfg2)
    tmp = Path(tempfile.mkdtemp())
    try:
        dev = torch.device("cpu")
        Etr = sum(r["part"] == "train" for r in rows)
        frames = [0, 1, 2, 3, 8, 9, 10, 11]
        pos = {"early": [0, 1, 2, 3], "late": [4, 5, 6, 7]}
        mean, std = torch.zeros(6), torch.ones(6)
        Y = EG.make_labels(c2, rows, np.asarray(C["mask_feat"]), c2["q1"]["primary_target"])
        Yp = Y[:, frames]
        cfgt = copy.deepcopy(cfg)
        cfgt["measure"]["probe_seeds"] = [0, 1, 2]
        cfgt["predictor"]["batch_episodes"] = 16
        c2p = MS.probe_cfg(cfgt, c2)
        paths = {"results_dir": tmp / "res", "cache": tmp / "cache", "exp": tmp, "results009": tmp / "none.json"}
        m_ints, ts = [2, 6], 0
        todo = S.units(m_ints, [ts])
        diag = S.run_seed12(cfgt, c2p, paths, w, C, H, mean, std, dev, frames, pos, Yp, rows, Etr, nvis, nact, ts, todo, m_ints)
        files = sorted(p.name for p in (paths["results_dir"] / "parts").glob("*.json"))
        check("run_seed12 wrote base_s0 + unit_m2_s0 + unit_m6_s0", files == ["base_s0.json", "unit_m2_s0.json", "unit_m6_s0.json"], str(files))
        base = B.read_json(S.part_path(paths, "base", ts))
        check("baseline part covers depths [2, 3, 6] (m_int and m_int+1 of both blocks; 7 does not exist)", sorted(int(k) for k in base["depths"]) == [2, 3, 6])
        u2, u6 = B.read_json(S.part_path(paths, "unit", ts, 2)), B.read_json(S.part_path(paths, "unit", ts, 6))
        check("block 2 unit has records at m_int and m_int+1; block 6 unit: m_int+1 is '해당 없음'", isinstance(u2["depth_m_int"], dict) and isinstance(u2["depth_m_int_plus_1"], dict) and isinstance(u6["depth_m_int"], dict) and str(u6["depth_m_int_plus_1"]).startswith("해당 없음"))
        rec = u2["depth_m_int"]
        check("record contains all five measures (b_attentive on tokens too) and the separation meta", all(isinstance(rec[m], dict) for m in MS.MEASURES) and rec["meta"]["n_probe_train_episodes"] == Etr - int(round(0.5 * Etr)) and rec["meta"]["n_direction_fit_episodes"] == int(round(0.5 * Etr)))
        fit_ep = S.fit_episodes(cfgt, ts, Etr)
        check("direction-fit episodes follow 009's rule (RandomState(9000+seed).permutation(Etr)[:round(0.5 Etr)], sorted)", np.array_equal(fit_ep, np.sort(np.random.RandomState(9000 + ts).permutation(Etr)[: int(round(0.5 * Etr))])))
        dk = diag["m2_s0"]
        check("diagnostic: after full-strength erasure the fitted readout has ~no variance on probe-train and test episodes (std ratio < 0.05)", dk["readout_std_ratio_after_over_fit_pre_test"] < 0.05 and dk["readout_std_ratio_after_over_fit_pre_probe_train"] < 0.05, str({k: round(v, 4) for k, v in dk.items() if isinstance(v, float)}))
        b2 = base["depths"]["2"]["d_linear_pooled"]["late"]["train"]["mean"]
        e2 = u2["depth_m_int"]["d_linear_pooled"]["late"]["train"]["mean"]
        check("toy: linear R2 at depth m_int=2: baseline high, erased far lower", b2 > 0.5 and e2 < b2 - 0.4, "(baseline %.3f, erased %.3f)" % (b2, e2))
        mt = {p.name: p.stat().st_mtime_ns for p in (paths["results_dir"] / "parts").glob("*.json")}
        todo2 = [u for u in S.units(m_ints, [ts]) if not S.unit_done(paths, u, m_ints)]
        S.run_seed12(cfgt, c2p, paths, w, C, H, mean, std, dev, frames, pos, Yp, rows, Etr, nvis, nact, ts, todo2, m_ints)
        check("resume: nothing to do, part files untouched", todo2 == [] and mt == {p.name: p.stat().st_mtime_ns for p in (paths["results_dir"] / "parts").glob("*.json")})
        S.part_path(paths, "unit", ts, 2).unlink()
        todo3 = [u for u in S.units(m_ints, [ts]) if not S.unit_done(paths, u, m_ints)]
        S.run_seed12(cfgt, c2p, paths, w, C, H, mean, std, dev, frames, pos, Yp, rows, Etr, nvis, nact, ts, todo3, m_ints)
        mt2 = {p.name: p.stat().st_mtime_ns for p in (paths["results_dir"] / "parts").glob("*.json")}
        check("resume: only the deleted unit is recomputed", todo3 == [("unit", 0, 2)] and all(mt2[k] == mt[k] for k in mt if k != "unit_m2_s0.json"))
        m_ints2 = [2, 4, 6]
        todo4 = [u for u in S.units(m_ints2, [ts]) if not S.unit_done(paths, u, m_ints2)]
        check("adding block 4 later: baseline is incomplete (needs depths 4, 5), unit_m4 pending", ("base", 0, None) in todo4 and ("unit", 0, 4) in todo4 and ("unit", 0, 2) not in todo4)
        S.run_seed12(cfgt, c2p, paths, w, C, H, mean, std, dev, frames, pos, Yp, rows, Etr, nvis, nact, ts, todo4, m_ints2)
        base = B.read_json(S.part_path(paths, "base", ts))
        check("baseline part merged: depths [2, 3, 4, 5, 6]; the already measured depth 2 was not recomputed", sorted(int(k) for k in base["depths"]) == [2, 3, 4, 5, 6])
        # aggregate
        cfga = copy.deepcopy(cfgt)
        cfga["success"] = dict(cfg["success"], min_seeds=1)
        cfga["_config_path"] = cfg["_config_path"]
        res = S.aggregate(cfga, None, cfg2, paths, m_ints2, True, "cli", [ts], True, c9)
        check("results.json: interpretation null, pilot true with reasons (config flag, separate dir, overrides)", res["interpretation"] is None and res["pilot"] is True and len(res["pilot_reasons"]) >= 3, str(res["pilot_reasons"]))
        check("results.json written to the results dir only (no results.json in the exp dir for a separate results dir)", (paths["results_dir"] / "results.json").exists() and not (tmp / "results.json").exists())
        for key in ("code_version", "config_sha256", "environment", "pilot", "pilot_config_flag", "m_int_used", "train_seeds", "blocks"):
            if key not in res:
                check("results.json has key %s" % key, False)
        check("results.json has code_version(git_head, git_status_dirty, repo_url, requested_ref), config_sha256, environment", set(res["code_version"]) == {"git_head", "git_status_dirty", "repo_url", "requested_ref"} and len(res["config_sha256"]) == 64 and "python" in res["environment"])
        b6 = res["blocks"]["6"]
        check("block 6: auxiliary m_int+1 observation is '해당 없음'", str(b6["compact_depth_m_int_plus_1_auxiliary"]).startswith("해당 없음") and b6["records_depth_m_int_plus_1"] is None)
        check("block 2: auxiliary m_int+1 is a dict with strength_1 and strength_0", isinstance(res["blocks"]["2"]["compact_depth_m_int_plus_1_auxiliary"], dict) and "0" in res["blocks"]["2"]["compact_depth_m_int_plus_1_auxiliary"]["strength_1"])
        check("every block has a verdict with items a/b/c and an overall status", all(set(b["verdict"]["items"]) == set(VD.ITEMS.values()) and "status" in b["verdict"]["overall"] for b in res["blocks"].values()))
        check("009 results.json absent -> (d) reproduction is 'not_available' (no number invented)", res["blocks"]["2"]["d_reproduction_vs_009_at_m_int"]["status"] == "not_available")
        # (d) reproduction against a fake 009 results.json
        fake = {"blocks": {"2": {"strengths": {"1": {"0": {"curves": {"train": {"2": 0.1}, "interp": {"2": 0.1}, "extrap": {"2": 0.1}}}}, "0": {"0": {"curves": {"train": {"2": 0.9}, "interp": {"2": 0.9}, "extrap": {"2": 0.9}}}}}}}}
        B.write_json(fake, paths["results009"])
        res_r = S.aggregate(cfga, None, cfg2, paths, [2], True, "cli", [ts], True, c9)
        dr = res_r["blocks"]["2"]["d_reproduction_vs_009_at_m_int"]["by_strength"]["1"]["0"]["train"]
        check("(d) reports 012 minus 009 difference per split and strength", dr["r2_009"] == 0.1 and abs(dr["diff_012_minus_009"] - (dr["r2_012"] - 0.1)) < 1e-12)
        S.part_path(paths, "unit", ts, 4).unlink()
        res2 = S.aggregate(cfga, None, cfg2, paths, m_ints2, True, "cli", [ts], True, c9)
        check("a missing unit -> that block not_evaluable, other blocks unaffected", res2["blocks"]["4"]["verdict"]["overall"]["status"] == VD.NOT_EVAL and res2["blocks"]["2"]["verdict"]["overall"]["status"] != VD.NOT_EVAL)
        cm = copy.deepcopy(cfg)
        cm["pilot"] = False
        check("pilot logic: config pilot false + experiment results dir + no overrides -> no reasons (main run)", S.pilot_reasons(cm, B.default_results_dir(), False, False, [1], [0]) == [])
        check("pilot logic: separate results dir -> pilot; config flag true -> pilot; overrides -> pilot", len(S.pilot_reasons(cm, tmp / "x", False, False, [1], [0])) == 1 and len(S.pilot_reasons(cfg, B.default_results_dir(), False, False, [1], [0])) == (1 if cfg["pilot"] else 0) and len(S.pilot_reasons(cm, B.default_results_dir(), True, True, [1], [0])) == 2)
    finally:
        M.P_TOK, M.D_VIS = old
        shutil.rmtree(tmp, ignore_errors=True)


def run(cfg, c4, cfg2, c9):
    for name, fn in (("A", lambda: t_verdict(cfg)), ("B", t_split), ("D", t_logistic), ("C", lambda: t_measures(cfg, cfg2)), ("E", lambda: t_bookkeeping(cfg, cfg2, c9)), ("F", lambda: t_end_to_end(cfg, cfg2, c9))):
        try:
            fn()
        except Exception:
            REC["fail"].append(name + " (exception)")
            traceback.print_exc()
    print("selftest summary: passed=%d failed=%d skipped=%d" % (len(REC["pass"]), len(REC["fail"]), len(REC["skipped"])))
    if REC["fail"]:
        print("FAILED:", REC["fail"])
    return 1 if REC["fail"] else 0
