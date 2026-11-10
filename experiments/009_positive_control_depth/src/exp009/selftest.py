"""CPU self-test of experiment 009 on SYNTHETIC data and a toy world model (built with 004's selftest builders). Numbers printed here are NOT results.
Covers: eraser algebra, torch/numpy agreement, forward with intervention vs 004's ee_forward, unit bookkeeping and resume, shared baseline, per-block verdicts,
override recording, end-to-end toy run (extract -> probes -> aggregate) where erasing at block m_int must make the 004 rule fire at m* <= m_int + 1."""
from __future__ import annotations

import copy
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from . import base as B
from . import curves as CU
from . import erase as ER
from . import stages as S

REC = {"pass": [], "fail": [], "skipped": []}


def check(name, cond, detail=""):
    (REC["pass"] if cond else REC["fail"]).append(name)
    print("  [%s] %s %s" % ("ok" if cond else "FAIL", name, detail))


def t_eraser():
    print("A  eraser algebra (numpy)")
    rng = np.random.RandomState(0)
    n, d = 600, 12
    y = rng.randn(n)
    mix = rng.randn(d)
    X = np.outer(y, mix) + rng.randn(n, d) * 0.5 + 3.0          # size info spread over all coordinates, offset mean
    for method in ("leace_scalar", "probe_direction"):
        E = ER.fit_eraser(X, y, method, 1e-3)
        X1 = ER.apply_eraser_np(E, X, 1.0)
        cov1 = np.abs(np.cov(X1.T, y)[:-1, -1]).max()
        cov0 = np.abs(np.cov(X.T, y)[:-1, -1]).max()
        if method == "leace_scalar":
            check("leace_scalar: after alpha=1 the covariance of every coordinate with the label is ~0", cov1 < 1e-2 * cov0 + 1e-3, "(%.2g vs %.2g before)" % (cov1, cov0))
            res = y - np.c_[X1, np.ones(n)] @ np.linalg.lstsq(np.c_[X1, np.ones(n)], y, rcond=None)[0]
            check("leace_scalar: best linear probe on the erased features explains ~0 of the variance (in-sample R2 < 0.05)", 1 - res.var() / y.var() < 0.05, "(R2 %.3f)" % (1 - res.var() / y.var()))
        else:
            check("probe_direction: component along u is zero after alpha=1", np.abs(X1 @ E["u"]).max() < 1e-9)
            res = y - np.c_[X1, np.ones(n)] @ np.linalg.lstsq(np.c_[X1, np.ones(n)], y, rcond=None)[0]
            check("probe_direction: other coordinates still carry size info (in-sample linear R2 > 0.3) -> weaker than leace", 1 - res.var() / y.var() > 0.3, "(R2 %.3f)" % (1 - res.var() / y.var()))
        X5 = ER.apply_eraser_np(E, X, 0.5)
        check("%s: alpha=0.5 is the midpoint of alpha=0 and alpha=1" % method, np.allclose(X5, 0.5 * X + 0.5 * X1, atol=1e-9))
        check("%s: alpha=0 is identity" % method, np.allclose(ER.apply_eraser_np(E, X, 0.0), X))
    try:
        ER.fit_eraser(X, np.ones(n), "leace_scalar", 1e-3)
        check("constant labels in the fit set raise", False)
    except ValueError:
        check("constant labels in the fit set raise", True)
    check("unknown method raises", _raises(lambda: ER.fit_eraser(X, y, "pca", 1e-3)))


def _raises(f):
    try:
        f()
    except ValueError:
        return True
    return False


def t_bookkeeping(cfg):
    print("B  units, resume, shared baseline, override, verdicts")
    tmp = Path(tempfile.mkdtemp())
    try:
        paths = {"results_dir": tmp}
        us = S.units([1, 2, 3, 4, 5, 6], [0, 1.0], [0, 1, 2])
        check("units: 3 baselines (one per train seed, shared by all blocks) + 6 blocks x 1 strength (1.0) x 3 seeds = 21", len(us) == 21 and sum(u[0] == "base" for u in us) == 3)
        check("part file names distinct for every unit", len({S.part_path(paths, u[0], u[1], u[2], u[3]) for u in us}) == 21)
        S.write_rows([], S.part_path(paths, "unit", 0, 3, 1.0))
        S.write_rows([], S.part_path(paths, "base", 0))
        check("unit_done recognises finished units only", S.unit_done(paths, ("unit", 0, 3, 1.0)) and S.unit_done(paths, ("base", 0, None, 0.0)) and not S.unit_done(paths, ("unit", 0, 3, 0.5)))
        check("a single-block list needs 1/6 of the intervention units", len(S.units([3], [0, 1.0], [0, 1, 2])) == 3 + 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    old = os.environ.pop("EXP009_M_INT", None)
    try:
        v, o, src = S.resolve_m_int(cfg, None)
        check("default m_int [1..6] from config (user decision 2026-11-10), not overridden", v == [1, 2, 3, 4, 5, 6] and not o and src == "config")
        v, o, src = S.resolve_m_int(cfg, "6,2")
        check("CLI override sorted, recorded as overridden (source cli)", v == [2, 6] and o and src == "cli")
        os.environ["EXP009_M_INT"] = "4"
        v, o, src = S.resolve_m_int(cfg, None)
        check("env override recorded (source env)", v == [4] and o and src.startswith("env"))
        v, o, src = S.resolve_m_int(cfg, "1,2,3,4,5,6")
        check("CLI value equal to config is not 'overridden'", v == [1, 2, 3, 4, 5, 6] and not o)
    finally:
        os.environ.pop("EXP009_M_INT", None)
        if old is not None:
            os.environ["EXP009_M_INT"] = old
    # verdict per block
    check("verdict: 2 of 3 seeds with m* <= m_int+1 -> criterion_met", CU.block_verdict({0: 4, 1: 3, 2: None}, 3, 1, 2)["status"] == "criterion_met")
    check("verdict: m* = m_int+2 does not count", CU.block_verdict({0: 5, 1: 5, 2: 4}, 3, 1, 2)["n_seeds_detected"] == 1)
    check("verdict: no detection -> criterion_not_met", CU.block_verdict({0: None, 1: None, 2: None}, 3, 1, 2)["status"] == "criterion_not_met")
    cur = {0: 0.9, 1: 0.9, 2: 0.88, 3: 0.2, 4: 0.1}
    check("detect uses 004's first_drop (0.2 below running max)", CU.detect(cur, 0.2)["depth"] == 3 and CU.detect({0: 0.9, 1: 0.8, 2: 0.75}, 0.2) is None)
    mb = {"train": {0: 0.9, 1: 0.9, 2: 0.9}, "extrap": {0: 0.9, 1: 0.5, 2: 0.4}}
    check("m_star = earliest vanishing depth over splits", CU.m_star_of(mb, 0.2)[0] == 1)


def toy_setup(cfg2):
    import torch
    from exp002 import model_io as M
    from exp004 import selftest as T4
    nvis, nact, P, L, H = 16, 4, 9, 6, 1
    old = (M.P_TOK, M.D_VIS)
    M.P_TOK, M.D_VIS = P, nvis
    c2 = T4.small_cfg2(cfg2, n_train=16, n_test=10)
    c2["probe"]["kinds"] = ["linear_pooled"]
    rows = T4.make_rows(c2)
    E = len(rows)
    s = np.array([r["nominal_log_area"] for r in rows])
    rng = np.random.RandomState(0)
    u8 = np.clip(128 + 40 * s[:, None, None, None, None] + 12 * rng.randn(E, 13, 12, 12, 3), 0, 255).astype(np.uint8)
    raw = rng.randn(E, 60, 6).astype(np.float32)
    World = T4.build_toys(nvis, nact, P, L, erase_from=None, identity_norm=True)
    torch.manual_seed(0)
    w = World(H).eval()
    mf = np.zeros((E, 13, 3), np.float32)
    mf[..., 0] = np.exp(5 + s)[:, None]
    mf[..., 1:] = 10
    C = {"rows": rows, "frames_u8": u8, "actions_raw": raw, "mask_feat": mf}
    return M, old, c2, rows, C, w, (nvis, nact, P, L, H), T4


def t_torch(cfg, cfg2):
    print("C  toy world model: forward with intervention, extraction, probes, per-block verdict end to end")
    try:
        import torch
    except ImportError:
        REC["skipped"].append("C")
        print("  [skipped(no torch)]")
        return
    from exp004 import deep as DP
    from exp004 import stages as S4
    from exp002 import evalgrid as EG
    M, old, c2, rows, C, w, (nvis, nact, P, L, H), T4 = toy_setup(cfg2)
    tmp = Path(tempfile.mkdtemp())
    try:
        dev = torch.device("cpu")
        Etr = sum(r["part"] == "train" for r in rows)
        te = T4.te_rows_of(rows)
        levels = list(c2["data"]["levels"])
        frames = [0, 1, 2, 3, 8, 9, 10, 11]
        pos = {"early": [0, 1, 2, 3], "late": [4, 5, 6, 7]}
        mean, std = torch.zeros(6), torch.ones(6)
        Y = EG.make_labels(c2, rows, np.asarray(C["mask_feat"]), c2["q1"]["primary_target"])
        Yp = Y[:, frames]
        # torch erase == numpy reference
        rng = np.random.RandomState(1)
        X = rng.randn(50, 9, nvis + nact).astype(np.float32)
        y = rng.randn(50 * 9)
        Ef = ER.fit_eraser(X[..., :nvis].reshape(-1, nvis), y, "leace_scalar", 1e-3)
        got = ER.erase_stream(torch.as_tensor(X), Ef, 0.7, nvis).numpy()
        ref = np.concatenate([ER.apply_eraser_np(Ef, X[..., :nvis].reshape(-1, nvis), 0.7).reshape(50, 9, nvis), X[..., nvis:]], -1)
        check("torch erase_stream == numpy reference; action coordinates untouched", np.abs(got - ref).max() < 1e-3 and np.array_equal(got[..., nvis:], X[..., nvis:]))
        Eo = ER.fit_eraser(X[..., :nvis].reshape(-1, nvis), y, "probe_direction", 1e-3)
        got = ER.erase_stream(torch.as_tensor(X), Eo, 1.0, nvis).numpy()
        check("torch probe_direction erase removes the u component", np.abs(got[..., :nvis] @ Eo["u"]).max() < 1e-3)
        # forward without intervention == 004 ee_forward
        z = torch.randn(5, H * P, nvis + nact)
        h1 = DP.ee_forward(w.predictor, z)
        h2 = ER.forward_hidden(w.predictor, z, None)
        h3 = ER.forward_hidden(w.predictor, z, {"m_int": 3, "alpha": 0.0, "E": Ef, "dz": nvis})
        check("forward_hidden(no erase) == exp004.deep.ee_forward at every depth; alpha=0 is the same", all(torch.equal(a, b) for a, b in zip(h1, h2)) and all(torch.equal(a, b) for a, b in zip(h1, h3)))
        h4 = ER.forward_hidden(w.predictor, z, {"m_int": 3, "alpha": 1.0, "E": Ef, "dz": nvis})
        check("intervention at m_int=3 leaves h_0..h_2 unchanged and changes h_3 and later", all(torch.equal(h1[i], h4[i]) for i in range(3)) and all(not torch.equal(h1[i], h4[i]) for i in range(3, L + 1)))
        # end to end: run_seed on the toy
        cfgt = copy.deepcopy(cfg)
        cfgt["probe"]["seeds"] = [0, 1, 2]
        cfgt["predictor"]["batch_episodes"] = 16
        cfgt["intervention"]["fit_fraction"] = 0.5
        paths = {"results_dir": tmp / "res", "cache": tmp / "cache", "exp": tmp, "results004": tmp / "none"}
        m_ints, strengths = [2, 4, 6], [0.0, 0.5, 1.0]
        t0 = time.time()
        ts = 0
        todo = S.units(m_ints, strengths, [ts])
        diag = S.run_seed(cfgt, c2, paths, w, C, H, mean, std, dev, frames, pos, Yp, Etr, te, levels, nvis, ts, todo, m_ints, strengths)
        files = sorted(p.name for p in (paths["results_dir"] / "parts").glob("*.csv"))
        check("run_seed wrote 1 baseline + 3 blocks (incl. last block 6) x 2 strengths = 7 part files (baseline shared across blocks)", len(files) == 7 and files.count("base_s0.csv") == 1, str(files))
        mt = {p.name: p.stat().st_mtime_ns for p in (paths["results_dir"] / "parts").glob("*.csv")}
        todo2 = [u for u in S.units(m_ints, strengths, [ts]) if not S.unit_done(paths, u)]
        check("resume: nothing left to do after a full run", todo2 == [])
        S.run_seed(cfgt, c2, paths, w, C, H, mean, std, dev, frames, pos, Yp, Etr, te, levels, nvis, ts, todo2, m_ints, strengths)
        check("resume: part files untouched on a second call", mt == {p.name: p.stat().st_mtime_ns for p in (paths["results_dir"] / "parts").glob("*.csv")})
        # delete one unit and re-run only that one
        victim = S.part_path(paths, "unit", ts, 4, 0.5)
        victim.unlink()
        todo3 = [u for u in S.units(m_ints, strengths, [ts]) if not S.unit_done(paths, u)]
        S.run_seed(cfgt, c2, paths, w, C, H, mean, std, dev, frames, pos, Yp, Etr, te, levels, nvis, ts, todo3, m_ints, strengths)
        mt2 = {p.name: p.stat().st_mtime_ns for p in (paths["results_dir"] / "parts").glob("*.csv")}
        check("resume: only the deleted unit is recomputed", todo3 == [("unit", 0, 4, 0.5)] and victim.exists() and all(mt2[k] == mt[k] for k in mt if k != victim.name))
        # aggregate with the toy (single train seed, sd across seeds not needed)
        cfga = copy.deepcopy(cfgt)
        cfga["success"] = {"strength": 1.0, "min_seeds": 1, "slack": 1}
        cfga["_config_path"] = cfg["_config_path"]
        res = S.aggregate(cfga, None, c2, paths, m_ints, False, "config", strengths, [ts])
        a0 = res["blocks"]["2"]["strengths"]["1"][ts]
        check("aux A without 004 random-init rows -> 계산 불가 (no guess)", a0["aux_A_delta_vs_random_init"]["status"] == "계산 불가")
        check("aux B computable on the toy (self-check vs ALL rows < 1e-6)", a0["aux_B_common_variance_r2"]["status"] in ("관측됨", "관측 안 됨") and a0["aux_B_common_variance_r2"]["selfcheck_max_abs_diff"] < 1e-6,
              str({k: a0["aux_B_common_variance_r2"].get(k) for k in ("status", "m_star", "selfcheck_max_abs_diff")}))
        # block 6 (last block): detection verdict, no recovery
        b6 = res["blocks"]["6"]
        check("block 6: verdict computed (curve has depths 0..6; intervention only at depth 6)", b6["verdict"]["status"] in ("criterion_met", "criterion_not_met"), str(b6["verdict"].get("m_star_by_train_seed")))
        check("block 6: recovery observation is '해당 없음' (no later block)", isinstance(b6["aux_recovery_r2_minus_r2_at_m_int"], str) and b6["aux_recovery_r2_minus_r2_at_m_int"].startswith("해당 없음"))
        check("block 2: recovery observation is still computed (dict)", isinstance(res["blocks"]["2"]["aux_recovery_r2_minus_r2_at_m_int"], dict))
        check("block 6 curve: depth 0..5 equal the baseline, depth 6 from the unit", sorted(int(d) for d in b6["strengths"]["1"][ts]["curves"]["train"]) == list(range(7)))
        check("aux summary per block at success strength recorded for A and B", set(b6["aux_rules_at_success_strength"]) == {"A", "B"} and b6["aux_rules_at_success_strength"]["A"]["n_seeds_not_computable"] == 1)
        # aux A with a synthetic random-init reference (004 rows stand-in): random-init R2 = 0.05 at all depths -> delta curve = intervened curve - 0.05
        import csv as _csv
        p004 = tmp / "r004"
        p004.mkdir()
        from exp004 import depthlib as _DL
        brows = _DL.read_rows(S.part_path(paths, "base", ts))
        rnd = [dict(r, control="random_init", r2=0.05) for r in brows if r["level"] == "ALL" and r["module"] == "predictor"]
        _DL.write_rows(rnd, p004 / "depth_r2.csv", fields=S.FIELDS)
        paths2 = dict(paths, results004=p004)
        res_a = S.aggregate(cfga, None, c2, paths2, m_ints, False, "config", strengths, [ts])
        for m in m_ints:
            A = res_a["blocks"][str(m)]["strengths"]["1"][ts]["aux_A_delta_vs_random_init"]
            B_ = res_a["blocks"][str(m)]["strengths"]["1"][ts]["aux_B_common_variance_r2"]
            v = res_a["blocks"][str(m)]["verdict"]["status"]
            check("aux A (stand-in random-init 0.05) block %d: computed, delta = real - 0.05, same m* as main rule when constant offset" % m,
                  A["status"] in ("관측됨", "관측 안 됨") and abs(A["delta_curves"]["train"]["0"] - (res_a["blocks"][str(m)]["strengths"]["1"][ts]["curves"]["train"]["0"] - 0.05)) < 1e-9
                  and A["m_star"] == res_a["blocks"][str(m)]["strengths"]["1"][ts]["m_star"], str((A["status"], A["m_star"], B_["status"], B_["m_star"], v)))
            check("aux A/B block %d: within_m_int_plus_slack reported as bool when observed" % m, (A["within_m_int_plus_slack"] is None) == (A["status"] != "관측됨") and (B_["within_m_int_plus_slack"] is None) == (B_["status"] != "관측됨"))
        # keys mismatch -> 계산 불가
        _DL.write_rows([r for r in rnd if r["seed"] != "0" and str(r["seed"]) != "0"], p004 / "depth_r2.csv", fields=S.FIELDS)
        res_m = S.aggregate(cfga, None, c2, paths2, m_ints, False, "config", strengths, [ts])
        check("aux A with mismatching random-init keys -> 계산 불가", res_m["blocks"]["2"]["strengths"]["1"][ts]["aux_A_delta_vs_random_init"]["status"] == "계산 불가")
        check("primary verdict unchanged by aux rules", all(res_m["blocks"][str(m)]["verdict"] == res["blocks"][str(m)]["verdict"] for m in m_ints))
        for m in m_ints:
            v = res["blocks"][str(m)]["verdict"]
            check("toy: erasing at block %d with alpha=1 -> 004 rule fires at m* <= %d (criterion_met)" % (m, m + 1), v["status"] == "criterion_met", str(v.get("m_star_by_train_seed")))
            check("toy: baseline (alpha=0) shows no 0.2 drop (m* None)", res["blocks"][str(m)]["strengths"]["0"][ts]["m_star"] is None)
        c05 = res["blocks"]["2"]["strengths"]["0.5"][ts]["curves"]["train"]
        c10 = res["blocks"]["2"]["strengths"]["1"][ts]["curves"]["train"]
        c00 = res["blocks"]["2"]["strengths"]["0"][ts]["curves"]["train"]
        check("toy: R2 at depth m_int with the full-strength erasure is far below the baseline", c10["2"] < c00["2"] - 0.5, "(%.2f vs %.2f)" % (c10["2"], c00["2"]))
        check("toy FINDING (characterization): partial strength 0.5 only rescales the size component, a re-fitted linear probe is scale invariant -> R2 at m_int stays within 0.05 of the baseline",
              abs(c05["2"] - c00["2"]) < 0.05, "(0: %.3f, 0.5: %.3f, 1: %.3f)" % (c00["2"], c05["2"], c10["2"]))
        check("toy: depths below m_int are identical to the baseline at every strength", all(res["blocks"]["4"]["strengths"]["1"][ts]["curves"]["train"][str(d)] == res["blocks"]["4"]["strengths"]["0"][ts]["curves"]["train"][str(d)] for d in range(4)))
        check("results.json records m_int_config / m_int_used / m_int_overridden and interpretation null", res["m_int_used"] == [2, 4, 6] and res["m_int_overridden"] is False and "m_int_config" in res and res["interpretation"] is None)
        dk = diag["m2_a1_s0"]
        check("diagnostic: the fitted readout has ~no variance after full-strength erasure (std ratio < 0.05 on held-out train and on test episodes)",
              dk["readout_std_ratio_after_over_fit_pre_test"] < 0.05 and dk["readout_std_ratio_after_over_fit_pre_train_heldout"] < 0.05, str({k: round(v, 4) for k, v in dk.items() if v is not None}))
        # missing unit -> not_evaluable, not a guess
        S.part_path(paths, "unit", ts, 2, 1.0).unlink()
        res2 = S.aggregate(cfga, None, c2, paths, m_ints, False, "config", strengths, [ts])
        check("missing unit at the success strength -> that block not_evaluable (other block unaffected)", res2["blocks"]["2"]["verdict"]["status"] == "not_evaluable" and res2["blocks"]["4"]["verdict"]["status"] == "criterion_met")
        # B pure-numpy planted check: common-variance R2 of a planted collapse is detected; Var_all cancels in r2_common
        from exp009 import auxrules as AXr
        lab = AXr.common_variance(cfg2)
        lv = list(c2["data"]["levels"])[:3]
        check("auxrules.common_variance = log(area_ratio) of the 002 config", all(abs(lab[l] - np.log(cfg2["data_impl"]["area_ratio"][l])) < 1e-12 for l in lv))
        # direction fit uses train episodes only (never test)
        fit_ep = np.sort(np.random.RandomState(int(cfgt["intervention"]["fit_seed"]) + ts).permutation(Etr)[: int(round(0.5 * Etr))])
        check("direction-fit episodes are a subset of the train split (evaluation episodes never used)", fit_ep.max() < Etr and len(fit_ep) == Etr // 2)
    finally:
        M.P_TOK, M.D_VIS = old
        shutil.rmtree(tmp, ignore_errors=True)


def t_config(cfg, c4):
    print("D  config explicitness")
    check("m_int is a list default [1,2,3,4,5,6]; strengths include 0", cfg["intervention"]["m_int"] == [1, 2, 3, 4, 5, 6] and 0 in cfg["intervention"]["strengths"])
    check("seeds explicit (train seeds, probe seeds, fit seed, global)", cfg["predictor"]["train_seeds"] == [0, 1, 2] and cfg["probe"]["seeds"] == [0, 1, 2, 3, 4] and isinstance(cfg["intervention"]["fit_seed"], int) and cfg["seed"] == 0)
    check("decision rule equals 004 config (threshold 0.2, primary)", cfg["depth"]["drop_threshold"] == c4["depth"]["drop_threshold"] and dict(cfg["depth"]["primary"]) == dict(c4["depth"]["primary"]))
    check("confirmed 2026-11-10: method leace_scalar, strengths [0, 1.0]", cfg["intervention"]["method"] == "leace_scalar" and [float(a) for a in cfg["intervention"]["strengths"]] == [0.0, 1.0])
    check("success rule: strength 1.0, 2 seeds, slack 1", cfg["success"] == {"strength": 1.0, "min_seeds": 2, "slack": 1})


def t_metadata_preflight():
    print("E  metadata.pt preflight (no attempt consumed)")
    tmp = Path(tempfile.mkdtemp())
    try:
        import json as _j
        d1, d2 = tmp / "data" / "a060_train", tmp / "data" / "b_train"
        d1.mkdir(parents=True); d2.mkdir(parents=True)
        man = tmp / "train_manifest.json"
        man.write_text(_j.dumps({"episodes": [{"dir": str(d1), "name": "e0.mp4"}, {"dir": str(d1), "name": "e1.mp4"}, {"dir": str(d2), "name": "e2.mp4"}]}))
        res = tmp / "results"
        res.mkdir()
        bad = S.metadata_missing([("s0", man)])
        check("preflight: metadata.pt absent in 2 episode dirs -> 2 problems", len(bad) == 2 and "a060_train" in bad[0], str(bad))
        (d1 / "metadata.pt").write_bytes(b"x")
        check("preflight: one dir still missing -> 1 problem", len(S.metadata_missing([("s0", man)])) == 1)
        (d2 / "metadata.pt").write_bytes(b"x")
        check("preflight: all present -> no problem", S.metadata_missing([("s0", man)]) == [])
        check("preflight: missing manifest -> problem", len(S.metadata_missing([("s1", tmp / "nope.json")])) == 1)
        check("preflight does not create/modify attempts.json", not (res / "attempts.json").exists())
        src = Path(S.__file__).parent.parent.parent / "run.py"
        t = src.read_text()
        check("run.py: preflight runs before attempts_check and exits 4 without attempts_record", t.index("metadata_preflight") < t.index("B.attempts_check") and "sys.exit(4)" in t)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(cfg, c4, cfg2):
    for name, fn in (("A", t_eraser), ("B", lambda: t_bookkeeping(cfg)), ("D", lambda: t_config(cfg, c4)), ("E", t_metadata_preflight), ("C", lambda: t_torch(cfg, cfg2))):
        try:
            fn()
        except Exception:
            REC["fail"].append(name + " (exception)")
            traceback.print_exc()
    print("selftest summary: passed=%d failed=%d skipped=%d" % (len(REC["pass"]), len(REC["fail"]), len(REC["skipped"])))
    if REC["fail"]:
        print("FAILED:", REC["fail"])
    return 1 if REC["fail"] else 0
