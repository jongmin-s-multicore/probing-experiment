"""CPU selftest on SYNTHETIC data with known answers. Every number printed here is a synthetic number, NOT an experimental result.
Checks (see README section 8): persistence == encoder-token copy, retrained probe never sees test rows, slope/level alignment, seed-level CI formula, t quantile,
reproduction machinery (zero difference on identical inputs, nonzero when perturbed), 001/002/003 files unchanged, read-only input guard, loud failure of a
non-writable output folder, untrained-predictor / rebuild_aux toy checks (torch), --dry-run of every stage, Python 3.9 syntax, notebook JSON + cell compile."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from . import base as B
from . import lib as L

RES = {"pass": 0, "fail": 0, "skip": 0, "rows": []}


def check(name, cond, detail=""):
    ok = bool(cond)
    RES["pass" if ok else "fail"] += 1
    RES["rows"].append((name, ok, detail))
    print("[%s] %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return ok


def skip(name, why):
    RES["skip"] += 1
    RES["rows"].append((name, None, why))
    print("[SKIP] %s: %s" % (name, why), flush=True)


def guarded(fn):
    def w(*a, **k):
        import time
        t0 = time.time()
        try:
            return fn(*a, **k)
        except Exception:                                  # noqa: BLE001
            check(fn.__name__ + " (no exception)", False, traceback.format_exc().splitlines()[-1])
            traceback.print_exc()
        finally:
            print("[time] %s %.1fs" % (fn.__name__, time.time() - t0), flush=True)
    w.__name__ = fn.__name__
    return w


def tree_snapshot(roots):
    snap = {}
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        for p in sorted(r.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc"):
                st = p.stat()
                snap[str(p)] = (st.st_size, st.st_mtime_ns)
    return snap


# ------------------------------------------------------------------------------------------------ pure-numpy tests
@guarded
def t_stats():
    check("t_crit(df=2) = 0.95 / sqrt(2*0.975*0.025) (closed form, 4.3026527297)", abs(L.t_crit(2) - 0.95 / np.sqrt(2 * 0.975 * 0.025)) < 1e-12 and abs(L.t_crit(2) - 4.30265273) < 1e-7, "%.12f" % L.t_crit(2))
    try:
        from scipy import stats as sps
        worst = max(abs(L.t_ppf(0.975, df) - sps.t.ppf(0.975, df)) for df in (1.0, 2.0, 2.7, 3.0, 4.5, 10.0, 30.0))
        check("t_ppf matches scipy (df 1..30, incl. non-integer)", worst < 1e-7, "max diff %.2e" % worst)
        a, b = np.array([0.8, 0.9, 0.7]), np.array([0.1, 0.2, 0.05])
        d = L.diff_agg(a, b)
        sp = sps.ttest_ind(a, b, equal_var=False)
        se = float(d["welch_se"])
        tc = sps.t.ppf(0.975, float(d["welch_df"]))
        check("Welch CI matches scipy (df and bounds)", abs(float(d["welch_df"]) - float(sp.df)) < 1e-9 and abs(float(d["welch_ci_lo"]) - (a.mean() - b.mean() - tc * se)) < 1e-6,
              "df=%.4f" % float(d["welch_df"]))
    except ImportError:
        skip("t_ppf / Welch vs scipy", "scipy not installed")
    g = L.seed_agg(np.array([1.0, 2.0, 3.0]))
    check("seed_agg [1,2,3]: mean 2, sd 1, n 3, df 2", g["mean"] == 2.0 and abs(g["sd"] - 1.0) < 1e-12 and g["n"] == 3 and g["df"] == 2)
    half = L.t_crit(2) / np.sqrt(3)
    check("seed_agg CI = mean +- t(2) sd / sqrt(3)", abs(g["ci_lo"] - (2 - half)) < 1e-9 and abs(g["ci_hi"] - (2 + half)) < 1e-9, "[%.4f, %.4f]" % (g["ci_lo"], g["ci_hi"]))
    g2 = L.seed_agg(np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]]))
    check("seed_agg vector: second column sd 0", g2["sd"][1] == 0 and g2["ci_lo"][1] == 10.0)
    p = L.diff_agg(np.array([3.0, 4.0, 5.0]), np.array([1.0, 1.0, 2.0]))
    check("diff_agg paired by seed index: d = [2,3,3]", abs(float(p["paired_mean"]) - 8 / 3) < 1e-12 and abs(float(p["diff_mean"]) - 8 / 3) < 1e-12)


@guarded
def t_slope_alignment():
    levels = ["a026", "a040", "a060", "a080", "ind", "a125", "a160", "a200", "a248"]
    area = {"a026": 0.259, "a040": 0.399, "a060": 0.599, "a080": 0.801, "ind": 1.0, "a125": 1.249, "a160": 1.6, "a200": 2.0, "a248": 2.479}
    rng = np.random.RandomState(0)
    li = np.repeat(np.arange(9), 10)
    rng.shuffle(li)                                          # episode order is NOT sorted by level
    y = np.log(np.array([area[levels[i]] for i in li]))
    h = 0.5 * y + 0.3
    roles = {"all": np.ones(len(li), bool), "trained": np.isin(li, [2, 4, 6]), "unseen": ~np.isin(li, [2, 4, 6])}
    q = L.a_stats(h, y, li, 9, roles)
    check("slope/intercept recovered (episode level, all roles): 0.5 / 0.3", all(abs(q["slope_episode|" + r] - 0.5) < 1e-10 and abs(q["intercept_episode|" + r] - 0.3) < 1e-10 for r in roles))
    check("slope of level means = 0.5", all(abs(q["slope_levelmeans|" + r] - 0.5) < 1e-10 for r in roles))
    exp_true = np.log(np.array([area[l] for l in levels]))
    check("level labels aligned with level index (shuffled episodes)", np.allclose(q["per_level_true"], exp_true) and np.allclose(q["per_level_mean_pred"], 0.5 * exp_true + 0.3))
    bias = q["per_level_bias"]
    check("per-level bias = pred - true", np.allclose(bias, 0.5 * exp_true + 0.3 - exp_true))
    h2 = y + np.where(li == 3, 0.7, 0.0)                     # only level index 3 (a080) shifted
    q2 = L.a_stats(h2, y, li, 9, roles)
    check("a single shifted level appears only at its own index", abs(q2["per_level_bias"][3] - 0.7) < 1e-10 and np.abs(np.delete(q2["per_level_bias"], 3)).max() < 1e-10)
    hc = np.full(len(li), 0.2)
    q3 = L.a_stats(hc, y, li, 9, roles)
    check("constant prediction: slope 0, mean per level 0.2", abs(q3["slope_episode|all"]) < 1e-12 and np.allclose(q3["per_level_mean_pred"], 0.2))
    lz = np.repeat(1.0, 4)
    s, i = L.ols_line(lz, np.arange(4.0))
    check("ols_line with constant x -> NaN (no crash)", np.isnan(s) and np.isnan(i))


@guarded
def t_cumulative_and_mask():
    from . import stages_pre as SP
    a = np.ones((1, 60, 6))
    c = SP.cumulative_action(a)
    check("cumulative_action: frame f = 5f * (1,...)", c.shape == (1, 13, 6) and np.allclose(c[0, :, 0], 5 * np.arange(13)))
    a2 = np.zeros((1, 60, 6))
    a2[0, 7, 2] = 3.0
    c2 = SP.cumulative_action(a2)
    check("cumulative_action uses actions with index < 5f (action 7 first counted at frame 2)", c2[0, 1, 2] == 0 and c2[0, 2, 2] == 3.0)
    m = np.zeros((10, 20), bool)
    m[2:4, 5:9] = True
    cx, cy, bx, by = SP.mask_props(m)
    check("mask_props centroid / bbox centre", abs(cx - np.mean(np.arange(5, 9)) / 20) < 1e-12 and abs(cy - 2.5 / 10) < 1e-12 and abs(bx - 6.5 / 20) < 1e-12 and abs(by - 2.5 / 10) < 1e-12)
    check("mask_props empty -> NaN", all(np.isnan(v) for v in SP.mask_props(np.zeros((4, 4), bool))))


# ------------------------------------------------------------------------------------------------ paths / guards
@guarded
def t_guards(tmp):
    cfg = B.load_config()
    inp, work = Path(tmp) / "in", Path(tmp) / "work"
    inp.mkdir()
    os.environ["EXP006_INPUT_ROOT"], os.environ["EXP006_WORK_ROOT"] = str(inp), str(work)
    paths = B.resolve_paths(cfg)
    try:
        B.checked_write_bytes(paths, inp / "x.txt", b"a")
        check("write inside INPUT_ROOT refused", False)
    except B.InputWriteRefused:
        check("write inside INPUT_ROOT refused", not (inp / "x.txt").exists())
    for other in (B.exp002_dir() / "zz_test.txt", B.EXP_DIR.parent / "001_size_collapse_diagnosis" / "zz.txt", B.EXP_DIR.parent / "003_train_dist_dependence" / "zz.txt"):
        try:
            B.checked_write_bytes(paths, other, b"a")
            check("write into %s refused" % other.parent.name, False)
        except B.InputWriteRefused:
            check("write into %s refused" % other.parent.name, not other.exists())
    os.environ["EXP006_WORK_ROOT"] = str(inp / "sub")
    try:
        B.resolve_paths(cfg)
        check("WORK_ROOT inside INPUT_ROOT refused", False)
    except B.InputWriteRefused:
        check("WORK_ROOT inside INPUT_ROOT refused", True)
    # non-writable output (parent is a file) must fail loudly
    blocker = Path(tmp) / "blocker"
    blocker.write_text("x")
    os.environ["EXP006_WORK_ROOT"] = str(blocker / "child")
    p2 = B.resolve_paths(cfg)
    try:
        B.probe_output_dir(p2)
        check("unwritable WORK_ROOT -> OutputCheckError", False)
    except B.OutputCheckError as e:
        check("unwritable WORK_ROOT -> OutputCheckError", True, str(e)[:60])
    os.environ["EXP006_WORK_ROOT"] = str(work)
    p3 = B.resolve_paths(cfg)
    B.probe_output_dir(p3)
    n = B.write_json(p3, {"a": 1.0}, work / "t.json")
    check("checked write + read-back works", (work / "t.json").exists() and n > 0)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise ValueError("boom")
    try:
        B.run_with_retries(flaky, 3, "flaky")
        check("retry wrapper stops after 3 attempts", False)
    except B.StageFailed as e:
        check("retry wrapper stops after 3 attempts", calls["n"] == 3 and len(e.attempts) == 3)
    calls["n"] = 0

    def flaky2():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("once")
        return 7
    r, att = B.run_with_retries(flaky2, 3, "flaky2")
    check("retry wrapper succeeds on 2nd attempt", r == 7 and len(att) == 2)


# ------------------------------------------------------------------------------------------------ synthetic world
def make_world(tmp, cfg, cfg2, seed=0, n_tr=10, n_te=8, P=4, D=384):
    """Writes a 002-like Drive folder (cache, results, q1, per-run q2_cells.npz, 002 results.json) built with the 002 functions themselves, plus extract/ outputs."""
    from exp002 import evalgrid as EG
    from exp002 import probes as PR
    from exp002 import q2 as Q2
    from exp002 import stats as ST
    from . import extract as EX
    rng = np.random.RandomState(seed)
    levels = list(cfg2["data"]["levels"])
    area = cfg2["data_impl"]["area_ratio"]
    roles = {l: "trained" for l in cfg2["data"]["train_levels_b"]}
    for r, ls in cfg2["data"]["roles"].items():
        for l in ls:
            roles[l] = r
    rows = []
    for part, lvls, n in (("train", cfg2["data"]["train_levels_b"], n_tr), ("test", levels, n_te)):
        part_rows = []
        for l in lvls:
            for i in range(n):
                part_rows.append({"id": -1, "part": part, "level": l, "role": roles[l], "split": "%s_%s" % (l, part), "name": "ep%d.mp4" % i, "dir": "x", "rank_in_level": i,
                                  "nominal_log_area": float(np.log(area[l]))})
        if part == "test":
            order = rng.permutation(len(part_rows))                   # test rows NOT in level order
            part_rows = [part_rows[j] for j in order]
        rows += part_rows
    for i, r in enumerate(rows):
        r["id"] = i
    E = len(rows)
    Etr = sum(r["part"] == "train" for r in rows)
    y = np.array([r["nominal_log_area"] for r in rows])
    u = rng.randn(D)
    u /= np.linalg.norm(u)
    drift = rng.randn(13, D) * 0.2
    base = rng.randn(D) * 0.5
    tokens = np.zeros((E, 13, P, D), np.float32)
    for f in range(13):
        tokens[:, f] = base + (y[:, None, None] * u * 3.0) + drift[f] + rng.randn(E, P, D) * 0.3
    tokens = tokens.astype(np.float16)
    inp = Path(os.environ["EXP006_INPUT_ROOT"])
    cd = inp / "exp002_cache" / "q1_tokens"
    cd.mkdir(parents=True, exist_ok=True)
    np.save(cd / "tokens_dino.npy", tokens)
    np.save(cd / "frames_u8.npy", rng.randint(0, 255, size=(E, 13, 16, 16, 3)).astype(np.uint8))
    np.save(cd / "mask_feat.npy", (rng.rand(E, 13, 3) * 100 + 5).astype(np.float32))
    np.save(cd / "actions_raw.npy", rng.randn(E, 60, 6).astype(np.float32))
    B.write_json_raw(rows, cd / "episodes.json")
    B.write_json_raw({"encoder_fingerprint": "synthetic"}, cd / "meta.json")
    (cd / "DONE").write_text("ok\n")
    te_rows = [r for r in rows if r["part"] == "test"]
    B.write_json_raw(te_rows, inp / "results" / "q1" / "test_rows.json")
    Yp = EG.make_labels(cfg2, rows, np.zeros((E, 13, 3)), cfg["task"]["primary_target"])
    seeds = [int(s) for s in cfg["seeds"]["probe"]]
    tok32 = np.asarray(tokens, np.float32)
    mu_pool = PR.pool(tok32[:Etr][:, :12]).reshape(-1, D).mean(0)
    # q1 single-frame cells (as 002 q1.py computes them)
    q1cells = {}
    for f in range(1, 13):
        c, p = EG.fit_predict_cells(cfg2, "linear_pooled", tokens[:Etr], Yp[:Etr], tokens[Etr:], Yp[Etr:], [f], {"frame%d" % f: [f]}, seeds)
        q1cells["%s|linear_pooled|dino|f%d|frame%d" % (cfg["task"]["primary_target"], f, f)] = c["frame%d" % f]
    (inp / "results" / "q1").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(inp / "results" / "q1" / "cells.npz", **q1cells)
    grid, rowsd = {}, {}
    shrink = {"multi_60_ind_160": 0.8, "single_ind": 0.5}
    boot = ST.Bootstrap(EG.level_index(te_rows, levels), 50, 0)
    rm = EG.role_masks(te_rows)
    world = {"rows": rows, "Etr": Etr, "tokens": tokens, "Yp": Yp, "mu_pool": mu_pool, "shrink": shrink, "D": D, "P": P}
    rolls = {}
    for cond, sd, run in B.all_runs(cfg):
        s = shrink[cond] + 0.02 * sd
        R = (s * tok32[:, 1:13] + (1 - s) * mu_pool[None, None, None, :] + rng.randn(E, 12, P, D) * 0.05).astype(np.float16)
        rolls[run] = R
        cells, pms, digs = Q2.output_space(cfg2, tokens, Yp, Etr, R[Etr:], 0, 12)
        z = {}
        z.update(cells)
        z.update({"predmean|" + k: v for k, v in pms.items()})
        (inp / "results" / run).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(inp / "results" / run / "q2_cells.npz", **z)
        g = {}
        for k_, S_ in cells.items():
            g[k_] = {"unseen": boot.stat(S_, rm["unseen"], "r2")["point"], "trained": boot.stat(S_, rm["trained"], "r2")["point"]}
        grid["%s/seed%d" % (cond, sd)] = g
        prim = cfg["task"]["primary_target"]
        ce, q2_ = g["%s|linear_pooled|OUT|encoder|step12" % prim]["unseen"], g["%s|linear_pooled|OUT|rollout|step12" % prim]["unseen"]
        rowsd.setdefault(cond, {})[str(sd)] = {"r2_q1_ceiling": ce, "r2_q2": q2_, "drop": ce - q2_}
        # extraction outputs (as the GPU stage would write them)
        tr_roll = PR.pool(R[:Etr])
        te_roll = PR.pool(R[Etr:])
        te_unt = {}
        for v in cfg["extract"]["untrained"]["variants"]:
            for us in cfg["seeds"]["untrained_predictor"]:
                te_unt[(v, int(us))] = (mu_pool + rng.randn(E - Etr, 12, D) * 0.2).astype(np.float32)
        paths = B.resolve_paths(cfg)
        EX.write_run_outputs(paths, paths["extract"] / run, tr_roll, te_roll, te_unt,
                             {"run": run, "epoch": 15, "untrained": [{"variant": k[0], "seed": k[1]} for k in te_unt], "synthetic": True, "elapsed_s": 0.0})
    r002 = {"exploratory": {"q2_layer_time_grid_r2_point_estimates": grid},
            "decision_table": [{"id": "H_b:last_rollout_step", "result": {"per_training_seed": rowsd["multi_60_ind_160"]}},
                               {"id": "cond_a_vs_b:last_rollout_step", "result": {"per_training_seed_a": rowsd["single_ind"], "per_training_seed_b": rowsd["multi_60_ind_160"]}}]}
    B.write_json_raw(r002, inp / "results" / "results.json")
    world["rolls"] = rolls
    world["r002_path"] = inp / "results" / "results.json"
    return world


@guarded
def t_pipeline(tmp):
    from . import ctx as CX
    from . import stages_an as SA
    from . import stages_agg as SG
    from . import stages_pre as SP
    from . import figs
    from exp002 import directions as D
    from exp002 import evalgrid as EG
    cfg = B.load_config()
    cfg2 = B.load_cfg002(cfg)
    cfg2["probe_impl"]["n_bootstrap"] = 100                                  # selftest only (speed)
    inp, work = Path(tmp) / "in2", Path(tmp) / "work2"
    inp.mkdir()
    os.environ["EXP006_INPUT_ROOT"], os.environ["EXP006_WORK_ROOT"] = str(inp), str(work)
    cfg["base"]["results002"] = "does_not_exist.json"                      # force the INPUT_ROOT/results/results.json copy
    paths = B.resolve_paths(cfg)
    paths["logs"] = Path(tmp) / "logs"
    B.probe_output_dir(paths)
    world = make_world(tmp, cfg, cfg2)
    check("synthetic world built (SYNTHETIC numbers, not results)", True, "E=%d Etr=%d" % (len(world["rows"]), world["Etr"]))
    ctx = CX.Ctx(cfg, cfg2, paths)
    check("test rows are not in level order (alignment is tested by level index)", not (np.diff(ctx.lvl_idx) >= 0).all())
    SP.stage_pool(cfg, cfg2, paths, ctx)
    pooled = CX.load_pooled(paths)
    check("pooled tokens = fp32 mean over patches of the fp16 tokens", np.allclose(pooled, np.asarray(world["tokens"], np.float32).mean(2), atol=1e-6))
    # labels stage with an injected mask function
    SP.stage_labels(cfg, cfg2, paths, ctx, mask_fn=lambda f: f[..., 0] > 128)
    lz = np.load(Path(paths["cache"]) / "d_labels.npz")
    check("labels: shape (E,13,10) and cumulative action columns finite", lz["labels"].shape == (ctx.E, 13, 10) and np.isfinite(lz["labels"][:, :, 4:]).all())
    # transfer
    res, pm = SP.stage_transfer(cfg, cfg2, paths, ctx)
    tp = np.load(Path(paths["cache"]) / "transfer_pred.npz")
    # persistence == probe applied to a copy of the encoder tokens of the context frame (original 002 path with patch tokens)
    t = ctx.t
    sets = {"copy": (np.asarray(world["tokens"][ctx.Etr:, t], np.float32), ctx.label_te(1))}
    cells, pm2, _ = EG.fit_apply_cells(cfg2, "linear_pooled", world["tokens"][:ctx.Etr], ctx.Yp[:ctx.Etr], ctx.frames_tr, sets, ctx.probe_seeds)
    diffs = [np.abs(tp[SP.n_per(k)] - pm2["copy"]).max() for k in ctx.steps]
    check("persistence predictions == probe on copied encoder tokens (002 path), all steps", max(diffs) < 1e-6, "max diff %.2e" % max(diffs))
    check("persistence is the same prediction at every step", all(np.allclose(tp[SP.n_per(1)], tp[SP.n_per(k)]) for k in ctx.steps))
    # encoder step set == probe on tokens of frame k
    sets = {"e5": (np.asarray(world["tokens"][ctx.Etr:, 5], np.float32), ctx.label_te(5))}
    _, pm3, _ = EG.fit_apply_cells(cfg2, "linear_pooled", world["tokens"][:ctx.Etr], ctx.Yp[:ctx.Etr], ctx.frames_tr, sets, ctx.probe_seeds)
    check("encoder|5 == probe on tokens of frame 5", np.abs(tp[SP.n_enc(5)] - pm3["e5"]).max() < 1e-6)
    # repro: identical machinery -> ~0
    rp = SA.stage_repro(cfg, cfg2, paths, ctx, tp)
    mx = rp["summary"]["rollout_vs_results_json"]["max_abs_r2_diff"]
    check("repro on identical synthetic inputs: rollout R2 diff ~ 0", mx < 1e-6, "max |diff| %.2e (synthetic)" % mx)
    check("repro encoder-probe diff ~ 0", rp["summary"]["encoder_vs_results_json"]["max_abs_r2_diff"] < 1e-6, "%.2e" % rp["summary"]["encoder_vs_results_json"]["max_abs_r2_diff"])
    check("repro compares Drive cells with results.json (consistent for synthetic)", rp["summary"]["drive_cells_vs_results_json_max_abs_r2_diff"] < 1e-9)
    # repro detects a perturbation
    ex_bad = CX.load_extract(paths, "exp002_single_ind_s0")
    ex_bad["roll"] = ex_bad["roll"] + 0.3 * np.random.RandomState(1).randn(*ex_bad["roll"].shape).astype(np.float32)
    exs = {run: CX.load_extract(paths, run) for _, _, run in B.all_runs(cfg)}
    exs["exp002_single_ind_s0"] = ex_bad
    _, pm_bad = SP.stage_transfer(cfg, cfg2, paths, ctx, extracts=exs, out_name=None)
    rp_bad = SA.stage_repro(cfg, cfg2, paths, ctx, {k: v for k, v in pm_bad.items()})
    check("repro reports a difference when the extraction is perturbed (> 1e-3)", rp_bad["summary"]["rollout_vs_results_json"]["max_abs_r2_diff"] > 1e-3,
          "%.3f" % rp_bad["summary"]["rollout_vs_results_json"]["max_abs_r2_diff"])
    SA.stage_repro(cfg, cfg2, paths, ctx, tp)                                 # restore the part file with the unperturbed comparison
    # A: stored / reproduced, known shrink ratio
    A = SA.stage_a(cfg, cfg2, paths, ctx, transfer_pred=tp)
    for src in ("stored", "reproduced"):
        st = A["sources"][src]
        check("A[%s] computed" % src, st["status"] == "ok")
        ratios = []
        for cond, sh in world["shrink"].items():
            for sd in (0, 1, 2):
                q = st["per_run"][cond][str(sd)]
                r_roll = np.array(q["rollout"]["q"]["slope_episode|all"])
                r_enc = np.array(q["encoder"]["q"]["slope_episode|all"])
                ratios.append(np.abs(r_roll / r_enc - (sh + 0.02 * sd)).max())
        check("A[%s] slope(rollout)/slope(encoder) = known shrink factor (fp16 tolerance)" % src, max(ratios) < 0.05, "max dev %.3f" % max(ratios))
    s0 = A["sources"]["stored"]["per_run"]["single_ind"]["0"]["rollout"]["q"]["per_level_true"]
    check("A per-level true = log area of the level order (config)", np.allclose(np.array(s0)[0], [np.log(cfg2["data_impl"]["area_ratio"][l]) for l in cfg2["data"]["levels"]]))
    d_st = np.array(A["sources"]["stored"]["per_run"]["single_ind"]["0"]["rollout"]["q"]["per_level_mean_pred"])
    d_rp = np.array(A["sources"]["reproduced"]["per_run"]["single_ind"]["0"]["rollout"]["q"]["per_level_mean_pred"])
    check("A stored == reproduced level means (identical inputs)", np.abs(d_st - d_rp).max() < 1e-6, "%.2e" % np.abs(d_st - d_rp).max())
    # B
    Bq = SA.stage_b(cfg, cfg2, paths, ctx, tp)
    check("B: persistence / ceiling / untrained series have 12 steps", len(Bq["persistence"]["r2"]["unseen"]) == 12 and len(Bq["ceiling_encoder"]["r2"]["trained"]) == 12)
    check("B: first-frame == last-context-frame flag (num_hist=1)", Bq["first_frame_equals_last_context_frame"] is True)
    check("B: ceiling - persistence difference stored", len(Bq["persistence_minus_ceiling"]["r2"]["unseen"]) == 12)
    # C : no test leakage
    seen = []
    orig = D.ridge_fit

    def spy(X, y, groups, alphas, k, seed):
        seen.append((np.asarray(X).shape[0], len(np.unique(groups))))
        return orig(X, y, groups, alphas, k, seed)
    D.ridge_fit = spy
    try:
        C = SA.stage_c(cfg, cfg2, paths, ctx, pooled=pooled, extracts=exs_clean(paths, cfg), transfer_pred=tp)
    finally:
        D.ridge_fit = orig
    check("C: every ridge fit used exactly the %d training rows (no test rows)" % ctx.Etr, len(seen) > 0 and all(n == ctx.Etr and g == ctx.Etr for n, g in seen), "%d fits" % len(seen))
    rng = np.random.RandomState(5)
    Xtr = exs_clean(paths, cfg)["exp002_single_ind_s0"]["roll"][:ctx.Etr, 3]
    ytr = ctx.Yp[:ctx.Etr, 4]
    Xte1 = exs_clean(paths, cfg)["exp002_single_ind_s0"]["roll"][ctx.Etr:, 3]
    H1, a1 = SA.retrain_fit_apply(ctx, Xtr, ytr, Xte1, ctx.probe_seeds)
    H2, a2 = SA.retrain_fit_apply(ctx, Xtr, ytr, rng.randn(*Xte1.shape).astype(np.float32) * 9, ctx.probe_seeds)
    Hn, a3 = SA.retrain_fit_apply(ctx, Xtr, ytr, np.full_like(Xte1, np.nan), ctx.probe_seeds)
    check("C: alpha and fit identical whatever the test latents are (random / NaN test)", a1 == a2 == a3 and np.isnan(Hn).all() and not np.allclose(H1, H2))
    e = C["encoder_retrained_vs_002_q1_single_frame_cells"]["unseen"]
    dmax = max(abs(r["mine"] - r["q1_cell_f%d" % r["step"]]) for r in e)
    check("C: encoder retrained at frame k == 002 Q1 single-frame cell (same data, same fit path)", dmax < 1e-9, "max diff %.2e" % dmax)
    check("C: alpha chosen per probe seed stored", len(C["runs"]["exp002_single_ind_s0"]["retrained"]["alpha_per_probe_seed"]) == 12)
    # D
    Dq = SA.stage_d(cfg, cfg2, paths, ctx, pooled=pooled, labels=lz["labels"], extracts=exs_clean(paths, cfg))
    check("D: 10 properties, sets for encoder/persist/6 runs x 12 steps", len(Dq["properties"]) == 10 and len(Dq["per_property"]["cum_action_g1_x"]["sets"]) == 12 * 8)
    # D known answer: label that IS a linear function of tokens is recovered by the encoder probe; use cum_action label replaced
    lab2 = lz["labels"].copy()
    lab2[:, :, 4] = pooled[:, :, :5].sum(-1)                              # label = sum of the first 5 pooled coordinates
    Dq2 = SA.stage_d(cfg, cfg2, paths, ctx, pooled=pooled, labels=lab2, extracts=exs_clean(paths, cfg))
    r2e = Dq2["per_property"]["cum_action_g1_x"]["sets"]["encoder|3"]["r2|all"]
    check("D: label linear in the tokens is recovered by the encoder-trained probe (R2 > 0.9)", r2e > 0.9, "R2 %.3f (synthetic)" % r2e)
    # E
    Eq = SA.stage_e(cfg, cfg2, paths, ctx)
    v = np.array([Eq["per_run"]["exp002_single_ind_s%d" % s]["r2|rollout|unseen"][-1] for s in range(3)])
    agg = Eq["seed_summary"]["single_ind"]["r2|rollout|unseen"]
    half = L.t_crit(2) * v.std(ddof=1) / np.sqrt(3)
    check("E: seed mean/SD/CI of step-12 R2 equal the hand formula", abs(agg["mean"][-1] - v.mean()) < 1e-12 and abs(agg["sd"][-1] - v.std(ddof=1)) < 1e-12 and abs(agg["ci_hi"][-1] - (v.mean() + half)) < 1e-9 and agg["df"] == 2)
    cons = [abs(x) for r in Eq["consistency_with_results002_json"].values() for k_, x in r.items() if "minus" in k_]
    check("E: Drive-cell values reproduce the 002 results.json values (synthetic)", len(cons) > 0 and max(cons) < 1e-9, "max %.2e" % max(cons))
    dv = Eq["a_minus_b"]["values"]["drop|unseen"]
    va = np.array([Eq["per_run"]["exp002_single_ind_s%d" % s]["drop|unseen"][-1] for s in range(3)])
    vb = np.array([Eq["per_run"]["exp002_multi_60_ind_160_s%d" % s]["drop|unseen"][-1] for s in range(3)])
    check("E: (a)-(b) difference of mean drops", abs(dv["diff_mean"][-1] - (va.mean() - vb.mean())) < 1e-12 and abs(dv["paired_mean"][-1] - (va - vb).mean()) < 1e-12)
    # aggregate + figures
    res_json = SG.stage_aggregate(cfg, cfg2, paths, "2026-01-01 00:00:00")
    exp_copy = B.EXP_DIR / "results.json"
    check("aggregate wrote results.json to the experiment folder and WORK_ROOT", exp_copy.exists() and (Path(paths["work_root"]) / "results" / "results.json").exists())
    if exp_copy.exists():
        exp_copy.unlink()                                                     # synthetic results.json must not stay in the experiment folder
    text = json.dumps(res_json).lower()
    check("results.json: interpretation is null", res_json["interpretation"] is None)
    check("results.json: no pass/fail/p-value/threshold keys", not any(('"%s"' % w) in text for w in ("pass", "fail", "p_value", "threshold", "significant")))
    try:
        os.environ["MPLBACKEND"] = "Agg"      # unconditional (a kernel-inherited backend is invalid here)
        import matplotlib  # noqa: F401
        made, failed = figs.make_all(json.loads(json.dumps(res_json, default=float)), Path(tmp) / "figs", "(SYNTHETIC)")
        check("figures: 6 PNGs from synthetic results", len(made) == 6 and not failed, "made %d failed %s" % (len(made), failed))
    except ImportError:
        skip("figures", "matplotlib not installed")
    return paths, cfg, cfg2


def exs_clean(paths, cfg):
    from . import ctx as CX
    return {run: CX.load_extract(paths, run) for _, _, run in B.all_runs(cfg)}


# ------------------------------------------------------------------------------------------------ torch toy: extraction path
@guarded
def t_toy_extract(tmp):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        skip("toy extraction (torch)", "torch not installed")
        return
    from . import extract as EX
    from exp002 import model_io as M
    from exp002 import probes as PR
    cfg = B.load_config()
    cfg2 = B.load_cfg002(cfg)
    D_, P_ = 384, 4

    class Enc(nn.Module):
        emb_dim = 384

    class ToyPred(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(404, 404)
            self.pos = nn.Parameter(torch.randn(1, P_, 404))

        def forward(self, x):
            return x + 0.5 * torch.tanh(self.lin(x + self.pos[:, :x.shape[1]]))

    class Emb(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.l = nn.Linear(i, o)
            self.emb_dim = o

        def forward(self, x):
            return self.l(x)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(123)
            self.encoder = Enc()
            self.vis = nn.Linear(3, P_ * D_)
            self.proprio_encoder, self.action_encoder = Emb(1, 10), Emb(30, 10)
            self.predictor = ToyPred()
            self.num_proprio_repeat = self.num_action_repeat = 1

        def encode(self, obs, act):
            v = obs["visual"]                                              # (B,K,3,H,W)
            vis = self.vis(v.mean((-1, -2))).reshape(v.shape[0], v.shape[1], P_, D_)
            pro = self.proprio_encoder(obs["proprio"]).unsqueeze(2).expand(-1, -1, P_, -1)
            ac = self.action_encoder(act).unsqueeze(2).expand(-1, -1, P_, -1)
            return torch.cat([vis, pro, ac], -1)

        def predict(self, z):
            b, t, p, d = z.shape
            return self.predictor(z.reshape(b, t * p, d)).reshape(b, t, p, d)

        def replace_actions_from_z(self, z, act):
            ac = self.action_encoder(act).unsqueeze(2).expand(-1, -1, z.shape[2], -1)
            z[..., -10:] = ac
            return z

    torch.manual_seed(0)
    model = Toy().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def factory(seed, variant):
        torch.manual_seed(int(seed))
        mods = {"predictor": ToyPred().eval()}
        if variant == "pred_and_action_proprio":
            mods["proprio_encoder"], mods["action_encoder"] = Emb(1, 10).eval(), Emb(30, 10).eval()
        return mods

    rng = np.random.RandomState(0)
    E, Etr = 12, 6
    C = {"frames_u8": rng.randint(0, 255, size=(E, 13, 8, 8, 3)).astype(np.uint8), "actions_raw": rng.randn(E, 60, 6).astype(np.float32)}
    mean, std = torch.zeros(6), torch.ones(6)
    dev = torch.device("cpu")
    cfg["extract"]["batch_episodes"] = 4
    variants, useeds = cfg["extract"]["untrained"]["variants"], cfg["seeds"]["untrained_predictor"]
    untrained = EX.build_untrained(cfg, factory, model, variants, useeds)
    tr, _ = EX.run_pass(cfg, cfg2, model, C, (0, Etr), mean, std, dev, 0, 12, 1, None, 4, "train")
    te, un = EX.run_pass(cfg, cfg2, model, C, (Etr, E), mean, std, dev, 0, 12, 1, untrained, 4, "test")
    # manual reference with exp002 model_io (the 002 code path), whole set in one batch
    with torch.no_grad():
        x = M.frames_to_float(C["frames_u8"]).to(dev)
        acts = M.acts_from_raw(torch.as_tensor(C["actions_raw"]), mean, std)
        z_all = M.encode_batch(model, x, acts)
        ref = M.rollout(model, z_all, acts, 0, 12, 1)
        ref_pooled = PR.pool(ref.half().cpu().numpy())
    check("toy extract: trained rollout (pooled, batched) == 002 model_io.rollout reference", np.abs(np.concatenate([tr, te]) - ref_pooled).max() < 1e-5, "max diff %.2e" % np.abs(np.concatenate([tr, te]) - ref_pooled).max())
    check("toy extract: first latent step differs from copy of frame 0 (rollout is not persistence)", np.abs(te[:, 0] - PR.pool(z_all[Etr:, 0:1, :, :384].numpy())[:, 0]).max() > 1e-3)
    ref_pred_only = None
    with torch.no_grad(), EX.Swap(model, untrained[("pred_only", 0)]):
        ref_pred_only = PR.pool(M.rollout(model, z_all[Etr:], acts[Etr:], 0, 12, 1).half().numpy())
    check("toy extract: untrained pred_only (seed 0) == manual swap reference", np.abs(un[("pred_only", 0)] - ref_pred_only).max() < 1e-5)
    check("toy extract: model modules restored after Swap", model.predictor is not untrained[("pred_only", 0)]["predictor"])
    check("toy extract: untrained differs from trained", np.abs(un[("pred_only", 0)] - te).max() > 1e-3)
    check("toy extract: different untrained seeds differ", np.abs(un[("pred_only", 0)] - un[("pred_only", 1)]).max() > 1e-4)
    check("toy extract: variants differ (fresh action/proprio encoders change the rollout)", np.abs(un[("pred_only", 0)] - un[("pred_and_action_proprio", 0)]).max() > 1e-4)
    un2 = EX.build_untrained(cfg, factory, model, variants, useeds)
    with torch.no_grad(), EX.Swap(model, un2[("pred_only", 1)]):
        again = PR.pool(M.rollout(model, z_all[Etr:], acts[Etr:], 0, 12, 1).half().numpy())
    check("toy extract: same untrained seed -> same weights -> same rollout", np.abs(again - un[("pred_only", 1)]).max() < 1e-6)
    zr = EX.rebuild_aux(model, z_all, acts, {})
    check("rebuild_aux reproduces encode() with the trained encoders", float((zr - z_all).abs().max()) < 1e-6)
    zf = EX.rebuild_aux(model, z_all, acts, untrained[("pred_and_action_proprio", 0)])
    check("rebuild_aux leaves the visual part untouched, changes the aux part", float((zf[..., :384] - z_all[..., :384]).abs().max()) == 0 and float((zf[..., 384:] - z_all[..., 384:]).abs().max()) > 1e-4)
    # structure mismatch must raise
    class BadPred(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(404, 100)

    try:
        EX.build_untrained(cfg, lambda s, v: {"predictor": BadPred()}, model, ["pred_only"], [0])
        check("structure mismatch of the untrained predictor raises", False)
    except RuntimeError:
        check("structure mismatch of the untrained predictor raises", True)
    # action stats helper: normalise False -> zeros/ones
    m0, s0, n0 = EX.action_stats_local("/nonexistent", "/nonexistent", [0, 1, 2, 4, 5, 6], False)
    check("action_stats_local: normalize_action False -> zeros/ones without reading files", float(m0.abs().sum()) == 0 and float(s0.sum()) == 6 and n0 is False)


@guarded
def t_check_and_data_meta(tmp):
    try:
        import torch
    except ImportError:
        skip("check / data_meta (torch)", "torch not installed")
        return
    import tarfile
    from . import stages_pre as SP
    from . import extract as EX
    cfg = B.load_config()
    cfg2 = B.load_cfg002(cfg)
    inp, work, ds = Path(tmp) / "in3", Path(tmp) / "work3", Path(tmp) / "data3"
    inp.mkdir()
    os.environ["EXP006_INPUT_ROOT"], os.environ["EXP006_WORK_ROOT"], os.environ["DATASET_DIR"] = str(inp), str(work), str(ds)
    paths = B.resolve_paths(cfg)
    rng = np.random.RandomState(0)
    # synthetic metadata + archive (with a path-traversal member that must be ignored)
    arc_dir = Path(tmp) / "arc_src"
    metas = {}
    for sp in ("a060_train", "ind_train", "a160_train"):
        (arc_dir / "cloth_sim" / sp).mkdir(parents=True)
        m = [{"video_path": "/x/%s/episode_%05d.mp4" % (sp, i), "actions": torch.as_tensor(rng.randn(64, 8).astype(np.float32)), "length": 65} for i in range(5)]
        torch.save(m, arc_dir / "cloth_sim" / sp / "metadata.pt")
        metas[sp] = m
    inp.mkdir(exist_ok=True)
    with tarfile.open(inp / "cloth_sim_data.tar", "w") as tf:
        tf.add(str(arc_dir / "cloth_sim"), arcname="cloth_sim")
        evil = Path(tmp) / "evil.pt"
        evil.write_bytes(b"x")
        tf.add(str(evil), arcname="../evil/metadata.pt")
    arc_before = (inp / "cloth_sim_data.tar").stat()
    for _, _, run in B.all_runs(cfg):
        rd = inp / "train_runs" / "outputs" / run
        rd.mkdir(parents=True)
        eps = [{"split": sp, "name": "episode_%05d.mp4" % i, "dir": "/elsewhere", "group": sp} for sp in ("a060_train", "ind_train", "a160_train") for i in range(5)]
        B.write_json_raw({"episodes": eps}, rd / "train_manifest.json")
    r = SP.stage_data_meta(cfg, cfg2, paths)
    ok = all((ds / "cloth_sim" / sp / "metadata.pt").exists() for sp in metas) and not (Path(tmp) / "evil").exists()
    check("data_meta: metadata.pt of the 3 splits extracted from the (read only) archive; traversal member ignored", ok)
    a2 = (inp / "cloth_sim_data.tar").stat()
    check("data_meta: archive unchanged (size, mtime)", (a2.st_size, a2.st_mtime_ns) == (arc_before.st_size, arc_before.st_mtime_ns))
    check("data_meta: metadata_fields.json written with keys", "actions" in B.read_json(Path(paths["results"]) / "metadata_fields.json")["splits"]["ind_train"]["keys"])
    mean, std, norm = EX.action_stats_local(inp / "train_runs" / "outputs" / "exp002_single_ind_s0" / "train_manifest.json", paths["data_root"], [0, 1, 2, 4, 5, 6], True)
    allact = np.concatenate([m_["actions"].numpy()[:, [0, 1, 2, 4, 5, 6]] for sp in ("a060_train", "ind_train", "a160_train") for m_ in metas[sp]])
    check("action_stats_local: mean/std = torch.mean/std(+1e-6) of the manifest episodes' raw actions", norm and np.allclose(mean.numpy(), allact.mean(0), atol=1e-6) and np.allclose(std.numpy(), allact.std(0, ddof=1) + 1e-6, atol=1e-6))
    # check stage on a folder that lacks the cache / checkpoints: must run, list the missing files, write the manifest
    (inp / "results" / "q1").mkdir(parents=True, exist_ok=True)
    (inp / "results" / "q1" / "cells.npz").write_bytes(b"abc")
    res = SP.stage_check(cfg, cfg2, paths)
    check("check: runs on an incomplete input folder, lists missing files, hashes small files", "train_runs/outputs/exp002_single_ind_s0/checkpoints/model_latest.pth" in res["missing"]
          and res["files"]["results/q1/cells.npz"]["sha256"] == B.sha256_bytes(b"abc") and (Path(paths["results"]) / "input_manifest.json").exists())
    check("check: model_load_compat says the resume patch is not needed", res["model_load_compat"]["resume_patch_needed_for_load"] is False and res["model_load_compat"]["model_io_imports_train_py"] is False)
    check("check: wrote nothing into INPUT_ROOT", not any(p.name.endswith((".tmp", ".write_test")) for p in inp.rglob("*")))


# ------------------------------------------------------------------------------------------------ static checks
@guarded
def t_static():
    root = B.EXP_DIR
    bad = []
    for p in sorted((root / "src").rglob("*.py")) + [root / "run.py", root / "colab" / "build_notebook.py", root / "analysis" / "make_figures.py"]:
        if p.exists():
            try:
                compile(p.read_text(), str(p), "exec", dont_inherit=True)
            except SyntaxError as e:
                bad.append("%s: %s" % (p.name, e))
    check("all .py files compile on Python %s" % sys.version.split()[0], not bad, "; ".join(bad))
    if sys.version_info[:2] == (3, 9):
        check("selftest ran on Python 3.9", True)
    else:
        skip("Python 3.9 syntax", "interpreter is %s, not 3.9 (compile check above used this interpreter)" % sys.version.split()[0])
    nb = root / "colab" / "exp006_colab.ipynb"
    if nb.exists():
        d = json.loads(nb.read_text())
        errs = []
        for i, c in enumerate(d["cells"]):
            if c["cell_type"] == "code":
                try:
                    compile("".join(c["source"]), "cell%d" % i, "exec")
                except SyntaxError as e:
                    errs.append("cell %d: %s" % (i, e))
        check("notebook is valid JSON; %d cells; every code cell compiles" % len(d["cells"]), not errs and d.get("nbformat") == 4, "; ".join(errs))
        emb = "".join(d["cells"][3]["source"]) if len(d["cells"]) > 3 else ""
        check("notebook embeds the code snapshot", "EMBEDDED" in emb and len(emb) > 10000)
        check("notebook does not write to the 002 folder name EXP002_WORK for outputs", "exp006_work" in "".join("".join(c["source"]) for c in d["cells"]))
    else:
        skip("notebook checks", "colab/exp006_colab.ipynb not built yet")
    cfg = B.load_config()
    check("config: all seeds present", all(k in cfg["seeds"] for k in ("global", "probe", "untrained_predictor", "bootstrap", "retrain_probe")))
    check("config: rollout horizon 12, 002 config sha present", cfg["task"]["rollout_horizon"] == 12 and len(cfg["base"]["config002_sha256"]) == 64)
    cfg2 = B.load_cfg002(cfg)
    check("002 config sha256 equals the pinned value", cfg2["_config002_sha256"] == cfg["base"]["config002_sha256"])
    src = (B.exp002_dir() / "src" / "exp002" / "model_io.py").read_text()
    check("model_io.py does not import train.py (resume patch not needed for loading)", "import train" not in src and "from train" not in src)


@guarded
def t_dry_run(tmp):
    env = dict(os.environ, EXP006_INPUT_ROOT=str(Path(tmp) / "ri"), EXP006_WORK_ROOT=str(Path(tmp) / "rw"), PYTHONDONTWRITEBYTECODE="1")
    bad = []
    for st in ("check", "data_meta", "extract", "pool", "labels", "transfer", "repro", "analyze_a", "analyze_b", "analyze_c", "analyze_d", "analyze_e", "aggregate", "figures"):
        r = subprocess.run([sys.executable, str(B.EXP_DIR / "run.py"), st, "--dry-run"], env=env, capture_output=True, text=True, cwd=str(tmp))
        if r.returncode != 0 or "DRY-RUN plan" not in r.stdout:
            bad.append("%s rc=%s %s" % (st, r.returncode, (r.stderr or r.stdout)[-200:]))
    check("--dry-run of 14 stages exits 0 and prints a plan", not bad, "; ".join(bad))
    check("--dry-run wrote nothing into the work/input folders", not (Path(tmp) / "rw").exists() and not (Path(tmp) / "ri").exists())


def run():
    print("SYNTHETIC SELFTEST: numbers printed here are not experimental results. Python %s" % sys.version.split()[0])
    watch = [B.EXP_DIR.parent / "001_size_collapse_diagnosis", B.exp002_dir(), B.EXP_DIR.parent / "003_train_dist_dependence"]
    before = tree_snapshot(watch)
    tmp = tempfile.mkdtemp(prefix="exp006_selftest_")
    old_env = {k: os.environ.get(k) for k in ("EXP006_INPUT_ROOT", "EXP006_WORK_ROOT")}
    try:
        t_stats()
        t_slope_alignment()
        t_cumulative_and_mask()
        t_guards(tmp)
        t_pipeline(tmp)
        t_toy_extract(tmp)
        t_check_and_data_meta(tmp)
        t_static()
        t_dry_run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    after = tree_snapshot(watch)
    changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
    check("001, 002, 003 folders unchanged by the selftest (size + mtime of %d files)" % len(before), not changed, str(changed[:3]))
    leftovers = [p for p in (B.EXP_DIR / "results.json",) if p.exists()]
    check("selftest left no results.json in the experiment folder", not leftovers)
    print("\nSELFTEST SUMMARY: PASS %d  FAIL %d  SKIP %d   (synthetic data; not experimental results)" % (RES["pass"], RES["fail"], RES["skip"]))
    return RES["fail"] == 0
