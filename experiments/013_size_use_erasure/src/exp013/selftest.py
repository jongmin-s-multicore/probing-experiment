"""CPU self-test of experiment 013 on SYNTHETIC data and a toy world model (009's toy builders). Numbers printed here are NOT results.
A verdict rule (known answers, boundaries)   B bootstrap / Delta_use known answers   C erasers: norm matching, LEACE/level-mean properties, numpy == torch
D fit/eval separation (poison tests)         E stage-0 gate (planted level-dependent dynamics) and gate refusal in run.py (exit 5, no attempt)
F metric definitions vs an independent numpy computation   G stage-2 comparison   H planted-use end to end (fit -> eval -> Delta_use -> verdict -> aggregate)
I toy world model (compute_unit, determinism, batch invariance, erased extraction, readout drop)   J config / attempts / run.py order."""
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
from . import core as K
from . import lib as LB
from . import stages as S

REC = {"pass": [], "fail": [], "skipped": []}


def check(name, cond, detail=""):
    (REC["pass"] if cond else REC["fail"]).append(name)
    print("  [%s] %s %s" % ("ok" if cond else "FAIL", name, detail))


def _raises(f, exc=Exception):
    try:
        f()
    except exc:
        return True
    return False


# ------------------------------------------------------------------------------------------------ A
def mk(delta, lo, gap, hi=None):
    ci = {"point": delta, "ci_lo": lo, "ci_hi": (delta + 1.0 if hi is None else hi)}
    return {"delta_use": ci, "gap": {"point": gap, "ci_lo": gap, "ci_hi": gap}}


def t_verdict(cfg):
    print("A  verdict rule (confirmed 2026-11-11)")
    sc = cfg["success"]
    seeds = [0, 1, 2]
    J = lambda b, a: LB.judge_method(dict(zip(seeds, b)), dict(zip(seeds, a)), sc, seeds)
    a0 = [mk(0.01, -0.01, 0.4)] * 3
    good = [mk(0.10, 0.02, 0.4)] * 3
    check("config: min_seeds 2, 10% of gap, b>a required", (sc["min_seeds"], sc["min_fraction_of_persistence_gap"], sc["require_b_greater_than_a"]) == (2, 0.10, True))
    check("all three conditions met -> criterion_met", J(good, a0)["status"] == LB.MET)
    check("2 of 3 seeds with CI excluding 0 -> met (cond1)", J([mk(0.1, 0.02, 0.4), mk(0.1, 0.02, 0.4), mk(0.1, -0.01, 0.4)], a0)["status"] == LB.MET)
    check("1 of 3 seeds -> not met", J([mk(0.1, 0.02, 0.4), mk(0.1, -0.01, 0.4), mk(0.1, -0.01, 0.4)], a0)["status"] == LB.NOT_MET)
    check("CI lower bound exactly 0 does not exclude 0", J([mk(0.1, 0.0, 0.4)] * 3, a0)["cond1_seeds_positive_ci_excl_0"]["n"] == 0)
    check("point estimate <= 0 never counts even with lo > 0 (inconsistent input)", J([mk(-0.1, 0.02, 0.4)] * 3, a0)["cond1_seeds_positive_ci_excl_0"]["n"] == 0)
    check("Delta_use exactly 10% of the gap counts (>=)", J([mk(0.05, 0.01, 0.5)] * 3, [mk(0.0, -0.1, 0.5)] * 3)["cond2_mean_delta_use_ge_10pct_mean_gap"]["met"] is True)
    check("Delta_use just below 10% of the gap does not count", J([mk(0.0499, 0.01, 0.5)] * 3, [mk(0.0, -0.1, 0.5)] * 3)["status"] == LB.NOT_MET)
    check("gap <= 0 (model not better than persistence) -> cond2 false", J([mk(0.5, 0.2, 0.0)] * 3, a0)["cond2_mean_delta_use_ge_10pct_mean_gap"]["met"] is False)
    check("(b) Delta_use equal to (a) -> cond3 false (strict >)", J([mk(0.1, 0.02, 0.4)] * 3, [mk(0.1, 0.02, 0.4)] * 3)["cond3_b_greater_than_a"]["met"] is False)
    check("(b) larger than (a) -> cond3 true", J(good, [mk(0.0999, 0.02, 0.4)] * 3)["cond3_b_greater_than_a"]["met"] is True)
    check("missing (b) seed -> not_evaluable", J([good[0], good[1], None], a0)["status"] == LB.NOT_EVAL)
    check("NaN in (b) -> not_evaluable", J([good[0], good[1], mk(float("nan"), float("nan"), 0.4)], a0)["status"] == LB.NOT_EVAL)
    check("missing (a) seed -> not_evaluable", J(good, [a0[0], a0[1], None])["status"] == LB.NOT_EVAL)
    check("not-detected reading says it does NOT mean 'unused'", "사용하지 않는다는 뜻이 아님" in J([mk(0.1, -0.01, 0.4)] * 3, a0)["reading"])


# ------------------------------------------------------------------------------------------------ B
def t_bootstrap(cfg):
    print("B  bootstrap and Delta_use known answers")
    g = np.repeat(np.arange(3), [10, 20, 30])
    ix = LB.boot_indices(g, 200, 14000)
    ix2 = LB.boot_indices(g, 200, 14000)
    check("boot_indices reproducible for seed 14000, different for another seed", np.array_equal(ix, ix2) and not np.array_equal(ix, LB.boot_indices(g, 200, 14001)))
    check("stratified: every resample keeps the group sizes and draws only inside the group", all((g[ix[b]] == g).all() for b in range(200)))
    n = 40
    sel = {"none": np.full(n, 0.5), "size": np.full(n, 0.7), "rand": np.stack([np.full(n, 0.55)] * 4, 0), "persist": np.full(n, 1.0)}
    st = LB.use_stats(sel, LB.boot_indices(np.zeros(n, int), 500, 14000), 0.90)
    check("constants: motion_norm none 0.5, size 0.7, rand 0.55; increases 0.2 / 0.05; Delta_use 0.15; gap 0.5",
          abs(st["motion_norm_none"]["point"] - 0.5) < 1e-12 and abs(st["increase_size"]["point"] - 0.2) < 1e-12 and abs(st["increase_rand_mean"]["point"] - 0.05) < 1e-12
          and abs(st["delta_use"]["point"] - 0.15) < 1e-12 and abs(st["gap"]["point"] - 0.5) < 1e-12, str(st["delta_use"]))
    check("constants: CI collapses to the point", abs(st["delta_use"]["ci_lo"] - 0.15) < 1e-12 and abs(st["delta_use"]["ci_hi"] - 0.15) < 1e-12)
    rng = np.random.RandomState(0)
    for n, name in ((200, "n=200"), (2000, "n=2000")):
        base = rng.rand(n)
        sel = {"none": 0.5 * np.ones(n), "size": 0.5 + 0.3 * rng.rand(n), "rand": np.stack([0.5 + 0.3 * rng.rand(n) for _ in range(5)], 0), "persist": np.ones(n)}
        s2 = LB.use_stats(sel, LB.boot_indices(np.zeros(n, int), 2000, 14000), 0.90)
        w = s2["delta_use"]["ci_hi"] - s2["delta_use"]["ci_lo"]
        check("noise toy %s: point inside the 90%% CI; no true effect -> |point| within one CI width of 0" % name, s2["delta_use"]["ci_lo"] <= s2["delta_use"]["point"] <= s2["delta_use"]["ci_hi"] and abs(s2["delta_use"]["point"]) < w, "(point %.4f, width %.4f)" % (s2["delta_use"]["point"], w))
        if n == 200:
            w200 = w
    check("CI narrows with more episodes", w < w200)
    sel = {"none": 0.5 * np.ones(100), "size": 0.8 + 0.05 * rng.rand(100), "rand": np.stack([0.5 + 0.05 * rng.rand(100) for _ in range(5)], 0), "persist": np.ones(100)}
    s3 = LB.use_stats(sel, LB.boot_indices(np.zeros(100, int), 2000, 14000), 0.90)
    check("planted effect: CI excludes 0", s3["delta_use"]["ci_lo"] > 0.2)


# ------------------------------------------------------------------------------------------------ C
def toy_dims():
    return 9, 16, 4     # P, dz, nact


class patched:
    """Temporarily set exp002.model_io P_TOK / D_VIS to the toy sizes (core.py reads them at call time)."""

    def __enter__(self):
        self.old = (K.M.P_TOK, K.M.D_VIS)
        K.M.P_TOK, K.M.D_VIS = toy_dims()[:2]

    def __exit__(self, *a):
        K.M.P_TOK, K.M.D_VIS = self.old


def t_erasers(cfg):
    print("C  erasers: norm matching, properties, numpy == torch")
    import torch
    P, dz, _ = toy_dims()
    rng = np.random.RandomState(0)
    n, nlev = 400, 9
    u = rng.randn(dz)
    y = rng.randn(n)
    pm = (y[:, None] * u[None] + 0.5 * rng.randn(n, dz))[:, None, :].repeat(13, 1) + 0.05 * rng.randn(n, 13, dz)
    E = LB.fit_leace(pm, y, np.arange(n), 1e-3)
    X = pm[:, :12].reshape(-1, dz)
    Xe = X + LB.leace_disp_np(E, X)
    yy = np.repeat(y, 12)
    cov = np.abs(np.cov(Xe.T, yy)[:-1, -1]).max()
    cov0 = np.abs(np.cov(X.T, yy)[:-1, -1]).max()
    check("LEACE (numpy displacement): covariance of every coordinate with y ~ 0 after erasure", cov < 1e-6 * cov0 + 1e-9, "(%.2g vs %.2g)" % (cov, cov0))
    check("LEACE: fitted readout a.x is constant after erasure", float((Xe @ E["a"]).std()) < 1e-9 * max(1.0, float((X @ E["a"]).std())), "(%.2g)" % float((Xe @ E["a"]).std()))
    check("LEACE numpy displacement == 009 apply_eraser_np(alpha=1) - x", np.allclose(Xe, ER.apply_eraser_np(E, X, 1.0)))
    R = LB.random_dirs(20, dz, 13000)
    check("random directions: 20 unit vectors, reproducible with seed 13000, different with another seed", np.allclose(np.linalg.norm(R, axis=1), 1) and np.array_equal(R, LB.random_dirs(20, dz, 13000)) and not np.array_equal(R, LB.random_dirs(20, dz, 13001)))
    Sg = LB.random_signs(20, nlev, P, 13000)
    check("random signs: +-1, shape (20, 9, P), reproducible", set(np.unique(Sg)) == {-1.0, 1.0} and Sg.shape == (20, nlev, P) and np.array_equal(Sg, LB.random_signs(20, nlev, P, 13000)))
    # level means
    Mn = rng.randn(nlev, P, dz)
    D = LB.level_mean_shifts(Mn)
    check("level_mean_match: Mn + D is the overall mean for every level and patch (all levels end up with the same mean)", np.allclose(Mn + D, Mn.mean(0, keepdims=True)))
    check("level_mean_match: shifts sum to zero over levels", np.allclose(D.sum(0), 0))
    # torch Eraser
    er = K.Eraser(E, D, R, Sg, torch.device("cpu"))
    N = 50
    z = torch.as_tensor(rng.randn(N, P, dz), dtype=torch.float32)
    lvl = torch.as_tensor(rng.randint(0, nlev, N))
    pre = er.prepare(z, lvl)
    d_l = er.displacement("leace:size", pre, lvl).numpy()
    ref = np.stack([LB.leace_disp_np(E, z[i].numpy().astype(np.float64)) for i in range(N)])
    check("torch LEACE displacement == numpy reference", np.abs(d_l - ref).max() < 1e-4, "(%.2g)" % np.abs(d_l - ref).max())
    d_m = er.displacement("lmm:size", pre, lvl).numpy()
    ref = np.stack([LB.lmm_disp_np(D, int(lvl[i])) for i in range(N)])
    check("torch level-mean displacement == numpy reference", np.abs(d_m - ref).max() < 1e-5)
    worst = {"leace": 0.0, "lmm": 0.0}
    rel = {"leace": 0.0, "lmm": 0.0}
    for k in range(20):
        for m in ("leace", "lmm"):
            ds = torch.as_tensor(er.displacement("%s:size" % m, pre, lvl)).norm(dim=-1)
            dr = er.displacement("%s:rand:%d" % (m, k), pre, lvl).norm(dim=-1)
            worst[m] = max(worst[m], float((ds - dr).abs().max()))
            rel[m] = max(rel[m], float(((ds - dr).abs() / (ds + 1e-12)).max()))
    check("norm matching: per-token displacement norm of every random direction == size erasure (LEACE and level-mean-match)", worst["leace"] < 1e-4 and worst["lmm"] < 1e-5, "(max abs diff %.2g / %.2g)" % (worst["leace"], worst["lmm"]))
    dsr = er.displacement("leace:rand:3", pre, lvl).numpy()
    check("random control is NOT the size direction (cosine with the size displacement < 0.9)", float(np.abs((dsr * d_l).sum() / (np.linalg.norm(dsr) * np.linalg.norm(d_l)))) < 0.9)
    check("level-mean random control: displacement is not the size shift", np.abs(er.displacement("lmm:rand:0", pre, lvl).numpy() - d_m).max() > 1e-3)


# ------------------------------------------------------------------------------------------------ toy rows and planted data
def toy_rows(cfg2, n_train=16, n_test=10):
    from exp004 import selftest as T4
    c2 = T4.small_cfg2(cfg2, n_train=n_train, n_test=n_test)
    c2["probe"]["kinds"] = ["linear_pooled"]
    return c2, T4.make_rows(c2)


def planted_tokens(rows, P, dz, seed=0, g=0.3):
    """tokens (E,13,P,dz): z_t = base + 2 s u + 0.1 noise + t * g s v ; real change per step Delta = g s v + 0.02 noise. s = nominal log area of the episode's level."""
    rng = np.random.RandomState(seed)
    s = np.array([r["nominal_log_area"] for r in rows])
    u = rng.randn(dz)
    u /= np.linalg.norm(u)
    v = rng.randn(dz)
    v -= (v @ u) * u
    v /= np.linalg.norm(v)
    base = rng.randn(P, dz) * 0.5
    E = len(rows)
    T = np.zeros((E, 13, P, dz))
    z0 = base[None] + 2.0 * s[:, None, None] * u[None, None] + 0.1 * rng.randn(E, P, dz)
    T[:, 0] = z0
    for t in range(12):
        T[:, t + 1] = T[:, t] + g * s[:, None, None] * v[None, None] + 0.02 * rng.randn(E, P, dz)
    return T.astype(np.float32), {"u": u, "v": v, "base": base, "g": g, "c1": 2.0}


def oracle_fn(info, dz, dev=None):
    import torch
    u, v, base = (torch.as_tensor(info[k], dtype=torch.float32) for k in ("u", "v", "base"))
    g, c1 = info["g"], info["c1"]

    def fn(zin):                                           # zin (N,1,P,D)
        zv = zin[..., :dz]
        s_hat = ((zv - base) @ u).mean(2) / c1             # (N,1) size read from the patch mean
        out = zin.clone()
        out[..., :dz] = zv + g * s_hat[..., None, None] * v
        return out
    return fn


def copy_fn(dz):
    return lambda zin: zin.clone()


def build_z_all(tokens, ids, nact, rng):
    import torch
    t = torch.as_tensor(tokens[ids])
    za = torch.as_tensor(rng.randn(len(ids), 13, 1, nact).astype(np.float32)).expand(-1, -1, t.shape[2], -1)
    return torch.cat([t, za], -1)


def synth_unit(cfg, cfg2, rows, Etr, lvl, F, tokens, ts, fns, ep_sel, nact, bs=8):
    """Like stages.compute_unit but with synthetic z_all and given predict fns (no world model). Returns (arrs, meta)."""
    import torch
    dev = torch.device("cpu")
    er = K.Eraser(F["leace"][ts], F["D"], LB.random_dirs(20, tokens.shape[-1], cfg["erasure"]["random_dir_seed"]), LB.random_signs(20, len(F["levels"]), tokens.shape[2], cfg["erasure"]["random_dir_seed"]), dev)
    keys = K.config_keys(20)
    acc = {k: {key: [] for key in keys} for k in fns}
    dh = {k: [] for k in fns}
    dr = []
    rng = np.random.RandomState(1)
    for s in range(0, len(ep_sel), bs):
        sel = ep_sel[s:s + bs]
        ids = Etr + sel
        z_all = build_z_all(tokens, ids, nact, rng)
        o, d, r = K.eval_batch(z_all, torch.as_tensor(lvl[ids]), fns, er, keys)
        for k in fns:
            for key in keys:
                acc[k][key].append(o[k][key])
            dh[k].append(d[k])
        dr.append(r)
    arrs = {"metrics_%s" % k: np.stack([np.concatenate(acc[k][key], 0) for key in keys], 0) for k in fns}
    arrs.update({"dhat_%s" % k: np.concatenate(dh[k], 0).astype(np.float32) for k in fns})
    arrs["dreal"] = np.concatenate(dr, 0).astype(np.float32)
    meta = S.unit_meta(cfg, rows, Etr, F["levels"], lvl, F["fit_mask"], F["eval_mask"], ep_sel, keys)
    return arrs, meta


def synth_fit(cfg, cfg2, rows, tokens, seeds):
    rows_, Etr, levels, lvl = S.rows_info(cfg2, {"rows": rows})
    pm = tokens.mean(2)
    F = S.compute_fit(cfg, cfg2, rows, Etr, lvl, pm, lambda ids: tokens[np.sort(ids)], seeds, tokens.shape[2], tokens.shape[3])
    return F, Etr, lvl


# ------------------------------------------------------------------------------------------------ D
def t_separation(cfg, cfg2):
    print("D  fit / evaluation separation")
    c2, rows = toy_rows(cfg2)
    P, dz, nact = toy_dims()
    tokens, info = planted_tokens(rows, P, dz)
    rows_, Etr, levels, lvl = S.rows_info(c2, {"rows": rows})
    fit_ep = LB.fit_episodes(cfg, 0, Etr)
    check("LEACE fit episodes follow the 009 rule and are train rows only", np.array_equal(fit_ep, np.sort(np.random.RandomState(9000).permutation(Etr)[: int(round(0.5 * Etr))])) and fit_ep.max() < Etr and len(fit_ep) == Etr // 2)
    check("fit episodes depend on the train seed (9000 + seed)", not np.array_equal(fit_ep, LB.fit_episodes(cfg, 1, Etr)))
    rows_test = rows[Etr:]
    fm, em = LB.ood_halves(cfg, rows_test, levels, cfg["data"]["judge_levels"])
    lv_t = np.array([r["level"] for r in rows_test])
    ood = ~np.isin(lv_t, cfg["data"]["judge_levels"])
    check("OoD halves: disjoint, half/half per OoD level, train levels all evaluated and never fit", not (fm & em).any() and all(fm[lv_t == l].sum() == 5 and em[lv_t == l].sum() == 5 for l in levels if l not in cfg["data"]["judge_levels"])
          and em[~ood].all() and not fm[~ood].any())
    fm2, em2 = LB.ood_halves(cfg, rows_test, levels, cfg["data"]["judge_levels"])
    check("OoD halves are reproducible (seed 9000)", np.array_equal(fm, fm2) and np.array_equal(em, em2))
    F, Etr, lvl = synth_fit(cfg, c2, rows, tokens, [0])
    # poison 1 (LEACE): garbage in every train row that is not a fit episode and in all test rows
    not_fit_train = np.setdiff1d(np.arange(Etr), fit_ep)
    test_ids = Etr + np.arange(len(rows_test))
    used_test = Etr + np.where(fm)[0]
    p1 = tokens.copy()
    p1[not_fit_train] = 1e3
    p1[test_ids] = -1e3
    Fp1 = S.compute_fit(cfg, c2, rows, Etr, lvl, p1.mean(2), lambda ids: tokens[np.sort(ids)], [0], P, dz)
    check("poison test (LEACE): garbage in every non-fit train row and every test row does not change the eraser", all(np.array_equal(F["leace"][0][k], Fp1["leace"][0][k]) for k in ("mu", "c", "a")))
    # poison 2 (level means): garbage in every test row except the OoD fit half (train rows of the train levels are legitimately used)
    p2 = tokens.copy()
    p2[np.setdiff1d(test_ids, used_test)] = -1e3
    seen = []

    def tok_fn(ids):
        seen.extend(np.asarray(ids).tolist())
        return p2[np.sort(ids)]
    Fp2 = S.compute_fit(cfg, c2, rows, Etr, lvl, tokens.mean(2), tok_fn, [0], P, dz)
    check("poison test (level means): garbage in every test row except the OoD fit half does not change the level-mean shifts", np.array_equal(F["D"], Fp2["D"]))
    check("level-mean fit read only: train rows (train levels) + the OoD fit-half test rows (never the OoD eval half or train-level test rows)", set(seen) <= set(range(Etr)) | set(used_test.tolist()) and not (set(seen) & set((Etr + np.where(em & ood)[0]).tolist())) and not (set(seen) & set((Etr + np.where(~ood)[0]).tolist())))
    check("level-mean fit counts: train levels = n_train rows of the level, OoD levels = fit half", F["counts"][levels.index("ind")] == 16 and F["counts"][levels.index("a026")] == 5)


# ------------------------------------------------------------------------------------------------ E
def synth_gate_data(same, seed=0):
    rng = np.random.RandomState(seed)
    dz, da = 6, 4
    levels = ["a060", "ind", "a160"]
    rows = []
    for part, n in (("train", 40), ("test", 20)):
        for l in levels:
            for i in range(n):
                rows.append({"level": l, "part": part, "rank_in_level": i})
    E = len(rows)
    W = {l: (rng.randn(dz, dz + da) * 0.35) for l in levels}
    if same:
        W = {l: W[levels[0]] for l in levels}
    pm = np.zeros((E, 13, dz))
    act = rng.randn(E, 12, da)
    pm[:, 0] = rng.randn(E, dz)
    for e, r in enumerate(rows):
        for t in range(12):
            pm[e, t + 1] = pm[e, t] + 0.3 * np.tanh(W[r["level"]] @ np.r_[pm[e, t], act[e, t]]) + 0.02 * rng.randn(dz)
    return pm, act, rows, levels


def t_gate(cfg, cfg2):
    print("E  stage-0 gate and its refusal")
    alphas = [0.1, 1.0, 10.0, 100.0]
    pm, act, rows, levels = synth_gate_data(False)
    g = LB.stage0_gate(pm, act, rows, levels, alphas, 3, cfg["stage0"]["ratio_threshold"], cfg["stage0"]["min_pairs"])
    check("gate: level-dependent dynamics -> 6 directed pairs, passed", g["n_pairs"] == 6 and g["passed"] and g["n_pairs_pass"] >= 4, "(pass %d/6, ratios %s)" % (g["n_pairs_pass"], [round(p["ratio_other_over_same"], 2) for p in g["pairs"]]))
    pm, act, rows, levels = synth_gate_data(True)
    g2 = LB.stage0_gate(pm, act, rows, levels, alphas, 3, cfg["stage0"]["ratio_threshold"], cfg["stage0"]["min_pairs"])
    check("gate: identical dynamics in all levels -> NOT passed (< 4 pairs with ratio >= 1.1)", (not g2["passed"]) and g2["n_pairs_pass"] < 4, "(pass %d/6, ratios %s)" % (g2["n_pairs_pass"], [round(p["ratio_other_over_same"], 2) for p in g2["pairs"]]))
    check("gate: threshold boundary (>= 1.1 counts)", LB.stage0_gate(*synth_gate_data(False)[:3], synth_gate_data(False)[3], alphas, 3, 0.0, 6)["passed"] and not LB.stage0_gate(*synth_gate_data(False)[:3], synth_gate_data(False)[3], alphas, 3, 1e9, 1)["passed"])
    check("gate: ridge alpha chosen by episode groups (alpha from the grid), ratio uses the SAME regressor on same vs other level", g["alpha_by_level"]["ind"] in alphas and all(abs(p["ratio_other_over_same"] - p["mse_other"] / p["mse_same"]) < 1e-12 for p in g["pairs"]))
    f = LB.kfold_groups(30, 3, 0)
    check("kfold_groups: every group in exactly one of 3 balanced folds", set(f) == {0, 1, 2} and all((f == j).sum() == 10 for j in range(3)))
    # refusal in run.py
    run_py = Path(__file__).resolve().parents[2] / "run.py"
    src = run_py.read_text()
    check("run.py: gate check (exit 5) comes before the preflight (exit 4) and before attempts_check / attempts_record", src.index("gate_status") < src.index("input_preflight") < src.index("B.attempts_check") and "B.EXIT_GATE" in src and B.EXIT_GATE == 5)
    tmp = Path(tempfile.mkdtemp())
    try:
        env = dict(os.environ, DINO_TRAIN=str(tmp / "train_runs"), DINO_WORK=str(tmp / "work"), EXP013_CACHE=str(tmp / "cache"), MPLBACKEND="Agg")
        env.pop("EXP002_CACHE", None)
        sha = B.gate_config_sha(cfg)

        def run(stage, res, *extra):
            p = subprocess.run([sys.executable, str(run_py), stage, "--results-dir", str(res)] + list(extra), capture_output=True, text=True, env=env, timeout=600)
            return p.returncode, p.stderr

        res = tmp / "r1"
        rc, err = run("measure", res)
        check("measure with NO stage0.json: exit 5, no attempts.json", rc == 5 and not (res / "attempts.json").exists() and "GATE" in err, "(rc=%s)" % rc)
        res.mkdir(parents=True, exist_ok=True)
        B.write_json({"passed": False, "n_pairs_pass": 2, "n_pairs": 6, "ratio_threshold": 1.1, "min_pairs": 4, "gate_config_sha256": sha}, res / "stage0.json")
        rc, err = run("measure", res)
        check("measure with a FAILED gate: exit 5, message names the pairs, no attempts.json", rc == 5 and "NOT PASSED" in err and not (res / "attempts.json").exists(), "(rc=%s)" % rc)
        rc, err = run("readout", res)
        check("readout with a FAILED gate: exit 5", rc == 5 and not (res / "attempts.json").exists())
        B.write_json({"passed": True, "n_pairs_pass": 5, "n_pairs": 6, "ratio_threshold": 1.1, "min_pairs": 4, "gate_config_sha256": "stale"}, res / "stage0.json")
        rc, err = run("measure", res)
        check("measure with a STALE gate record (other gate definition): exit 5", rc == 5 and "stale" in err)
        B.write_json({"passed": True, "n_pairs_pass": 5, "n_pairs": 6, "ratio_threshold": 1.1, "min_pairs": 4, "gate_config_sha256": sha}, res / "stage0.json")
        rc, err = run("measure", res)
        check("measure with a PASSED gate goes on to the preflight (inputs missing here -> exit 4, no attempts.json)", rc == 4 and "PREFLIGHT" in err and not (res / "attempts.json").exists(), "(rc=%s)" % rc)
        B.write_json({"passed": False, "n_pairs_pass": 2, "n_pairs": 6, "ratio_threshold": 1.1, "min_pairs": 4, "gate_config_sha256": sha}, res / "stage0.json")
        rc, err = run("measure", res, "--override-gate")
        check("--override-gate with a failed gate passes the gate (then exit 4 on missing inputs) and is logged", rc == 4 and "GATE (exit 5" not in err)
        rc, err = run("measure", res, "--dry-run")
        check("dry-run with a failed gate does not exit 5", rc == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    s0 = S.pilot_reasons(cfg, B.default_results_dir(), False, False, None, True)
    check("pilot_reasons: gate override listed", any("override-gate" in r for r in s0))


# ------------------------------------------------------------------------------------------------ F
def t_metrics(cfg, cfg2):
    print("F  metric definitions against an independent numpy computation")
    import torch
    c2, rows = toy_rows(cfg2)
    P, dz, nact = toy_dims()
    D = dz + nact
    tokens, info = planted_tokens(rows, P, dz)
    F, Etr, lvl = synth_fit(cfg, c2, rows, tokens, [0])
    dev = torch.device("cpu")
    R = LB.random_dirs(20, dz, 13000)
    Sg = LB.random_signs(20, 9, P, 13000)
    er = K.Eraser(F["leace"][0], F["D"], R, Sg, dev)
    rng = np.random.RandomState(3)
    Wm = rng.randn(D, D) * 0.1
    Wt = torch.as_tensor(Wm, dtype=torch.float32)
    fn = lambda zin: zin @ Wt
    ids = Etr + np.arange(5)
    z_all = build_z_all(tokens, ids, nact, rng)
    keys = ["none", "leace:size", "leace:rand:2", "lmm:size", "lmm:rand:5"]
    out, dhat, dreal = K.eval_batch(z_all, torch.as_tensor(lvl[ids]), {"m": fn}, er, keys)
    za = z_all.numpy().astype(np.float64)
    bad = []
    for key in keys:
        exp = np.zeros((5, 6))
        for b in range(5):
            tot = np.zeros(6)
            for f in range(12):
                zf = za[b, f]                              # (P, D)
                z, z1 = zf[:, :dz], za[b, f + 1][:, :dz]
                if key == "none":
                    dl = np.zeros_like(z)
                elif key.startswith("leace"):
                    dl = np.stack([LB.leace_disp_np(F["leace"][0], z, None if key.endswith("size") else R[int(key.split(":")[2])])])[0]
                else:
                    dl = LB.lmm_disp_np(F["D"], int(lvl[ids[b]]), None if key.endswith("size") else R[int(key.split(":")[2])], None if key.endswith("size") else Sg[int(key.split(":")[2])])
                ze = z + dl
                zin = zf.copy()
                zin[:, :dz] = ze
                pred = (zin @ Wm)[:, :dz]
                nr = np.linalg.norm(dl, axis=-1)
                tot += np.array([((pred - z1) ** 2).mean(), (((pred - ze) - (z1 - z)) ** 2).mean(), ((z1 - z) ** 2).mean(), ((ze - z1) ** 2).mean(), nr.mean(), (nr ** 2).mean()])
            exp[b] = tot / 12
        bad.append(np.abs(out["m"][key] - exp).max())
    check("eval_batch == independent numpy (total, motion, persist, copy, displacement norms) for none / LEACE size+random / level-mean size+random", max(bad) < 1e-4, "(max abs diff %.2g)" % max(bad))
    o0 = out["m"]["none"]
    check("no erasure: total == motion, displacement 0", np.allclose(o0[:, K.MI["total"]], o0[:, K.MI["motion"]]) and np.all(o0[:, K.MI["disp_norm_mean"]] == 0))
    out2, _, _ = K.eval_batch(z_all, torch.as_tensor(lvl[ids]), {"copy": lambda z: z.clone()}, er, keys)
    check("copy predictor (pred = erased input): motion == persistence for EVERY erasure; total == copy error", all(np.allclose(out2["copy"][k][:, K.MI["motion"]], out2["copy"][k][:, K.MI["persist"]]) and np.allclose(out2["copy"][k][:, K.MI["total"]], out2["copy"][k][:, K.MI["copy"]]) for k in keys))
    check("persist does not depend on the erasure", all(np.allclose(out["m"][k][:, K.MI["persist"]], o0[:, K.MI["persist"]]) for k in keys))
    check("stage-2 inputs: dreal == mean real change; shapes (B, dz)", dreal.shape == (5, dz) and np.allclose(dreal, np.stack([(za[b, 1:13, :, :dz] - za[b, :12, :, :dz]).mean((0, 1)) for b in range(5)]), atol=1e-5))


# ------------------------------------------------------------------------------------------------ G
def t_stage2():
    print("G  stage-2 level-mean comparison")
    rng = np.random.RandomState(0)
    n, nl, dz = 90, 9, 16
    lvl = np.repeat(np.arange(nl), n // nl)
    lm = rng.randn(nl, dz)
    real = lm[lvl] + 0.01 * rng.randn(n, dz)
    r = LB.level_mean_comparison(2.0 * real + 5.0 + 0.001 * rng.randn(n, dz), real, lvl, nl)
    check("pred = 2 * real + constant: correlation ~1, slope ~2 (global mean removed)", abs(r["corr"] - 1) < 1e-3 and abs(r["slope_pred_on_real"] - 2) < 0.02, str({k: round(v, 4) for k, v in r.items() if isinstance(v, float)}))
    r2 = LB.level_mean_comparison(rng.randn(n, dz), real, lvl, nl)
    check("unrelated prediction: |correlation| small", abs(r2["corr"]) < 0.3)
    r3 = LB.level_mean_comparison(real + 100.0 * np.ones(dz), real, lvl, nl)
    check("a constant offset on all levels does not change the result", abs(r3["corr"] - 1) < 1e-6 and abs(r3["slope_pred_on_real"] - 1) < 1e-6)


# ------------------------------------------------------------------------------------------------ H
def t_planted(cfg, cfg2, c9):
    print("H  planted size-use world: fit -> eval -> Delta_use -> verdict -> aggregate (results.json)")
    import torch
    c2, rows = toy_rows(cfg2)
    P, dz, nact = toy_dims()
    tokens, info = planted_tokens(rows, P, dz)
    seeds = [0, 1, 2]
    F, Etr, lvl = synth_fit(cfg, c2, rows, tokens, seeds)
    ep_all = np.arange(len(rows) - Etr)
    tmp = Path(tempfile.mkdtemp())
    try:
        cfgt = copy.deepcopy(cfg)
        paths = {"results_dir": tmp / "res", "exp": tmp, "cache": tmp / "cache"}
        B.write_json({"passed": True, "n_pairs_pass": 5, "n_pairs": 6, "gate_config_sha256": B.gate_config_sha(cfgt)}, paths["results_dir"] / "stage0.json")

        def write_units(use_b, use_a):
            for cond, uses in (("multi_60_ind_160", use_b), ("single_ind", use_a)):
                for ts in seeds:
                    fns = {"trained": oracle_fn(info, dz) if uses else copy_fn(dz), "random_init": copy_fn(dz)}
                    arrs, meta = synth_unit(cfgt, c2, rows, Etr, lvl, F, tokens, ts, fns, ep_all, nact)
                    meta.update({"condition": cond, "train_seed": ts, "run": "toy", "epoch": 0, "checkpoint_sha256": "x", "random_init_seed": 1234 + ts, "encoder_fingerprint": "toy", "cache_encoder_fingerprint": "toy",
                                 "encoder_matches_cache": True, "first_batch_max_abs_diff_encoder_vs_cache": 0.0, "gate_overridden": False, "episodes_per_level": None})
                    S.part_npz(paths, cond, ts).parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(S.part_npz(paths, cond, ts), **arrs)
                    B.write_json(meta, S.part_json(paths, "unit", cond, ts))
        write_units(True, False)
        res = S.aggregate(cfgt, c9, c2, paths, ["single_ind", "multi_60_ind_160"], seeds, False, False, None, False)
        v = res["verdict"]
        st = res["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["trained"]["leace_scalar"]["train"]["motion"]
        stc = res["stage1"]["conditions"]["single_ind"]["seeds"]["0"]["trained"]["leace_scalar"]["train"]["motion"]
        stl = res["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["trained"]["level_mean_match"]["train"]["motion"]
        print("     planted b: motion_norm none %.3f size %.3f rand %.3f Delta_use %.3f CI [%.3f, %.3f]" % (st["motion_norm_none"]["point"], st["motion_norm_size"]["point"], st["motion_norm_rand_mean"]["point"], st["delta_use"]["point"], st["delta_use"]["ci_lo"], st["delta_use"]["ci_hi"]))
        check("planted use in (b), none in (a): LEACE verdict = criterion_met", v["leace_scalar"]["status"] == LB.MET, str({k: v["leace_scalar"][k] for k in ("cond1_seeds_positive_ci_excl_0", "cond2_mean_delta_use_ge_10pct_mean_gap", "cond3_b_greater_than_a")}))
        check("planted use in (b), none in (a): level-mean-match verdict = criterion_met", v["level_mean_match"]["status"] == LB.MET, str({k: v["level_mean_match"][k] for k in ("cond1_seeds_positive_ci_excl_0", "cond2_mean_delta_use_ge_10pct_mean_gap", "cond3_b_greater_than_a")}))
        check("planted use: un-erased motion error is far below persistence (gap large) and size erasure removes it", st["motion_norm_none"]["point"] < 0.5 and st["motion_norm_size"]["point"] > 0.8 and st["delta_use"]["ci_lo"] > 0.3)
        check("a predictor that ignores the input size (copy) has Delta_use == 0 and gap 0", abs(stc["delta_use"]["point"]) < 1e-9 and abs(stc["gap"]["point"]) < 1e-9)
        check("random-init (= copy here) reference Delta_use ~ 0 is reported per condition/seed", abs(res["random_init_delta_use_reference"]["multi_60_ind_160"]["0"]["leace_scalar"]["point"]) < 1e-9)
        nm = res["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["trained"]["leace_scalar"]["train"]["norm_matching"]
        check("results.json records the norm matching (ratio random/size == 1)", abs(nm["ratio_random_over_size"] - 1) < 1e-4 and nm["max_abs_diff_per_episode"] < 1e-4, str(nm))
        write_units(False, False)
        res2 = S.aggregate(cfgt, c9, c2, paths, ["single_ind", "multi_60_ind_160"], seeds, False, False, None, False)
        check("no predictor uses size (both copy): verdict criterion_not_met for both methods", all(res2["verdict"][m]["status"] == LB.NOT_MET for m in res2["verdict"]))
        write_units(True, True)
        res3 = S.aggregate(cfgt, c9, c2, paths, ["single_ind", "multi_60_ind_160"], seeds, False, False, None, False)
        check("size use equally strong in (a) and (b): cond3 (b > a) fails -> criterion_not_met", res3["verdict"]["leace_scalar"]["cond3_b_greater_than_a"]["met"] is False and res3["verdict"]["leace_scalar"]["status"] == LB.NOT_MET)
        # sets and judged split
        r0 = res["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["trained"]
        check("sets: leace has train/interp/extrap (+ held-half variants); level-mean-match evaluates OoD levels on the held-out half only (n smaller)",
              set(r0["leace_scalar"]) == {"train", "interp", "interp_heldhalf", "extrap", "extrap_heldhalf"} and set(r0["level_mean_match"]) == {"train", "interp", "extrap"}
              and r0["leace_scalar"]["extrap"]["motion"]["n_episodes"] == 40 and r0["level_mean_match"]["extrap"]["motion"]["n_episodes"] == 20 and r0["leace_scalar"]["train"]["motion"]["n_episodes"] == 30)
        check("results.json fields: pilot (true: config flag + separate dir), pilot_reasons, config_sha256, code_version, environment, gate, interpretation null, stage2, missing_units empty",
              res["pilot"] is True and len(res["pilot_reasons"]) >= 2 and len(res["config_sha256"]) == 64 and set(res["code_version"]) == {"git_head", "git_status_dirty", "repo_url", "requested_ref"}
              and "python" in res["environment"] and res["gate"]["ok"] is True and res["interpretation"] is None and res["missing_units"] == []
              and "stage2_level_mean_delta_z" in res["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"] and "erasers_shared_across_conditions" in res)
        s2 = res["stage1"]["conditions"]["multi_60_ind_160"]["seeds"]["0"]["stage2_level_mean_delta_z"]["trained"]
        check("stage 2: oracle predictor tracks the level-mean change (corr > 0.9, slope ~1)", s2["corr"] > 0.9 and 0.6 < s2["slope_pred_on_real"] < 1.4, str(s2))
        S.part_json(paths, "unit", "single_ind", 1).unlink()
        res4 = S.aggregate(cfgt, c9, c2, paths, ["single_ind", "multi_60_ind_160"], seeds, False, False, None, False)
        check("a missing (a) unit -> verdict not_evaluable, listed in missing_units (no estimate substituted)", res4["verdict"]["leace_scalar"]["status"] == LB.NOT_EVAL and "unit_single_ind_s1" in res4["missing_units"])
        cm = copy.deepcopy(cfg)
        cm["pilot"] = False
        check("pilot logic: config pilot false + experiment results dir + no overrides -> no reasons; each override adds a reason",
              S.pilot_reasons(cm, B.default_results_dir(), False, False, None, False) == [] and len(S.pilot_reasons(cm, B.default_results_dir(), True, True, 10, True)) == 4 and len(S.pilot_reasons(cm, tmp, False, False, None, False)) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ I
def t_toy_world(cfg, cfg2, c9):
    print("I  toy world model: compute_unit, determinism, batch invariance, erased extraction, readout drop")
    try:
        import torch
    except ImportError:
        REC["skipped"].append("I")
        return
    from exp009.selftest import toy_setup
    from exp004 import deep as DP
    from exp002 import evalgrid as EG
    M, old, c2, rows, C, w, (nvis, nact, P, L, H), T4 = toy_setup(cfg2)
    tmp = Path(tempfile.mkdtemp())
    try:
        dev = torch.device("cpu")
        mean, std = torch.zeros(6), torch.ones(6)
        E = len(rows)
        Etr = sum(r["part"] == "train" for r in rows)
        # cached tokens exactly as encode_batch produces them
        toks = np.zeros((E, 13, P, nvis), np.float32)
        with torch.no_grad():
            for s in range(0, E, 16):
                sl = slice(s, min(E, s + 16))
                x = M.frames_to_float(C["frames_u8"][sl])
                acts = M.acts_from_raw(torch.as_tensor(C["raw"][sl] if "raw" in C else C["actions_raw"][sl]), mean, std)
                toks[sl] = M.encode_batch(w, x, acts)[..., :nvis].numpy()
        C["tokens_dino"] = toks
        C["meta"] = {"encoder_fingerprint": "toy"}
        F, Etr_, lvl = synth_fit(cfg, c2, rows, toks, [0])
        er = K.Eraser(F["leace"][0], F["D"], LB.random_dirs(20, nvis, 13000), LB.random_signs(20, 9, P, 13000), dev)
        rnd = DP.random_init_predictor(w, 1234)
        ep_sel = np.arange(E - Etr)
        a1, d1 = S.compute_unit(cfg, c2, C, w, rnd, mean, std, er, dev, ep_sel, 8, "toy")
        a2, _ = S.compute_unit(cfg, c2, C, w, rnd, mean, std, er, dev, ep_sel, 16, "toy")
        a3, _ = S.compute_unit(cfg, c2, C, w, rnd, mean, std, er, dev, ep_sel, 8, "toy")
        M1 = a1["metrics_trained"]
        check("compute_unit shapes: (43 configs, 90 test episodes, 6 metrics) for trained and random-init", M1.shape == (43, 90, 6) and a1["metrics_random_init"].shape == (43, 90, 6) and a1["dhat_trained"].shape == (90, nvis))
        check("deterministic (same inputs twice) and batch-size invariant (8 vs 16 episodes)", all(np.array_equal(a1[k], a3[k]) for k in a1) and all(np.allclose(a1[k], a2[k], atol=1e-6) for k in a1))
        check("encoder == cached tokens in the toy (first-batch difference ~ 0)", d1 < 1e-5, "(%.2g)" % d1)
        check("trained and random-init predictors give different scores", not np.allclose(M1, a1["metrics_random_init"]))
        keys = K.config_keys(20)
        ki = {k: i for i, k in enumerate(keys)}
        check("config keys: 1 + 2 x (1 + 20) = 43, none first", len(keys) == 43 and keys[0] == "none" and keys[1] == "leace:size" and keys[22] == "lmm:size")
        check("size vs random displacement norms are equal per episode for all 20 directions in the full pipeline", max(np.abs(M1[ki["leace:rand:%d" % k], :, K.MI["disp_norm_mean"]] - M1[ki["leace:size"], :, K.MI["disp_norm_mean"]]).max() for k in range(20)) < 1e-4
              and max(np.abs(M1[ki["lmm:rand:%d" % k], :, K.MI["disp_norm_mean"]] - M1[ki["lmm:size"], :, K.MI["disp_norm_mean"]]).max() for k in range(20)) < 1e-4)
        # erased extraction vs 009 extract_hidden
        frames = [0, 1, 2, 3, 8, 9, 10, 11]
        Cf = dict(C)
        Cf["rows"] = rows
        ER.extract_hidden(w, Cf, H, tmp / "ref", mean, std, dev, frames, nvis, [0, 3, 6], None, 8)
        K.extract_erased(w, Cf, H, tmp / "mine", mean, std, dev, frames, nvis, [0, 3, 6], er, None, lvl, 8)
        same = all(np.array_equal(np.load(tmp / "ref" / ("feat_%d.npy" % m)), np.load(tmp / "mine" / ("feat_%d.npy" % m))) for m in (0, 3, 6))
        check("extract_erased without erasure == 009 extract_hidden (depths 0, 3, 6)", same)
        K.extract_erased(w, Cf, H, tmp / "le", mean, std, dev, frames, nvis, [0], er, "leace:size", lvl, 8)
        F0 = np.load(tmp / "ref" / "feat_0.npy").astype(np.float64)
        Fl = np.load(tmp / "le" / "feat_0.npy").astype(np.float64)
        pos = w.predictor.pos_embedding.detach().numpy()[0]               # (P, D)
        # depth 0 = input + pos embedding, so (erased - plain) = displacement of the input z part
        zin = np.stack([toks[e, f] for e in range(E) for f in frames]).reshape(E, len(frames), P, nvis)
        dl = np.stack([[LB.leace_disp_np(F["leace"][0], zin[e, j].astype(np.float64)) for j in range(len(frames))] for e in range(E)])
        check("erasure is applied to the predictor INPUT z part (depth-0 hidden differs from the plain one by the LEACE displacement)", np.abs((Fl - F0) - dl).max() < 5e-3, "(%.2g, fp16 storage)" % np.abs((Fl - F0) - dl).max())
        a = F["leace"][0]["a"]
        pooled = lambda X: X.mean(2) @ a
        check("after the input erasure the pooled fitted readout is (almost) constant over episodes/frames at depth 0 (std ratio < 0.02)", pooled(Fl).std() < 0.02 * pooled(F0).std(), "(%.3g vs %.3g)" % (pooled(Fl).std(), pooled(F0).std()))
        # readout drop with the 009 probe
        fr, pos_w = S.S4.frames_and_windows(c2)
        c2r = copy.deepcopy(c2)
        c2r["probe"]["seeds"] = [0, 1, 2]
        r_none = S.readout_method(c2r, c9, w, Cf, mean, std, dev, 8, er, "none", lvl, tmp / "r0", 0)
        r_le = S.readout_method(c2r, c9, w, Cf, mean, std, dev, 8, er, "leace_scalar", lvl, tmp / "r1", 0)
        r_lm = S.readout_method(c2r, c9, w, Cf, mean, std, dev, 8, er, "level_mean_match", lvl, tmp / "r2", 0)
        g = lambda r, d: r["r2_by_split_depth"]["train"][str(d)]
        print("     toy linear size readout R2 (train split) depth 0/3/6: none %s | leace %s | level-mean %s" % ([round(g(r_none, d), 3) for d in (0, 3, 6)], [round(g(r_le, d), 3) for d in (0, 3, 6)], [round(g(r_lm, d), 3) for d in (0, 3, 6)]))
        check("readout check: size readable without erasure (R2 > 0.5 at depth 0), strongly reduced by LEACE and by level-mean-match", g(r_none, 0) > 0.5 and g(r_le, 0) < g(r_none, 0) - 0.4 and g(r_lm, 0) < g(r_none, 0) - 0.4)
        check("readout record: depths 0..L for splits train/interp/extrap", sorted(int(k) for k in r_none["r2_by_split_depth"]["extrap"]) == list(range(L + 1)))
        # unit resume bookkeeping
        paths = {"results_dir": tmp / "res"}
        check("part file names distinct per (condition, seed)", len({S.part_json(paths, "unit", c, t).name for c in ("single_ind", "multi_60_ind_160") for t in (0, 1, 2)}) == 6)
    finally:
        M.P_TOK, M.D_VIS = old
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ J
def t_config(cfg, cfg2, c9):
    print("J  config consistency, attempts, preflight")
    res = B.consistency_checks(cfg, c9, cfg2)
    check("config consistency with 009/002 (%d checks)" % len(res), all(ok for _, ok, _ in res), str([n for n, ok, _ in res if not ok]))
    bad = copy.deepcopy(cfg)
    bad["erasure"]["fit_seed"] = 1
    bad["success"]["bootstrap"]["seed"] = 5
    names = [n for n, ok, _ in B.consistency_checks(bad, c9, cfg2) if not ok]
    check("a changed fit_seed / bootstrap seed is detected", any("fit_seed" in n for n in names) and any("bootstrap" in n for n in names), str(names))
    check("config: seeds explicit (global 0, train seeds, random-init 1234, fit 9000, random directions 13000, bootstrap 14000) and pilot flag present",
          cfg["seed"] == 0 and cfg["predictor"]["train_seeds"] == [0, 1, 2] and cfg["predictor"]["random_init_seed"] == 1234 and cfg["erasure"]["fit_seed"] == 9000 and cfg["erasure"]["random_dir_seed"] == 13000
          and cfg["success"]["bootstrap"]["seed"] == 14000 and "pilot" in cfg)
    tmp = Path(tempfile.mkdtemp())
    try:
        paths = {"results_dir": tmp}
        for i in range(3):
            from exp002.common import attempts_record
            attempts_record(paths, "stage:measure", False)
        check("attempts: after 3 consecutive failures the 4th run is refused (max_attempts 3)", _raises(lambda: B.attempts_check(cfg, paths, "stage:measure"), RuntimeError) and cfg["run"]["max_attempts"] == 3)
        B.attempts_check(cfg, paths, "stage:aggregate")
        check("attempts are counted per stage", True)
        out = tmp / "train_runs" / "outputs"
        (out / "multi_60_ind_160_s0").mkdir(parents=True)
        pth = {"train_base": tmp / "train_runs"}
        bad_ = S.run_problems(pth, ["multi_60_ind_160"], [0])
        check("preflight: run dir without the exp002_ prefix is not accepted, message names the expected folder", len(bad_) == 1 and "exp002_multi_60_ind_160_s0" in bad_[0])
        rd = out / "exp002_single_ind_s0"
        (rd / "checkpoints").mkdir(parents=True)
        for f in ("DONE", "hydra.yaml", "train_manifest.json"):
            (rd / f).write_text("x")
        check("preflight: model_latest.pth missing reported; with it present the (a) condition run dir is accepted", len(S.run_problems(pth, ["single_ind"], [0])) == 1)
        (rd / "checkpoints" / "model_latest.pth").write_bytes(b"x")
        check("preflight: complete exp002_single_ind_s0 -> no problem", S.run_problems(pth, ["single_ind"], [0]) == [])
        check("run names: exp002_<cond>_s<seed>", B.run_name("single_ind", 2) == "exp002_single_ind_s2")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _with(fn, *a):
    with patched():
        return fn(*a)


def run(cfg, c4, cfg2, c9):
    for name, fn in (("A", lambda: t_verdict(cfg)), ("B", lambda: t_bootstrap(cfg)), ("C", lambda: t_erasers(cfg)), ("D", lambda: t_separation(cfg, cfg2)), ("E", lambda: t_gate(cfg, cfg2)),
                     ("F", lambda: _with(t_metrics, cfg, cfg2)), ("G", t_stage2), ("H", lambda: _with(t_planted, cfg, cfg2, c9)), ("I", lambda: t_toy_world(cfg, cfg2, c9)), ("J", lambda: t_config(cfg, cfg2, c9))):
        try:
            fn()
        except Exception:
            REC["fail"].append(name + " (exception)")
            traceback.print_exc()
    print("selftest summary: passed=%d failed=%d skipped=%d" % (len(REC["pass"]), len(REC["fail"]), len(REC["skipped"])))
    if REC["fail"]:
        print("FAILED:", REC["fail"])
    return 1 if REC["fail"] else 0
