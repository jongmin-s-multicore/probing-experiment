"""CPU self-test on SYNTHETIC data with known answers. The numbers printed here are NOT experiment results.
Sub-tests that need torch are reported as 'skipped(no torch)' when torch is missing (never counted as passed).
Covers: metrics/bootstrap/Holm/Spearman, matched random directions, INLP/LEACE, Q1 probes + 3 controls (end to end through the cache format),
H-a/H-b/H-c verdict logic with known outcomes (incl. null threshold -> not_evaluable, (a) missing -> pending, threshold wiring),
hooks + Q3 run_layer on a toy predictor with the branch's module layout (uses it / ignores it / nonlinear information -> not_removed)."""
from __future__ import annotations

import copy
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np

from . import directions as D
from . import stats as ST
from .common import load_config, write_json, read_json, ConfigNull, require, PreregReader

REC = {"pass": [], "fail": [], "skipped": []}


def check(name, cond, detail=""):
    (REC["pass"] if cond else REC["fail"]).append(name)
    print("  [%s] %s %s" % ("ok" if cond else "FAIL", name, detail))


def skip(name, why):
    REC["skipped"].append(name)
    print("  [skipped(%s)] %s" % (why, name))


def have_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ------------------------------------------------------------------------------------------------ A. numpy parts
def t_stats():
    print("A1 stats")
    rng = np.random.RandomState(0)
    n_eps, per = 40, 5
    ep = np.repeat(np.arange(n_eps), per)
    y = rng.randn(n_eps * per)
    h = y + 0.5 * rng.randn(n_eps * per)
    S = ST.episode_stats(y, h, ep, n_eps)
    tot = S.sum(0)
    check("R2 from sums == direct", abs(ST.metric_from_sums(tot, "r2") - (1 - ((y - h) ** 2).sum() / ((y - y.mean()) ** 2).sum())) < 1e-9)
    check("pearson from sums == np.corrcoef", abs(ST.metric_from_sums(tot, "pearson") - np.corrcoef(y, h)[0, 1]) < 1e-9)
    check("MAE from sums", abs(ST.metric_from_sums(tot, "mae") - np.abs(y - h).mean()) < 1e-9)
    lv = np.repeat(np.arange(4), n_eps // 4)
    b = ST.Bootstrap(lv, 300, 1)
    st = b.stat(S[None], np.ones(n_eps), "r2")
    lo, hi = ST.ci95(st["boot"])
    check("bootstrap CI contains point", lo <= st["point"] <= hi, "(%.3f in [%.3f, %.3f])" % (st["point"], lo, hi))
    sw = b.stat(S[None], np.ones(n_eps), "r2_within")
    ys, hs = y.reshape(-1), h
    ref_sst = sum(((y[(lv[ep] == l)] - y[(lv[ep] == l)].mean()) ** 2).sum() for l in range(4))
    ref_sse = sum((((y - y[lv[ep] == l].mean())[lv[ep] == l] - (h - h[lv[ep] == l].mean())[lv[ep] == l]) ** 2).sum() for l in range(4))
    check("within-level R2 == direct", abs(sw["point"] - (1 - ref_sse / ref_sst)) < 1e-9)
    adj, rej = ST.holm([0.01, 0.04, 0.03], 0.05)
    check("Holm adjusted p", np.allclose(adj, [0.03, 0.06, 0.06]) and rej == [True, False, False], str(adj))
    adj, rej = ST.holm([0.001, None], 0.05)
    check("Holm: None counts in family size (m=2)", abs(adj[0] - 0.002) < 1e-12 and rej == [True, False])
    check("Spearman monotone = 1, reversed = -1", ST.spearman([1, 2, 3, 4], [1, 5, 6, 9]) == 1.0 and ST.spearman([1, 2, 3], [3, 2, 1]) == -1.0)
    check("Spearman ties (average ranks)", abs(ST.spearman([1, 2, 2, 3], [1, 2, 3, 4]) - np.corrcoef(ST.rankdata([1, 2, 2, 3]), ST.rankdata([1, 2, 3, 4]))[0, 1]) < 1e-12)
    check("pair order accuracy perfect", ST.pair_order_accuracy([1, 2, 3, 4], [1, 1, 2, 3]) == 1.0)


def t_directions():
    print("A2 directions / INLP / LEACE")
    rng = np.random.RandomState(0)
    N, d = 600, 20
    A = rng.randn(d, d) @ np.diag(np.linspace(0.2, 3, d))
    base = rng.randn(N, d) @ A.T
    a = rng.randn(d)
    a /= np.linalg.norm(a)
    y = rng.randn(N)
    X = base + 4.0 * y[:, None] * a[None]
    ep = np.arange(N)
    m = D.ridge_fit(X[:400], y[:400], ep[:400], [0.1, 1, 10, 100], 3, 0)
    check("ridge held-out R2 high on linear data", D.r2(y[400:], D.ridge_predict(m, X[400:])) > 0.7, "%.3f" % D.r2(y[400:], D.ridge_predict(m, X[400:])))
    lam, V = D.cov_eig(X)
    tgt = float(np.var(X @ m["u"], ddof=1))
    rs = [D.matched_random_direction(lam, V, tgt, np.random.RandomState(s)) for s in range(30)]
    check("matched random directions: unit norm", all(abs(np.linalg.norm(r) - 1) < 1e-8 for r, _, _ in rs))
    check("matched random directions: variance matched (<1%)", max(abs(q - tgt) / tgt for _, q, _ in rs) < 0.01, "max rel err %.4f" % max(abs(q - tgt) / tgt for _, q, _ in rs))
    iso = np.random.RandomState(1).randn(200, d)
    iso /= np.linalg.norm(iso, axis=1, keepdims=True)
    viso = np.array([np.var(X @ v, ddof=1) for v in iso])
    check("isotropic Gaussian directions would NOT be variance matched (control is meaningful)", np.std(viso) / tgt > 0.2, "iso rel std %.2f" % (np.std(viso) / tgt))
    r0, r1 = rs[0][0], rs[1][0]
    check("different RNG streams give different directions", abs(r0 @ r1) < 0.99)
    ss = [np.random.SeedSequence([1000, ts, 3, 0]).generate_state(1)[0] for ts in (0, 1, 2)]
    check("per-training-seed direction streams are distinct", len(set(ss)) == 3)
    B, r2s = D.inlp(X[:400], y[:400], ep[:400], 20, [1.0, 10.0], 3, 0)
    Xr = X - (X @ B) @ B.T
    check("INLP: train linear R2 decreases over iterations", r2s[-1] < r2s[0] - 0.3, "%.2f -> %.2f" % (r2s[0], r2s[-1]))
    mr = D.ridge_fit(Xr[:400], y[:400], ep[:400], [1.0, 10.0], 3, 0)
    check("INLP: linear info gone (held-out ridge R2 < 0.05)", D.r2(y[400:], D.ridge_predict(mr, Xr[400:])) < 0.05, "r2 %.3f, rank %d" % (D.r2(y[400:], D.ridge_predict(mr, Xr[400:])), B.shape[1]))
    M, mu = D.leace_fit(X[:400], y[:400])
    Xl = D.apply_removal(X, M, mu)
    ml = D.ridge_fit(Xl[:400], y[:400], ep[:400], [1.0, 10.0], 3, 0)
    check("LEACE: linear info gone (held-out ridge R2 < 0.05)", D.r2(y[400:], D.ridge_predict(ml, Xl[400:])) < 0.05)
    check("LEACE: rank-1 change only", np.linalg.matrix_rank(M, tol=1e-6) == 1)
    check("removal strength 0 is identity", np.allclose(D.apply_removal(X, M, mu, 0.0), X))


# ------------------------------------------------------------------------------------------------ B. synthetic cache -> Q1 -> aggregate
def small_cfg(base, mask_noise):
    cfg = copy.deepcopy(base)
    cfg["data"]["train_episodes_per_level"] = 12
    cfg["data"]["eval_episodes_per_level"] = 8
    cfg["probe"]["seeds"] = [0, 1]
    cfg["probe_impl"]["n_bootstrap"] = 200
    cfg["probe_impl"]["attentive"].update({"d_model": 16, "n_heads": 2, "mlp_hidden": 16, "epochs": 15, "batch_size": 64})
    cfg["probe_impl"]["device"] = "cpu"
    cfg["probe_impl"]["mlp_check"].update({"hidden": 32, "epochs": 600, "lr": 3.0e-3})
    cfg["train"]["seeds"] = [0, 1, 2]
    cfg["_mask_noise"] = mask_noise
    return cfg


def make_rows(cfg):
    from .data import level_roles, nominal_log_area
    roles = level_roles(cfg)
    rows = []
    for part, levels, n in (("train", cfg["data"]["train_levels_b"], cfg["data"]["train_episodes_per_level"]), ("test", cfg["data"]["levels"], cfg["data"]["eval_episodes_per_level"])):
        for l in levels:
            for i in range(n):
                rows.append({"id": len(rows), "part": part, "level": l, "role": roles[l], "split": "x", "name": "e%d" % len(rows), "dir": "x", "rank_in_level": i,
                             "nominal_log_area": nominal_log_area(cfg, l)})
    return rows


P_SYN, D_SYN = 6, 16


def feat_from_size(s, rng, scale, noise=0.3, frames=13):
    """tokens (E,frames,P,D): scale * s * a + noise. a fixed direction."""
    a = np.random.RandomState(123).randn(D_SYN)
    a /= np.linalg.norm(a)
    t = scale * s[:, None, None, None] * a[None, None, None, :] + noise * rng.randn(len(s), frames, P_SYN, D_SYN)
    return t.astype(np.float16)


def write_synth_cache(cfg, paths, tokens_scale=1.0, mask_noise=0.6):
    rng = np.random.RandomState(7)
    rows = make_rows(cfg)
    s = np.array([r["nominal_log_area"] for r in rows])
    E = len(rows)
    cd = paths["cache"] / "q1_tokens"
    cd.mkdir(parents=True, exist_ok=True)
    np.save(cd / "tokens_dino.npy", feat_from_size(s, rng, tokens_scale))
    np.save(cd / "tokens_rand.npy", (0.3 * rng.randn(E, 13, P_SYN, D_SYN)).astype(np.float16))
    area = np.exp(5 + s[:, None] + mask_noise * rng.randn(E, 13))
    mf = np.stack([area, np.sqrt(area), np.sqrt(area)], -1).astype(np.float32)
    np.save(cd / "mask_feat.npy", mf)
    np.save(cd / "frames_u8.npy", np.zeros((E, 13, 2, 2, 3), np.uint8))
    np.save(cd / "actions_raw.npy", np.zeros((E, 60, 6), np.float32))
    write_json(rows, cd / "episodes.json")
    write_json({"encoder_fingerprint": "synthetic", "K": 13}, cd / "meta.json")
    (cd / "DONE").write_text("ok\n")
    return rows


def synth_paths(tmp):
    return {"results_dir": Path(tmp) / "results", "exp": Path(tmp), "cache": Path(tmp) / "cache", "logs": Path(tmp) / "logs", "manifests": Path(tmp) / "manifests"}


def t_q1_aggregate(base):
    print("B  synthetic cache -> Q1 (probes + 3 controls) -> H-a verdicts")
    from . import q1 as Q1, aggregate as AG
    kinds = ["linear_pooled", "attentive"] if have_torch() else ["linear_pooled"]
    if not have_torch():
        skip("attentive probe (Q1)", "no torch")
    out = {}
    for case, (scale, mnoise, expect) in {"signal": (1.0, 0.6, "pass"), "no_signal": (0.0, 0.6, "fail"), "saturated_baseline": (1.0, 0.0, "not_evaluable")}.items():
        tmp = tempfile.mkdtemp()
        try:
            cfg = small_cfg(base, mnoise)
            paths = synth_paths(tmp)
            write_synth_cache(cfg, paths, scale, mnoise)
            cfg["probe"]["kinds"] = kinds
            Q1.run(cfg, paths, only_kinds=kinds)
            res = AG.run(cfg, paths)
            rows = [r for r in res["decision_table"] if r["id"].startswith("H_a") and r["id"].split(":")[-1] in kinds]
            check("H-a [%s]: %d rows for the probe kinds that ran" % (case, len(rows)), len(rows) == 2 * len(kinds))
            check("H-a [%s]: all statuses == %s" % (case, expect), all(r["status"] == expect for r in rows), str([(r["id"], r["status"]) for r in rows]))
            er = res["h_a_early_flat_report"]
            check("H-a [%s]: early_flat reported (not judged) with baseline R2 + saturation flag" % case, er["status"] == "reported_not_judged" and len(er["rows"]) == 2 * len(kinds)
                  and all("r2_mask_baseline" in x for x in er["rows"]) and all(x["baseline_saturated"] == (case == "saturated_baseline") for x in er["rows"]))
            if case == "signal":
                cell = np.load(paths["results_dir"] / "q1" / "cells.npz")
                r0 = rows[0]["result"]
                check("signal: real R2 high, shuffle control low, random-init low",
                      r0["r2_real"] > 0.5 and r0["r2_random_init"] < 0.3, "real %.2f rand %.2f base %.2f" % (r0["r2_real"], r0["r2_random_init"], r0["r2_mask_baseline"]))
                sh = json_cell(paths, "%s|linear_pooled|shuffle|all|all" % cfg["q1"]["primary_target"], "unseen")
                check("label-shuffle control R2 near 0 or negative", sh < 0.25, "%.3f" % sh)
                check("results.json has CI caveat + prereg values used", "training-seed" in res["ci_caveat"] and "q2_drop_threshold" in res["prereg_values_used"])
                # threshold wiring: margin 0.99 must flip pass -> fail
                cfg2 = copy.deepcopy(cfg)
                cfg2["prereg"]["q1_margin"]["delta_r2_vs_mask_area_baseline"] = 0.99
                res2 = AG.run(cfg2, paths)
                rows2 = [r for r in res2["decision_table"] if r["id"].startswith("H_a") and r["id"].split(":")[-1] in kinds]
                check("threshold wiring: raising q1 margin to 0.99 changes status to fail", all(r["status"] == "fail" for r in rows2))
                cfg3 = copy.deepcopy(cfg)
                cfg3["prereg"]["q1_margin"]["delta_r2_vs_mask_area_baseline"] = None
                rows3 = [r for r in AG.run(cfg3, paths)["decision_table"] if r["id"].startswith("H_a") and r["id"].split(":")[-1] in kinds]
                check("null margin -> not_evaluable", all(r["status"] == "not_evaluable" for r in rows3))
                out["paths"], out["cfg"], out["tmp"] = paths, cfg, tmp
                tmp = None
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
    return out


def json_cell(paths, key, role):
    s = read_json(paths["results_dir"] / "q1" / "q1_summary.json")
    return s["cells"][key][role]["r2"]["point"]


def t_h_b(ctx):
    print("C  H-b verdict logic on synthetic Q2 cells")
    from . import aggregate as AG, evalgrid as EG
    cfg, paths = ctx["cfg"], ctx["paths"]
    from . import q2 as Q2
    prim = cfg["q1"]["primary_target"]
    cfgo = copy.deepcopy(cfg)
    cfgo["probe"]["kinds"] = ["linear_pooled"]
    rows = make_rows(cfg)
    tr, te = EG.split_rows(rows)
    Etr, Ete = len(tr), len(te)
    mf = np.load(paths["cache"] / "q1_tokens" / "mask_feat.npy")
    Y = EG.make_labels(cfg, rows, mf, prim)
    W = EG.windows(cfg)
    frames = W["union"]
    pos = {w: [frames.index(f) for f in W[w]] for w in ("early", "late", "union")}
    for case, (keep_scale, expect) in {"preserved": (0.9, "pass"), "destroyed": (0.0, "fail")}.items():
        for seed in (0, 1, 2):
            rng = np.random.RandomState(100 + seed)
            s = np.array([r["nominal_log_area"] for r in rows])
            cells, L = {}, 3
            for j in range(L + 1):
                sc = 1.0 if j < L else keep_scale
                F = feat_from_size(s, rng, sc, frames=len(frames))
                Yp = Y[:, frames]
                c, _ = EG.fit_predict_cells(cfg, "linear_pooled", F[:Etr], Yp[:Etr], F[Etr:], Yp[Etr:], list(range(len(frames))), pos, [0, 1])
                for w in c:
                    cells["%s|linear_pooled|layer%d|%s" % (prim, j, w)] = c[w]
            tokens = feat_from_size(s, rng, 1.0)                                  # encoder tokens (E,13,P,D)
            R_te = feat_from_size(s[Etr:], rng, keep_scale, frames=12)            # predicted z of TEST episodes only
            oc, _, _ = Q2.output_space(cfgo, tokens, Y, Etr, R_te, 0, 12)
            cells.update(oc)
            d = paths["results_dir"] / ("exp002_multi_60_ind_160_s%d" % seed)
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "q2_cells.npz", **cells)
            write_json({"meta": {"n_blocks": L, "t": 0, "horizon": 12}, "diagnostics_4c": {"verdict": "small"}}, d / "q2_summary.json")
        cfgx = copy.deepcopy(cfg)
        cfgx["q2_impl"]["primary_probe"] = "linear_pooled"
        r = [x for x in AG.run(cfgx, paths)["decision_table"] if x["id"].startswith("H_b")]
        check("H-b [%s]: statuses == %s" % (case, expect), all(x["status"] == expect for x in r), str([(x["id"], x["status"], round(x["result"].get("drop_mean_over_seeds", -9), 3)) for x in r]))
        if case == "preserved":
            cfgy = copy.deepcopy(cfgx)
            cfgy["q2"]["drop_threshold"] = -0.5
            r2 = [x for x in AG.run(cfgy, paths)["decision_table"] if x["id"].startswith("H_b")]
            check("threshold wiring: q2.drop_threshold=-0.5 flips H-b to fail", all(x["status"] == "fail" for x in r2))
            cfgz = copy.deepcopy(cfgx)
            cfgz["q2"]["drop_threshold"] = None
            r3 = [x for x in AG.run(cfgz, paths)["decision_table"] if x["id"].startswith("H_b")]
            check("null q2.drop_threshold -> not_evaluable", all(x["status"] == "not_evaluable" for x in r3))
    r = [x for x in AG.run(cfg, paths)["decision_table"] if x["id"].startswith("cond_a_vs_b")]
    check("cond (a) vs (b) is 'pending' when (a) results are missing", all(x["status"] == "pending" for x in r) and len(r) == 2)
    rows_all = AG.run(cfg, paths)["decision_table"]
    check("exploratory row has no p-value", all(x["p_value"] is None for x in rows_all if x["status"] == "exploratory"))


# ------------------------------------------------------------------------------------------------ F. new in 확정 기록 6/7
def t_budget_manifest(base):
    print("F1 equal budget + manifests (episodes per condition)")
    from . import budget as B, data as DA
    pl = B.plan({"multi_60_ind_160": 1000, "single_ind": 1000, "multi_60_ind_160_small": 334}, 10, "multi_60_ind_160")
    check("equal-episode conditions get the same epochs and steps", pl["single_ind"]["epochs"] == 10 and pl["single_ind"]["steps_computed"] == pl["multi_60_ind_160"]["steps_computed"])
    check("small condition: epochs converted to (nearly) the same total steps, rel_diff recorded", pl["multi_60_ind_160_small"]["epochs"] == 30 and abs(pl["multi_60_ind_160_small"]["rel_diff"]) < 0.01,
          str(pl["multi_60_ind_160_small"]))
    pl2 = B.plan({"a": 100, "b": 33}, 3, "a", budget_steps=600)
    check("train.budget (total steps) overrides: epochs = round(steps/spe)", pl2["a"]["epochs"] == 6 and pl2["b"]["epochs"] == 18 and pl2["b"]["steps_computed"] == 594)
    check("steps_per_epoch ceil/floor", B.steps_per_epoch(101, 24) == 5 and B.steps_per_epoch(101, 24, "floor") == 4)
    listing = {}
    for l in ("a060", "ind", "a160"):
        listing["%s_train" % l] = ["%s_%d.mp4" % (l, i) for i in range(2000)]
    m = {c["name"]: DA.build_train_manifest(base, listing, "/d", c["name"]) for c in base["conditions"]["list"]}
    check("(a) single_ind = 900 ind episodes", m["single_ind"]["n_episodes"] == 900 and m["single_ind"]["group_counts"] == {"ind": 900})
    check("(b) = 3 x 300 = 900", m["multi_60_ind_160"]["n_episodes"] == 900 and set(m["multi_60_ind_160"]["group_counts"].values()) == {300})
    check("small = 3 x 100 = 300", m["multi_60_ind_160_small"]["n_episodes"] == 300 and set(m["multi_60_ind_160_small"]["group_counts"].values()) == {100})
    key = lambda mm: {(e["split"], e["name"]) for e in mm["episodes"]}
    check("small subset of (b); (b)'s ind 300 subset of (a)", key(m["multi_60_ind_160_small"]) <= key(m["multi_60_ind_160"]) and
          {k for k in key(m["multi_60_ind_160"]) if k[0] == "ind_train"} <= key(m["single_ind"]))
    check("run_order: small last and optional; train.budget null; torch_version set", base["conditions"]["run_order"][-1] == "multi_60_ind_160_small" and
          "multi_60_ind_160_small" in base["conditions"]["optional"] and base["train"]["budget"] is None and base["env"]["torch_version"] == "2.0.1+cu117")
    try:
        DA.build_train_manifest(base, {"ind_train": ["x"] * 100}, "/d", "single_ind")
        check("too few episodes -> error", False)
    except ValueError:
        check("too few ind episodes (<900) raises", True)


def t_output_space(base):
    print("F2 output-space probe (Q1 probe applied to predicted z; no refit on predictor outputs)")
    from . import q2 as Q2, evalgrid as EG
    cfg = small_cfg(base, 0.5)
    cfg["probe"]["kinds"] = ["linear_pooled"]
    rows = make_rows(cfg)
    tr, te = EG.split_rows(rows)
    Etr = len(tr)
    s = np.array([r["nominal_log_area"] for r in rows])
    rng = np.random.RandomState(3)
    tokens = feat_from_size(s, rng, 1.0)
    Y = EG.make_labels(cfg, rows, np.zeros((len(rows), 13, 3)), cfg["q1"]["primary_target"])
    R1 = feat_from_size(s[Etr:], rng, 1.0, frames=4)
    R2 = feat_from_size(s[Etr:], rng, 0.0, frames=4)
    c1, p1, d1 = Q2.output_space(cfg, tokens, Y, Etr, R1, 0, 4)
    c2, p2, d2 = Q2.output_space(cfg, tokens, Y, Etr, R2, 0, 4)
    pre = "%s|linear_pooled|OUT|" % cfg["q1"]["primary_target"]
    check("encoder-side reads and probe digest do not depend on the predictor output (probe not fitted on it)", all(np.array_equal(c1[pre + "encoder|step%d" % k], c2[pre + "encoder|step%d" % k]) for k in range(1, 5))
          and d1 == d2)
    rm = EG.role_masks(te)
    boot = ST.Bootstrap(EG.level_index(te, list(cfg["data"]["levels"])), 100, 0)
    r_keep = boot.stat(c1[pre + "rollout|step4"], rm["unseen"], "r2")["point"]
    r_kill = boot.stat(c2[pre + "rollout|step4"], rm["unseen"], "r2")["point"]
    r_base = boot.stat(c1[pre + "rollout|step4"], rm["trained"], "r2_within")["point"]
    check("rollout read: preserved signal high, destroyed signal low", r_keep > 0.5 and r_kill < r_keep - 0.5, "%.2f vs %.2f" % (r_keep, r_kill))
    check("trained-role baseline cell exists for the same probe", np.isfinite(boot.stat(c1[pre + "rollout|step1"], rm["trained"], "mae")["point"]))
    check("predicted-size means stored per level source (pm shape = probe seeds x test eps)", p1[pre + "rollout|step1"].shape == (2, len(te)))


def t_diag(base):
    print("F3 4c diagnostics (rule of 확정 기록 7 A) incl. boundary values")
    from . import diag as DG
    sm, mm, el = base["diag_4c"]["activation_small_max_ratio"], base["diag_4c"]["activation_medium_max_ratio"], base["diag_4c"]["error_large_min_ratio"]
    check("config diag_4c = 2.0 / 5.0 / 2.0 and seed set", (sm, mm, el) == (2.0, 5.0, 2.0) and base["diag_4c"]["bootstrap_seed"] is not None)
    cr = lambda r: DG.classify_ratio(r, sm, mm)
    check("boundaries: 1.999 small, 2.0 medium, 4.999 medium, 5.0 large", (cr(1.999), cr(2.0), cr(4.999), cr(5.0)) == ("small", "medium", "medium", "large"))
    check("error boundary: 1.999 not large, 2.0 large", (DG.classify_err(1.999, el), DG.classify_err(2.0, el)) == ("not_large", "large"))
    check("final: any large -> large; else worst activation class", DG.final_verdict(["small", "small"], ["not_large", "large"]) == "large" and DG.final_verdict(["small", "medium"], ["not_large", "not_large"]) == "medium"
          and DG.final_verdict(["large", "small"], ["not_large"] * 2) == "large" and DG.final_verdict(["small"], ["not_large"]) == "small")
    check("final: missing -> not_evaluable", DG.final_verdict([], []) == "not_evaluable" and DG.final_verdict(["small"], ["not_evaluable"]) == "not_evaluable")
    rng = np.random.RandomState(0)
    mk = lambda n, mu, sd: [DG.episode_stats(mu + sd * rng.randn(2, 3, 5)) for _ in range(n)]
    st = {0: mk(40, 0.0, 1.0) + mk(20, 0.0, 1.0) + mk(20, 3.0, 1.0)}          # 0..39 ind, 40..59 same distribution, 60..79 mean shifted by 3 sd
    ne = {"ind": np.ones(40), "same": np.ones(20), "shifted": np.ones(20), "bigerr": 2.5 * np.ones(20)}
    res = DG.evaluate(st, list(range(40)), {"same": list(range(40, 60)), "shifted": list(range(60, 80))}, {**ne}, 100, 1, base["diag_4c"])
    check("same distribution as ind -> activation class small", res["levels"]["same"]["activation_class"] == "small", "ratio %.2f" % res["levels"]["same"]["max_ratio_over_layers"])
    check("mean shift of 3 sd -> large", res["levels"]["shifted"]["activation_class"] == "large" and res["verdict"] == "large")
    ne2 = {"ind": np.ones(40), "same": 2.5 * np.ones(20)}
    res2 = DG.evaluate(st, list(range(40)), {"same": list(range(40, 60))}, ne2, 100, 1, base["diag_4c"])
    check("same activations but next-frame error 2.5x ind -> large (error criterion)", res2["verdict"] == "large" and res2["levels"]["same"]["error_class"] == "large")
    res3 = DG.evaluate(st, list(range(40)), {"same": list(range(40, 60))}, {"ind": np.ones(40), "same": 1.5 * np.ones(20)}, 100, 1, base["diag_4c"])
    check("verdict follows activation class when error < 2x; rule text stored", res3["verdict"] == res3["levels"]["same"]["activation_class"] and res3["interpretation_rule_for_verdict"])
    r_a = DG.evaluate(st, list(range(40)), {"same": list(range(40, 60))}, ne2, 100, 1, base["diag_4c"])["noise_floor_per_layer"]
    r_b = DG.evaluate(st, list(range(40)), {"same": list(range(40, 60))}, ne2, 100, 1, base["diag_4c"])["noise_floor_per_layer"]
    check("noise floor reproducible for a fixed seed; uses only ind episodes", r_a == r_b)
    from . import aggregate as AG
    tmp = tempfile.mkdtemp()
    try:
        paths = synth_paths(tmp)
        cfg = copy.deepcopy(base)
        r0 = [r for r in AG.diagnostics_rows(cfg, paths) if r["id"] == "diag_4c:single_ind"][0]
        check("diagnostics row without data -> not_evaluable", r0["status"] == "not_evaluable")
        d = paths["results_dir"] / "exp002_single_ind_s0"
        d.mkdir(parents=True)
        np.savez_compressed(d / "q2_cells.npz", x=np.zeros(1))
        write_json({"meta": {"n_blocks": 1}, "diagnostics_4c": res}, d / "q2_summary.json")
        r1 = [r for r in AG.diagnostics_rows(cfg, paths) if r["id"] == "diag_4c:single_ind"][0]
        check("diagnostics row carries the verdict + rule (large: numbers only)", r1["result"]["verdict_worst_over_seeds"] == "large" and "do NOT interpret" in r1["result"]["interpretation_rule"])
        rows = AG.cond_a_vs_b(cfg, PreregReader(cfg), paths, ST.Bootstrap(np.zeros(10, int), 10, 0), EG_masks(), None)
        check("cond_a_vs_b with (a) 4c large and (b) missing -> still pending (no judgment)", all(r["status"] in ("pending",) for r in rows))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def EG_masks():
    return {"unseen": np.ones(10, bool)}


def t_q3_pure(base):
    print("F4 Q3: episode-mean alpha curve, Spearman threshold, tiers")
    from . import q3 as Q3
    ds = np.array([[1.0, 1.0, 5.0, 5.0], [2.0, 2.0, 6.0, 6.0]])
    ep = np.array([0, 0, 1, 1])
    check("episode_mean: mean within episode then over episodes", np.allclose(Q3.episode_mean(ds, ep), [3.0, 4.0]))
    ds2 = np.array([[0.0, 0.0, 0.0, 10.0]])
    check("episode mean differs from sample mean when episodes have unequal sizes", abs(Q3.episode_mean(ds2, np.array([0, 0, 0, 1]))[0] - 5.0) < 1e-12 and abs(ds2.mean() - 2.5) < 1e-12)
    check("monotonic_min_spearman == 0.9 (from config)", base["q3_impl"]["monotonic_min_spearman"] == 0.9)
    r = ST.spearman([-2, -1, -0.5, 0, 0.5, 1, 2], [1, 2, 3, 4, 5, 7, 6])
    check("7-point curve with one adjacent swap: rho >= 0.9 passes, < 1.0 would fail", 0.9 <= abs(r) < 1.0, "rho %.4f" % r)
    check("Q3 confirmatory = ridge + INLP (config); others exploratory", base["q3_impl"]["primary_steering_direction"] == "ridge" and base["q3_impl"]["primary_removal_method"] == "INLP")


# ------------------------------------------------------------------------------------------------ D. torch: hooks + Q3 toy
def build_toy(dim, ptok, nvis, kind):
    import torch
    nn = torch.nn

    class Lin(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.l = nn.Linear(dim, dim, bias=False)
            nn.init.normal_(self.l.weight, std=scale)

        def forward(self, x):
            return self.l(x)

    class Tr(nn.Module):
        def __init__(self, L):
            super().__init__()
            self.layers = nn.ModuleList([nn.ModuleList([Lin(0.0 if kind == "ignore" else 0.02), Lin(0.0 if kind == "ignore" else 0.02)]) for _ in range(L)])
            self.norm = nn.Identity()

        def forward(self, x):
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return self.norm(x)

    class Pred(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Tr(4)

        def forward(self, x):
            return self.transformer(x)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = Pred()

        def predict(self, z):
            N, H, P, d = z.shape
            out = self.predictor(z.reshape(N, H * P, d)).reshape(N, H, P, d)
            if kind == "ignore":                       # output = function of action dims only (visual part ignored)
                out = torch.cat([z[..., nvis:].mean(-1, keepdim=True).expand(N, H, P, nvis) * 0.0, out[..., nvis:]], -1)
            return out
    return Toy()


def t_torch_q3(base):
    print("D  hooks + Q3 on a toy predictor (module layout like the branch: transformer.layers[(attn, ff)], transformer.norm)")
    if not have_torch():
        skip("hooks / Q3 run_layer / H-c verdicts", "no torch")
        return
    import torch
    from . import model_io as M, q3 as Q3, aggregate as AG
    M.P_TOK, Q3.D_VIS = 4, 8
    dim, nvis, ptok, H = 12, 8, 4, 1
    dev = torch.device("cpu")
    torch.manual_seed(0)
    model = build_toy(dim, ptok, nvis, "use")
    hooks = M.StreamHooks(model).install()
    z = torch.randn(5, H, ptok, dim)
    hooks.reset()
    hooks.capture_layers = {0, 2, 4}
    out = model.predict(z)
    cap = {k: v.clone() for k, v in hooks.store.items()}
    x = z.reshape(5, ptok, dim)
    manual = {0: x.clone()}
    for i, (a, f) in enumerate(model.predictor.transformer.layers):
        x = a(x) + x
        x = f(x) + x
        manual[i + 1] = x.clone()
    check("hook captures residual stream at layers 0/2/4(=L)", all(torch.allclose(cap[k], manual[k], atol=1e-6) for k in (0, 2, 4)))
    hooks.reset()
    hooks.modify = {2: lambda t: t * 0 + t}
    o2 = model.predict(z)
    check("identity modification leaves the output unchanged", torch.allclose(o2, out, atol=1e-6))
    hooks.reset()
    hooks.modify = {4: lambda t: t + 1.0}
    o3 = model.predict(z)
    check("shift at last layer changes the output by exactly the shift", torch.allclose(o3 - out, torch.ones_like(out), atol=1e-5))
    hooks.reset()
    hooks.modify = {2: lambda t: t + 1.0}
    o4 = model.predict(z)
    check("shift at a MIDDLE layer travels along the residual path (output change ~ shift, not ~0)", abs((o4 - out).mean().item() - 1.0) < 0.15, "mean change %.3f" % (o4 - out).mean().item())
    check("layer index rule: 6 blocks -> middle=3,last=6", Q3.layer_indices(base, 6)[0] == {"middle": 3, "last": 6})
    hooks.remove()

    cfg = small_cfg(base, 0.5)
    cfg["q3"]["n_random_directions"] = 100
    cfg["q3_impl"].update({"batch_samples": 64, "inlp_iterations": 4})
    cfg["probe"]["seeds"] = [0, 1]
    results = {}
    rng = np.random.RandomState(0)
    for kind, nonlinear in (("use", False), ("ignore", False), ("use", True)):
        model = build_toy(dim, ptok, nvis, kind)
        hooks = M.StreamHooks(model).install()

        class Ad(Q3.ModelAdapter):
            pass
        ad = Ad(model, hooks, H, dev)
        a = np.random.RandomState(5).randn(dim)
        a /= np.linalg.norm(a)

        nuis_bulk, nuis_hi = (0.2, 0.0) if nonlinear else (1.0, 4.0)       # high-variance nuisance dims (like action dims) keep matched random directions non-degenerate
        if nonlinear:
            lv = np.array([0.0, 1.0, 3.0])
            ytr, yev = np.repeat(lv, 40), np.repeat(lv, 20)
            b = np.cov(lv ** 2, lv, bias=True)[0, 1] / lv.var()
            f = lambda y: (y ** 2 - b * y)
            fs = f(lv)
            sc = 2.0 / (fs - fs.mean()).std()
            sig_of = lambda y: (f(y) - fs.mean()) * sc          # linearly uncorrelated with y on the training levels, but injective
        else:
            ytr = np.repeat(np.linspace(-1, 1, 5), 24)
            yev = np.repeat(np.linspace(-1.5, 1.5, 6), 10)
            sig_of = lambda y: 2.0 * y

        def mk(n, y):
            nsc = np.array([nuis_bulk] * nvis + [nuis_hi] * (dim - nvis))
            zz = 0.5 * rng.randn(n, H, ptok, dim) + nsc * rng.randn(n, 1, 1, dim) + sig_of(y)[:, None, None, None] * a[None, None, None, :]
            return torch.tensor(zz, dtype=torch.float32)

        Ztr, Zev = mk(len(ytr), ytr), mk(len(yev), yev)
        eptr = np.arange(len(ytr))
        wv = np.zeros(nvis)
        wv[:] = (a[:nvis] / (np.linalg.norm(a[:nvis]) + 1e-9))
        readout = {"w": torch.tensor(wv, dtype=torch.float32), "b": 0.0}
        t0 = __import__("time").time()
        res = Q3.run_layer(cfg, ad, 2, "middle", Ztr, ytr, eptr, Zev, yev, np.arange(len(yev)) // 2, readout, 0, 0, 2)
        name = "%s%s" % (kind, "+nonlinear" if nonlinear else "")
        results[name] = res
        hooks.remove()
        print("  (toy %s: %.0fs)" % (name, __import__("time").time() - t0))
        check("[%s] zero-shift scan equals baseline (identity)" % name, res["identity_check_max_abs_dsize_zero_shift"] < 1e-5)
        check("[%s] random directions variance-matched (<5%%)" % name, res["steering"]["ridge"]["random_variance_match_ok"])
    s_use = results["use"]["steering"]["ridge"]
    s_ign = results["ignore"]["steering"]["ridge"]
    check("toy 'uses size': size direction outside 95th percentile of 100 random, monotone", s_use["outside_percentile"] and s_use["monotone"], "p_rank %.3f rho %.2f" % (s_use["p_rank"], s_use["size_direction"]["spearman_signed_dsize_vs_alpha"]))
    check("toy 'ignores size': NOT outside the random distribution", not s_ign["outside_percentile"], "p_rank %.3f" % s_ign["p_rank"])
    check("tiers: ridge/INLP confirmatory, meandiff/LEACE exploratory", results["use"]["steering"]["ridge"]["tier"] == "confirmatory" and results["use"]["steering"]["meandiff"]["tier"] == "exploratory"
          and results["use"]["removal"]["INLP"]["tier"] == "confirmatory" and results["use"]["removal"]["LEACE"]["tier"] == "exploratory")
    r_use = results["use"]["removal"]["INLP"]
    check("toy linear info: INLP removed (MLP R2 after <= tol)", r_use["removed"], str(np.round(r_use["check"]["mlp_r2_after_removal_per_probe_seed"], 3)))
    r_nl = results["use+nonlinear"]["removal"]["INLP"]
    check("toy nonlinear info: INLP flagged not_removed", (not r_nl["removed"]) and r_nl["flag"] == "not_removed",
          "mlp after %s before %s" % (np.round(r_nl["check"]["mlp_r2_after_removal_per_probe_seed"], 2), np.round(r_nl["check"]["mlp_r2_before_per_probe_seed"], 2)))
    # H-c verdicts from the stored q3.json files
    for case, toy, expect in (("uses", "use", "pass"), ("ignores", "ignore", "fail")):
        tmp = tempfile.mkdtemp()
        try:
            paths = synth_paths(tmp)
            for seed in (0, 1, 2):
                lay = {"middle": results[toy], "last": results[toy]}
                write_json({"layers": lay}, paths["results_dir"] / ("exp002_multi_60_ind_160_s%d" % seed) / "q3.json")
            rows = [r for r in AG.h_c(cfg, PreregReader(cfg), paths, "multi_60_ind_160")]
            st = {r["id"]: r["status"] for r in rows}
            if toy == "use":
                for seed in (0, 1, 2):                      # remove exploratory blocks: verdicts must not change
                    lay2 = {t: {**v, "steering": {"ridge": v["steering"]["ridge"]}, "removal": {"INLP": v["removal"]["INLP"]}} for t, v in lay.items()}
                    write_json({"layers": lay2}, paths["results_dir"] / ("exp002_multi_60_ind_160_s%d" % seed) / "q3.json")
                st2 = {r["id"]: r["status"] for r in AG.h_c(cfg, PreregReader(cfg), paths, "multi_60_ind_160")}
                check("H-c verdicts identical without the exploratory blocks (meandiff, LEACE)", st2 == st)
            print("   H-c [%s]: %s" % (case, st))
            if toy == "use":
                check("H-c steering [uses] == pass", st["H_c:steering"] == "pass")
                check("H-c removal [uses] is pass or fail (removal verified), not not_evaluable", st["H_c:subspace_removal"] in ("pass", "fail"))
            else:
                check("H-c steering [ignores] == fail", st["H_c:steering"] == "fail")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    tmp = tempfile.mkdtemp()
    try:
        paths = synth_paths(tmp)
        for seed in (0, 1, 2):
            write_json({"layers": {"middle": results["use+nonlinear"], "last": results["use+nonlinear"]}}, paths["results_dir"] / ("exp002_multi_60_ind_160_s%d" % seed) / "q3.json")
        st = {r["id"]: r["status"] for r in AG.h_c(cfg, PreregReader(cfg), paths, "multi_60_ind_160")}
        check("H-c removal with not_removed flag -> not_evaluable (no conclusion)", st["H_c:subspace_removal"] == "not_evaluable")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_vram_poller():
    print("G  smoke.VramPoller without nvidia-smi")
    from . import smoke as SM
    SM.VramPoller._read = staticmethod(lambda: 0)
    p = SM.VramPoller(period=0.01)
    p.start()
    import time as _t
    _t.sleep(0.05)
    r = p.finish()
    check("VramPoller start -> finish returns, thread ended", not p.is_alive() and set(r) == {"baseline_mib", "peak_mib", "delta_mib"} and r["delta_mib"] == 0, str(r))
    SM.VramPoller._read = staticmethod(lambda: 7)
    p = SM.VramPoller(period=0.01)
    p.start()
    _t.sleep(0.05)
    r = p.finish()
    check("VramPoller records peak from _read (7) with delta 0 vs baseline 7", r["peak_mib"] == 7 and r["baseline_mib"] == 7 and r["delta_mib"] == 0, str(r))
    SM.VramPoller._read = staticmethod(lambda: 7 if _t.time() < 0 else 9)
    p = SM.VramPoller.__new__(SM.VramPoller)
    import threading as _th
    _th.Thread.__init__(p, daemon=True)
    p.period, p.peak, p.base, p._halt = 0.01, 0, 5, False
    p.start()
    _t.sleep(0.05)
    r = p.finish()
    check("VramPoller delta = peak - baseline", r["delta_mib"] == 4, str(r))


def t_static_and_slices(base):
    print("H  static attribute check (AST) + slice-count plausibility")
    import tempfile as _tf
    from . import static_check as SC, data as DA
    root = Path(__file__).resolve().parents[1]
    probs, n = SC.check(root, [root.parent / "run.py"])
    check("static check: every module-alias attribute / from-import name exists (%d references)" % n, not probs, "; ".join(probs[:5]))
    check("data.count_train_slices is defined", hasattr(DA, "count_train_slices") and hasattr(DA, "expected_slices"))
    tmp = Path(_tf.mkdtemp())
    try:
        (tmp / "exp002").mkdir()
        (tmp / "exp002" / "a.py").write_text("def f():\n    pass\n")
        (tmp / "exp002" / "b.py").write_text("from . import a as A\nfrom .a import f, g\ndef h():\n    A.f(); A.missing()\n")
        pr, _ = SC.check(tmp)
        check("static check detects a missing attribute and a missing from-import name", len(pr) == 2 and any("A.missing" in x for x in pr) and any("import g" in x for x in pr), str(pr))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # server smoke (num_hist=1, 48 episodes, batch 24): 99 train / 12 valid iterations. 0.9 split of 48 -> 43 / 5 (assumed int(0.9*48)); 65 frames, nf=2, frameskip 5
    ntr, nva = DA.expected_slices([65] * 43, 2, 5), DA.expected_slices([65] * 5, 2, 5)
    import math
    check("observed smoke iterations (99 train, 12 valid at batch 24) consistent with the slice formula + ceil", math.ceil(ntr / 24) == 99 and math.ceil(nva / 24) == 12, "train %d valid %d" % (ntr, nva))


def t_config(base):
    print("E  config guards")
    nb = copy.deepcopy(base)
    nb["train"]["batch_size"] = None                     # the real config now has a value; test the guard on a copy
    try:
        require(nb, "train.batch_size")
        check("null batch_size raises ConfigNull", False)
    except ConfigNull as e:
        check("null batch_size raises ConfigNull with a clear message", "TODO" in str(e))
    import yaml
    from .common import _strict_loader
    try:
        yaml.load("a: 1\na: 2\n", Loader=_strict_loader())
        check("duplicate YAML keys rejected", False)
    except ValueError:
        check("duplicate YAML keys rejected", True)
    check("Q1 primary window = late (late_deformed), report window = early", base["q1"]["primary_window"] == "late" and base["q1"]["report_window"] == "early" and base["q1"]["baseline_saturated_r2"] == 0.99)
    check("config seeds present", base["seed"] is not None and base["train"]["seeds"] and base["probe"]["seeds"] and base["q3_impl"]["random_direction_seed"] is not None)


def t_colab(base):
    """Colab path (확정 기록 10): resolve_paths with Colab-style env vars, copy_from_mount (copy / skip / size-mismatch repair), colab_tools prepare/prune."""
    import os, tempfile, json, importlib.util, copy, time
    from pathlib import Path
    from exp002 import drive
    from exp002.common import resolve_paths
    tmp = Path(tempfile.mkdtemp(prefix="exp002_colab_"))
    saved = {k: os.environ.get(k) for k in ("DINO_WORK", "DATASET_DIR", "DINO_TRAIN", "EXP002_CACHE", "ACWM_PHYS_SRC")}
    try:
        os.environ.update({"DINO_WORK": str(tmp / "content/dinowm"), "DATASET_DIR": str(tmp / "content/dinowm/data"), "DINO_TRAIN": str(tmp / "drive/exp002_work/train_runs"),
                           "EXP002_CACHE": str(tmp / "drive/exp002_work/exp002_cache"), "ACWM_PHYS_SRC": str(tmp / "content/dinowm/acwm_phys_src")})
        pa = resolve_paths(base)
        check("colab: resolve_paths follows env vars", pa["train_base"] == tmp / "drive/exp002_work/train_runs" and pa["cache"] == tmp / "drive/exp002_work/exp002_cache"
              and pa["data_root"] == tmp / "content/dinowm/data" / base["data_impl"]["subdir"] and pa["acwm_src"] == tmp / "content/dinowm/acwm_phys_src")
        # copy_from_mount with tiny fake splits (no torch needed: metadata length injected)
        cfg = copy.deepcopy(base)
        cfg["data_impl"]["expected_episodes"] = {"train": 3, "test": 3}
        src = tmp / "mnt/shared"
        splits = ["ind_train", "a060_test"]
        for sp in splits:
            (src / sp).mkdir(parents=True)
            for i in range(3):
                (src / sp / ("ep%d.mp4" % i)).write_bytes(b"x" * (100 + i))
            (src / sp / "metadata.pt").write_bytes(b"meta")
        mc = lambda p: 3
        pa["data_root"] = tmp / "local/cloth_sim"
        os.environ["EXP002_DRIVE_MOUNT_SRC"] = str(tmp / "mnt")
        check("colab: find_mount_src finds a child folder containing all splits", drive.find_mount_src(cfg, splits) == src)
        # bracket + space folder names (expected real layout): default path, fallback search, no glob/shell interpretation
        fake_mount = tmp / "fakedrive"
        real = fake_mount / "MyDrive/dino-wm-videos/[data]/[cloth-sim] cloth_move"
        for sp in splits:
            (real / sp).mkdir(parents=True)
        os.environ.pop("EXP002_DRIVE_MOUNT_SRC", None)
        check("colab: default path with [brackets] and space is found", drive.find_mount_src(cfg, splits, mount=str(fake_mount)) == real)
        (fake_mount / "MyDrive/dino-wm-videos/[data]/[cloth-sim] cloth_move").rename(fake_mount / "MyDrive/dino-wm-videos/[data]/[cloth-sim] renamed [x]")
        found = drive.find_mount_src(cfg, splits, mount=str(fake_mount))
        check("colab: fallback search below MyDrive/dino-wm-videos finds the split folder after a rename", found is not None and found.name == "[cloth-sim] renamed [x]", str(found))
        (fake_mount / "MyDrive/dino-wm-videos/[data]/[cloth-sim] renamed [x]/ind_train").rmdir()
        check("colab: incomplete folder (missing split) is not accepted", drive.find_mount_src(cfg, splits, mount=str(fake_mount)) is None)
        os.environ["EXP002_DRIVE_MOUNT_SRC"] = str(tmp / "mnt")
        # copy from a source whose path has brackets/space (pathlib only)
        br = tmp / "mnt2/[data]/[cloth-sim] cloth_move"
        shutil.copytree(src, br)
        pb = dict(pa); pb["data_root"] = tmp / "local2/cloth_sim"
        orig_check2 = drive.check
        drive.check = lambda *a, **k: None
        try:
            check("colab: copy_from_mount works from a [bracket] path", drive.copy_from_mount(cfg, pb, splits, br, meta_counter=lambda p: 3) == 0 and (pb["data_root"] / "ind_train" / "ep0.mp4").exists())
        finally:
            drive.check = orig_check2
        # the notebook passes the path through os.environ: check that os.environ round trip + shell-free use keeps it intact
        os.environ["EXP002_DRIVE_MOUNT_SRC"] = str(br)
        check("colab: env var path with brackets/space is used first", drive.mount_src_candidates(cfg)[0] == br and drive.find_mount_src(cfg, splits) == br)
        os.environ["EXP002_DRIVE_MOUNT_SRC"] = str(tmp / "mnt")
        orig_check = drive.check
        drive.check = lambda *a, **k: None
        try:
            rc = drive.copy_from_mount(cfg, pa, splits, src, meta_counter=mc)
            check("colab: copy_from_mount completes", rc == 0 and all((pa["data_root"] / s / drive.OK_MARK).exists() for s in splits))
            mark = json.loads((pa["data_root"] / "ind_train" / drive.OK_MARK).read_text())
            check("colab: marker records bytes", mark.get("bytes") == 100 + 101 + 102 + 4, str(mark))
            m0 = (pa["data_root"] / "ind_train" / "ep0.mp4").stat().st_mtime_ns
            rc = drive.copy_from_mount(cfg, pa, splits, src, meta_counter=mc)
            check("colab: second run skips complete splits (file untouched)", rc == 0 and (pa["data_root"] / "ind_train" / "ep0.mp4").stat().st_mtime_ns == m0)
            (pa["data_root"] / "ind_train" / "ep1.mp4").write_bytes(b"y")          # truncated local file
            rc = drive.copy_from_mount(cfg, pa, splits, src, meta_counter=mc)
            check("colab: truncated local file is detected via recorded bytes and repaired", rc == 0 and (pa["data_root"] / "ind_train" / "ep1.mp4").stat().st_size == 101)
            (src / "ind_train" / "ep2.mp4").unlink()
            (pa["data_root"] / "ind_train" / "ep2.mp4").unlink()
            rc = drive.copy_from_mount(cfg, pa, splits, src, meta_counter=mc, force=True)
            check("colab: missing source episode -> incomplete, rc=3, no marker", rc == 3 and not (pa["data_root"] / "ind_train" / drive.OK_MARK).exists())
        finally:
            drive.check = orig_check
        # colab_tools
        ct_path = Path(__file__).resolve().parents[2] / "colab" / "colab_tools.py"
        spec = importlib.util.spec_from_file_location("colab_tools", ct_path)
        ct = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ct)
        rd = tmp / "drive/exp002_work/train_runs/outputs/exp002_x_s0/checkpoints"
        rd.mkdir(parents=True)
        for e in range(1, 8):
            (rd / ("model_%d.pth" % e)).write_bytes(b"ok%d" % e)
        (rd / "model_latest.pth").write_bytes(b"corrupt")
        check("colab_tools: prune without DONE keeps everything", ct.prune_run(rd.parent, [5, 10, 15], require_done=True) == [])
        (rd.parent / "DONE").write_text("ok\n")
        rem = ct.prune_run(rd.parent, [5, 10, 15], require_done=True)
        check("colab_tools: prune after DONE removes only non-kept epoch files", sorted(rem) == ["model_1.pth", "model_2.pth", "model_3.pth", "model_4.pth", "model_6.pth", "model_7.pth"]
              and (rd / "model_5.pth").exists() and (rd / "model_latest.pth").exists())
        # prepare (resume / repair / quarantine)
        loadable = lambda p: p.read_bytes().startswith(b"ok")
        tr = rd.parent.parent.parent
        (rd.parent / "DONE").unlink()
        for e_ in (6, 7):
            (rd / ("model_%d.pth" % e_)).write_bytes(b"ok%d" % e_)
        (rd / "model_latest.pth").write_bytes(b"ok7")
        r = ct.prepare_run(rd.parent, tr / "quarantine", loadable=loadable)
        check("colab_tools: valid checkpoint of an unfinished run is KEPT (resume), nothing moved", r["action"] == "resume" and (rd / "model_latest.pth").exists() and not (tr / "quarantine").exists(), str(r))
        (rd / "model_latest.pth").write_bytes(b"corrupt")
        r = ct.prepare_run(rd.parent, tr / "quarantine", loadable=loadable)
        check("colab_tools: corrupt model_latest.pth is replaced by the newest loadable epoch file, then resume", r["action"] == "repaired_resume" and r.get("restored_from") == "model_7.pth"
              and (rd / "model_latest.pth").read_bytes() == b"ok7", str(r))
        for f in rd.glob("model_*.pth"):
            f.write_bytes(b"corrupt")
        (rd.parent / "epoch_logs.jsonl").write_text("{}\n")
        r = ct.prepare_run(rd.parent, tr / "quarantine", loadable=loadable, now=time.localtime(0))
        dst = Path(r.get("to", "/nonexistent"))
        check("colab_tools: no loadable checkpoint -> checkpoints and epoch_logs MOVED to quarantine/<run>__<time> (fresh start), nothing deleted",
              r["action"] == "quarantined" and dst.parent == tr / "quarantine" and dst.name.startswith("exp002_x_s0__") and (dst / "checkpoints" / "model_latest.pth").exists()
              and (dst / "epoch_logs.jsonl").exists() and not (rd.parent / "checkpoints").exists(), str(r))
        check("colab_tools: after quarantine the run is 'fresh' (no checkpoint)", ct.prepare_run(rd.parent, tr / "quarantine", loadable=loadable)["action"] == "fresh")
        done_rd = tr / "outputs" / "exp002_done_s1"
        (done_rd / "checkpoints").mkdir(parents=True)
        (done_rd / "checkpoints" / "model_latest.pth").write_bytes(b"corrupt")
        (done_rd / "DONE").write_text("ok\n")
        smoke_rd = tr / "outputs" / "exp002_smoke_h1_b24"
        (smoke_rd / "checkpoints").mkdir(parents=True)
        (smoke_rd / "checkpoints" / "model_latest.pth").write_bytes(b"corrupt")
        check("colab_tools: DONE run and smoke run are never touched by prepare",
              ct.prepare_run(done_rd, tr / "quarantine", loadable=loadable)["action"] == "skip" and ct.prepare_run(smoke_rd, tr / "quarantine", loadable=loadable)["action"] == "skip"
              and (done_rd / "checkpoints" / "model_latest.pth").exists() and (smoke_rd / "checkpoints" / "model_latest.pth").exists())
        check("colab_tools: prune ignores the quarantine folder (not under outputs/)", all(r_.name != "quarantine" for r_ in ct._runs(tr)))
        # resume patch: epoch range of the patched train.py (budget), evaluated from the REAL patched source when the pinned train.py is available
        here = Path(__file__).resolve().parents[2]
        patch_txt = (here / "colab/train_resume.patch").read_text()
        check("resume patch text: contains the marker and the TOTAL-epoch loop, keeps optimizer state_dicts and action_encoder_optimizer",
              "[exp002 operational]" in patch_txt and "range(init_epoch, self.total_epochs + 1)" in patch_txt and '"action_encoder_optimizer"' in patch_txt and "load_state_dict" in patch_txt)
        from exp002 import trainrun as TRN
        check("trainrun: patch marker constant equals the one in the patch", TRN.RESUME_MARKER in patch_txt)
        pinned = None
        for cand in (os.environ.get("ACWM_PHYS_SRC"), str(here.parents[1] / "thread0")):
            if cand and (Path(cand) / ".git").exists():
                import subprocess
                r_ = subprocess.run(["git", "-C", cand, "show", "%s:train.py" % base["code"]["commit"]], capture_output=True, text=True)
                if r_.returncode == 0:
                    pinned = r_.stdout
                    break
        if pinned is None:
            skip("resume patch applied to the pinned train.py (epoch ranges)", "pinned commit train.py not available on this machine")
        else:
            import ast, subprocess, types
            d_ = tmp / "pin"
            d_.mkdir()
            (d_ / "train.py").write_text(pinned)
            ra = subprocess.run(["patch", "-p1", "--dry-run", "-i", str(here / "colab/train_resume.patch")], cwd=d_, capture_output=True, text=True)
            check("resume patch applies cleanly to the pinned train.py", ra.returncode == 0, ra.stdout[-200:] + ra.stderr[-200:])
            subprocess.run(["patch", "-p1", "-i", str(here / "colab/train_resume.patch")], cwd=d_, capture_output=True, text=True)
            tree = ast.parse((d_ / "train.py").read_text())
            run_fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run"][0]
            loop = [n for n in ast.walk(run_fn) if isinstance(n, ast.For)][0]
            rng_src = ast.unparse(loop.iter)
            def epochs_for(start_epoch, total):
                ns = {"self": types.SimpleNamespace(epoch=start_epoch, total_epochs=total), "init_epoch": start_epoch + 1}
                return list(eval(rng_src, {"range": range}, ns))
            fresh, res5, res15 = epochs_for(0, 15), epochs_for(5, 15), epochs_for(15, 15)
            check("patched loop: fresh run trains epochs 1..15 (same as the original)", fresh == list(range(1, 16)), rng_src)
            check("patched loop: resume after the epoch-5 checkpoint trains 6..15", res5 == list(range(6, 16)))
            check("patched loop: resume at the final epoch trains nothing", res15 == [])
            spe = 1857
            check("budget: epochs trained before (5) + after resume (10) x steps_per_epoch == fresh total (steps_per_epoch is an example value)", (5 + len(res5)) * spe == len(fresh) * spe)
            check("unpatched pinned loop would train epochs MORE (documents why the patch is needed)", "init_epoch + self.total_epochs" in pinned)
        check("resume wiring: overnight.sh calls prepare before train attempts, trainrun refuses an existing checkpoint with an unpatched train.py, notebook applies the patch and calls prepare",
              'colab_tools.py" prepare' in (here / "overnight.sh").read_text() and "Refusing" in (here / "src/exp002/trainrun.py").read_text()
              and "train_py_is_patched" in (here / "src/exp002/trainrun.py").read_text() and "'apply', PATCH" in (here / "colab/exp002_colab.ipynb").read_text().replace('\\"', '"')
              and "colab_tools.py prepare" in (here / "colab/exp002_colab.ipynb").read_text())
        pd = tmp / "src_patched"
        pd.mkdir()
        (pd / "train.py").write_text("x = 1\n")
        check("trainrun: unpatched train.py is detected", TRN.train_py_is_patched(pd) is False)
        (pd / "train.py").write_text("# " + TRN.RESUME_MARKER + "\n")
        check("trainrun: patched train.py is detected", TRN.train_py_is_patched(pd) is True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        os.environ.pop("EXP002_DRIVE_MOUNT_SRC", None)
        shutil.rmtree(tmp, ignore_errors=True)


def t_archive(base):
    """Archive mode of the Colab data stage (확정 기록 10): create / extract / verify / layouts / path escape / truncated or incomplete archives / interrupted run."""
    import os, tempfile, copy, tarfile, zipfile, io, json
    from pathlib import Path
    from exp002 import drive
    tmp = Path(tempfile.mkdtemp(prefix="exp002_arch_"))
    saved = {k: os.environ.get(k) for k in ("EXP002_WORK_ROOT", "EXP002_DATA_ARCHIVE", "EXP002_DRIVE_MOUNT_SRC", "EXP002_MAKE_ARCHIVE", "EXP002_ARCHIVE_DIRECT")}
    orig_check = drive.check
    drive.check = lambda *a, **k: None
    try:
        for k in saved:
            os.environ.pop(k, None)
        cfg = copy.deepcopy(base)
        cfg["data_impl"]["expected_episodes"] = {"train": 3, "test": 3}
        splits = ["ind_train", "a060_test"]
        mc = lambda p: 3

        def mkpaths(name):
            return {"data_root": tmp / name / "cloth_sim", "dataset_dir": tmp / name}

        def make_src(root):
            for sp in splits:
                (root / sp).mkdir(parents=True, exist_ok=True)
                for i in range(3):
                    (root / sp / ("ep%d.mp4" % i)).write_bytes(b"v" * (50 + i))
                (root / sp / "metadata.pt").write_bytes(b"meta")

        def mk_tar(path, entries, mode="w"):
            with tarfile.open(path, mode) as tf:
                for name, data in entries:
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))

        def all_files(prefix):
            out = []
            for sp in splits:
                for i in range(3):
                    out.append(("%s%s/ep%d.mp4" % (prefix, sp, i), b"v" * (50 + i)))
                out.append(("%s%s/metadata.pt" % (prefix, sp), b"meta"))
            return out
        # 1. make_archive from verified local data, atomic rename
        pa = mkpaths("local1")
        make_src(pa["data_root"])
        for sp in splits:
            drive._finish_split(cfg, pa["data_root"] / sp, sp, "test", mc)
        wr = tmp / "workroot"
        wr.mkdir()
        arc = wr / "cloth_sim_data.tar"
        ok = drive.make_archive(cfg, pa, splits, arc)
        names = tarfile.open(arc).getnames()
        check("archive: make_archive writes cloth_sim/<split>/<file> and leaves no .tmp", ok and "cloth_sim/ind_train/ep0.mp4" in names and not (wr / "cloth_sim_data.tar.tmp").exists()
              and not any(n.endswith(drive.OK_MARK) for n in names), str(names[:3]))
        # 2. extract into a fresh root, verification + recorded bytes
        pb = mkpaths("local2")
        os.environ["EXP002_WORK_ROOT"] = str(wr)
        check("archive: find_archive finds cloth_sim_data.tar in the work root", drive.find_archive(cfg) == arc)
        rc = drive.extract_archive(cfg, pb, splits, arc, meta_counter=mc)
        rec = json.loads((pb["data_root"] / "ind_train" / drive.OK_MARK).read_text())
        check("archive: extract + verify (mp4 == metadata == expected, bytes recorded)", rc == 0 and rec.get("source") == "archive" and rec.get("bytes") == 50 + 51 + 52 + 4 and (pb["data_root"] / "a060_test" / "ep2.mp4").exists(), str(rec))
        check("archive: local temporary archive copy removed after extraction", not (tmp / "local2" / "_archive_tmp" / "cloth_sim_data.tar").exists())
        m0 = (pb["data_root"] / "ind_train" / "ep0.mp4").stat().st_mtime_ns
        check("archive: second run skips complete splits", drive.extract_archive(cfg, pb, splits, arc, meta_counter=mc) == 0 and (pb["data_root"] / "ind_train" / "ep0.mp4").stat().st_mtime_ns == m0)
        # 3. prepare_data prefers the archive over the mount; with no archive it copies in parallel and then writes the archive once
        pc = mkpaths("local3")
        os.environ["EXP002_DRIVE_MOUNT_SRC"] = str(tmp / "nowhere")
        rc = drive.prepare_data(cfg, pc, splits, meta_counter=mc)
        check("archive: prepare_data uses the archive when present (mount not needed)", rc == 0 and json.loads((pc["data_root"] / "a060_test" / drive.OK_MARK).read_text())["source"] == "archive")
        wr2 = tmp / "workroot2"
        wr2.mkdir()
        os.environ["EXP002_WORK_ROOT"] = str(wr2)
        src = tmp / "mount_src"
        make_src(src)
        os.environ["EXP002_DRIVE_MOUNT_SRC"] = str(src)
        pd = mkpaths("local4")
        rc = drive.prepare_data(cfg, pd, splits, meta_counter=mc)
        check("archive: no archive -> parallel mount copy, verified, then archive created once", rc == 0 and (wr2 / "cloth_sim_data.tar").exists() and not (wr2 / "cloth_sim_data.tar.tmp").exists()
              and json.loads((pd["data_root"] / "ind_train" / drive.OK_MARK).read_text())["source"] == "drive_mount")
        pe = mkpaths("local5")
        os.environ["EXP002_DRIVE_MOUNT_SRC"] = str(tmp / "nowhere")
        check("archive: next session uses only the archive", drive.prepare_data(cfg, pe, splits, meta_counter=mc) == 0 and (pe["data_root"] / "ind_train" / "ep1.mp4").exists())
        os.environ["EXP002_MAKE_ARCHIVE"] = "0"
        wr3 = tmp / "workroot3"
        wr3.mkdir()
        os.environ["EXP002_WORK_ROOT"] = str(wr3)
        os.environ["EXP002_DRIVE_MOUNT_SRC"] = str(src)
        drive.prepare_data(cfg, mkpaths("local6"), splits, meta_counter=mc)
        check("archive: EXP002_MAKE_ARCHIVE=0 writes no archive", not (wr3 / "cloth_sim_data.tar").exists())
        os.environ.pop("EXP002_MAKE_ARCHIVE")
        os.environ.pop("EXP002_DRIVE_MOUNT_SRC")
        # 4. layout variants
        for label, prefix, kind in (("<split>/ at top level (tar)", "", "tar"), ("other/<split>/ (tar)", "data_dump/", "tar"), ("cloth_sim/<split>/ (zip)", "cloth_sim/", "zip")):
            a2 = tmp / ("variant_%s.%s" % (abs(hash(label)) % 10000, kind))
            if kind == "tar":
                mk_tar(a2, all_files(prefix))
            else:
                with zipfile.ZipFile(a2, "w") as zf:
                    for n, d in all_files(prefix):
                        zf.writestr(n, d)
            pv = mkpaths("lv_%s" % abs(hash(label)))
            check("archive: layout %s extracts and verifies" % label, drive.extract_archive(cfg, pv, splits, a2, meta_counter=mc) == 0 and (pv["data_root"] / "ind_train" / drive.OK_MARK).exists())
        a3 = tmp / "nosplit.tar"
        mk_tar(a3, [("readme/x.txt", b"hi"), ("foo/bar.mp4", b"x")])
        pn = mkpaths("lnosplit")
        check("archive: no split directories -> clear error, nothing written", drive.extract_archive(cfg, pn, splits, a3, meta_counter=mc) == 3 and not (pn["data_root"] / "ind_train" / drive.OK_MARK).exists())
        a4 = tmp / "mixed.tar"
        mk_tar(a4, all_files("cloth_sim/")[:4] + all_files("other/")[4:])
        check("archive: inconsistent top-level layout is rejected", drive.extract_archive(cfg, mkpaths("lmixed"), splits, a4, meta_counter=mc) == 3)
        # 5. path escape
        outside = tmp / "evil.txt"
        for label, name in (("../ in the path", "cloth_sim/ind_train/../../../evil.txt"), ("absolute path", str(outside))):
            a5 = tmp / ("escape_%d.tar" % abs(hash(label)))
            mk_tar(a5, all_files("cloth_sim/") + [(name, b"owned")])
            pf = mkpaths("lesc_%d" % abs(hash(label)))
            rc = drive.extract_archive(cfg, pf, splits, a5, meta_counter=mc)
            check("archive: %s is refused, no file written outside / no marker" % label, rc == 3 and not outside.exists() and not (pf["data_root"] / "ind_train" / drive.OK_MARK).exists())
        a6 = tmp / "symlink.tar"
        with tarfile.open(a6, "w") as tf:
            for n, d in all_files("cloth_sim/"):
                ti = tarfile.TarInfo(n); ti.size = len(d); tf.addfile(ti, io.BytesIO(d))
            li = tarfile.TarInfo("cloth_sim/ind_train/link.mp4"); li.type = tarfile.SYMTYPE; li.linkname = "/etc/passwd"; tf.addfile(li)
        check("archive: symlink member inside a split is refused", drive.extract_archive(cfg, mkpaths("lsym"), splits, a6, meta_counter=mc) == 3)
        # 6. partial archives
        data = arc.read_bytes()
        a7 = tmp / "truncated.tar"
        a7.write_bytes(data[: len(data) // 2])
        p7 = mkpaths("ltrunc")
        check("archive: truncated archive is rejected (rc 3); the split cut off by the truncation gets no completion marker", drive.extract_archive(cfg, p7, splits, a7, meta_counter=mc) == 3 and not (p7["data_root"] / "a060_test" / drive.OK_MARK).exists())
        a8 = tmp / "missing_episode.tar"
        mk_tar(a8, [e for e in all_files("cloth_sim/") if not e[0].endswith("ind_train/ep2.mp4")])
        p8 = mkpaths("lmiss")
        check("archive: archive with a missing episode (mp4 count != metadata) is rejected, no marker", drive.extract_archive(cfg, p8, splits, a8, meta_counter=mc) == 3 and not (p8["data_root"] / "ind_train" / drive.OK_MARK).exists())
        a9 = tmp / "missing_split.tar"
        mk_tar(a9, [e for e in all_files("cloth_sim/") if "a060_test" not in e[0]])
        check("archive: archive without a needed split is rejected", drive.extract_archive(cfg, mkpaths("lmsplit"), splits, a9, meta_counter=mc) == 3)
        a10 = tmp / "garbage.tar"
        a10.write_bytes(b"not a tar" * 100)
        check("archive: non-archive file gives a clear error (rc 3)", drive.extract_archive(cfg, mkpaths("lgarb"), splits, a10, meta_counter=mc) == 3)
        # 7. interrupted extraction is safe to re-run (leftover .part + a stale short file)
        pg = mkpaths("linterrupt")
        (pg["data_root"] / "ind_train").mkdir(parents=True)
        (pg["data_root"] / "ind_train" / "ep0.mp4.part").write_bytes(b"half")
        (pg["data_root"] / "ind_train" / "ep1.mp4").write_bytes(b"short")
        rc = drive.extract_archive(cfg, pg, splits, arc, meta_counter=mc)
        check("archive: re-run after an interruption repairs short files and removes .part leftovers", rc == 0 and (pg["data_root"] / "ind_train" / "ep1.mp4").stat().st_size == 51
              and not list((pg["data_root"] / "ind_train").glob("*.part")))
    finally:
        drive.check = orig_check
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


def run(base_cfg=None):
    print("=" * 100)
    print("SELFTEST on synthetic data. The numbers below are NOT experiment results.")
    print("=" * 100)
    base = base_cfg or load_config()
    for name, fn, args in (("stats", t_stats, ()), ("directions", t_directions, ()), ("config", t_config, (base,))):
        try:
            fn(*args)
        except Exception:
            traceback.print_exc()
            REC["fail"].append(name + " (exception)")
    for name, fn in (("budget/manifest", t_budget_manifest), ("output-space probe", t_output_space), ("diag 4c", t_diag), ("q3 pure", t_q3_pure), ("vram poller", lambda b: t_vram_poller()), ("static/slices", t_static_and_slices), ("colab", t_colab), ("archive", t_archive)):
        try:
            fn(base)
        except Exception:
            traceback.print_exc()
            REC["fail"].append(name + " (exception)")
    ctx = {}
    try:
        ctx = t_q1_aggregate(base)
        if ctx:
            t_h_b(ctx)
    except Exception:
        traceback.print_exc()
        REC["fail"].append("q1/aggregate synthetic (exception)")
    finally:
        if ctx.get("tmp"):
            shutil.rmtree(ctx["tmp"], ignore_errors=True)
    try:
        t_torch_q3(base)
    except Exception:
        traceback.print_exc()
        REC["fail"].append("torch/q3 (exception)")
    print("=" * 100)
    print("selftest summary: passed=%d failed=%d skipped=%d" % (len(REC["pass"]), len(REC["fail"]), len(REC["skipped"])))
    for n in REC["skipped"]:
        print("  skipped:", n)
    for n in REC["fail"]:
        print("  FAILED:", n)
    print("(synthetic numbers are not results; skipped sub-tests were NOT run)")
    return 0 if not REC["fail"] else 1
