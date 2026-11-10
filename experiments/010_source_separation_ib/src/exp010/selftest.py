"""CPU self-test of experiment 010 on SYNTHETIC data. Numbers printed here are NOT experiment results.
Covers: TOST statistics on planted data, task-seed independence, the 005 metric code driven through 010's per-task-seed wrapper, four planted representations."""
from __future__ import annotations

import copy
import traceback

import numpy as np

from . import stats as ST
from . import stages as S
from . import base as B

REC = {"pass": [], "fail": [], "skipped": []}


def check(name, cond, detail=""):
    (REC["pass"] if cond else REC["fail"]).append(name)
    print("  [%s] %s %s" % ("ok" if cond else "FAIL", name, detail))


def t_stats():
    print("A  TOST and tables on planted data")
    rng = np.random.RandomState(0)
    r = ST.tost(rng.randn(5) * 0.005 + 0.0, 0.05)
    check("equivalence supported when diffs ~ 0 with small sd", r["status"] == "supported", str(round(r["ci90"][0], 4)) + ".." + str(round(r["ci90"][1], 4)))
    r = ST.tost(np.array([0.2, 0.21, 0.19]), 0.05)
    check("refuted when the whole 90% CI is outside +-delta", r["status"] == "refuted")
    r = ST.tost(np.array([0.0, 0.1, -0.1, 0.05]), 0.05)
    check("inconclusive when the CI straddles delta", r["status"] == "inconclusive")
    check("fewer than 3 units -> not_evaluable", ST.tost(np.array([0.0, 0.0]), 0.05)["status"] == "not_evaluable")
    check("delta null -> not_evaluable", ST.tost(np.zeros(4), None)["status"] == "not_evaluable")
    check("identical diffs (sd 0) inside delta -> supported", ST.tost(np.full(3, 0.01), 0.05)["status"] == "supported")
    rows = [{"layer": "L", "model": "ours", "metric": "D-IB", "pair_group": "all", "task_seed": 1, "seed": 0, "score": 0.5}] * 2
    try:
        ST.unit_table(rows)
        check("duplicate (key, task_seed, head seed) rows raise", False)
    except ValueError:
        check("duplicate (key, task_seed, head seed) rows raise", True)


def t_pipeline(cfg, c2):
    print("B  per-task-seed metrics + analysis on planted representations (005 code re-used)")
    try:
        import torch
    except ImportError:
        REC["skipped"].append("B")
        print("  [skipped(no torch)]")
        return
    from exp005 import selftest as T5
    from exp005 import stages as S5
    c5 = copy.deepcopy(cfg["_c5"])
    c, cc2 = T5.small_cfgs(c5, c2)
    rows, frames, mf = T5.synth_cache(cc2)
    dev = torch.device("cpu")
    Etr = sum(r["part"] == "train" for r in rows)
    nF = len(frames)
    ac, hc = cc2["probe_impl"]["attentive"], c["head"]
    st = T5.synth_tokens(None, rows, frames, "state")
    nz = T5.synth_tokens(None, rows, frames, "noise", seed=9)
    names = [r["name"] for r in cfg["representations"]]
    plan = S.plan_of(cfg)
    reps = {(names[0], "ours"): st, (names[0], "random"): nz, (names[1], "ours"): st, (names[1], "random"): nz}   # two planted representations, state vs noise
    tseeds, hseeds = [2000, 2001, 2002], [0, 1]
    tasks, preds = {}, {}
    cfg_t = copy.deepcopy(cfg)
    cfg_t["_c5"] = c
    for ts in tseeds:
        ci = B.cfg_for_task_seed(c, ts)
        T, _ = S5.build_tasks(ci, cc2, rows, mf, frames, [0, 1, 2, 3], [4, 5, 6, 7])
        tasks[ts] = T
        P = {}
        bs = S5.base_sources(ci, T, dev)
        for hs in hseeds:
            for nm, src in bs.items():
                P[("base", nm, hs)] = S5.predict_model(src, T, {"A"}, hs, ac, hc)
            for (lay, m), arr in reps.items():
                P[(lay, m, hs)] = S5.predict_model(S5._sources(ci, T, arr, Etr, nF, dev), T, {"A"}, hs, ac, hc)
        preds[ts] = P
    check("task seeds give different tasks (labels / D_k differ) but identical evaluation inputs",
          not np.array_equal(tasks[2000]["labels_a"], tasks[2001]["labels_a"]) and not np.array_equal(tasks[2000]["dk_a"], tasks[2001]["dk_a"]) and
          np.array_equal(tasks[2000]["row"], tasks[2001]["row"]) and np.array_equal(tasks[2000]["row"], tasks[2002]["row"]))
    ib = S.collect_ib_rows(cfg_t, plan, tasks, preds, hseeds)
    metrics = {r["metric"] for r in ib}
    check("only 5-A/5-C metrics are produced (no calibration / law rows needed)", metrics == {"R-IB", "D-IB", "D-IB_area=", "D-IB_area≠", "R-IB_fold"}, str(sorted(metrics)))
    check("rows tagged with all 3 task seeds, unique per (task seed, layer, model, metric, group, head seed)",
          {r["task_seed"] for r in ib} == set(tseeds) and len({(r["task_seed"], r["layer"], r["model"], r["metric"], r["pair_group"], r["seed"]) for r in ib}) == len(ib))
    A = S.analyze(cfg, plan, ib)
    check("analysis has representation scores, H-f TOST for the 4 layer/model cells, pairwise table", len(A["representation_scores"]) > 0 and len(A["H_f_tost"]) == 4 and len(A["pairwise_differences"]) > 0)
    rt = {(r["layer"], r["model"], r["metric"], r["pair_group"]): r for r in A["representation_scores"]}
    k = (names[0], "ours", "D-IB", "all")
    check("planted state tokens score higher D-IB than noise tokens (all pairs)", rt[k]["mean"] > rt[(names[0], "random", "D-IB", "all")]["mean"] + 0.1,
          "(state %.3f noise %.3f)" % (rt[k]["mean"], rt[(names[0], "random", "D-IB", "all")]["mean"]))
    pw = [p for p in A["pairwise_differences"] if p["a"] == "%s/ours" % names[0] and p["b"] == "%s/ours" % names[1] and p["metric"] == "D-IB" and p["pair_group"] == "all"][0]
    check("identical planted representations in two layers -> difference exactly 0 and TOST supported", abs(pw["mean"]) < 1e-12 and pw["status"] == "supported")
    pw2 = [p for p in A["pairwise_differences"] if p["a"] == "%s/ours" % names[0] and p["b"] == "%s/random" % names[0] and p["metric"] == "D-IB" and p["pair_group"] == "all"][0]
    check("state vs noise -> TOST does not claim equivalence", pw2["status"] != "supported", pw2["status"])
    A2 = S.analyze(cfg, plan, ib)
    check("analysis deterministic", A2["H_f_tost"] == A["H_f_tost"])


def t_config(cfg):
    print("C  config explicitness")
    check("task seeds >= 3, head seeds explicit, eval n_inputs 900, delta present", len(cfg["seeds"]["task_a"]) >= 3 and all(isinstance(s, int) for s in cfg["seeds"]["head"]) and cfg["eval"]["n_inputs"] == 900 and cfg["tost"]["delta"] == 0.05)
    check("four representations (2 layers x ours/random) defined", len(cfg["representations"]) == 2 and [r["name"] for r in cfg["representations"]] == ["enc_k12", "pred_out6"])
    check("005 normalization read from 005 config (D const / R iid)", cfg["_c5"]["normalization"]["noninf"] == {"R": "iid", "D": "const"})
    check("first task seed equals 005 seeds.tasks_a (replicates 005 tasks)", cfg["seeds"]["task_a"][0] == B.B5.load_config((B.EXP_DIR / cfg["base"]["config005"]).resolve())["seeds"]["tasks_a"])


def run(cfg, c2):
    for name, fn in (("A", t_stats), ("C", lambda: t_config(cfg)), ("B", lambda: t_pipeline(cfg, c2))):
        try:
            fn()
        except Exception:
            REC["fail"].append(name + " (exception)")
            traceback.print_exc()
    print("selftest summary: passed=%d failed=%d skipped=%d" % (len(REC["pass"]), len(REC["fail"]), len(REC["skipped"])))
    if REC["fail"]:
        print("FAILED:", REC["fail"])
    return 1 if REC["fail"] else 0
