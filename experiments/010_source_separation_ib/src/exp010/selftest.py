"""CPU self-test of experiment 010 on SYNTHETIC data. Numbers printed here are NOT experiment results.
Covers: TOST statistics on planted data, task-seed independence, the 005 metric code driven through 010's per-task-seed wrapper, four planted representations."""
from __future__ import annotations

import copy
import os
import traceback

import numpy as np

from . import stats as ST
from . import stages as S
from . import base as B

C2 = B.C2

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


def t_colab(cfg, c2):
    print("D  Colab notebook / preflight / code_version (CPU, no Colab)")
    import ast
    import json
    import tempfile
    from pathlib import Path
    from . import stages as S
    nbp = B.EXP_DIR / "colab" / "exp010_colab.ipynb"
    if not nbp.exists():
        REC["skipped"].append("colab notebook missing")
        print("  [skip] colab/exp010_colab.ipynb not built")
        return
    nb = json.loads(nbp.read_text())
    cells = nb["cells"]
    src = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    check("notebook JSON is nbformat 4 with code cells", nb["nbformat"] == 4 and len(src) > 10, "%d code cells" % len(src))
    bad = []
    for i, t in enumerate(src):
        try:
            ast.parse(t)
        except SyntaxError as e:
            bad.append((i, str(e)))
    check("every code cell parses as Python", not bad, str(bad))
    allsrc = "\n".join(src)
    check("three approval gates default to False", all(("%s = False" % g) in allsrc for g in ("APPROVED_REGEN_010", "APPROVED_PILOT_010", "APPROVED_MAIN_010")))
    check("no embedded code snapshot (git clone only)", "base64" not in allsrc and "git', 'clone'" in allsrc)
    check("MPLBACKEND=Agg set for sub-processes", "'MPLBACKEND': 'Agg'" in allsrc)
    check("separate folders from 006/009 (mylab010, venv010, exp010_work)", all(x in allsrc for x in ("/content/mylab010", "/content/venv010", "exp010_work")) and "/content/mylab'" not in allsrc)
    run0 = C2.run_name(cfg["base"]["condition"], 0)
    check("checkpoint run name used in the notebook equals exp002 run_name(condition, 0) (006 prefix lesson)", ("want_run = '%s'" % run0) in allsrc, run0)
    check("regeneration calls 004 enc_cache and 005 feat through run.py (no source edit)", "'enc_cache'" in allsrc and "stage='feat'" in allsrc)
    check("main/pilot cells do not pass --task-seeds/--delta overrides to the main run", "run010('heads', mode='main')" in allsrc and "--delta" not in allsrc)
    # compare_logs / plan_fields from cell R4 against the committed original logs
    r4 = [t for t in src if t.lstrip().startswith("# R4.")][0]
    ns = {}
    head = r4.split("REF = {")[0]
    exec(head.replace("import re, glob, time, hashlib\nimport numpy as np\n", "import re, glob, time, hashlib\n"), ns)
    e4 = B.EXP_DIR.parent / "004_depth_truncation" / "logs" / "colab_20261109" / "enc_cache_20261109-092707.log"
    e5 = B.EXP_DIR.parent / "005_inductive_bias_probe" / "logs" / "colab_20261109" / "feat_20261109-112051.log"
    if e4.exists() and e5.exists():
        t4, t5 = e4.read_text(errors="replace"), e5.read_text(errors="replace")
        r = ns["compare_logs"](t4, t4)
        check("compare_logs parses the original 004 enc_cache log (plan, 3 consistency lines, 1800/1800)", len(r) == 5 and all(x[3] != "DIFFERENT" for x in r) and r[0][1] is not None, str([x[0] for x in r]))
        r = ns["compare_logs"](t4, t4.replace("k_grid=[0, 2, 4, 6, 8, 10, 12]", "k_grid=[0, 2, 4, 6, 8, 10]"))
        check("compare_logs flags a changed k_grid", any(x[0] == "enc_cache plan" and x[3] == "DIFFERENT" for x in r))
        r = ns["compare_logs"](t5, t5)
        check("compare_logs parses the original 005 feat log (plan, 1800/1800)", len(r) == 2 and r[0][1] is not None and all(x[3] != "DIFFERENT" for x in r), str(r))
        r = ns["compare_logs"](t5, t5.replace("['pred_out6']", "['pred_out6', 'pred_out5']"))
        check("compare_logs flags a changed layer list", any(x[3] == "DIFFERENT" for x in r))
    else:
        REC["skipped"].append("original 004/005 logs missing")
    # preflight and code_version
    with tempfile.TemporaryDirectory() as td:
        env_keys = {"EXP004_CACHE": td, "EXP005_CACHE": td, "EXP002_CACHE": td}
        old = {k: os.environ.get(k) for k in env_keys}
        os.environ.update(env_keys)
        try:
            p2 = B.resolve_paths(cfg, cfg["_c5"], c2)
            miss = S.missing_inputs(cfg, p2)
            check("missing_inputs lists the 4 feature files and the 002 cache when absent", len(miss) == 5, str(len(miss)))
            for l in S.plan_of(cfg)["layers"]:
                for ctl, _ in S.MODELS:
                    pth = S.FE.layer_path(p2, l, ctl, cfg["_c5"]["exp4"]["cache_suffix"])
                    pth.parent.mkdir(parents=True, exist_ok=True)
                    pth.write_bytes(b"x")
            miss2 = S.missing_inputs(cfg, p2)
            check("missing_inputs only reports the 002 cache once the 4 files exist", len(miss2) == 1 and "002" in miss2[0], str(miss2))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    os.environ["EXP010_CODE_REPO_URL"], os.environ["EXP010_CODE_REF"] = "u", "r"
    cv = S.code_version(B.EXP_DIR)
    check("code_version has git_head/git_status_dirty/repo_url/requested_ref and reads the env", set(cv) == {"git_head", "git_status_dirty", "repo_url", "requested_ref"} and cv["repo_url"] == "u" and cv["requested_ref"] == "r")
    os.environ.pop("EXP010_CODE_REPO_URL"), os.environ.pop("EXP010_CODE_REF")
    # config values the regeneration preflight (cell R1) expects from the 004 / 005 / 002 config files
    import yaml
    c4 = yaml.safe_load((B.EXP_DIR.parent / "004_depth_truncation" / "config.yaml").read_text())
    c5 = yaml.safe_load((B.EXP_DIR.parent / "005_inductive_bias_probe" / "config.yaml").read_text())
    check("004/005 config values equal the ones expected by cell R1",
          c4["encoder"]["k_grid"] == [0, 2, 4, 6, 8, 10, 12] and c4["encoder"]["random_init_control"] is True and c4["encoder"]["source_train_seed"] == 0
          and c5["feat"]["train_seed"] == 0 and c5["feat"]["random_init_seed"] == 1234 and c2["probe_impl"]["random_init"]["seed"] == 0)
    pl = json.loads((B.EXP_DIR.parent / "005_inductive_bias_probe" / "results" / "plan.json").read_text())
    check("005 results/plan.json (copied for `feat`) has exactly the layer pred_out6", [(l["name"], l["module"], l["depth"]) for l in pl["layers"]] == [("pred_out6", "predictor", 6)])


def run(cfg, c2):
    for name, fn in (("A", t_stats), ("C", lambda: t_config(cfg)), ("B", lambda: t_pipeline(cfg, c2)), ("D", lambda: t_colab(cfg, c2))):
        try:
            fn()
        except Exception:
            REC["fail"].append(name + " (exception)")
            traceback.print_exc()
    print("selftest summary: passed=%d failed=%d skipped=%d" % (len(REC["pass"]), len(REC["fail"]), len(REC["skipped"])))
    if REC["fail"]:
        print("FAILED:", REC["fail"])
    return 1 if REC["fail"] else 0
