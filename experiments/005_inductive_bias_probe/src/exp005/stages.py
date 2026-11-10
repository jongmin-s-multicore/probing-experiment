"""Stage implementations of experiment 005 (plan / tasks / feat / heads / metrics / report). Heavy parts (feat, heads on tokens) need GPU, the 002 cache,
the 004 cache and a 002 condition-(b) checkpoint; none of it ran on the authoring machine except through toy data in selftest."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import time
from pathlib import Path

import numpy as np

from exp002 import cache as CA
from exp002 import evalgrid as EG
from exp002 import model_io as M
from exp002 import probes as PR
from . import branch as BR
from . import feats as FE
from . import ibmetrics as IB
from . import tasks as TK
from . import verdicts as VD
from .base import as002, log, read_json, write_json, run_dir002, ConfigNull

PRED_DIR = "predictions"


# ------------------------------------------------------------------------------------------------ plan
def load_summary(cfg, paths, path=None):
    p = Path(path) if path else (paths["exp"] / cfg["exp4"]["summary"]).resolve()
    return (read_json(p) if p.exists() else None), p


def plan_stage(cfg, paths, summary_path=None, dry_run=False):
    summ, p = load_summary(cfg, paths, summary_path)
    plan = BR.select(summ, cfg["branch"]["exp1_negative"])
    plan["summary_path"] = str(p)
    plan["pilot"] = bool(cfg["pilot"]) or bool(plan.get("pilot"))
    log("plan from %s: rows=%s hold=%s layers=%s layer_subexps=%s unresolved=%s pilot=%s" % (
        p, plan["rows_matched"], plan["hold"], [l["name"] for l in plan["layers"]], plan["layer_subexps"], plan["unresolved"], plan["pilot"]))
    if summ is None and not dry_run:
        raise FileNotFoundError("experiment 4 summary.json not found at %s - run experiment 4 aggregate first" % p)
    if not dry_run:
        write_json(plan, paths["results_dir"] / "plan.json")
    return plan


def read_plan(paths):
    p = paths["results_dir"] / "plan.json"
    if not p.exists():
        raise FileNotFoundError("results/plan.json missing - run `python run.py plan` first")
    return read_json(p)


# ------------------------------------------------------------------------------------------------ tasks
def build_tasks(cfg, cfg2, rows, mask_feat, frames, early_pos, late_pos):
    """Deterministic evaluation inputs + tasks for 5-A, 5-B, 5-D (numpy)."""
    nF = len(frames)
    levels = list(cfg2["data"]["levels"])
    roles = {l: r for r, ls in cfg2["data"]["roles"].items() for l in ls}
    for l in cfg2["data"]["train_levels_b"]:
        roles[l] = "trained"
    sd = cfg["seeds"]
    ev = TK.eval_inputs(rows, frames, early_pos, late_pos, cfg["eval"]["n_inputs"], sd["eval"])
    mf = np.asarray(mask_feat)
    lvl = np.array([levels.index(rows[r]["level"]) for r in ev["row"]])
    role = np.array([roles[rows[r]["level"]] for r in ev["row"]])
    s_eval = np.array([rows[r]["nominal_log_area"] for r in ev["row"]])
    area_feat = mf[ev["row"], ev["frame"]]                                          # (n,3) [area px, bbox w, bbox h]
    q = TK.area_quantile_bins(area_feat[:, 0], cfg["q"]["n_bins"])
    pool = TK.train_pool(rows, nF)
    Etr = sum(r["part"] == "train" for r in rows)
    s_all = np.array([r["nominal_log_area"] for r in rows])
    pool_row, pool_fpos = pool // nF, pool % nF
    lvl_flat = np.full(Etr * nF, -1)
    lvl_flat[pool] = [levels.index(rows[r]["level"]) for r in pool_row]
    s_flat = np.zeros(Etr * nF)
    s_flat[pool] = s_all[pool_row]
    area_flat = np.zeros((Etr * nF, 3), np.float32)
    area_flat[pool] = mf[pool_row, np.array(frames)[pool_fpos]]
    ib = cfg["ib"]
    labels, dk_a, pool_levels_used = TK.tasks_a(rows, nF, levels, ib["n_tasks"], ib["n_per_task"], cfg["ib"]["pool_levels"] or levels, sd["tasks_a"])
    cb = cfg["calib"]
    sin = cfg["sinusoid"]
    s_lo, s_hi = float(min(np.log(cfg2["data_impl"]["area_ratio"][l]) for l in levels)), float(max(np.log(cfg2["data_impl"]["area_ratio"][l]) for l in levels))
    prm = TK.sinusoid_params(cb["n_tasks"], sin["n_components"], sin["freq_min"], sin["freq_max"], sd["sinus"])
    dk_b = TK.tasks_b(rows, nF, levels, cfg2["data"]["train_levels_b"], cb["n_tasks"], cb["n_per_task"], sd["tasks_b"])
    g_eval = TK.sinusoid_eval(prm, s_eval, s_lo, s_hi)
    g_dk = np.stack([TK.sinusoid_eval({k: v[i:i + 1] for k, v in prm.items()}, s_flat[dk_b[i]], s_lo, s_hi)[0] for i in range(cb["n_tasks"])])
    law = cfg["law"]
    n_lab = max(int(law["min_labels"]), int(math.ceil(float(law["label_fraction"]) * len(pool))))
    ids_d = [TK.label_subset_d(rows, nF, levels, cfg2["data"]["train_levels_b"], n_lab, sd["d"])]
    names_d = ["labels_" + "+".join(cfg2["data"]["train_levels_b"])]
    for j, combo in enumerate(law["combos"]):
        ids_d.append(TK.label_subset_d(rows, nF, levels, combo, n_lab, sd["d"] + 1 + j))
        names_d.append("labels_" + "+".join(combo))
    T = {"row": ev["row"], "fpos": ev["fpos"], "frame": ev["frame"], "kind": ev["kind"], "lvl": lvl, "role": role, "s_eval": s_eval, "q": q,
         "is_ind": role == "trained", "is_ood": np.isin(role, cfg["ib"]["ood_roles"]), "is_extrap": np.isin(role, cfg["calib"]["extrap_roles"]),
         "area_eval": area_feat, "lvl_flat": lvl_flat, "s_flat": s_flat, "area_flat": area_flat, "pool": pool,
         "labels_a": labels, "dk_a": dk_a, "sin_f": prm["f"], "sin_phi": prm["phi"], "sin_a": prm["a"], "dk_b": dk_b, "g_dk": g_dk, "g_eval": g_eval,
         "ids_d": np.stack(ids_d), "s_range": np.array([s_lo, s_hi])}
    meta = {"levels": levels, "pool_levels_used_5A": pool_levels_used, "n_pool_train_inputs": int(len(pool)), "n_labels_5D": n_lab, "names_d": names_d,
            "n_eval": int(len(ev["row"])), "eval_per_level": {l: int((lvl == i).sum()) for i, l in enumerate(levels)}, "eval_kind_counts": {"early": int((ev["kind"] == 0).sum()), "late": int((ev["kind"] == 1).sum())},
            "n_q_bins": cfg["q"]["n_bins"], "ood_roles": list(cfg["ib"]["ood_roles"]), "frames": list(map(int, frames))}
    return T, meta


def tasks_stage(cfg, cfg2, paths, suffix="", dry_run=False):
    ib, cb = cfg["ib"], cfg["calib"]
    log("tasks plan: eval inputs %d (half early / half late), 5-A tasks %d x %d inputs, 5-B tasks %d x %d inputs (sum of %d sinusoids, f in [%s, %s] cycles per unit log-area), 5-D heads %d (labels %s)" % (
        cfg["eval"]["n_inputs"], ib["n_tasks"], ib["n_per_task"], cb["n_tasks"], cb["n_per_task"], cfg["sinusoid"]["n_components"], cfg["sinusoid"]["freq_min"], cfg["sinusoid"]["freq_max"],
        1 + len(cfg["law"]["combos"]), "fraction %s (min %s)" % (cfg["law"]["label_fraction"], cfg["law"]["min_labels"])))
    if dry_run:
        return None
    C = CA.load(as002(paths), suffix)
    W = EG.windows(cfg2)
    frames = list(W["union"])
    T, meta = build_tasks(cfg, cfg2, C["rows"], np.asarray(C["mask_feat"]), frames, [frames.index(f) for f in W["early"]], [frames.index(f) for f in W["late"]])
    out = paths["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "tasks.npz", **T)
    meta["config_sha256"] = hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest()
    write_json(meta, out / "tasks_meta.json")
    log("tasks done: %s" % {k: meta[k] for k in ("n_eval", "pool_levels_used_5A", "n_pool_train_inputs", "n_labels_5D")})
    return T, meta


def load_tasks(paths):
    p = paths["results_dir"] / "tasks.npz"
    if not p.exists():
        raise FileNotFoundError("results/tasks.npz missing - run `python run.py tasks` first")
    z = np.load(p)
    return {k: z[k] for k in z.files}, read_json(paths["results_dir"] / "tasks_meta.json")


# ------------------------------------------------------------------------------------------------ feat (predictor layers)
def feat_stage(cfg, cfg2, paths, plan, suffix="", dry_run=False):
    pl = [l for l in plan["layers"] if l["module"] == "predictor"]
    el = [l for l in plan["layers"] if l["module"] == "encoder"]
    log("feat plan: predictor layers %s (extracted here, real + random-init predictor), encoder layers %s (read from the 004 cache %s)" % (
        [l["name"] for l in pl], [l["name"] for l in el], FE.enc_cache_dir(paths, suffix)))
    if dry_run:
        return None
    for l in el:
        for ctl in ("real", "rand"):
            p = FE.layer_path(paths, l, ctl, suffix)
            if not p.exists():
                raise FileNotFoundError("%s missing - encoder layer k=%d must be in the 004 k_grid and `python run.py enc_cache` (004) must have run (random-init control on)" % (p, l["depth"]))
    if not pl:
        return []
    import torch
    from exp004 import deep as DP
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = CA.load(as002(paths), suffix)
    seed = int(cfg["feat"]["train_seed"])
    rdir = run_dir002(paths, cfg, seed)
    if not (rdir / "DONE").exists():
        raise FileNotFoundError("002 run %s has no DONE" % rdir)
    model, tcfg, ep = M.load_world_model(paths["acwm_src"], rdir, dev)
    if M.encoder_fingerprint(model.encoder) != C["meta"]["encoder_fingerprint"]:
        raise RuntimeError("encoder differs from the Q1 cache encoder")
    H = int(tcfg.num_hist)
    mean, std, _ = M.action_stats_for_run(cfg2, rdir, tcfg)
    W = EG.windows(cfg2)
    models = {"real": model, "rand": DP.random_init_predictor(model, int(cfg["feat"]["random_init_seed"]) + seed)}
    for l in pl:
        if l["depth"] > M.n_blocks(model):
            raise ValueError("predictor layer %s deeper than the model's %d blocks" % (l["name"], M.n_blocks(model)))
    return FE.extract_layers(models, C, H, list(W["union"]), pl, paths["cache"], mean, std, dev, int(cfg["feat"]["batch_episodes"]), suffix, label="feat")


# ------------------------------------------------------------------------------------------------ heads
def _sources(cfg, T, feat_arr, Etr, nF, dev, ctl_arch="attn"):
    import torch
    from .heads import Src
    a = np.asarray(feat_arr[:Etr]).reshape(Etr * nF, *feat_arr.shape[2:])
    tr = torch.as_tensor(a).to(dev)
    ev = torch.as_tensor(np.asarray(feat_arr[T["row"], T["fpos"]])).to(dev)
    return Src(tr, ev, "attn")


def base_sources(cfg, T, dev):
    """Layer-independent models: oracle (true s), area/bbox head."""
    import torch
    from .heads import Src
    from exp002 import probes as PR2
    s_tr = torch.as_tensor(T["s_flat"][:, None, None].astype(np.float32)).to(dev)
    s_ev = torch.as_tensor(T["s_eval"][:, None, None].astype(np.float32)).to(dev)
    af = PR2.mask_baseline_features(T["area_flat"]).astype(np.float32)
    ae = PR2.mask_baseline_features(T["area_eval"]).astype(np.float32)
    pool = T["pool"]
    mu, sd = af[pool].mean(0), af[pool].std(0) + 1e-6
    a_tr = torch.as_tensor(((af - mu) / sd)[:, None, :].astype(np.float32)).to(dev)
    a_ev = torch.as_tensor(((ae - mu) / sd)[:, None, :].astype(np.float32)).to(dev)
    return {"oracle_linear": Src(s_tr, s_ev, "linear"), "oracle_mlp": Src(s_tr, s_ev, "mlp5"), "area": Src(a_tr, a_ev, "attn", no_input_norm=True)}


def predict_model(src, T, need, seed, ac, hc, max_tasks=None, tag=""):
    """All heads of one model/seed. Returns {'A': (Ta,n) 0/1, 'B': (Tb,n), 'D': (nD,n)} for the requested sub-experiments."""
    from .heads import fit_predict
    n_eval = len(T["s_eval"])
    out = {}
    t0 = time.time()
    if "A" in need:
        Ta = len(T["dk_a"]) if not max_tasks else min(max_tasks, len(T["dk_a"]))
        YA = np.zeros((len(T["dk_a"]), n_eval), np.float32)
        for k in range(Ta):
            ids = T["dk_a"][k]
            y = T["labels_a"][k][T["lvl_flat"][ids]].astype(np.float64)
            YA[k] = fit_predict(src, ids, y, "bin", seed, k, ac, hc)
            if k == 0:
                log("%s 5-A first head: %.2fs -> projected %.0fs for %d tasks" % (tag, time.time() - t0, (time.time() - t0) * len(T["dk_a"]), len(T["dk_a"])))
        out["A"] = YA[:Ta] if max_tasks else YA
    if "B" in need:
        Tb = len(T["dk_b"]) if not max_tasks else min(max_tasks, len(T["dk_b"]))
        YB = np.zeros((len(T["dk_b"]), n_eval), np.float32)
        for k in range(Tb):
            YB[k] = fit_predict(src, T["dk_b"][k], T["g_dk"][k], "reg", seed, 100000 + k, ac, hc)
        out["B"] = YB[:Tb] if max_tasks else YB
    if "D" in need:
        YD = np.zeros((len(T["ids_d"]), n_eval), np.float32)
        for j in range(len(T["ids_d"])):
            ids = T["ids_d"][j]
            YD[j] = fit_predict(src, ids, T["s_flat"][ids], "reg", seed, 200000 + j, ac, hc)
        out["D"] = YD
    return out


def pred_file(paths, layer, model, seed):
    return paths["results_dir"] / PRED_DIR / ("%s__%s__seed%d.npz" % (layer, model, seed))


def need_set(subs):
    n = set()
    if "5-A" in subs or "5-C" in subs:
        n.add("A")
    if "5-B" in subs:
        n.add("B")
    if "5-D" in subs:
        n.add("D")
    return n


def heads_stage(cfg, cfg2, paths, plan, seeds, suffix="", max_tasks=None, layers_only=None, dry_run=False):
    layers = [l for l in plan["layers"] if (not layers_only or l["name"] in layers_only)]
    ibn, cbn = (max_tasks or cfg["ib"]["n_tasks"]), (max_tasks or cfg["calib"]["n_tasks"])
    per_model = lambda subs: (ibn if ("5-A" in subs or "5-C" in subs) else 0) + (cbn if "5-B" in subs else 0) + (len(cfg["law"]["combos"]) + 1 if "5-D" in subs else 0)
    n_attn = sum(2 * len(seeds) * per_model(plan["layer_subexps"][l["name"]]) for l in layers)
    union = set(s for l in layers for s in plan["layer_subexps"][l["name"]])
    n_base = len(seeds) * per_model(union) * 3
    log("heads plan: layers %s ; seeds %s ; attentive heads (ours + random-init) = %d ; base heads (oracle linear/MLP, area) = %d ; "
        "measure the first head (logged) before the full run; estimates are in README (추정)" % ([l["name"] for l in layers], seeds, n_attn, n_base))
    if dry_run:
        return None
    if plan.get("hold"):
        raise RuntimeError("plan.hold is true (experiment 1 negative): experiment 5 is on hold - report to the user instead of running heads")
    import torch
    T, meta = load_tasks(paths)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = CA.load(as002(paths), suffix)
    Etr = sum(r["part"] == "train" for r in C["rows"])
    nF = len(meta["frames"])
    ac, hc = cfg2["probe_impl"]["attentive"], cfg["head"]
    (paths["results_dir"] / PRED_DIR).mkdir(parents=True, exist_ok=True)
    cost = read_json(paths["results_dir"] / "cost.json") if (paths["results_dir"] / "cost.json").exists() else {"entries": []}
    t_all = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    bs = base_sources(cfg, T, dev)
    for seed in seeds:
        for name, src in bs.items():
            f = pred_file(paths, "base", name, seed)
            if f.exists():
                continue
            t0 = time.time()
            res = predict_model(src, T, need_set(union), seed, ac, hc, max_tasks, tag="base/%s/s%d" % (name, seed))
            np.savez_compressed(f, **res)
            cost["entries"].append({"what": "base/%s/seed%d" % (name, seed), "seconds": time.time() - t0})
    for l in layers:
        need = need_set(plan["layer_subexps"][l["name"]])
        for ctl, mname in (("real", "ours"), ("rand", "random")):
            p = FE.layer_path(paths, l, ctl, suffix)
            if not p.exists():
                raise FileNotFoundError("%s missing - run `python run.py feat` (and 004 enc_cache for encoder layers)" % p)
            arr = np.load(p, mmap_mode="r")
            src = _sources(cfg, T, arr, Etr, nF, dev)
            for seed in seeds:
                f = pred_file(paths, l["name"], mname, seed)
                if f.exists():
                    continue
                t0 = time.time()
                res = predict_model(src, T, need, seed, ac, hc, max_tasks, tag="%s/%s/s%d" % (l["name"], mname, seed))
                np.savez_compressed(f, **res)
                cost["entries"].append({"what": "%s/%s/seed%d" % (l["name"], mname, seed), "seconds": time.time() - t0})
                log("heads %s %s seed %d done (%.0fs)" % (l["name"], mname, seed, time.time() - t0))
            del src
    cost.update({"gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                 "max_vram_mb": (torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None), "last_heads_stage_seconds": time.time() - t_all})
    write_json(cost, paths["results_dir"] / "cost.json")
    return True


# ------------------------------------------------------------------------------------------------ metrics
def load_preds(paths, plan, seeds):
    P = {}
    for seed in seeds:
        for name in ("oracle_linear", "oracle_mlp", "area"):
            f = pred_file(paths, "base", name, seed)
            if f.exists():
                P[("base", name, seed)] = dict(np.load(f))
        for l in plan["layers"]:
            for m in ("ours", "random"):
                f = pred_file(paths, l["name"], m, seed)
                if f.exists():
                    P[(l["name"], m, seed)] = dict(np.load(f))
    return P


def pysr_available():
    return importlib.util.find_spec("pysr") is not None


def pysr_fit(x, y, cfg):
    """Optional symbolic regression of yhat as a function of s (PySR). Returns an expression string or a status string; never silently ignored."""
    if not pysr_available():
        return "pysr_not_installed"
    try:
        from pysr import PySRRegressor
        m = PySRRegressor(niterations=int(cfg["law"]["pysr"]["niterations"]), binary_operators=["+", "*", "-"], unary_operators=[], random_state=int(cfg["law"]["pysr"]["seed"]),
                          deterministic=True, parallelism="serial", progress=False, verbosity=0)
        m.fit(np.asarray(x, float).reshape(-1, 1), np.asarray(y, float))
        return str(m.sympy())
    except Exception as e:                              # recorded, not hidden
        return "pysr_error: %s: %s" % (type(e).__name__, e)


def compute_metrics(cfg, T, P, plan, seeds):
    """Pure function of tasks + saved predictions. Returns dict(ib, cal_bins, cal_dev, law) lists of row dicts."""
    ctx = IB.Ctx(T["lvl"], T["q"], T["is_ind"], T["is_ood"])
    prim = "oracle_" + cfg["oracle"]["primary"]
    other = "oracle_linear" if prim == "oracle_mlp" else "oracle_mlp"
    ib_rows, cal_bins, cal_dev, law = [], [], [], []
    n_eval = len(T["s_eval"])
    for l in plan["layers"]:
        lay = l["name"]
        subs = plan["layer_subexps"][lay]
        metrics = (["R-IB", "D-IB"] if "5-A" in subs else []) + (["D-IB_area=", "D-IB_area≠", "R-IB_fold"] if "5-C" in subs else [])
        for seed in seeds:
            get = lambda lay_, m: P.get((lay_, m, seed))
            if metrics:
                srcs = {"ours": get(lay, "ours"), "random": get(lay, "random"), "area": get("base", "area"), "oracle": get("base", prim), other: get("base", other)}
                if any(v is None or "A" not in v for v in srcs.values()):
                    continue
                Ys = {k: v["A"] for k, v in srcs.items()}
                Ys["noninf"] = IB.noninformative(Ys["ours"].shape[0], n_eval, cfg["seeds"]["noninf"] + seed)
                Ys["const"] = IB.constant_model(Ys["ours"].shape[0], n_eval)
                raw = IB.raw_scores_multi(Ys, ctx, metrics)
                nz = cfg["normalization"]
                for mod in Ys:
                    for (m_, g), v in raw[mod].items():
                        mode = nz["noninf"][IB.metric_type(m_)]
                        alt = "const" if mode == "iid" else "iid"
                        ref = lambda md: raw["noninf" if md == "iid" else "const"][(m_, g)]
                        ib_rows.append({"layer": lay, "model": mod, "metric": m_, "pair_group": g, "seed": seed, "raw": v,
                                        "score": IB.normalize(v, raw["oracle"][(m_, g)], ref(mode), nz["min_gap"]),
                                        "score_alt": IB.normalize(v, raw["oracle"][(m_, g)], ref(alt), nz["min_gap"]),
                                        "ref_mode": mode})
            if "5-B" in subs:
                srcs = {"ours": get(lay, "ours"), "random": get(lay, "random"), "area": get("base", "area"), "oracle": get("base", prim), other: get("base", other)}
                if any(v is None or "B" not in v for v in srcs.values()):
                    continue
                Dm = {k: IB.dist_matrix(v["B"]) for k, v in srcs.items()}
                Dm["truth"] = IB.dist_matrix(T["g_eval"][:srcs["ours"]["B"].shape[0]])
                Dor = Dm["oracle"]
                norm = float(np.triu(Dor, 1).max())
                for g in cfg["calib"]["groups"]:
                    sel = IB.pair_selector(ctx, g, T["is_extrap"])
                    for mod, D in Dm.items():
                        bins, mad, npairs = IB.calibration(Dor, D, sel, cfg["calib"]["n_bins"], norm)
                        cal_dev.append({"layer": lay, "model": mod, "group": g, "seed": seed, "value": mad, "n_pairs": npairs, "norm": norm})
                        for b in bins:
                            r = {"layer": lay, "model": mod, "group": g, "seed": seed}
                            r.update(b)
                            cal_bins.append(r)
            if "5-D" in subs:
                srcs = {"ours": get(lay, "ours"), "random": get(lay, "random"), "area": get("base", "area"), "oracle": get("base", prim), other: get("base", other)}
                if any(v is None or "D" not in v for v in srcs.values()):
                    continue
                slices0 = {"interp": T["role"] == "interp", "extrap": T["role"] == "extrap", "early": T["kind"] == 0, "late": T["kind"] == 1}
                for mod, v in srcs.items():
                    D = v["D"]
                    for sl, mk in slices0.items():
                        a, b, r2 = IB.ols_line(T["s_eval"][mk], D[0][mk])
                        law.append({"layer": lay, "model": mod, "slice": sl, "seed": seed, "slope": a, "intercept": b, "r2": r2, "n": int(mk.sum()), "pysr_expr": ""})
                    for j in range(1, D.shape[0]):
                        a, b, r2 = IB.ols_line(T["s_eval"], D[j])
                        law.append({"layer": lay, "model": mod, "slice": "labels_" + "+".join(cfg["law"]["combos"][j - 1]), "seed": seed, "slope": a, "intercept": b, "r2": r2,
                                    "n": int(len(T["s_eval"])), "pysr_expr": ""})
    return {"ib": ib_rows, "cal_bins": cal_bins, "cal_dev": cal_dev, "law": law}


def write_csv(rows, path, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(r.get(k), float) and not np.isfinite(r.get(k))) else ("" if r.get(k) is None else r.get(k))) for k in fields})


IB_FIELDS = ["layer", "model", "metric", "pair_group", "seed", "raw", "score", "score_alt", "ref_mode"]
CAL_FIELDS = ["layer", "model", "group", "seed", "bin", "n_pairs", "x_oracle_mean", "y_model_mean", "x_norm", "y_norm"]
CALDEV_FIELDS = ["layer", "model", "group", "seed", "value", "n_pairs", "norm"]
LAW_FIELDS = ["layer", "model", "slice", "seed", "slope", "intercept", "r2", "n", "pysr_expr"]


def metrics_stage(cfg, cfg2, paths, plan, seeds, dry_run=False):
    log("metrics plan: layers %s ; seeds %s ; reads results/predictions/*.npz and results/tasks.npz ; writes ib_scores.csv calibration.csv calibration_dev.csv law_fits.csv" % ([l["name"] for l in plan["layers"]], seeds))
    if dry_run:
        return None
    T, meta = load_tasks(paths)
    P = load_preds(paths, plan, seeds)
    if not P:
        raise FileNotFoundError("no results/predictions/*.npz - run `python run.py heads` first")
    R = compute_metrics(cfg, T, P, plan, seeds)
    out = paths["results_dir"]
    if R["law"] and cfg["law"]["pysr"]["enabled"]:
        W = [r for r in R["law"] if r["model"] == "ours" and r["seed"] == seeds[0] and not r["slice"].startswith("labels_")]
        for r in W:
            pr = P[(r["layer"], "ours", r["seed"])]["D"][0]
            mk = {"interp": T["role"] == "interp", "extrap": T["role"] == "extrap", "early": T["kind"] == 0, "late": T["kind"] == 1}[r["slice"]]
            r["pysr_expr"] = pysr_fit(T["s_eval"][mk], pr[mk], cfg)
    write_csv(R["ib"], out / "ib_scores.csv", IB_FIELDS)
    write_csv(R["cal_bins"], out / "calibration.csv", CAL_FIELDS)
    write_csv(R["cal_dev"], out / "calibration_dev.csv", CALDEV_FIELDS)
    write_csv(R["law"], out / "law_fits.csv", LAW_FIELDS)
    figs = make_figs(R, cfg, out)
    log("metrics done: ib rows %d, calibration bin rows %d, law rows %d, figures %s" % (len(R["ib"]), len(R["cal_bins"]), len(R["law"]), figs))
    return R


def make_figs(R, cfg, out):
    try:
        import os
        os.environ["MPLBACKEND"] = "Agg"      # unconditional (see 004 figs.py); figures are only saved to files
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        return "skipped(no matplotlib: %s)" % e
    done = []
    if R["cal_bins"]:
        layers = sorted({r["layer"] for r in R["cal_bins"]})
        fig, axs = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4), squeeze=False)
        for ax, lay in zip(axs[0], layers):
            for mod in ("ours", "oracle", "area", "random", "truth"):
                pts = {}
                for r in R["cal_bins"]:
                    if r["layer"] == lay and r["model"] == mod and r["group"] == "all" and r["x_norm"] is not None:
                        pts.setdefault(r["bin"], []).append((r["x_norm"], r["y_norm"]))
                if pts:
                    xs = [np.mean([p[0] for p in v]) for _, v in sorted(pts.items())]
                    ys = [np.mean([p[1] for p in v]) for _, v in sorted(pts.items())]
                    ax.plot(xs, ys, marker=".", label=mod)
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_title(lay)
            ax.set_xlabel("oracle pairwise distance (norm)")
            ax.set_ylabel("model pairwise distance (norm)")
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / "fig_calibration.png", dpi=120)
        plt.close(fig)
        done.append("fig_calibration.png")
    dib = [r for r in R["ib"] if r["metric"] in ("D-IB_area=", "D-IB_area≠") and r["pair_group"] == "all" and r["model"] in ("ours", "area", "random", "oracle")]
    if dib:
        layers = sorted({r["layer"] for r in dib})
        fig, ax = plt.subplots(figsize=(1.5 + 2.0 * len(layers), 4))
        w = 0.18
        for j, mod in enumerate(("ours", "area", "random", "oracle")):
            for mi, (metric, hatch) in enumerate((("D-IB_area=", ""), ("D-IB_area≠", "//"))):
                ys = []
                for lay in layers:
                    v = [r["raw"] for r in dib if r["layer"] == lay and r["model"] == mod and r["metric"] == metric]
                    ys.append(np.nanmean(v) if v else np.nan)
                ax.bar(np.arange(len(layers)) + (2 * j + mi) * w / 1.0 - 3.5 * w, ys, w, hatch=hatch, color="C%d" % j, edgecolor="k", label="%s %s" % (mod, metric))
        ax.set_xticks(np.arange(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_ylabel("D-IB raw")
        ax.legend(fontsize=5, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "fig_dib_decomp.png", dpi=120)
        plt.close(fig)
        done.append("fig_dib_decomp.png")
    return done or "skipped(no rows)"


# ------------------------------------------------------------------------------------------------ report
def read_csv_rows(path, floats):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in floats:
            if k in r:
                r[k] = float(r[k]) if r[k] != "" else float("nan")
        if "seed" in r:
            r["seed"] = int(r["seed"])
    return rows


def verdict_rows(cfg, plan, ib_rows, cal_dev, law_rows):
    vc = cfg["verdict"]
    out = []
    for l in plan["layers"]:
        lay = l["name"]
        subs = plan["layer_subexps"][lay]
        row = {"layer": lay, "sub_experiments": subs}
        if plan.get("hold"):
            row["H-e"] = row["H-f"] = row["H-g"] = {"status": "not_evaluable", "reason": "plan_hold_exp1_negative"}
        else:
            row["H-e"] = VD.h_e(ib_rows, cal_dev, lay, vc) if ("5-A" in subs and "5-B" in subs) else {"status": "not_evaluable", "reason": "5-A_and_5-B_not_both_selected_for_this_layer"}
            row["H-f"] = VD.h_f(ib_rows, lay, vc) if "5-C" in subs else {"status": "not_evaluable", "reason": "5-C_not_selected_for_this_layer"}
            row["H-g"] = VD.h_g(law_rows, lay, vc) if "5-D" in subs else {"status": "not_evaluable", "reason": "5-D_not_selected_for_this_layer"}
        out.append(row)
    return out


def report_stage(cfg, cfg2, paths, plan, dry_run=False):
    log("report plan: reads results/ib_scores.csv calibration_dev.csv law_fits.csv, applies verdict rules with thresholds from config, writes REPORT.md and results.json")
    if dry_run:
        return None
    out = paths["results_dir"]
    ib = read_csv_rows(out / "ib_scores.csv", ["raw", "score", "score_alt"])
    cd = read_csv_rows(out / "calibration_dev.csv", ["value", "n_pairs", "norm"])
    law = read_csv_rows(out / "law_fits.csv", ["slope", "intercept", "r2"])
    vr = verdict_rows(cfg, plan, ib, cd, law)
    cost = read_json(out / "cost.json") if (out / "cost.json").exists() else None
    from exp002.common import env_info
    res = {"experiment": cfg["name"], "pilot": bool(plan.get("pilot")), "plan": plan, "verdicts": vr, "thresholds": cfg["verdict"], "cost": cost,
           "config_sha256": hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest(), "config002_sha256": cfg2.get("_config002_sha256"),
           "environment": env_info(), "interpretation": None}
    write_json(res, paths["exp"] / "results.json")
    L = ["# REPORT (자동 생성: 규칙 적용 결과와 수치만, 해석 없음)", "",
         "판정 규칙 적용 결과 요약: " + "; ".join("%s: H-e=%s H-f=%s H-g=%s" % (r["layer"], r["H-e"]["status"], r["H-f"]["status"], r["H-g"]["status"]) for r in vr),
         "", "- pilot: %s" % res["pilot"], "- 4.6 분기 행: %s ; hold: %s ; unresolved: %s" % (plan["rows_matched"], plan["hold"], plan["unresolved"]), "",
         "| layer | 하위 실험 | H-e | H-f | H-g |", "|---|---|---|---|---|"]
    for r in vr:
        L.append("| %s | %s | %s (%s) | %s (%s) | %s (%s) |" % (r["layer"], ",".join(r["sub_experiments"]), r["H-e"]["status"], r["H-e"].get("reason", ""), r["H-f"]["status"], r["H-f"].get("reason", ""),
                                                            r["H-g"]["status"], r["H-g"].get("reason", "")))
    L += ["", "## 근거 수치 (results.json 의 verdicts 에서)", "", "```json", __import__("json").dumps(vr, indent=1, ensure_ascii=False, default=str), "```", "",
          "## 임계값 (config.yaml verdict)", "", "```json", __import__("json").dumps(cfg["verdict"], indent=1, ensure_ascii=False), "```", "",
          "## 실행 비용", "", "```json", __import__("json").dumps(cost, indent=1, ensure_ascii=False, default=str), "```", "",
          "그림: fig_calibration.png, fig_dib_decomp.png (있는 경우). 표: ib_scores.csv, calibration.csv, calibration_dev.csv, law_fits.csv, predictions/."]
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("report done")
    return res


def check(cfg, cfg2, paths):
    from exp002.common import env_info
    info = env_info()
    print("python executable :", info["python_executable"])
    print("CONDA_DEFAULT_ENV :", info["CONDA_DEFAULT_ENV"])
    print("packages          :", info["packages"], "pysr installed:", pysr_available())
    print("cuda_available    :", info.get("cuda_available"), info.get("gpu"), info.get("gpu_mem_gb"))
    s, p = load_summary(cfg, paths)
    print("004 summary.json  :", p, "exists=%s" % (s is not None), "pilot=%s" % (s or {}).get("pilot"))
    for k in ("cache002", "cache004", "cache"):
        print("  %-9s %s exists=%s" % (k, paths[k], paths[k].exists()))
    print("002 Q1 cache DONE:", (CA.cache_dir(as002(paths)) / "DONE").exists())
    print("results/tasks.npz:", (paths["results_dir"] / "tasks.npz").exists(), " plan.json:", (paths["results_dir"] / "plan.json").exists())
    return 0
