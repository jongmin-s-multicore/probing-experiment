"""008: 004 depth_r2.csv 사후 재집계 (A 통제 대비 곡선, B 척도 통일, C 겹침 재계산). CPU, numpy 만. 해석 없음."""
import csv, json, math, pathlib, sys
from collections import defaultdict
import numpy as np
import yaml

here = pathlib.Path(__file__).parent
cfg = yaml.safe_load(open(here / "config.yaml"))
np.random.seed(cfg["seed"])
P = lambda k: (here / cfg["inputs"][k]).resolve()
c004 = yaml.safe_load(open(P("config004")))
# 004 config 값과 008 config 값 일치 확인(바꾸지 않았음을 보증)
assert c004["depth"]["drop_threshold"] == cfg["drop_threshold"]
assert c004["depth"]["overlap_sd_mult"] == cfg["overlap_sd_mult"]
assert c004["depth"]["overlap_depth_fraction"] == cfg["overlap_depth_fraction"]
SPL = cfg["splits"]; THR = cfg["drop_threshold"]

rows = list(csv.DictReader(open(P("depth_r2_csv"))))
for r in rows:
    r["depth"] = int(r["depth"])
    for k in ("r2", "mae"):
        r[k] = float(r[k]) if r[k] != "" else float("nan")
    r["n_episodes"] = int(r["n_episodes"])
pr = cfg["primary"]


def sel(module, control, level="ALL", probe=None, fit=None, tp=None):
    probe = probe or pr["probe"]; fit = fit or pr["fit"]; tp = tp or pr["timepoint"]
    return [r for r in rows if r["module"] == module and r["control"] == control and r["level"] == level
            and r["probe"] == probe and r["fit"] == fit and r["timepoint"] == tp]


def first_drop(ds, vs, thr):  # 004 depthlib.first_drop 과 같은 규칙
    m = None
    for d, v in sorted(zip(ds, vs)):
        if not math.isfinite(v): continue
        if m is not None and (m - v) >= thr - 1e-12:
            return {"depth": int(d), "value": v, "running_max_before": m, "drop": m - v}
        m = v if m is None else max(m, v)
    return None


def key(r): return (r["depth"], r["split"], r["seed"], r["train_seed"])


out = {"config": cfg, "items": {}}
# ------------------------------------------------------------------ 사전 확인: random-init 대응 행
pre = {}
for mod in ("encoder", "predictor"):
    real = {key(r): r for r in sel(mod, "real")}
    rnd = {key(r): r for r in sel(mod, "random_init")}
    pre[mod] = {"n_real_rows": len(real), "n_random_init_rows": len(rnd), "n_matched_keys": len(set(real) & set(rnd)),
                "train_seeds": sorted({r["train_seed"] for r in real.values()}), "depths": sorted({r["depth"] for r in real.values()}),
                "probe_seeds": sorted({r["seed"] for r in real.values()})}
    pre[mod]["all_matched"] = pre[mod]["n_real_rows"] == pre[mod]["n_random_init_rows"] == pre[mod]["n_matched_keys"] > 0
out["precheck_random_init_rows"] = pre

# ------------------------------------------------------------------ A
A = {}
for mod in ("encoder", "predictor"):
    if not pre[mod]["all_matched"]:
        A[mod] = {"status": "계산 불가", "reason": "random-init 대응 행 없음/불일치"}; continue
    real = {key(r): r["r2"] for r in sel(mod, "real")}
    rnd = {key(r): r["r2"] for r in sel(mod, "random_init")}
    res = {"unit_note": ("encoder 는 동결이라 학습 시드가 없다(train_seed=NA). Δ>0 보조 판정의 sd 는 probe seed 5개 사이 sd 이다." if mod == "encoder"
                         else "Δ>0 보조 판정의 sd 는 학습 시드 3개의 (probe seed 평균) 사이 sd 이다."), "by_split": {}}
    for s in SPL:
        ds = pre[mod]["depths"]
        # Δ per (depth, train_seed, probe_seed)
        delta = defaultdict(list); realv = defaultdict(list); rndv = defaultdict(list)
        for k, v in real.items():
            if k[1] != s: continue
            d = v - rnd[k]
            delta[k[0]].append((k[3], k[2], d)); realv[k[0]].append(v); rndv[k[0]].append(rnd[k])
        curve_d, curve_r, curve_x, per_depth = [], [], [], []
        for d in ds:
            L = delta[d]
            dm = float(np.mean([x[2] for x in L]))
            if mod == "predictor":
                g = defaultdict(list)
                for ts, ps, x in L: g[ts].append(x)
                units = [float(np.mean(v)) for v in g.values()]
            else:
                units = [x[2] for x in L]
            sd = float(np.std(units, ddof=1)) if len(units) > 1 else float("nan")
            lo = float(np.mean(units) - cfg["delta_gt0_sd_mult"] * sd)
            curve_d.append(dm); curve_r.append(float(np.mean(realv[d]))); curve_x.append(float(np.mean(rndv[d])))
            per_depth.append({"depth": d, "delta_mean": dm, "real_mean": curve_r[-1], "random_init_mean": curve_x[-1],
                              "n_units": len(units), "unit_sd": sd, "mean_minus_2sd": lo, "delta_gt0": bool(lo > 0)})
        fd = first_drop(ds, curve_d, THR); fr = first_drop(ds, curve_r, THR)
        res["by_split"][s] = {"delta_curve": per_depth, "vanish_delta": fd, "vanish_real_same_rule": fr,
                              "n_depths_delta_gt0": int(sum(p["delta_gt0"] for p in per_depth)), "n_depths": len(ds),
                              "status_vanish_delta": "관측됨" if fd else "관측 안 됨"}
    res["status"] = "관측됨" if any(res["by_split"][s]["vanish_delta"] for s in SPL) else "관측 안 됨"
    res["status_rule"] = "어느 split 이든 Δ 곡선에서 직전 최고값 대비 0.2 이상 하락 깊이가 있으면 관측됨"
    A[mod] = res
out["items"]["A_control_relative_curve"] = A

# ------------------------------------------------------------------ B
# 척도: 레이블 = log(명목 면적비) (002 config analysis.primary_target). level 별 값은 002 config area_ratio.
c002 = yaml.safe_load(open(P("config002")))
area = c002["data"]["area_ratio"] if "area_ratio" in c002.get("data", {}) else None
if area is None:
    def find(o):
        if isinstance(o, dict):
            if "a026" in o and "a248" in o and isinstance(o["a026"], (int, float)): return o
            for v in o.values():
                f = find(v)
                if f: return f
    area = find(c002)
lab = {k: math.log(v) for k, v in area.items()}
Bres = {"label_source": "002 config area_ratio 의 log (목표 log_nominal_area_ratio, 에피소드 상수)"}
# 자기검증: 레벨 행 n_episodes 로 공통분산을 계산하고 CSV 의 'ALL' split 행과 맞는지 본다
# (가) MSE_level = (1 - r2_level) * Var_all   (나) split R2(CSV) = 1 - MSE_split / Var_split
roles = c002["data"]["roles"] if "roles" in c002.get("data", {}) else None
lvl_split = {r["level"]: r["split"] for r in rows if r["level"] != "ALL" and r["module"] == "encoder"}
nlev = {}
for r in rows:
    if r["level"] != "ALL" and r["module"] == "encoder" and r["depth"] == 0 and r["control"] == "real" and r["seed"] == "0" \
            and r["probe"] == "linear" and r["timepoint"] == "late":
        nlev[r["level"]] = r["n_episodes"]
Nall = sum(nlev.values())
mu = sum(nlev[l] * lab[l] for l in nlev) / Nall
Var_all = sum(nlev[l] * (lab[l] - mu) ** 2 for l in nlev) / Nall
Bres["n_episodes_per_level"] = nlev; Bres["common_label_variance_all_eval_episodes"] = Var_all


def split_var(s):
    ls = [l for l in nlev if lvl_split[l] == s]; n = sum(nlev[l] for l in ls)
    m = sum(nlev[l] * lab[l] for l in ls) / n
    return sum(nlev[l] * (lab[l] - m) ** 2 for l in ls) / n


def metrics(mod, control):
    rec = {}
    lv = defaultdict(dict)
    for r in sel(mod, control, level="ALL"): pass
    # 수집: (depth, probe seed, train_seed) 별 level r2/mae
    per = defaultdict(dict)
    for r in rows:
        if r["module"] == mod and r["control"] == control and r["probe"] == pr["probe"] and r["fit"] == pr["fit"] and r["timepoint"] == pr["timepoint"]:
            per[(r["depth"], r["seed"], r["train_seed"])][(r["level"], r["split"])] = r
    return per


per = metrics("predictor", "real")
chk = []
for k, dct in per.items():
    mse = {l: (1 - dct[(l, lvl_split[l])]["r2"]) * Var_all for l in nlev}
    for s in SPL:
        ls = [l for l in nlev if lvl_split[l] == s]
        msesp = sum(nlev[l] * mse[l] for l in ls) / sum(nlev[l] for l in ls)
        r2_csv = dct[("ALL", s)]["r2"]
        chk.append(abs((1 - msesp / split_var(s)) - r2_csv))
Bres["selfcheck_max_abs_diff_split_r2_vs_csv_ALL_rows"] = float(max(chk))
Bres["selfcheck_pass"] = bool(max(chk) < 1e-6)
if not Bres["selfcheck_pass"]:
    B = {"status": "계산 불가", "reason": "레벨 레이블로 복원한 공통/ split 분산이 CSV 의 ALL 행 R2 와 1e-6 이내로 일치하지 않음", "detail": Bres}
else:
    B = {"detail": Bres, "modules": {}}
    for mod in ("encoder", "predictor"):
        mres = {}
        for control in ("real", "random_init"):
            per = metrics(mod, control)
            ds = sorted({k[0] for k in per})
            cur = {}
            for s in SPL:
                ls = [l for l in nlev if lvl_split[l] == s]; n = sum(nlev[l] for l in ls)
                vals = defaultdict(lambda: {"mae": [], "mse": [], "r2_common": [], "r2_split": []})
                for k, dct in per.items():
                    mse = sum(nlev[l] * (1 - dct[(l, s)]["r2"]) * Var_all for l in ls) / n
                    vals[k[0]]["mse"].append(mse); vals[k[0]]["mae"].append(dct[("ALL", s)]["mae"])
                    vals[k[0]]["r2_common"].append(1 - mse / Var_all); vals[k[0]]["r2_split"].append(dct[("ALL", s)]["r2"])
                cur[s] = {d: {m: float(np.mean(v)) for m, v in vals[d].items()} for d in ds}
            mres[control] = cur
        # 곡선과 0.2 규칙 (r2_common)
        summ = {}
        for control in mres:
            summ[control] = {}
            for s in SPL:
                ds = sorted(mres[control][s])
                summ[control][s] = {"first_drop_r2_common": first_drop(ds, [mres[control][s][d]["r2_common"] for d in ds], THR),
                                    "first_drop_r2_split": first_drop(ds, [mres[control][s][d]["r2_split"] for d in ds], THR),
                                    "mae_range": [min(mres[control][s][d]["mae"] for d in ds), max(mres[control][s][d]["mae"] for d in ds)],
                                    "mse_range": [min(mres[control][s][d]["mse"] for d in ds), max(mres[control][s][d]["mse"] for d in ds)]}
        B["modules"][mod] = {"curves_by_control_split_depth": {c: {s: {str(d): v for d, v in mres[c][s].items()} for s in SPL} for c in mres}, "summary": summ}
    B["status"] = "관측됨" if any(B["modules"][m]["summary"]["real"][s]["first_drop_r2_common"] for m in B["modules"] for s in SPL) else "관측 안 됨"
    B["status_rule"] = "공통 분산 R2 곡선에서 어느 split 이든 0.2 이상 하락 깊이가 있으면 관측됨(MAE/MSE 곡선은 curves 에 수치만 기록)"
    B["note"] = "MSE/MAE 단위: log 면적비. 값은 probe seed(및 predictor 는 학습 seed) 평균. R2 common = 1 - MSE_split / Var(전체 평가 episode 900)."
out["items"]["B_unified_scale"] = B

# ------------------------------------------------------------------ C
C = {}
for mod in ("encoder", "predictor"):
    if mod == "encoder":
        C[mod] = {"status": "계산 불가", "reason": "인코더는 동결이라 학습 시드 단위가 없다(train_seed=NA, 서로 다른 학습 시드 3개의 seed 평균이 존재하지 않음). 004 가 쓴 sd 는 probe seed 5개 사이 sd 이며 004 원 결과와 동일하다."}
        continue
    per_ts = defaultdict(lambda: defaultdict(list))   # (split, depth) -> train_seed -> list
    for r in sel(mod, "real"):
        per_ts[(r["split"], r["depth"])][r["train_seed"]].append(r["r2"])
    ds = sorted({k[1] for k in per_ts}); res = {}
    for s in ("interp", "extrap"):
        pd_ = []
        for d in ds:
            tr = [np.mean(v) for v in per_ts[("train", d)].values()]
            oo = [np.mean(v) for v in per_ts[(s, d)].values()]
            tol = cfg["overlap_sd_mult"] * max(np.std(tr, ddof=1), np.std(oo, ddof=1))
            gap = float(np.mean(tr) - np.mean(oo))
            pd_.append({"depth": d, "gap": gap, "tol": float(tol), "inside": bool(abs(gap) <= tol), "n_train_seeds": len(tr),
                        "sd_train": float(np.std(tr, ddof=1)), "sd_ood": float(np.std(oo, ddof=1))})
        fr = sum(p["inside"] for p in pd_) / len(pd_)
        res[s] = {"fraction_depths_inside": fr, "overlap": bool(fr >= cfg["overlap_depth_fraction"] - 1e-12), "per_depth": pd_}
    C[mod] = {"by_split": res, "status": "관측됨" if all(res[s]["overlap"] for s in res) else "관측 안 됨",
              "status_rule": "interp, extrap 둘 다 inside 비율 >= 0.8 이면 겹침 관측됨(004 판정 규칙과 동일 구조)"}
# 004 원 결과(참고): summary.json
s004 = json.load(open(P("summary_json")))
out["reference_004_summary_ind_ood_gap_fractions"] = {m: {s: s004["ind_ood_gap"][m][s]["fraction_depths_inside_seed_sd"] for s in ("interp", "extrap")} for m in ("encoder", "predictor")}
out["items"]["C_overlap_recompute_train_seed_units"] = C
out["notes"] = "해석 없음. /review-experiment 대기. 004 판정(H-d refuted)은 바꾸지 않는다."
(here / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=lambda o: None))
print(json.dumps({"precheck": pre, "A": {m: A[m].get("status") for m in A}, "B": B["status"], "C": {m: C[m]["status"] for m in C}}, ensure_ascii=False))
