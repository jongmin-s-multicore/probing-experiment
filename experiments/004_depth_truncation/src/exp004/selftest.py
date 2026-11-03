"""CPU self-test on SYNTHETIC data with planted structure. The numbers printed here are NOT experiment results.
torch-dependent items are reported 'skipped(no torch)' (never counted as passed)."""
from __future__ import annotations

import copy
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np

from exp002 import model_io as M
from exp002 import stats as ST
from exp002.data import level_roles, nominal_log_area
from . import depthlib as DL
from .base import ConfigNull, write_json, read_json

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


def small_cfg2(cfg2, n_train=12, n_test=8):
    c = copy.deepcopy(cfg2)
    c["data"]["train_episodes_per_level"] = n_train
    c["data"]["eval_episodes_per_level"] = n_test
    c["probe"]["seeds"] = [0, 1, 2]
    c["probe_impl"]["n_bootstrap"] = 50
    c["probe_impl"]["attentive"].update({"d_model": 16, "n_heads": 2, "mlp_hidden": 16, "epochs": 12, "batch_size": 64})
    c["probe_impl"]["device"] = "cpu"
    return c


def make_rows(cfg2):
    roles = level_roles(cfg2)
    rows = []
    for part, levels, n in (("train", cfg2["data"]["train_levels_b"], cfg2["data"]["train_episodes_per_level"]), ("test", cfg2["data"]["levels"], cfg2["data"]["eval_episodes_per_level"])):
        for l in levels:
            for i in range(n):
                rows.append({"id": len(rows), "part": part, "level": l, "role": roles[l], "split": "x", "name": "e%d" % len(rows), "dir": "x", "rank_in_level": i,
                             "nominal_log_area": nominal_log_area(cfg2, l)})
    return rows


def te_rows_of(rows):
    return [r for r in rows if r["part"] == "test"]


# ------------------------------------------------------------------------------------------------ A. numpy
def t_first_drop(cfg):
    print("A1 vanishing point (first depth >= threshold below the running max)")
    D = [0, 2, 4, 6, 8, 10, 12]
    fd = lambda v, thr=0.2: DL.first_drop(D, v, thr)
    r = fd([0.5, 0.9, 0.92, 0.9, 0.4, 0.3, 0.2])
    check("planted drop at depth 8 is found", r is not None and r["depth"] == 8, str(r))
    check("no drop -> None", fd([0.5, 0.6, 0.7, 0.8, 0.8, 0.79, 0.7]) is None)
    check("slow decline: each step < 0.2 but 0.2 below the running max -> found at the first depth that reaches it", fd([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]) == {"depth": 4, "value": 0.7, "running_max_before": 0.9, "drop": 0.9 - 0.7} or fd([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])["depth"] == 4)
    check("boundary: drop exactly 0.2 counts, 0.199 does not", fd([0.9, 0.7]*1 + [0.7]*5)["depth"] == 2 and fd([0.9, 0.7001] + [0.7001] * 5) is None)
    check("NaN depths are skipped", fd([0.9, float("nan"), 0.5, 0.5, 0.5, 0.5, 0.5])["depth"] == 4)
    check("increase after a dip does not reset the running max", fd([0.5, 0.9, 0.85, 0.95, 0.74, 0.7, 0.7])["depth"] == 8)
    try:
        DL.first_drop(D, [0.9] * 7, None)
        check("null threshold raises ConfigNull", False)
    except ConfigNull:
        check("null threshold raises ConfigNull", True)
    check("config drop_threshold is 0.2", cfg["depth"]["drop_threshold"] == 0.2)


def rows_from_curves(spec, seeds=(0, 1, 2, 3, 4), noise=0.01, module_depths=None, rng=None):
    """spec {(module, split): {depth: value}} -> level=='ALL' rows of the primary configuration."""
    rng = rng or np.random.RandomState(0)
    out = []
    for (mod, sp), dd in spec.items():
        for d, v in dd.items():
            for s in seeds:
                out.append({"module": mod, "depth": d, "probe": "linear", "fit": "new", "level": "ALL", "split": sp, "timepoint": "late", "seed": s,
                            "r2": v + noise * rng.randn(), "mae": 0.0, "pearson": 0.0, "control": "real", "train_seed": "NA", "pred_mean": 0.0, "n_episodes": 10})
    return out


def t_summary(cfg):
    print("A2 summary.json: k*/m*, loss_ood_only, l_best, H-d verdicts")
    E = [0, 2, 4, 6, 8, 10, 12]
    P = [0, 1, 2, 3, 4, 5, 6]
    hi = lambda ds, until: {d: (0.95 if d <= until else 0.3) for d in ds}
    meta = {"pilot": True, "encoder_layers": 12, "predictor_layers": 6}
    # planted: encoder extrap drops after 4 (k*=6), interp after 6 (->8), train never; predictor: all splits drop after 3 (m*=4)
    spec = {("encoder", "train"): {d: 0.95 for d in E}, ("encoder", "interp"): hi(E, 6), ("encoder", "extrap"): hi(E, 4),
            ("predictor", "train"): hi(P, 3), ("predictor", "interp"): hi(P, 3), ("predictor", "extrap"): hi(P, 3)}
    s = DL.build_summary(rows_from_curves(spec), cfg, meta)
    check("k_star = 6 (earliest over splits), per split {train:None, interp:8, extrap:6}", s["k_star"] == 6 and s["k_star_by_split"] == {"train": None, "interp": 8, "extrap": 6}, str(s["k_star_by_split"]))
    check("m_star = 4", s["m_star"] == 4 and s["m_star_by_split"]["train"] == 4)
    check("loss_ood_only False because the predictor also loses InD", s["loss_ood_only"] is False)
    check("pilot field carried; layers carried", s["pilot"] is True and s["encoder_layers"] == 12 and s["predictor_layers"] == 6)
    check("refine_suggestion names the interval before the drop", s["refine_suggestion"]["encoder/extrap"]["between_depths"] == [4, 6])
    check("verdict H-d supported (OoD earlier, curves separate)", s["verdict_H_d"]["status"] == "supported", str(s["verdict_H_d"]["reason"]))
    spec2 = {k: v for k, v in spec.items() if k[0] == "encoder"}
    s2 = DL.build_summary(rows_from_curves(spec2), cfg, meta)
    check("encoder only: loss_ood_only True, m_star None", s2["loss_ood_only"] is True and s2["m_star"] is None)
    check("l_best: argmax of mean(interp, extrap) R2 -> encoder depth <= 4", s2["l_best"]["module"] == "encoder" and s2["l_best"]["depth"] <= 4)
    flat = {(m, sp): {d: 0.9 for d in (E if m == "encoder" else P)} for m in ("encoder", "predictor") for sp in DL.SPLITS}
    s3 = DL.build_summary(rows_from_curves(flat), cfg, meta)
    check("no drop anywhere -> k*/m* null and verdict refuted(no_drop)", s3["k_star"] is None and s3["m_star"] is None and s3["verdict_H_d"]["status"] == "refuted" and "no_drop" in s3["verdict_H_d"]["reason"])
    same = {(m, sp): {d: (0.95 if d <= 4 else 0.3) for d in (E if m == "encoder" else P)} for m in ("encoder", "predictor") for sp in DL.SPLITS}
    s4 = DL.build_summary(rows_from_curves(same), cfg, meta)
    check("InD and OoD curves identical -> refuted (overlap within seed sd)", s4["verdict_H_d"]["status"] == "refuted" and "overlap" in s4["verdict_H_d"]["reason"], str(s4["verdict_H_d"]["reason"]))
    s5 = DL.build_summary(rows_from_curves(spec, seeds=(0,)), cfg, meta)
    check("single seed -> sd undefined -> not_evaluable", s5["verdict_H_d"]["status"] == "not_evaluable", str(s5["verdict_H_d"]))
    c6 = copy.deepcopy(cfg)
    c6["depth"]["drop_threshold"] = None
    try:
        DL.build_summary(rows_from_curves(spec), c6, meta)
        check("null drop_threshold in config -> ConfigNull (not silently defaulted)", False)
    except ConfigNull:
        check("null drop_threshold in config -> ConfigNull (not silently defaulted)", True)
    c7 = copy.deepcopy(cfg)
    c7["depth"]["overlap_sd_mult"] = None
    s7 = DL.build_summary(rows_from_curves(spec), c7, meta)
    check("null overlap_sd_mult -> verdict not_evaluable", s7["verdict_H_d"]["status"] == "not_evaluable")
    ood_later = {("encoder", "train"): hi(E, 4), ("encoder", "interp"): hi(E, 8), ("encoder", "extrap"): hi(E, 8)}
    for k in list(ood_later):
        ood_later[("predictor",) + k[1:]] = hi(P, 6)
    s8 = DL.build_summary(rows_from_curves(ood_later), cfg, meta)
    check("OoD vanishes later than InD -> refuted(ood_not_earlier)", s8["verdict_H_d"]["status"] == "refuted" and "not_earlier" in s8["verdict_H_d"]["reason"], str(s8["verdict_H_d"]["reason"]))
    few = {(m, sp): {0: 0.9, 12: 0.1} for m in ("encoder", "predictor") for sp in DL.SPLITS}
    s9 = DL.build_summary(rows_from_curves(few), cfg, meta)
    check("fewer than min_depths -> not_evaluable", s9["verdict_H_d"]["status"] == "not_evaluable")


def t_rows():
    print("A3 cell_rows (level rows use pooled-variance R2), csv round trip")
    rng = np.random.RandomState(1)
    levels = ["a", "b", "c"]
    te = [{"level": l, "role": ro} for l, ro in zip(levels, ["trained", "interp", "extrap"]) for _ in range(10)]
    y = np.repeat([0.0, 1.0, 2.0], 10)
    h = y + 0.3 * rng.randn(30)
    S = ST.episode_stats(y, h, np.arange(30), 30)[None]
    rows = DL.cell_rows(S, h[None], te, levels, {"module": "encoder", "depth": 4, "probe": "linear", "fit": "new", "timepoint": "late", "control": "real", "train_seed": "NA"}, [7])
    allr = [r for r in rows if r["level"] == "ALL" and r["split"] == "all"][0]
    check("ALL/all R2 equals direct", abs(allr["r2"] - (1 - ((y - h) ** 2).sum() / ((y - y.mean()) ** 2).sum())) < 1e-9)
    lr = [r for r in rows if r["level"] == "b"][0]
    check("level row R2 = 1 - MSE_level / Var_all", abs(lr["r2"] - (1 - ((y[10:20] - h[10:20]) ** 2).mean() / y.var())) < 1e-9 and np.isnan(lr["pearson"]) and lr["split"] == "interp")
    check("one row per probe seed x group; seed carried", all(r["seed"] == 7 for r in rows) and len(rows) == 3 + 3 + 1)
    d = tempfile.mkdtemp()
    try:
        p = Path(d) / "x.csv"
        DL.write_rows(rows, p)
        back = DL.read_rows(p)
        check("csv round trip keeps r2 and empty pearson as NaN", len(back) == len(rows) and abs(back[0]["r2"] - rows[0]["r2"]) < 1e-12 and np.isnan([r for r in back if r["level"] == "b"][0]["pearson"]))
        hdr = open(p).readline().strip().split(",")
        check("spec column names present in order", hdr[:11] == ["module", "depth", "probe", "fit", "level", "split", "timepoint", "seed", "r2", "mae", "pearson"], str(hdr[:11]))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_pixels():
    print("A4 pixel tokens")
    from . import stages as S
    u8 = (np.arange(2 * 28 * 28 * 3) % 251).astype(np.uint8).reshape(2, 28, 28, 3)
    t = S.pixel_tokens(u8, grid=14, q=4)
    check("shape (n, 196, q*q*3) with q reduced to the patch size", t.shape == (2, 196, 2 * 2 * 3))
    ones = np.full((1, 56, 56, 3), 255, np.uint8)
    check("constant white image -> all ones", np.allclose(S.pixel_tokens(ones, grid=14, q=4), 1.0))


# ------------------------------------------------------------------------------------------------ B. planted probes (numpy + probes)
def t_encoder_probe_planted(cfg, cfg2):
    print("B1 planted encoder depth structure -> probes -> rows -> summary (k* recovered)")
    from . import stages as S
    from exp002 import evalgrid as EG
    c2 = small_cfg2(cfg2)
    kinds = ["linear_pooled", "attentive"] if have_torch() else ["linear_pooled"]
    c2["probe"]["kinds"] = kinds
    rows = make_rows(c2)
    Etr = sum(r["part"] == "train" for r in rows)
    te = te_rows_of(rows)
    levels = list(c2["data"]["levels"])
    s = np.array([r["nominal_log_area"] for r in rows])
    roles = np.array([r["role"] for r in rows])
    P_, D_ = 4, 8
    a = np.random.RandomState(5).randn(D_)
    a /= np.linalg.norm(a)
    ks = [0, 2, 4, 6, 8, 10, 12]
    frames = [0, 1, 2, 3, 8, 9, 10, 11]
    pos = {"early": [0, 1, 2, 3], "late": [4, 5, 6, 7]}
    rng = np.random.RandomState(3)

    def feats(k, until_ind, until_ood):
        until = np.where(roles == "extrap", until_ood, until_ind)
        sig = (k <= until).astype(float)
        return (sig[:, None, None, None] * s[:, None, None, None] * a[None, None, None, :] * 1.0
                + 0.15 * rng.randn(len(s), len(frames), P_, D_)).astype(np.float16)
    Y = np.repeat(s[:, None], len(frames), 1)

    def run_case(until_ind, until_ood):
        allrows = []
        for k in ks:
            F = feats(k, until_ind, until_ood)
            allrows += S.probe_hidden(c2, kinds, c2["probe"]["seeds"], F, Y[:Etr], Y[Etr:], Etr, pos, te, levels,
                                      {"module": "encoder", "depth": k, "fit": "new", "train_seed": "NA"}, ctrl_name="real")
        return allrows
    meta = {"pilot": True, "encoder_layers": 12, "predictor_layers": 6}
    r1 = run_case(until_ind=12, until_ood=4)
    sm1 = DL.build_summary(r1, cfg, meta)
    check("planted: extrap info lost after k=4 -> k_star_by_split.extrap == 6", sm1["k_star_by_split"]["extrap"] == 6, str(sm1["k_star_by_split"]))
    check("planted: train/interp keep info at every k -> None", sm1["k_star_by_split"]["train"] is None and sm1["k_star_by_split"]["interp"] is None)
    check("k_star == 6 and loss_ood_only True (only OoD loses)", sm1["k_star"] == 6 and sm1["loss_ood_only"] is True)
    a_rows = [r for r in r1 if r["probe"] == "attentive" and r["timepoint"] == "late" and r["level"] == "ALL" and r["split"] == "train"]
    if have_torch():
        check("attentive rows also produced (probe name 'attentive', 3 seeds x 7 depths)", len(a_rows) == 21)
    else:
        skip("attentive rows", "no torch")
    r2 = run_case(until_ind=8, until_ood=8)
    sm2 = DL.build_summary(r2, cfg, meta)
    check("planted: all splits lose info after k=8 -> k_star == 10 for every split, loss_ood_only False", sm2["k_star"] == 10 and set(sm2["k_star_by_split"].values()) == {10} and sm2["loss_ood_only"] is False, str(sm2["k_star_by_split"]))
    shuf = S.probe_hidden(c2, ["linear_pooled"], c2["probe"]["seeds"], feats(0, 12, 12), Y[:Etr], Y[Etr:], Etr, pos, te, levels,
                          {"module": "encoder", "depth": 0, "fit": "new", "train_seed": "NA"}, shuffle_base=int(c2["probe_impl"]["shuffle_seed"]), ctrl_name="label_shuffle")
    real = [r for r in r1 if r["depth"] == 0 and r["probe"] == "linear" and r["timepoint"] == "late" and r["level"] == "ALL" and r["split"] == "interp"]
    sh = [r for r in shuf if r["timepoint"] == "late" and r["level"] == "ALL" and r["split"] == "interp"]
    check("label-shuffle control has lower R2 than the real probe", np.mean([r["r2"] for r in sh]) < np.mean([r["r2"] for r in real]) - 0.5, "(%.2f vs %.2f)" % (np.mean([r["r2"] for r in sh]), np.mean([r["r2"] for r in real])))


# ------------------------------------------------------------------------------------------------ C. torch
def build_toys(nvis, nact, P, L, erase_from=None, seed=0, identity_norm=False):
    import torch
    nn = torch.nn
    D = nvis + nact

    class Lin(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.l = nn.Linear(D, D, bias=False)
            nn.init.normal_(self.l.weight, std=scale)

        def forward(self, x):
            return self.l(x)

    class Erase(nn.Module):
        def forward(self, x):                                     # x + Erase(x) = x with the visual coordinates zeroed
            m = torch.zeros(D)
            m[:nvis] = 1.0
            return -x * m

    class Tr(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.ModuleList([Lin(0.02), Erase() if (erase_from is not None and j + 1 == erase_from) else Lin(0.02)]) for j in range(L)])
            self.norm = nn.Identity() if identity_norm else nn.LayerNorm(D)

        def forward(self, x):
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return self.norm(x)

    class Pred(nn.Module):
        def __init__(self, n_tok):
            super().__init__()
            self.pos_embedding = nn.Parameter(torch.randn(1, n_tok, D) * 0.01)
            self.dropout = nn.Dropout(0.0)
            self.transformer = Tr()

        def forward(self, x):
            x = x + self.pos_embedding[:, :x.shape[1]]
            return self.transformer(self.dropout(x))

    class World(nn.Module):
        def __init__(self, H):
            super().__init__()
            self.predictor = Pred(H * P)
            self.R = nn.Parameter(torch.randn(48, nvis) * 0.2, requires_grad=False)
            self.A = nn.Parameter(torch.randn(30, nact) * 0.2, requires_grad=False)

        def encode(self, obs, act):
            v = obs["visual"]                                         # (B,K,3,12,12) in [-1,1]
            B, K = v.shape[:2]
            pt = v.reshape(B, K, 3, 3, 4, 3, 4).permute(0, 1, 3, 5, 2, 4, 6).reshape(B, K, P, 48)
            zv = pt @ self.R
            za = (act @ self.A)[:, :, None, :].expand(B, K, P, nact)
            return torch.cat([zv, za], -1)

        def predict(self, z):
            N, T, P_, D_ = z.shape
            return self.predictor(z.reshape(N, T * P_, D_)).reshape(N, T, P_, D_)

        def replace_actions_from_z(self, z, act):
            za = (act @ self.A)[:, :, None, :].expand(z.shape[0], z.shape[1], z.shape[2], nact)
            z[..., -nact:] = za
            return z
    return World


def t_torch_units():
    print("C1 torch: encoder truncation and early exit against reference implementations")
    import torch
    from . import deep as DP
    nn = torch.nn
    torch.manual_seed(0)

    class Blk(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.l = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())

        def forward(self, x):
            return x + self.l(x)

    class ToyViT(nn.Module):
        def __init__(self, d=12, L=12, regs=0, chunked=False):
            super().__init__()
            self.d, self.num_register_tokens = d, regs
            self.pe = nn.Linear(6, d)
            self.cls = nn.Parameter(torch.randn(1, 1, d))
            self.reg = nn.Parameter(torch.randn(1, regs, d)) if regs else None
            bl = [Blk(d) for _ in range(L)]
            if chunked:
                self.blocks = nn.ModuleList([nn.ModuleList([nn.Identity()] * 4 + bl[:4]), nn.ModuleList([nn.Identity()] * 4 + bl[4:8]), nn.ModuleList([nn.Identity()] * 4 + bl[8:])])
            else:
                self.blocks = nn.ModuleList(bl)
            self.norm = nn.LayerNorm(d)

        def prepare_tokens_with_masks(self, x):
            p = self.pe(x)
            t = [self.cls.expand(x.shape[0], -1, -1)] + ([self.reg.expand(x.shape[0], -1, -1)] if self.reg is not None else []) + [p]
            return torch.cat(t, 1)

        def forward_features(self, x):
            h = self.prepare_tokens_with_masks(x)
            for b in DP.flat_blocks(self):
                h = b(h)
            return {"x_norm_patchtokens": self.norm(h)[:, 1 + self.num_register_tokens:]}

        def get_intermediate_layers(self, x, n, norm=True):                 # reference: DINOv2 semantic (block indices, 0-based)
            h = self.prepare_tokens_with_masks(x)
            outs = []
            for i, b in enumerate(DP.flat_blocks(self)):
                h = b(h)
                if i in n:
                    outs.append(h)
            return [self.norm(o)[:, 1 + self.num_register_tokens:] for o in outs]

    for regs, chunked in ((0, False), (4, False), (0, True)):
        v = ToyViT(regs=regs, chunked=chunked).eval()
        x = torch.randn(3, 9, 6)
        with torch.no_grad():
            T = DP.tokens_by_k(v, x, [0, 3, 12])
            ref = v.forward_features(x)["x_norm_patchtokens"]
            gi = v.get_intermediate_layers(x, [2])[0]
            h0 = v.norm(v.prepare_tokens_with_masks(x))[:, 1 + regs:]
        tag = "regs=%d chunked=%s" % (regs, chunked)
        check("tokens_by_k(k=L) == forward_features patch tokens (%s)" % tag, torch.allclose(T[12], ref, atol=1e-6))
        check("tokens_by_k(k=3) == get_intermediate_layers(n=[2], norm=True) (%s)" % tag, torch.allclose(T[3], gi, atol=1e-6))
        check("tokens_by_k(k=0) == norm(patch embedding + pos) without cls/registers (%s)" % tag, torch.allclose(T[0], h0, atol=1e-6) and T[0].shape == (3, 9, 12))
    try:
        DP.tokens_by_k(v, x, [13])
        check("k > number of blocks raises", False)
    except ValueError:
        check("k > number of blocks raises", True)

    from exp002.model_io import StreamHooks
    nvis, nact, P, L = 16, 4, 9, 6
    World = build_toys(nvis, nact, P, L)
    w = World(2).eval()
    z = torch.randn(3, 2, P, nvis + nact)
    with torch.no_grad():
        hid = DP.ee_forward(w.predictor, z.reshape(3, 2 * P, -1))
        full = w.predict(z)
        hk = StreamHooks(w).install()
        hk.reset()
        hk.capture_layers = set(range(L + 1))
        w.predict(z)
        cap = dict(hk.store)
        hk.remove()
    check("early exit at m=L equals model.predict", torch.allclose(DP.predict_exit(w, z, L), full, atol=1e-6))
    check("hidden after m blocks equals exp002 StreamHooks capture at layer index m (all m)", all(torch.allclose(hid[m], cap[m], atol=1e-6) for m in range(L + 1)))
    check("early exit at m<L differs from the full output", not torch.allclose(DP.predict_exit(w, z, 2), full, atol=1e-3))
    before = [p.clone() for p in w.predictor.parameters()]
    rw = DP.random_init_predictor(w, 5)
    after = list(w.predictor.parameters())
    check("random_init_predictor leaves the original predictor untouched", all(torch.equal(a, b) for a, b in zip(before, after)))
    check("random_init_predictor weights differ from the original", not torch.allclose(rw.predictor.transformer.layers[0][0].l.weight, w.predictor.transformer.layers[0][0].l.weight))
    rw2 = DP.random_init_predictor(w, 5)
    check("random_init_predictor reproducible for a fixed seed", torch.equal(rw.predictor.pos_embedding, rw2.predictor.pos_embedding))


def t_torch_pipeline(cfg, cfg2):
    print("C2 torch: toy world model with the size state erased after block 4 -> extraction -> probes (i)(ii)(iii) -> m* recovered")
    import torch
    from . import deep as DP
    from . import stages as S
    old = (M.P_TOK, M.D_VIS)
    M.P_TOK, M.D_VIS = 9, 16
    tmp = Path(tempfile.mkdtemp())
    try:
        nvis, nact, P, L, H = 16, 4, 9, 6, 1
        c2 = small_cfg2(cfg2)
        c2["probe"]["kinds"] = ["linear_pooled", "attentive"]
        cc = copy.deepcopy(cfg)
        rows = make_rows(c2)
        Etr = sum(r["part"] == "train" for r in rows)
        te = te_rows_of(rows)
        levels = list(c2["data"]["levels"])
        E = len(rows)
        s = np.array([r["nominal_log_area"] for r in rows])
        rng = np.random.RandomState(0)
        u8 = np.clip(128 + 40 * s[:, None, None, None, None] + 12 * rng.randn(E, 13, 12, 12, 3), 0, 255).astype(np.uint8)
        raw = rng.randn(E, 60, 6).astype(np.float32)
        World = build_toys(nvis, nact, P, L, erase_from=4, identity_norm=True)   # identity final norm: the toy is not trained, LayerNorm would rescale the outputs away from the encoder-token scale the exp1 probe was fitted on
        torch.manual_seed(0)
        w = World(H).eval()
        mean, std = torch.zeros(6), torch.ones(6)
        acts = M.acts_from_raw(torch.as_tensor(raw), mean, std)
        with torch.no_grad():
            toks = np.concatenate([M.encode_batch(w, M.frames_to_float(u8[i:i + 16]), torch.zeros(len(u8[i:i + 16]), 13, 30))[..., :nvis].numpy() for i in range(0, E, 16)])
        mf = np.zeros((E, 13, 3), np.float32)
        mf[..., 0] = np.exp(5 + s)[:, None]
        mf[..., 1:] = 10
        C = {"rows": rows, "frames_u8": u8, "actions_raw": raw, "tokens_dino": toks.astype(np.float16), "mask_feat": mf}
        frames = [0, 1, 2, 3, 8, 9, 10, 11]
        pos = {"early": [0, 1, 2, 3], "late": [4, 5, 6, 7]}
        steps = [1, 4, 8]
        meta = DP.extract(cc, c2, C, w, H, tmp / "real", mean, std, torch.device("cpu"), frames, Etr, steps, nvis, with_outputs=True, batch_episodes=16, label="toy")
        check("extract returns n_blocks read from the model (6) and files for every m", meta["n_blocks"] == 6 and all((tmp / "real" / ("feat_%d.npy" % m)).exists() for m in range(7)))
        Y = EG_labels(c2, C)
        out = S.pred_probe_rows(cc, c2, C, tmp / "real", meta, Y, Etr, te, levels, 0, "real", pos, 0, {})
        rr = out["rows"]
        hid = [r for r in rr if r["fit"] == "new" and r["control"] == "real" and r["probe"] == "linear" and r["timepoint"] == "late" and r["level"] == "ALL" and r["split"] == "train"]
        byd = {d: np.mean([r["r2"] for r in hid if r["depth"] == d]) for d in range(7)}
        check("(i) hidden z-part probe: R2 high for m=0..3 and low for m>=4 (planted)", all(byd[d] > 0.8 for d in range(4)) and all(byd[d] < 0.3 for d in range(4, 7)), str({d: round(v, 2) for d, v in byd.items()}))
        sm = DL.build_summary(rr, cc, {"pilot": True, "encoder_layers": 12, "predictor_layers": meta["n_blocks"]})
        # the toy has a single train seed and only 3 probe seeds -> InD/OoD all drop together; m_star must still be 4
        check("summary m_star == 4 recovered from the pipeline rows (train split)", sm["m_star"] == 4 and sm["m_star_by_split"]["train"] == 4, str(sm["m_star_by_split"]))
        e1 = {d: np.mean([r["r2"] for r in rr if r["fit"] == "exp1_probe" and r["module"] == "predictor" and r["depth"] == d and r["probe"] == "linear" and r["timepoint"] == "late" and r["level"] == "ALL" and r["split"] == "train"]) for d in range(7)}
        ref = np.mean([r["r2"] for r in rr if r["module"] == "reference" and r["probe"] == "linear" and r["timepoint"] == "late" and r["level"] == "ALL" and r["split"] == "train"])
        check("(ii) exp1 probe on early-exit z: present for every m, reference (encoder tokens) high", len(e1) == 7 and ref > 0.8, "ref %.2f" % ref)
        check("(ii) exp1 probe on early-exit z: m>=4 output has lower R2 than m<=3 (visual part erased)", max(e1[d] for d in range(4, 7)) < min(e1[d] for d in range(0, 4)), str({d: round(v, 2) for d, v in e1.items()}))
        roll = [r for r in rr if r["fit"] == "exp1_probe" and r["module"] == "predictor" and r["timepoint"] == "rollout_t+8" and r["level"] == "ALL" and r["split"] == "all" and r["probe"] == "linear"]
        check("(ii) rollout timepoints t+1, t+4, t+8 exist", {r["timepoint"] for r in rr if r["fit"] == "exp1_probe"} >= {"early", "late", "rollout_t+1", "rollout_t+4", "rollout_t+8"} and len(roll) == 7 * 3)
        mse = out["mse_rows"]
        mm = {d: np.mean([r["mse"] for r in mse if r["depth"] == d and r["timepoint"] == "late" and r["level"] == "ALL" and r["split"] == "all"]) for d in range(7)}
        check("(iii) next-z MSE rows for all depths/timepoints/levels", len(mse) == 5 * (9 + 3 + 1) * 7 and all(np.isfinite(v) for v in mm.values()), "")
        # ModelView / random-init branch of extract (no outputs)
        rm = DP.random_init_predictor(w, 9)
        meta2 = DP.extract(cc, c2, C, rm, H, tmp / "rand", mean, std, torch.device("cpu"), frames, Etr, steps, nvis, with_outputs=False, batch_episodes=16, label="toy-rand")
        rows_r = S.pred_probe_rows(cc, c2, C, tmp / "rand", meta2, Y, Etr, te, levels, 0, "random_init", pos, 0, {})["rows"]
        check("random-init predictor rows produced with control name and no exp1-probe rows", rows_r and {r["control"] for r in rows_r} == {"random_init"})
        # feature_part=full (404-like) shape
        meta3 = DP.extract(cc, c2, C, w, H, tmp / "full", mean, std, torch.device("cpu"), frames, Etr, steps, nvis + nact, with_outputs=False, batch_episodes=16, label="toy-full")
        check("feature_part full keeps action coordinates (dpart = nvis + nact)", np.load(tmp / "full" / "feat_3.npy", mmap_mode="r").shape[-1] == nvis + nact)
        # corrupt predictor structure -> the m=L consistency check must raise
        class Bad(type(w)):
            def predict(self, z):
                return super().predict(z) + 1.0
        w.__class__ = Bad
        try:
            DP.extract(cc, c2, C, w, H, tmp / "bad", mean, std, torch.device("cpu"), frames, Etr, steps, nvis, with_outputs=True, batch_episodes=16, label="toy-bad")
            check("early-exit/predict mismatch is detected", False)
        except RuntimeError:
            check("early-exit/predict mismatch is detected", True)
        # aggregate end to end on the toy rows
        res = Path(tmp) / "results"
        (res / "parts").mkdir(parents=True)
        DL.write_rows(rr, res / "parts" / "predictor_s0_real.csv")
        DL.write_rows(out["mse_rows"], res / "parts" / "predictor_s0_real_mse.csv", fields=S.MSE_FIELDS)
        write_json({"predictor_layers": 6, "encoder_layers": 12, "partial": False}, res / "parts" / "meta.json")
        cfg_a = copy.deepcopy(cfg)
        cfg_a["pilot"] = False
        cfg_a["_config_path"] = cfg["_config_path"]
        paths = {"results_dir": res, "exp": Path(tmp)}
        sm2 = S.aggregate(cfg_a, c2, paths)
        check("aggregate writes depth_r2.csv, summary.json, next_z_mse.csv, results.json", all((res / f).exists() for f in ("depth_r2.csv", "summary.json", "next_z_mse.csv")) and (Path(tmp) / "results.json").exists())
        check("summary.json has pilot field (false from config; true if any part is partial)", read_json(res / "summary.json")["pilot"] is False)
        write_json({"predictor_layers": 6, "partial": True}, res / "parts" / "meta.json")
        check("partial cache forces pilot true", S.aggregate(cfg_a, c2, paths)["pilot"] is True)
        figs = read_json(Path(tmp) / "results.json")["figures"]
        if isinstance(figs, list):
            check("figures written (heatmap, curves)", all((res / f).exists() for f in figs))
        else:
            skip("figures", str(figs))
    finally:
        M.P_TOK, M.D_VIS = old
        shutil.rmtree(tmp, ignore_errors=True)


def EG_labels(c2, C):
    from exp002 import evalgrid as EG
    return EG.make_labels(c2, C["rows"], np.asarray(C["mask_feat"]), c2["q1"]["primary_target"])


def t_part_and_config(cfg, cfg2):
    print("D  Part resume, config explicitness")
    from . import stages as S
    d = tempfile.mkdtemp()
    try:
        paths = {"results_dir": Path(d)}
        p = S.Part(paths, "x", fresh=False)
        p.add([{"module": "encoder", "depth": 4, "probe": "linear", "fit": "new", "level": "ALL", "split": "all", "timepoint": "late", "seed": 0, "r2": 0.5, "mae": 0.1, "pearson": 0.7,
                "control": "real", "train_seed": "NA", "pred_mean": 0.0, "n_episodes": 3}])
        p2 = S.Part(paths, "x", fresh=False)
        check("Part.done sees rows written by an earlier run (resume)", p2.done(module="encoder", depth=4, control="real") and not p2.done(module="encoder", depth=6))
        check("fresh=True ignores old rows", not S.Part(paths, "x", fresh=True).rows)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    check("probe seeds explicit, 5", cfg["probe"]["seeds"] == [0, 1, 2, 3, 4])
    check("002 config loaded with 004 seeds/kinds override and sha recorded", cfg2["probe"]["seeds"] == [0, 1, 2, 3, 4] and len(cfg2["_config002_sha256"]) == 64)
    check("predictor feature_part is z_only (spec 3.2 (i))", cfg["predictor"]["feature_part"] == "z_only")
    check("002 num_hist=1, frameskip=5 (확정 기록 8)", cfg2["train"]["num_hist"] == 1 and cfg2["data_impl"]["frameskip"] == 5)


def run(cfg, cfg2):
    tests = [("A1", lambda: t_first_drop(cfg)), ("A2", lambda: t_summary(cfg)), ("A3", t_rows), ("A4", t_pixels), ("B1", lambda: t_encoder_probe_planted(cfg, cfg2)),
             ("D", lambda: t_part_and_config(cfg, cfg2))]
    if have_torch():
        tests += [("C1", t_torch_units), ("C2", lambda: t_torch_pipeline(cfg, cfg2))]
    else:
        skip("C1/C2 torch tests", "no torch")
    for name, fn in tests:
        try:
            fn()
        except Exception:
            REC["fail"].append(name + " (exception)")
            traceback.print_exc()
    print("selftest summary: passed=%d failed=%d skipped=%d" % (len(REC["pass"]), len(REC["fail"]), len(REC["skipped"])))
    if REC["fail"]:
        print("FAILED:", REC["fail"])
    return 1 if REC["fail"] else 0
