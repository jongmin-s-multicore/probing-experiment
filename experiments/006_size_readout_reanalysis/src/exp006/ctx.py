"""Shared loading for the CPU stages: 002 cache (read only), rows, labels, role masks, stored 002 cells, extracted rollouts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import base as B
from .base import log


def load_cache002(paths):
    """Read-only memmaps of the 002 Q1 cache (same files as exp002 cache.load, but tokens_rand.npy is not needed here)."""
    cd = Path(paths["cache002"])
    if not (cd / "DONE").exists():
        raise FileNotFoundError("%s/DONE missing (002 cache)" % cd)
    out = {n: np.load(cd / (n + ".npy"), mmap_mode="r") for n in ("frames_u8", "tokens_dino", "mask_feat", "actions_raw")}
    out["rows"] = B.read_json(cd / "episodes.json")
    out["meta"] = B.read_json(cd / "meta.json")
    return out


class Ctx:
    def __init__(self, cfg, cfg2, paths):
        from exp002 import evalgrid as EG
        self.cfg, self.cfg2, self.paths = cfg, cfg2, paths
        self.C = load_cache002(paths)
        rows = self.C["rows"]
        self.rows = rows
        self.tr_rows, self.te_rows = EG.split_rows(rows)
        self.Etr, self.Ete, self.E = len(self.tr_rows), len(self.te_rows), len(rows)
        assert all(r["part"] == "train" for r in rows[:self.Etr]) and all(r["part"] == "test" for r in rows[self.Etr:])
        self.levels = list(cfg2["data"]["levels"])
        self.lvl_idx = EG.level_index(self.te_rows, self.levels)
        self.nominal = np.array([r["nominal_log_area"] for r in self.te_rows])
        self.rmasks = EG.role_masks(self.te_rows)
        self.prim = cfg["task"]["primary_target"]
        self.Yp = EG.make_labels(cfg2, rows, np.asarray(self.C["mask_feat"]), self.prim)       # (E,13)
        self.t = int(cfg["task"]["context_frame_t"])
        self.steps = [int(s) for s in cfg["analysis"]["steps"]]
        self.horizon = int(cfg["task"]["rollout_horizon"])
        self.frames_tr = EG.windows(cfg2)[cfg["task"]["probe_train_regime"]]
        self.probe_seeds = [int(s) for s in cfg["seeds"]["probe"]]
        self.EG = EG

    def role_masks(self):
        return {r: self.rmasks[r] for r in self.cfg["analysis"]["roles"]}

    def label_te(self, k):
        return self.Yp[self.Etr:, self.t + k]

    def results002_cells(self, run):
        p = Path(self.paths["results002"]) / run / "q2_cells.npz"
        if not p.exists():
            raise FileNotFoundError("%s missing (002 Drive output)" % p)
        return np.load(p)

    def out_key(self, kind, k):
        return "%s|linear_pooled|OUT|%s|step%d" % (self.prim, kind, k)

    def stored_pred(self, z, run, kind, k):
        """(P, Ete) stored per-episode probe predictions of 002 (predmean|...)."""
        return np.asarray(z["predmean|" + self.out_key(kind, k)], float)

    def stored_cell(self, z, run, kind, k):
        return np.asarray(z[self.out_key(kind, k)], float)

    def results002_json(self):
        cfg = self.cfg
        cands = [B.EXP_DIR / cfg["base"]["results002"], Path(self.paths["results002"]) / "results.json"]
        for p in cands:
            p = Path(p).resolve()
            if p.exists():
                sha = B.sha256_file(p)
                return B.read_json(p), {"path": str(p), "sha256": sha, "matches_config_sha256": sha == cfg["base"]["results002_sha256"]}
        raise FileNotFoundError("002 results.json not found in: %s" % [str(c) for c in cands])


def load_pooled(paths):
    p = Path(paths["cache"]) / "pooled_tokens.npy"
    if not p.exists():
        raise FileNotFoundError("%s missing - run `python run.py pool`" % p)
    return np.load(p)


def load_extract(paths, run, with_untrained=True):
    d = Path(paths["extract"]) / run
    if not (d / "DONE").exists():
        raise FileNotFoundError("%s/DONE missing - run `python run.py extract` (GPU) first" % d)
    out = {"roll": np.load(d / "rollout_pooled.npy"), "meta": B.read_json(d / "meta.json"), "untrained": {}, "tf": None}
    if (d / "teacher_forced_pooled.npy").exists():              # analysis F (teacher forcing); absent in extractions made before F existed
        out["tf"] = np.load(d / "teacher_forced_pooled.npy")
    if with_untrained:
        for m in out["meta"]["untrained"]:
            out["untrained"][(m["variant"], int(m["seed"]))] = np.load(d / ("untrained_%s_s%d.npy" % (m["variant"], m["seed"])))
    return out


def save_part(ctx_or_paths, name, obj, stage_info=None):
    paths = ctx_or_paths.paths if hasattr(ctx_or_paths, "paths") else ctx_or_paths
    obj = dict(obj)
    obj["stage"] = name
    if stage_info:
        obj["stage_info"] = stage_info
    B.write_json(paths, obj, Path(paths["parts"]) / (name + ".json"))
    log("wrote part %s" % (Path(paths["parts"]) / (name + ".json")))


def jsonable_agg(d):
    """seed_agg dict -> json-able (arrays -> lists)."""
    return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
