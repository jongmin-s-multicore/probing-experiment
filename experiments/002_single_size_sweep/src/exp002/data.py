"""Datasets and manifests. Adapted from experiments/001_size_collapse_diagnosis/src/exp001/data.py (ManifestDataset logic copied from the
branch class datasets/acwm_phys_dset.py @ code.commit as in 001; selection logic rewritten for 002 conditions).

Conditions (config.conditions.list, episodes_per_level per condition; 확정 기록 6 equal budget): (b) multi_60_ind_160 = first 300 episodes of
a060_train + ind_train + a160_train; (a) single_ind = first 900 episodes of ind_train (the first 300 are the ones (b) uses for ind);
multi_60_ind_160_small (optional) = first 100 of each of the three levels (subset of (b)). Evaluation: the first 100 episodes of every *_test split.
Probe training episodes = the (b) training episodes (config q1.probe_train_levels_from = data.train_levels_b) for BOTH conditions.
hydra target for train.py:  exp002.data.load_manifest_slice_train_val   (needs src/ on PYTHONPATH, see run.py train)"""
from __future__ import annotations

import math
from pathlib import Path

from .common import write_json, read_json


# ------------------------------------------------------------------------------------------ pure logic (no torch)
def split_name(cfg, level, part):
    return cfg["data_impl"]["split_pattern"].format(size=level, part=part)


def level_roles(cfg):
    """level -> trained | interp | extrap (from config; trained = data.train_levels_b, since probes are trained on those)."""
    roles = {l: "trained" for l in cfg["data"]["train_levels_b"]}
    for r, ls in cfg["data"]["roles"].items():
        for l in ls:
            roles[l] = r
    missing = [l for l in cfg["data"]["levels"] if l not in roles]
    if missing:
        raise ValueError("levels without a role: %s" % missing)
    return roles


def nominal_log_area(cfg, level):
    return math.log(cfg["data_impl"]["area_ratio"][level])


def select_first(names, n):
    if len(names) < n:
        raise ValueError("only %d episodes available, %d requested" % (len(names), n))
    return names[:n]


def cond_episodes_per_level(cfg, cond):
    """Episodes per training level of a condition: conditions.list[].episodes_per_level, else data.train_episodes_per_level."""
    for c in cfg["conditions"]["list"]:
        if c["name"] == cond:
            return int(c.get("episodes_per_level", cfg["data"]["train_episodes_per_level"]))
    raise KeyError(cond)


def build_train_manifest(cfg, listing, data_root, cond):
    n = cond_episodes_per_level(cfg, cond)
    eps = []
    for lvl in _cond_levels(cfg, cond):
        sp = split_name(cfg, lvl, "train")
        for nm in select_first(listing[sp], n):
            eps.append({"dir": str(Path(data_root) / sp), "split": sp, "name": nm, "group": lvl})
    keys = [(e["split"], e["name"]) for e in eps]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate episodes")
    gc = {}
    for e in eps:
        gc[e["group"]] = gc.get(e["group"], 0) + 1
    return {"name": cond, "n_episodes": len(eps), "group_counts": gc, "episodes": eps}


def _cond_levels(cfg, cond):
    for c in cfg["conditions"]["list"]:
        if c["name"] == cond:
            return c["levels"]
    raise KeyError(cond)


def build_cache_table(cfg, listing, data_root):
    """Episodes whose frames are cached: probe-training episodes (part=train, levels = data.train_levels_b, first N of *_train)
    followed by evaluation episodes (part=test, all levels, first M of *_test). Row order is fixed and shared by all stages."""
    roles = level_roles(cfg)
    rows = []
    for part, levels, n in (("train", cfg["data"]["train_levels_b"], cfg["data"]["train_episodes_per_level"]),
                            ("test", cfg["data"]["levels"], cfg["data"]["eval_episodes_per_level"])):
        for lvl in levels:
            sp = split_name(cfg, lvl, part)
            for i, nm in enumerate(select_first(listing[sp], n)):
                rows.append({"id": len(rows), "part": part, "level": lvl, "role": roles[lvl], "split": sp, "name": nm,
                             "dir": str(Path(data_root) / sp), "rank_in_level": i,
                             "nominal_log_area": nominal_log_area(cfg, lvl)})
    return rows


# ------------------------------------------------------------------------------------------ torch part
def list_episodes(data_root, splits):
    import torch
    names, meta_out = {}, {}
    for s in splits:
        d = Path(data_root) / s
        if not (d / "metadata.pt").exists():
            continue
        meta = torch.load(d / "metadata.pt", map_location="cpu", weights_only=True)
        names[s] = [Path(e["video_path"]).name for e in meta]
        meta_out[s] = {Path(e["video_path"]).name: {k: (int(e[k]) if k in e and e[k] is not None else None)
                                                    for k in ("cloth_dimx", "cloth_dimy", "seed", "n_attempts", "length")} for e in meta}
    return names, meta_out


def make_dataset_class():
    """ManifestDataset(ACWMPhysDataset) over an arbitrary list of episodes. Requires the branch checkout on sys.path."""
    import torch
    from datasets.acwm_phys_dset import ACWMPhysDataset

    class ManifestDataset(ACWMPhysDataset):
        def __init__(self, episodes, transform=None, normalize_action=False, action_dims=None, action_format="raw", action_stats=None):
            self.transform = transform
            cache, meta = {}, []
            for ep in episodes:
                d = ep["dir"]
                if d not in cache:
                    m = torch.load(Path(d) / "metadata.pt", map_location="cpu", weights_only=True)
                    cache[d] = {Path(e["video_path"]).name: e for e in m}
                meta.append(cache[d][ep["name"]])
            self.ep_dirs = [Path(ep["dir"]) for ep in episodes]
            self.video_names = [ep["name"] for ep in episodes]
            self.ep_groups = [ep.get("group") for ep in episodes]
            if action_format != "raw":
                raise ValueError("Cloth Move uses action_format=raw")
            dims = list(action_dims) if action_dims is not None else list(range(meta[0]["actions"].shape[1]))
            self.raw_actions = [e["actions"][:, dims].float() for e in meta]
            self.seq_lengths = torch.tensor([len(a) for a in self.raw_actions])
            self.n_obs = [min(int(e["length"]), len(a) + 1) for e, a in zip(meta, self.raw_actions)]
            self.action_dim = self.raw_actions[0].shape[1]
            self.state_dim = 1
            self.proprio_dim = 1
            if normalize_action:
                if action_stats is None:
                    allact = torch.cat(self.raw_actions)
                    action_stats = (allact.mean(0), allact.std(0) + 1e-6)
                self.action_mean, self.action_std = action_stats
            else:
                self.action_mean = torch.zeros(self.action_dim)
                self.action_std = torch.ones(self.action_dim)
            self.actions = [(a - self.action_mean) / self.action_std for a in self.raw_actions]

        def get_frames(self, idx, frames):
            import decord
            from einops import rearrange
            from datasets.acwm_pour_water_dset import _to_tensor
            frames = list(frames)
            reader = decord.VideoReader(str(self.ep_dirs[idx] / self.video_names[idx]), num_threads=1)
            image = rearrange(_to_tensor(reader.get_batch(frames)), "T H W C -> T C H W").float() / 255.0
            if self.transform:
                image = self.transform(image)
            zeros = torch.zeros(len(frames), 1)
            return {"visual": image, "proprio": zeros}, zeros.clone()

    return ManifestDataset


def load_manifest_slice_train_val(transform, data_path, manifest, n_rollout=None, exclude=(), normalize_action=False, action_dims=None,
                                  action_format="raw", split_ratio=0.9, num_hist=0, num_pred=0, frameskip=0, val_split_seed=42):
    """Drop-in replacement of datasets.acwm_phys_dset.load_acwm_phys_slice_train_val for a manifest (same as 001, 0.9/0.1 split seed 42)."""
    from datasets.traj_dset import split_traj_datasets
    from datasets.acwm_pour_water_dset import LazyTrajSlicerDataset
    eps = read_json(manifest)["episodes"]
    if n_rollout:
        eps = eps[:n_rollout]
    ds = make_dataset_class()(eps, transform=transform, normalize_action=normalize_action, action_dims=action_dims, action_format=action_format)
    dset_train, dset_val = split_traj_datasets(ds, train_fraction=split_ratio, random_seed=val_split_seed)
    nf = num_hist + num_pred
    datasets = {"train": LazyTrajSlicerDataset(dset_train, nf, frameskip), "valid": LazyTrajSlicerDataset(dset_val, nf, frameskip)}
    return datasets, {"train": dset_train, "valid": dset_val}


def count_train_slices(manifest, num_hist, num_pred, frameskip, action_dims, split_ratio=0.9, val_split_seed=42):
    """(n_train_slices, n_valid_slices) of the branch loader for a manifest. Reuses load_manifest_slice_train_val (same 0.9/0.1 episode split, seed
    val_split_seed, same LazyTrajSlicerDataset) with transform=None, so only metadata.pt is read and no video is decoded (get_frames is never called).
    Needs the branch checkout on sys.path. Consistency with the first server smoke (num_hist=1, 48 episodes, batch 24: 99 train / 12 valid iterations)
    is checked arithmetically in selftest (see expected_slices); the branch slicer code itself was not read here."""
    ds, _ = load_manifest_slice_train_val(None, None, manifest, action_dims=action_dims, split_ratio=split_ratio, num_hist=num_hist, num_pred=num_pred,
                                          frameskip=frameskip, val_split_seed=val_split_seed)
    return len(ds["train"]), len(ds["valid"])


def expected_slices(n_obs_list, nf, frameskip):
    """ASSUMPTION inferred from the server smoke observation (not from branch code): an episode with n_obs frames gives max(0, n_obs - nf*frameskip)
    slices, nf = num_hist + num_pred. Used only for a plausibility check of count_train_slices output, never to replace it."""
    return sum(max(0, int(n) - nf * frameskip) for n in n_obs_list)
