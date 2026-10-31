"""Datasets and manifests.

Manifest = ordered list of episodes {dir, split, name, group}. The training distribution (single size vs. mixed sizes) is
chosen by config.train.distributions; the same loader is used for every distribution so that data amount, epochs and seed
are identical across them. `baseline_single` takes ind_train in the original metadata order, i.e. the same 2000 episodes as
the branch's own Cloth Move run (the 0.9/0.1 split uses the same seed 42 as the branch loader).

hydra target for train.py:  exp001.data.load_manifest_slice_train_val   (needs src/ on PYTHONPATH, see run.py train)
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import group_of, write_json, read_json


# ----------------------------------------------------------------------------- pure manifest logic (no torch)
def select_episodes(listing, specs, cfg, data_root):
    """listing: {split: [file names in metadata order]}. Returns list of episode dicts."""
    out = []
    for spec in specs:
        split_key = spec["source"]
        names = [n for n in listing[split_key] if group_of(split_key, n, cfg) == spec["group"]]
        if split_key == "ood_test":
            names = sorted(names)  # deterministic: sorted file names, then first/last
        if spec["take"] == "first":
            chosen = names[: spec["n"]]
        elif spec["take"] == "last":
            chosen = names[len(names) - spec["n"] :]
        else:
            raise ValueError(spec["take"])
        if len(chosen) != spec["n"]:
            raise ValueError(f"{spec}: only {len(chosen)} episodes available")
        for n in chosen:
            out.append({"dir": str(Path(data_root) / split_key), "split": split_key, "name": n, "group": group_of(split_key, n, cfg)})
    keys = [(e["split"], e["name"]) for e in out]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate episodes in manifest")
    return out


def build_manifests(cfg, listing, data_root):
    """Returns {dist_name: manifest dict}. Checks that all distributions have the same number of episodes."""
    total = cfg["train"]["total_episodes"]
    res = {}
    for name, specs in cfg["train"]["distributions"].items():
        eps = select_episodes(listing, specs, cfg, data_root)
        if len(eps) != total:
            raise ValueError(f"distribution {name}: {len(eps)} episodes != train.total_episodes {total} (data amount must match)")
        groups = {}
        for e in eps:
            groups[e["group"]] = groups.get(e["group"], 0) + 1
        res[name] = {"name": name, "seed": cfg["train"]["seed"], "n_episodes": len(eps), "group_counts": groups, "episodes": eps}
    return res


def eval_episode_table(listing, cfg, data_root):
    """All episodes that may be evaluated: every split that is not a *_train split (cloth_sim: the 9 *_test splits; legacy: ind_test,
    ood_test). Group inferred from split/file name. Training splits are never evaluated, so training and evaluation episodes are disjoint;
    the legacy mix distribution (which takes ood_test episodes) is handled by the in_train flag in extract."""
    rows = []
    for split, names in listing.items():
        if split.endswith("_train"):
            continue
        ordered = sorted(names) if split == "ood_test" else list(names)
        counters = {}
        for n in ordered:
            g = group_of(split, n, cfg)
            counters[g] = counters.get(g, 0) + 1
            rows.append({"dir": str(Path(data_root) / split), "split": split, "name": n, "group": g, "rank_in_group": counters[g]})
    return rows


# ----------------------------------------------------------------------------- torch part
def list_episodes(data_root, splits):
    """Reads metadata.pt of each split dir (a pickle: only with weights_only=True).
    Returns ({split: [names in metadata order]}, {split: {name: {"dimx","dimy","seed","n_attempts","length"}}}).
    cloth_dimx/cloth_dimy are particle counts in cloth_sim (they differ by size by design)."""
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


def _make_dataset_class():
    import torch
    from datasets.acwm_phys_dset import ACWMPhysDataset

    class ManifestDataset(ACWMPhysDataset):
        """ACWMPhysDataset over an arbitrary list of episodes (different split dirs). Logic copied from the branch class
        (datasets/acwm_phys_dset.py @ code.commit); only the episode source differs."""

        def __init__(self, episodes, transform=None, normalize_action=False, action_dims=None, action_format="raw", action_stats=None):
            self.transform = transform
            cache = {}
            meta = []
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
            self.actions = [e["actions"][:, dims].float() for e in meta]
            self.seq_lengths = torch.tensor([len(a) for a in self.actions])
            self.n_obs = [min(int(e["length"]), len(a) + 1) for e, a in zip(meta, self.actions)]
            self.action_dim = self.actions[0].shape[1]
            self.state_dim = 1
            self.proprio_dim = 1
            if normalize_action:
                if action_stats is None:
                    allact = torch.cat(self.actions)
                    action_stats = (allact.mean(0), allact.std(0) + 1e-6)
                self.action_mean, self.action_std = action_stats
            else:
                self.action_mean = torch.zeros(self.action_dim)
                self.action_std = torch.ones(self.action_dim)
            self.actions = [(a - self.action_mean) / self.action_std for a in self.actions]

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


def load_manifest_slice_train_val(
    transform,
    data_path,          # unused (episodes carry their own dirs); kept because conf/env passes it
    manifest,
    n_rollout=None,
    exclude=(),
    normalize_action=False,
    action_dims=None,
    action_format="raw",
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    """Drop-in replacement of datasets.acwm_phys_dset.load_acwm_phys_slice_train_val for a manifest."""
    from datasets.traj_dset import split_traj_datasets
    from datasets.acwm_pour_water_dset import LazyTrajSlicerDataset

    man = read_json(manifest)
    eps = man["episodes"]
    if n_rollout:
        eps = eps[:n_rollout]
    ds = _make_dataset_class()(eps, transform=transform, normalize_action=normalize_action, action_dims=action_dims, action_format=action_format)
    dset_train, dset_val = split_traj_datasets(ds, train_fraction=split_ratio, random_seed=42)
    num_frames = num_hist + num_pred
    datasets = {
        "train": LazyTrajSlicerDataset(dset_train, num_frames, frameskip),
        "valid": LazyTrajSlicerDataset(dset_val, num_frames, frameskip),
    }
    return datasets, {"train": dset_train, "valid": dset_val}


def build_eval_dataset(episodes, train_manifest_path, transform, normalize_action, action_dims):
    """Eval episodes with the action statistics of the training manifest (as the branch's load_acwm_phys_split does)."""
    cls = _make_dataset_class()
    train = read_json(train_manifest_path)["episodes"]
    tr = cls(train, normalize_action=normalize_action, action_dims=action_dims)
    return cls(episodes, transform=transform, normalize_action=normalize_action, action_dims=action_dims,
               action_stats=(tr.action_mean, tr.action_std) if normalize_action else None)
