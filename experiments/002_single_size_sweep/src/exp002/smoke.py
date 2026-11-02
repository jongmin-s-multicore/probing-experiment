"""`python run.py smoke` - measure memory/time on the 12GB GPU server (GPU, needs data + manifest). Writes results/smoke.json.
 (b) predictor training: branch train.py is run for 1 epoch on a few episodes for batch candidates (descending) x num_hist candidates;
     peak VRAM = max GPU memory.used seen by nvidia-smi polling minus the value before the start (other processes on the GPU are counted -> noted);
     OOM = non-zero exit + 'out of memory' in the output. Step time is NOT isolated (wall time of the whole short run + parsed tqdm rate if any).
 (a) one episode: mp4 decode -> branch transform (224) -> encoder tokens, time and torch peak VRAM  (uses the checkpoint written by (b)).
 (c) Q2 hook inference on a batch: predictor hooks, rollout, shift probe; interface assumptions are checked and recorded.
Never edits config.yaml; prints a recommendation only."""
from __future__ import annotations

import math
import re
import subprocess
import threading
import time

import numpy as np

from .common import log, read_json, write_json, env_info, warn_conda, run_dir, config_sha256
from . import trainrun as TR


class VramPoller(threading.Thread):
    def __init__(self, period=0.5):
        super().__init__(daemon=True)
        self.period, self.peak, self.base, self._stop = period, 0, self._read(), False

    @staticmethod
    def _read():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            return max(int(x) for x in r.stdout.split())
        except Exception:
            return 0

    def run(self):
        while not self._halt:
            self.peak = max(self.peak, self._read())
            time.sleep(self.period)

    def finish(self):
        self._halt = True
        self.join(2)
        return {"baseline_mib": self.base, "peak_mib": self.peak, "delta_mib": self.peak - self.base}


def step_b(cfg, paths, cond, seed, out):
    sc = cfg["smoke"]
    man = paths["manifests"] / ("train_%s.json" % cond)
    if not man.exists():
        raise FileNotFoundError("%s missing - run `python run.py manifest` first" % man)
    res = {}
    last_ok = None
    for H in sc["num_hist_list"]:
        res[str(H)] = {"trials": []}
        for bs in sc["batch_candidates"]:
            name = "exp002_smoke_h%d_b%d" % (H, bs)
            rd = run_dir(paths, name)
            cmd, env = TR.build_cmd(cfg, paths, man, seed, rd, smoke={"num_hist": H, "batch_size": bs, "n_rollout": sc["train_episodes"]})
            rd.mkdir(parents=True, exist_ok=True)
            read_json(man)
            write_json(read_json(man), rd / "train_manifest.json")
            log("smoke (b) num_hist=%d batch=%d" % (H, bs))
            poll = VramPoller()
            poll.start()
            t0 = time.time()
            rc, last = TR.stream(cmd, paths["acwm_src"], env, paths["logs"] / ("smoke_train_h%d_b%d.log" % (H, bs)))
            wall = time.time() - t0
            v = poll.finish()
            txt = "\n".join(last)
            oom = rc != 0 and bool(re.search(r"out of memory|CUDA error: out of memory|OutOfMemory", txt, re.I))
            rates = re.findall(r"([0-9.]+)\s*(it/s|s/it)", txt)
            trial = {"batch_size": bs, "returncode": rc, "oom": oom, "wall_s": round(wall, 1), "vram": v, "last_tqdm_rate": " ".join(rates[-1]) if rates else None,
                     "n_rollout_episodes": sc["train_episodes"], "note": "vram delta includes any other process on the GPU; wall time includes start-up"}
            if rc != 0 and not oom:
                trial["error_tail"] = last[-15:]
            res[str(H)]["trials"].append(trial)
            write_json({**out, "step_b_partial": res}, paths["results_dir"] / "smoke.json")
            if rc == 0:
                res[str(H)]["max_batch_that_fits"] = bs
                last_ok = (name, H, bs)
                break
            if not oom:
                log("smoke (b): non-OOM failure (rc=%d); see logs/smoke_train_h%d_b%d.log. Stopping this num_hist." % (rc, H, bs))
                break
        if "max_batch_that_fits" not in res[str(H)]:
            res[str(H)]["max_batch_that_fits"] = None
    out["step_b"] = res
    return last_ok


def step_a_c(cfg, paths, run_name_ok, out):
    import torch
    import hydra
    from . import model_io as M
    from .data import make_dataset_class
    sc = cfg["smoke"]
    dev = torch.device("cuda")
    rows = read_json(paths["manifests"] / "cache_episodes.json")[: max(sc["n_extract_episodes"], sc["q2_batch_episodes"])]
    rd = run_dir(paths, run_name_ok)
    model, tcfg, ep = M.load_world_model(paths["acwm_src"], rd, dev)
    tf = hydra.utils.instantiate(tcfg.env.dataset.transform)
    ds = make_dataset_class()(rows, transform=tf, normalize_action=False, action_dims=list(cfg["data_impl"]["action_dims"]))
    K = 13
    torch.cuda.reset_peak_memory_stats()
    ta = {}
    t0 = time.time()
    obs = [ds.get_frames(i, range(0, 5 * (K - 1) + 1, 5))[0]["visual"] for i in range(sc["n_extract_episodes"])]
    ta["decode_and_resize_s_per_episode"] = (time.time() - t0) / sc["n_extract_episodes"]
    vis = torch.stack(obs).to(dev)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        z = M.encode_batch(model, vis, torch.zeros(vis.shape[0], K, 30, device=dev))
    torch.cuda.synchronize()
    ta["encode_s_per_episode"] = (time.time() - t0) / vis.shape[0]
    ta["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    ta["z_all_shape"] = list(z.shape)
    n_all = cfg["data"]["train_episodes_per_level"] * len(cfg["data"]["train_levels_b"]) + cfg["data"]["eval_episodes_per_level"] * len(cfg["data"]["levels"])
    ta["cache_time_estimate_min_for_%d_episodes_decode_plus_encode_x1.5_for_random_encoder" % n_all] = round(n_all * (ta["decode_and_resize_s_per_episode"] + 1.5 * ta["encode_s_per_episode"]) / 60, 1)
    out["step_a_one_episode"] = ta
    # ---- (c)
    chk = {}
    L = M.n_blocks(model)
    H = int(tcfg.num_hist)
    chk.update({"n_blocks": L, "num_hist": H, "encoder_emb_dim": int(model.encoder.emb_dim), "emb_dim_equals_D_VIS": int(model.encoder.emb_dim) == M.D_VIS})
    nb = sc["q2_batch_episodes"]
    obs = [ds.get_frames(i, range(0, 5 * (K - 1) + 1, 5))[0]["visual"] for i in range(nb)]
    vis = torch.stack(obs).to(dev)
    acts = torch.zeros(nb, K, 30, device=dev)
    torch.cuda.reset_peak_memory_stats()
    hooks = M.StreamHooks(model).install()
    frames = list(range(max(H - 1, 0), 12))
    t0 = time.time()
    with torch.no_grad():
        z_all = M.encode_batch(model, vis, acts)
        zw = M.make_windows(z_all, H, frames)
        hooks.reset()
        hooks.capture_layers = set(range(L + 1))
        o = model.predict(zw)
        chk["predict_out_shape"] = list(o.shape)
        chk["captured_shape_layer0"] = list(hooks.store[0].shape)
        chk["tokens_per_frame_is_196_frame_major_assumed"] = hooks.store[0].shape[1] == H * M.P_TOK
        cap = {j: hooks.store[j].clone() for j in range(L + 1)}
        hooks.reset()
        hooks.modify = {3 if L > 3 else 0: lambda x: x}
        o2 = model.predict(zw)
        chk["identity_modification_max_abs_diff"] = float((o2 - o).abs().max())
        hooks.reset()
        j0 = L - 1
        delta = torch.zeros(cap[j0].shape[-1], device=dev)
        delta[0] = 1.0
        hooks.modify = {j0: lambda x: x + delta}
        hooks.capture_layers = {L}
        model.predict(zw)
        proj = float(((hooks.store[L] - cap[L])[..., 0]).mean())
        chk["residual_path_probe_mean_change_along_delta_at_layer_L"] = proj
        chk["residual_path_probe_ok_if_ge_0.5"] = proj >= 0.5
        hooks.reset()
        R = M.rollout(model, z_all, acts, H - 1, min(12, 12 - (H - 1)), H)
        chk["rollout_shape"] = list(R.shape)
    torch.cuda.synchronize()
    chk["q2_hook_inference_s_for_%d_episodes" % nb] = time.time() - t0
    chk["q2_hook_inference_peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    # shifted predictor pass of q3_impl.batch_samples windows
    torch.cuda.reset_peak_memory_stats()
    bsz = cfg["q3_impl"]["batch_samples"]
    big = zw.repeat(math.ceil(bsz / zw.shape[0]), 1, 1, 1)[:bsz]
    hooks.modify = {L // 2: lambda x: x * 1.0}
    with torch.no_grad():
        model.predict(big)
    torch.cuda.synchronize()
    chk["q3_shifted_pass_peak_vram_mib_for_%d_windows" % bsz] = torch.cuda.max_memory_allocated() / 2 ** 20
    hooks.remove()
    out["step_c_q2_q3_interface"] = chk


def run(cfg, paths, dry_run=False):
    sc = cfg["smoke"]
    cond = cfg["conditions"]["run_order"][0]
    log("smoke plan: (b) train.py 1 epoch on %d episodes, num_hist %s x batch %s ; (a) one-episode extraction ; (c) hook inference. Output results/smoke.json" % (
        sc["train_episodes"], sc["num_hist_list"], sc["batch_candidates"]))
    out = {"env": env_info(), "config_sha256": config_sha256(cfg["_config_path"]), "conda_env_active": warn_conda(),
           "note": "numbers only; 12GB-server measurement. config.yaml is not modified."}
    if dry_run:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("smoke needs a CUDA GPU (torch.cuda.is_available() is False)")
        out["gpu_total_mib"] = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    except ImportError:
        raise RuntimeError("smoke needs torch")
    t0 = time.time()
    ok = step_b(cfg, paths, cond, cfg["train"]["seeds"][0], out)
    if ok is None:
        out["step_a_one_episode"] = out["step_c_q2_q3_interface"] = "skipped: no successful smoke training run (no checkpoint)"
    else:
        step_a_c(cfg, paths, ok[0], out)
    # recommendation (printed + stored, config untouched)
    rec = []
    for H, r in out["step_b"].items():
        mb = r.get("max_batch_that_fits")
        if mb is None:
            rec.append("num_hist=%s: no batch candidate trained successfully" % H)
        elif mb >= 24:
            rec.append("num_hist=%s: batch 24 fits (peak delta %s MiB); batch_size 24 as in 001 is possible" % (H, r["trials"][-1]["vram"]["delta_mib"]))
        else:
            rec.append("num_hist=%s: batch 24 does NOT fit; largest candidate that fits = %d. Recommendation: train.batch_size=%d (and gradient accumulation %d to keep the effective 24 IF the branch train.py supports it - unverified; otherwise record the batch change vs 001)" % (H, mb, mb, math.ceil(24 / mb)))
    out["recommendation"] = rec
    out["total_wall_s"] = time.time() - t0
    write_json(out, paths["results_dir"] / "smoke.json")
    for r in rec:
        log("RECOMMENDATION (config not changed): " + r)
    log("wrote results/smoke.json")
    return out
