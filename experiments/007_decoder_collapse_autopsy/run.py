"""007 decoder collapse autopsy. 재학습 없이 003 체크포인트(Drive 휴지통, 읽기 전용)를 CPU 로 검사한다.

사용:
  python run.py fetch            # 코드(git archive), 체크포인트, 프레임 표본을 scratchpad 로 내려받고 slim 파일 생성
  python run.py analyze          # 검사 B..E 와 판단 기준 표 -> results.json, figures/
  python run.py analyze --smoke  # 체크포인트 1개, 프레임 몇 장 (smoke_results.json)
해석은 쓰지 않는다. 수치와 판단 기준 해당/비해당만 기록한다.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

here = pathlib.Path(__file__).parent
KST = datetime.timezone(datetime.timedelta(hours=9))


def load_cfg():
    return yaml.safe_load((here / "config.yaml").read_text())


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def all_ckpts(cfg):
    out = []
    for grp in ("collapsed", "control"):
        for run, eps in cfg["runs"][grp].items():
            for e in eps:
                out.append((grp, run, e))
    return out


def slim_path(cfg, run, e):
    return pathlib.Path(cfg["paths"]["ckpt_dir"]) / "slim" / f"{run}__model_{e}.pt"


def tstats(t):
    """텐서 요약. NaN/Inf 는 세고 나머지로 통계를 낸다."""
    import torch
    t = t.detach().float()
    fin = torch.isfinite(t)
    nan = int(torch.isnan(t).sum())
    inf = int(torch.isinf(t).sum())
    tf = t[fin]
    n = int(t.numel())
    if tf.numel() == 0:
        return {"n": n, "nan": nan, "inf": inf}
    return {"n": n, "norm": float(tf.norm()), "mean": float(tf.mean()), "std": float(tf.std(unbiased=False)),
            "absmax": float(tf.abs().max()), "zero_frac": float((tf == 0).float().mean()), "nan": nan, "inf": inf}



# 검사 A. 코드 사실 (commit 5a640bed 의 파일, 줄 번호는 `git show 5a640bed:<file>` 기준; 003 은 colab/train_resume.patch 를 추가 적용)
CODE_FACTS = {
    "decoder_class": "models/vqvae.py:166-219 VQVAE (conf/decoder/vqvae.yaml: channel 384, n_embed 2048, n_res_block 4, n_res_channel 128, quantize False). 003 hydra_overrides 에 decoder 지정 없음 -> conf/train.yaml:7 기본값 vqvae. 체크포인트의 decoder 모듈 구조도 같음(관측).",
    "decoder_structure": "VQVAE.forward(models/vqvae.py:198-214): 입력 (b,t,196 토큰,384) -> 14x14 격자 -> upsample_b(Decoder stride 4: Conv3x3, ResBlock x4, ReLU, ConvT(384->192), ReLU, ConvT(192->384)) -> dec(같은 구조, 마지막 ConvT 192->3). Decoder: models/vqvae.py:131-163, ResBlock: 82-97 (ReLU, Conv3x3 384->128, ReLU, Conv1x1 128->384, 잔차 합). 출력 224x224, 출력 활성함수 없음.",
    "batchnorm": "없음 (models/vqvae.py 에 BatchNorm, GroupNorm, LayerNorm 없음; grep). 디코더에 dropout 없음.",
    "activation": "ReLU (models/vqvae.py:88,90,123,142 등). inplace=True 인 ReLU 포함.",
    "quantize": "quantize False (conf/decoder/vqvae.yaml:5, models/vqvae.py:206-209) -> diff 는 0, 코드북 갱신 없음.",
    "amp": "코드에 autocast, GradScaler, mixed_precision 인자 없음 (train.py, models/ grep). Accelerator(log_with='wandb') 는 train.py:61 (mixed_precision 인자 없음). Accelerate 설정 파일이나 환경변수에 의한 mixed precision 여부는 체크포인트로 구분 불가.",
    "grad_clip": "없음 (train.py, models/ 에 clip_grad 없음; grep).",
    "lr_scheduler": "없음 (grep).",
    "decoder_optimizer": "torch.optim.Adam(decoder.parameters(), lr=training.decoder_lr) train.py:336-338. decoder_lr 3e-4 (conf/train.yaml:40), 003 은 덮어쓰지 않음(체크포인트 param_group lr 로 확인). weight_decay 0.",
    "other_optimizers": "predictor: AdamW lr=predictor_lr (train.py:317-320; 003 은 5e-5 로 덮어씀, config.yaml:98), action/proprio encoder: AdamW lr=action_encoder_lr 5e-4 (train.py:325-330). encoder: Adam 이 만들어지지만 train_encoder False (conf/train.yaml:68) 라 step 안 함 (train.py:463).",
    "decoder_input_detach": "예. models/visual_world_model.py:254-256 self.decode(z.detach()) (재구성 경로). 예측 경로도 z_pred.detach() 이고 torch.no_grad (211-214) 이므로 손실에 기여하지 않음.",
    "decoder_loss_to_predictor": "역전파되지 않음: 재구성 손실 decoder_loss_reconstructed 는 loss 에 더해지지만(visual_world_model.py:271) 입력이 z.detach() 이고 encoder 는 동결이라 gradient 는 decoder 파라미터에만 닿음. predictor 는 z_loss(MSE, 목표 detach; 245-246)로만 학습. 한 번의 accelerator.backward(loss) (train.py:461).",
    "train_step": "train.py:443-536: model.train() -> forward -> 모든 optimizer zero_grad (454-459) -> backward (461) -> decoder, predictor, action_encoder optimizer step (463-469). 한 step 에 decoder 갱신 1회.",
    "val_structure": "train.py:538-632, @torch.no_grad. model.eval() 후 openloop_rollout(train/val), 이어 valid dataloader 전체에 대해 model(obs, act) 를 돌려 loss_components 를 로그. val_img_*_reconstructed 이미지 지표는 plot = (i == 0) 인 첫 valid 배치에서만 계산(574-620). 즉 val_img_mse_reconstructed 는 첫 배치 한 개의 값.",
    "eval_mode_effect": "val 은 model.eval() (visual_world_model.py:82-90) -> predictor dropout(0.1, conf/predictor/vit.yaml:5) 꺼짐. 디코더에는 BatchNorm, dropout 이 없어 train/eval 차이 없음(구조상).",
    "checkpoint_contents": "train.py:177-196 save_ckpt: _keys_to_save(156-170) = epoch, predictor, predictor_optimizer, decoder, decoder_optimizer, action_encoder, proprio_encoder (encoder 는 train_encoder False 라 제외) 를 torch.save. 원본 5a640bed 는 nn.Module 과 optimizer 객체 자체를 pickle. 003 패치(colab/train_resume.patch)는 action_encoder_optimizer 추가, optimizer 는 state_dict 로 저장. 관측된 키: epoch, predictor, predictor_optimizer, action_encoder_optimizer, decoder, decoder_optimizer, action_encoder, proprio_encoder. 저장되지 않는 것: encoder, RNG 상태, dataloader 위치, step 단위 loss, gradient.",
    "not_distinguishable_from_checkpoints": ["epoch 3 도중 정확한 시점", "단일 큰 gradient 여부", "Accelerate 의 mixed precision 설정 여부", "epoch 사이 batch 별 loss 궤적"],
}

# ----------------------------------------------------------------------------- fetch

def sh(cmd, timeout=900, check=True):
    log("$", " ".join(cmd) if isinstance(cmd, list) else cmd)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {r.stderr[-500:]}")
    return r


def retry(fn, n=3):
    last = None
    for i in range(n):
        try:
            return fn()
        except Exception as ex:  # noqa
            last = ex
            log(f"attempt {i + 1}/{n} failed: {ex}")
    raise RuntimeError(f"3회 실패: {last}")


def prepare_code(cfg):
    cd = pathlib.Path(cfg["paths"]["code_dir"])
    if (cd / "train.py").exists():
        return
    cd.mkdir(parents=True, exist_ok=True)
    p1 = subprocess.Popen(["git", "-C", cfg["code"]["thread0_dir"], "archive", cfg["code"]["repo_commit"], "--",
                           "train.py", "utils.py", "models", "conf", "distributed_fn", "datasets", "metrics"],
                          stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(cd)], stdin=p1.stdout, check=True)
    p1.wait()


def build_slim(cfg, run, e, raw, meta):
    """원본(368 MB)에서 predictor, decoder, action/proprio encoder 모듈과 optimizer 요약만 남긴다 (/tmp 가 tmpfs 3.8 GB)."""
    import torch
    sys.path.insert(0, cfg["paths"]["code_dir"])
    ck = torch.load(raw, map_location="cpu")
    assert ck["epoch"] == e, (ck["epoch"], e)
    nonfinite = {}

    def count_nonfinite(obj, key):
        n = 0
        for t in obj:
            n += int((~torch.isfinite(t.float())).sum())
        nonfinite[key] = n

    def opt_summary(osd, named):
        names = [n for n, _ in named]
        g = {k: v for k, v in osd["param_groups"][0].items() if k != "params"}
        per = {}
        tensors = []
        for idx, st in osd["state"].items():
            nm = names[int(idx)]
            tensors += [st["exp_avg"], st["exp_avg_sq"]]
            per[nm] = {"step": float(st["step"]), "exp_avg": tstats(st["exp_avg"]), "exp_avg_sq": tstats(st["exp_avg_sq"])}
        return {"param_group": g, "n_state": len(osd["state"]), "per_param": per}, tensors

    slim = {"run": run, "epoch": e, "ckpt_keys": list(ck.keys()), "meta": meta}
    for k in ("predictor", "decoder", "action_encoder", "proprio_encoder"):
        m = ck[k]
        count_nonfinite([p for p in m.parameters()], f"{k}_params")
        count_nonfinite([b for b in m.buffers()], f"{k}_buffers")
        slim[k] = m.cpu()
    for k, mods in (("decoder_optimizer", ["decoder"]), ("predictor_optimizer", ["predictor"]),
                    ("action_encoder_optimizer", ["action_encoder", "proprio_encoder"])):
        if k in ck:
            osd = ck[k]
            if not isinstance(osd, dict):
                osd = osd.state_dict()
            named = []
            for mn in mods:
                named += list(ck[mn].named_parameters())
            summ, tensors = opt_summary(osd, named)
            count_nonfinite(tensors, f"{k}_state")
            slim[k] = summ
    slim["nonfinite_in_checkpoint"] = nonfinite
    out = slim_path(cfg, run, e)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out)


def fetch(cfg):
    prepare_code(cfg)
    ckd = pathlib.Path(cfg["paths"]["ckpt_dir"])
    raw_dir = ckd / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    root = cfg["drive"]["remote_root"]
    meta_all = {}
    runs = sorted({r for _, r, _ in all_ckpts(cfg)})
    for run in runs:
        d = ckd / run
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("frames_sample.npz", "episodes.json"):
            if not (d / fn).exists():
                retry(lambda: sh(["rclone", "copy", f"{root}/{cfg['drive']['cache_subdir'].format(run=run)}", str(d),
                                  "--include", fn], 600))
        for fn in ("train_env.json", "run_info.txt", "DONE"):
            if not (d / fn).exists():
                retry(lambda: sh(["rclone", "copy", f"{root}/train_runs/outputs/{run}", str(d), "--include", fn], 300))
        sub = f"{root}/{cfg['drive']['trashed_ckpt_subdir'].format(run=run)}"
        r = retry(lambda: sh(["rclone", "lsf", sub, "--drive-trashed-only", "--format", "psti", "--separator", ";"], 300))
        listing = {}
        for line in r.stdout.strip().splitlines():
            p, s, t, i = line.split(";")
            listing[p] = {"size": int(s), "modtime_utc": t, "id": i,
                          "modtime_kst": (datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
                                          .astimezone(KST).strftime("%Y-%m-%d %H:%M:%S"))}
        for grp, rr, e in all_ckpts(cfg):
            if rr != run:
                continue
            sp = slim_path(cfg, run, e)
            fname = f"model_{e}.pth"
            meta = {"drive_path": f"{sub}/{fname}", "trashed": True, **listing[fname]}
            meta_all[f"{run}/model_{e}"] = meta
            if sp.exists():
                continue
            raw = raw_dir / f"{run}__model_{e}.pth"
            if not (raw.exists() and raw.stat().st_size == listing[fname]["size"]):
                tmp = raw_dir / f"tmp_{run}_{e}"
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir()
                retry(lambda: sh(["rclone", "copy", sub, str(tmp), "--drive-trashed-only", "--include", fname], 900))
                (tmp / fname).rename(raw)
                tmp.rmdir()
            h = hashlib.sha256()
            with open(raw, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            meta["sha256"] = h.hexdigest()
            meta["local_size"] = raw.stat().st_size
            build_slim(cfg, run, e, raw, meta)
            raw.unlink()   # 로컬 사본만 삭제. Drive 는 건드리지 않는다.
            log("slim done", sp)
    (ckd / "fetch_meta.json").write_text(json.dumps(meta_all, indent=2, ensure_ascii=False))
    log("fetch finished")


# ----------------------------------------------------------------------------- analyze

def load_set(npz_path, cfg, n_limit=None):
    z = np.load(npz_path)
    groups = sorted({k.split("/")[0] for k in z.files if k.endswith("/real")})
    frames, labels = [], []
    for g in groups:
        for ep in cfg["inputs"]["episodes"]:
            key = f"{g}/{ep}/real"
            if key not in z.files:
                continue
            for t in cfg["inputs"]["time_idx"]:
                frames.append(z[key][t])
                labels.append(f"{g}/ep{ep}/t{t}")
    x = np.stack(frames)
    if n_limit:
        sel = np.linspace(0, len(x) - 1, n_limit).round().astype(int)
        x, labels = x[sel], [labels[i] for i in sel]
    return x, labels


def encode_frames(cfg, u8):
    import torch
    from torchvision import transforms
    sys.path.insert(0, cfg["paths"]["code_dir"])
    torch.hub.set_dir(cfg["paths"]["hub_dir"])
    from models.dino import DinoV2Encoder
    enc = DinoV2Encoder(cfg["inputs"]["dino"]["name"], "x_norm_patchtokens").eval()
    tf = transforms.Resize(cfg["inputs"]["dino"]["encoder_input"])
    x = torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), 18):
            outs.append(enc(tf(x[i:i + 18])))
    return x, torch.cat(outs)


class Hooks:
    """디코더 층별 출력의 입력 간 std, ReLU 비활성 비율을 잰다. 배치마다 합계만 누적(텐서 저장 없음).
    입력 간 std = 샘플 축 std(모집단)를 원소별로 구해 평균. ReLU 입력 <= 0 비율과 채널 단위 완전 비활성 비율도 잰다."""

    def __init__(self, decoder):
        import torch.nn as nn
        from models.vqvae import ResBlock
        self.st, self.order, self.handles = {}, [], []
        for name, m in decoder.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.ReLU, ResBlock)):
                self.st[name] = {"type": type(m).__name__}
                self.handles.append(m.register_forward_hook(self._post(name)))
                if isinstance(m, nn.ReLU):
                    self.handles.append(m.register_forward_pre_hook(self._pre(name)))

    def _pre(self, name):
        def f(mod, inp):
            x = inp[0]
            s = self.st[name]
            le0 = (x <= 0)
            s["le0"] = s.get("le0", 0) + int(le0.sum())
            s["in_n"] = s.get("in_n", 0) + x.numel()
            act = (~le0).any(dim=0).any(dim=-1).any(dim=-1)      # (C,) 어떤 입력, 위치에서든 양수였던 채널
            s["chan_active"] = act if "chan_active" not in s else (s["chan_active"] | act)
            s["in_nan"] = s.get("in_nan", 0) + int(x.isnan().sum())
            s["in_inf"] = s.get("in_inf", 0) + int(x.isinf().sum())
        return f

    def _post(self, name):
        def f(mod, inp, out):
            o = out.detach()
            s = self.st[name]
            if name not in self.order:
                self.order.append(name)
                s["shape"] = list(o.shape[1:])
            od = o.double()
            s["n"] = s.get("n", 0) + o.shape[0]
            s["sum"] = od.sum(0) if "sum" not in s else s["sum"] + od.sum(0)
            s["sumsq"] = od.pow(2).sum(0) if "sumsq" not in s else s["sumsq"] + od.pow(2).sum(0)
            s["nan"] = s.get("nan", 0) + int(o.isnan().sum())
            s["inf"] = s.get("inf", 0) + int(o.isinf().sum())
        return f

    def finalize(self):
        recs = []
        for i, name in enumerate(self.order):
            s = self.st[name]
            mean = s["sum"] / s["n"]
            var = (s["sumsq"] / s["n"] - mean ** 2).clamp(min=0)
            rms = float((s["sumsq"].sum() / (s["n"] * mean.numel())).sqrt())
            rec = {"name": name, "type": s["type"], "shape": s["shape"], "order": i,
                   "across_input_std": float(var.sqrt().mean()), "rms": rms,
                   "across_input_std_over_rms": float(var.sqrt().mean()) / (rms + 1e-30),
                   "nan": s["nan"], "inf": s["inf"]}
            if "le0" in s:
                rec.update({"frac_input_le0": s["le0"] / s["in_n"],
                            "dead_channel_frac": float((~s["chan_active"]).float().mean()),
                            "input_nan": s["in_nan"], "input_inf": s["in_inf"]})
            recs.append(rec)
        return recs

    def remove(self):
        for h in self.handles:
            h.remove()


def run_decoder(slim, z, batch=9):
    """z (N,P,384) -> 층별 요약, 출력 이미지 (N,3,224,224)."""
    import torch
    dec = slim["decoder"].eval()
    hk = Hooks(dec)
    outs = []
    with torch.no_grad():
        for i in range(0, len(z), batch):
            img, _ = dec(z[i:i + batch].unsqueeze(1))
            outs.append(img)
    hk.remove()
    return hk.finalize(), torch.cat(outs)


def run_predictor(cfg, slim, z):
    import torch
    N = z.shape[0]
    with torch.no_grad():
        pro = slim["proprio_encoder"].eval()(torch.full((N, 1, 1), cfg["inputs"]["proprio_value"]))   # (N,1,10)
        act = slim["action_encoder"].eval()(torch.full((N, 1, 30), cfg["inputs"]["action_value"]))     # (N,1,10)
        P = z.shape[1]
        zin = torch.cat([z, pro.unsqueeze(2).expand(N, 1, P, -1)[:, 0], act.unsqueeze(2).expand(N, 1, P, -1)[:, 0]], dim=-1)  # (N,P,404)
        out = slim["predictor"].eval()(zin)
    vis_out = out[..., :z.shape[-1]]
    sd_out = vis_out.std(dim=0, unbiased=False).mean()
    sd_in = z.std(dim=0, unbiased=False).mean()
    return {"out_across_input_std": float(sd_out), "in_across_input_std": float(sd_in),
            "ratio_out_over_in": float(sd_out / sd_in), "out_rms": float(vis_out.pow(2).mean().sqrt()),
            "out_nan": int(vis_out.isnan().sum()), "out_inf": int(vis_out.isinf().sum()),
            "full_out_across_input_std": float(out.std(dim=0, unbiased=False).mean())}, vis_out


def module_stats(m):
    return {n: tstats(p) for n, p in m.named_parameters()}


def weight_delta(m_a, m_b):
    out = {}
    pa = dict(m_a.named_parameters())
    for n, pb in m_b.named_parameters():
        d = (pb.detach().float() - pa[n].detach().float())
        na = float(pa[n].detach().float().norm())
        nd = float(d.norm())
        out[n] = {"delta_norm": nd, "base_norm": na, "rel_change": nd / (na + 1e-30)}
    return out


def analyze(cfg, smoke=False):
    import torch
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(8)
    sys.path.insert(0, cfg["paths"]["code_dir"])
    ckd = pathlib.Path(cfg["paths"]["ckpt_dir"])
    figs = here / "figures"
    figs.mkdir(exist_ok=True)
    t0 = time.time()
    ck_list = all_ckpts(cfg)
    if smoke:
        ck_list = [("control", "exp003_mix_adjacent_s1", 2)]
    n_limit = cfg["smoke"]["n_frames"] if smoke else None

    # ---- 입력 집합: common (ma_s0 의 frames_sample 을 모든 체크포인트에), own (각 run 의 frames_sample)
    sets = {}
    u8, labels = load_set(cfg["paths"]["common_npz"], cfg, n_limit)
    x, z = encode_frames(cfg, u8)
    sets["common"] = {"labels": labels, "x": x, "z": z, "u8": u8}
    if not smoke:
        for run in sorted({r for _, r, _ in ck_list}):
            u8r, lab = load_set(ckd / run / "frames_sample.npz", cfg)
            xr, zr = encode_frames(cfg, u8r)
            sets[f"own:{run}"] = {"labels": lab, "x": xr, "z": zr, "u8": u8r}
    log("inputs encoded", {k: len(v["labels"]) for k, v in sets.items()}, f"{time.time() - t0:.0f}s")

    res = {"A_code_facts": CODE_FACTS, "run_train_env": {}, "ckpts": {}}
    for run in sorted({r for _, r, _ in ck_list}):
        p = ckd / run / "train_env.json"
        if p.exists():
            te = json.loads(p.read_text())
            res["run_train_env"][run] = {k: te.get(k) for k in ("n_attempts", "n_resumes", "gpus_used", "finished_ok")}
            res["run_train_env"][run]["attempts"] = [{k: a.get(k) for k in ("start_time", "end_time", "action", "epoch_checkpoint_at_start", "gpu", "patched_train_py", "returncode")} for a in te.get("attempts", [])]
    slims = {}
    for grp, run, e in ck_list:
        key = f"{run}/model_{e}"
        sl = torch.load(slim_path(cfg, run, e), map_location="cpu")
        slims[key] = sl
        r = {"group": grp, "run": run, "epoch": e, "drive_meta": sl["meta"], "ckpt_keys": sl["ckpt_keys"],
             "nonfinite_in_checkpoint": sl["nonfinite_in_checkpoint"]}
        # B: 가중치
        r["B_weights"] = {k: module_stats(sl[k]) for k in ("decoder", "predictor", "action_encoder", "proprio_encoder")}
        # C: 옵티마이저
        r["C_optimizer"] = {k: sl[k] for k in ("decoder_optimizer", "predictor_optimizer", "action_encoder_optimizer") if k in sl}
        # D, E: 두 입력 집합
        r["D"], r["E"] = {}, {}
        use_sets = ["common"] if smoke else ["common", f"own:{run}"]
        for sname in use_sets:
            S = sets[sname]
            recs, img = run_decoder(sl, S["z"])
            mse = float(((img - S["x"]) ** 2).mean())
            per_frame_mse = ((img - S["x"]) ** 2).mean(dim=(1, 2, 3))
            r["D"][sname] = {"n_frames": len(S["labels"]),
                             "layers": recs,
                             "output_across_input_std": float(img.std(dim=0, unbiased=False).mean()),
                             "output_nan": int(img.isnan().sum()), "output_inf": int(img.isinf().sum()),
                             "mse_vs_real_in_[-1,1]": mse,
                             "mse_per_frame_min": float(per_frame_mse.min()), "mse_per_frame_max": float(per_frame_mse.max()),
                             "real_across_input_std": float(S["x"].std(dim=0, unbiased=False).mean())}
            pr, _ = run_predictor(cfg, sl, S["z"])
            r["E"][sname] = pr
            if sname == "common":
                S.setdefault("imgs", {})[key] = img
        res["ckpts"][key] = r
        log("done", key, f"{time.time() - t0:.0f}s")

    # B: epoch 간 변화량
    res["B_delta"] = {}
    for grp, run, e in ck_list:
        k_a, k_b = f"{run}/model_{e}", f"{run}/model_{e + 1}"
        if k_b in slims:
            res["B_delta"][f"{run}/{e}->{e + 1}"] = {m: weight_delta(slims[k_a][m], slims[k_b][m])
                                                     for m in ("decoder", "predictor", "action_encoder", "proprio_encoder")}

    # 이미지 시트
    if not smoke:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        S = sets["common"]
        labs = S["labels"]
        pick = [i for i, l in enumerate(labs) if l.split("/")[1] == "ep1" and l.split("/")[2] in ("t0", "t12")]
        sizes = sorted({labs[i].split("/")[0] for i in pick})
        sizes = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]
        pick = [i for i in pick if labs[i].split("/")[0] in sizes]
        for run in sorted({r for _, r, _ in ck_list}):
            eps = sorted(e for _, r, e in ck_list if r == run)
            fig, axs = plt.subplots(1 + len(eps), len(pick), figsize=(1.8 * len(pick), 1.8 * (1 + len(eps))))
            for c, i in enumerate(pick):
                axs[0, c].imshow(((S["x"][i].permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy())
                axs[0, c].set_title(labs[i].replace("/ep1", ""), fontsize=6)
                for rr, e in enumerate(eps, start=1):
                    im = S["imgs"][f"{run}/model_{e}"][i]
                    axs[rr, c].imshow(((im.permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy())
            for rr, name in enumerate(["real"] + [f"epoch {e}" for e in eps]):
                axs[rr, 0].set_ylabel(name, fontsize=7)
            for a in axs.ravel():
                a.set_xticks([]); a.set_yticks([])
            fig.suptitle(f"{run} (decoder(DINO z of real frame), common input set)", fontsize=7)
            fig.tight_layout()
            fig.savefig(figs / f"recon_sheet_{run}.png", dpi=110)
            plt.close(fig)

    # 판단 기준
    res["criteria"] = {} if smoke else judge(cfg, res)
    res["meta"] = {"date_kst": datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"), "smoke": smoke,
                   "elapsed_s": round(time.time() - t0, 1), "torch": torch.__version__, "device": "cpu",
                   "input_sets": {k: {"n_frames": len(v["labels"]), "labels": v["labels"]} for k, v in sets.items()},
                   "seed": cfg["seed"]}
    out = here / ("smoke_results.json" if smoke else "results.json")
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote", out)


def judge(cfg, res):
    """사전 기록된 판단 기준 (notes/hypotheses.md '007 진단 질문')을 기계적으로 적용한다."""
    C = res["ckpts"]
    ratio_thr = cfg["criteria"]["dependence_lost_ratio"]
    chg_thr = cfg["criteria"]["weight_change_ratio"]
    controls = list(cfg["runs"]["control"].keys())
    out = {}
    # (1) NaN/Inf
    c1 = {}
    for key, r in C.items():
        n = sum(r["nonfinite_in_checkpoint"].values())
        for w in r["B_weights"].values():
            n += sum(v.get("nan", 0) + v.get("inf", 0) for v in w.values())
        act = 0
        for sname, d in r["D"].items():
            act += d["output_nan"] + d["output_inf"] + sum(l["nan"] + l["inf"] + l.get("input_nan", 0) + l.get("input_inf", 0) for l in d["layers"])
        for sname, d in r["E"].items():
            act += d["out_nan"] + d["out_inf"]
        c1[key] = {"group": r["group"], "nonfinite_checkpoint_and_weights": n, "nonfinite_activations_outputs": act}
    out["1_nan_inf"] = {"per_ckpt": c1, "applies": any(v["nonfinite_checkpoint_and_weights"] + v["nonfinite_activations_outputs"] > 0
                                                       for v in c1.values() if v["group"] == "collapsed"),
                        "control_any_nonfinite": any(v["nonfinite_checkpoint_and_weights"] + v["nonfinite_activations_outputs"] > 0
                                                     for v in c1.values() if v["group"] == "control")}

    def layers(key, sname):
        return {l["name"]: l for l in C[key]["D"][sname]["layers"]}

    # (2) 입력 의존성 소실 층, (3) 전부 비활성 ReLU
    c2, c3 = {}, {}
    for run, eps in cfg["runs"]["collapsed"].items():
        for e in eps:
            key = f"{run}/model_{e}"
            for sname in C[key]["D"]:
                ref_e = e if e in cfg["runs"]["control"][controls[0]] else max(cfg["runs"]["control"][controls[0]])
                ctrl = [layers(f"{c}/model_{ref_e}", sname if sname == "common" else f"own:{c}") for c in controls]
                cur = layers(key, sname)
                order = sorted(cur.values(), key=lambda l: l["order"])
                rows, first = [], None
                for l in order:
                    cm = float(np.mean([c[l["name"]]["across_input_std"] for c in ctrl]))
                    ratio = l["across_input_std"] / cm if cm > 0 else float("nan")
                    lost = bool(ratio < ratio_thr)
                    if lost and first is None:
                        first = l["name"]
                    rows.append({"name": l["name"], "type": l["type"], "std": l["across_input_std"], "control_mean_std": cm,
                                 "ratio": ratio, "lost": lost})
                c2[f"{key}|{sname}"] = {"control_epoch_used": ref_e, "first_lost_layer": first,
                                        "n_layers_lost": sum(r["lost"] for r in rows), "layers": rows}
                relu_rows = []
                for l in order:
                    if l["type"] == "ReLU":
                        cvals = [c[l["name"]]["frac_input_le0"] for c in ctrl]
                        relu_rows.append({"name": l["name"], "frac_input_le0": l["frac_input_le0"],
                                          "dead_channel_frac": l["dead_channel_frac"], "control_frac_input_le0": cvals,
                                          "all_inactive": bool(l["frac_input_le0"] >= 1.0)})
                c3[f"{key}|{sname}"] = {"all_inactive_layers": [r["name"] for r in relu_rows if r["all_inactive"]], "relu_layers": relu_rows}
    out["2_dependence_lost"] = {"per_ckpt": c2,
                                "applies": {k: v["first_lost_layer"] for k, v in c2.items()}}
    out["3_all_inactive_relu"] = {"per_ckpt": c3, "applies": {k: v["all_inactive_layers"] for k, v in c3.items()}}

    # (4) epoch 3 가중치 변화 10배 이상 (decoder 층별). 대조 = 3개 run 전체 최댓값, 같은 분포 짝도 병기.
    c4 = {}
    for run in cfg["runs"]["collapsed"]:
        for mod in ("decoder", "predictor", "action_encoder", "proprio_encoder"):
            dk = f"{run}/2->3"
            if dk not in res["B_delta"]:
                continue
            rows = []
            for pn, v in res["B_delta"][dk][mod].items():
                ctrl_all = [res["B_delta"][f"{c}/2->3"][mod][pn]["rel_change"] for c in controls]
                ctrl_same = [res["B_delta"][f"{c}/2->3"][mod][pn]["rel_change"] for c in cfg["runs"]["control_same_distribution"][run]]
                rows.append({"param": pn, "rel_change": v["rel_change"], "control_rel_changes": ctrl_all,
                             "ratio_vs_max_control": v["rel_change"] / (max(ctrl_all) + 1e-30),
                             "ratio_vs_max_same_dist_control": v["rel_change"] / (max(ctrl_same) + 1e-30),
                             "flag_all_controls": bool(v["rel_change"] >= chg_thr * max(ctrl_all)),
                             "flag_same_dist_controls": bool(v["rel_change"] >= chg_thr * max(ctrl_same))})
            c4[f"{run}|{mod}"] = {"flagged_all_controls": [r["param"] for r in rows if r["flag_all_controls"]],
                                  "flagged_same_dist": [r["param"] for r in rows if r["flag_same_dist_controls"]],
                                  "max_ratio_vs_max_control": max(r["ratio_vs_max_control"] for r in rows),
                                  "rows": rows}
    out["4_weight_change_10x"] = {"per_run_module": c4,
                                  "applies": {k: v["flagged_all_controls"] for k, v in c4.items() if k.endswith("|decoder")}}
    return out


def summarize(cfg):
    """results.json 의 criteria 에서 판단 기준 표를 만들어 `criteria_table` 로 추가한다 (새 측정 없음)."""
    p = here / "results.json"
    d = json.loads(p.read_text())
    c, C = d["criteria"], d["ckpts"]
    tab = {}
    nf = c["1_nan_inf"]["per_ckpt"]
    tab["1_nan_inf"] = {"applies": bool(c["1_nan_inf"]["applies"]),
                        "collapsed_nonfinite_counts": {k: v["nonfinite_checkpoint_and_weights"] + v["nonfinite_activations_outputs"] for k, v in nf.items() if v["group"] == "collapsed"},
                        "control_nonfinite_counts": {k: v["nonfinite_checkpoint_and_weights"] + v["nonfinite_activations_outputs"] for k, v in nf.items() if v["group"] == "control"}}
    t2 = {}
    for k, v in c["2_dependence_lost"]["per_ckpt"].items():
        if not k.endswith("|common"):
            continue
        row = {l["name"]: l for l in v["layers"]}
        fl = v["first_lost_layer"]
        t2[k.split("|")[0]] = {"applies": fl is not None, "first_lost_layer": fl, "n_layers_lost": v["n_layers_lost"],
                               "std_at_first_lost": row[fl]["std"] if fl else None, "control_mean_std_at_first_lost": row[fl]["control_mean_std"] if fl else None,
                               "control_epoch_used": v["control_epoch_used"]}
    tab["2_dependence_lost"] = t2
    t3 = {}
    for k, v in c["3_all_inactive_relu"]["per_ckpt"].items():
        if not k.endswith("|common"):
            continue
        key = k.split("|")[0]
        L = {l["name"]: l for l in C[key]["D"]["common"]["layers"]}
        t = {"applies_strict_all_inputs_all_positions_le0": bool(v["all_inactive_layers"]), "all_inactive_layers": v["all_inactive_layers"]}
        fl = c["2_dependence_lost"]["per_ckpt"][k]["first_lost_layer"]
        if fl and fl in L and "frac_input_le0" in L[fl]:
            r = [x for x in v["relu_layers"] if x["name"] == fl][0]
            t["first_lost_layer_relu"] = {"name": fl, "frac_input_le0": L[fl]["frac_input_le0"], "dead_channel_frac": L[fl]["dead_channel_frac"], "control_frac_input_le0": r["control_frac_input_le0"]}
        t["max_frac_input_le0_over_relu_layers"] = max(x["frac_input_le0"] for x in v["relu_layers"])
        t3[key] = t
    tab["3_all_inactive_relu"] = t3
    t4 = {}
    for run in cfg["runs"]["collapsed"]:
        v = c["4_weight_change_10x"]["per_run_module"][f"{run}|decoder"]
        top = max(v["rows"], key=lambda r: r["ratio_vs_max_control"])
        t4[run] = {"applies": bool(v["flagged_all_controls"]), "flagged_layers_all_controls": v["flagged_all_controls"], "flagged_layers_same_dist_controls": v["flagged_same_dist"],
                   "max_ratio_param": top["param"], "max_ratio": top["ratio_vs_max_control"], "rel_change_at_max": top["rel_change"], "control_rel_changes_at_max": top["control_rel_changes"],
                   "other_modules_max_ratio": {m: c["4_weight_change_10x"]["per_run_module"][f"{run}|{m}"]["max_ratio_vs_max_control"] for m in ("predictor", "action_encoder", "proprio_encoder")}}
    tab["4_weight_change_10x"] = t4
    any_hit = bool(tab["1_nan_inf"]["applies"] or any(x["applies"] for x in t2.values()) or any(x["applies_strict_all_inputs_all_positions_le0"] for x in t3.values()) or any(x["applies"] for x in t4.values()))
    tab["5_none"] = {"applies": not any_hit}
    tab["not_distinguishable_from_checkpoints"] = ["epoch 3 도중 정확한 시점 (체크포인트로 구분 불가)", "단일 큰 gradient 여부 (체크포인트로 구분 불가)"]
    d["criteria_table"] = tab
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    log("criteria_table written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "analyze", "summarize"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.stage == "summarize":
        summarize(cfg)
    elif a.stage == "fetch":
        fetch(cfg)
    else:
        analyze(cfg, a.smoke)
