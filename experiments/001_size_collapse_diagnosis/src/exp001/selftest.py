"""CPU-only self test with synthetic data (no torch, no GPU, no dataset). Checks manifest logic, mask measurement, probe protocols,
diagnose tables and result writing. The synthetic numbers mean nothing; they only show that the code paths run."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np

from .common import write_json, log
from . import data as D, probes as P, probe_stage, diagnose
from .masks import cloth_area


def synth_cache(cache, rng, cfg, K=13, Dp=8, n_per=10, train_groups=("ind",)):
    from .common import size_roles
    sizes = [(x["name"], x["area_ratio"], x["particles"]) for x in cfg["_ds"]["sizes"]]
    eps, scale = [], []
    for g, ar, pt in sizes:
        for i in range(n_per):
            eps.append({"dir": "x", "split": f"{g}_test", "name": f"episode_{i:05d}.mp4", "group": g, "rank_in_group": i + 1, "in_train": False,
                        "particles": (pt[0] * pt[1]) if len(pt) == 2 else None, "role": None})
            scale.append(ar * np.exp(rng.randn() * 0.03))
    E = len(eps)
    scale = np.array(scale)
    prof = 0.7 + 0.3 * np.sin(np.linspace(0, 2, K))
    area = (2700 * scale[:, None] * prof[None]).astype(np.int64)
    ly = np.log(area)
    W = rng.randn(384)
    W /= np.linalg.norm(W)

    def feat(l, shrink):
        return (rng.randn(E, l.shape[1], 384) * 0.5 + (shrink * (l - 7.9))[..., None] * W * 3).astype(np.float32)

    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache / "scalars.npz", area_real=area, area_recon=area, area_tf1=(area[:, 1:] * 0.9).astype(int),
             area_ol=(area[:, 1:] * 0.5 * scale[:, None] ** -0.5).astype(int), cov_real=np.zeros((E, K, 196), np.float16))
    np.savez(cache / "pooled_mean.npz", gt=feat(ly, 1.0), tf1=feat(ly[:, 1:], 0.5), ol=feat(ly[:, 1:], 0.2), h0=feat(ly[:, :-1], 1.0),
             h3=feat(ly[:, :-1], 0.6), rvit=rng.randn(E, K, 384).astype(np.float32))
    np.savez(cache / "pix_grid.npz", pix=(rng.randn(E, K, 588) + ly[..., None] * 0.1).astype(np.float32))
    for n, k in (("gt", K), ("tf1", K - 1), ("ol", K - 1)):
        np.save(cache / f"patch_{n}.npy", rng.randn(E, k, 196, Dp).astype(np.float16))
    roles = size_roles(cfg, list(train_groups))
    for e in eps:
        e["role"] = roles[e["group"]]
    write_json({"run_name": "synth", "epoch_loaded": 0, "episodes": eps, "n_episodes": E, "K": K, "pools": ["mean"], "code_commit": "synthetic",
                "dataset": cfg["data"]["active"], "train_groups": list(train_groups), "roles": roles,
                "reference_group": cfg["diagnose"]["reference_group"]}, cache / "episodes.json")
    (cache / "DONE").write_text("ok\n")


def _manifest_checks(cfg):
    blk = cfg["_ds"]
    if blk["kind"] == "cloth_sim":
        listing = {f"{s['name']}_{p}": [f"episode_{i:05d}.mp4" for i in range(2000 if p == "train" else 100)] for s in blk["sizes"] for p in ("train", "test")}
    else:
        listing = {"ind_train": [f"t{i}.mp4" for i in range(2000)], "ind_test": [f"i{i}.mp4" for i in range(100)],
                   "ood_test": [f"smaller_{i:03d}.mp4" for i in range(50)] + [f"larger_{i:03d}.mp4" for i in range(50)]}
    man = D.build_manifests(cfg, listing, "/data/root")
    for name, m in man.items():
        assert m["n_episodes"] == cfg["train"]["total_episodes"], (name, m["n_episodes"])
    tr_keys = {(e["split"], e["name"]) for m in man.values() for e in m["episodes"]}
    tab = D.eval_episode_table(listing, cfg, "/data/root")
    assert tab and not any(r["split"].endswith("_train") for r in tab)
    if blk["kind"] == "cloth_sim":
        assert len(tab) == 900 and not any((r["split"], r["name"]) in tr_keys for r in tab), "train/eval episodes overlap"
        assert man["baseline_single"]["group_counts"] == {"ind": 2000}
        assert man["mix_adjacent"]["group_counts"] == {"a080": 667, "ind": 666, "a125": 667}
        assert man["mix_wide"]["group_counts"] == {"a040": 667, "ind": 666, "a160": 667}
        from .common import size_roles
        r1, r2 = size_roles(cfg, ["ind"]), size_roles(cfg, list(man["mix_wide"]["group_counts"]))
        assert r1["a026"] == "extrap_small" and r1["a248"] == "extrap_large" and "interp" not in r1.values()
        assert r2["a060"] == r2["a080"] == r2["a125"] == "interp" and r2["a026"] == "extrap_small" and r2["a200"] == "extrap_large"
        assert size_roles(cfg, ["a080", "ind", "a125"])["a060"] == "extrap_small"
    else:
        assert man["mix_ood3"]["group_counts"] == {"ind": 1950, "smaller": 25, "larger": 25}
    return man


def run(cfg):
    import os
    from .common import resolve_dataset
    rng = np.random.RandomState(0)
    # 1. manifest logic with fake listings: selected dataset and the legacy dataset
    _manifest_checks(cfg)
    other = "acwm_phys" if cfg["data"]["active"] == "cloth_sim" else "cloth_sim"
    c_other = copy.deepcopy(cfg)
    os.environ["EXP001_DATASET"] = other
    try:
        resolve_dataset(c_other)
        _manifest_checks(c_other)
    finally:
        os.environ.pop("EXP001_DATASET", None)
    log("selftest: manifest logic OK (cloth_sim and legacy acwm_phys; role classification; train/eval disjoint)")
    # 2. mask measurement on a synthetic image (orange patch)
    try:
        import cv2
    except ImportError:
        cv2 = None
        log("selftest: cv2 not installed, mask check skipped")
    if cv2 is not None:
        hsv = np.zeros((224, 224, 3), np.uint8)
        hsv[50:100, 50:100] = (12, 200, 200)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        assert cloth_area(img, cfg["mask"]) == 2500, cloth_area(img, cfg["mask"])
        log("selftest: mask OK")
    from . import maskcheck
    g = np.array(["a", "a", "b", "b"])
    t = maskcheck.group_area_table(g, np.array([100., 100., 200., 200.]), {"a": 1.0, "b": 2.0}, "a")
    assert abs(t["b"]["ratio_to_ref_mean"] - 2.0) < 1e-9
    log("selftest: maskcheck table OK")
    # 3. probes + diagnose on synthetic cache (9 sizes; baseline trained on 'ind' only, then a mixed training set with interpolation)
    c2 = copy.deepcopy(cfg)
    c2["diagnose"]["pools"] = ["mean"]
    c2["probe"].update(seeds=[0], n_bootstrap=30, protocols=["strat_cv", "logo", "gt_transfer", "upper", "heldout_size"])
    c2["probe"]["models"]["mlp"]["max_iter"] = 30
    c2["diagnose"]["rollout_ks"] = [1, 4, 12]
    tmp = Path(tempfile.mkdtemp(prefix="exp001_selftest_"))
    paths = {"cache": tmp / "cache", "results_dir": tmp / "results"}
    summ = {}
    for tag, tg in (("synth", ("ind",)), ("synth_wide", ("a040", "ind", "a160"))):
        synth_cache(paths["cache"] / tag, rng, c2, train_groups=tg)
        out = probe_stage.run(c2, paths, tag)
        r = out["pools"]["mean"]["ridge"]
        assert out["group_roles"]["a060"] == ("extrap_small" if tag == "synth" else "interp")
        for proto in ("strat_cv", "logo", "gt_transfer", "upper", "heldout_size"):
            for ty in ("gt", "tf1", "ol"):
                if ty in r[proto]:
                    a = r[proto][ty]["all"]
                    summ[f"{tag}/{proto}/{ty}"] = {m: a[m]["mean"] for m in ("r2_overall", "r2_within_group", "acc_cls", "acc_cls_pm1", "slope")}
        assert any(k.startswith("role_") for k in r["heldout_size"]["gt"]), "heldout_size role summaries missing"
    diag = diagnose.run(c2, paths, "synth_wide")
    diagnose.run(c2, paths, "synth", out_name="repro_check.json", area_only=True)
    assert set(diag["size_curve"]["by_group"]) == {s["name"] for s in c2["_ds"]["sizes"]}
    assert diag["repro_check"]["groups"]["a248"]["n_episodes"] == 10 and "label_check_particles_vs_mask_area" in diag
    log("selftest: probe + diagnose + repro_check OK (synthetic numbers, not results)")
    for k, v in summ.items():
        log(f"  {k}: " + ", ".join(f"{m}={'nan' if x is None else round(x, 3)}" for m, x in v.items()))
    _selftest_sync_hf(cfg)
    _selftest_drive(cfg)
    return {"ok": True, "tmp": str(tmp)}


def _selftest_sync_hf(cfg):
    """sync: secret scan, dry-run, skip-when-unconfigured, local-mount copy+verification (into a temp dir, nothing leaves the machine);
    hf: error explanation. No rclone, no network."""
    import os
    from . import sync, hf
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        exp, mount = td / "exp", td / "drive"
        (exp / "results" / "r1").mkdir(parents=True)
        mount.mkdir()
        (exp / "results.json").write_text("{}")
        (exp / "config.yaml").write_text("seed: 0\n")
        (exp / "results" / "r1" / "diagnose.json").write_text("{}")
        (exp / "results" / "r1" / "leak.txt").write_text("token = hf_" + "A" * 30)
        (exp / "results" / "r1" / "mail.txt").write_text("someone@example.com")
        paths = {"exp": exp, "results_dir": exp / "results", "acwm_src": td / "none", "train_base": td / "tr", "cache": td / "cache"}
        c = copy.deepcopy(cfg)
        c["sync"] = {"drive": {"rclone_remote": "", "remote_path": "", "local_mount": ""}}
        for k in ("EXP001_RCLONE_REMOTE", "EXP001_REMOTE_PATH", "EXP001_DRIVE_MOUNT"):
            os.environ.pop(k, None)
        files, skipped, _ = sync.collect(exp, ["results.json", "results", "config.yaml"])
        assert sorted(r for r, _ in files) == ["config.yaml", "results.json", "results/r1/diagnose.json"], files
        assert {r for r, _ in skipped} == {"results/r1/leak.txt", "results/r1/mail.txt"}, skipped
        assert sync.run(c, paths, dry_run=True)["status"] == "dry_run"
        assert not (exp / "results" / "sync_status.json").exists()
        assert sync.run(c, paths)["status"] == "skipped" and (exp / "results" / "sync_status.json").exists()
        c["sync"]["drive"].update({"local_mount": str(mount), "remote_path": "x"})
        st = sync.run(c, paths)
        assert st["status"] == "ok" and st["mode"] == "mount", st
        out = next((mount / "x").iterdir())
        assert (out / "results" / "run_meta.json").exists() and (out / "results" / "sync_status.json").exists()
        assert not (out / "results" / "r1" / "leak.txt").exists()
        assert st["skipped_secret_like"]
    class _E(Exception):
        response = type("R", (), {"status_code": 401})()
    assert "HF_TOKEN" in hf.explain_error(_E(), "o/r") and hf.explain_error(ValueError("x"), "o/r") is None
    log("selftest: sync (scan, dry-run, skip, local-mount copy+verify) and hf error text OK")


FAKE_RCLONE = r"""#!__PY__
# fake rclone for the selftest: logs argv, fakes copy/lsf/size/listremotes. FAKE_N = episodes per split, FAKE_FAIL=quota -> exit 1 with a 403 text.
import json, os, sys
a = sys.argv[1:]
open(os.environ["FAKE_LOG"], "a").write(json.dumps(a) + "\n")
n = int(os.environ.get("FAKE_N", "3"))
if a[0] == "listremotes":
    print("gdrive:")
elif a[0] == "copy":
    if os.environ.get("FAKE_FAIL") == "quota":
        print("ERROR : Failed to copy: googleapi: Error 403: rate limit exceeded", file=sys.stderr); sys.exit(1)
    dest = a[2]; os.makedirs(dest, exist_ok=True)
    for i in range(n):
        open(os.path.join(dest, "episode_%05d.mp4" % i), "w").write("x")
    open(os.path.join(dest, "metadata.pt"), "w").write(str(n))
elif a[0] == "lsf":
    for i in range(n): print("episode_%05d.mp4" % i)
    print("metadata.pt")
elif a[0] == "size":
    print(json.dumps({"count": n + 1, "bytes": 1000 * (n + 1)}))
"""


def _selftest_drive(cfg):
    """drive download with a FAKE rclone executable: command construction for both access modes, split selection, skip/resume logic,
    quota-pattern detection, download_check.json. No network, no real rclone. Fake mp4/metadata files are text; counters are injected."""
    import copy
    import json
    import os
    import stat
    import sys
    from . import drive
    c = copy.deepcopy(cfg)
    if c["data"]["active"] != "cloth_sim":
        return
    for k in ("EXP001_RCLONE_REMOTE", "EXP001_DATA_REMOTE_PATH", "EXP001_RCLONE_BIN", "EXP001_RCLONE_FLAGS", "EXP001_DATA_SOURCE"):
        os.environ.pop(k, None)
    c["data"]["drive"].update({"rclone_remote": "gdrive", "remote_path": "", "root_folder_id": "FOLDERID"})
    c["sync"] = {"drive": {"rclone_remote": "", "remote_path": "", "local_mount": ""}}
    # split selection
    sp = drive.select_splits(c, [])
    assert sp[0] == "ind_train" and len(sp) == 10 and all(x.endswith("_test") for x in sp[1:]), sp
    assert "a040_train" in drive.select_splits(c, ["mix_wide"]) and drive.select_splits(c, ["mix_wide"]).count("ind_train") == 1
    assert drive.expected_n(c, "ind_train") == 2000 and drive.expected_n(c, "a026_test") == 100
    # command strings
    d = drive.drive_cfg(c)
    assert drive.rclone_source(d, "ind_test") == ("gdrive,root_folder_id=FOLDERID:ind_test", [], "root_folder_id")
    d2 = drive.drive_cfg({**c, "data": {**c["data"], "drive": {**c["data"]["drive"], "root_folder_style": "flag"}}})
    assert drive.rclone_source(d2, "ind_test") == ("gdrive:ind_test", ["--drive-root-folder-id", "FOLDERID"], "root_folder_id")
    d3 = drive.drive_cfg({**c, "data": {**c["data"], "drive": {**c["data"]["drive"], "remote_path": "/my/copy/"}}})
    assert drive.rclone_source(d3, "ind_test") == ("gdrive:my/copy/ind_test", [], "remote_path")
    # sync.drive.rclone_remote is shared when data.drive.rclone_remote is empty
    c2 = copy.deepcopy(c)
    c2["data"]["drive"]["rclone_remote"] = ""
    c2["sync"]["drive"]["rclone_remote"] = "shared"
    assert drive.drive_cfg(c2)["rclone_remote"] == "shared"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake = td / "fake_rclone"
        fake.write_text(FAKE_RCLONE.replace("__PY__", sys.executable))
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        logf = td / "calls.jsonl"
        os.environ.update({"FAKE_LOG": str(logf), "FAKE_N": "3"})
        c["data"]["drive"]["rclone_bin"] = str(fake)
        c["_ds"]["expected_episodes"] = {"train": 3, "test": 3}
        paths = {"data_root": td / "data" / "cloth_sim", "results_dir": td / "results"}
        mc = lambda p: int(Path(p).read_text())                                                     # noqa: E731
        vp = lambda p: {"file": Path(p).name, "width": 512, "height": 512, "frames": 65, "error": None}   # noqa: E731
        calls = lambda: [json.loads(x) for x in logf.read_text().splitlines()] if logf.exists() else []  # noqa: E731
        two = ["ind_train", "ind_test"]
        # dry-run: lsf+size only, nothing written
        assert drive.download(c, paths, two, dry_run=True, meta_counter=mc, video_prober=vp) == 0
        assert {x[0] for x in calls()} <= {"lsf", "size"} and not paths["data_root"].exists()
        logf.unlink()
        # real (fake) download
        assert drive.download(c, paths, two, meta_counter=mc, video_prober=vp) == 0
        cp = [x for x in calls() if x[0] == "copy"]
        assert [x[1] for x in cp] == ["gdrive,root_folder_id=FOLDERID:ind_train", "gdrive,root_folder_id=FOLDERID:ind_test"], cp
        assert cp[0][2] == str(paths["data_root"] / "ind_train")
        chk = json.loads((td / "results" / "download_check.json").read_text())
        assert chk["warnings"] == [] and chk["splits"]["ind_test"]["n_mp4"] == 3, chk
        # second run: complete splits are skipped (no rclone copy at all)
        logf.unlink()
        assert drive.download(c, paths, two, meta_counter=mc, video_prober=vp) == 0
        assert not [x for x in calls() if x[0] == "copy"]
        # resume: a deleted episode file -> marker no longer matches -> split is fetched again, the other stays skipped
        (paths["data_root"] / "ind_test" / "episode_00001.mp4").unlink()
        assert drive.download(c, paths, two, meta_counter=mc, video_prober=vp) == 0
        cp = [x for x in calls() if x[0] == "copy"]
        assert [x[1].split(":")[-1] for x in cp] == ["ind_test"], cp
        # mismatch -> warning in download_check.json, not an error
        (paths["data_root"] / "ind_test" / "episode_00002.mp4").unlink()
        out = drive.check(c, paths, two, meta_counter=mc, video_prober=vp)
        assert any("ind_test" in w and "mp4 count 2" in w for w in out["warnings"]), out["warnings"]
        # quota / 403 pattern: exit code 3, other splits are still attempted
        os.environ["FAKE_FAIL"] = "quota"
        paths2 = {"data_root": td / "data2", "results_dir": td / "results2"}
        assert drive.download(c, paths2, ["a026_test"], meta_counter=mc, video_prober=vp) == 3
        assert drive.QUOTA_RX.search("Error 403: Rate Limit Exceeded") and drive.QUOTA_RX.search("download quota exceeded")
        assert not drive.QUOTA_RX.search("copied 3 files")
        del os.environ["FAKE_FAIL"]
        # no remote name -> setup help, rc 2, no copy
        c3 = copy.deepcopy(c)
        c3["data"]["drive"]["rclone_remote"] = ""
        logf.unlink()
        assert drive.download(c3, paths2, ["a026_test"], meta_counter=mc, video_prober=vp) == 2 and not calls()
        # remote_path mode builds the (b) command
        c4 = copy.deepcopy(c)
        c4["data"]["drive"]["remote_path"] = "cloth_copy"
        assert drive.download(c4, paths2, ["a026_test"], meta_counter=mc, video_prober=vp) == 0
        assert calls()[-1][:2] == ["copy", "gdrive:cloth_copy/a026_test"] or any(x[1] == "gdrive:cloth_copy/a026_test" for x in calls())
        for k in ("FAKE_LOG", "FAKE_N"):
            os.environ.pop(k, None)
    log("selftest: drive download (fake rclone: commands for both access modes, split selection, skip/resume, quota pattern, download_check) OK")
