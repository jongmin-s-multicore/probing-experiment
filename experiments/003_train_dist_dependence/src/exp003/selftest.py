"""CPU-only self test with synthetic data (no torch, no GPU, no dataset, no network). Checks manifest logic (4 distributions, train/eval
disjoint), metrics (ratio, slope, center c) against synthetic curves with known answers, the H-H1 / H-H2 verdict rules, aggregate output,
sync / drive code paths (fake rclone). The synthetic numbers mean nothing; they only show that the code paths run and the rules do what
they say. NOT a validation of the GPU stages (train / extract), which cannot run on this machine."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np

from .common import write_json, read_json, log, area_ratio_of
from . import data as D, metrics as M, verdict as V, aggregate as AG


def _listing(cfg):
    return {f"{s['name']}_{p}": [f"episode_{i:05d}.mp4" for i in range(2000 if p == "train" else 100)] for s in cfg["_ds"]["sizes"] for p in ("train", "test")}


def _manifest_checks(cfg):
    listing = _listing(cfg)
    man = D.build_manifests(cfg, listing, "/data/root")
    assert set(man) == {"single_a026", "single_a248", "mix_adjacent", "mix_wide"}, list(man)
    for name, m in man.items():
        assert m["n_episodes"] == cfg["train"]["total_episodes"] == 2000, (name, m["n_episodes"])
        assert all(e["split"].endswith("_train") for e in m["episodes"]), name
    assert man["single_a026"]["group_counts"] == {"a026": 2000} and man["single_a248"]["group_counts"] == {"a248": 2000}
    assert man["mix_adjacent"]["group_counts"] == {"a080": 667, "ind": 667, "a125": 666}, man["mix_adjacent"]["group_counts"]
    assert man["mix_wide"]["group_counts"] == {"a040": 667, "ind": 667, "a160": 666}, man["mix_wide"]["group_counts"]
    tr_keys = {(e["split"], e["name"]) for m in man.values() for e in m["episodes"]}
    tab = D.eval_episode_table(listing, cfg, "/data/root")
    assert len(tab) == 900 and not any(r["split"].endswith("_train") for r in tab) and not any((r["split"], r["name"]) in tr_keys for r in tab), "train/eval overlap"
    from .common import size_roles
    assert size_roles(cfg, ["a026"])["a248"] == "extrap_large" and size_roles(cfg, ["a248"])["a026"] == "extrap_small"
    r = size_roles(cfg, ["a080", "ind", "a125"])
    assert r["a060"] == "extrap_small" and r["a160"] == "extrap_large" and "interp" not in r.values()
    r = size_roles(cfg, ["a040", "ind", "a160"])
    assert r["a060"] == r["a080"] == r["a125"] == "interp" and r["a026"] == "extrap_small" and r["a200"] == "extrap_large"


def synth_run(cfg, rng, train_groups, c, gamma, n=100, noise=0.03, ratio_fn=None):
    """Synthetic extract result. Real area = 2400 * a_g * noise. Open-loop area = real * ratio_g with ratio_g = (a_g/c)^(-gamma)
    (crosses 1 exactly at a_g = c; log-log slope of ol on real = 1 - gamma). ratio_fn overrides the curve."""
    from .common import size_roles
    eps, real, ol = [], [], []
    roles = size_roles(cfg, list(train_groups))
    for s in cfg["_ds"]["sizes"]:
        a = s["area_ratio"]
        for i in range(n):
            eps.append({"split": s["name"] + "_test", "name": "episode_%05d.mp4" % i, "group": s["name"], "in_train": False, "role": roles[s["name"]]})
            r = 2400 * a * np.exp(rng.randn() * noise)
            rat = (ratio_fn(a) if ratio_fn else (a / c) ** (-gamma)) * np.exp(rng.randn() * noise)
            real.append(np.full(13, r))
            ol.append(np.full(12, r * rat))
    ep = {"episodes": eps, "roles": roles, "train_groups": sorted(train_groups), "epoch_loaded": 15, "n_episodes": len(eps), "code_commit": "synthetic"}
    return ep, {"area_real": np.array(real).round().astype(np.int64), "area_ol": np.array(ol).round().astype(np.int64)}


def _metrics_checks(cfg, rng):
    ep, sc = synth_run(cfg, rng, ["a026"], c=0.35, gamma=0.5)
    m = M.compute(cfg, ep, sc)
    assert m["center"]["status"] == "ok" and abs(np.log(m["center"]["c"] / 0.35)) < 0.15, m["center"]
    assert abs(m["slope_ol"]["all_sizes"]["slope"] - 0.5) < 0.05, m["slope_ol"]["all_sizes"]
    assert m["slope_ol"]["all_sizes"]["lo"] <= m["slope_ol"]["all_sizes"]["slope"] <= m["slope_ol"]["all_sizes"]["hi"]
    assert m["by_size"]["a026"]["ratio_ol_over_real"]["mean"] > 1.0 > m["by_size"]["a248"]["ratio_ol_over_real"]["mean"]
    assert len(m["by_size"]["ind"]["area_ol_per_episode"]) == 100
    # no crossing: ratio below 1 everywhere / above 1 everywhere / two crossings
    m2 = M.compute(cfg, *synth_run(cfg, rng, ["ind"], c=1.0, gamma=0.0, ratio_fn=lambda a: 0.8))
    assert m2["center"]["c"] is None and "below" in m2["center"]["reason"], m2["center"]
    m3 = M.compute(cfg, *synth_run(cfg, rng, ["ind"], c=1.0, gamma=0.0, ratio_fn=lambda a: 1.3))
    assert m3["center"]["c"] is None and "above" in m3["center"]["reason"]
    m4 = M.compute(cfg, *synth_run(cfg, rng, ["ind"], c=1.0, gamma=0.0, ratio_fn=lambda a: 1.3 if (a < 0.5 or a > 1.7) else 0.7))
    assert m4["center"]["c"] is None and m4["center"]["n_crossings"] == 2 and "ambiguous" in m4["center"]["reason"], m4["center"]
    # find_center interpolation on a toy grid: ratio 2 at size1, 0.5 at size2 -> log crossing halfway in log area
    fc = M.find_center(["x", "y"], [0.0, np.log(4.0)], [2.0, 0.5])
    assert abs(fc["c"] - 2.0) < 1e-9, fc
    # slope subsets
    assert m["slope_ol"]["trained_and_interp_sizes"] == ["a026"] and m["slope_ol"]["trained_and_interp"] is None
    log("selftest: metrics (ratio, slope, center c, null reasons) OK on synthetic curves with known answers")
    return m


def _verdict_checks(cfg, rng):
    tmp = Path(tempfile.mkdtemp(prefix="exp003_selftest_"))
    base_path = tmp / "base" / "results.json"
    cfg = copy.deepcopy(cfg)
    cfg["criteria"]["h2_baseline"]["results_json"] = "base/results.json"
    seeds = cfg["train"]["seeds"]

    def runs_for(a026_kind, a248_kind, mix_gamma):
        runs = {}
        spec = {"center": dict(c=0.95, gamma=0.4)}
        for s in seeds:
            for dist, tg, kind, cc in (("single_a026", ["a026"], a026_kind, 0.35), ("single_a248", ["a248"], a248_kind, 2.2)):
                kw = dict(c=cc, gamma=0.5) if kind == "h1" else dict(c=0.95, gamma=0.4)
                runs["%s:%d" % (dist, s)] = M.compute(cfg, *synth_run(cfg, rng, tg, **kw))
            for dist, tg in (("mix_adjacent", ["a080", "ind", "a125"]), ("mix_wide", ["a040", "ind", "a160"])):
                runs["%s:%d" % (dist, s)] = M.compute(cfg, *synth_run(cfg, rng, tg, c=1.0 + 0.01 * s, gamma=mix_gamma[dist]))
        return runs

    # baseline file missing -> H2 not_evaluable
    v0 = V.build(cfg, runs_for("h1", "h1", {"mix_adjacent": 0.2, "mix_wide": 0.2}), tmp)
    assert v0["H-H2"]["status"] == "not_evaluable" and "not readable" in v0["H-H2"]["distributions"]["mix_wide"]["reason"], v0["H-H2"]
    assert v0["H-H1"]["status"] == "pass", v0["H-H1"]["patterns"]
    # baseline slopes 0.25 (gamma 0.75); mixes with slope 0.8 -> all pairs >= 0.2 -> pass
    base_path.parent.mkdir()
    sl = cfg["criteria"]["h2_baseline"]["slope_path"]

    def put(vals):
        d = {"runs": {}}
        for rn, v in zip(cfg["criteria"]["h2_baseline"]["runs"], vals):
            cur = d["runs"].setdefault(rn, {})
            for p in sl[:-1]:
                cur = cur.setdefault(p, {})
            cur[sl[-1]] = v
        base_path.write_text(json.dumps(d))
    put([0.24, 0.25, 0.26])
    v1 = V.build(cfg, runs_for("h1", "h1", {"mix_adjacent": 0.2, "mix_wide": 0.2}), tmp)
    assert v1["H-H2"]["status"] == "pass", v1["H-H2"]
    v2 = V.build(cfg, runs_for("center", "center", {"mix_adjacent": 0.75, "mix_wide": 0.75}), tmp)
    assert v2["H-H1"]["status"] == "fail" and v2["H-H2"]["status"] == "fail", (v2["H-H1"]["patterns"], v2["H-H2"])
    v3 = V.build(cfg, runs_for("h1", "center", {"mix_adjacent": 0.2, "mix_wide": 0.75}), tmp)
    assert v3["H-H1"]["status"] == "not_evaluable" and v3["H-H2"]["status"] == "not_evaluable", (v3["H-H1"]["patterns"], v3["H-H2"]["status"])
    # missing seeds -> not_evaluable
    r4 = runs_for("h1", "h1", {"mix_adjacent": 0.2, "mix_wide": 0.2})
    del r4["single_a026:2"], r4["mix_wide:1"]
    v4 = V.build(cfg, r4, tmp)
    assert v4["H-H1"]["status"] == "not_evaluable" and v4["H-H2"]["distributions"]["mix_wide"]["status"] == "not_evaluable"
    # the thresholds in the verdict are read from config: changing criteria changes the outcome, and they are recorded
    c5 = copy.deepcopy(cfg)
    c5["criteria"]["h2_slope_gain_min"] = 0.9
    v5 = V.build(c5, runs_for("h1", "h1", {"mix_adjacent": 0.2, "mix_wide": 0.2}), tmp)
    assert v5["H-H2"]["status"] == "fail" and v5["H-H2"]["distributions"]["mix_wide"]["status"] == "fail" and v5["criteria_used"]["criteria.h2_slope_gain_min"] == 0.9
    assert v1["criteria_used"]["criteria.h2_slope_gain_min"] == 0.2
    log("selftest: verdict table (H-H1 pass/fail/mixed, H-H2 pass/fail/mixed, missing baseline, missing seeds, config-driven threshold) OK")
    # aggregate end to end on a temp results dir
    paths = {"results_dir": tmp / "results", "maskcheck_json": tmp / "none.json", "acwm_src": tmp / "none", "train_base": tmp / "tr"}
    for k, m in runs_for("h1", "h1", {"mix_adjacent": 0.2, "mix_wide": 0.2}).items():
        d, s = k.split(":")
        write_json(m, paths["results_dir"] / cfg["train"]["run_name_template"].format(dist=d, seed=s) / "metrics.json")
    res = AG.run(cfg, paths, out_path=tmp / "results.json", base_dir=tmp)
    back = read_json(tmp / "results.json")
    assert len(back["runs"]) == 12 and back["missing_runs"] == [] and back["verdict"]["H-H2"]["status"] == "pass"
    assert back["verdict"]["H-H1"]["models"]["single_a026"]["per_seed"]["0"]["c"] is not None
    log("selftest: aggregate -> results.json (12 runs, verdict) OK")


def run(cfg):
    rng = np.random.RandomState(cfg["seed"])
    _manifest_checks(cfg)
    log("selftest: manifests OK (4 distributions x 2000 episodes, *_train only, counts 667/667/666, 900 eval episodes disjoint from training)")
    try:
        import cv2
        from .masks import cloth_area
        hsv = np.zeros((224, 224, 3), np.uint8)
        hsv[50:100, 50:100] = (12, 200, 200)
        assert cloth_area(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), cfg["mask"]) == 2500
        log("selftest: mask OK")
    except ImportError:
        log("selftest: cv2 not installed, mask check SKIPPED")
    _metrics_checks(cfg, rng)
    _verdict_checks(cfg, rng)
    _selftest_sync_hf(cfg)
    _selftest_drive(cfg)
    _selftest_mount(cfg)
    res = _selftest_resume(cfg)
    log("selftest: resume/guard/baseline group: %s (synthetic data, not results)" % res)
    return {"ok": True}


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
        for k in ("EXP003_RCLONE_REMOTE", "EXP003_REMOTE_PATH", "EXP003_DRIVE_MOUNT"):
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
    for k in ("EXP003_RCLONE_REMOTE", "EXP003_DATA_REMOTE_PATH", "EXP003_RCLONE_BIN", "EXP003_RCLONE_FLAGS", "EXP003_DATA_SOURCE"):
        os.environ.pop(k, None)
    c["data"]["drive"].update({"rclone_remote": "gdrive", "remote_path": "", "root_folder_id": "FOLDERID"})
    c["sync"] = {"drive": {"rclone_remote": "", "remote_path": "", "local_mount": ""}}
    # split selection
    sp = drive.select_splits(c, [])
    assert len(sp) == 9 and all(x.endswith("_test") for x in sp), sp
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


def _selftest_mount(cfg):
    """Colab mount-copy mode with a fake 'mounted Drive' directory (no network, no rclone): candidate paths incl. bracketed names, find_mount_src,
    parallel copy + marker/bytes, skip, resume after a deleted file, size-mismatch recopy, tar and zip archive extraction, archive creation by
    prepare_data, unsafe archive rejection, mode switch. Fake mp4/metadata are text files; counters are injected."""
    import copy
    import os
    import tarfile
    import zipfile
    from . import drive
    c = copy.deepcopy(cfg)
    if c["data"]["active"] != "cloth_sim":
        return
    for k in ("EXP003_DATA_MODE", "EXP003_DRIVE_MOUNT_SRC", "EXP003_DATA_ARCHIVE", "EXP003_WORK_ROOT", "EXP003_MAKE_ARCHIVE", "EXP003_ARCHIVE_DIRECT"):
        os.environ.pop(k, None)
    c["data"]["drive"]["root_folder_id"] = "FOLDERID"
    c["_ds"]["expected_episodes"] = {"train": 3, "test": 3}
    two = ["ind_train", "ind_test"]
    def mc(p):      # counter for the fake text metadata.pt; a real binary (torch.save) file falls back to the real counter
        try:
            return int(Path(p).read_text())
        except (UnicodeDecodeError, ValueError):
            return drive.meta_len(p)
    assert not drive.mount_mode_requested()
    os.environ["EXP003_DATA_MODE"] = "mount"
    assert drive.mount_mode_requested()
    os.environ["EXP003_DATA_MODE"] = "rclone"
    assert not drive.mount_mode_requested()
    os.environ.pop("EXP003_DATA_MODE")
    os.environ["EXP003_DRIVE_MOUNT_SRC"] = "/x"
    assert drive.mount_mode_requested()
    os.environ.pop("EXP003_DRIVE_MOUNT_SRC")
    assert {"a026_train", "a248_test", "ind_train"} <= drive.all_split_names(c) and len(drive.all_split_names(c)) == 18

    def make_src(base):
        for s in two:
            (base / s).mkdir(parents=True)
            for i in range(3):
                (base / s / f"episode_{i:05d}.mp4").write_text("v" * (10 + i))
            (base / s / "metadata.pt").write_text("3")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mount = td / "drive"
        # candidates: default bracketed path, shortcut-target path, explicit env first
        cands = drive.mount_src_candidates(c, str(mount))
        assert cands[0] == mount / "MyDrive/dino-wm-videos/[data]/[cloth-sim] cloth_move" and cands[1] == mount / ".shortcut-targets-by-id" / "FOLDERID", cands
        os.environ["EXP003_DRIVE_MOUNT_SRC"] = str(td / "explicit")
        assert drive.mount_src_candidates(c, str(mount))[0] == td / "explicit"
        os.environ.pop("EXP003_DRIVE_MOUNT_SRC")
        assert drive.find_mount_src(c, two, str(mount)) is None
        # found via default candidate's direct child, via the shortcut path, and via the bounded search below dino-wm-videos
        src = mount / "MyDrive/dino-wm-videos/[data]/[cloth-sim] cloth_move" / "cloth_sim"
        make_src(src)
        assert drive.find_mount_src(c, two, str(mount)) == src
        src2 = td / "drive2"
        make_src(src2 / ".shortcut-targets-by-id" / "FOLDERID")
        assert drive.find_mount_src(c, two, str(src2)) == src2 / ".shortcut-targets-by-id" / "FOLDERID"
        src3 = td / "drive3"
        make_src(src3 / "MyDrive/dino-wm-videos/a/b")
        assert drive.find_mount_src(c, two, str(src3)) == src3 / "MyDrive/dino-wm-videos/a/b"
        # copy
        paths = {"data_root": td / "w" / "data" / "cloth_sim", "dataset_dir": td / "w" / "data", "results_dir": td / "w" / "results"}
        assert drive.copy_from_mount(c, paths, two, src, dry_run=True, meta_counter=mc) == 0 and not paths["data_root"].exists()
        assert drive.copy_from_mount(c, paths, two, src, meta_counter=mc, workers=2) == 0
        for s in two:
            assert (paths["data_root"] / s / drive.OK_MARK).exists() and drive.count_mp4(paths["data_root"] / s) == 3
            assert drive.split_complete(c, paths["data_root"] / s, s, mc)[0] and drive._split_done(c, paths["data_root"] / s, s, mc)[0]
        assert json.loads((paths["data_root"] / "ind_test" / drive.OK_MARK).read_text())["source"] == "drive_mount"
        assert not list(paths["data_root"].rglob("*.part"))
        assert (td / "w" / "results" / "download_check.json").exists()
        # skip when complete (a changed source file is NOT re-read)
        (src / "ind_test" / "episode_00000.mp4").write_text("changed-source-longer")
        assert drive.copy_from_mount(c, paths, two, src, meta_counter=mc) == 0
        assert (paths["data_root"] / "ind_test" / "episode_00000.mp4").read_text() == "v" * 10
        # resume: deleted file refetched; size mismatch refetched (via force)
        (paths["data_root"] / "ind_test" / "episode_00001.mp4").unlink()
        assert drive.copy_from_mount(c, paths, two, src, meta_counter=mc) == 0
        assert (paths["data_root"] / "ind_test" / "episode_00001.mp4").exists()
        (src / "ind_test" / "episode_00000.mp4").write_text("v" * 10)
        # source missing a file -> rc 3, no marker
        (src / "ind_test" / "episode_00002.mp4").rename(td / "moved.mp4")
        (paths["data_root"] / "ind_test" / "episode_00002.mp4").unlink()
        assert drive.copy_from_mount(c, paths, two, src, meta_counter=mc) == 3
        assert not (paths["data_root"] / "ind_test" / drive.OK_MARK).exists()
        (td / "moved.mp4").rename(src / "ind_test" / "episode_00002.mp4")
        assert drive.copy_from_mount(c, paths, two, src, meta_counter=mc) == 0
        # prepare_data: no archive, mount found -> copy, then archive written to EXP003_WORK_ROOT; second fresh data root uses the archive
        os.environ["EXP003_WORK_ROOT"] = str(td / "work")
        assert drive.find_archive(c) is None and drive.archive_target(c) == td / "work" / "cloth_sim_data.tar"
        os.environ["EXP003_DRIVE_MOUNT_SRC"] = str(src)
        pathsA = {"data_root": td / "A" / "data" / "cloth_sim", "dataset_dir": td / "A" / "data", "results_dir": td / "A" / "results"}
        assert drive.prepare_data(c, pathsA, two, meta_counter=mc) == 0
        arc = td / "work" / "cloth_sim_data.tar"
        assert arc.is_file() and drive.find_archive(c) == arc
        with tarfile.open(arc) as tf:
            assert "cloth_sim/ind_train/metadata.pt" in tf.getnames()
        os.environ.pop("EXP003_DRIVE_MOUNT_SRC")
        pathsB = {"data_root": td / "B" / "data" / "cloth_sim", "dataset_dir": td / "B" / "data", "results_dir": td / "B" / "results"}
        assert drive.prepare_data(c, pathsB, two, dry_run=True, meta_counter=mc) == 0 and not pathsB["data_root"].exists()
        assert drive.prepare_data(c, pathsB, two, meta_counter=mc) == 0
        for s in two:
            assert drive.split_complete(c, pathsB["data_root"] / s, s, mc)[0]
        assert json.loads((pathsB["data_root"] / "ind_test" / drive.OK_MARK).read_text())["source"] == "archive"
        assert not (pathsB["dataset_dir"] / "_archive_tmp" / arc.name).exists()
        # zip archive with a different top-level layout (<split>/<file>)
        arc.unlink()
        with zipfile.ZipFile(td / "work" / "cloth_sim_data.zip", "w") as zf:
            for s in two:
                for f in sorted((src / s).iterdir()):
                    zf.write(f, f"{s}/{f.name}")
        assert drive.find_archive(c).name == "cloth_sim_data.zip"
        pathsC = {"data_root": td / "C" / "data" / "cloth_sim", "dataset_dir": td / "C" / "data", "results_dir": td / "C" / "results"}
        assert drive.prepare_data(c, pathsC, two, meta_counter=mc) == 0
        assert drive.count_mp4(pathsC["data_root"] / "ind_train") == 3
        # incomplete archive (only ind_train) -> rc 3, nothing marked for the missing split
        (td / "work" / "cloth_sim_data.zip").unlink()
        with tarfile.open(td / "work" / "cloth_sim_data.tar", "w") as tf:
            for f in sorted((src / "ind_train").iterdir()):
                tf.add(str(f), arcname=f"cloth_sim/ind_train/{f.name}", recursive=False)
        pathsD = {"data_root": td / "D" / "data" / "cloth_sim", "dataset_dir": td / "D" / "data", "results_dir": td / "D" / "results"}
        assert drive.prepare_data(c, pathsD, two, meta_counter=mc) == 3
        # unsafe path in an archive is refused before anything is extracted
        (td / "work" / "cloth_sim_data.tar").unlink()
        with tarfile.open(td / "work" / "cloth_sim_data.tar", "w") as tf:
            ti = tarfile.TarInfo("../evil.txt")
            tf.addfile(ti)
        pathsE = {"data_root": td / "E" / "data" / "cloth_sim", "dataset_dir": td / "E" / "data", "results_dir": td / "E" / "results"}
        assert drive.prepare_data(c, pathsE, two, meta_counter=mc) == 3 and not (td / "E" / "evil.txt").exists()
        # no archive and no mount source -> rc 2
        (td / "work" / "cloth_sim_data.tar").unlink()
        pathsF = {"data_root": td / "F" / "data" / "cloth_sim", "dataset_dir": td / "F" / "data", "results_dir": td / "F" / "results"}
        assert drive.prepare_data(c, pathsF, two, meta_counter=mc, mount=str(td / "empty_mount")) == 2     # never look at a real /content/drive
        os.environ.pop("EXP003_WORK_ROOT")
    log("selftest: mount-copy mode (fake mounted Drive: candidate paths incl. brackets, find_mount_src, parallel copy, marker/bytes, skip/resume, tar/zip archive, archive creation, unsafe path, rc 2/3) OK")


# ------------------------------------------------------------------------------------------------ resume / guard / baseline (2026-11-05)
def _tree(root):
    import hashlib
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


FAKE_TRAIN = '''import os, sys
# [exp002 operational] range(init_epoch, self.total_epochs + 1)
a = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
rd, total, save = a["hydra.run.dir"], int(a["training.epochs"]), int(a["training.save_every_x_epoch"])
ck = os.path.join(rd, "checkpoints"); os.makedirs(ck, exist_ok=True)
lat = os.path.join(ck, "model_latest.pth")
ep = int(open(lat).read().split()[1]) if os.path.exists(lat) else 0
stop = int(os.environ.get("FAKE_STOP_AFTER", "0"))
for e in range(ep + 1, total + 1):
    open(os.path.join(rd, "fake_trained.txt"), "a").write("%d\\n" % e)
    if e % save == 0:
        for n in ("model_%d.pth" % e, "model_latest.pth"):
            open(os.path.join(ck, n), "w").write("epoch %d" % e)
    if stop and e == stop:
        sys.exit(1)
'''


def _selftest_resume(cfg):
    import ast
    import re
    import importlib.util
    import os
    import subprocess
    import types
    from . import resume as R, baseline as B
    from .common import EXP_DIR, code_snapshot_sha256
    n = {"ok": 0, "skip": 0}

    # --- Python 3.9 syntax of every source file
    for p in list((EXP_DIR / "src").rglob("*.py")) + [EXP_DIR / "run.py"]:
        ast.parse(p.read_text(), feature_version=(3, 9))
    n["ok"] += 1

    # --- the patch on the pinned train.py: clean apply, epoch ranges
    pinned = None
    for src in (os.environ.get("ACWM_PHYS_SRC"), str(EXP_DIR.parents[1] / "thread0")):
        if not src:
            continue
        r = subprocess.run(["git", "-C", src, "show", "%s:train.py" % cfg["code"]["commit"]], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout:
            pinned = r.stdout
            break
        r = subprocess.run(["git", "-C", src, "show", "HEAD:train.py"], capture_output=True, text=True)      # Colab checkout is at the pinned commit
        if r.returncode == 0 and r.stdout and cfg["code"]["commit"] == subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip():
            pinned = r.stdout
            break
    patch = EXP_DIR / "colab" / "train_resume.patch"
    if pinned is None or not patch.exists():
        log("selftest: patch-on-pinned-train.py SKIPPED (pinned train.py or patch file not available here)")
        n["skip"] += 1
    else:
        td = Path(tempfile.mkdtemp(prefix="exp003_patch_"))
        (td / "train.py").write_text(pinned)
        assert not R.train_py_is_patched(td)
        chk = subprocess.run(["git", "apply", "--check", str(patch)], cwd=td, capture_output=True, text=True)
        assert chk.returncode == 0, chk.stderr
        assert subprocess.run(["git", "apply", str(patch)], cwd=td, capture_output=True).returncode == 0
        assert R.train_py_is_patched(td)
        assert subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=td, capture_output=True).returncode == 0

        def epochs(text, init):
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and getattr(node.iter.func, "id", "") == "range" and "init_epoch" in ast.dump(node.iter):
                    ex = ast.Expression(ast.Call(ast.Name("list", ast.Load()), [node.iter], []))
                    ast.fix_missing_locations(ex)
                    return eval(compile(ex, "<loop>", "eval"), {"list": list, "range": range, "init_epoch": init, "self": types.SimpleNamespace(total_epochs=15)})
            raise AssertionError("loop not found")
        patched = (td / "train.py").read_text()
        assert epochs(patched, 1) == list(range(1, 16)) and epochs(patched, 6) == list(range(6, 16)) and epochs(patched, 16) == []
        assert re.search(r"@torch\.no_grad\(\)[^\n]*\n\s*def val\(self\)", patched), "@torch.no_grad() above val() must remain after the patch"
        assert re.search(r"@torch\.no_grad\(\)[^\n]*\n\s*def val\(self\)", pinned)
        assert epochs(pinned, 6) == list(range(6, 21))       # the original problem: 15 MORE epochs after a restart at epoch 5
        log("selftest: patch applies cleanly to the pinned train.py; epochs fresh 1..15, resume after 5 -> 6..15, after 15 -> none; unpatched restart after 5 -> 6..20 (15 more). "
            "'@torch.no_grad()' above val() is kept")
        n["ok"] += 1

    # --- prepare_run branches (text checkpoints; 'BROKEN' content is unreadable)
    def loadable(p):
        t = Path(p).read_text()
        if t.startswith("BROKEN"):
            raise ValueError("corrupt")
        return True

    def ckdir(root, name, files):
        ck = Path(root) / name / "checkpoints"
        ck.mkdir(parents=True)
        for fn, txt in files.items():
            (ck / fn).write_text(txt)
        return Path(root) / name
    td = Path(tempfile.mkdtemp(prefix="exp003_prep_"))
    q = td / "quarantine"
    assert R.prepare_run(td / "none", q, loadable)["action"] == "fresh"
    d = ckdir(td, "r_ok", {"model_latest.pth": "epoch 6", "model_6.pth": "epoch 6"})
    assert R.prepare_run(d, q, loadable)["action"] == "resume"
    d = ckdir(td, "r_broken", {"model_latest.pth": "BROKEN", "model_5.pth": "epoch 5", "model_6.pth": "BROKEN", "model_4.pth": "epoch 4"})
    r = R.prepare_run(d, q, loadable)
    assert r["action"] == "repaired_resume" and r["restored_from"] == "model_5.pth" and (d / "checkpoints" / "model_latest.pth").read_text() == "epoch 5", r
    d = ckdir(td, "r_none", {"model_latest.pth": "BROKEN", "model_5.pth": "BROKEN"})
    (d / "epoch_logs.jsonl").write_text("{}\n")
    r = R.prepare_run(d, q, loadable)
    assert r["action"] == "quarantined" and not (d / "checkpoints").exists() and not (d / "epoch_logs.jsonl").exists()
    moved = list(q.glob("r_none__*"))
    assert len(moved) == 1 and (moved[0] / "checkpoints" / "model_5.pth").exists(), "quarantine moves, never deletes"
    d = ckdir(td, "r_imp", {"model_latest.pth": "epoch 6", "model_6.pth": "epoch 6"})
    def imp(p):
        raise ImportError("No module named models")
    try:
        R.prepare_run(d, q, imp)
        raise AssertionError("ImportError must propagate")
    except ImportError:
        pass
    assert (d / "checkpoints" / "model_latest.pth").exists() and not list(q.glob("r_imp__*")), "ImportError must not quarantine"
    d = ckdir(td, "r_done", {"model_latest.pth": "BROKEN"})
    (d / "DONE").write_text("ok\n")
    assert R.prepare_run(d, q, loadable)["action"] == "skip" and (d / "checkpoints" / "model_latest.pth").read_text() == "BROKEN"
    d = ckdir(td, "x_smoke", {"model_latest.pth": "BROKEN"})
    assert R.prepare_run(d, q, loadable)["action"] == "skip" and R.prune_run(d, [15]) == []
    log("selftest: prepare_run (fresh, resume, broken latest repaired from newest readable epoch, quarantine by move, ImportError not quarantined, DONE and smoke untouched) OK")
    n["ok"] += 1

    # --- _train_one end to end with a fake train.py
    spec = importlib.util.spec_from_file_location("exp003_run_selftest", str(EXP_DIR / "run.py"))
    runmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runmod)
    c = copy.deepcopy(cfg)
    c["colab"] = {"keep_ckpt_epochs": [5, 10, 15], "baseline_001_bytes": 12237220}
    tt = Path(tempfile.mkdtemp(prefix="exp003_train_"))
    paths = {"train_base": tt / "tr", "manifests": tt / "man", "acwm_src": tt / "src", "dataset_dir": tt / "data", "results_dir": tt / "res", "cache": tt / "cache"}
    for k in ("manifests", "acwm_src", "results_dir"):
        paths[k].mkdir(parents=True)
    for dist in c["train"]["distributions"]:
        write_json({"n_episodes": 1, "group_counts": {}, "episodes": []}, paths["manifests"] / ("train_%s.json" % dist))
    old_env = {k: os.environ.get(k) for k in ("EXP003_APPROVED", "FAKE_STOP_AFTER", "EXP003_ALLOW_RETRAIN")}
    os.environ["EXP003_APPROVED"] = "single_a026 mix_wide mix_adjacent single_a248"
    os.environ.pop("FAKE_STOP_AFTER", None)
    os.environ.pop("EXP003_ALLOW_RETRAIN", None)
    old_load, old_gpu = R.default_loadable, R.gpu_info
    R.default_loadable = lambda p, src=None: loadable(p)
    R.gpu_info = lambda: {"gpu": "SYNTHETIC GPU", "memory_total_mib": 1}
    args = types.SimpleNamespace(dry_run=False, force=False)
    try:
        # mix_wide: finished before this code existed (DONE + metrics + save_every=5 checkpoints); must stay byte-identical
        mw = paths["train_base"] / "outputs" / "exp003_mix_wide_s0"
        (mw / "checkpoints").mkdir(parents=True)
        for fn in ("model_5.pth", "model_10.pth", "model_15.pth", "model_latest.pth"):
            (mw / "checkpoints" / fn).write_text("epoch " + fn)
        (mw / "DONE").write_text("ok\n")
        (paths["results_dir"] / "exp003_mix_wide_s0").mkdir()
        (paths["results_dir"] / "exp003_mix_wide_s0" / "metrics.json").write_text("{}")
        sm = paths["train_base"] / "outputs" / "exp003_mix_wide_s0_smoke"
        (sm / "checkpoints").mkdir(parents=True)
        (sm / "checkpoints" / "model_latest.pth").write_text("epoch 1")
        (sm / "DONE").write_text("ok\n")
        before = _tree(paths["train_base"]), _tree(paths["results_dir"])
        # unpatched train.py: refuse (fresh and with checkpoint), rc 3, nothing changes, not counted as an attempt
        (paths["acwm_src"] / "train.py").write_text(FAKE_TRAIN.replace("# [exp002 operational]", "#").replace("total + 1", "total + ep"))
        rc = runmod._train_one(c, paths, "single_a026", 0, args)
        assert rc == runmod.REFUSED and not (paths["train_base"] / "outputs" / "exp003_single_a026_s0" / "checkpoints").exists()
        sa = paths["train_base"] / "outputs" / "exp003_single_a026_s0"
        (sa / "checkpoints").mkdir(parents=True, exist_ok=True)
        (sa / "checkpoints" / "model_latest.pth").write_text("epoch 5")
        (sa / "checkpoints" / "model_5.pth").write_text("epoch 5")
        assert runmod._train_one(c, paths, "single_a026", 0, args) == runmod.REFUSED and not (sa / "DONE").exists() and not (sa / "train_env.json").exists()
        assert runmod.guarded(c, paths, "train:t", lambda: runmod.REFUSED, False) == runmod.REFUSED and not (paths["results_dir"] / "attempts.json").exists()
        # patched: resume from epoch 5 -> trains 6..15 only
        (paths["acwm_src"] / "train.py").write_text(FAKE_TRAIN)
        assert R.train_py_is_patched(paths["acwm_src"])
        assert runmod._train_one(c, paths, "single_a026", 0, args) == 0 and (sa / "DONE").exists()
        assert (sa / "fake_trained.txt").read_text().split() == [str(e) for e in range(6, 16)]
        te = R.read_train_env(sa)
        assert te["recorded"] and te["resumed"] and te["gpu"] == "SYNTHETIC GPU" and te["attempts"][0]["action"] == "resume" and te["attempts"][0]["epoch_checkpoint_at_start"] == 5 and te["attempts"][0]["target_total_epochs"] == 15
        # fresh run, interrupted after epoch 6, latest broken, then resume: 1..6 then 7..15 (total 15 epochs, no duplicates)
        os.environ["FAKE_STOP_AFTER"] = "6"
        sb = paths["train_base"] / "outputs" / "exp003_single_a248_s1"
        assert runmod._train_one(c, paths, "single_a248", 1, args) == 1 and not (sb / "DONE").exists()
        os.environ.pop("FAKE_STOP_AFTER")
        (sb / "checkpoints" / "model_latest.pth").write_text("BROKEN")
        assert runmod._train_one(c, paths, "single_a248", 1, args) == 0 and (sb / "DONE").exists()
        assert (sb / "fake_trained.txt").read_text().split() == [str(e) for e in range(1, 16)], (sb / "fake_trained.txt").read_text()
        te = R.read_train_env(sb)
        assert [a["action"] for a in te["attempts"]] == ["fresh", "repaired_resume"] and te["attempts"][1]["restored_from"] == "model_6.pth" and te["n_resumes"] == 1 and te["finished_ok"]
        assert sorted(f.name for f in (sb / "checkpoints").iterdir()) == ["model_10.pth", "model_15.pth", "model_5.pth", "model_latest.pth"], "prune keeps 5/10/15 + latest"
        # mix_wide and smoke: unchanged by all of the above, skipped, --force refused
        assert runmod._train_one(c, paths, "mix_wide", 0, args) == 0
        args.force = True
        assert runmod._train_one(c, paths, "mix_wide", 0, args) == runmod.REFUSED
        args.force = False
        assert R.prune_run(mw, [15]) == [] and R.prepare_run(mw, paths["train_base"] / "quarantine", loadable)["action"] == "skip"
        after = _tree(paths["train_base"]), _tree(paths["results_dir"])
        for key in before[0]:
            if "mix_wide" in key:
                assert after[0].get(key) == before[0][key], key
        assert all(after[1].get(k) == v for k, v in before[1].items())
        env_mw = R.read_train_env(mw)
        assert env_mw["recorded"] is False and env_mw["gpu"] == "unknown"
        log("selftest: _train_one guard (unpatched refused rc 3 not counted as attempt; patched resume 6..15; fresh+interrupt+broken latest -> repaired resume, 15 epochs total; prune keeps 5/10/15+latest; "
            "finished mix_wide and smoke byte-identical, skipped, --force refused; mix_wide train_env gpu 'unknown') OK")
        n["ok"] += 1
    finally:
        R.default_loadable, R.gpu_info = old_load, old_gpu
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # --- aggregate: train_env in runs, copy to the work root, code sha
    td = Path(tempfile.mkdtemp(prefix="exp003_agg_"))
    out = td / "results.json"
    write_json({"x": 1}, out)
    dst = AG.persist(out, str(td / "work"))
    assert dst == td / "work" / "results" / "results.json" and read_json(dst) == {"x": 1} and AG.persist(out, None) is None
    assert code_snapshot_sha256(EXP_DIR) == code_snapshot_sha256(EXP_DIR) and len(code_snapshot_sha256(EXP_DIR)) == 64
    log("selftest: aggregate.persist copies results.json to WORK_ROOT/results; code snapshot sha256 computed OK")
    n["ok"] += 1

    # --- 001 baseline file: not found fails, found is copied and read, ambiguous fails, unreadable fails
    td = Path(tempfile.mkdtemp(prefix="exp003_base_"))
    size = 12237220
    bc = copy.deepcopy(cfg)
    bc["colab"] = {"baseline_001_bytes": size}
    base_dir = td / "exp" / "003"
    base_dir.mkdir(parents=True)
    spec_b = bc["criteria"]["h2_baseline"]

    def make(path, vals=None, pad=size):
        d = {"runs": {}}
        for rn, v in zip(spec_b["runs"], vals or [0.24, 0.25, 0.26]):
            cur = d["runs"].setdefault(rn, {})
            for p in spec_b["slope_path"][:-1]:
                cur = cur.setdefault(p, {})
            cur[spec_b["slope_path"][-1]] = v
        b = json.dumps(d).encode()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b + b" " * (pad - len(b)))
    ok, info = B.run(bc, base_dir=base_dir, roots=[str(td / "drive")])
    assert not ok and info["action"] == "not_found" and not B.target_path(bc, base_dir).exists()
    make(td / "drive" / "MyDrive" / "old" / "results.json", pad=size - 5)             # decoy: wrong size
    make(td / "drive" / "MyDrive" / "exp001_stuff" / "sub" / "results.json")
    ok, info = B.run(bc, base_dir=base_dir, roots=[str(td / "drive")])
    assert ok and info["action"] == "copied" and info["found"] and info["baseline"]["values"][spec_b["runs"][0]] == 0.24, info
    assert B.target_path(bc, base_dir).stat().st_size == size
    ok, info = B.run(bc, base_dir=base_dir, roots=[])
    assert ok and info["action"] == "already_present"
    B.target_path(bc, base_dir).unlink()
    make(td / "drive" / "MyDrive" / "other" / "results.json", vals=[0.1, 0.2, 0.3])
    ok, info = B.run(bc, base_dir=base_dir, roots=[str(td / "drive")])
    assert not ok and info["action"] == "ambiguous"
    t2 = Path(tempfile.mkdtemp(prefix="exp003_base2_"))
    (t2 / "d").mkdir()
    (t2 / "d" / "results.json").write_bytes(b'{"runs": {}}' + b" " * (size - 12))
    ok, info = B.run(bc, base_dir=t2 / "a" / "b", roots=[str(t2 / "d")])
    assert not ok and info["found"] and not info["baseline"]["values_readable"], info
    log("selftest: 001 baseline install (not found -> fail, wrong-size decoy ignored, copied + baseline.found + values read, already present, ambiguous -> fail, unreadable runs -> fail) OK")
    n["ok"] += 1
    return n
