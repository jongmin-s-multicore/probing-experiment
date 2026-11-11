"""Experiment 014 entry point.  python run.py [--smoke] | python run.py selftest
Reads EXP004_CACHE (enc/tokens_k12.npy, enc/rand_k12.npy) and EXP005_CACHE (feat_pred_out6.npy, feat_pred_out6_rand.npy). Writes results.json. No training, CPU only."""
import argparse
import datetime
import os
import sys
from pathlib import Path

import yaml

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / "src"))

from exp014 import data as D  # noqa: E402
from exp014 import runner as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="main", choices=["main", "selftest", "check"])
    ap.add_argument("--smoke", action="store_true", help="few bootstrap/null reps -> results_smoke.json (not a result)")
    a = ap.parse_args()
    if a.mode == "selftest":
        from exp014 import selftest
        return selftest.main()
    cfg_path = here / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    inputs = Path(os.environ.get("EXP014_INPUTS", str(D.DEFAULT_INPUTS)))
    try:
        paths = D.cache_paths(os.environ)
    except KeyError as e:
        print("STOP:", e); return 2
    miss = [str(p) for p in list(paths.values()) + [inputs] if not p.exists()]
    if miss:
        print("STOP: input files missing (main analysis not run):\n  " + "\n  ".join(miss)); return 2
    if a.mode == "check":
        print("all inputs present"); return 0
    log = R.make_logger(here / "logs" / ("%s_%s.log" % ("smoke" if a.smoke else "main", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))))
    R.run_main(cfg, cfg_path, paths, inputs, here / ("results_smoke.json" if a.smoke else "results.json"), log, smoke=a.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
