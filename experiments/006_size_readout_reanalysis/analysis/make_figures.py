#!/usr/bin/env python
"""Regenerate analysis/fig*.png from ../results.json (no recomputation). Needs matplotlib. Same code as `python run.py figures`."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from exp006 import figs  # noqa: E402

res = json.loads((HERE.parent / "results.json").read_text())
made, failed = figs.make_all(res, HERE)
print("made:", made)
print("failed:", failed)
