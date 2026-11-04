"""Hugging Face access helpers. The token is only ever passed to huggingface_hub calls; it is never printed, logged or written to a file.
Token sources, in order: env HF_TOKEN, env HUGGING_FACE_HUB_TOKEN, the token stored by `huggingface-cli login`."""
from __future__ import annotations

import os
from pathlib import Path

HELP = """\
Hugging Face access to the dataset repo failed ({what}).
What to do (on the machine that runs the experiment):
  1. Make sure your HF account has access to the repo '{repo}' (it is private or gated; unauthenticated calls return 401).
  2. Create a READ token at https://huggingface.co/settings/tokens
  3. Give it to the scripts, either
       export HF_TOKEN=hf_xxx            # for this shell only; do not put it in config.yaml, README or a script
     or
       huggingface-cli login             # stores it in ~/.cache/huggingface/token (mode 600)
  4. Re-run:  python run.py download --dry-run --query-size   (checks access), then  python run.py download
The token value is never written to logs or result files by this experiment."""


def get_token():
    """Return (token or None, source label). The label never contains the token."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(name, "").strip()
        if v:
            return v, f"env {name}"
    try:
        import huggingface_hub as hh
        for fn in ("get_token",):
            f = getattr(hh, fn, None)
            if f:
                v = f()
                if v:
                    return v, "huggingface-cli login (stored token)"
        from huggingface_hub import HfFolder          # older versions
        v = HfFolder.get_token()
        if v:
            return v, "huggingface-cli login (stored token)"
    except Exception:
        pass
    for p in (Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token",):
        try:
            v = p.read_text().strip()
            if v:
                return v, "huggingface-cli login (stored token)"
        except OSError:
            pass
    return None, "none"


def status_code(e):
    r = getattr(e, "response", None)
    return getattr(r, "status_code", None)


def explain_error(e, repo):
    """Message for an exception raised by huggingface_hub, or None when it does not look like an auth/permission problem."""
    sc = status_code(e)
    name = type(e).__name__
    if sc in (401, 403) or name in ("GatedRepoError",):
        return HELP.format(what=f"HTTP {sc or name}", repo=repo)
    if sc == 404 or name == "RepositoryNotFoundError":
        return HELP.format(what="HTTP 404 repo not found or not visible to this token (HF also answers 404/401 for private repos)", repo=repo)
    return None


def require_token(repo):
    tok, src = get_token()
    if tok is None:
        raise SystemExit(HELP.format(what="no token found: HF_TOKEN is not set and no `huggingface-cli login` token exists", repo=repo))
    return tok, src
