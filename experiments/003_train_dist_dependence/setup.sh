#!/usr/bin/env bash
# Experiment 003 environment setup. Run once on the machine that will run the experiment (needs git, conda, internet).
#   bash setup.sh [--dry-run] [--minimal]
# What it does
#   1. makes a read-only detached checkout of the acwm-phys code at the commit recorded in config.yaml
#      (git worktree if LOCAL_THREAD0 points to a clone that has the commit, otherwise a fresh clone of the upstream repo)
#   2. creates the conda env from that checkout's environment.yaml (identical to thread0/environment.yaml on main; verified by diff)
#      --minimal: python 3.9.19 + only the pinned packages this experiment imports (fallback if the full 217-package pip list fails)
# Variables: DINO_WORK (default ~/dinowm)  ACWM_PHYS_SRC (default $DINO_WORK/acwm_phys_src)  CONDA_ENV (default dino_wm)
#            LOCAL_THREAD0 (optional path of an existing thread0 clone)
# Status: written but NOT executed by the author (no install was run in the authoring session).
# Note: the hanbin5/cloth-sim branch/HF dataset are reported unusable by the user (2026-10-31); the code checkout stays the acwm-phys commit (config code.commit)
#       and the data come from Google Drive via rclone (config data.source: drive).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DRY=0; MINIMAL=0
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; --minimal) MINIMAL=1 ;; *) echo "unknown option $a"; exit 2 ;; esac; done
run() { echo "+ $*"; if [ "$DRY" = 0 ]; then eval "$@"; fi; }

DINO_WORK="${DINO_WORK:-$HOME/dinowm}"
ACWM_PHYS_SRC="${ACWM_PHYS_SRC:-$DINO_WORK/acwm_phys_src}"
CONDA_ENV="${CONDA_ENV:-dino_wm}"
REPO_URL="https://github.com/jongmin-s-multicore/thread0.git"
BRANCH="hanbin5/acwm-phys"
COMMIT="5a640bed66002eca7bdb31d10b2bde59ad944551"   # keep equal to config.yaml code.commit
grep -q "$COMMIT" "$HERE/config.yaml" || { echo "commit in setup.sh differs from config.yaml"; exit 1; }

echo "DINO_WORK=$DINO_WORK ACWM_PHYS_SRC=$ACWM_PHYS_SRC CONDA_ENV=$CONDA_ENV"
run "mkdir -p '$DINO_WORK'"

# 1. code checkout -------------------------------------------------------------------------------------------
if [ -d "$ACWM_PHYS_SRC/.git" ] || [ -f "$ACWM_PHYS_SRC/.git" ]; then
  echo "checkout exists: $(git -C "$ACWM_PHYS_SRC" rev-parse HEAD 2>/dev/null || echo '?')"
elif [ -n "${LOCAL_THREAD0:-}" ] && git -C "$LOCAL_THREAD0" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
  run "git -C '$LOCAL_THREAD0' worktree add --detach '$ACWM_PHYS_SRC' $COMMIT"
else
  run "git clone --no-checkout '$REPO_URL' '$ACWM_PHYS_SRC'"
  run "git -C '$ACWM_PHYS_SRC' checkout --detach $COMMIT"
fi
if [ "$DRY" = 0 ]; then
  [ "$(git -C "$ACWM_PHYS_SRC" rev-parse HEAD)" = "$COMMIT" ] || { echo "checkout is not at $COMMIT"; exit 1; }
  # the branch must contain the commit (documentation only; branch name: $BRANCH)
fi

# 2. conda env -----------------------------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "conda env $CONDA_ENV exists, skip creation"
elif [ "$MINIMAL" = 1 ]; then
  run "conda create -y -n '$CONDA_ENV' python=3.9.19 pip"
  # versions copied from environment.yaml of the pinned commit
  run "conda run -n '$CONDA_ENV' pip install torch==2.3.0 torchvision==0.18.0 numpy==1.26.4 einops==0.4.1 hydra-core==1.2.0 hydra-submitit-launcher==1.2.0 submitit==1.5.1 omegaconf==2.3.0 accelerate==0.26.1 wandb==0.13.1 decord==0.6.0 opencv-python==4.6.0.66 scikit-learn==1.5.0 scipy==1.13.1 scikit-image==0.19.3 huggingface-hub==0.23.2 pyyaml tqdm imageio"
else
  # NOTE: the file also pins mujoco-py, d4rl, tensorflow, jax ... (not needed by train/diagnose, but part of the user-requested env)
  run "conda env create -f '$ACWM_PHYS_SRC/environment.yaml' -n '$CONDA_ENV'"
fi
run "conda run -n '$CONDA_ENV' python -c \"import torch, hydra, decord, cv2, yaml, scipy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\""
echo
SRC="${EXP003_DATA_SOURCE:-$(sed -n 's/^  source: *"\{0,1\}\([a-z]*\)"\{0,1\}.*/\1/p' "$HERE/config.yaml" | head -1)}"
echo "Dataset source: ${SRC:-drive}  (config.yaml data.source / env EXP003_DATA_SOURCE: drive | hf)"
if [ "${SRC:-drive}" = "hf" ]; then
  echo "  source=hf: Hugging Face token needed for the dataset repo hanbin5/cloth-sim (unauthenticated access returned 401):"
  if [ -n "${HF_TOKEN:-}" ]; then echo "    HF_TOKEN is set in this shell (value not shown)"
  elif [ -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then echo "    found a token stored by huggingface-cli login (value not shown)"
  else echo "    NO token found. Create a read token at https://huggingface.co/settings/tokens, then 'export HF_TOKEN=hf_...' or 'huggingface-cli login'. Never put it in config.yaml."; fi
else
  echo "  source=drive: data is downloaded with rclone from the Google Drive folder (README 'Drive 에서 데이터 받기')."
  echo "  rclone found: $(command -v rclone || echo 'no -> install from https://rclone.org/install/')"
  if command -v rclone >/dev/null 2>&1; then echo "  rclone remotes: $(rclone listremotes 2>/dev/null | tr '\n' ' ')"; fi
  echo "  needed: a Drive remote (rclone config; same one can be used to upload results) + its NAME in config.yaml data.drive.rclone_remote (or env EXP003_RCLONE_REMOTE)"
  echo "          and access to the data folder: (a) default root_folder_id, or (b) shortcut/copy in your Drive + data.drive.remote_path. Both untested by the author."
  echo "  check: python run.py download --dry-run     (lists remote files/sizes; downloads nothing)"
fi
echo "Google Drive upload of results (optional): see README 'Drive 동기화' (fill config.yaml sync.drive.*; shares the rclone remote)."
echo "next: bash run_all.sh --dry-run"
