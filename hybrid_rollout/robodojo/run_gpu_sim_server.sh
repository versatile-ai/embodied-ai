#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: $0 OUTPUT_DIR [TASK] [PORT] [EVAL_SEED]" >&2
  exit 2
fi

output_dir=$1
task=${2:-put_bottles_into_dustbin}
port=${3:-19113}
eval_seed=${4:-0}
robodojo_root=${ROBODOJO_ROOT:-/root/RoboDojo}
adapter_root=${ROBODOJO_ADAPTER_ROOT:-/root/gpt_eval_code_20260919}
python_bin=${ROBODOJO_PYTHON:-/opt/Anaconda3/envs/RoboDojo/bin/python}

command -v nvidia-smi >/dev/null
nvidia-smi -L | grep -q '^GPU 0:'
test -x "$python_bin"
test -d "$robodojo_root"
test -d "$adapter_root/hybrid_rollout"

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ROBODOJO_SKIP_EVAL_SCENE_SETUP=1
export ROBODOJO_DIRECT_RGB_CAPTURE=1
export ROBODOJO_POST_RESET_WARMUP_RENDERS=3
export ROBODOJO_LIVE_DIR=${ROBODOJO_LIVE_DIR:-/tmp/robodojo_live_frames}
export PYTHONPATH="$adapter_root:$robodojo_root:$robodojo_root/XPolicyLab:$robodojo_root/third_party/curobo${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$output_dir" "$ROBODOJO_LIVE_DIR"
exec "$python_bin" -m hybrid_rollout.robodojo.robodojo_server.server \
  --task "$task" \
  --output "$output_dir" \
  --port "$port" \
  --eval-seed "$eval_seed" \
  --device cuda:0
