#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

command -v "$PYTHON_BIN" >/dev/null || { echo "python3 is required" >&2; exit 1; }
command -v swiftc >/dev/null || {
  echo "swiftc is required to build the macOS video encoder (install Xcode Command Line Tools)." >&2
  exit 1
}

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ ! -x "$ROOT_DIR/encode_video" ]]; then
  swiftc -module-cache-path /tmp/astra-swift-module-cache \
    "$ROOT_DIR/encode_video.swift" -o "$ROOT_DIR/encode_video"
fi

PYTHONPYCACHEPREFIX=/tmp/astra-pycache python -m py_compile \
  "$ROOT_DIR/harness/simsvc.py" "$ROOT_DIR/harness/control.py" \
  "$ROOT_DIR/client.py" "$ROOT_DIR/test_service.py" "$ROOT_DIR/verify_http.py"
echo "Environment ready. Run: $ROOT_DIR/validate_colleague.sh"
