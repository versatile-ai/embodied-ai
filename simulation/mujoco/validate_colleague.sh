#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
[[ -f "$VENV_DIR/bin/activate" ]] || {
  echo "Missing virtualenv. Run ./setup_colleague.sh first." >&2
  exit 1
}
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

PYTHONPYCACHEPREFIX=/tmp/astra-pycache python -u test_service.py
PYTHONPYCACHEPREFIX=/tmp/astra-pycache python -u check_render.py
python start_server.py
PYTHONPYCACHEPREFIX=/tmp/astra-pycache python -u verify_http.py

echo "Validation passed. Open http://127.0.0.1:8763/live"
