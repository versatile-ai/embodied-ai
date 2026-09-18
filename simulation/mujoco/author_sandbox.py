"""WSL-only analysis sandbox; never silently fall back to Windows/full access."""

import json
import os
from pathlib import Path
import subprocess
import shlex

DISTRO = os.environ.get("ASTRA_ANALYSIS_WSL", "Ubuntu-22.04")
LINUX_HOME = os.environ.get("ASTRA_ANALYSIS_HOME", "/home/ken")
NODE = LINUX_HOME + "/.nvm/versions/node/v24.15.0/bin/node"
CODEX = (
    LINUX_HOME + "/.local/share/robodojo-codex/node_modules/@openai/codex/bin/codex.js"
)
RUNTIME = LINUX_HOME + "/.local/share/robodojo-analysis-runtime"
PYTHON = RUNTIME + "/bin/python"


def linux_path(path: Path) -> str:
    resolved = path.resolve().as_posix()
    if len(resolved) < 3 or resolved[1:3] != ":/":
        raise ValueError("Expected absolute Windows drive path for WSL bridge")
    return "/mnt/" + resolved[0].lower() + resolved[2:]


def prefix() -> list[str]:
    return [NODE, CODEX]


def wsl_command(args: list[str]) -> list[str]:
    return ["wsl.exe", "-d", DISTRO, "--", "bash", "-lc", shlex.join(args)]


def permissions(root: Path) -> list[str]:
    filesystem = {
        ":root": "deny",
        ":minimal": "read",
        ":tmpdir": "deny",
        ":slash_tmp": "deny",
        linux_path(root): "write",
        RUNTIME: "read",
        LINUX_HOME + "/.local/share/robodojo-codex": "read",
        LINUX_HOME + "/.local/share/uv/python": "read",
        linux_path(root / "evidence"): "read",
    }
    table = (
        "{"
        + ",".join(json.dumps(k) + "=" + json.dumps(v) for k, v in filesystem.items())
        + "}"
    )
    return [
        "-c",
        'default_permissions="author"',
        "-c",
        "permissions.author.filesystem=" + table,
        "-c",
        "permissions.author.network.enabled=false",
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="disabled"',
        "-c",
        'shell_environment_policy.inherit="none"',
    ]


def probe(root: Path, canary: Path) -> dict:
    """Read only a harmless canary, and test denied loopback without robot actions."""
    if not canary.is_file():
        raise ValueError("Probe canary must exist outside sandbox")
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "sandbox_canary.txt").write_text("immutable evidence", encoding="utf-8")
    code = """import json,pathlib,socket
import numpy as np
from PIL import Image
result={}
evidence=pathlib.Path('evidence/sandbox_canary.txt')
result['evidence_readable']=evidence.read_text()=='immutable evidence'
try:
    evidence.write_text('modified')
    result['evidence_write_blocked']=False
except OSError:
    result['evidence_write_blocked']=True
try:
    evidence.unlink()
    result['evidence_delete_blocked']=False
except OSError:
    result['evidence_delete_blocked']=True
try:
    pathlib.Path(CANARY).read_bytes()
    result['outside_read_blocked']=False
except (PermissionError,FileNotFoundError):
    result['outside_read_blocked']=True
try:
    socket.socket().connect(('127.0.0.1',8763))
    result['network_blocked']=False
except OSError:
    result['network_blocked']=True
try:
    socket.socket().bind(('0.0.0.0',0))
    result['network_bind_blocked']=False
except OSError:
    result['network_bind_blocked']=True
pathlib.Path('sandbox_notes.txt').write_text('sandbox capability test')
Image.new('RGB',(32,32)).crop((0,0,16,16)).save('sandbox_crop.png')
result['notes_crop_math_ok']=int(np.arange(4).sum())==6
print(json.dumps(result))
""".replace("CANARY", repr(linux_path(canary)))
    cmd = prefix() + [
        "sandbox",
        "-C",
        linux_path(root),
        "-P",
        "author",
        *permissions(root),
        "--",
        PYTHON,
        "-c",
        code,
    ]
    result = subprocess.run(
        wsl_command(cmd), capture_output=True, text=True, timeout=60
    )
    if result.returncode:
        raise RuntimeError("WSL sandbox probe failed: " + result.stderr[-3000:])
    data = json.loads(result.stdout.strip().splitlines()[-1])
    if not all(data.values()):
        raise RuntimeError("WSL confinement failed: " + json.dumps(data))
    return data


def canonical_image_allowed(path: str, root: Path) -> bool:
    """Audit actual WSL path, including symlinks; not a pre-read security hook."""
    base = linux_path(root)
    if not path.startswith(base + "/") or ".." in path.split("/"):
        return False
    result = subprocess.run(
        wsl_command(["realpath", "-e", "--", path]),
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0 and result.stdout.strip().startswith(base + "/")
