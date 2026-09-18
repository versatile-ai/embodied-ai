"""Check native Codex read/write/network confinement without any model call."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile

RUNTIME = Path("C:/Users/qingyan/AppData/Local/robodojo-analysis-runtime")


def permission_args(root: Path) -> list[str]:
    filesystem = {
        ":root": "deny",
        ":minimal": "read",
        str(root): "write",
        "C:/Python313": "read",
        str(RUNTIME): "read",
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
    ]


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="robodojo_sandbox_probe_"))
    # This inert sibling is deliberately outside the allowlisted workspace.
    with tempfile.NamedTemporaryFile(prefix="robodojo_canary_", delete=False) as f:
        f.write(b"harmless permission test")
        canary = Path(f.name)
    code = """import json, pathlib, socket
import numpy as np
from PIL import Image
r=pathlib.Path.cwd()
result={}
try:
    pathlib.Path(CANARY).read_bytes()
    result['outside_read_blocked']=False
except PermissionError:
    result['outside_read_blocked']=True
try:
    s=socket.create_connection(('127.0.0.1',8763),timeout=2)
    s.close()
    result['sim_network_blocked']=False
except OSError:
    result['sim_network_blocked']=True
(r/'NOTES.md').write_text('sandbox test')
Image.new('RGB',(32,32)).crop((0,0,16,16)).save(r/'crop.png')
result['notes_crop_math_ok']=int(np.arange(4).sum())==6
print(json.dumps(result))
""".replace("CANARY", repr(str(canary)))
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("Codex unavailable")
    cmd = [
        executable,
        "sandbox",
        "-C",
        str(root),
        "-P",
        "author",
        *permission_args(root),
        "--",
        str(RUNTIME / "Scripts/python.exe"),
        "-c",
        code,
    ]
    print("PROBE_WORKSPACE", root, flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    print(result.stdout)
    print(result.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    data = json.loads(result.stdout.strip().splitlines()[-1])
    if not all(data.values()):
        raise RuntimeError("Sandbox confinement probe failed")


if __name__ == "__main__":
    main()
