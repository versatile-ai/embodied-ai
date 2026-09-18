"""User-authorized native Windows functional backend, without sandbox isolation."""

import json
import os
from pathlib import Path
import shutil
import subprocess

PYTHON = str(
    Path(os.environ["LOCALAPPDATA"]) / "robodojo-analysis-runtime/Scripts/python.exe"
)


def linux_path(path: Path) -> str:
    """Backend-compatible path adapter; no WSL conversion on Windows."""
    return path.resolve().as_posix()


def prefix() -> list[str]:
    executable = shutil.which("codex.exe")
    if not executable:
        raise RuntimeError("Native Windows Codex executable not found")
    return [executable]


def wsl_command(args: list[str]) -> list[str]:
    return args


def permissions(root: Path) -> list[str]:
    return [
        "-c",
        'default_permissions=":danger-full-access"',
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="disabled"',
    ]


def transport_config() -> list[str]:
    """Same official ChatGPT endpoint/auth, using HTTP rather than WebSocket."""
    return [
        "-c",
        'model_provider="author_http"',
        "-c",
        'model_providers.author_http.name="OpenAI HTTP"',
        "-c",
        'model_providers.author_http.base_url="https://chatgpt.com/backend-api/codex"',
        "-c",
        "model_providers.author_http.requires_openai_auth=true",
        "-c",
        "model_providers.author_http.supports_websockets=false",
    ]


def probe(root: Path, canary: Path) -> dict:
    """Verify native tools, not an isolation guarantee."""
    result = subprocess.run(
        [
            PYTHON,
            "-c",
            "import json,numpy as np; from PIL import Image; "
            "print(json.dumps({'numpy_sum':int(np.arange(10).sum()),'pillow':Image.new('RGB',(2,2)).size== (2,2)}))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise RuntimeError("Windows analysis runtime failed: " + result.stderr[-2000:])
    checks = json.loads(result.stdout)
    if checks != {"numpy_sum": 45, "pillow": True}:
        raise RuntimeError("Windows analysis capability check failed")
    return dict(
        backend="windows",
        permission_profile=":danger-full-access",
        read_confinement_verified=False,
        checks=checks,
    )


def canonical_image_allowed(path: str, root: Path) -> bool:
    try:
        target = Path(path)
        return target.is_absolute() and target.resolve(strict=True).is_relative_to(
            root.resolve(strict=True)
        )
    except (OSError, ValueError):
        return False
