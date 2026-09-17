"""Isolated local/server comparison; never connects to the live HTTP service."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "harness"))

import mujoco
from OpenGL import GL
from simsvc import Session


def peak_rss_mib() -> float:
    """Return peak process resident memory, not dedicated GPU memory."""
    if os.name != "nt":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / 1024**2


def main() -> None:
    """Measure six warm control steps with all three recorded camera streams."""
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "harness/simsvc.py",
            "assets/x5/dual_x5_scene.xml",
            "layouts/put_bottles_into_dustbin_0.json",
        )
    }
    layout = json.loads((ROOT / "layouts/put_bottles_into_dustbin_0.json").read_text())
    # Only this unique temporary directory is cleaned; historical runs are untouched.
    with tempfile.TemporaryDirectory(
        prefix="benchmark-", dir=ROOT / "runs"
    ) as temporary:
        began = time.perf_counter()
        session = Session(
            layout,
            "Isolated rendering benchmark",
            record_dir=Path(temporary) / "episode",
        )
        try:
            setup_seconds = time.perf_counter() - began
            renderer = GL.glGetString(GL.GL_RENDERER).decode()
            target = session.state14()
            target[0] = target[7] = 0.15
            target[6] = target[13] = 1.0
            session.act([target])  # Untimed warm-up, same on both hosts.
            times = []
            for _ in range(6):
                began = time.perf_counter()
                session.act([target])
                times.append(time.perf_counter() - began)
            assert not session.error, session.error
            print(
                json.dumps(
                    {
                        "platform": platform.platform(),
                        "python": platform.python_version(),
                        "mujoco": mujoco.__version__,
                        "renderer": renderer,
                        "camera_count": 3,
                        "resolution": [640, 480],
                        "shadow_size": int(session.model.vis.quality.shadowsize),
                        "mesh_faces": session.model.nmeshface,
                        "setup_seconds": setup_seconds,
                        "step_seconds": times,
                        "mean_seconds": statistics.mean(times),
                        "median_seconds": statistics.median(times),
                        "peak_rss_mib": peak_rss_mib(),
                        "hashes": hashes,
                    },
                    indent=2,
                ),
                flush=True,
            )
        finally:
            session.renderer.close()


if __name__ == "__main__":
    main()
