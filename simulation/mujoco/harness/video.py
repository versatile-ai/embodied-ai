"""Encode recorded frames with AVFoundation on Mac or FFmpeg elsewhere."""
import os
from pathlib import Path
import subprocess
import sys


def encode_video(root: Path, frames: Path, destination: Path, fps: int) -> None:
    """Encode a session's contiguous, zero-based PNG frames to MP4."""
    if sys.platform == "darwin":
        binary = root / "encode_video"
        if not binary.exists():
            source = root / "encode_video.swift"
            if not source.exists():
                raise RuntimeError(f"Missing macOS video encoder source: {source}")
            subprocess.run([
                "swiftc", "-module-cache-path", "/tmp/astra-swift-module-cache",
                str(source), "-o", str(binary),
            ], check=True, timeout=120, capture_output=True)
        command = [str(binary), str(frames), str(destination), str(fps)]
    else:
        import imageio_ffmpeg

        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-framerate", str(fps), "-start_number", "0",
            "-i", str(frames / "%06d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
        ]
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    subprocess.run(command, check=True, timeout=60, capture_output=True, **options)
