"""Encode recorded frames with AVFoundation on Mac or FFmpeg elsewhere."""
import os
from pathlib import Path
import subprocess
import sys


def encode_video(root: Path, frames: Path, destination: Path, fps: int) -> None:
    """Encode a session's contiguous, zero-based PNG frames to MP4."""
    if sys.platform == "darwin":
        command = [str(root / "encode_video"), str(frames), str(destination), str(fps)]
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
