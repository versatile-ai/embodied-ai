#!/usr/bin/env python3
"""把实验保存的frames.npz转成mp4视频（agentview）。
用法: python3 frames_to_video.py <frames.npz> <out.mp4> [fps]
"""
import sys
import numpy as np

def npz_to_mp4(npz_path, mp4_path, fps=10):
    import cv2
    d = np.load(npz_path)
    key = "main_images" if "main_images" in d else list(d.keys())[0]
    frames = d[key]
    if len(frames) == 0:
        print("no frames"); return False
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        img = f if f.dtype == np.uint8 else (f * 255).astype(np.uint8)
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        vw.write(img)
    vw.release()
    print(f"{mp4_path}: {len(frames)} frames @ {fps}fps")
    return True

if __name__ == "__main__":
    npz_to_mp4(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
