"""Temporal flicker metric: mean abs difference between consecutive frames inside the static OSD area.
Lower = more stable. Compares the source (scaled to the same size) with one or more processed outputs.

usage: flicker_test.py SOURCE.mp4 start_sec n_frames OUT1.mp4 [OUT2.mp4 ...]
"""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import enhance

args = [a for a in sys.argv[1:] if a != "--full"]
src, start, n = args[0], float(args[1]), int(args[2])
outs = args[3:]
W, H = 1440, 1080
# static OSD regions in the 1440x1080 frame: top-left telemetry block and bottom-right battery block
REGIONS = [(40, 60, 560, 300), (1100, 820, 1400, 1000)]
if "--full" in sys.argv:   # whole frame: only meaningful on a static scene (e.g. after landing)
    REGIONS = [(0, 0, W, H)]


def frames(path, ss):
    cmd = [enhance.FFMPEG, "-v", "error", "-ss", str(ss), "-i", path, "-frames:v", str(n),
           "-vf", f"scale={W}:{H}:flags=lanczos,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)


def metric(f):
    vals = []
    for (x0, y0, x1, y1) in REGIONS:
        r = f[:, y0:y1, x0:x1]
        vals.append(np.abs(np.diff(r, axis=0)).mean())
    return np.mean(vals)


print(f"{'file':60s} flicker (mean |dt| in static OSD)")
print(f"{'source (scaled)':60s} {metric(frames(src, start)):.3f}")
for o in outs:
    print(f"{os.path.basename(o):60s} {metric(frames(o, 0)):.3f}")
