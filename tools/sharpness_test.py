"""Sharpness metrics on matching frames of several videos: mean gradient magnitude and Laplacian variance
(higher = sharper / more fine detail). Also writes a 2x zoom crop grid of the first two videos.
usage: sharpness_test.py t1,t2,... OUT.png video1.mp4 video2.mp4 [...]
"""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import enhance
from PIL import Image, ImageDraw

times = [float(t) for t in sys.argv[1].split(",")]
outp = sys.argv[2]
vids = sys.argv[3:]


def frame(v, t):
    raw = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-ss", str(t), "-i", v, "-frames:v", "1",
                                   "-vf", "scale=1440:1080:flags=lanczos,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"])
    return np.frombuffer(raw, np.uint8).reshape(1080, 1440).astype(np.float32)


def metrics(g):
    gx = np.abs(np.diff(g, axis=1)).mean(); gy = np.abs(np.diff(g, axis=0)).mean()
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return (gx + gy) / 2, lap.var()


print(f"{'video':45s} {'gradient':>9s} {'laplacian var':>14s}")
for v in vids:
    gs = [metrics(frame(v, t)) for t in times]
    print(f"{os.path.basename(v):45s} {np.mean([g[0] for g in gs]):9.3f} {np.mean([g[1] for g in gs]):14.1f}")

# zoom grid: 320x180 region at 2x for the first two videos, at each time
cw, ch = 320, 180
grid = Image.new("RGB", (2 * cw * 2, len(times) * (ch * 2 + 22)), "black"); d = ImageDraw.Draw(grid)
for i, t in enumerate(times):
    for j, v in enumerate(vids[:2]):
        raw = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-ss", str(t), "-i", v, "-frames:v", "1",
                                       "-vf", f"crop={cw}:{ch}:560:450,format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        im = Image.fromarray(np.frombuffer(raw, np.uint8).reshape(ch, cw, 3)).resize((cw * 2, ch * 2), Image.NEAREST)
        x, y = j * cw * 2, i * (ch * 2 + 22)
        grid.paste(im, (x, y + 22)); d.text((x + 6, y + 4), f"{os.path.basename(v)}  t={t}s  [2x zoom]", fill="yellow")
grid.save(outp); print("wrote", outp)
