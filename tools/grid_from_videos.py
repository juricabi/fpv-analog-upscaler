"""Labelled crop grid from the same timestamp of several videos (1:1 pixels).
usage: grid_from_videos.py t_seconds OUT.png crop_x,crop_y[,w,h] video1.mp4 [video2.mp4 ...]
"""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import enhance
from PIL import Image, ImageDraw

t, outp = float(sys.argv[1]), sys.argv[2]
c = [int(v) for v in sys.argv[3].split(",")]
x0, y0 = c[0], c[1]
cw, ch = (c[2], c[3]) if len(c) > 3 else (640, 360)
vids = sys.argv[4:]
cells = []
for v in vids:
    raw = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-ss", str(t), "-i", v, "-frames:v", "1",
                                   "-vf", f"crop={cw}:{ch}:{x0}:{y0},format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    cells.append((os.path.basename(v), Image.fromarray(np.frombuffer(raw, np.uint8).reshape(ch, cw, 3))))
cols = 2; rows = (len(cells) + 1) // 2
grid = Image.new("RGB", (cols * cw, rows * (ch + 22)), "black"); d = ImageDraw.Draw(grid)
for i, (label, im) in enumerate(cells):
    x, y = (i % cols) * cw, (i // cols) * (ch + 22)
    grid.paste(im, (x, y + 22)); d.text((x + 6, y + 4), label, fill="yellow")
grid.save(outp); print("wrote", outp)
