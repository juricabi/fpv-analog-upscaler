"""Build a grid: model output at native res with different post-sharpen (CAS) amounts, 1:1 crops at 1080p."""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import enhance
from PIL import Image, ImageDraw

src, t, outdir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
model = sys.argv[4] if len(sys.argv) > 4 else "liveaction"
os.makedirs(outdir, exist_ok=True)
info = enhance.probe(src)
raw = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-ss", str(t), "-i", src, "-frames:v", "1",
                               "-vf", "format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
frame = np.frombuffer(raw, np.uint8).reshape(info["h"], info["w"], 3)
ups = enhance.Upscaler(os.path.join(enhance.MODELS_DIR, enhance.MODELS[model][0]))
up = ups.run(frame)
Image.fromarray(up).save(os.path.join(outdir, "up_full.png"))

cells = []
for cas in [0.0, 0.25, 0.5]:
    vf = "scale=1440:1080:flags=lanczos" + (f",cas={cas}" if cas > 0 else "")
    out = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-i", os.path.join(outdir, "up_full.png"),
                                   "-vf", vf + ",format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    im = Image.fromarray(np.frombuffer(out, np.uint8).reshape(1080, 1440, 3))
    cells.append((f"{model} native, cas={cas}", im.crop((400, 360, 1040, 720))))
srcim = Image.fromarray(frame).resize((1440, 1080), Image.LANCZOS)
cells.insert(0, ("source lanczos", srcim.crop((400, 360, 1040, 720))))
grid = Image.new("RGB", (1280, 2 * 382), "black"); d = ImageDraw.Draw(grid)
for i, (label, c) in enumerate(cells):
    x, y = (i % 2) * 640, (i // 2) * 382
    grid.paste(c, (x, y + 22)); d.text((x + 6, y + 4), label, fill="yellow")
p = os.path.join(outdir, f"sharpen_t{int(t)}.png"); grid.save(p); print("wrote", p)
