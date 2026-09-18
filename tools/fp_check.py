"""Check a model's fp16 (wrapped) output against fp32 on a real frame: max/mean diff and a side-by-side crop.
usage: fp_check.py CLIP t OUTDIR modelkey [prescale WxH]
"""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import enhance
from PIL import Image, ImageDraw

src, t, outdir, key = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
vf = f"scale={sys.argv[5].replace('x', ':')}:flags=lanczos" if len(sys.argv) > 5 else "null"
os.makedirs(outdir, exist_ok=True)
info = enhance.probe(src)
raw = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-ss", str(t), "-i", src, "-frames:v", "1",
                               "-vf", vf + ",format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
n = len(raw) // 3
h = int(round((n / (info["w"] / info["h"])) ** 0.5)); w = n // h
frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
path = os.path.join(enhance.MODELS_DIR, enhance.MODELS[key][0])
y32 = enhance.Upscaler(enhance.prepared_model(path, fp16=False)).run(frame)
y16 = enhance.Upscaler(enhance.prepared_model(path, fp16=True)).run(frame)
d = np.abs(y32.astype(int) - y16.astype(int))
print(f"{key}: fp16 vs fp32 max diff {d.max()}  mean diff {d.mean():.3f}  pixels differing >8: {(d.max(axis=2) > 8).mean() * 100:.2f}%")
cw, ch = 640, 360
x0, y0 = y32.shape[1] // 2 - cw // 2, y32.shape[0] // 2 - ch // 2
grid = Image.new("RGB", (2 * cw, ch + 22), "black"); dr = ImageDraw.Draw(grid)
for i, (label, img) in enumerate([("fp32", y32), ("fp16", y16)]):
    grid.paste(Image.fromarray(img[y0:y0 + ch, x0:x0 + cw]), (i * cw, 22)); dr.text((i * cw + 6, 4), f"{key} {label}", fill="yellow")
p = os.path.join(outdir, f"fp_{key}.png"); grid.save(p); print("wrote", p)
