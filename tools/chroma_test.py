"""Pre-filter variants (chroma denoise, deblock) on one frame, crops around the OSD text and a fast-motion area.
usage: chroma_test.py CLIP t OUTDIR [model=liveaction]
"""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import enhance
from PIL import Image, ImageDraw

src, t, outdir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
model = sys.argv[4] if len(sys.argv) > 4 else "liveaction"
os.makedirs(outdir, exist_ok=True)
info = enhance.probe(src)
ups = enhance.Upscaler(enhance.prepared_model(os.path.join(enhance.MODELS_DIR, enhance.MODELS[model][0]), fp16=True))


def grab(vf):
    raw = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-ss", str(t), "-i", src, "-frames:v", "1",
                                   "-vf", vf + ",format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return np.frombuffer(raw, np.uint8).reshape(info["h"], info["w"], 3)


def post(img):
    h, w = img.shape[:2]
    out = subprocess.run([enhance.FFMPEG, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "pipe:0",
                          "-vf", "cas=0.4,scale=1440:1080:flags=bicubic,format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         input=img.tobytes(), stdout=subprocess.PIPE, check=True).stdout
    return Image.fromarray(np.frombuffer(out, np.uint8).reshape(1080, 1440, 3))


variants = [
    ("light (current) hqdn3d=1.5:1.5:3:3", "hqdn3d=1.5:1.5:3:3"),
    ("chroma stronger hqdn3d=1.5:4:3:6", "hqdn3d=1.5:4:3:6"),
    ("chroma strong hqdn3d=1.5:6:3:8", "hqdn3d=1.5:6:3:8"),
    ("medium all hqdn3d=3:2.5:5:4", "hqdn3d=3:2.5:5:4"),
    ("deblock + light", "deblock=filter=weak:block=8,hqdn3d=1.5:1.5:3:3"),
    ("no prefilter", "null"),
]
regions = [("OSD text 1:1", (30, 40, 670, 400)), ("scene 1:1", (400, 400, 1040, 760))]
cw, ch = 640, 360
grid = Image.new("RGB", (2 * cw, len(variants) * (ch + 22)), "black"); d = ImageDraw.Draw(grid)
for i, (label, vf) in enumerate(variants):
    im = post(ups.run(grab(vf)))
    for j, (rname, box) in enumerate(regions):
        x, y = j * cw, i * (ch + 22)
        grid.paste(im.crop(box), (x, y + 22)); d.text((x + 6, y + 4), f"{label}  [{rname}]", fill="yellow")
p = os.path.join(outdir, f"chroma_t{int(t)}.png"); grid.save(p); print("wrote", p)
