"""Run selected models on a few frames of a clip and build labelled 1:1 crop grids at 1080p scale.

usage: compare_models.py CLIP  t1,t2  OUTDIR  [models=liveaction,vhs2hd] [variants=native,analog,analog+dn] [cx,cy]
"""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import enhance
from PIL import Image, ImageDraw

src = sys.argv[1]
times = [float(t) for t in sys.argv[2].split(",")]
outdir = sys.argv[3]
model_keys = (sys.argv[4] if len(sys.argv) > 4 else "liveaction,vhs2hd").split(",")
variant_keys = (sys.argv[5] if len(sys.argv) > 5 else "native,analog,analog+dn").split(",")
center = [float(v) for v in sys.argv[6].split(",")] if len(sys.argv) > 6 else (0.5, 0.45)
crop_w, crop_h = 640, 360
os.makedirs(outdir, exist_ok=True)
info = enhance.probe(src)
aw, ah = enhance.analog_size(info["w"], info["h"], info["fps"])

VARIANTS = {
    "native": "null",
    "native+dn": "hqdn3d=1.5:1.5:3:3",
    "analog": f"scale={aw}:{ah}:flags=lanczos",
    "analog+dn": f"scale={aw}:{ah}:flags=lanczos,hqdn3d=1.5:1.5:3:3",
    "analog+dn2": f"scale={aw}:{ah}:flags=lanczos,hqdn3d=3:2.5:5:4",
}


def grab(t, vf):
    raw = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-ss", str(t), "-i", src, "-frames:v", "1",
                                   "-vf", vf + ",format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    n = len(raw) // 3
    h = int(round((n / (info["w"] / info["h"])) ** 0.5)); w = n // h
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def to_1080(img):
    out_w = int(round(1080 * img.shape[1] / img.shape[0] / 2.0)) * 2
    return Image.fromarray(img).resize((out_w, 1080), Image.LANCZOS)


def crop(im):
    cx, cy = int(im.width * center[0]), int(im.height * center[1])
    x0 = max(0, min(im.width - crop_w, cx - crop_w // 2)); y0 = max(0, min(im.height - crop_h, cy - crop_h // 2))
    return im.crop((x0, y0, x0 + crop_w, y0 + crop_h))


models = {k: enhance.Upscaler(os.path.join(enhance.MODELS_DIR, enhance.MODELS[k][0])) for k in model_keys}

for t in times:
    cells = [("source (lanczos to 1080p)", crop(to_1080(grab(t, "null"))))]
    for vname in variant_keys:
        base = grab(t, VARIANTS[vname])
        for mk, ups in models.items():
            cells.append((f"{mk}  [{vname}]", crop(to_1080(ups.run(base)))))
    cols = 2; rows = (len(cells) + 1) // 2
    grid = Image.new("RGB", (cols * crop_w, rows * (crop_h + 22)), "black")
    d = ImageDraw.Draw(grid)
    for i, (label, c) in enumerate(cells):
        x, y = (i % cols) * crop_w, (i // cols) * (crop_h + 22)
        grid.paste(c, (x, y + 22)); d.text((x + 6, y + 4), label, fill="yellow")
    p = os.path.join(outdir, f"cmp_t{int(t)}.png"); grid.save(p); print("wrote", p)
