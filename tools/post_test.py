"""Post-processing variants on one frame: sharpen order, downscale filter (anti-aliasing), double pass.
Builds a labelled 1:1 crop grid at 1080p.

usage: post_test.py CLIP t OUTDIR [model=liveaction]
"""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import enhance
from PIL import Image, ImageDraw

src, t, outdir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
model = sys.argv[4] if len(sys.argv) > 4 else "liveaction"
os.makedirs(outdir, exist_ok=True)
info = enhance.probe(src)
aw, ah = enhance.analog_size(info["w"], info["h"], info["fps"])
mp = enhance.prepared_model(os.path.join(enhance.MODELS_DIR, enhance.MODELS[model][0]), fp16=True)
ups = enhance.Upscaler(mp)


def grab(vf):
    raw = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-ss", str(t), "-i", src, "-frames:v", "1",
                                   "-vf", vf + ",format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    n = len(raw) // 3
    h = int(round((n / (info["w"] / info["h"])) ** 0.5)); w = n // h
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def ff(img, vf):
    """run an ffmpeg filter chain on an RGB numpy image, return RGB numpy image"""
    h, w = img.shape[:2]
    out = subprocess.run([enhance.FFMPEG, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "pipe:0",
                          "-vf", vf + ",format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         input=img.tobytes(), stdout=subprocess.PIPE, check=True).stdout
    n = len(out) // 3
    oh = int(round((n / (w / h)) ** 0.5)); ow = n // oh
    return np.frombuffer(out, np.uint8).reshape(oh, ow, 3)


native = grab("hqdn3d=1.5:1.5:3:3")
analog = grab(f"scale={aw}:{ah}:flags=lanczos,hqdn3d=1.5:1.5:3:3")
up_native = ups.run(native)            # 2560x1920
up_analog = ups.run(analog)            # 1536x1152
up_double = ups.run(up_analog)         # 3072x2304

variants = [
    ("A current: lanczos down, then cas 0.25", ff(up_native, "scale=1440:1080:flags=lanczos,cas=0.25")),
    ("B cas 0.25 at 2x, then lanczos down", ff(up_native, "cas=0.25,scale=1440:1080:flags=lanczos")),
    ("C lanczos down, no sharpen", ff(up_native, "scale=1440:1080:flags=lanczos")),
    ("D bicubic down + cas 0.25 (softer AA)", ff(up_native, "scale=1440:1080:flags=bicubic,cas=0.25")),
    ("E spline down + cas 0.25", ff(up_native, "scale=1440:1080:flags=spline,cas=0.25")),
    ("F cas 0.4 at 2x, then bicubic down", ff(up_native, "cas=0.4,scale=1440:1080:flags=bicubic")),
    ("G double pass at PAL res (2x, 2x), lanczos down + cas 0.25", ff(up_double, "scale=1440:1080:flags=lanczos,cas=0.25")),
    ("H unsharp 0.6 at 2x, then lanczos down", ff(up_native, "unsharp=5:5:0.6:5:5:0,scale=1440:1080:flags=lanczos")),
]
cw, ch = 640, 360
x0, y0 = 400, 360
grid = Image.new("RGB", (2 * cw, 4 * (ch + 22)), "black"); d = ImageDraw.Draw(grid)
for i, (label, img) in enumerate(variants):
    c = Image.fromarray(img).crop((x0, y0, x0 + cw, y0 + ch))
    x, y = (i % 2) * cw, (i // 2) * (ch + 22)
    grid.paste(c, (x, y + 22)); d.text((x + 6, y + 4), label, fill="yellow")
p = os.path.join(outdir, f"post_t{int(t)}.png"); grid.save(p); print("wrote", p)
# also a 2x zoom of a small area to show edge aliasing
zoom = Image.new("RGB", (2 * cw, 4 * (ch + 22)), "black"); d = ImageDraw.Draw(zoom)
for i, (label, img) in enumerate(variants):
    c = Image.fromarray(img).crop((x0 + 160, y0 + 90, x0 + 160 + cw // 2, y0 + 90 + ch // 2)).resize((cw, ch), Image.NEAREST)
    x, y = (i % 2) * cw, (i // 2) * (ch + 22)
    zoom.paste(c, (x, y + 22)); d.text((x + 6, y + 4), label + "  [2x zoom]", fill="yellow")
p = os.path.join(outdir, f"post_zoom_t{int(t)}.png"); zoom.save(p); print("wrote", p)
