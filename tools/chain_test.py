"""Test model chains on one frame: e.g. 1x denoiser -> 2x SPAN, at native or PAL res. 1:1 crop grid at 1080p.
usage: chain_test.py CLIP t OUTDIR
"""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import enhance
from PIL import Image, ImageDraw

src, t, outdir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
os.makedirs(outdir, exist_ok=True)
info = enhance.probe(src)
aw, ah = enhance.analog_size(info["w"], info["h"], info["fps"])


def model(key):
    return enhance.Upscaler(enhance.prepared_model(os.path.join(enhance.MODELS_DIR, enhance.MODELS[key][0]), fp16=True))


def grab(vf):
    raw = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-ss", str(t), "-i", src, "-frames:v", "1",
                                   "-vf", vf + ",format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    n = len(raw) // 3
    h = int(round((n / (info["w"] / info["h"])) ** 0.5)); w = n // h
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def post(img):
    h, w = img.shape[:2]
    out = subprocess.run([enhance.FFMPEG, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "pipe:0",
                          "-vf", "cas=0.4,scale=1440:1080:flags=bicubic,format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         input=img.tobytes(), stdout=subprocess.PIPE, check=True).stdout
    return Image.fromarray(np.frombuffer(out, np.uint8).reshape(1080, 1440, 3))


la, dn = model("liveaction"), model("denoise1x")
native = grab("hqdn3d=1.5:4:3:6")
analog = grab(f"scale={aw}:{ah}:flags=lanczos,hqdn3d=1.5:4:3:6")
native_nodn = grab("null")
cells = [
    ("A v2 current: liveaction native", post(la.run(native))),
    ("B denoise1x -> liveaction, native", post(la.run(dn.run(native)))),
    ("C denoise1x -> liveaction, native, no hqdn3d", post(la.run(dn.run(native_nodn)))),
    ("D denoise1x -> liveaction, PAL res", post(la.run(dn.run(analog)))),
    ("E liveaction -> denoise1x (after), native", post(dn.run(la.run(native)))),
    ("F liveaction PAL res, 2 passes", post(la.run(la.run(analog)))),
]
cw, ch = 640, 360
for name, box in [("scene", (400, 360, 1040, 720)), ("osd", (30, 40, 670, 400))]:
    grid = Image.new("RGB", (2 * cw, 3 * (ch + 22)), "black"); d = ImageDraw.Draw(grid)
    for i, (label, im) in enumerate(cells):
        x, y = (i % 2) * cw, (i // 2) * (ch + 22)
        grid.paste(im.crop(box), (x, y + 22)); d.text((x + 6, y + 4), label, fill="yellow")
    p = os.path.join(outdir, f"chain_{name}_t{int(t)}.png"); grid.save(p); print("wrote", p)
