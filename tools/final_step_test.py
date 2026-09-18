"""Final-step variants: CAS+bicubic (current) vs downscale + 1x SuperScale anti-aliasing model, and DeH264 pre-chain.
usage: final_step_test.py CLIP t OUTDIR
"""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import enhance
from PIL import Image, ImageDraw

src, t, outdir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
os.makedirs(outdir, exist_ok=True)
info = enhance.probe(src)


def model(key):
    return enhance.Upscaler(enhance.prepared_model(os.path.join(enhance.MODELS_DIR, enhance.MODELS[key][0]), fp16=True))


def grab(vf):
    raw = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-ss", str(t), "-i", src, "-frames:v", "1",
                                   "-vf", vf + ",format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    n = len(raw) // 3
    h = int(round((n / (info["w"] / info["h"])) ** 0.5)); w = n // h
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def ff(img, vf):
    h, w = img.shape[:2]
    out = subprocess.run([enhance.FFMPEG, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "pipe:0",
                          "-vf", vf + ",format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         input=img.tobytes(), stdout=subprocess.PIPE, check=True).stdout
    n = len(out) // 3
    oh = int(round((n / (w / h)) ** 0.5)); ow = n // oh
    return np.frombuffer(out, np.uint8).reshape(oh, ow, 3)


la, ss, dh = model("liveaction"), model("superscale"), model("deh264")
native = grab("hqdn3d=1.5:4:3:6")
up = la.run(native)  # 2560x1920
up_dh = la.run(dh.run(native))
cells = [
    ("A v2 current: cas0.4 @2x -> bicubic", ff(up, "cas=0.4,scale=1440:1080:flags=bicubic")),
    ("B area down -> SuperScale 1x", ss.run(ff(up, "scale=1440:1080:flags=area"))),
    ("C bicubic down -> SuperScale 1x", ss.run(ff(up, "scale=1440:1080:flags=bicubic"))),
    ("D cas0.4 @2x -> area -> SuperScale", ss.run(ff(up, "cas=0.4,scale=1440:1080:flags=area"))),
    ("E area -> SuperScale -> cas 0.2", ff(ss.run(ff(up, "scale=1440:1080:flags=area")), "cas=0.2")),
    ("F DeH264 -> liveaction, cas0.4 @2x -> bicubic", ff(up_dh, "cas=0.4,scale=1440:1080:flags=bicubic")),
    ("G DeH264 -> liveaction, area -> SuperScale", ss.run(ff(up_dh, "scale=1440:1080:flags=area"))),
    ("H source lanczos (reference)", ff(native, "scale=1440:1080:flags=lanczos")),
]
cw, ch = 640, 360
for name, box in [("scene", (400, 360, 1040, 720)), ("osd", (30, 40, 670, 400))]:
    grid = Image.new("RGB", (2 * cw, 4 * (ch + 22)), "black"); d = ImageDraw.Draw(grid)
    for i, (label, img) in enumerate(cells):
        x, y = (i % 2) * cw, (i // 2) * (ch + 22)
        grid.paste(Image.fromarray(img).crop(box), (x, y + 22)); d.text((x + 6, y + 4), label, fill="yellow")
    p = os.path.join(outdir, f"final_{name}_t{int(t)}.png"); grid.save(p); print("wrote", p)
