"""Check whether DVR frames are bobbed fields (ABAB pattern, half vertical resolution) or full frames.
Prints frame-to-frame differences and vertical/horizontal detail energy, and saves a 1:1 zoomed crop.

usage: field_check.py CLIP t_seconds OUTDIR
"""
import os, sys, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import enhance
from PIL import Image

src, t, outdir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
os.makedirs(outdir, exist_ok=True)
info = enhance.probe(src)
W, H, N = info["w"], info["h"], 12
raw = subprocess.check_output([enhance.FFMPEG, "-v", "error", "-ss", str(t), "-i", src, "-frames:v", str(N),
                               "-vf", "format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"])
f = np.frombuffer(raw, np.uint8).reshape(N, H, W).astype(np.float32)

d1 = [np.abs(f[i + 1] - f[i]).mean() for i in range(N - 1)]
d2 = [np.abs(f[i + 2] - f[i]).mean() for i in range(N - 2)]
print("consecutive-frame diffs (t,t+1):", " ".join(f"{v:5.2f}" for v in d1))
print("two-apart diffs        (t,t+2):", " ".join(f"{v:5.2f}" for v in d2))
print("ratio mean(d2)/mean(d1) = %.2f   (<1 suggests ABAB field alternation, >1 normal motion)" % (np.mean(d2) / np.mean(d1)))

# OSD region is static: diff there isolates field/noise behaviour
osd = (slice(40, 260), slice(40, 520))
o1 = [np.abs(f[i + 1][osd] - f[i][osd]).mean() for i in range(N - 1)]
o2 = [np.abs(f[i + 2][osd] - f[i][osd]).mean() for i in range(N - 2)]
print("OSD region  (t,t+1):", " ".join(f"{v:5.2f}" for v in o1))
print("OSD region  (t,t+2):", " ".join(f"{v:5.2f}" for v in o2))

# vertical vs horizontal detail: energy of gradients at 1, 2, 4 pixel spacing
g = f[0]
for k in (1, 2, 4, 8):
    gv = np.abs(g[k:, :] - g[:-k, :]).mean(); gh = np.abs(g[:, k:] - g[:, :-k]).mean()
    print(f"gradient energy spacing {k}: vertical {gv:6.2f}  horizontal {gh:6.2f}  v/h {gv / gh:.2f}")

# vertical frequency spectrum: how much energy above the 576-line and 288-line limits
col = g - g.mean(axis=0, keepdims=True)
spec = np.abs(np.fft.rfft(col, axis=0)).mean(axis=1)
freqs = np.fft.rfftfreq(H, d=1.0) * H  # cycles per frame height
def band(lo, hi): return spec[(freqs >= lo) & (freqs < hi)].mean()
print(f"vertical spectrum energy: 0-144 cyc {band(0,144):.1f} | 144-288 {band(144,288):.1f} | 288-480 {band(288,480):.1f} (>288 = beyond one field) ")

# 1:1 crops, zoomed 3x nearest, of frames 0..3 for the OSD text and an image edge
for i in range(4):
    Image.fromarray(f[i].astype(np.uint8)[60:160, 40:340]).resize((900, 300), Image.NEAREST).save(os.path.join(outdir, f"osd_zoom_f{i}.png"))
strip = np.concatenate([f[i].astype(np.uint8)[60:160, 40:340] for i in range(4)], axis=0)
Image.fromarray(strip).resize((900, 1200), Image.NEAREST).save(os.path.join(outdir, "osd_zoom_4frames.png"))
print("wrote", os.path.join(outdir, "osd_zoom_4frames.png"))
