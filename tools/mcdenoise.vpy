# Temporal denoise pre-filter for analog FPV footage, run by enhance.py --mc-denoise (VapourSynth + Zsmooth).
# Methods (--arg method=...):
#   ttmpsm : TTempSmooth, motion-adaptive temporal smoothing over +-radius frames (default)
#   analog : Cnr4 chroma noise reducer (made for camcorder/VHS chroma noise) + TTempSmooth
#   dgm    : DegrainMedian (classic VHS/analog degrainer) + TTempSmooth
#   flux   : FluxSmoothST (spatio-temporal fluctuation smoother) + TTempSmooth
# Direct use:
#   vspipe -c y4m mcdenoise.vpy --arg infile=clip.mp4 --arg method=analog --arg strength=200 - | ffmpeg -i - ...
import vapoursynth as vs
from vapoursynth import core

infile = globals()["infile"]
method = globals().get("method", "ttmpsm")
strength = int(globals().get("thsad", globals().get("strength", 200)))   # 100 light .. 400 strong
radius = int(globals().get("radius", 2))
start = float(globals().get("start", 0))
duration = float(globals().get("duration", 0))

clip = core.bs.VideoSource(source=infile)
fps = clip.fps.numerator / clip.fps.denominator
if start or duration:
    first = int(round(start * fps))
    last = min(clip.num_frames, first + int(round(duration * fps))) if duration else clip.num_frames
    clip = clip.std.Trim(first=first, last=last - 1)

s = strength / 200.0          # 1.0 = default strength
r = max(1, min(radius, 7))
src = clip.resize.Bicubic(format=vs.YUV420P8)


def ttmpsm(c, k=1.0):
    t = max(1, min(256, int(round(4 * s * k)))); m = max(0, min(255, int(round(2 * s * k))))
    return core.zsmooth.TTempSmooth(c, maxr=r, thresh=[t, t + 1, t + 1], mdiff=[m, m + 1, m + 1],
                                    strength=2, scthresh=0.0, fp=True)   # scthresh 0: no scene-change plugin needed


if method == "ttmpsm":
    den = ttmpsm(src)
elif method == "analog":
    den = ttmpsm(core.zsmooth.Cnr4(src, scenechange=0))
elif method == "dgm":
    lim = 4 * s
    den = ttmpsm(core.zsmooth.DegrainMedian(src, limit=[lim, lim * 1.5, lim * 1.5], mode=[1, 1, 1]))
elif method == "flux":
    th = 7 * s
    den = ttmpsm(core.zsmooth.FluxSmoothST(src, temporal_threshold=[th, th, th], spatial_threshold=[th, th, th]), k=0.5)
else:
    raise ValueError("unknown method " + method)

den.set_output()
