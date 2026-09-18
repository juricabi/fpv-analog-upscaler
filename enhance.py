#!/usr/bin/env python3
"""
enhance.py - free AI enhancer for analog FPV DVR footage (Orqa / NTSC / PAL) on AMD GPUs.

Pipeline:  ffmpeg (decode, downscale to true analog res, light temporal denoise)
           -> ONNX Runtime on DirectML (AI 2x restoration model on the AMD GPU)
           -> ffmpeg (lanczos to 1080p, optional sharpen, encode, copy audio)

Usage:
  python enhance.py ORQA0084.MP4                 # -> output/ORQA0084_<model>_1080p.mp4
  python enhance.py *.MP4 --model vhs2hd          # best quality, slow
  python enhance.py clip.mp4 --start 60 --duration 10 --model liveaction
  python enhance.py --list-models
"""
import argparse, json, os, queue, subprocess, sys, threading, time
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = os.path.join(HERE, "tools", "ffmpeg", "bin", "ffmpeg.exe")
FFPROBE = os.path.join(HERE, "tools", "ffmpeg", "bin", "ffprobe.exe")
MODELS_DIR = os.path.join(HERE, "models")

MODELS = {
    "liveaction": ("2xLiveActionV1_SPAN.onnx",
                   "SPAN trained on live-action video: fixes compression, blur, halos, keeps real detail. Fast."),
    "vhs2hd":     ("2xVHS2HD.onnx",
                   "RealPLKSR trained on PAL VHS: dehalo, chroma noise, ghosting. Best cleanup, ~15x slower."),
    "nomos-otf":  ("2xNomosUni_compact_otf_medium.onnx",
                   "Compact model trained with blur+noise+jpeg degradations. Strong denoise, fast."),
    "nomos-jpg":  ("2xNomosUni_span_multijpg.onnx",
                   "SPAN trained on multi-pass JPEG compression. Fast."),
    "lsdir4x":    ("4xLSDIRCompactR3.onnx",
                   "Compact 4x trained with realistic degradations. Fast."),
    "esrgan2x":   ("2xNomosUni_esrgan_multijpg.onnx",
                   "ESRGAN (RRDB) 2x, multi-JPEG degradations. Heavier network, slower."),
    "esrgan4x":   ("4xNomos2_otf_esrgan.onnx",
                   "ESRGAN (RRDB) 4x, blur+noise+jpeg degradations. Heavier network, slower."),
    "denoise1x":  ("1xDeNoise_realplksr_otf.onnx",
                   "RealPLKSR 1x denoiser (no upscale), can be chained. Slow."),
    "superscale": ("1xSuperScale_SPAN.onnx",
                   "SPAN 1x anti-aliasing sharpener meant for downscaled images (final step). Fast."),
}

DENOISE = {  # hqdn3d = luma_spatial:chroma_spatial:luma_temporal:chroma_temporal
    # analog chroma is low-res and noisy (rainbow fringes), luma carries the detail: keep luma light, chroma stronger
    "off": None,
    "light": "hqdn3d=1.5:4:3:6",
    "medium": "hqdn3d=3:5:5:7",
    "strong": "hqdn3d=5:6:8:8",
}


def probe(path):
    out = subprocess.check_output([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", path])
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = v.get("avg_frame_rate", "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    duration = float(info["format"].get("duration", 0) or 0)
    nb = int(v.get("nb_frames") or 0) or int(round(duration * fps))
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": fps, "fps_str": v.get("avg_frame_rate", "50/1"),
            "frames": nb, "duration": duration, "audio": has_audio}


def resolve_model(name):
    if name in MODELS:
        return os.path.join(MODELS_DIR, MODELS[name][0]), name
    if os.path.isfile(name):
        return name, os.path.splitext(os.path.basename(name))[0]
    p = os.path.join(MODELS_DIR, name)
    if os.path.isfile(p):
        return p, os.path.splitext(name)[0]
    sys.exit(f"model not found: {name}  (use --list-models)")


class Chain:
    """Several models run one after another on each frame, e.g. 'denoise1x+liveaction'."""
    def __init__(self, upscalers):
        self.ups = upscalers
        self.scale = 1
        for u in upscalers:
            self.scale *= u.scale
        self.provider = upscalers[0].provider

    def run(self, frame_u8):
        for u in self.ups:
            frame_u8 = u.run(frame_u8)
        return frame_u8


def prepared_model(model_path, fp16=True):
    """Return a cached variant of the model that takes/returns uint8 HxWx3 directly (conversion runs on the GPU),
    optionally with fp16 weights (about 2x faster on RDNA3, max 4/255 difference)."""
    cache_dir = os.path.join(MODELS_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    dst = os.path.join(cache_dir, f"{stem}_u8_{'fp16' if fp16 else 'fp32'}.onnx")
    if not os.path.isfile(dst) or os.path.getmtime(dst) < os.path.getmtime(model_path):
        print(f"  preparing {os.path.basename(dst)} (one-time) ...")
        cmd = [sys.executable, os.path.join(HERE, "tools", "wrap_onnx.py"), model_path, dst] + (["--fp16"] if fp16 else [])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return dst


def sample_frame(src, args):
    """One pre-filtered frame from the middle of the clip, used to validate fp16 on real content."""
    try:
        info = probe(src)
        pre = build_prefilter(args, info)
        raw = subprocess.check_output([FFMPEG, "-v", "error", "-ss", str(max(0.0, info["duration"] / 2)), "-i", src,
                                       "-frames:v", "1", "-vf", pre, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        ratio = info["w"] / info["h"]
        h = int(round((len(raw) / 3 / ratio) ** 0.5)); w = len(raw) // 3 // h
        return np.frombuffer(raw, np.uint8).reshape(h, w, 3)
    except Exception:
        return None


def fp16_ok(model_path, device, sample):
    """fp16 halves the GPU time but some models overflow (2xVHS2HD outputs garbage). Compare fp16 with fp32 on a
    real frame once per model and remember the verdict next to the cache."""
    if device != "dml" or sample is None:
        return False
    stem = os.path.splitext(os.path.basename(model_path))[0]
    verdict = os.path.join(MODELS_DIR, "cache", stem + ".fp16_verdict")
    if os.path.isfile(verdict) and os.path.getmtime(verdict) > os.path.getmtime(model_path):
        return open(verdict).read().strip() == "ok"
    p16 = prepared_model(model_path, fp16=True)
    p32 = prepared_model(model_path, fp16=False)
    ok = False
    if p16.endswith("_fp16.onnx"):
        y16 = Upscaler(p16, device).run(sample)
        y32 = Upscaler(p32, device).run(sample)
        d = np.abs(y16.astype(np.int16) - y32.astype(np.int16))
        ok = bool(d.max() <= 24 and d.mean() <= 1.0)
        print(f"  fp16 check {stem}: max diff {d.max()}, mean {d.mean():.2f} -> {'ok' if ok else 'rejected, using fp32'}")
    with open(verdict, "w") as f:
        f.write("ok" if ok else "bad")
    return ok


class Upscaler:
    def __init__(self, model_path, device="dml"):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if device == "dml" else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, so, providers=providers)
        self.provider = self.sess.get_providers()[0]
        i = self.sess.get_inputs()[0]
        self.inp = i.name
        self.u8 = "uint8" in i.type           # wrapped model: uint8 NHWC in/out
        self.fp16 = "float16" in i.type
        if self.u8:
            probe_in = np.zeros((1, 32, 32, 3), dtype=np.uint8)
            self.scale = self.sess.run(None, {self.inp: probe_in})[0].shape[1] // 32
        else:
            probe_in = np.zeros((1, 3, 32, 32), dtype=np.float16 if self.fp16 else np.float32)
            self.scale = self.sess.run(None, {self.inp: probe_in})[0].shape[2] // 32

    def run(self, frame_u8):  # HxWx3 uint8 -> (H*s)x(W*s)x3 uint8
        if self.u8:
            return self.sess.run(None, {self.inp: frame_u8[None]})[0][0]
        x = frame_u8.transpose(2, 0, 1)[None].astype(np.float16 if self.fp16 else np.float32)
        x *= (1.0 / 255.0)
        y = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0][0]
        y = y.astype(np.float32) if self.fp16 else y
        np.multiply(y, 255.0, out=y)
        np.clip(y, 0, 255, out=y)
        return np.ascontiguousarray(y.transpose(1, 2, 0).astype(np.uint8))


def analog_size(src_w, src_h, fps):
    """True resolution of the analog signal (square pixels): PAL 576 lines, NTSC 480 lines."""
    lines = 576 if 45 <= fps < 55 else 480
    w = int(round(lines * src_w / src_h / 2.0)) * 2
    return w, lines


def build_prefilter(args, info):
    f = []
    if args.prescale == "auto":
        if info["h"] > 620:  # DVR stored an upscaled frame -> bring it back to analog resolution
            w, h = analog_size(info["w"], info["h"], info["fps"])
            f.append(f"scale={w}:{h}:flags=lanczos")
    elif args.prescale not in ("none", "off"):
        w, h = args.prescale.lower().split("x")
        f.append(f"scale={w}:{h}:flags=lanczos")
    d = DENOISE.get(args.denoise, args.denoise if args.denoise.startswith("hqdn3d") else None)
    if d:
        f.append(d)
    f.append("format=rgb24")
    return ",".join(f)


def build_postfilter(args, in_w, in_h):
    f = []
    out_h = args.height
    out_w = int(round(out_h * in_w / in_h / 2.0)) * 2
    # sharpen at the model's 2x resolution, then downscale: the downscale averages away halos and
    # stair-stepping on diagonals (supersampled anti-aliasing) while keeping the edge contrast
    if args.sharpen > 0 and not args.sharpen_last:
        f.append(f"cas={args.sharpen}")
    if (out_w, out_h) != (in_w, in_h):
        f.append(f"scale={out_w}:{out_h}:flags={args.downscale}")
    if args.sharpen > 0 and args.sharpen_last:
        f.append(f"cas={args.sharpen}")
    if args.pad:
        pw, ph = args.pad.lower().split("x")
        f.append(f"pad={pw}:{ph}:(ow-iw)/2:(oh-ih)/2")
    f.append("scale=out_color_matrix=bt709:out_range=tv,format=yuv420p")
    return ",".join(f)


def encoder_args(args):
    q = args.crf
    return {
        "libx264": ["-c:v", "libx264", "-preset", args.preset, "-crf", str(q), "-profile:v", "high"],
        "libx265": ["-c:v", "libx265", "-preset", args.preset, "-crf", str(q + 2), "-tag:v", "hvc1"],
        "h264_amf": ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q), "-qp_b", str(q)],
        "hevc_amf": ["-c:v", "hevc_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q)],
    }[args.encoder]


VSPIPE = os.path.join(HERE, "tools", "vapoursynth-portable", "vspipe.bat")
MCDENOISE_VPY = os.path.join(HERE, "tools", "mcdenoise.vpy")


def start_decoder(src, args, pre, frames_limit=0):
    """Returns (procs, stdout) producing raw rgb24 frames. With --mc-denoise the frames come from VapourSynth
    (MVTools motion-compensated denoise) piped through ffmpeg for the remaining pre-filters."""
    tail = ["-vf", pre, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    if frames_limit:
        tail = ["-frames:v", str(frames_limit)] + tail
    if args.mc_denoise > 0:
        vs_cmd = ["cmd", "/c", VSPIPE, "-c", "y4m", MCDENOISE_VPY, "--arg", f"infile={os.path.abspath(src)}",
                  "--arg", f"strength={args.mc_denoise}", "--arg", f"radius={args.mc_radius}", "--arg", f"method={args.mc_method}",
                  "--arg", f"start={args.start}", "--arg", f"duration={args.duration}"]
        if frames_limit:
            vs_cmd += ["--end", str(frames_limit - 1)]
        vs_cmd += ["-"]
        vs_log = open(os.path.join(os.environ.get("TEMP", HERE), "vspipe_last.log"), "w")
        vsp = subprocess.Popen(vs_cmd, stdout=subprocess.PIPE, stderr=vs_log)
        dec = subprocess.Popen([FFMPEG, "-v", "error", "-i", "pipe:0"] + tail, stdin=vsp.stdout, stdout=subprocess.PIPE)
        vsp.stdout.close()
        return [vsp, dec], dec.stdout
    head = [FFMPEG, "-v", "error", "-nostdin"]
    if args.start:
        head += ["-ss", str(args.start)]
    head += ["-i", src]
    if args.duration:
        head += ["-t", str(args.duration)]
    dec = subprocess.Popen(head + tail, stdout=subprocess.PIPE)
    return [dec], dec.stdout


def process(src, dst, args, ups, model_name):
    info = probe(src)
    pre = build_prefilter(args, info)

    # find the frame size after the pre-filter by decoding one frame
    procs, out = start_decoder(src, args, pre, frames_limit=1)
    hdr = out.read()
    for p in procs:
        p.wait()
    if not hdr:
        sys.exit("decoder produced no frame" + (" (see vspipe_last.log in %TEMP%)" if args.mc_denoise > 0 else ""))
    ratio = info["w"] / info["h"]
    h_in = int(round((len(hdr) / 3 / ratio) ** 0.5))
    w_in = len(hdr) // 3 // h_in
    assert w_in * h_in * 3 == len(hdr), "could not infer pre-filtered frame size"
    total_scale = ups.scale ** args.passes
    w_out, h_out = w_in * total_scale, h_in * total_scale
    post = build_postfilter(args, w_out, h_out)

    total = info["frames"]
    if args.duration:
        total = min(total, int(round(args.duration * info["fps"])))

    enc_cmd = [FFMPEG, "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w_out}x{h_out}", "-r", info["fps_str"], "-i", "pipe:0"]
    if info["audio"] and not args.no_audio:
        if args.start:
            enc_cmd += ["-ss", str(args.start)]
        enc_cmd += ["-i", src]
        if args.duration:
            enc_cmd += ["-t", str(args.duration)]
        enc_cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy"]
    enc_cmd += ["-vf", post] + encoder_args(args)
    enc_cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                "-movflags", "+faststart", "-shortest", dst]

    print(f"\n{os.path.basename(src)}  {info['w']}x{info['h']} @ {info['fps']:.2f} fps, {info['duration']:.0f}s, {total} frames")
    print(f"  pre : {('[vapoursynth %s %d] ' % (args.mc_method, args.mc_denoise)) if args.mc_denoise > 0 else ''}{pre}")
    print(f"  ai  : {model_name} x{ups.scale}{' x%d passes' % args.passes if args.passes > 1 else ''} on {ups.provider}  ({w_in}x{h_in} -> {w_out}x{h_out})")
    print(f"  post: {post}")
    print(f"  enc : {' '.join(encoder_args(args))}")
    print(f"  out : {dst}")

    dec_procs, dec_out = start_decoder(src, args, pre)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, bufsize=w_out * h_out * 3 * 4)
    q_in, q_out = queue.Queue(maxsize=12), queue.Queue(maxsize=12)
    nbytes = w_in * h_in * 3

    def reader():
        while True:
            buf = dec_out.read(nbytes)
            if len(buf) < nbytes:
                break
            q_in.put(np.frombuffer(buf, np.uint8).reshape(h_in, w_in, 3))
        q_in.put(None)

    def writer():
        while True:
            item = q_out.get()
            if item is None:
                break
            enc.stdin.write(item.tobytes())
        enc.stdin.close()

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=writer, daemon=True).start()

    t0 = time.time(); n = 0; last = 0.0
    t_wait_in = t_infer = t_wait_out = t_sleep = 0.0
    idle_ratio = (1.0 / args.throttle - 1.0) if 0 < args.throttle < 1 else 0.0  # GPU duty cycle
    while True:
        ta = time.perf_counter()
        frame = q_in.get()
        if frame is None:
            break
        tb = time.perf_counter()
        out = ups.run(frame)
        for _ in range(args.passes - 1):
            out = ups.run(out)
        tc = time.perf_counter()
        if idle_ratio:
            time.sleep((tc - tb) * idle_ratio)
        ts = time.perf_counter()
        q_out.put(out)
        td = time.perf_counter()
        t_wait_in += tb - ta; t_infer += tc - tb; t_sleep += ts - tc; t_wait_out += td - ts
        n += 1
        now = time.time()
        if now - last > 1.0 or n == total:
            fps = n / (now - t0)
            eta = (total - n) / fps if fps > 0 and total else 0
            sys.stdout.write(f"\r  {n}/{total} frames  {fps:5.1f} fps  ETA {int(eta // 60):02d}:{int(eta % 60):02d}   ")
            sys.stdout.flush()
            last = now
    q_out.put(None)
    for p in dec_procs:
        p.wait()
    enc.wait()
    dt = time.time() - t0
    print(f"\n  done: {n} frames in {dt / 60:.1f} min ({n / dt:.1f} fps), encoder exit {enc.returncode}")
    print(f"  time: waiting for decoder {t_wait_in:.1f}s | AI inference {t_infer:.1f}s | throttle idle {t_sleep:.1f}s | waiting for encoder {t_wait_out:.1f}s")
    return enc.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="video files or folders")
    ap.add_argument("--model", "-m", default="liveaction", help="model key, file in models/, or path (default: liveaction). "
                    "Chain models with +, e.g. denoise1x+liveaction (cleaner, ~4x slower)")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--out", "-o", default=os.path.join(HERE, "output"), help="output folder (default: output/)")
    ap.add_argument("--height", type=int, default=1080, help="output height (default 1080)")
    ap.add_argument("--pad", default="", help="pad to canvas e.g. 1920x1080 (black bars for 4:3 sources)")
    ap.add_argument("--prescale", default="none", help="none | auto | WxH  (auto = downscale to true analog res first, PAL 768x576 / NTSC 640x480; faster, slightly softer)")
    ap.add_argument("--denoise", default="light", help="off | light | medium | strong | hqdn3d=a:b:c:d  (temporal denoise before the AI, default light)")
    ap.add_argument("--mc-denoise", type=int, default=150, help="temporal denoise strength via VapourSynth (DegrainMedian + TTempSmooth) before the AI: 150 = sharper and calmer than none (default), 300 = calmest but slightly softer, 0 = off (skipped if tools/vapoursynth-portable is missing)")
    ap.add_argument("--mc-radius", type=int, default=2, help="frames each side for --mc-denoise (1..3, default 2)")
    ap.add_argument("--mc-method", default="dgm", help="--mc-denoise method: dgm (DegrainMedian + TTempSmooth, default) | analog (Cnr4 chroma + TTempSmooth) | ttmpsm | flux")
    ap.add_argument("--sharpen", type=float, default=0.4, help="FidelityFX CAS amount 0..1, applied at the 2x stage before downscaling (default 0.4, 0 = off)")
    ap.add_argument("--sharpen-last", action="store_true", help="apply the sharpen after the downscale instead (old behaviour, more aliasing)")
    ap.add_argument("--downscale", default="bicubic", help="downscale filter to 1080p: bicubic (smooth edges, default) | lanczos (crisper, slight ringing) | spline")
    ap.add_argument("--passes", type=int, default=1, help="run the AI model N times in a row (2 = more texture, slower, riskier)")
    ap.add_argument("--encoder", default="libx264", choices=["libx264", "libx265", "h264_amf", "hevc_amf"])
    ap.add_argument("--preset", default="slow", help="x264/x265 preset (default slow)")
    ap.add_argument("--crf", type=int, default=18, help="quality: lower = better/bigger (default 18)")
    ap.add_argument("--start", type=float, default=0, help="start at N seconds (testing)")
    ap.add_argument("--duration", type=float, default=0, help="process only N seconds (testing)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    ap.add_argument("--throttle", type=float, default=1.0, help="GPU duty cycle 0.1..1.0, e.g. 0.5 leaves half the GPU free for other work (default 1.0)")
    ap.add_argument("--low-priority", action="store_true", help="run at below-normal CPU priority (encoder too)")
    ap.add_argument("--fp32", action="store_true", help="use fp32 weights (2x slower, reference quality)")
    ap.add_argument("--raw-model", action="store_true", help="run the original ONNX as-is (no uint8/fp16 wrapping)")
    ap.add_argument("--suffix", default="", help="extra output name suffix")
    args = ap.parse_args()

    if args.list_models:
        for k, (f, d) in MODELS.items():
            ok = "  " if os.path.isfile(os.path.join(MODELS_DIR, f)) else "!!"
            print(f"{ok} {k:12s} {f:40s} {d}")
        return

    files = []
    for p in args.inputs:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".ts"))]
        else:
            files.append(p)
    if not files:
        ap.print_help(); sys.exit(1)

    if args.mc_denoise > 0 and not os.path.isfile(VSPIPE):
        print("  note: tools/vapoursynth-portable not found, temporal denoise (--mc-denoise) skipped")
        args.mc_denoise = 0

    if args.low_priority and os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL, inherited by the ffmpeg children

    device = "cpu" if args.cpu else "dml"
    sample = None if args.raw_model or args.fp32 else sample_frame(files[0], args)
    stages, names = [], []
    for part in args.model.split("+"):
        model_path, model_name = resolve_model(part.strip())
        if not args.raw_model:
            use16 = not args.fp32 and fp16_ok(model_path, device, sample)
            model_path = prepared_model(model_path, fp16=use16)
        stage = Upscaler(model_path, device)
        stages.append(stage)
        names.append(model_name)
        if sample is not None:
            sample = stage.run(sample)  # next stage is validated on what it will actually receive
    ups = stages[0] if len(stages) == 1 else Chain(stages)
    model_name = "+".join(names)
    os.makedirs(args.out, exist_ok=True)
    ok = True
    for src in files:
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(args.out, f"{stem}_{model_name}{args.suffix}_{args.height}p.mp4")
        ok &= process(src, dst, args, ups, model_name)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
