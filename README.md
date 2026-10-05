# FPV analog DVR enhancer (free, AMD GPU)

Cleans up and upscales analog FPV DVR recordings (Orqa / NTSC / PAL) to 1080p using a
restoration AI model that runs on the AMD GPU through DirectML. Everything is free and
portable inside this folder.

```
ffmpeg decode + light temporal denoise
   -> AI 2x model on the GPU (ONNX Runtime, DirectML)
   -> lanczos to 1080p + FidelityFX CAS sharpen + H.264 encode, audio copied
```

## Use

Drag one or more DVR files onto `enhance.bat`, or from a terminal:

```
enhance.bat ORQA0084.MP4
enhance.bat D:\dvr\*.MP4 --pad 1920x1080
enhance.bat clip.MP4 --model vhs2hd
enhance.bat clip.MP4 --start 30 --duration 10        (quick test of a section)
```

Results land in `output\<name>_<model>_1080p.mp4`. A 4:3 source becomes 1440x1080; add
`--pad 1920x1080` for black side bars on a 16:9 canvas.

| option | default | notes |
|---|---|---|
| `--model` | `liveaction` | see models below, `--list-models` |
| `--sharpen` | `0.4` | CAS amount 0..1 applied at the 2x stage before downscaling, `0` = off |
| `--downscale` | `bicubic` | filter for the final downscale: `bicubic` smooths diagonals (anti-aliasing), `lanczos` is crisper with slight ringing |
| `--denoise` | `light` | `off`, `light`, `medium`, `strong`: temporal denoise before the AI, luma light and chroma stronger |
| `--prescale` | `none` | `auto` = downscale to the true analog res (PAL 768x576) first: 2.5x faster, a bit softer |
| `--passes` | `1` | `2` runs the model twice (more texture, slower, more risk of artifacts) |
| `--mc-denoise` | `150` | temporal denoise strength through VapourSynth before the AI: `150` sharper and calmer, `300` calmest but slightly softer, `0` off; `--mc-method dgm/analog/ttmpsm/flux` |
| `--throttle` | `1.0` | GPU duty cycle, `0.5` leaves half the GPU free so the PC stays usable |
| `--low-priority` | off | below-normal CPU priority for the whole pipeline |
| `--encoder` | `libx264` | `h264_amf` / `hevc_amf` use the GPU encoder, `libx265` for smaller files |
| `--crf` | `18` | lower = better quality, bigger file |
| `--height` | `1080` | output height |

## Models (in `models/`)

Tested on ORQA0084.MP4 (Orqa DVR, PAL analog, stored as 1280x960 @ 50 fps):

| key | what it is | verdict on analog FPV | speed at native res |
|---|---|---|---|
| `liveaction` | 2xLiveActionV1_SPAN, trained on live-action video compression, blur, halos, deinterlace jaggies | **best**: clearly sharper, natural, keeps real detail | ~20 fps |
| `vhs2hd` | 2xVHS2HD RealPLKSR, trained on PAL VHS captures | cleaner chroma noise but softer, loses detail | ~2 fps |
| `nomos-jpg` | 2xNomosUni SPAN, multi-JPEG degradation | close to source, little gain | ~20 fps |
| `nomos-otf` | 2xNomosUni compact, blur+noise+jpeg | over-painted, artifacts | ~25 fps |
| `lsdir4x` | 4xLSDIRCompactR3 | over-painted, artifacts | ~15 fps |
| `esrgan2x` | 2xNomosUni ESRGAN, multi-JPEG | a bit crisper than liveaction but crunchy texture | ~3 fps (11 fps with `--prescale auto`) |
| `esrgan4x` | 4xNomos2 ESRGAN, blur+noise+jpeg | very clean but plastic, smears grass and trees | ~1 fps (2.7 fps with `--prescale auto`) |
| `denoise1x` | 1xDeNoise RealPLKSR | 1x denoiser; `--model denoise1x+liveaction` gives cleaner edges and calmer grass, ~4x slower | ~1.3 fps native, ~4 fps at PAL res |
| (removed) | 1xDeH264 RealPLKSR | leaves a checkerboard pattern on this footage | |
| `superscale` | 1xSuperScale SPAN | anti-aliasing model for downscaled images; no visible gain over CAS + bicubic | ~60 fps |

Models can be chained with `+` (`--model denoise1x+liveaction`). The fp16 build of each model is
checked against fp32 on first use and rejected automatically if it differs (2xVHS2HD overflows in fp16
and runs in fp32).

Things checked that did not help: the 4xNomosUniDAT transformer (6 s per frame on DirectML, removed),
the 2xPublic RealPLKSR "real" model (DirectML rejects some of its ops, removed), H.264 deblocking before
the AI (no visible change), and double-passing the model (slightly more texture, not clearly better,
kept as `--passes 2`).

Things checked that did help and are now default: sharpen at the 2x stage then bicubic downscale
(crisper edges, smoother diagonals), and stronger chroma-only denoise (less rainbow fringing on OSD text).

The DVR frames were verified to be full frames, not bobbed half-resolution fields, so no field
deinterlacing is needed (`tools/field_check.py`).

## Temporal denoise through VapourSynth (on by default)

`tools/vapoursynth-portable` is a self-contained VapourSynth R80 with BestSource and the Zsmooth
filter pack, driven by `tools/mcdenoise.vpy`. `--mc-denoise` runs it as a pre-filter on the CPU at
about 150 fps, so it does not slow the pipeline; it is skipped automatically if the folder is missing.
Measured on the test clip (temporal noise = mean frame-to-frame difference on a static scene, lower is
calmer; sharpness = mean gradient on in-flight frames, higher is sharper):

| | temporal noise | sharpness |
|---|---|---|
| untouched source | 3.06 | 3.31 |
| no pre-denoise (v2) | 3.26 | 3.90 |
| DegrainMedian 150 (default) | 2.92 | 3.95 |
| DegrainMedian 300 | 2.51 | 3.76 |

150 is both sharper and calmer than no pre-denoise, because the AI sharpens cleaner edges instead of
noise. 300 is the calmest but removes some fine texture that extra CAS sharpening does not bring back.
Fast flight shows no ghosting at either setting. TTempSmooth alone and the Cnr4 chroma filter did less
than the built-in hqdn3d. Real motion-compensated MVTools is not available for this VapourSynth version
(only an old API 3 build exists), and the BM3D, dfttest2 and nlm-ispc plugins crashed on load.

Beyond this, the remaining upside would come from true multi-frame AI video models (BasicVSR++,
RealViformer, SeedVR2). They need PyTorch on the GPU, which on AMD under Windows means the ROCm preview,
and they are far heavier than the per-frame models used here.

On first use each model is wrapped once into `models/cache/` with uint8 in/out and fp16 weights, which
runs about 2x faster on RDNA3 with a maximum pixel difference of 4/255 versus fp32 (`--fp32` to disable).

To try another model from <https://openmodeldb.info>: drop its `.onnx` into `models/` and pass
`--model file.onnx`. For `.pth` / `.safetensors` models run `tools\convert_pth_to_onnx.py` first
(needs `pip install torch spandrel`, CPU torch is enough).

## Tools

- `tools/compare_models.py CLIP t1,t2 OUTDIR [models] [variants]` - labelled 1:1 crop grids to compare models and pre-filters
- `tools/sharpen_test.py CLIP t OUTDIR [model]` - CAS sharpen amounts side by side
- `tools/flicker_test.py SRC start n OUT...` - temporal flicker metric in the static OSD area
- `tools/post_test.py CLIP t OUTDIR` - sharpen order, downscale filter and double-pass variants
- `tools/chroma_test.py CLIP t OUTDIR` - denoise / deblock pre-filter variants around the OSD
- `tools/field_check.py CLIP t OUTDIR` - checks for bobbed fields vs full frames
- `tools/bench_models.py WxH model.onnx...` - GPU speed benchmark
- `tools/wrap_onnx.py in.onnx out.onnx [--fp16]` - adds GPU-side uint8 conversion (done automatically)
- `tools/convert_pth_to_onnx.py model.pth [out.onnx]` - PyTorch model to ONNX via spandrel

## Setup on another PC

`powershell -ExecutionPolicy Bypass -File setup.ps1` creates the venv, installs the Python packages,
downloads portable FFmpeg and the models. Needs Python 3.10+ and a DirectX 12 GPU (AMD, Intel or NVIDIA).
Dropping a video onto `enhance.bat` before the setup has run offers to run it first.

## Why these choices

- The Orqa DVR already upscales the ~576-line analog signal to 1280x960. Running the model at that
  native size and downscaling to 1080p gave visibly more detail than downscaling to 768x576 first.
- Per-frame models can shimmer on noisy sources, so a light temporal denoise (hqdn3d) runs before the AI.
  Measured flicker in static OSD areas was slightly lower than the untouched source.
- Output is tagged BT.709 limited range so players and editors show the colors correctly.
