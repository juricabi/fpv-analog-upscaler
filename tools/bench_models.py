"""Benchmark ONNX models on the GPU (wrapped uint8 + fp16) at given sizes.
usage: bench_models.py WxH[,WxH...] model1.onnx [model2.onnx ...]
"""
import os, sys, time, subprocess, numpy as np, onnxruntime as ort
HERE = os.path.dirname(os.path.abspath(__file__))
sizes = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1].split(",")]
so = ort.SessionOptions(); so.log_severity_level = 3
tmp = os.environ.get("TMP", ".")
for m in sys.argv[2:]:
    stem = os.path.splitext(os.path.basename(m))[0]
    wrapped = os.path.join(tmp, stem + "_bench_u8fp16.onnx")
    try:
        subprocess.run([sys.executable, os.path.join(HERE, "wrap_onnx.py"), m, wrapped, "--fp16"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        s = ort.InferenceSession(wrapped, so, providers=["DmlExecutionProvider"])
        name = s.get_inputs()[0].name
        for (w, h) in sizes:
            x = np.random.randint(0, 256, (1, h, w, 3), dtype=np.uint8)
            y = s.run(None, {name: x})[0]
            n = 3; t = time.time()
            for _ in range(n): s.run(None, {name: x})
            ms = (time.time() - t) / n * 1000
            print(f"{stem:50s} @{w}x{h} -> {y.shape[2]}x{y.shape[1]} : {ms:7.1f} ms/frame ({1000 / ms:5.1f} fps)")
    except Exception as e:
        print(f"{stem:50s} FAILED: {str(e)[:200]}")
