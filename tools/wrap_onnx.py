"""Wrap an upscale ONNX model so it takes/returns uint8 HxWx3 images directly (all pre/post-processing on the GPU),
optionally converting the core network to fp16. The fp16 build is checked against fp32 on a noise image and
rejected (fp32 kept) if it does not load or its output differs too much (some models overflow in fp16).

usage: wrap_onnx.py in.onnx out.onnx [--fp16]
"""
import sys, os, onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
want_fp16 = "--fp16" in sys.argv


def build(fp16):
    m = onnx.load(src)
    in_type = m.graph.input[0].type.tensor_type.elem_type
    already_fp16 = in_type == TensorProto.FLOAT16
    if fp16 and not already_fp16:
        try:
            from onnxruntime.transformers.float16 import convert_float_to_float16
        except ImportError:
            from onnxconverter_common.float16 import convert_float_to_float16
        m = convert_float_to_float16(m, keep_io_types=False)
    core = TensorProto.FLOAT16 if (fp16 or already_fp16) else TensorProto.FLOAT
    np_dtype = np.float16 if core == TensorProto.FLOAT16 else np.float32
    g = m.graph
    in_name, out_name = g.input[0].name, g.output[0].name
    g.initializer.extend([
        numpy_helper.from_array(np.array(1.0 / 255.0, dtype=np_dtype), "pre_scale"),
        numpy_helper.from_array(np.array(255.0, dtype=np_dtype), "post_scale"),
        numpy_helper.from_array(np.array(0.0, dtype=np_dtype), "clip_min"),
        numpy_helper.from_array(np.array(255.0, dtype=np_dtype), "clip_max"),
    ])
    pre = [
        helper.make_node("Cast", ["image_u8"], ["pre_f"], to=core),
        helper.make_node("Transpose", ["pre_f"], ["pre_t"], perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["pre_t", "pre_scale"], [in_name]),
    ]
    post = [
        helper.make_node("Mul", [out_name, "post_scale"], ["post_m"]),
        helper.make_node("Clip", ["post_m", "clip_min", "clip_max"], ["post_c"]),
        helper.make_node("Round", ["post_c"], ["post_r"]),
        helper.make_node("Transpose", ["post_r"], ["post_t"], perm=[0, 2, 3, 1]),
        helper.make_node("Cast", ["post_t"], ["image_out_u8"], to=TensorProto.UINT8),
    ]
    nodes = pre + list(g.node) + post
    del g.node[:]
    g.node.extend(nodes)
    del g.input[:]
    g.input.append(helper.make_tensor_value_info("image_u8", TensorProto.UINT8, ["batch", "height", "width", 3]))
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("image_out_u8", TensorProto.UINT8, ["batch", "height_out", "width_out", 3]))
    return m, core == TensorProto.FLOAT16


def run(path, x):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    s = ort.InferenceSession(path, so, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    return s.run(None, {s.get_inputs()[0].name: x})[0]


rng = np.random.default_rng(0)
test = (rng.random((1, 96, 128, 3)) * 255).astype(np.uint8)
# smooth part + noise part, like real video
test[:, :48] = (np.linspace(30, 220, 128, dtype=np.uint8)[None, None, :, None] + rng.integers(0, 12, (1, 48, 128, 3))).clip(0, 255)

m32, _ = build(False)
tmp32 = dst + ".fp32.tmp.onnx"
onnx.save(m32, tmp32)
ref = run(tmp32, test)

chosen = None
if want_fp16:
    try:
        m16, _ = build(True)
        onnx.save(m16, dst)
        out = run(dst, test)
        d = np.abs(out.astype(int) - ref.astype(int))
        if d.max() > 24 or d.mean() > 1.0:
            raise RuntimeError(f"fp16 output differs from fp32 (max {d.max()}, mean {d.mean():.2f})")
        chosen = "fp16"
    except Exception as e:
        print("fp16 rejected (%s), using fp32" % str(e).splitlines()[0][:120])
        if os.path.exists(dst):
            os.remove(dst)
if chosen is None:
    os.replace(tmp32, dst)
    chosen = "fp32"
else:
    os.remove(tmp32)
print("wrote", dst, chosen)
