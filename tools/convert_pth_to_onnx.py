"""Convert a PyTorch upscale model (.pth/.safetensors from OpenModelDB) to ONNX using spandrel."""
import sys, os, torch
from spandrel import ModelLoader

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".onnx"
desc = ModelLoader().load_from_file(src)
model = desc.model.eval()
print("arch:", desc.architecture.name, "| scale:", desc.scale, "| in/out ch:", desc.input_channels, desc.output_channels)
dummy = torch.rand(1, 3, 64, 64)
with torch.no_grad():
    torch.onnx.export(
        model, dummy, dst,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17, dynamo=False, do_constant_folding=True,
    )
print("wrote", dst, round(os.path.getsize(dst) / 1e6, 1), "MB")
