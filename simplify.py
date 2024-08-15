import onnx
from onnxsim import simplify
import sys

filename = sys.argv[1]

# load your predefined ONNX model
model = onnx.load(filename)

# convert model
model_simp, check = simplify(model)


assert check, "Simplified ONNX model could not be validated"

new_filename = filename.replace(".onnx", "_simplified.onnx")
print(f"New file name: {new_filename}")
onnx.save(model_simp, new_filename)