import onnx
from onnxsim import simplify

filename = "weights/test.onnx"

# load your predefined ONNX model
model = onnx.load(filename)

# convert model
model_simp, check = simplify(model)


assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simp, "weights/test_simplified.onnx")