import onnx
import sys

model = onnx.load(sys.argv[1])
onnx.checker.check_model(model)