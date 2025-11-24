import sys
import onnx
import onnxruntime
import onnx_graphsurgeon as gs
import numpy as np

print("ONNX version:", onnx.__version__)
print("ONNX Runtime version:", onnxruntime.__version__)
print("ONNX GraphSurgeon version:", gs.__version__)
print("NumPy version:", np.__version__)
print("Python", sys.version)


# Работает с версиями:
# ONNX: 1.19.1
# ONNXRUNTIME: 1.23.2
# ONNX GraphSurgeon: 0.5.8
# NumPy: 2.2.6
# Python 3.13.4 (tags/v3.13.4:8a526ec, Jun  3 2025, 17:46:04) [MSC v.1943 64 bit (AMD64)]