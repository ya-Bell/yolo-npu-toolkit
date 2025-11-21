from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
pt_path = str(ROOT / 'models' / 'yolo11n.pt')

model = YOLO(pt_path)
model.export(format='onnx', opset=12, dynamic=False)
