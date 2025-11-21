from pathlib import Path
from ultralytics import YOLO

# Определяем корень проекта (на два уровня вверх от этого скрипта: scripts/ -> корень проекта)
ROOT = Path(__file__).resolve().parents[1]
pt_path = str(ROOT / 'models' / 'yolov8n.pt')

# Загружаем предобученную модель YOLOv8n из папки `models/`
model = YOLO(pt_path)
# Экспортируем модель в ONNX (место сохранения зависит от реализации ultralytics)
model.export(format='onnx', opset=12, dynamic=False)
