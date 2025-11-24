# Проект: YOLO → NPU Toolkit

Коротко
-------
Этот проект преобразует YOLO ONNX-модель в формат, удобный для NPU: вместо одного большого
flatten-выхода модель выдаёт по каждому детектору три карты (`feat64`, `cls80`, `conf1`) для
трёх масштабов (80×80, 40×40, 20×20). Это даёт лучшую оптимизацию компилятора под аппаратные
ускорители.

Файлы и структура
------------------
- `edit_model.py`: основной скрипт — делает трансформацию и может протестировать
  результат через `onnxruntime`.
- Модифицированные модели по умолчанию сохраняются в папку `mod_onnx/` как `<orig_name>_mod.onnx`.
- `scripts/`: вспомогательные скрипты. Активная логика в
  корневом `edit_model.py`.
- `scripts/export_model.py` — скрипт для экспорта PyTorch-моделей YOLO в форматы .pt и .onnx. 
  По умолчанию файл .pt сохраняется в папку `models/`, а файл .onnx — в папку `onnx/`. 
  Лишние файлы .pt удаляются из корня после завершения работы.  
  Модели экспортируются с параметрами: `format="onnx"`, `opset=12`, `dynamic=False`.
- `scripts/checking_version.py` — выводит текущие версии Python и зависимостей (ONNX, ONNX Runtime, ONNX GraphSurgeon, NumPy).
- `requirements.txt` — список необходимых пакетов и их версий. 

Quick Start
-----------
1) Создайте виртуальное окружение и установите зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Преобразование и тест (пример для `yolov8n.onnx`):

```powershell
python "d:\yolo-npu-toolkit\edit_model.py" -m "d:\yolo-npu-toolkit\onnx\yolov8n.onnx" --test
```

3) Указать конкретный путь для сохранения модифицированной модели:

```powershell
python "d:\yolo-npu-toolkit\edit_model.py" -m "d:\yolo-npu-toolkit\onnx\yolov8n.onnx" -o "d:\yolo-npu-toolkit\mod_onnx\yolov8n_mod.onnx" --test
```

Input
-----
- **Тип:** `float32` тензор изображения, формат NCHW (обычно `[1,3,H,W]`).
- **Форма:** скрипт использует первый вход модели (`sess.get_inputs()[0]`). Динамические размеры при тесте заменяются на `1`.
- **Нормализация:** значения зависимы от модели/пайплайна (часто в диапазоне `[0,1]` или `[0,255]`); приведите вход к нужному формату перед компиляцией/инференсом.
- **Как проверить быстро:** запустите команду, чтобы увидеть имя и форму входа:

```powershell
python - <<'PY'
import onnxruntime as ort
sess = ort.InferenceSession(r"d:\yolo-npu-toolkit\yolov8n.onnx")
inp = sess.get_inputs()[0]
print("name:", inp.name, "shape:", inp.shape, "type:", inp.type)
PY
```
- **Пример:** если вход модели `[1,3,640,640]`, то при `--test` скрипт создаст случайный тензор формы `[1,3,640,640]` и выполнит инференс.

Что делает скрипт
-----------------
- Находит Concat из трёх Reshape с формой `[1,144,-1]`.
- Для каждой ветки восстанавливает `[1,144,H,W]`, затем создаёт:
  - `feat64` — Slice(0:64) → [1,64,H,W]
  - `cls80`  — Slice(64:144) → [1,80,H,W]
  - `conf1`  — Sigmoid(cls80) → ReduceSum(axes=[1],keepdims=1) → Clip → [1,1,H,W]
- Обновляет `graph.outputs` и сохраняет модель в `mod_onnx/` (если не указан `-o`).

Параметры и советы
------------------
- `-m, --model`: путь к исходной ONNX-модели (обязательно).
- `-o, --out`: путь для сохранения модифицированной модели (по умолчанию `mod_onnx/<name>_mod.onnx`).
- `--test`: проверить модифицированную модель через `onnxruntime` (создаёт случайный тестовый ввод).

Проверка
-------
Для проверки используйте `--test` — скрипт попытается загрузить модель в `onnxruntime` и
выполнит тестовый прогон с случайным входом, печатая список входов/выходов и форму тензоров.

Требования
----------
- Python 3.13.4
- onnx 1.19.1
- onnxruntime 1.23.2
- onnx-graphsurgeon 0.5.8
- numpy 2.2.6
- ultralytics (нужен только для export_model.py)

Contributing
------------
- 

License / Contact
-----------------
- Лицензия MIT


