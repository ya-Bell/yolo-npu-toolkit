# ⚡ YOLO → NPU Toolkit

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-GraphSurgeon-005CED?logo=onnx&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-YOLOv8-EE4C2C?logo=pytorch&logoColor=white)
![Target](https://img.shields.io/badge/Hardware-NPU%20%2F%20Edge%20AI-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Инструмент трансформации и оптимизации вычислительных графов моделей **YOLO (ONNX)** для аппаратных нейроускорителей (NPU, Edge TPU, RKNN и встроенные чипы).

---

## 🎯 Проблема и решение

### Проблема
Большинство стандартных YOLO-экспортов (например, Ultralytics YOLOv8) объединяют все предсказания в один общий flatten-тензор на выходе (`[1, 84, 8400]`), либо содержат сложные операторы постобработки (Non-Max Suppression / NMS). 
Компиляторы аппаратных NPU-ускорителей часто не поддерживают динамические NMS-операторы или теряют квантованную производительность при работе с единым конкатенированным тензором.

### Решение
`yolo-npu-toolkit` препарирует выходную голову модели с помощью **ONNX GraphSurgeon**:
* Разделяет единый монолитный выход на независимые карты признаков по трём масштабам детекции: **$80 \times 80$**, **$40 \times 40$** и **$20 \times 20$**.
* Для каждого масштаба выделяет отдельные ветки:
  * `feat` (координаты боксов / регрессия)
  * `cls` (вероятности классов)
  * `conf` (уверенность детектора)
* Позволяет компиляторам NPU эффективно квантовать (INT8/FP16) и распределять вычисления по аппаратным ядрам, снижая общую задержку инференса (latency).

---

## 🏗️ Структура проекта

```text
├── edit_model.py              # Основной модуль трансформации графа ONNX
├── requirements.txt           # Зависимости проекта
├── scripts/
│   ├── export_model.py        # Утилита экспорта PyTorch YOLO в чистый ONNX
│   └── checking_version.py    # Проверка совместимости версий библиотек
├── onnx/                      # Каталог для исходных ONNX-моделей
└── mod_onnx/                  # Каталог с оптимизированными под NPU моделями
```

---

## 🚀 Быстрый старт

### 1. Установка окружения

```bash
# Клонирование репозитория
git clone https://github.com/ya-Bell/yolo-npu-toolkit.git
cd yolo-npu-toolkit

# Создание и активация виртуального окружения
python -m venv .venv
source .venv/bin/activate  # Для Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Для Windows PowerShell

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Трансформация модели

Преобразование исходной модели `yolov8n.onnx` и верификация через `onnxruntime`:

```bash
python edit_model.py -m onnx/yolov8n.onnx --test
```

Параметры запуска:
* `-m, --model` — путь к входному `.onnx` файлу (обязательный).
* `-o, --output` — (опционально) путь для сохранения модифицированной модели. По умолчанию сохраняется в `mod_onnx/<name>_mod.onnx`.
* `--test` — автоматический валидационный прогон тестового тензора через `onnxruntime` для проверки целостности графа и формы выходных тензоров.

---

## 🛠 Стек технологий

* **Python 3.10+**
* **ONNX** и **ONNX GraphSurgeon** (манипуляция узлами графа, разрезание и переподключение тензоров)
* **ONNX Runtime** (валидация инференса)
* **NumPy**

---

## 👤 Автор

* **Алексей Языков** ([@ya-Bell](https://github.com/ya-Bell)) — СПбГУ, факультет математики и механики.
