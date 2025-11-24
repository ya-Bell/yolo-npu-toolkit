from pathlib import Path
import shutil
import traceback

try:
    from ultralytics import YOLO
except:
    YOLO = None


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ONNX_DIR = ROOT / "onnx"


MENU = {
    '1': ('yolov8x', 'YOLOv8 X'),
    '2': ('yolov8l', 'YOLOv8 L'),
    '3': ('yolov8n', 'YOLOv8 N'),
    '4': ('yolo11x', 'YOLO11 X'),
    '5': ('yolo11l', 'YOLO11 L'),
    '6': ('yolo11n', 'YOLO11 N'),
}


# Создание директорий
def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)


# Выбор моделей
def choose_model():
    print("Выберите модель:")
    for k, v in MENU.items():
        print(f"{k}. {v[1]}")

    raw = input("Введите номера (через пробел или запятую): ")

    # Заменяем запятые на пробелы, оставляем только цифры и пробелы
    raw_clean = "".join(c if c.isdigit() or c.isspace() else " " for c in raw)

    # Разбиваем на отдельные элементы и фильтруем только допустимые ключи MENU
    selected = [MENU[x][0] for x in raw_clean if x in MENU]

    return selected or None

# Экспорт ONNX
def export_model(model_src: str, opset=12, dynamic=False):
    model_name = Path(model_src).stem
    pt_path = MODELS_DIR / f"{model_name}.pt"
    onnx_path = ONNX_DIR / f"{model_name}.onnx"

    print(f"Загружаю модель: {model_src}")
    model = YOLO(model_src)

    # Сохраняем PT только в /models
    if not pt_path.exists():
        try:
            model.save(str(pt_path))
            print(f"[PT] сохранён в /models/: {pt_path.name}")
        except:
            print(f"Не удалось сохранить PT через model.save()")

    # Экспорт ONNX сразу в /onnx
    print(f"Экспорт в ONNX...")
    out = model.export(format="onnx", opset=opset, dynamic=dynamic)
    exported_onnx = Path(out["onnx"]) if isinstance(out, dict) else Path(out)
    
    # Перемещаем ONNX в /onnx, если он там не оказался
    if exported_onnx.parent != ONNX_DIR:
        shutil.move(str(exported_onnx), str(onnx_path))
        print(f"[ONNX] перемещён в /onnx/: {onnx_path.name}")
    else:
        print(f"[ONNX] сохранён в /onnx/: {exported_onnx.name}")

    return onnx_path


# MAIN
def main():
    ensure_dirs()

    selected = choose_model()
    if not selected:
        return

    for model_key in selected:
        print("\n---")
        print(f"Обработка: {model_key}")

        # Если PT уже есть в models — используем его
        local_pt = MODELS_DIR / f"{model_key}.pt"
        model_src = str(local_pt if local_pt.exists() else model_key)

        try:
            out = export_model(model_src)
            print(f"Готово: {out}")
        except:
            print(f"Ошибка при экспорте {model_key}:")
            traceback.print_exc()

    # После обработки всех моделей удаляем PT из корня, если он там остался
    for pt in list(Path.cwd().glob("*.pt")):
        try:
            pt.unlink()
            print(f"[CLEAN] удалён лишний PT из корня: {pt.name}")
        except Exception as e:
            print(f"Не удалось удалить {pt.name}: {e}")


if __name__ == "__main__":
    main()
