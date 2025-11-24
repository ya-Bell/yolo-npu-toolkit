import onnx
import onnxruntime as ort
import onnx_graphsurgeon as gs

import numpy as np
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

# Обновляет словари:
#  - outputsDict: имя выхода → индекс узла
#  - inputsDict: имя входа → индекс узла
#  - namesDict: имя узла → индекс узла
def updateDicts(graph):
    outputsDict = {}
    inputsDict = {}
    namesDict = {}

    for idx, node in enumerate(graph.nodes):
        namesDict[node.name] = idx

        for out in node.outputs:
            outputsDict[out.name] = idx

        for inp in node.inputs:
            inputsDict[inp.name] = idx

    return outputsDict, inputsDict, namesDict

# Преобразует список имён выходов → индексы узлов
def outToIdx(outputs, outputsDict):
    return [outputsDict[x] for x in outputs]

# Поиск узлов определённого типа вверх по графу
# start — индекс узла, откуда начинаем
def searchElem(graph, start, opType, outputsDict, maxDepth=7):
    if maxDepth == 0:
        return set()

    node = graph.nodes[start]

    if node.op == opType:
        return {out.name for out in node.outputs}

    found = set()
    for inp in node.inputs:
        if inp.name in outputsDict:
            found |= searchElem(graph, outputsDict[inp.name], opType, outputsDict, maxDepth - 1)

    return found

# Получить все выходные gs.Variable для списка индексов узлов
def getOutput(graph, idxs):
    arr = []
    for i in idxs:
        arr.extend(graph.nodes[i].outputs)
    return arr


# Удаление узлов по списку индексов (в обратном порядке)
def delNodes(graph, nodesToDel):
    nodesToDel = sorted(list(nodesToDel), reverse=True)
    for i in nodesToDel:
        del graph.nodes[i]
    graph.cleanup()


# Генератор уникальных имён
counts = defaultdict(int)
def getNewName(name='', prefix=''):
    counts[name] += 1

    if prefix:
        return f"/{prefix}/{name}_{counts[name]}"
    if name:
        return f"{name}_{counts[name]}"
    return f"{counts[name]}"

# Создание выходной переменной (gs.Variable) с опциональной формой/типом
# Если shape передан — используем его, иначе пытаемся унаследовать.
def create_output(name_hint='', dtype="float32", shape=None, prefix=''):
    base = {
        "dtype": dtype,
        "shape": shape,
    }
    if name_hint:
        base["name"] = getNewName(name_hint, prefix)
    else:
        base["name"] = getNewName(prefix)
    return gs.Variable(**base)


def get_shape(var):
    """Возвращает форму переменной или None, если форма неизвестна."""
    if hasattr(var, 'shape') and var.shape is not None:
        return list(var.shape)
    return None


# Создаёт два константа 0 и 1 для Clip (или другого узла, если нужны константы)
# Возвращает список gs.Constant
def get_clip_consts():
    consts = []
    consts.append(gs.Constant(name=getNewName("Constant"), values=np.array(0.0, dtype=np.float32)))
    consts.append(gs.Constant(name=getNewName("Constant"), values=np.array(1.0, dtype=np.float32)))
    return consts


def create_slice_constants(start, end, axis, name_prefix=''):
    """Создаёт и возвращает три gs.Constant: starts, ends, axes для Slice-узла."""
    starts = gs.Constant(name=getNewName(f"{name_prefix}_starts"), values=np.array([start], dtype=np.int64))
    ends = gs.Constant(name=getNewName(f"{name_prefix}_ends"), values=np.array([end], dtype=np.int64))
    axes = gs.Constant(name=getNewName(f"{name_prefix}_axes"), values=np.array([axis], dtype=np.int64))
    return starts, ends, axes


def ensure_output_dtype_shape(outv):
    """Гарантирует, что у выходной переменной есть dtype и shape (или None)."""
    if not hasattr(outv, 'dtype') or outv.dtype is None:
        outv.dtype = 'float32'
    if not hasattr(outv, 'shape'):
        outv.shape = None


# ------------------------------------------------------------
# Вставляет в граф цепочку узлов Sigmoid -> ReduceSum -> Clip,
# начиная с входной переменной `inp_var`.
#
# - Sigmoid: создаётся без жёстко заданной формы — наследует форму входа.
# - ReduceSum: axes=[1], keepdims=1 -> если вход имел shape, результирующая форма
#   будет той же, но на оси 1 размер = 1.
# - Clip: берёт константы 0 и 1 как min/max.
#
# graph.outputs будет расширен соответствующими выходами в зависимости от аргумента.
# Возвращает список выходных переменных, добавленных в graph.outputs (для тестов).
# ------------------------------------------------------------
def insert_sigmoid_reducesum_clip(graph, inp_var, add_to_graph_outputs=True):
    # Возьмём shape входа (может быть None)
    inp_shape = None
    if hasattr(inp_var, "shape") and inp_var.shape is not None:
        inp_shape = list(inp_var.shape)

    # 1) Sigmoid node
    sigmoid_out_shape = None
    if inp_shape is not None:
        # Sigmoid не меняет форму входа
        sigmoid_out_shape = list(inp_shape)

    sigmoid_out = create_output(name_hint="Sigmoid_out", shape=sigmoid_out_shape, dtype="float32", prefix=inp_var.name.strip("/").split("/")[0])
    sigmoid_node = gs.Node(op="Sigmoid",
                           name=getNewName("Sigmoid"),
                           inputs=[inp_var],
                           outputs=[sigmoid_out])
    graph.nodes.append(sigmoid_node)

    # 2) ReduceSum node (axes=[1], keepdims=1)
    # Вычисляем уменьшенную форму, если известна входная форма
    reducesum_out_shape = None
    if sigmoid_out_shape is not None:
        reducesum_out_shape = list(sigmoid_out_shape)
        # устанавливаем размер оси 1 в 1, если такая ось существует
        if len(reducesum_out_shape) > 1:
            reducesum_out_shape[1] = 1

    reducesum_out = create_output(name_hint="ReduceSum_out", shape=reducesum_out_shape, dtype="float32", prefix=inp_var.name.strip("/").split("/")[0])
    axes_const = gs.Constant(
        name=getNewName("ReduceSum_axes"),
        values=np.array([1], dtype=np.int64)
    )

    reducesum_node = gs.Node(
        op="ReduceSum",
        name=getNewName("ReduceSum"),
        attrs={"keepdims": 1},          # axes больше НЕ атрибут!
        inputs=[sigmoid_out, axes_const],   # axes теперь ВХОД
        outputs=[reducesum_out]
    )

    graph.nodes.append(reducesum_node)

    # 3) Clip node (min=0, max=1)
    clip_out_shape = reducesum_out_shape if reducesum_out_shape is not None else None
    clip_out = create_output(name_hint="Clip_out", shape=clip_out_shape, dtype="float32", prefix=inp_var.name.strip("/").split("/")[0])

    # константы для Clip
    consts = get_clip_consts()
    clip_inputs = [reducesum_out] + consts

    clip_node = gs.Node(op="Clip",
                        name=getNewName("Clip"),
                        inputs=clip_inputs,
                        outputs=[clip_out])
    graph.nodes.append(clip_node)

    # Если нужно, добавить в graph.outputs (для того чтобы получился новый выход модели)
    if add_to_graph_outputs:
        graph.outputs.append(clip_out)

    # Возвращаем последний выход (clip)
    return clip_out

# Основная функция: модифицирует модель, заменяя Concat- или Concat-связи на цепочки
# Sigmoid -> ReduceSum -> Clip (кроме случаев, где размер канала = 64,
# тогда просто делаем этот выход graph output).
def modify_model(input_model_path: Path, output_model_path: Path):
    # загрузим onnx-сюда и импортируем в graphsurgeon
    model = onnx.load(str(input_model_path))
    graph = gs.import_onnx(model)

    # обновляем словари для поиска по графу
    outputsDict, inputsDict, namesDict = updateDicts(graph)

    # Найдём все имена выходов (строго: имена тензоров), которые происходят от Conv
    # Для этого используем обход от последнего узла вверх и ищем Conv
    conv_out_names = searchElem(graph, len(graph.nodes) - 1, "Conv", outputsDict, maxDepth=20)
    if not conv_out_names:
        print("Внимание: не найдено выходов от Conv (conv_out_names пуст). Продолжаем, но проверь модель.")
    conv_out_idxs = outToIdx(list(conv_out_names), outputsDict) if conv_out_names else []

    # Соберём набор узлов, которые нужно удалить (в примере: Concat/Conv объединители)
    nodes_to_del = set()
    for name in conv_out_names:
        if name in inputsDict:
            nodes_to_del.add(inputsDict[name])

    # Очистим текущие выходы — мы построим новые
    graph.outputs = []

    # Пройдёмся по каждому найденному Conv-выходу и вставим нашу логику
    for idx in conv_out_idxs:
        node = graph.nodes[idx]
        # берем основной выход переменной (обычно один)
        if not node.outputs:
            continue
        out_var = node.outputs[0]
        shape = None
        if hasattr(out_var, "shape"):
            shape = out_var.shape

        # Защитимся: если shape известен и channels == 64 — просто делаем его выходом
        if shape is not None and len(shape) > 1 and shape[1] == 64:
            graph.outputs.append(out_var)
        else:
            # Вставляем цепочку Sigmoid->ReduceSum->Clip, начиная с out_var
            insert_sigmoid_reducesum_clip(graph, out_var, add_to_graph_outputs=True)

    # Очистим граф (graphsurgeon сделает это при экспорте, но полезно вызвать)
    graph.cleanup()
    # Biuld NPU-friendly 9 outputs: for scale 80,40,20 produce
    # feat64_[i] (1,64,H,W), cls80_[i] (1,80,H,W), conf1_[i] (1,1,H,W)
    try:
        outputsDict, inputsDict, namesDict = updateDicts(graph)

        # Определяем кандидат на целевой Concat из трёх Reshape [1,144,-1]
        target_concat = None
        for idx, node in enumerate(graph.nodes):
            if node.op != 'Concat':
                continue
            reshape_nodes = []
            ok = True
            for inp in node.inputs:
                if inp.name not in outputsDict:
                    ok = False; break
                prod = graph.nodes[outputsDict[inp.name]]
                if prod.op != 'Reshape' or len(prod.inputs) < 2:
                    ok = False; break
                const = prod.inputs[1]
                if not isinstance(const, gs.Constant):
                    ok = False; break
                try:
                    arr = list(const.values.reshape(-1))
                    if len(arr) >= 3 and int(arr[0]) == 1 and int(arr[1]) == 144:
                        reshape_nodes.append(prod)
                    else:
                        ok = False; break
                except Exception:
                    ok = False; break
            if ok and len(reshape_nodes) == 3:
                target_concat = reshape_nodes
                break

        new_outputs = []
        if target_concat is not None:
            # Получить размер входного изображения и при возможности определить размеры сетки
            grid_sizes = []
            try:
                in_proto = model.graph.input[0]
                dims = [d.dim_value for d in in_proto.type.tensor_type.shape.dim]
                if len(dims) >= 4 and dims[2] and dims[3]:
                    input_h = int(dims[2]); input_w = int(dims[3])
                    grid_sizes = [input_h // 8, input_h // 16, input_h // 32]
            except Exception:
                grid_sizes = []

            # Для каждой ветки reshape создаём feat64, cls80, conf1
            for rnode in target_concat:
                src = rnode.inputs[0]
                # попытка найти H,W
                H = None; W = None
                sshape = getattr(src, 'shape', None)
                if sshape is not None and len(sshape) == 4:
                    H = int(sshape[2]); W = int(sshape[3])
                else:
                    # извлечь значение -1 из константы reshape
                    try:
                        vals = list(rnode.inputs[1].values.reshape(-1))
                        HW = int(vals[-1])
                        for g in grid_sizes:
                            if g * g == HW:
                                H = g; W = g; break
                    except Exception:
                        pass

                if H is None or W is None:
                    # не удалось определить форму — пропускаем эту ветку
                    continue

                # Убедимся, что у нас есть 4D тензор [1,144,H,W]
                reshaped = src
                if not (hasattr(src, 'shape') and src.shape is not None and len(src.shape) == 4 and src.shape[1] == 144):
                    shape_const = gs.Constant(name=getNewName('ReshapeShape'), values=np.array([1,144,H,W], dtype=np.int64))
                    resh_out = create_output(name_hint='reshaped', shape=[1,144,H,W], dtype='float32', prefix=src.name.strip('/').split('/')[0])
                    graph.nodes.append(gs.Node(op='Reshape', name=getNewName('Reshape_fix'), inputs=[src, shape_const], outputs=[resh_out]))
                    reshaped = resh_out

                # Срезаем первые 64 канала -> feat64
                starts1, ends1, axes = create_slice_constants(0, 64, 1, name_prefix=f'slice_{H}_feat')
                feat64 = create_output(name_hint=f'feat64_{H}', shape=[1,64,H,W], dtype='float32', prefix=reshaped.name.strip('/').split('/')[0])
                graph.nodes.append(gs.Node(op='Slice', name=getNewName('Slice_feat'), inputs=[reshaped, starts1, ends1, axes], outputs=[feat64]))

                # Срезаем следующие 80 каналов -> cls80
                starts2, ends2, _ = create_slice_constants(64, 144, 1, name_prefix=f'slice_{H}_cls')
                cls80 = create_output(name_hint=f'cls80_{H}', shape=[1,80,H,W], dtype='float32', prefix=reshaped.name.strip('/').split('/')[0])
                graph.nodes.append(gs.Node(op='Slice', name=getNewName('Slice_cls'), inputs=[reshaped, starts2, ends2, axes], outputs=[cls80]))

                # conf: Sigmoid -> ReduceSum(axes=[1], keepdims=1) -> Clip
                # Используем унифицированную функцию, чтобы не дублировать логику
                conf = insert_sigmoid_reducesum_clip(graph, cls80, add_to_graph_outputs=False)

                # собираем выходы
                new_outputs.append(feat64)
                new_outputs.append(cls80)
                new_outputs.append(conf)

        # Если мы построили 9 выходов, заменяем graph.outputs и гарантируем наличие dtype
        if len(new_outputs) >= 9:
            # Сохраняем порядок: scale1(feat,cls,conf), scale2(...), scale3(...)
            graph.outputs = new_outputs[:9]
            # Гарантируем наличие dtype и форм
            for outv in graph.outputs:
                ensure_output_dtype_shape(outv)
        else:
            print('Не удалось автоматически построить 9 ожидаемых выходов — оставляем текущие outputs')

    except Exception as e:
        print('Ошибка при построении NPU-выходов:', e)
        traceback.print_exc()

    # Сохраняем результирующую onnx-модель
    new_onnx = gs.export_onnx(graph)
    try:
        new_onnx.ir_version = 11
    except Exception:
        pass
    onnx.save(new_onnx, str(output_model_path))

    print(f"Модель сохранена: {output_model_path}")
    return output_model_path

# Тестовый инференс модели в onnxruntime
# Загружает модель, выводит входы/выходы, прогоняет случайный ввод.
def test_inference(model_path: Path):
    print("\n=== Проверка модели через onnxruntime ===")
    
    try:
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as e:
        print("Ошибка загрузки модели в onnxruntime:")
        print(e)
        return
    
    # входы
    inputs_info = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
    outputs_info = [(o.name, o.shape, o.type) for o in sess.get_outputs()]

    print("\nВходы модели:")
    for i in inputs_info:
        print("  ", i)

    print("\nВыходы модели:")
    for o in outputs_info:
        print("  ", o)

    # возьмём первый вход и создадим тестовый тензор
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    # заменим динамические размеры на 1
    test_shape = [s if isinstance(s, int) else 1 for s in input_shape]

    print(f"\nСоздаю тестовый ввод shape={test_shape}")
    x = np.random.rand(*test_shape).astype(np.float32)

    print("Запускаю инференс...")
    try:
        res = sess.run(None, {input_name: x})
        print("OK — инференс выполнен.")
        print(f"Количество выходов: {len(res)}")
    except Exception as e:
        print("Ошибка при инференсе:")
        print(e)


# Главная функция — CLI
def main():
    parser = argparse.ArgumentParser(description="Модификация ONNX модели: вставка Sigmoid->ReduceSum->Clip")
    parser.add_argument("-m", "--model", required=True, help="Путь к исходной ONNX модели")
    parser.add_argument("-o", "--out", help="Путь для сохранения модифицированной модели")
    parser.add_argument("--test", action="store_true", help="Проверить модифицированную модель через onnxruntime")

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Файл не найден: {model_path}")

    # Формируем путь выхода
    if args.out:
        out_path = Path(args.out)
    else:
        # Создаём папку mod_onnx, если она не существует
        output_dir = Path('mod_onnx')
        output_dir.mkdir(exist_ok=True)
        out_path = output_dir / (model_path.stem + "_mod.onnx")

    print(f"Загрузка модели: {model_path}")

    # Модифицируем модель
    modify_model(model_path, out_path)

    # Опциональная проверка
    if args.test:
        test_inference(out_path)

# Защищённый запуск модуля
if __name__ == "__main__":
    main()
    