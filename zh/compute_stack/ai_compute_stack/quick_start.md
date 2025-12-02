# 快速入门指南

本指南演示如何将 YOLOv8 模型部署到 SpacemiT AI CPU，并通过 **SpacemiT Execution Provider** 进行高效推理。

## 模型转换与量化

### 导出 ONNX 模型（可选）

**注：** 若已有 ONNX 模型，可跳过此步

若尚未有 ONNX 模型，从 `ultralytics` 导出一个 yolov8n 模型：

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=(640, 640), opset=17)
```

### 模型量化

通过 [xslim 量化工具](./xslim.md) 安装并准备好量化数据与量化参数

```bash
# 静态量化
python -m xslim -c ./yolov8n.json -i yolov8n.onnx -o yolov8n.q.onnx
```

```bash
# 动态量化
python -m xslim -i yolov8n.onnx -o yolov8n.fp16.onnx --dynq
```

```bash
# FP16
python -m xslim -i yolov8n.onnx -o yolov8n.fp16.onnx --fp16
```

## 模型性能测试

```bash
# 例如在spacemit-ort.riscv64.2.0.1文件夹内
export LD_LIBRARY_PATH=./lib

# 选择正确的模型路径
./bin/onnxruntime_perf_test yolov8n.q.onnx -e spacemit -r 10 -x 4 -S 1 -s -I -c 1
```

示例输出（K1 平台）如下，可以大致估模目前模型性能情况

```
./bin/onnxruntime_perf_test /mnt/modelzoo/detection/yolov8n/yolov8n.q.onnx -e spacemit -r 10 -x 4 -S 1 -s -I -c 1
using SpaceMITExecutionProvider
Setting intra_op_num_threads to 4
Session creation time cost: 1.14476 s
First inference time cost: 248 ms
Total inference time cost: 0.823382 s
Total inference requests: 10
Average inference time cost: 82.3382 ms
Total inference run time: 0.823483 s
Number of inferences per second: 12.1435
Avg CPU usage: 48 %
Peak working set size: 78946304 bytes
Avg CPU usage:48
Peak working set size:78946304
Runs:10
Min Latency: 0.0808129 s
Max Latency: 0.0838817 s
P50 Latency: 0.0826095 s
P90 Latency: 0.0838817 s
P95 Latency: 0.0838817 s
P99 Latency: 0.0838817 s
P999 Latency: 0.0838817 s
```

## 推理应用集成

参考 [yolov8_cpp_ort推理源码] (https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime-CPP/inference.cpp), 仅需两处修改即可启用 SpacemiT EP：

**修改前**

```C++
session = new Ort::Session(env, modelPath, sessionOption);
```

**修改后**（启用 SpacemiT EP）

``` C++
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"
//....
//....
//....
SessionOptionsSpaceMITEnvInit(session_options, provider_options);
session = new Ort::Session(env, modelPath, sessionOption);
```
