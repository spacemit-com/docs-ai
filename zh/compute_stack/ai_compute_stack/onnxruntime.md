# ONNXRuntime

**SpacemiT-ONNXRuntime** 包含 [ONNXRuntime](https://github.com/microsoft/onnxruntime) 基础推理库与 SpacemiT-ExecutionProvider 加速后端，并保持了整体架构的解耦，因此其使用方法与社区版本 ONNXRuntime 几乎一致。

## 快速入门

### 资源获取

预编译包定期更新，地址：
[https://archive.spacemit.com/spacemit-ai/onnxruntime/](https://archive.spacemit.com/spacemit-ai/onnxruntime/)

```bash
# 示例：拉取 2.0.1 版本
wget https://archive.spacemit.com/spacemit-ai/onnxruntime/spacemit-ort.riscv64.2.0.1.tar.gz
```

### ONNXRuntime 模型推理

通用用法请参考官方文档：

- [ONNXRuntime 推理指南](https://onnxruntime.ai/docs/#onnx-runtime-for-inferencing)
- [ONNXRuntime 推理示例代码库](https://github.com/microsoft/onnxruntime-inference-examples)

### SpacemiT-ExecutionProvider 后端

#### C&C++ 示例

```cpp
#include <onnxruntime_cxx_api.h>
#include "spacemit_ort_env.h"

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "demo");
Ort::SessionOptions session_options;
std::unordered_map<std::string, std::string> provider_options;

// 以下为ep的配置，可选
// provider_options["SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE"] = "1"; 禁止使用近似后处理
// provider_options["SPACEMIT_EP_DUMP_SUBGRAPHS"] = "1"; 导出ep编译子图，在执行程序的目录下，以SpaceMITExecutionProvider_SpineSubgraph_为前缀的ONNX模型
// provider_options["SPACEMIT_EP_DEBUG_PROFILE"] = "demo"; 导出ep执行profile, 以传入字符串为前缀的json
SessionOptionsSpaceMITEnvInit(session_options, provider_options);
Ort::Session session(env, net_param_path, session_options);

// ...后续与社区版ONNXRuntime一致
```

#### Python 示例

```python
import onnxruntime as ort
import numpy as np
import spacemit_ort

eps = ort.get_available_providers()
net_param_path = "resnet18.q.onnx"

# provider_options的定义与C++版本一致
ep_provider_options = {}
# 导出ep编译子图，在执行程序的目录下，以SpaceMITExecutionProvider_SpineSubgraph_为前缀的ONNX模型
# ep_provider_options ["SPACEMIT_EP_DUMP_SUBGRAPHS"] = "1";
# 导出ep执行profile, 以传入字符串为前缀的json
# ep_provider_options ["SPACEMIT_EP_DEBUG_PROFILE"] = "demo";

session = ort.InferenceSession(net_param_path,
                providers=["SpaceMITExecutionProvider"],
                provider_options=[ep_provider_options ])

input_tensor = np.ones((1, 3, 224, 224), dtype=np.float32)
outputs = session.run(None, {"data": input_tensor})
```

### 快速验证模型性能

可参考 [spacemit-demo](https://github.com/spacemit-com/spacemit-demo)

## 配置选项 (ProviderOption) 说明

### `SPACEMIT_EP_INTRA_THREAD_NUM`

- 单独指定 EP 的线程数，独立于 ONNXRuntime 的 `intra_thread_num`
- 如果只设置 `session_options` 的 `intra_thread_num`，则相当于
   `SPACEMIT_EP_INTRA_THREAD_NUM=intra_thread_num`
   并启动 `2 * intra_thread_num` 个计算线程

```C++
std::unordered_map<std::string, std::string> provider_options;
provider_options["SPACEMIT_EP_INTRA_THREAD_NUM"] = "4";
```

### `SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD`

在同一个进程内，对所有带 EP 的 session 使用同一个 intra 线程池

```C++
std::unordered_map<std::string, std::string> provider_options;
// "1"表示enable，其他表示disable

provider_options["SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD"] = "1";

SessionOptionsSpaceMITEnvInit(session_options, provider_options);
Ort::Session session0(env, net_param_path, session_options);

SessionOptionsSpaceMITEnvInit(session_options, provider_options);
Ort::Session session1(env, net_param_path, session_options);

// session0和session1将共享EP的线程资源，请保证session和session1不同时执行推理
```

### `SPACEMIT_EP_DUMP_SUBGRAPHS`

- 导出 EP 编译子图，在执行程序的目录下，以`SpaceMITExecutionProvider_SpineSubgraph_` 为前缀的 ONNX 模型
- 可用于检查ep在推理这个模型时，切分了多少子图，子图数量越少越好
- 支持通过环境变量设置

```C++
std::unordered_map<std::string, std::string> provider_options;
// "1"表示enable，其他表示disable
provider_options["SPACEMIT_EP_DUMP_SUBGRAPHS"] = "1";
```

### `SPACEMIT_EP_DEBUG_PROFILE`

- 导出 EP 执行 profile, 以传入字符串为前缀的 json 格式文件，与 ort 的 profile 独立
- 可以使用 Google Trace、或 [perfetto](https://www.ui.perfetto.dev/) 打开Json 文件并查看每个算子的执行情况
- 支持通过环境变量设置

```C++
std::unordered_map<std::string, std::string> provider_options;
// 该值表示profile json文件的前缀
provider_options["SPACEMIT_EP_DEBUG_PROFILE"] = "profile_";
```

### `SPACEMIT_EP_DUMP_TENSORS`

- 导出 EP 执行时的每层结果, 以传入字符串为文件夹
- 支持通过环境变量设置

```C++
std::unordered_map<std::string, std::string> provider_options;
// 该值表示导出Tensor NPY文件的文件夹，若不存在，EP会自行创建这个文件夹
provider_options["SPACEMIT_EP_DUMP_TENSORS"] = "dump";
```

### `SPACEMIT_EP_DISABLE_OP_TYPE_FILTER`

- 禁止 EP 推理某些 OP 类型
- 支持通过环境变量设置

```C++
std::unordered_map<std::string, std::string> provider_options;
// 使用;间隔表示
provider_options["SPACEMIT_EP_DISABLE_OP_TYPE_FILTER"] = "Conv;Gemm";
```

### `SPACEMIT_EP_DISABLE_OP_NAME_FILTER`

- 禁止 EP 推理某些命名的 OP
- 支持通过环境变量设置

```C++
std::unordered_map<std::string, std::string> provider_options;
// 使用;间隔表示
provider_options["SPACEMIT_EP_DISABLE_OP_TYPE_FILTER"] = "Conv1;Conv2";
```

### `SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE`

- 禁止使用 FP16 后处理，例如量化模式下 Conv、Gemm 的 Scale 会内部优化为 FP16
- 支持通过环境变量设置

```C++
std::unordered_map<std::string, std::string> provider_options;
// "1"表示禁用，其他表示无效
provider_options["SPACEMIT_EP_DISABLE_FLOAT16_EPILOGUE"] = "1";
```

## Demo 说明

### `onnxruntime_perf_test`

```bash
# cd spacemit-ort.riscv64.x.x.x
export LD_LIBRARY_PATH=./lib
# 执行
./bin/onnxruntime_perf_test resnet50.q.onnx -e spacemit -r 1 -x 1 -c 1 -S 1 -I
```

**常用参数说明:**
- `-e`: 指定使用的后端，例如 `-e spacemit`
- `-r`: 指定测试次数，例如 `-r 10`
- `-x`: 指定使用线程数
- `-c`: 指定同时发起多少个推理会话
- `-S`: 给定随机种子，产生相同的输入数据。这默认为-1
- `-I`: 生成输入绑定
- `-s`: 显示统计结果
- `-p`: 生成运行日志

### `onnx_test_runner`

`onnx_test_runner` 主要用于验证和测试 ONNX 模型是否正确实现了其预期功能，以及在不同执行后端（例如 SpaceMITExecutionProvider）上的正确性和一致性。其核心用途是：
- **验证模型正确性​：** 确保从训练框架（如 PyTorch、TensorFlow）导出的 ONNX 模型在 ONNX Runtime 中的推理结果与原始框架一致。
- **检查算子支持​：** 验证模型所使用的算子是否被特定的执行提供程序（如 CPU, GPU, NPU 等）支持。
- **跨平台一致性检查​：** 保证同一个模型在不同硬件后端（如 CPU、ArmNN、ACL、CUDA 等）上运行的结果是可靠的。
- **性能测试（辅助）​​：** 虽然主要关注正确性，但其输出的运行时间也能提供初步的性能参考（更专业的性能测试建议使用 `onnxruntime_perf_test`）。

``` bash
onnx_test_runner [选项] <测试数据目录>

#添加环境变量
export LD_LIBRARY_PATH=PATH_TO_YOUR_LIB
#执行
./onnx_test_runner {.....}/resnet50.q-imagenet -j 1 -X 4 -e spacemit
```

**测试数据目录组织**

```
mobilenetv2
├── mobilenetv2.onnx      # ONNX 模型文件
└── test_data0            # 测试数据目录
    ├── input0.pb         # 输入张量（protocol buffer 格式）
    └── output_0.pb       # 期望的输出张量
└── test_data1
    ├── input0.pb
    └── output_0.pb
```

**常用参数说明：**
- `-e`: 指定EP。这是最关键的选项之一，用于选择在哪个硬件后端上运行测试。
- `-X`: 并行运行测试的线程数。
- `-c`: 并发运行的会话数。
- `-r`: 重复运行测试的次数，可用于稳定性测试。
- `-v`: 输出更详细的信息。

## 附录

- **[EP算子说明](./onnxruntime_ep_ops.md)**
   - EP 支持的算子表示能够被 EP 组织为子图并由 EP 后端推理，不支持的算子将回退至ONNXRuntime 的 CPUProvider
   - EP 支持的算子算子列表会不断扩充，可根据相关模型瓶颈情况提出需求

- **[模型性能数据](./modelzoo.md)**
   - 以 ONNXRuntime+SpaceMITExecutionProvider 进行推理，通过`onnxruntime_perf_test` 获得模型性能数据

- **[FAQ](./onnxruntime_ep_faq.md)**
   - 可在 [进迭时空开发者社区](https://forum.spacemit.com/) 提出问题，我们将尽快给出答复