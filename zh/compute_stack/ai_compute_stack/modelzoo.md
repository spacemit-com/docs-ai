sidebar_position: 9

# ModelZoo
>
> ModelZoo内模型数据定期更新，模型应用示例请参考[ai-sdk
> ](https://github.com/spacemit-com/ai-sdk)

- [ModelZoo](#modelzoo)
  - [基础模型](#基础模型)
    - [测试方式](#测试方式)
    - [resnet](#resnet)
    - [mobilenet](#mobilenet)
    - [efficientnet](#efficientnet)
    - [vit](#vit)
    - [yolov5](#yolov5)
    - [yolov6](#yolov6)
    - [yolov8](#yolov8)
    - [yolov8-seg](#yolov8-seg)
    - [yolov8-pose](#yolov8-pose)
    - [yolov12](#yolov12)
    - [音频模型](#音频模型)
  - [大模型](#大模型)
    - [测试方式](#测试方式-1)
    - [Qwen](#qwen)
    - [HunYuan](#hunyuan)
    - [Llama](#llama)
  - [多模态大模型](#多模态大模型)
    - [测试方式](#测试方式-2)
    - [VLM](#vlm)
    - [ASR](#asr)

## 基础模型
- K1
>- 推理引擎版本: spacemit-ort-2.0.6
>- OS：bianbu-3.0
>- date：2026-7-27

- K3
>- 推理引擎版本: [v2.0.6](https://github.com/spacemit-com/onnxruntime/releases/download/2.0.6/spacemit-ort.riscv64.2.0.6.tar.gz)
>- OS：bianbu-4.0rc3
>- date：2026-7-27

### 测试方式
~~~
# 进入spacemit-ort库路径
# cd {spacemit_ort_lib}/
export LD_LIBRARY_PATH=./lib/

# 调整为自己的${model_path}(模型文件路径)，${num of cores}(选择跑几个核心)
./bin/onnxruntime_perf_test ${model_path} -e spacemit -r 10 -x 1 -S 1 -s -c 1 -i "SPACEMIT_EP_INTRA_THREAD_NUM|${num of cores}" -I

# 输出信息如下
using SpaceMITExecutionProvider
setting SPACEMIT_EP_INTRA_THREAD_NUM : 4
Setting intra_op_num_threads to 1
Session creation time cost: 0.169475 s
First inference time cost: 109 ms
Total inference time cost: 0.0727021 s
Total inference requests: 10
Average inference time cost total: 7.270205 ms
Total inference run time: 0.0727619 s
Number of inferences per second: 137.435
Avg CPU usage: 62 %
Peak working set size: 91336704 bytes
Avg CPU usage:62
Peak working set size:91336704
Runs:10
Min Latency: 0.00720383 s
Max Latency: 0.00730163 s
P50 Latency: 0.00727787 s
P90 Latency: 0.00730163 s
P95 Latency: 0.00730163 s
P99 Latency: 0.00730163 s
P999 Latency: 0.00730163 s

# Average inference time cost total即单帧推理耗时
~~~

### resnet

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| [resnet18](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/resnet/resnet18.q.onnx) | int8 | 224x224 | 40.68 | 22.15 | 12.99 |
| [resnet50](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/resnet/resnet50.q.onnx) | int8 | 224x224 | 95.36 | 52.89 | 32.29 |
| [resnet50](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/resnet/resnet50.fp16.onnx) | fp16 | 224x224 | 674.48 | 363.79 | 227.39 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- | --- |
| [resnet18](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/resnet/resnet18.q.onnx) | int8 | 224x224 | 8.00 | 4.74 | 2.90 | 2.07 |
| [resnet50](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/resnet/resnet50.q.onnx) | int8 | 224x224 | 21.00 | 12.21 | 7.72 | 5.47 |
| [resnet50.batch4](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/resnet/resnet50.b4.q.onnx) | int8 | 224x224 | 73.37 | 40.19 | 23.19 | 15.55 |
| [resnet50](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/resnet/resnet50.fp16.onnx) | fp16 | 224x224 | 37.64 | 21.95 | 14.89 | 11.36 |

### mobilenet

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| [mobilenet_v1](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v1.q.onnx) | int8 | 224x224 | 31.06 | 16.16 | 10.21 |
| [mobilenet_v2](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v2.q.onnx) | int8 | 224x224 | 30.48 | 19.30 | 13.70 |
| [mobilenet_v3_small](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v3_small.fp16.onnx) | fp16 | 224x224 | 26.91 | 16.31 | 10.60 |
| [mobilenet_v3_large](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v3_large.fp16.onnx) | fp16 | 224x224 | 67.58 | 40.64 | 28.37 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- |---|
| [mobilenet_v1](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v1.q.onnx) | int8 | 224x224 | 12.67 | 7.21 | 3.91 | 2.35 |
| [mobilenet_v2](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v2.q.onnx) | int8 | 224x224 | 17.69 | 9.92 | 5.22 | 3.26 |
| [mobilenet_v3_small](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v3_small.fp16.onnx) | fp16 | 224x224 | 8.71 | 5.14 | 3.23 | 2.75 |
| [mobilenet_v3_large](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobilenet/mobilenet_v3_large.fp16.onnx) | fp16 | 224x224 | 16.59 | 9.67 | 5.95 | 4.52 |

### efficientnet

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| [efficientnet_v1_b0](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b0.q.onnx) | int8 | 224x224 | 82.84 | 49.03 | 33.74 |
| [efficientnet_v1_b1](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b1.q.onnx) | int8 | 224x224 | 119.96 | 68.63 | 47.31 |
| [efficientnet_v2_s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v2_s.q.onnx) | int8 | 224x224 | 170.51 | 95.03 | 60.96 |
| [efficientnet_v1_b0](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b0.fp16.onnx) | fp16 | 224x224 | 140.58 | 83.16 | 59.58 |
| [efficientnet_v1_b1](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b1.fp16.onnx) | fp16 | 224x224 | 198.59 | 120.75 | 83.44 |
| [efficientnet_v2_s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v2_s.fp16.onnx) | fp16 | 224x224 | 621.82 | 337.15 | 200.57 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- |---|
| [efficientnet_v1_b0](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b0.q.onnx) | int8 | 224x224 | 40.84 | 22.51 | 12.89 | 9.85 |
| [efficientnet_v1_b1](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b1.q.onnx) | int8 | 224x224 | 63.51 | 34.50 | 20.16 | 15.49 |
| [efficientnet_v2_s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v2_s.q.onnx) | int8 | 224x224 | 58.41 | 33.46 | 19.80 | 13.53 |
| [efficientnet_v1_b0](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b0.fp16.onnx) | fp16 | 224x224 | 43.35 | 23.67 | 14.30 | 10.91 |
| [efficientnet_v1_b1](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v1_b1.fp16.onnx) | fp16 | 224x224 | 63.37 | 35.19 | 20.96 | 15.63 |
| [efficientnet_v2_s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/efficientnet/efficientnet_v2_s.fp16.onnx) | fp16 | 224x224 | 82.89 | 47.68 | 29.77 | 20.67 |

### vit

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| [vit_b_16](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/vit/vit_b_16.q.onnx) | int8 | 224x224 | 518.05 | 344.59 | 179.51 |
| [vit_b_16](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/vit/vit_b_16.fp16.onnx) | fp16 | 224x224 | 2549.14 | 1410.04 | 766.46 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- | --- |
| [vit_b_16](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/vit/vit_b_16.q.onnx) | int8 | 224x224 | 100.90 | 56.64 | 35.19 | 23.76 |
| [vit_b_16](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/vit/vit_b_16.fp16.onnx) | fp16 | 224x224 | 138.46 | 86.37 | 62.11 | 48.75 |

### yolov5

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| [yolov5n](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5/yolov5n.q.onnx) | int8 | 640x640 | 248.40 | 138.78 | 85.24 |
| [yolov5s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5/yolov5s.q.onnx) | int8 | 640x640 | 473.59 | 252.89 | 150.95 |
| [yolov5m](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5/yolov5m.q.onnx) | int8 | 640x640 | 996.62 | 516.62 | 286.32 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- | --- |
| [yolov5n](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5/yolov5n.q.onnx) | int8 | 640x640 | 43.76 | 24.11 | 14.36 | 9.57 |
| [yolov5s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5/yolov5s.q.onnx) | int8 | 640x640 | 73.40 | 40.20 | 23.95 | 15.81 |
| [yolov5m](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5/yolov5m.q.onnx) | int8 | 640x640 | 152.03 | 81.90 | 45.88 | 28.97 |

### yolov6

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| [yolov6n](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov6/yolov6n.q.onnx) | int8 | 640x640 | 173.42 | 93.21 | 55.98 |
| [yolov6s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov6/yolov6s.q.onnx) | int8 | 640x640 | 438.71 | 224.39 | 123.60 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- | --- |
| [yolov6n](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov6/yolov6n.q.onnx) | int8 | 640x640 | 31.90 | 18.04 | 10.95 | 7.53 |
| [yolov6s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov6/yolov6s.q.onnx) | int8 | 640x640 | 65.37 | 35.98 | 21.33 | 13.42 |

### yolov8
- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| [yolov8n](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8/yolov8n.q.onnx) | int8 | 640x640 | 233.76 | 123.09 | 73.74 |
| [yolov8s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8/yolov8s.q.onnx) | int8 | 640x640 | 517.42 | 274.48 | 153.08 |
| [yolov8m](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8/yolov8m.q.onnx) | int8 | 640x640 | 1079.16 | 529.76 | 297.77 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- | --- |
| [yolov8n](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8/yolov8n.q.onnx) | int8 | 640x640 | 41.41 | 23.08 | 13.85 | 9.51 |
| [yolov8s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8/yolov8s.q.onnx) | int8 | 640x640 | 74.50 | 41.19 | 24.89 | 16.65 |
| [yolov8m](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8/yolov8m.q.onnx) | int8 | 640x640 | 157.80 | 85.68 | 48.90 | 32.31 |

### yolov8-seg

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- | --- |
| [yolov8n-seg](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_seg/yolov8n-seg.q.onnx) | int8 | 640x640 | 61.59 | 33.57 | 19.40 | 12.75 |
| [yolov8s-seg](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_seg/yolov8s-seg.q.onnx) | int8 | 640x640 | 103.27 | 56.33 | 33.24 | 21.51 |
| [yolov8m-seg](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_seg/yolov8m-seg.q.onnx) | int8 | 640x640 | 204.94 | 109.94 | 61.93 | 39.61 |

### yolov8-pose

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- | --- |
| [yolov8n-pose](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_pose/yolov8n-pose.q.onnx) | int8 | 640x640 | 46.42 | 26.18 | 16.16 | 11.16 |
| [yolov8s-pose](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_pose/yolov8s-pose.q.onnx) | int8 | 640x640 | 81.61 | 45.33 | 27.87 | 18.64 |
| [yolov8m-pose](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_pose/yolov8m-pose.q.onnx) | int8 | 640x640 | 165.60 | 90.30 | 51.75 | 33.84 |

### yolo12

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| [yolo12n](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo12/yolo12n.q.onnx) | int8 | 640x640 | 377.06 | 203.86 | 130.24 |
| [yolo12s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo12/yolo12s.q.onnx) | int8 | 640x640 | 831.48 | 464.78 | 265.56 |
| [yolo12m](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo12/yolo12m.q.onnx) | int8 | 640x640 | 1979.32 | 1076.22 | 590.99 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- | --- |
| [yolo12n](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo12/yolo12n.q.onnx) | int8 | 640x640 | 107.56 | 57.85 | 32.55 | 21.73 |
| [yolo12s](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo12/yolo12s.q.onnx) | int8 | 640x640 | 191.31 | 102.05 | 56.90 | 36.24 |
| [yolo12m](https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo12/yolo12m.q.onnx) | int8 | 640x640 | 378.55 | 200.75 | 110.23 | 69.17 |

### 音频模型

- K1

| 模型名 | type | 4 Core/rtf |
| --- | --- | --- |
| melotts | dyn_int8  | 0.984 |
| [sensevoice](https://archive.spacemit.com/spacemit-ai/model_zoo/asr/sensevoice.tar.gz) | dyn_int8 | --- |

- K3

| 模型名 | type | 4 Core/rtf | 8 Core/rtf |
| --- | --- | --- | --- |
| melotts | dyn_int8  | 0.530 | --- |
| [sensevoice](https://archive.spacemit.com/spacemit-ai/model_zoo/asr/sensevoice.tar.gz) | dyn_int8 | 0.1124 | 0.1380 |

## 大模型

- K3
>- llama.cpp版本：[0.1.1](https://github.com/spacemit-com/llama.cpp/releases/download/spacemit-llama.cpp.riscv64.0.1.1/spacemit-llama.cpp.riscv64.0.1.1.tar.gz)
>- OS：bianbu-4.0rc3
>- date：2026-5-26

### 测试方式
~~~
# 进入spacemit-llama.cpp库路径
# cd {spacemit-llama.cpp}/
export LD_LIBRARY_PATH=./lib/

# 调整为自己的${model_path}(模型文件路径)，${num of cores}(选择跑几个核心)
./bin/llama-bench -m ${model_path} -t ${num of cores} -p 128 -n 128 -mmp 0 -fa 1 -ub 128

# 输出信息如下
CPU_RISCV64_SPACEMIT: tcm is available, blk_size: 393216, blk_num: 8, is_fake_tcm: 0
CPU_RISCV64_SPACEMIT: num_cores: 16, num_perfer_cores: 8, perfer_core_arch_id: a064, exclude_main_thread: 0, use_ime1: 0, use_ime2: 1, mem_backend: HPAGE, cpu_mask: ff00, aicpu_id_offset: 8
CPU_RISCV64_SPACEMIT: alloc_chunk: open(/dev/tcm_sync_mem) failed, errno=2
CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap
| model                          |       size |     params | backend    | threads | n_ubatch | fa | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | -: | ---: | --------------: | -------------------: |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | CPU        |       8 |      128 |  1 |    0 |           pp128 |        499.75 ± 0.22 |
| qwen3 0.6B Q4_0                | 358.78 MiB |   596.05 M | CPU        |       8 |      128 |  1 |    0 |           tg128 |         53.35 ± 0.03 |
~~~

### Qwen

- K3

| 模型名 | 量化类型 | PP128 (token/s) | TG128 (token/s) | PP1280 (token/s) | TG1280 (token/s) |
| --- | --- | --- | --- | --- | --- |
| [qwen3-0.6B](https://www.modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/file/view/master/Qwen3-0.6B-Q4_0.gguf?status=2) | Q4_0 | 499.75 | 53.35 | - | - |
| [qwen3-1.7B](https://www.modelscope.cn/models/unsloth/Qwen3-1.7B-GGUF/file/view/master/Qwen3-1.7B-Q4_0.gguf?status=2) | Q4_0 | 229.79 | 23.11 | - | - |
| [qwen3-4B](https://www.modelscope.cn/models/unsloth/Qwen3-4B-GGUF/file/view/master/Qwen3-4B-Q4_0.gguf?status=2) | Q4_0 | 76.44 | 11.03 | - | - |
| [qwen3-moe-30B-A3B](https://www.modelscope.cn/models/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/file/view/master/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf?status=2) | Q4_0 | 55.67 | 12.32 | 44.03 | 11.17
| [qwen3.5-0.8B](https://www.modelscope.cn/models/unsloth/Qwen3.5-0.8B-GGUF/file/view/master/Qwen3.5-0.8B-Q4_0.gguf?status=2) | Q4_0 | 182.69 | 29.33 | - | - |
| [qwen3.5-2B](https://www.modelscope.cn/models/unsloth/Qwen3.5-2B-GGUF/file/view/master/Qwen3.5-2B-Q4_0.gguf?status=2) | Q4_1 | 112.22 | 16.15 | - | - |

### HunYuan

- K3

| 模型名 | 量化类型 | PP128 (token/s) | TG128 (token/s) | PP1280 (token/s) | TG1280 (token/s) |
| --- | --- | --- | --- | --- | --- |
| [HY-MT1.5-1.8B](https://www.modelscope.cn/models/Tencent-Hunyuan/HY-MT1.5-1.8B-GGUF/resolve/master/HY-MT1.5-1.8B-Q4_K_M.gguf) | Q4_K_M | 157.81 | 20.15 | - | - |

### Llama

- K3

| 模型名 | 量化类型 | PP128 (token/s) | TG128 (token/s) | PP1280 (token/s) | TG1280 (token/s) |
| --- | --- | --- | --- | --- | --- |
| [llama2-7B](https://www.modelscope.cn/models/TheBloke/Llama-2-7B-GGUF/resolve/master/llama-2-7b.Q4_0.gguf) | Q4_0 | 50.40 | 7.07 | - | - |

## 多模态大模型

- K3
>- llama.cpp版本：[0.1.1](https://github.com/spacemit-com/llama.cpp/releases/download/spacemit-llama.cpp.riscv64.0.1.1/spacemit-llama.cpp.riscv64.0.1.1.tar.gz)
>- 推理引擎版本: [v2.0.3](https://github.com/spacemit-com/onnxruntime/releases/download/2.0.3/spacemit-ort.riscv64.2.0.3.tar.gz)
>- OS：bianbu-4.0rc3
>- date：2026-5-26

### 测试方式


> 以qwen3vlencoder为例

```bash
export LD_LIBRARY_PATH=./spacemit-llama.cpp/lib:./spacemit_ort/lib
export SPACEMIT_EP_DENSE_ACCURACY_LEVEL=1

llama-server -m qwen3vl-30b-text-q4_1.gguf --media-backend smt --smt-config-dir ./ -ctk f16 -ctv f16 -t 8 -c 1024 --host 0.0.0.0 --port 8080 --reasoning-budget 0 --reasoning off
```
> 详细参数含义见llama.cpp.md

### VLM

- K3

| 模型名 | 图像规格 | LLM 8 Core + VisionEncoder 4 Core/ms | LLM 8 Core + VisionEncoder 8 Core/ms |
| --- | --- | --- | --- |
| [fastvlm-0.5B](https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/fastvlm-mm-0.5b-q4_1.tar.gz) | 512*512 | 256.47 | 164.50 |
| [Qwen3-VL-30B-A3B](https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/qwen30ba3b-mm-q4_1.tar.gz) | 768*768 | 7928.13 | 4753.55 |
| [Qwen3.5-0.8B](https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/Qwen3.5-0.8B.tar.gz) | 384*384 | 340.42 | 245.61 |
| [Qwen3.5-2B](https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/Qwen3.5-2B.tar.gz) | 384*384 | 901.56 | 794.03 |
| [Qwen3.5-4B](https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/Qwen3.5-4B.tar.gz) | 384*384 | 904.73 | 798.71 |

### ASR

- K3

| 模型名 | LLM 8 Core + AudioEncoder 4 Core/rtf |
| --- | --- |
| [qwen3-ASR-0.6B](https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/qwen3-asr-0.6B.tar.gz) | 0.186 |
