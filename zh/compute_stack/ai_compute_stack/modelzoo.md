sidebar_position: 9

# ModelZoo
>
> ModelZoo内模型数据定期更新

- [ModelZoo](#modelzoo)
  - [基础模型](#基础模型)
    - [resnet](#resnet)
    - [mobilenet](#mobilenet)
    - [efficientnet](#efficientnet)
    - [vit](#vit)
    - [yolov5](#yolov5)
    - [yolov6](#yolov6)
    - [yolov8](#yolov8)
    - [yolov12](#yolov12)
  - [大模型](#大模型)
    - [Qwen3](#qwen3)

## 基础模型
- K1
>- 推理引擎版本: spacemit-ort-2.0.2+beta1
>- OS：bianbu-3.0
>- date：2026-2-9

- K3
>- 推理引擎版本: spacemit-ort-2.0.2+beta1
>- OS：bianbu-4.0aplha1
>- date：2026-2-9
>- 6400DDR，A100@1.8GHz

### resnet

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| resnet18 | int8 | 224x224 | 39.71 | 22.49 | 13.71 |
| resnet50 | int8 | 224x224 | 93.37 | 53.01 | 32.86 |
| resnet50 | fp16 | 224x224 | 667.55 | 349.34 | 217.27 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- | --- |
| resnet18 | int8 | 224x224 | 7.66 | 4.60 | 2.92 | - |
| resnet50 | int8 | 224x224 | 18.91 | 11.17 | 7.37 | 5.44 |
| resnet50.batch4 | int8 | 224x224 | - | - | - | 15.55 |
| resnet50 | fp16 | 224x224 | 35.10 | 24.13 | 19.34 | - |

### mobilenet

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| mobilenet_v1 | int8 | 224x224 | 32.10 | 16.56 | 10.72 |
| mobilenet_v2 | int8 | 224x224 | 28.44 | 18.17 | 13.03 |
| mobilenet_v3_small | fp16 | 224x224 | 24.22 | 16.84 | 12.44 |
| mobilenet_v3_large | fp16 | 224x224 | 61.62 | 38.90 | 26.61 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| mobilenet_v1 | int8 | 224x224 | 13.20 | 7.46 | 4.88 |
| mobilenet_v2 | int8 | 224x224 | 14.51 | 8.62 | 5.91 |
| mobilenet_v3_small | fp16 | 224x224 | 6.75 | 4.31 | 2.90 |
| mobilenet_v3_large | fp16 | 224x224 | 13.37 | 8.25 | 5.48 |

### efficientnet

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| efficientnet_v1_b0 | int8 | 224x224 | 68.81 | 40.65 | 26.30 |
| efficientnet_v1_b1 | int8 | 224x224 | 97.24 | 57.21 | 37.28 |
| efficientnet_v2_s | int8 | 224x224 | 144.81 | 83.11 | 52.66 |
| efficientnet_v1_b0 | fp16 | 224x224 | 121.70 | 71.87 | 46.47 |
| efficientnet_v1_b1 | fp16 | 224x224 | 172.87 | 102.10 | 65.98 |
| efficientnet_v2_s | fp16 | 224x224 | 563.58 | 305.40 | 176.87 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| efficientnet_v1_b0 | int8 | 224x224 | 28.30 | 16.33 | 10.46 |
| efficientnet_v1_b1 | int8 | 224x224 | 43.66 | 25.50 | 16.57 |
| efficientnet_v2_s | int8 | 224x224 | 42.52 | 24.34 | 15.20 |
| efficientnet_v1_b0 | fp16 | 224x224 | 30.51 | 18.41 | 12.33 |
| efficientnet_v1_b1 | fp16 | 224x224 | 44.64 | 27.09 | 18.22 |
| efficientnet_v2_s | fp16 | 224x224 | 54.56 | 31.69 | 20.53 |

### vit

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| vit_b_16 | int8 | 224x224 | 527.78 | 356.00 | 200.91 |
| vit_b_16 | fp16 | 224x224 | 2557.03 | 1425.90 | 774.00 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms | 8 Core/ms |
| --- | --- | --- | --- | --- | --- | --- |
| vit_b_16 | int8 | 224x224 | 152.84 | 88.43 | 55.92 | 37.46 |
| vit_b_16 | fp16 | 224x224 | 174.96 | 104.06 | 71.47 | - |

### yolov5

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| yolov5n | int8 | 640x640 | 233.24 | 149.24 | 111.18 |
| yolov5s | int8 | 640x640 | 450.00 | 238.84 | 140.92 |
| yolov5m | int8 | 640x640 | 996.12 | 483.86 | 269.41 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| yolov5n | int8 | 640x640 | 57.51 | 31.80 | 19.34 |
| yolov5s | int8 | 640x640 | 83.67 | 45.97 | 27.81 |
| yolov5m | int8 | 640x640 | 159.01 | 86.06 | 49.00 |

### yolov6

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| yolov6n | int8 | 640x640 | 177.65 | 100.04 | 62.43 |
| yolov6s | int8 | 640x640 | 462.12 | 237.01 | 132.61 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| yolov6n | int8 | 640x640 | 57.28 | 31.75 | 19.05 |
| yolov6s | int8 | 640x640 | 102.75 | 55.43 | 32.57 |

### yolov8

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| yolov8n | int8 | 640x640 | 211.49 | 118.88 | 76.18 |
| yolov8s | int8 | 640x640 | 463.19 | 240.62 | 142.38 |
| yolov8m | int8 | 640x640 | 994.91 | 510.06 | 284.39 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| yolov8n | int8 | 640x640 | 62.12 | 34.20 | 20.41 |
| yolov8s | int8 | 640x640 | 91.86 | 50.25 | 29.89 |
| yolov8m | int8 | 640x640 | 173.04 | 93.61 | 52.81 |

### yolov12

- K1

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| yolo12n | int8 | 640x640 | 405.57 | 238.88 | 161.90 |
| yolo12s | int8 | 640x640 | 912.32 | 533.02 | 312.74 |
| yolo12m | int8 | 640x640 | 2050.74 | 1096.84 | 661.23 |

- K3

| 模型名 | type | shape | 1 Core/ms | 2 Core/ms | 4 Core/ms |
| --- | --- | --- | --- | --- | --- |
| yolo12n | int8 | 640x640 | 162.82 | 96.70 | 67.21 |
| yolo12s | int8 | 640x640 | 284.73 | 165.38 | 118.05 |
| yolo12m | int8 | 640x640 | 518.51 | 297.74 | 203.24 |

## 大模型

### Qwen3

- K1

> TBD

- K3

| 模型名 | 量化类型 | PP128 (token/s) | TG128 (token/s) |
| --- | --- | --- | --- |
| qwen3-0.6B | Q4_0 | 286 | 38.2 |
| qwen3-4B | Q4_0 | 52.71 | 9.02 |
| qwen3-moe-30B-A3B | Q4_0 | 37.8 | 10.68 |
| qwen3next-80B-A3B | Q2_K | 12.84 | 4.01 |
