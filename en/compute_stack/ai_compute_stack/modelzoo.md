sidebar_position: 9

# ModelZoo

The models in ModelZoo are updated on a regular basis.

## Base Models
>
>- Inference engine version: **spacemit-ort-2.0.1**
>- OS: **bianbu-2.2**
>- Date: **2025-11-18**

### ResNet

- **K1**

| Model | Type | Input Shape | 1 Core (ms) | 2 Cores (ms) | 4 Cores (ms) |
| --- | --- | --- | --- | --- | --- |
| resnet18 | int8 | 224×224 | 39.2943 | 22.208 | 13.4936 |
| resnet50 | int8 | 224×224 | 92.0352 | 52.6714 | 32.3108 |
| resnet50 | fp16 | 224×224 | 695.636 | 349.947 | 221.555 |

- **K3**

> TBD

### MobileNet

- **K1**

| Model | Type | Input Shape | 1 Core (ms) | 2 Cores (ms) | 4 Cores (ms) |
| --- | --- | --- | --- | --- | --- |
| mobilenet_v1 | int8 | 224×224 | 34.9488 | 19.5954 | 13.2855 |
| mobilenet_v2 | int8 | 224×224 | 34.8 | 21.0984 | 13.723 |
| mobilenet_v3_small | fp16 | 224×224 | 53.95 | 30.97 | 19.55 |
| mobilenet_v3_large | fp16 | 224×224 | 111.82 | 63.73 | 37.67 |

- **K3**

> TBD

### EfficientNet

- **K1**

| Model | Type | Input Shape | 1 Core (ms) | 2 Cores (ms) | 4 Cores (ms) |
| --- | --- | --- | --- | --- | --- |
| efficientnet_v1_b0 | int8 | 224×224 | 76.1146 | 43.7698 | 28.0568 |
| efficientnet_v1_b1 | int8 | 224×224 | 111.128 | 63.4798 | 39.5626 |
| efficientnet_v2_s | int8 | 224×224 | 149.12 | 86.0311 | 54.5618 |
| efficientnet_v1_b0 | fp16 | 224×224 | 228.692 | 126.799 | 74.8569 |
| efficientnet_v1_b1 | fp16 | 224×224 | 331.06 | 183.432 | 107.16 |
| efficientnet_v2_s | fp16 | 224×224 | 565.195 | 313.629 | 182.697 |

- **K3**

> TBD

### Vision Transformer (ViT)

- **K1**

| Model | Type | Input Shape | 1 Core (ms) | 2 Cores (ms) | 4 Cores (ms) |
| --- | --- | --- | --- | --- | --- |
| vit_b_16 | int8 | 224×224 | 526.331 | 357.206 | 203.762 |
| vit_b_16 | fp16 | 224×224 | 3428.19 | 1962.19 | 1142.83 |

- **K3**

> TBD

### yolov5

- **K1**

| Model | Type | Input Shape | 1 Core (ms) | 2 Cores (ms) | 4 Cores (ms) |
| --- | --- | --- | --- | --- | --- |
| yolov5n | int8 | 640x640 | 229.827 | 127.649 | 81.7252 |
| yolov5s | int8 | 640x640 | 452.873 | 238.459 | 142.081 |
| yolov5m | int8 | 640x640 | 980.426 | 481.453 | 272.397 |

- **K3**

> TBD

### yolov6

- **K1**

| Model | Type | Input Shape | 1 Core (ms) | 2 Cores (ms) | 4 Cores (ms) |
| --- | --- | --- | --- | --- | --- |
| yolov6n | int8 | 640x640 | 160.075 | 90.2756 | 59.3966 |
| yolov6s | int8 | 640x640 | 431.835 | 219.982 | 125.631 |

- **K3**

> TBD

### yolov8

- **K1**

| Model | Type | Input Shape | 1 Core (ms) | 2 Cores (ms) | 4 Cores (ms) |
| --- | --- | --- | --- | --- | --- |
| yolov8n | int8 | 640x640 | 209.169 | 117.75 | 76.641 |
| yolov8s | int8 | 640x640 | 459.718 | 239.252 | 142.641 |
| yolov8m | int8 | 640x640 | 1000.35 | 510.228 | 286.011 |

- **K3**

> TBD

### yolov12

- **K1**

| Model | Type | Input Shape | 1 Core (ms) | 2 Cores (ms) | 4 Cores (ms) |
| --- | --- | --- | --- | --- | --- |
| yolo12n | int8 | 640x640 | 422.273 | 246.346 | 168.666 |
| yolo12s | int8 | 640x640 | 892.293 | 495.835 | 326.551 |
| yolo12m | int8 | 640x640 | 2080.74 | 1072.63 | 681.091 |

- **K3**

> TBD

## LLM

### Qwen3

- **K1**

> TBD

- **K3**

> TBD

### Qwen2.5

- **K1**

> TBD

- **K3**

> TBD