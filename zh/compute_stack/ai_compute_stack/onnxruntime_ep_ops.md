# EP 算子说明

- 本章节罗列 SpaceMITExecutionProvider 支持的加速算子及其规格
- [ONNX-OP 描述参考](https://onnx.ai/onnx/operators/index.html)
- [ONNX-Contrib-OP 描述参考](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md)

## Dense

| 算子名称 | Domain | Opset | Attributes | Type | 支持QDQ量化格式 | 补充说明 |
|---------|----|-------|---------|-------------|----------|----------|
| **Conv** | `ai.onnx` | 11 | W 需为常量 | `tensor(float)`<br>`tensor(float16)` | ✓ | - W 对称 perchannel <br />- X 非对称 pertensor |
| **ConvTranspose** | `ai.onnx` | 11 | W 需为常量 | `tensor(float)`<br>`tensor(float16)` | ✓ | - W 对称 perchannel <br />- X 非对称 pertensor |
| **Gemm** | `ai.onnx` | 13 | transA == 0 | `tensor(float)`<br>`tensor(float16)` | ✓ | - A 非对称 pertensor <br />- B 对称perchannel <br />- 当 B 为非常量时，B 需为非对称 pertensor |
| **MatMul** | `ai.onnx` | 13 | 无 | `tensor(float)`<br>`tensor(float16)` | ✓ | - A 非对称 pertensor <br />- B 对称 perchannel <br />- 当 B 为非常量时，B 需为非对称 pertensor |

## QDQ

| 算子名称 | Domain | Opset | Attributes | T1 输入类型 | T2 输出类型 |
|---------|----|-------|------------|---------|---------|
| **DynamicQuantizeLinear** | `ai.onnx` | 11 | per-tensor | `tensor(float)` | `tensor(int8)` |
| **QuantizeLinear** | `ai.onnx` | 19 | per-tensor<br>per-channel | `tensor(float)` | `tensor(int8)`<br>`tensor(int16)` |
| **DequantizeLinear** | `ai.onnx` | 19 | per-tensor<br>per-channel | `tensor(int8)`<br>`tensor(int16)`<br>`tensor(int32)` | `tensor(float)` |

## Pool

| 算子名称 | Domain | Opset | Attributes | Type |
|---------|----|-------|-------------|----------|
| **AveragePool** | `ai.onnx` | 22 | - | `tensor(float)`<br>`tensor(float16)` |
| **GlobalAveragePool** | `ai.onnx` | 1 | - | `tensor(float)`<br>`tensor(float16)` |
| **MaxPool** | `ai.onnx` | 12 | - | `tensor(float)`<br>`tensor(int8)`<br>`tensor(float16)` |
| **GlobalMaxPool** | `ai.onnx` | 1 | - | `tensor(float)`<br>`tensor(int8)`<br>`tensor(float16)` |

## Reduce

| 算子名称 | Domain | Opset | Attributes | Type |
|---------|----|-------|---------|-------------|
| **ReduceMean** | `ai.onnx` | 18 | axes需为连续常量，如 [1,2] | `tensor(float)`<br>`tensor(float16)`|
| **ReduceMax** | `ai.onnx` | 20 | axes需为连续常量，如 [1,2] | `tensor(float)`<br>`tensor(int8)`<br>`tensor(float16)`|

## Math

| 算子名称 | Domain | Opset | Attributes | Type |
|---------|----|-------|---|-------------|
| **Add** | `ai.onnx` | 14 | - | `tensor(float)`<br>`tensor(float16)` |
| **Sub** | `ai.onnx` | 14 | - |`tensor(float)`<br>`tensor(float16)` |
| **Mul** | `ai.onnx` | 14 | - |`tensor(float)`<br>`tensor(float16)` |
| **Div** | `ai.onnx` | 14 | - |`tensor(float)`<br>`tensor(float16)` |
| **Pow** | `ai.onnx` | 14 | - |`tensor(float)`<br>`tensor(float16)` |
| **Sqrt** | `ai.onnx` | 14 | - |`tensor(float)`<br>`tensor(float16)` |
| **Abs** | `ai.onnx` | 14 | - |`tensor(float)`<br>`tensor(float16)` |
| **Reciprocal** | `ai.onnx` | 14 | - |`tensor(float)`<br>`tensor(float16)` |

## Activation

| 算子名称 | Domain | Opset | Attributes |Type |
|---------|----|-------|---|----------|
| **Sigmoid** | `ai.onnx` | 13 | - | `tensor(float)`<br>`tensor(float16)` |
| **Swish** | `ai.onnx` | 24 | - | `tensor(float)`<br>`tensor(float16)` |
| **HardSigmoid** | `ai.onnx` | 22 | - | `tensor(float)`<br>`tensor(float16)` |
| **HardSwish** | `ai.onnx` | 22 | - | `tensor(float)`<br>`tensor(float16)` |
| **Tanh** | `ai.onnx` | 13 | - | `tensor(float)`<br>`tensor(float16)` |
| **LeakyRelu** | `ai.onnx` | 16 | - | `tensor(float)`<br>`tensor(float16)` |
| **Clip** | `ai.onnx` | 13 | - | `tensor(float)`<br>`tensor(float16)` |
| **Relu** | `ai.onnx` | 14 | - | `tensor(float)`<br>`tensor(float16)` |
| **Elu** | `ai.onnx` | 22 | - | `tensor(float)`<br>`tensor(float16)` |
| **Gelu** | `ai.onnx` | 20 | - | `tensor(float)`<br>`tensor(float16)` |
| **Erf** | `ai.onnx` | 13 | - | `tensor(float)`<br>`tensor(float16)` |
| **Softmax** | `ai.onnx` | 13 | - | `tensor(float)`<br>`tensor(float16)` |

## Tensor

| 算子名称 | Domain | Opset | Attributes |Type |
|---------|----|-------|---|----------|
| **Cast** | `ai.onnx` | 24 | - | All |
| **Concat** | `ai.onnx` | 13 | - | All |
| **Split** | `ai.onnx` | 18 | - | All |
| **Transpose** | `ai.onnx` | 24 | - | All |
| **Unsqueeze** | `ai.onnx` | 24 | - | All |
| **Squeeze** | `ai.onnx` | 24 | - | All |
| **Reshape** | `ai.onnx` | 24 | - | All |
| **Flatten** | `ai.onnx` | 24 | - | All |
| **Gather** | `ai.onnx` | 13 | - | All |
| **Resize** | `ai.onnx` | 19 | - | All |


## Norm

| 算子名称 | Domain | Opset |Attributes | Type |
|---------|----|-------|---------|-------------|
| **LayerNormalization** | `ai.onnx` | 17 | Scale、B 必须为常量 | `tensor(float)`<br>`tensor(float16)` |
| **BatchNormalization** | `ai.onnx` | 15 | Scale、B、input_mean、input_var 必须为常量 | `tensor(float)`<br>`tensor(float16)` |

