sidebar_position: 2

# ONNX Runtime EP Accelerated Operators

This section lists the operators accelerated by **SpaceMITExecutionProvider**, along with their supported specifications.

- [ONNX-OP Reference](https://onnx.ai/onnx/operators/index.html)  
- [ONNX-Contrib-OP Reference](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md)

## Dense

## Dense

### Conv

- **Domain**: ai.onnx  
- **Opset**: 11  
- **Attributes**: `W` must be constant  
- **Type**: T: tensor(float) | tensor(float16)  
- **Notes**:  
  - Supports QDQ quantization format  
  - `W`: symmetric per-channel  
  - `X`: asymmetric per-tensor  

### ConvTranspose

- **Domain**: ai.onnx  
- **Opset**: 11  
- **Attributes**: `W` must be constant  
- **Type**: T: tensor(float) | tensor(float16)  
- **Notes**:  
  - Supports QDQ quantization format  
  - `W`: symmetric per-channel  
  - `X`: asymmetric per-tensor  

### Gemm

- **Domain**: ai.onnx  
- **Opset**: 13  
- **Attributes**: `transA == 0`  
- **Type**: T: tensor(float) | tensor(float16)  
- **Notes**:  
  - Supports QDQ quantization format  
  - `A`: asymmetric per-tensor  
  - `B`: symmetric per-channel  
  - If `B` is non-constant, it must be asymmetric per-tensor  

### MatMul

- **Domain**: ai.onnx  
- **Opset**: 13  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  
- **Notes**:  
  - Supports QDQ quantization format  
  - `A`: asymmetric per-tensor  
  - `B`: symmetric per-channel  
  - If `B` is non-constant, it must be asymmetric per-tensor

## QDQ

### DynamicQuantizeLinear

- **Domain**: ai.onnx  
- **Opset**: 11  
- **Attributes**: Per-tensor only  
- **Type**:  
  - T1: tensor(float)  
  - T2: tensor(int8)  

### QuantizeLinear

- **Domain**: ai.onnx  
- **Opset**: 19  
- **Attributes**: Per-tensor, per-channel  
- **Type**:  
  - T1: tensor(float)  
  - T2: tensor(int8) | tensor(int16)  

### DequantizeLinear

- **Domain**: ai.onnx  
- **Opset**: 19  
- **Attributes**: Per-tensor, per-channel  
- **Type**:  
  - T1: tensor(int8) | tensor(int16) | tensor(int32)  
  - T2: tensor(float)  

## Pooling

### AveragePool

- **Domain**: ai.onnx  
- **Opset**: 22  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### GlobalAveragePool

- **Domain**: ai.onnx  
- **Opset**: 1  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### MaxPool

- **Domain**: ai.onnx  
- **Opset**: 12  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(int8) | tensor(float16)  

### GlobalMaxPool

- **Domain**: ai.onnx  
- **Opset**: 1  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(int8) | tensor(float16)  

## Reduce

### ReduceMean

- **Domain**: ai.onnx  
- **Opset**: 18  
- **Attributes**: `axes` must be a continuous constant range (e.g., `[1, 2]`)  
- **Type**: T: tensor(float) | tensor(float16)  

### ReduceMax

- **Domain**: ai.onnx  
- **Opset**: 20  
- **Attributes**: `axes` must be a continuous constant range (e.g., `[1, 2]`)  
- **Type**: T: tensor(float) | tensor(int8) | tensor(float16)

## Math

### Add

- **Domain**: ai.onnx  
- **Opset**: 14  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Sub

- **Domain**: ai.onnx  
- **Opset**: 14  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Mul

- **Domain**: ai.onnx  
- **Opset**: 14  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Div

- **Domain**: ai.onnx  
- **Opset**: 14  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Pow

- **Domain**: ai.onnx  
- **Opset**: 14  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Sqrt

- **Domain**: ai.onnx  
- **Opset**: 14  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Abs

- **Domain**: ai.onnx  
- **Opset**: 14  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Reciprocal

- **Domain**: ai.onnx  
- **Opset**: 14  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

## Activation

### Sigmoid

- **Domain**: ai.onnx  
- **Opset**: 13  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Swish

- **Domain**: ai.onnx  
- **Opset**: 24  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### HardSigmoid

- **Domain**: ai.onnx  
- **Opset**: 22  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### HardSwish

- **Domain**: ai.onnx  
- **Opset**: 22  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Tanh

- **Domain**: ai.onnx  
- **Opset**: 13  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### LeakyRelu

- **Domain**: ai.onnx  
- **Opset**: 16  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Clip

- **Domain**: ai.onnx  
- **Opset**: 13  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Relu

- **Domain**: ai.onnx  
- **Opset**: 14  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Elu

- **Domain**: ai.onnx  
- **Opset**: 22  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Gelu

- **Domain**: ai.onnx  
- **Opset**: 20  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Erf

- **Domain**: ai.onnx  
- **Opset**: 13  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

### Softmax

- **Domain**: ai.onnx  
- **Opset**: 13  
- **Attributes**: —  
- **Type**: T: tensor(float) | tensor(float16)  

## Tensor

### Cast

- **Domain**: ai.onnx  
- **Opset**: 24  
- **Attributes**: —  
- **Type**: All  

### Concat

- **Domain**: ai.onnx  
- **Opset**: 13  
- **Attributes**: —  
- **Type**: All  

### Split

- **Domain**: ai.onnx  
- **Opset**: 18  
- **Attributes**: —  
- **Type**: All  

### Transpose

- **Domain**: ai.onnx  
- **Opset**: 24  
- **Attributes**: —  
- **Type**: All  

### Unsqueeze

- **Domain**: ai.onnx  
- **Opset**: 24  
- **Attributes**: —  
- **Type**: All  

### Squeeze

- **Domain**: ai.onnx  
- **Opset**: 24  
- **Attributes**: —  
- **Type**: All  

### Reshape

- **Domain**: ai.onnx  
- **Opset**: 24  
- **Attributes**: —  
- **Type**: All  

### Flatten

- **Domain**: ai.onnx  
- **Opset**: 24  
- **Attributes**: —  
- **Type**: All  

### Gather

- **Domain**: ai.onnx  
- **Opset**: 13  
- **Attributes**: —  
- **Type**: All  

### Resize

- **Domain**: ai.onnx  
- **Opset**: 19  
- **Attributes**: —  
- **Type**: All  

## Norm

### LayerNormalization

- **Domain**: ai.onnx  
- **Opset**: 17  
- **Attributes**: `Scale` and `B` must be constant  
- **Type**: T: tensor(float) | tensor(float16)  

### BatchNormalization

- **Domain**: ai.onnx  
- **Opset**: 15  
- **Attributes**: `Scale`, `B`, `input_mean`, and `input_var` must be constant  
- **Type**: T: tensor(float) | tensor(float16)  
