sidebar_position: 2

# SpaceMITExecutionProvider加速算子

>+ 本章节罗列SpaceMITExecutionProvider支持的加速算子及其规格
>+ [ONNX-OP描述参考](https://onnx.ai/onnx/operators/index.html)
>+ [ONNX-Contrib-OP描述参考](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md)

- [SpaceMITExecutionProvider加速算子](#spacemitexecutionprovider加速算子)
  - [Dense](#dense)
    - [**Conv**](#conv)
    - [**ConvTranspose**](#convtranspose)
    - [**Gemm**](#gemm)
    - [**MatMul**](#matmul)
  - [QDQ](#qdq)
    - [**DynamicQuantizeMatMul**](#dynamicquantizematmul)
    - [**MatMulInteger**](#matmulinteger)
    - [**DynamicQuantizeLinear**](#dynamicquantizelinear)
    - [**QuantizeLinear**](#quantizelinear)
    - [**DequantizeLinear**](#dequantizelinear)
  - [Pool](#pool)
    - [**AveragePool**](#averagepool)
    - [**GlobalAveragePool**](#globalaveragepool)
    - [**MaxPool**](#maxpool)
    - [**GlobalMaxPool**](#globalmaxpool)
  - [Reduce](#reduce)
    - [**ReduceMean**](#reducemean)
    - [**ReduceMax**](#reducemax)
  - [Math](#math)
    - [**Add**](#add)
    - [**Sub**](#sub)
    - [**Mul**](#mul)
    - [**Div**](#div)
    - [**Pow**](#pow)
    - [**Sqrt**](#sqrt)
    - [**Abs**](#abs)
    - [**Log**](#log)
    - [**Reciprocal**](#reciprocal)
    - [**Sin**](#sin)
    - [**Cos**](#cos)
    - [**Tan**](#tan)
    - [**Sinh**](#sinh)
    - [**Cosh**](#cosh)
    - [**Floor**](#floor)
    - [**Ceil**](#ceil)
  - [Activation](#activation)
    - [**Sigmoid**](#sigmoid)
    - [**Swish**](#swish)
    - [**HardSigmoid**](#hardsigmoid)
    - [**HardSwish**](#hardswish)
    - [**Tanh**](#tanh)
    - [**LeakyRelu**](#leakyrelu)
    - [**Clip**](#clip)
    - [**Relu**](#relu)
    - [**Elu**](#elu)
    - [**Gelu**](#gelu)
    - [**Celu**](#celu)
    - [**Softplus**](#softplus)
    - [**Softsign**](#softsign)
    - [**Erf**](#erf)
    - [**Softmax**](#softmax)
  - [Tensor](#tensor)
    - [**Cast**](#cast)
    - [**Concat**](#concat)
    - [**Split**](#split)
    - [**Transpose**](#transpose)
    - [**Unsqueeze**](#unsqueeze)
    - [**Squeeze**](#squeeze)
    - [**Reshape**](#reshape)
    - [**Flatten**](#flatten)
    - [**Gather**](#gather)
    - [**Slice**](#slice)
    - [**Resize**](#resize)
    - [**Where**](#where)
  - [Norm](#norm)
    - [**LayerNormalization**](#layernormalization)
    - [**InstanceNormalization**](#instancenormalization)
    - [**BatchNormalization**](#batchnormalization)
  - [Compare](#compare)
    - [**Equal**](#equal)
    - [**Greater**](#greater)
    - [**GreaterOrEqual**](#greaterorequal)
    - [**Less**](#less)
    - [**LessOrEqual**](#lessorequal)

## Dense
### **Conv**
>+ Domain: ai.onnx
>+ Opset: 11
>+ Attributes: W需为常量
>+ Type: T：tensor(float) | tensor(float16)
>+ Notes: 支持QDQ量化格式，W对称perchannel，X非对称pertensor，支持1D、2D、3D

### **ConvTranspose**
>+ Domain: ai.onnx
>+ Opset: 11
>+ Attributes: W需为常量
>+ Type: T：tensor(float)
>+ Notes: 支持QDQ量化格式，W对称perchannel，X非对称pertensor，支持1D、2D

### **Gemm**
>+ Domain: ai.onnx
>+ Opset: 13
>+ Attributes: transA==0
>+ Type: T：tensor(float) | tensor(float16)
>+ Notes: 支持QDQ量化格式，A非对称pertensor，B对称perchannel，当B为非常量时，B需为非对称pertensor

### **MatMul**
>+ Domain: ai.onnx
>+ Opset: 13
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)
>+ Notes: 支持QDQ量化格式，A非对称pertensor，B对称perchannel，当B为非常量时，B需为非对称pertensor

## QDQ
### **DynamicQuantizeMatMul**
>+ Domain: com.microsoft
>+ Opset: 1（参考ONNX-Contrib）
>+ Attributes:
>+ Type: T1：tensor(float)
>+ Type: T2：tensor(float)

### **MatMulInteger**
>+ Domain: ai.onnx
>+ Opset: 10
>+ Attributes:
>+ Type: T1：tensor(int8)
>+ Type: T2：tensor(int32)

### **DynamicQuantizeLinear**
>+ Domain: ai.onnx
>+ Opset: 11
>+ Attributes: 仅支持pertensor
>+ Type: T1：tensor(float)
>+ Type: T2：tensor(int8)

### **QuantizeLinear**
>+ Domain: ai.onnx
>+ Opset: 19
>+ Attributes: 仅支持pertensor、perchannel
>+ Type: T1：tensor(float)
>+ Type: T2：tensor(int8) | tensor(uint8)

### **DequantizeLinear**
>+ Domain: ai.onnx
>+ Opset: 19
>+ Attributes: 仅支持pertensor、perchannel
>+ Type: T1：tensor(int8) | tensor(uint8) | tensor(int32)
>+ Type: T2：tensor(float)

## Pool
### **AveragePool**
>+ Domain: ai.onnx
>+ Opset: 22
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **GlobalAveragePool**
>+ Domain: ai.onnx
>+ Opset: 1
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **MaxPool**
>+ Domain: ai.onnx
>+ Opset: 12
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **GlobalMaxPool**
>+ Domain: ai.onnx
>+ Opset: 1
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

## Reduce
### **ReduceMean**
>+ Domain: ai.onnx
>+ Opset: 18
>+ Attributes: axes需为连续常量，如[1,2]
>+ Type: T：tensor(float) | tensor(float16)

### **ReduceMax**
>+ Domain: ai.onnx
>+ Opset: 20
>+ Attributes: axes需为连续常量，如[1,2]
>+ Type: T：tensor(float) | tensor(float16)

## Math
### **Add**
>+ Domain: ai.onnx
>+ Opset: 14
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Sub**
>+ Domain: ai.onnx
>+ Opset: 14
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Mul**
>+ Domain: ai.onnx
>+ Opset: 14
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Div**
>+ Domain: ai.onnx
>+ Opset: 14
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Pow**
>+ Domain: ai.onnx
>+ Opset: 14
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Sqrt**
>+ Domain: ai.onnx
>+ Opset: 14
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Abs**
>+ Domain: ai.onnx
>+ Opset: 14
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Log**
>+ Domain: ai.onnx
>+ Opset: 13
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Reciprocal**
>+ Domain: ai.onnx
>+ Opset: 14
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Sin**
>+ Domain: ai.onnx
>+ Opset: 7
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Cos**
>+ Domain: ai.onnx
>+ Opset: 7
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Tan**
>+ Domain: ai.onnx
>+ Opset: 7
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Sinh**
>+ Domain: ai.onnx
>+ Opset: 9
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Cosh**
>+ Domain: ai.onnx
>+ Opset: 9
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Floor**
>+ Domain: ai.onnx
>+ Opset: 6
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Ceil**
>+ Domain: ai.onnx
>+ Opset: 6
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

## Activation
### **Sigmoid**
>+ Domain: ai.onnx
>+ Opset: 13
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Swish**
>+ Domain: ai.onnx
>+ Opset: 24
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **HardSigmoid**
>+ Domain: ai.onnx
>+ Opset: 22
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **HardSwish**
>+ Domain: ai.onnx
>+ Opset: 22
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Tanh**
>+ Domain: ai.onnx
>+ Opset: 13
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **LeakyRelu**
>+ Domain: ai.onnx
>+ Opset: 16
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Clip**
>+ Domain: ai.onnx
>+ Opset: 13
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Relu**
>+ Domain: ai.onnx
>+ Opset: 14
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Elu**
>+ Domain: ai.onnx
>+ Opset: 22
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Gelu**
>+ Domain: ai.onnx
>+ Opset: 20
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Celu**
>+ Domain: ai.onnx
>+ Opset: 12
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Softplus**
>+ Domain: ai.onnx
>+ Opset: 1
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Softsign**
>+ Domain: ai.onnx
>+ Opset: 1
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Erf**
>+ Domain: ai.onnx
>+ Opset: 13
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **Softmax**
>+ Domain: ai.onnx
>+ Opset: 13
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

## Tensor
### **Cast**
>+ Domain: ai.onnx
>+ Opset: 24
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)

### **Concat**
>+ Domain: ai.onnx
>+ Opset: 13
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)

### **Split**
>+ Domain: ai.onnx
>+ Opset: 18
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)

### **Transpose**
>+ Domain: ai.onnx
>+ Opset: 24
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16) | tensor(int8)

### **Unsqueeze**
>+ Domain: ai.onnx
>+ Opset: 24
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)

### **Squeeze**
>+ Domain: ai.onnx
>+ Opset: 24
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)

### **Reshape**
>+ Domain: ai.onnx
>+ Opset: 24
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)

### **Flatten**
>+ Domain: ai.onnx
>+ Opset: 24
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)

### **Gather**
>+ Domain: ai.onnx
>+ Opset: 13
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)

### **Slice**
>+ Domain: ai.onnx
>+ Opset: 13
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)

### **Resize**
>+ Domain: ai.onnx
>+ Opset: 19
>+ Attributes:
>+ Type: T：tensor(float) | tensor(int8)

### **Where**
>+ Domain: ai.onnx
>+ Opset: 9
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)

## Norm
### **LayerNormalization**
>+ Domain: ai.onnx
>+ Opset: 17
>+ Attributes: Scale、B需为常量
>+ Type: T：tensor(float) | tensor(float16)

### **InstanceNormalization**
>+ Domain: ai.onnx
>+ Opset: 6
>+ Attributes:
>+ Type: T：tensor(float) | tensor(float16)

### **BatchNormalization**
>+ Domain: ai.onnx
>+ Opset: 15
>+ Attributes: Scale、B、input_mean、input_var需为常量
>+ Type: T：tensor(float) | tensor(float16)

## Compare
### **Equal**
>+ Domain: ai.onnx
>+ Opset: 11
>+ Attributes:
>+ Type: T1：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)
>+ Type: T2：tensor(uint8) | tensor(bool)

### **Greater**
>+ Domain: ai.onnx
>+ Opset: 9
>+ Attributes:
>+ Type: T1：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)
>+ Type: T2：tensor(uint8) | tensor(bool)

### **GreaterOrEqual**
>+ Domain: ai.onnx
>+ Opset: 12
>+ Attributes:
>+ Type: T1：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)
>+ Type: T2：tensor(uint8) | tensor(bool)

### **Less**
>+ Domain: ai.onnx
>+ Opset: 9
>+ Attributes:
>+ Type: T1：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)
>+ Type: T2：tensor(uint8) | tensor(bool)

### **LessOrEqual**
>+ Domain: ai.onnx
>+ Opset: 12
>+ Attributes:
>+ Type: T1：tensor(float) | tensor(float16) | tensor(int32) | tensor(uint32) | tensor(int8) | tensor(uint8) | tensor(bool)
>+ Type: T2：tensor(uint8) | tensor(bool)
