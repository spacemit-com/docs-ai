# 设计理念

## 概述

为了加速 AI 计算，芯片企业设计了多种专用处理器架构，如 GPGPU、NPU、TPU 等。这些专用处理器架构在执行调度代码及应用层代码时，需要主控 CPU 的配合，如下图所示。因此，通常需要构建复杂的异构调度系统来协调 CPU 和 XPU 的额外数据交互和同步。
![architect](./images/architect.webp)

为了保证AI算力的通用性和易用性，进迭时空基于自身 CPU 核的研发能力，以标准 RISC-V 核为基础，创新性地在 CPU 内集成 TensorCore，以 RISC-V 指令集为统一的软硬件接口，驱动 Scalar 标量算力、Vector 向量算力和 Matrix AI 算力，支持软件和 AI 模型同时在 RISC-V AI 核上运行，并通过程序正常跳转实现软件和 AI 模型之间的事件和数据交互，进而完成整个 AI 应用执行。我们把这种以 RISC-V 指令集为统一的软硬件接口，驱动 Scalar 标量算力、Vector 向量算力和 Matrix AI 算力的技术，叫做 **同构融合技术**，这种具有 AI 算力的 CPU 称为 **AI CPU** 或者**智算核**。

**AI CPU** 保留了 CPU 编程模型，开发者使用 Linux 线程就可以驱动 AI 算力，在硬件层面上对 AI 算力和通用 CPU 进行了更高层次的封装，开发者无需关注异构调度和复杂的驱动管理；并且以 ***RISC-V*** CPU 为基础，可以便捷接入开源生态，保留开源软件的使用习惯；此外 **AI CPU** 兼具并行计算和逻辑计算能力，适配 MOE 模型推理。

在 AI 计算中，Scalar 标量算力，Vector 向量算力及 Matrix AI 算力都会被用到。其中：
- Scalar 标量算力，采用 ***RISC-V*** 标准指令集提供；
- Vector 向量算力，采用 ***RISC-V*** vector 1.0 指令集提供；
- Matrix AI算力，采用 ***RISC-V*** [Matrix扩展指令集](./instruction.md) 提供。

## 架构实践

目前进迭时空已经发布了带有 **AI CPU** 的第一代芯片 **K1**。K1 包含：

- **4 × X60** 通用 RISC-V CPU 核  
- **4 × A60** 智算核（AI CPU）

在 **A60** 中，RISC-V Vector 的位宽为 256 位，其 matrix 及 vector 理论算力展示如下，可参考[算力换算方法](./instruction.md)。
- Matrix 算力: 
  - 0.5 TOPS/Core (Int8), 2 TOPS/Cluster (Int8) 
- Vector 算力：
  - 0.128 TOPS/Core (Int8), 0.5 TOPS/Cluster (Int8)
  - 0.064 TOPS/Core (FP16), 0.25 TOPS/Cluster (FP16)
  - 0.032 TOPS/Core (FP32)

基于开源项目 [cpfp](https://github.com/pigirons/cpufp)，对 K1 **AI CPU** 中的 **A60** 核进行测试，实测数据如下：

```
$ ./cpufp --thread_pool=[0]
Number Threads: 1
Thread Pool Binding: 0
---------------------------------------------------------------
| Instruction Set | Core Computation       | Peak Performance |
| ime             | vmadot(s32,s8,s8)      | 511.53 GOPS      |
| ime             | vmadotu(u32,u8,u8)     | 511.5 GOPS       |
| ime             | vmadotus(s32,u8,s8)    | 511.53 GOPS      |
| ime             | vmadotsu(s32,s8,u8)    | 511.51 GOPS      |
| ime             | vmadotslide(s32,s8,s8) | 511.51 GOPS      |
| vector          | vfmacc.vf(f16,f16,f16) | 66.722 GFLOPS    |
| vector          | vfmacc.vv(f16,f16,f16) | 63.936 GFLOPS    |
| vector          | vfmacc.vf(f32,f32,f32) | 33.36 GFLOPS     |
| vector          | vfmacc.vv(f32,f32,f32) | 31.968 GFLOPS    |
| vector          | vfmacc.vf(f64,f64,f64) | 16.679 GFLOPS    |
| vector          | vfmacc.vv(f64,f64,f64) | 15.985 GFLOPS    |
---------------------------------------------------------------
For cluster 0(with ime extension), 4 cores:
$ ./cpufp --thread_pool=[0-3]
Number Threads: 4
Thread Pool Binding: 0 1 2 3
---------------------------------------------------------------
| Instruction Set | Core Computation       | Peak Performance |
| ime             | vmadot(s32,s8,s8)      | 2.046 TOPS       |
| ime             | vmadotu(u32,u8,u8)     | 2.0462 TOPS      |
| ime             | vmadotus(s32,u8,s8)    | 2.0461 TOPS      |
| ime             | vmadotsu(s32,s8,u8)    | 2.0462 TOPS      |
| ime             | vmadotslide(s32,s8,s8) | 2.0461 TOPS      |
| vector          | vfmacc.vf(f16,f16,f16) | 266.88 GFLOPS    |
| vector          | vfmacc.vv(f16,f16,f16) | 255.75 GFLOPS    |
| vector          | vfmacc.vf(f32,f32,f32) | 133.43 GFLOPS    |
| vector          | vfmacc.vv(f32,f32,f32) | 127.85 GFLOPS    |
| vector          | vfmacc.vf(f64,f64,f64) | 66.709 GFLOPS    |
| vector          | vfmacc.vv(f64,f64,f64) | 63.935 GFLOPS    |
---------------------------------------------------------------
For 2 clusters, 8 cores:
$ ./cpufp --thread_pool=[0-7]
Number Threads: 8
Thread Pool Binding: 0 1 2 3 4 5 6 7
---------------------------------------------------------------
| Instruction Set | Core Computation       | Peak Performance |
| vector          | vfmacc.vf(f16,f16,f16) | 533.65 GFLOPS    |
| vector          | vfmacc.vv(f16,f16,f16) | 511.45 GFLOPS    |
| vector          | vfmacc.vf(f32,f32,f32) | 266.89 GFLOPS    |
| vector          | vfmacc.vv(f32,f32,f32) | 255.75 GFLOPS    |
| vector          | vfmacc.vf(f64,f64,f64) | 133.42 GFLOPS    |
| vector          | vfmacc.vv(f64,f64,f64) | 127.86 GFLOPS    |
---------------------------------------------------------------
```