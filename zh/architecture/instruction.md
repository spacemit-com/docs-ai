# AI 指令集

## Matrix 扩展指令集

### 指令集介绍

Matirx 扩展指令主要用于 AI 中最重要的计算场景——矩阵乘法。其的一般形式为：
\[
C += A \times B
\]

其中，**C 为输出矩阵，A 和 B 为输入矩阵**。根据输入/输出矩阵使用的寄存器不同，***RISC-V***社区把 Matirx 扩展指令分为三个方案 ，如下图所示：
![三个指令集对比](./images/matrix_inst.jpg)

- **IME 方案**
   矩阵计算的输入、输出矩阵都使用 vector 寄存器，详情可以加入 [IME subgroup](https://lists.riscv.org/g/tech-integrated-matrix-extension) 查看。

- **VME方案**
   矩阵计算的输入矩阵复用 vector 寄存器，输出矩阵使用专用扩展寄存器，详情可以加入 [VME subgroup](https://lists.riscv.org/g/tech-vme) 查看。

- **AME方案**
   矩阵计算的输入、输出矩阵都使用个专用寄存器，详情可以加入 [AME subgroup](https://lists.riscv.org/g/tech-attached-matrix-extension) 查看。

### 进迭时空指令集

进迭时空自定义的 Matirx 扩展指令集基于 **IME** 方案，可参考[完整说明](https://github.com/spacemit-com/riscv-ime-extension-spec)。

以 **K1（vlen = 256）** 为例，其 Matrix 计算单元形状为：
\[
4 \times 8 \times 4
\]

则单个 Matrix 计算单元的理论算力为：
\[
2 \times 4 \times 8 \times 4 \times 2\text{GHz} = 0.5\text{ TOPS}
\]

### 简单使用

进迭时空基于 IME 规范实现的扩展指令集整体延续了 RISC-V Vector 标准的编程模型：

- 标准 Vector 指令使用方法（链接待补充）
- 可参考[IME 指令集的使用示例](https://github.com/spacemit-com/riscv-ime-extension-spec/tree/master/example)

