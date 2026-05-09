# GMMSQ算子kernel实现设计说明书

## 总体设计

> 整体的代码实现可以参考[grouped_matmul_finalize_routing](../../../../grouped_matmul_finalize_routing/)算子，核心差异点只在于后处理部分本gmmsq需要将N分两半处理，并进行SwiGLU+Quant操作。
> - Tiling参考代码: `gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.cpp`
> - Kernel参考代码：`gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block`

TODO: 总体设计说明

## Tiling - 拦截与校验

TODO: 参考gmmfr给gmmsq添加新的tiling模板，这一部分主要完成如何实现新的tiling模板，适配gmmsq要求的拦截与校验

## Tiling - tiling data定义与tiling切分

TODO: 这一部分主要完成tiling data的定义和切分算法的实现

## kernel - buffer分配

### L1 buffer分配

> 参考gmmfr分配方案，开DB

``` plain
带bias：
L1 Address Space (0 ~ 512 KB):
├─ [0x00000, 0x10000)  WeightL1_P0        64 KB
├─ [0x10000, 0x11000)  Bias_P0             4 KB
├─ [0x11000, 0x19000)  ScaleAL1_P0        32 KB
├─ [0x19000, 0x21000)  ScaleBL1_P0        32 KB
├─ [0x21000, 0x40000)  AL1_P0            124 KB
├─ [0x40000, 0x5F000)  AL1_P1            124 KB
├─ [0x5F000, 0x67000)  ScaleBL1_P1        32 KB
├─ [0x67000, 0x6F000)  ScaleAL1_P1        32 KB
├─ [0x6F000, 0x70000)  Bias_P1             4 KB
└─ [0x70000, 0x80000)  WeightL1_P1        64 KB
```

``` plain
无bias：
L1 Address Space (0 ~ 512 KB):
├─ [0x00000, 0x10000)  WeightL1_P0        64 KB
├─ [0x10000, 0x18000)  ScaleAL1_P0        32 KB
├─ [0x18000, 0x20000)  ScaleBL1_P0        32 KB
├─ [0x20000, 0x40000)  AL1_P0            128 KB
├─ [0x40000, 0x60000)  AL1_P1            128 KB
├─ [0x60000, 0x68000)  ScaleBL1_P1        32 KB
├─ [0x68000, 0x70000)  ScaleAL1_P1        32 KB
└─ [0x70000, 0x80000)  WeightL1_P1        64 KB
```

### L0 buffer分配

| Buffer | Position | Size | Purpose |
|---|---|---|---|
| L0A	| A2	| 64 KB	| A-matrix (activation) tile |
| L0B	| B2	| 64 KB	| B-matrix (weight) tile |
| L0C	| CO1	| 256×256 floats	| MMA accumulation |
| BiasTable	| C2	| 4 KB	| Bias | lookup table |

### UB buffer分配

> 存在3种划分，GMM划分方法、GMMFR划分方法，第三种是针对GMMSQ适配的划分:

对于GMMSQ来说还需要额外的空间是:

- yF32保存gmm fp32输出结果
- maxB16 32个指数部分最大值
- scaleE8M0 结果的buffer，搬出到GM（存在性能瓶颈点，搬运小包scatter可能搬运性能很差，单独分配个空间尝试优化性能）
- yFp8 结果（单独一个buffer，避免y写出GM阻塞下一轮计算）

| Buffer | Offset | Size | Purpose |
|--------|--------|------|---------|
| `weightLowBit_` | 0 | 64 KB | Packed 4-bit weight input |
| `weightHighBit_` | 64 KB | 128 KB | Anti-quantized fp8 weight |
| `yF32Buffer_` | 64 KB | 128 KB | MMA fixpipe output (reuses weightHighBit_ space) |
| `bias_` | 192 KB | 1 KB | Bias input |
| `biasRescale_` | 193 KB | 1 KB | Rescaled bias |
| `maxB16Buffer_` | 194 KB | 2 KB | Per-block max exponent (ComputeMaxExp) |
| `scaleE8M0Buffer_` | 196 KB | 16 KB | MX E8M0 scales |
| `yFp8Buffer_` | 212 KB | 32 KB | Final quantized FP8/FP4 output |

## kernel 计算流

TODO：继续补充kernel部分完整的计算流，SwiGLU和Quant部分留空，由其他团队成员分工完成

SwiGLU划分计算：

GMM + SwiGLU的gated是通过weight(N, K)划分完成的，N轴划分为两半：

- `W[0:N/2, :]` 前一半用于Swish计算
- `W[N/2:N, :]` 后一半作为gate完成GLU计算

两部分计算共用一个X，可以一起完成：`X[:M, :] @ W[:N, :].T = O[:M, :N]`。对输出结果 `O[:, 0:N/2]` 为Swish分支，`O[:, N/2:N]` 为gate分支。如果基本块选择baseN=256，那么为了完成计算必须同时计算得到Swish和gate结果 `256*256*2*4`，超出UB空间。

为了解决这个问题，可以每个将整个N划分两半，每次取baseN=128构成256的基本块，保证GMM的性能。

> 尾块计算需要按照一半的N计算尾块

## NOTE

- N是128对齐，是因为N轴切两半，一半为输出另一半为gate，所以实际N是64对齐
- tiling按照256*256切分

