# vf_mul_sel_softmaxflashv2_cast_nz_dn.h softmax 计算类型 fp32→fp16 修改说明

## 修改文件

`attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h`

## 修改背景

原文件中 softmax 计算（求 max、exp(x-max)、sum 累加等核心运算）全部在 **fp32** 寄存器中完成，计算结束后结果被 cast 为模板类型 `T`（sum/max 输出）或 `T2`（概率矩阵 P 输出）再存储。本次修改将所有 softmax 中间计算过程改为按 **fp16** 执行。

## 修改内容

### 1. 计算寄存器类型（328 处）

所有 `RegTensor<float>` 声明改为 `RegTensor<half>`，涉及：

| 寄存器类别 | 用途 | 示例变量 |
|---|---|---|
| 输入加载寄存器 | 加载 QK^T 得分矩阵 | `src0` `src1` `src2` `src3` `vreg_x_f32_0`~`vreg_x_f32_7` |
| 求最大值寄存器 | row max 归约 | `max0` `max1` `max2` `max3` `vreg_x_max_f32_b` |
| exp 中间结果寄存器 | exp(x-max) 计算 | `vreg_x_exp_0`~`vreg_x_exp_7` |
| sum 累加寄存器 | row sum 归约 | `vreg_x_sum_0`~`vreg_x_sum_7` `vreg_x_sum0`~`vreg_x_sum3` |
| pScale / ln(pScale) | P-Scale 对数计算 | `vreg_p_scale` `vreg_ln_p_scale` |
| sink / min / 其他中间量 | 辅助计算 | `vreg_sink_input` `vreg_min` `vreg_l0` `vreg_subloop_update` |
| qScale / kScale | per-token-head 量化 scale | `vreg_qscale_vec` `vreg_kscale_val` |
| 临时数据寄存器 | PerTokenHeadTail 路径 | `vreg_data_tmp0`~`vreg_data_tmp3` |

### 2. Mask 粒度调整

```
preg_134 = CreateMask<uint8_t, MaskPattern::ALL>();
```
改为：
```
preg_134 = CreateMask<uint16_t, MaskPattern::ALL>();
```

**原因**：`preg_134` 配合 `preg_135`（已是 `CreateMask<T>`）用于 exp/sum 等运算的元素掩码。fp32 时一个 256-bit 寄存器含 8 个 float 元素，用 `uint8_t` mask；fp16 时含 16 个 half 元素，需用 `uint16_t` mask 以覆盖全部元素。

> 注：`preg_108 = CreateMask<uint16_t, MaskPattern::ALL>()` 原本已是 uint16_t，无需修改。
> `preg_135 = CreateMask<T, MaskPattern::ALL>()` 基于 T 模板参数，T=half 时自动适配。

### 3. 指针类型（VF 函数参数 + wrapper 局部变量）

VF 函数签名中的 `__ubuf__ float *` / `__ubuf__ float*` 参数类型改为 `__ubuf__ half *` / `__ubuf__ half*`，涉及：

- `input_x_local_UB` — 输入得分矩阵指针
- `exp_max_fp32` — exp(max) 中间结果指针
- `new_global_sum` — softmax sum 输出指针
- `new_global_max` — softmax max 输出指针
- `pre_loop_max` / `pre_loop_sum` / `first_loop_sum` — Mxfp8 跨 loop 状态指针
- `qScaleUb` / `kScaleUb` — per-token-head 量化 scale 指针
- wrapper 函数中 `(__ubuf__ T*)` 赋值给这些指针的局部变量

### 4. LoadAlign 模板参数

```
LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(...)
```
改为：
```
LoadAlign<half, MicroAPI::LoadDist::DIST_BRC_B32>(...)
```

涉及 kScale 广播加载（PerTokenHead / PerTokenHeadTail 路径），将 float 广播改为 half 广播。

### 5. Duplicate 标量类型

```
Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_x_sum_0, 0, preg_134);
```
改为：
```
Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, half>(vreg_x_sum_0, 0, preg_134);
```

sum 累加寄存器初始化为 0 时，标量类型从 float 改为 half，与寄存器类型一致。

### 6. pScale 类型转换

```
Duplicate(vreg_p_scale, static_cast<float>(pScale));
```
改为：
```
Duplicate(vreg_p_scale, static_cast<half>(pScale));
```

pScale 输入为 `float` 标量（函数参数），加载到 fp16 寄存器前先 cast 为 half。

### 7. preLoop 参数类型（Mxfp8 路径）

```
const LocalTensor<float>& preLoopMaxTensor, const LocalTensor<float>& preLoopSumTensor,
const LocalTensor<float>& firstLoopSumTensor,
```
改为：
```
const LocalTensor<T>& preLoopMaxTensor, const LocalTensor<T>& preLoopSumTensor,
const LocalTensor<T>& firstLoopSumTensor,
```

使跨 loop 的中间状态存储类型与计算类型一致（T=half）。

## 循环次数与性能分析

### 原始 fp32 循环结构

VF 函数中数据布局为 `[ubN][m]`（ubN 列，每列 m 个元素），使用 4 个并行寄存器 src0~src3 处理 4 列：

- 求 max 循环：`for (iter_m < ubN/4)`，每次迭代加载 4 列 × 1 寄存器（8 float = 256 bit）
- exp/sum 循环（非 fp8）：`loopNum = ubN/4`，每次迭代 4 列 × 1 寄存器
- exp/sum 循环（fp8）：`loopNum = ubN/8`，每次迭代 8 列 × 1 寄存器

其中 `m` 由调用方传入，fp32 时 `m = s1RealSizeAlign32 >> 1 = halfS1RealSize`，恰好等于 1 个 fp32 寄存器的元素数（8）。

### fp16 修改后的循环结构

改为 fp16 后，**软件循环次数保持不变**（`ubN/4`、`ubN/8`），但每条指令的硬件吞吐量翻倍：

| 指令 | fp32 每次处理元素 | fp16 每次处理元素 | 提升 |
|---|---|---|---|
| `LoadAlign` | 8 float (256 bit) | 16 half (256 bit) | 2× |
| `Max` | 8 float | 16 half | 2× |
| `FusedExpSub` | 8 float | 16 half | 2× |
| `Add` | 8 float | 16 half | 2× |
| `Muls` | 8 float | 16 half | 2× |

### 性能提升来源

性能提升来自 **硬件 SIMD 双倍吞吐**，而非减少软件循环次数：

1. **单指令处理元素翻倍**：同一个 256-bit 寄存器，fp32 容纳 8 个元素，fp16 容纳 16 个元素。所有向量指令（Max、Exp、Add、Muls 等）每条指令处理的元素数翻倍。

2. **内存带宽减半**：相同数量的元素，fp16 数据量仅为 fp32 的一半，LoadAlign/StoreAlign 的数据搬运量减半。

3. **寄存器利用率**：fp32 时 `m = halfS1RealSize = 8`，恰好填满 1 个寄存器；fp16 时同一 `m = 8` 仍可填满半个寄存器（8 half = 128 bit），硬件自动以 256-bit 宽度执行，等效吞吐翻倍。

### 为何不减少循环次数

考虑过方案 A（将 `m` 从 8 增至 16，使每次 LoadAlign 加载 16 half = 1 个完整寄存器），但经分析后**不采用**，原因如下：

1. **`m` 由算法决定**：`m = halfS1RealSize` 是 S1 维度的实际切分大小，由 tiling 层决定。将 `m` 翻倍意味着每个 sub-block 处理的 S1 行数翻倍，需要同步修改 tiling、buffer 分配、cube/vec 同步等多个模块。

2. **数据布局耦合**：VF 函数中的 `src_ub0~3` 按 `m` 步长分列，mask 按 `m >> 2` 步长分列。修改 `m` 需要同步修改所有 stride 计算，且 mask 的 32-bit 对齐约束可能被破坏。

3. **收益有限**：即使不减少循环次数，硬件 SIMD 双倍吞吐已经带来了接近 2× 的性能提升。减少循环次数主要省的是循环开销（分支、指针更新），占总执行时间比例很小。

4. **风险控制**：保持循环结构不变，只做类型替换，是最小化改动风险的策略。

## 未修改部分

| 项目 | 说明 |
|---|---|
| 循环次数 `ubN/4`、`ubN/8` | 保持不变，性能提升来自硬件 SIMD 双倍吞吐 |
| `blockStride = ubN >> 2` | fp8 路径的存储步长，保持不变 |
| 函数参数中的 `float` 标量 | `float deScaleQK`、`float pScale`、`float keepProb`、`float dScale`、`float sinkValue` 等保持 float，这些是宿主机传入的标量，不参与寄存器运算或在运算前 cast |
| `RegTensor<half>` 临时 pack 寄存器 | `vreg_x_exp_even_f16`、`vreg_x_exp_f16_pack` 等原本就是 half/bf16，用于 cast 后的 pack 存储，无需修改 |
| `RegTensor<T2>` 输出寄存器 | `vreg_x_exp_fp8_0` 等，T2 是输出类型（half/bf16/fp8），由 T cast 而来，无需修改 |
| `Cast<T2, T, ...>` 调用 | 从 T（现为 half）cast 到 T2，类型自动随 T 变化，无需修改 |
| `StoreAlign<T, ...>` 调用 | 存储类型为 T（现为 half），自动适配 |
| `INV_LN2` / `LN2` 常量 | Mxfp8 路径中的量化近似计算，保持原值 |
| `Truncate<T, ...>` 调用 | 模板参数 T，自动适配 |
| 调用方 `m` 参数 | 保持 `s1RealSizeAlign32 >> 1` 不变，无需修改调用方 |

## 影响的函数列表

| 函数名 | 路径说明 |
|---|---|
| `ProcessVec1DnNoUpdateVF` | DN 路径，非 flash update 模式 |
| `ProcessVec1DnNoUpdate` | wrapper |
| `ProcessVec1DnUpdateVF` | DN 路径，flash update 模式 |
| `ProcessVec1DnUpdate` | wrapper |
| `ProcessVec1DnNoUpdateMxfp8VF` | DN 路径，Mxfp8 量化，非 update |
| `ProcessVec1DnNoUpdateMxfp8` | wrapper |
| `ProcessVec1DnUpdateMxfp8VF` | DN 路径，Mxfp8 量化，update |
| `ProcessVec1DnUpdateMxfp8` | wrapper |
| `ProcessVec1DnCausalVF` | DN 路径，Causal mask |
| `ProcessVec1DnCausal` | wrapper |
| `ProcessVec1VfDn` | 顶层入口（dispatch update/noUpdate） |
| `ProcessVec1VfDnCausal` | 顶层入口（Causal） |
| `ProcessVec1VfDnMxfp8` | 顶层入口（Mxfp8） |
| `ProcessVec1DnNoUpdatePerTokenHeadVF` | PerTokenHead 量化，非 update |
| `ProcessVec1DnNoUpdatePerTokenHeadTailVF` | PerTokenHead 量化，tail 处理 |
| `ProcessVec1DnUpdatePerTokenHeadVF` | PerTokenHead 量化，update |
| `ProcessVec1DnUpdatePerTokenHeadTailVF` | PerTokenHead 量化，update + tail |

## 注意事项

1. **精度影响**：fp16 的有效位数约 3-4 位十进制，相比 fp32 的 7 位有精度损失。对于 softmax 中的 max 减法操作，当得分矩阵值域跨度较大时可能出现精度下降。

2. **数值范围**：fp16 最大值为 65504，exp(x-max) 的结果范围 [0,1]，在 fp16 范围内。但 sum 累加在极端情况（大量 KV token）下可能超出 fp16 范围，需关注。

3. **调用方适配**：调用这些 VF 函数的上层代码（如 `flash_attention_noquant_block_vec_base.h` 等）需确保传入的 `srcTensor`、`expSumTensor`、`maxTensor` 等 LocalTensor 的 buffer 分配大小适配 half 类型（2 bytes/element vs 4 bytes/element）。

4. **StoreDist 保持不变**：`StoreAlign` 中的 `DIST_NORM_B32` / `DIST_NORM_B16` / `DIST_NORM_B8` 存储对齐模式保持原样，这些是存储对齐指令，与数据类型无直接关联。
