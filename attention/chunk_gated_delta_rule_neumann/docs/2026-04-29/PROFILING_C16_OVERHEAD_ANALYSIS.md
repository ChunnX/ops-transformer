# C16 Neumann profiling 开销分析 - 2026-04-29

## 背景

用户指出：chunk 从 64 减到 16 后，按直觉 Stage2/Stage3 的矩阵乘应明显变小，但 profiling 中 AIC/AIV 时间和搬运相关指标接近 4 倍放大，需要核对是否存在重复计算。

对比对象：
- 普通 `ChunkGatedDeltaRule`：`chunkSize=64`。
- 当前 `ChunkGatedDeltaRuleNeumann`：`chunkSize=16`，stage selector 指向 `stage1_C16_FP16_CVInverse.h`。

## profiling 关键现象

从 `performance_raw.csv` 的 Phase4 measurement 看：

| 指标 | 平均 Neumann / Normal | 说明 |
|---|---:|---|
| Task Duration | 4.07x | 整体耗时约 4 倍 |
| AICore time | 4.09x | AIC 总时间约 4 倍 |
| AIC MAC time | 1.52x | 真实 cube MAC 时间不是 4 倍 |
| AIC scalar time | 5.01x | per-matmul/per-chunk 控制开销显著放大 |
| AIC MTE2 time | 3.16x | 搬入开销接近 chunk 数放大 |
| AIC FixPipe time | 2.56x | 搬出/写回开销明显放大 |
| AIV time | 4.08x | AIV 总时间约 4 倍 |
| AIV vec time | 0.68x | 纯向量计算不是瓶颈 |
| AIV scalar time | 2.78x | AIV 控制/同步开销放大 |
| AIV MTE2/MTE3 time | 2.93x / 3.39x | AIV 搬运接近 chunk 数放大 |

典型 case `prefill_b8_s1024_d128_h32_hkv8`：
- Normal duration：`7416.32 us`
- Neumann duration：`34102.62 us`
- Neumann/Normal：`4.60x`
- AIC MAC time：`816.469 us -> 1073.457 us`，仅 `1.32x`
- AIC scalar time：`3089.659 us -> 15005.163 us`，`4.86x`

结论：profiling 中 4 倍主要不是纯 MAC 计算量翻 4 倍，而是 C16 带来的 chunk 数、matmul 调用次数、同步和搬运次数放大。

## 代码级原因

### 1. Stage2 总计算量不会随 chunk=16 降到 1/4

`Stage2` 每个 chunk 做 3 个 matmul：

```text
CalVPrime:    [C, Dk] x [Dv, Dk]^T -> [C, Dv]
CalAttnInter: [C, Dk] x [Dv, Dk]^T -> [C, Dv]
CalStateNew:  [Dv, C]^T x [C, Dk] -> [Dv, Dk]
```

单 chunk 计算量约 `3 * C * Dv * Dk`。
固定序列长度 `S` 下 chunk 数为 `S / C`，所以总量：

```text
(S / C) * 3 * C * Dv * Dk = 3 * S * Dv * Dk
```

因此 Stage2 总 MAC 理论上基本不随 chunk 从 64 改 16 而减少。变化的是每个 matmul 更小、调用次数变 4 倍，利用率和调度效率变差。

此外 Stage2 AIV 每个 chunk 都会 `CopyIn/CopyOut` 当前 state，搬运 `Dv * Dk`，该搬运次数会随 chunk 数变 4 倍。

### 2. Stage3 MAC 会降到约 1/4，但调用次数变 4 倍

`Stage3` 每个 chunk 做：

```text
masked_qkt [C, C] x v_inner [C, Dv] -> [C, Dv]
```

总 MAC 为：

```text
(S / C) * C * C * Dv = S * C * Dv
```

所以 C16 相对 C64 的 Stage3 MAC 是 1/4。但每个 chunk 的 AIV mask、copy/cast、同步和 matmul launch 次数是 4 倍，profiling 会体现为 MTE/scalar 开销放大。

### 3. Stage1 当前并没有真正切到 FP16 MM

虽然当前 selector 指向：

```cpp
#include "chunk_gated_delta_rule_neumann_stage1_C16_FP16_CVInverse.h"
```

但该文件内真实 MatmulType 仍是：

```cpp
using aT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using bT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using cT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
```

host tiling 也仍是：

```cpp
DT_FLOAT / DT_FLOAT / DT_FLOAT
```

因此当前 `C16_FP16_CVInverse` 只是文件名是 FP16，实际 AIC matmul 仍是 FP32 GM 输入、FP32 输出。这会直接影响 cube 利用率和耗时判断。

### 4. Stage1 matmul 调用次数显著增加

普通 C64 每 chunk 主要 AIC matmul 调用：

```text
Stage1: kk, qk, inverse combine x2, inv@k, inv@v = 6
Stage2: 3
Stage3: 1
合计约 10 次/chunk
```

当前 C16 Neumann 每 chunk：

```text
Stage1: kk, qk, Neumann Lpow x3, Neumann INV@Lpow x3, inv@k, inv@v = 10
Stage2: 3
Stage3: 1
合计约 14 次/chunk
```

chunk 数又从 `S/64` 变为 `S/16`，因此 matmul 调用次数约：

```text
(14 * S/16) / (10 * S/64) = 5.6x
```

这与 profiling 中 AIC scalar time 约 `5x` 的现象一致。

## 已发现并修复的重复写

在 Neumann `InverseCompute()` 中，代码原先写了两份 `I-L`：

1. 写 `attnWsGm_`
2. 写 `INVWsGm_`

但 Round1 立刻执行：

```cpp
NeumannLpowCompute(L_powWsGm_, attnWsGm_);
```

会用 `L @ L` 覆盖 `attnWsGm_`，所以 `attnWsGm_` 的初始 `I-L` 是死写。

已删除该无效写入，保留：
- `L_powWsGm_ = L`
- `INVWsGm_ = I - L`

影响：
- 编译通过。
- 三方精度 smoke 通过。
- 单 case profiling 中 MTE/FixPipe 有小幅下降，但总体耗时无明显改善，说明 4x 主因不是这个死写，而是小 chunk 导致的 per-chunk 调用/同步/搬运放大。

## 当前判断

1. 没有发现 Stage2/Stage3 对同一 chunk 的重复计算；主要是 chunk 数变 4 倍导致调用次数变 4 倍。
2. 用户关于 Stage2 “应该小 3/4” 的假设不成立：Stage2 总 MAC 与 chunkSize 基本无关。
3. Stage3 和 Stage1 的部分矩阵乘总 MAC 确实下降，但被小矩阵 FP32 matmul 低效率、更多同步、更多 MTE/FixPipe、更多 state 搬运抵消。
4. 当前 `C16_FP16_CVInverse` 不是实际 FP16 MM，是需要优先修正的命名/实现不一致问题。
5. 真正想降低 profiling 中的 4x，需要减少 per-chunk 次数或合并跨 chunk 工作，而不是只把 C 从 64 改 16。

## 建议下一步

1. 先把 `C16_FP16_CVInverse` 名字和真实实现对齐：要么改回 FP32 命名，要么实现真实 half workspace + FP16 A/B + FP32 C。
2. 优先推进 C64 方案：保持 chunk 数不变，避免 Stage2 state copy 和小 matmul launch 放大。
3. 将 Neumann 三轮 `INV += tmp` 从 AIV 搬到 AIC FixPipe atomic add，减少 AIV 同步和 GM 往返。
4. 对 Stage2 state 读写做优化：当前每 chunk 搬 `Dv*Dk` state，C16 会直接 4 倍放大。
