# Tiling 与 UT 设计

## 1. 目标

本文定义 `QkvRmsNormRopeCacheWithKScale` 首期 arch35 fast path 的 tiling 策略、tiling data 字段、扩展路线和 UT 验收标准。

首期 fast path 目标是优先保证 kernel 实现复杂度可控：

1. 固定 `D = 128`。
2. `rotation` 接口形态为 optional，但当前 checker 强制存在，dtype 为 `bf16`，shape 为 `[128,128]`。
3. `v_scale` 接口形态为 optional，但当前 checker 强制存在，dtype 为 `float32`，shape 为 `[Nv]`。
4. 首期 fast path 只支持 `Nv == Nk`。
5. `rotation` 完整常驻 L1。
6. 不切 D，不切 Q/K head，不做 Q/K flatten row 均衡切分。
7. 主 tile 只沿 token 维切分，并强制 `tokenTile` 为 2 的倍数。
8. 单次 Cube tile 满足 `M_tile <= 256`，保证 L0C double buffer。

## 2. 首期 Tiling 原则

记：

```text
Hqk  = Nq + Nk
Nv   = Nk
Htot = Nq + Nk + Nv = Nq + 2 * Nk
D    = 128
```

这里 `Nv = Nk` 是首期 fast path 的显式支持约束，不是由接口隐式推导。host 仍从 `head_nums=[Nq,Nk,Nv]` 读取三者，并在进入本 tiling 前校验 `Nv == Nk`。

首期 tile 形态固定为：

```text
dimTile    = 128
qkHeadTile = Hqk
vHeadTile  = Nv

actualTokenTile = min(tokenTile, T - tileStart)
rowTile         = tokenTile * Hqk
M_tile          = actualTokenTile * Hqk
```

`rowTile` 和 `M_tile` 的语义必须区分：

1. `rowTile` 是 host tiling 写入 tiling data 的 **full tile Q/K row 容量**，由静态选出的 `tokenTile` 计算得到。它用于 L1 A buffer、UB output buffer 等容量分配，也用于描述非 tail tile 的 Q/K row 数。
2. `M_tile` 是 kernel 运行时当前 tile 实际送入 Cube 的矩阵 M 值，由 `actualTokenTile` 计算得到。非 tail tile 中 `M_tile == rowTile`；最后一个 tail tile 中 `M_tile <= rowTile`。
3. `M_tile` 不需要作为独立 tiling data 字段传入，kernel 可由当前 `actualTokenTile * qkHeadTile` 计算得到。

Q/K row 的逻辑映射为：

```text
row = localTokenOffset * Hqk + qkHeadOffset
```

首期不把 `row` 作为 host 分发维度，只作为 tile 内部 Q/K 到 Cube 的布局描述。host/controller 仍按连续 token tile 分发任务，以保持 Q/K/V 交织输入的生命周期简单。

两个 AIV 核对同一个 token tile 做严格 token 平分：

```text
tokenTilePerAiv = tokenTile / 2

AIV0 tokens: [tileStart, tileStart + tokenTilePerAiv)
AIV1 tokens: [tileStart + tokenTilePerAiv, tileStart + tokenTile)
```

最后一个 tail tile 允许 `actualTokenTile < tokenTile`。tail 内部可用：

```text
actualAiv0 = ceil(actualTokenTile / 2)
actualAiv1 = actualTokenTile - actualAiv0
```

tail 可能出现 1 个 token 的不均衡，但不为 tail 引入 Q/K flatten row split。两个 AIV 写入同一个 A 矩阵的 disjoint row 区间时必须按实际 token 数计算 offset，不能按固定 `tokenTilePerAiv` 预留半区导致中间空洞：

```text
AIV0 token range: [tileStart, tileStart + actualAiv0)
AIV1 token range: [tileStart + actualAiv0, tileStart + actualTokenTile)

AIV0 A rows: [0, actualAiv0 * Hqk)
AIV1 A rows: [actualAiv0 * Hqk, actualTokenTile * Hqk)

M_tile = actualTokenTile * Hqk
```

因此 AIC 看到的 `A_rot[M_tile,D]` 在 M 方向始终连续无空洞。AIC Fixpipe 若按 M 方向分发 C 输出给两个 AIV，也按同一 row 边界切分。若 tail 中 `actualAiv1 == 0`，AIV1 不写数据，但仍需按同步协议参与必要的 ready/consumed 通知，避免 AIC 等待缺失。

## 3. L0C 约束

arch35 当前按 256 KiB L0C 估算。为了保留 C buffer double buffer：

```text
L0C_TOTAL      = 256 KiB
L0C_BUFFER_NUM = 2
L0C_ONE_BUFFER = 128 KiB
```

Cube 输出为 FP32：

```text
C_one_buffer = align_up(M_tile, 16) * 128 * sizeof(float32)
```

要求：

```text
C_one_buffer <= 128 KiB
```

等价于：

```text
align_up(M_tile, 16) <= 256
```

首期 host 侧按 full tile 容量直接保证：

```text
rowTile = tokenTile * Hqk
rowTile <= 256
```

也就是：

```text
token_by_l0c = floor(256 / Hqk)
```

由于任意运行时 `M_tile <= rowTile`，该约束可保证所有 tile 的 Cube M 都不超过 256。`rowTile > 256` 的方案不进入首期 fast path，只作为后续扩展。

## 4. UB 静态划分

首期每个 AIV 负责 `tokenTile / 2` 个 token。记：

```text
Ta = tokenTile / 2
```

首期 fast path 不按 shape 动态拼 UB 空间，而是由 kernel 固定每个 AIV 的 UB layout。Tiling 只使用相同的静态预算做容量约束，不把 UB 总量、buffer 数或 reserve 子区大小写入 tiling data。以下 UB 大小均是 **单个 AIV 的本地 UB 容量**，不是两个 AIV 合计容量：

```text
UB usable = 248 KiB

reserve        = 40 KiB
input DB pool  = 80 KiB    // 2 buffers, each 40 KiB
output DB pool = 128 KiB   // 2 buffers, each 64 KiB

40 KiB + 80 KiB + 128 KiB = 248 KiB
```

每个 AIV 的输入 double buffer 覆盖本 AIV 的 Q/K/V 输入：

```text
input_one_buffer = Ta * Htot * 128 * sizeof(bf16)
input_one_buffer <= 40 KiB
```

其中 `input_one_buffer` 是单 AIV 实际使用量。整个 token tile 的两个 AIV 合计输入量为：

```text
input_used_per_tile = tokenTile * Htot * 128 * sizeof(bf16)
```

每个 AIV 的输出 double buffer 覆盖本 AIV 的 Q/K Cube 输出：

```text
output_one_buffer = Ta * Hqk * 128 * sizeof(float32)
output_one_buffer <= 64 KiB
```

该 64 KiB one buffer 是单 AIV 静态容量。AIV 在 Cube fix 输出占用对应 buffer 之前，可以复用 output pool 做 RMS/RoPE 的 fp32 中间量；实现侧必须保证该复用与 AIC fix 到 UB output、AIV quant 消费 UB output 的生命周期不重叠。

因此 UB 直接给出：

```text
Ta <= floor(40 KiB / (Htot * 128 * sizeof(bf16)))
Ta <= floor(64 KiB / (Hqk  * 128 * sizeof(float32)))
```

即：

```text
Ta <= floor(160 / Htot)
Ta <= floor(128 / Hqk)
```

40 KiB reserve 是单 AIV 本地 scratch。首期 reserve 中需要承载 `gamma_ub`、`cos_sin_ub[2]`、`slot_mapping_ub[2]`、`v_scale_ub`、`v_out_ub[2]`。其中 `cos_sin`、`slot_mapping` 和 `v_out` 均开 double buffer；`v_scale` 读入 UB 后复用，不开 DB。具体子区容量、offset 和对齐规则由 kernel 按静态布局定义，tiling 不参与分配，也不写入 tiling data。

Q/K dynamic quant 的 FP8 staging 不占 reserve；它复用 `output DB pool` 中当前 `qk_after_cube_ub` 所在 buffer，在按 row 或 chunk 消费完 FP32 C 后原地复用或直接写 GM/cache。其他标量数组不在 UB 中常驻；Q/K row scale 计算后直接写 `q_scale` 或 `k_scale_cache`。前处理阶段的中间临时量不在 reserve 中静态分配；大块 Q/K fp32 中间量仍按生命周期复用 output DB pool。

V cache 写回属于后处理语义，但不依赖 Cube 输出。AIV 在完成当前 tile 的 Q/K preprocess、写入 `a_rot_l1_nz` 并发送 `A_READY` 后，可以立即执行 V quant 与 `v_cache` 写回：

```text
V: cast_to_e4m3fn_rint_sat(float32(V) * v_scale[nv]) -> v_cache
```

V 不等待 Cube 输出；这样可以尽早释放输入 UB buffer，并与 AIC 的 `rope_out @ rotation` 并行。首期仍让同一个 AIV 处理自己 token 范围内的 Q/K/V，避免引入 V ownership 规则。

## 5. L1 静态划分

`rotation` 固定常驻 L1：

```text
rotation_l1_bytes = 128 * 128 * sizeof(bf16) = 32 KiB
```

首期固定 4 个 L1 A buffer，每个 buffer 按最大 `rowTileAligned = 256` 预留：

```text
single A_L1 buffer = 256 * 128 * sizeof(bf16) = 64 KiB
A_L1 buffer count  = 4
```

L1 静态 layout：

```text
rotation_l1 = 32 KiB
A_L1 pool   = 4 * 64 KiB = 256 KiB
reserve     = 512 KiB - 32 KiB - 256 KiB = 224 KiB
```

Q/K RoPE 输出写入 L1 A buffer，供 Cube 使用：

```text
rowTile        = tokenTile * Hqk
rowTileAligned = align_up(rowTile, 16)
A_actual_bytes = rowTileAligned * 128 * sizeof(bf16)
```

host 只需保证：

```text
rowTileAligned <= 256
A_actual_bytes <= 64 KiB
```

由于首期强制 `rowTile <= 256`，自然满足：

```text
A_actual_bytes <= 64 KiB
```

L1 静态总占用为：

```text
32 KiB + 4 * 64 KiB = 288 KiB <= 512 KiB
```

典型 shape 下 L1 不再是首要瓶颈，但 tiling UT 必须校验静态 layout 和 `rowTileAligned <= 256`。

## 6. Host Tiling 算法

首期只接受 `D = 128` fast path。若 `D != 128`，不进入当前 fast tiling。

算法：

```text
Hqk  = Nq + Nk
Nv   = Nk
Htot = Nq + Nk + Nv  // = Nq + 2 * Nk

if Nv != Nk:
    unsupported_fast_path

if D != 128:
    unsupported_fast_path

UB_INPUT_ONE_BUFFER_BYTES  = 40 KiB
UB_OUTPUT_ONE_BUFFER_BYTES = 64 KiB

TaMax = min(
    8,                    // tokenTile <= 16
    floor(128 / Hqk),     // output UB + L0C
    floor(160 / Htot)     // input UB
)

if TaMax < 1:
    unsupported_fast_path

tokenTile = 2 * TaMax
tokenTilePerAiv = TaMax

rowTile = tokenTile * Hqk
rowTileAligned = align_up(rowTile, 16)

if rowTile > 256 or rowTileAligned > 256:
    unsupported_fast_path
```

首期不把 `tokenTile` 强行拉到 2。若 `TaMax < 1`，该 shape 不进入当前 fast path。

## 7. 典型结果

在首期 `Nv = Nk` 约束下，典型 tiling 结果如下：

| Nq | Nk | Nv | Hqk | Htot | Ta | tokenTile | rowTile | vRows/AIV |
|----|----|----|-----|------|----|-----------|---------|-----------|
| 16 | 2 | 2 | 18 | 20 | 7 | 14 | 252 | 14 |
| 16 | 4 | 4 | 20 | 24 | 6 | 12 | 240 | 24 |
| 16 | 8 | 8 | 24 | 32 | 5 | 10 | 240 | 40 |
| 32 | 2 | 2 | 34 | 36 | 3 | 6 | 204 | 6 |
| 32 | 4 | 4 | 36 | 40 | 3 | 6 | 216 | 12 |
| 32 | 8 | 8 | 40 | 48 | 3 | 6 | 240 | 24 |
| 64 | 2 | 2 | 66 | 68 | 1 | 2 | 132 | 2 |
| 64 | 4 | 4 | 68 | 72 | 1 | 2 | 136 | 4 |
| 64 | 8 | 8 | 72 | 80 | 1 | 2 | 144 | 8 |

表中的 `rowTile` 是非 tail tile 的 `M_tile`。tail tile 的实际 `M_tile = actualTokenTile * Hqk`，可能小于表中值。reserve 内 `v_out` 的静态容量证明归属 kernel 设计，不作为 tiling 表格字段。

## 8. Tiling Data 字段

字段建议收敛为：

```cpp
struct QkvRmsNormRopeCacheWithKScaleTilingData {
    uint32_t T;
    uint32_t batch;
    uint32_t numQHeads;
    uint32_t numKHeads;
    uint32_t numVHeads;
    uint32_t headDim;
    uint32_t maxSeqLen;
    uint32_t blockNum;
    uint32_t blockSize;

    uint32_t qkvStrideT;
    uint32_t qkvStrideH;
    uint32_t cacheStrideBlock;
    uint32_t cacheStrideHead;
    uint32_t cacheStrideToken;

    uint32_t tokenTile;
    uint32_t tokenTilePerAiv;
    uint32_t rowTile;
    uint32_t rowTileAligned;
    uint32_t qkHeadTile;
    uint32_t vHeadTile;
    uint32_t dimTile;

    uint32_t flags;
    float epsilon;
};
```

建议 flags：

```text
D128_FAST
ROTATION_L1_RESIDENT
V_CACHE_WRITE_AFTER_A_READY
TOKEN_TILE_EVEN
L0C_DOUBLE_BUFFER
```

说明：

1. `numQHeads/numKHeads/numVHeads` 来自必选属性 `head_nums=[Nq,Nk,Nv]`。
2. `batch = cu_seq_len.shape[0] - 1`。
3. `maxSeqLen = cos_sin.shape[0]`，不强校验等于最大 batch 内长度。
4. `dimTile` 首期固定为 `128`。
5. `qkHeadTile` 首期固定为 `Nq + Nk`，不做 Q/K head split。
6. `vHeadTile` 首期固定为 `Nv`。
7. `rowTile = tokenTile * qkHeadTile`，表示 full tile 的 Q/K row 容量。
8. `rowTileAligned = align_up(rowTile, 16)`，表示 L1 A buffer 和 UB output buffer 的容量对齐值。
9. `M_tile = actualTokenTile * qkHeadTile` 是 kernel 运行时当前 tile 的 Cube M 参数；它不是独立 tiling data 字段。非 tail tile `M_tile == rowTile`，tail tile `M_tile <= rowTile`。
10. UB/L1 buffer 数、字节预算和 reserve 子区布局均为 kernel 静态常量；host tiling 使用相同常量计算 `tokenTile` 和校验资源上界，但不写入 tiling data。
11. reserve 内部的 `cos_sin` DB、`slot_mapping` DB、`v_scale_ub`、`v_out` DB 均由 kernel 按静态布局分配，不作为 tiling data 字段写入。
12. `V_CACHE_WRITE_AFTER_A_READY` 表示 V cache 写回属于后处理子阶段，但调度上允许在 `A_READY` 之后、等待 `C_READY` 之前提前执行。

## 9. Tiling Key

首期 tiling key 只表达平台和 fast path 必要项：

```text
arch35 only
D128 fast path
qkv dtype fixed bf16
rotation dtype fixed bf16
gamma/cos_sin/scale dtype fixed fp32
out/cache dtype fixed fp8_e4m3fn
L0C double buffer
```

不进入 tiling key：

1. `cu_seq_len` 模式。当前固定前缀和语义。
2. `head_nums` 具体数值。
3. `rotation` 是否存在。当前 checker 强制非空。
4. `v_scale` 是否存在。当前 checker 强制非空。
5. Hadamard/FWHT 模式。
6. `tokenTile` 具体数值。由 tiling data 承载。

## 10. 后续扩展路线

### 10.1 M 大于 256

首期主路径不允许 full tile `rowTile > 256`，因此运行时 `M_tile` 也不会超过 256。后续可评估两种扩展：

```text
Option A: AIC 内部 M_sub 切分
  host tokenTile 可允许 rowTile > 256
  AIC 内部按 M_sub <= 256 循环
  保持 L0C double buffer
  需要新增 sub-tile 级同步、UB output offset、C_READY_SUB

Option B: 关闭 L0C double buffer
  单个 C buffer 理论上可使用完整 256 KiB L0C
  D=128 时 M 上限可放宽到 512
  但 Cube/Fixpipe 无法 DB，流水性能需要实测
```

这两条都不进入首期实现。

### 10.2 Small-T 负载重平衡

首期只沿 token 维切分，因此任务数为：

```text
tileNum = ceil(T / tokenTile)
```

当 `T` 很小且 `tileNum < 可用 AIC/AIV group 数` 时，很多核组没有任务。例如 `T = 1` 时，无论 `Hqk` 多大，主路径只有一个 token tile。

后续可增加 small-T 二级 tiling：

```text
tokenTileSmall = min(tokenTile, T_rem)
qkHeadTileSmall < Hqk
M_tile = tokenTileSmall * qkHeadTileSmall <= 256
```

触发条件建议：

```text
mainTileNum = ceil(T / tokenTile)

if mainTileNum < targetCoreGroupNum
   and T * Hqk has enough rows:
       enable small-T head-split tiling
else:
       use token-only fast path
```

该模式需要新增 V ownership 规则，避免同一 token 的 V cache 被多个 Q/K head shard 重复写。建议约定：

```text
同一个 token tile 内：
  qkHeadOffset == 0 的 shard 负责 V quant + v_cache 写回
  其他 Q/K head shard 只处理自己的 Q/K
```

由于 Q/K/V 在输入 `[T, Nq+Nk+Nv, D]` 中交织，head split 会显著增加地址映射、V ownership、同步和测试复杂度，因此只作为后续扩展。

### 10.3 Core Wave 尾块均衡

首期 host tiling 使用资源允许的最大偶数 `tokenTile`，不额外按 core 数重调 tile 大小。因此当：

```text
tileNum = ceil(T / tokenTile)
tileNum % coreGroupNum != 0
```

时，最后一个 wave 可能只有少数核组仍在工作。例如 `tileNum = 65`、`coreGroupNum = 32` 时，前 64 个 tile 可以形成两个完整 wave，最后只剩 1 个 tile，会造成一个核组多做一轮。

首期接受该尾波不均衡，不在主路径中引入额外调度复杂度。后续可评估两类优化：

```text
Option A: tail two-phase rebalance
  将最后 coreGroupNum + r 个 token tile 覆盖的 token 重新切成更多小 tile
  例如 65 个 tile、32 个核时，可考虑 32 个大 tile + 64 个小 tile
  目标是让每个核组在尾部拿到更接近的工作量
  代价是需要 large/small 两套 tile 参数、tail phase 起点和更复杂的 tileId 映射

Option B: uniform smaller token tile
  使用比 tokenTileMax 更小的统一 token tile，提高 tileNum 对 coreGroupNum 的整除友好度
  该方案需要 profiling 才能确定性能下限 token tile
  即使找到性能下限，也不能保证彻底消除最后一个 tail tile
```

由于 Option B 的性能下限依赖实测，且不能从数学上保证消除尾块；Option A 又会增加调度状态和实现复杂度，两者均不进入首版。

### 10.4 Q/K Flatten Row 均衡切分

后续可按 Q/K flatten row 在两个 AIV 间做更细粒度负载均衡，用于支持奇数 `tokenTile` 或更均衡的 tail：

```text
row = tokenOffset * Hqk + qkHeadOffset
```

但该路线会解耦 Q/K 与 V 的 token 生命周期，也会增加 row 到 token/head 的反解和 V ownership 复杂度。首期不采用。

## 11. Host UT

Host UT 覆盖 checker、infer shape、tiling data 和资源公式。

### 11.1 正常用例

1. `D=128`，`cu_seq_len=[0,T]`。
2. 多 batch：`cu_seq_len=[0,2,9]`。
3. `B in {1,2}`。
4. `Nq in {16,32,64}`。
5. `Nk in {2,4,8}`。
6. `Nv = Nk`。
7. `T in {1,128,1024,8192,32768}`。
8. `rotation`、`v_scale` 均存在。
9. cache 输出与输入同 shape。

### 11.2 资源与 tile 校验

Tiling UT 必须校验：

1. `tokenTile % 2 == 0`。
2. `tokenTile >= 2`。
3. `tokenTile <= 16`。
4. `rowTile == tokenTile * (Nq + Nk)`。
5. `rowTile <= 256`。
6. 非 tail tile `M_tile == rowTile`，tail tile `M_tile <= rowTile`。
7. `rowTileAligned == align_up(rowTile, 16)`。
8. `rowTileAligned <= 256`。
9. host tiling 使用固定 UB 常量推导 `TaMax = min(8, floor(128 / Hqk), floor(160 / Htot))`。
10. `tokenTile / 2 <= floor(160 / Htot)`。
11. `tokenTile / 2 <= floor(128 / Hqk)`。
12. UB/L1 总量、buffer 数和 reserve 子区大小不写入 tiling data。
13. Q/K FP8 staging 不占 reserve，复用 output DB pool。
14. L1 静态预算满足 `32 KiB + 4 * 64 KiB <= 512 KiB`。
15. `dimTile == 128`。
16. `qkHeadTile == Nq + Nk`。
17. `vHeadTile == Nv`。
18. 典型 `Nv = Nk` case 的 `tokenTile` 与第 7 节表格一致。
19. tail tile 覆盖 `T < tokenTile` 和 `T % tokenTile != 0`。

### 11.3 错误用例

1. 缺失 `rotation`。
2. 缺失 `v_scale`。
3. `rotation.shape != [D,D]`。
4. `rotation.dtype != bf16`。
5. `v_scale.shape != [Nv]`。
6. `cos_sin.shape[1] != D`。
7. `q_gamma.shape != [D]` 或 `k_gamma.shape != [D]`。
8. `slot_mapping.shape != [T]`。
9. `cu_seq_len` rank 不是 1 或长度小于 2。
10. 若 host 可读 `cu_seq_len`，覆盖 `cu_seq_len[0] != 0`、非单调、`cu_seq_len[Batch] != T`。
11. `head_nums` 缺失、长度不是 3、任一值非正。
12. `qkv.shape[1] != sum(head_nums)`。
13. cache/v_scale head 维与 `head_nums` 不一致。
14. FP8 dtype 不是 `float8_e4m3fn`。
15. `Nv != Nk` 时不进入当前 fast path。
16. `D != 128` 时不进入当前 fast path。
17. `TaMax < 1` 时不进入当前 fast path。

## 12. Kernel UT/ST

Kernel UT/ST 按阶段覆盖：

1. 前处理：Q/K/V split、RMSNorm、RoPE 位置映射。
2. AIV token 平分：主 tile 两个 AIV 各处理 `tokenTile / 2` 个 token。
3. Tail token 分配：`actualTokenTile` 为奇数时只在 tail 出现不均衡。
4. Cube：`rope_out @ rotation`，且 `M_tile <= 256`。
5. V 提前写回：V cache 写回属于后处理子阶段，但在 `A_READY` 后即可执行，不等待 Cube 输出，与 golden 一致。
6. 后处理：Q/K dynamic quant/cache。
7. 端到端：完整输出对比 golden。

必须覆盖：

1. `slot_mapping` 连续、非连续、跨 block。
2. 多 batch 前缀和位置。
3. tile 跨 `cu_seq_len` 边界时 RoPE 位置正确。
4. `v_scale == 0`。
5. `max_abs == 0` row；不新增零行保护假设，行为对齐 arch35 dynamic_quant 路径。
6. random `rotation`。
7. cache alias。

## 13. UT 验收标准

1. Host UT 能证明当前接口、checker 和 fast path 选择行为稳定。
2. Tiling UT 能证明 UB/L1/L0C 公式全部满足。
3. Tiling UT 能证明 `tokenTile` 为偶数且 `M_tile <= 256`。
4. Tiling UT 能证明典型 shape 的 `tokenTile` 与表格一致。
5. Kernel ST 与 golden 对齐。
6. 不出现 `SEQ_LEN_MODE` 双模式相关测试。
7. arch35 binary/config 能正确选择本 kernel。
