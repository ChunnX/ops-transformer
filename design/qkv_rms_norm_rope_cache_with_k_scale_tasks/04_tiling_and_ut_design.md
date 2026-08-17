# Tiling 与 UT 设计

## 1. 目标

本文定义 `QkvRmsNormRopeCacheWithKScale` 首期 arch35 实现的 tiling 策略、tiling data 字段、扩展路线和 UT 验收标准。

首期实现目标是优先保证 kernel 实现复杂度可控：

1. 固定 `D = 128`。
2. `rotation` 当前必须非空，dtype 为 `bf16`，shape 为 `[128,128]`。
3. `v_scale` 当前必须非空，dtype 为 `float32`；RoPE shape 为 `[Nv]`，M-RoPE shape 为 `[Nv,D]`。
4. RoPE 与 M-RoPE 统一只支持 `0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`。
5. 外部只保留一个 `rotation`；kernel 内部暂不修改 L1 静态 buffer 分配，但只填充和使用一个 rotation resident slot，Q/K segment 复用同一份 L1/L0B rotation。
6. host tiling 不切 D、不切 Q/K head、不做 Q/K flatten row 均衡切分；kernel Cube 可在单个 Q 或 K segment 内根据当前 `segmentRows` 和 L0A buffer size 动态计算 D 方向 tile，并在 L0C 累加，以适配 L0A 容量。
7. 主 tile 只沿 token 维切分。host 优先选择偶数 `tokenTile`；若偶数候选均不满足资源约束，当前实现保留 `tokenTile = 1` fallback。
8. 单次 Cube tile 满足 `M_tile <= 256` 且 side-by-side 物理行数 `qkAlignedRows <= 256`，保证 L0C double buffer。

## 2. 首期 Tiling 原则

记：

```text
0 < Nq <= 64
Nq   = 8 * Nk
Nk   = Nv
qkN  = Nq + Nk
qkvN = Nq + Nk + Nv = Nq + 2 * Nk
D    = 128
```

这里三项 head 关系是 RoPE 与 M-RoPE 的显式支持约束，不是由接口或资源公式隐式推导。host 仍从
`head_nums=[Nq,Nk,Nv]` 读取三者，并在进入资源计算前校验 `0 < Nq <= 64`、`Nq == 8 * Nk`、
`Nk == Nv`。

首期 tile 形态固定为：

```text
dimTile    = 128      // kernel 静态约束，不作为 tiling data 字段
qkHeadTile = qkN      // 由 qHeadNum + kvHeadNum 实时计算
vHeadTile  = Nv       // host 校验完整 head 合同后，kernel 使用 kvHeadNum

actualTokenTile = min(tokenTile, T - tileStart)
cubeTokenSize   = align_up(actualTokenTile, 2)
rowTile         = tokenTile * qkN
validM_tile     = actualTokenTile * qkN
physicalQRows   = cubeTokenSize * Nq
physicalKRows   = cubeTokenSize * Nk
qkAlignedRows   = align_up(physicalQRows, 16) + align_up(physicalKRows, 16)
```

`rowTile`、有效行和 Cube/Fixpipe 物理行的语义必须区分：

1. `rowTile` 是 host 和文档中用于资源证明的 **full tile Q/K row 容量**，由静态选出的 `tokenTile` 计算得到。当前代码不把 `rowTile` 写入 tiling data，kernel 通过 `tokenTile * (qHeadNum + kvHeadNum)` 实时计算同等语义。
2. `validM_tile = actualTokenTile * qkN` 是 Vec 后处理的有效 Q/K row 数，Q/K quant 和 GM/cache 写回只消费这些有效行。
3. Cube/Fixpipe 物理行使用 `cubeTokenSize = align_up(actualTokenTile, 2)`，即 `physicalQRows = cubeTokenSize * Nq`、`physicalKRows = cubeTokenSize * Nk`。odd tail 或 `tokenTile = 1` fallback 中，pad 行会参与 Cube 计算并由 Fixpipe 实际搬出到 AIV UB，但 Vec 不量化、不写回 pad 行。
4. 这些运行时 row 值不作为独立 tiling data 字段传入，kernel 由 `tokenTile/tokenSize/qHeadNum/kvHeadNum` 实时计算。

Q/K row 的逻辑映射为：

```text
row = local_pos * qkN + qkHeadOffset
```

首期不把 `row` 作为 host 分发维度，只作为 tile 内部 Q/K 到 Cube 的布局描述。host/controller 仍按连续 token tile 分发任务，以保持 Q/K/V 交织输入的生命周期简单。

Cube 侧按 Q/K segment 分开循环：

```text
Q segment: qRows = actualTokenTile * Nq
K segment: kRows = actualTokenTile * Nk
```

L0C 仍保留 side-by-side 布局，Q segment 的尾地址后紧接 K segment 的首地址：

```text
qL0cOffset = 0
kL0cOffset = align_up(qRows, 16) * D
```

因此 L0C/L1 的物理容量约束使用 `qkAlignedRows = align_up(qRows,16) + align_up(kRows,16)`，而不是只看 `align_up(M_tile,16)`。旧方案中 “Q/K 同时放入 L0A” 的 `qkAlignedRows <= 128` 约束可以放开；新的 L0A 约束在单个 Q 或 K segment 内判断，若 full-D segment 放不下，则根据当前 `segmentRows` 动态计算 D 方向 tile 并累加，不在 N/M 轴切分 Q/K head。

两个 AIV 核对同一个 token tile 仍按 token 维协作，但 split 已由 controller 前置写入 `TileParam`，vec 内部不再维护独立 slice 结构：

```text
cubeTokenSize     = align_up(actualTokenTile, 2)
cubeHalfTokenSize = cubeTokenSize / 2

AIV0 tokenOffset = tileStart
AIV0 tokenSize   = min(cubeHalfTokenSize, actualTokenTile)

AIV1 tokenOffset = tileStart + cubeHalfTokenSize
AIV1 tokenSize   = actualTokenTile - cubeHalfTokenSize
```

主路径偶数非 tail tile 中，`cubeHalfTokenSize == tokenTile / 2`，两个 AIV 各处理一半 token。最后一个 tail tile 或 `tokenTile = 1` fallback 允许出现不均衡：

```text
actualAiv0 = min(cubeHalfTokenSize, actualTokenTile)
actualAiv1 = actualTokenTile - cubeHalfTokenSize
```

tail/fallback 可能出现 1 个 token 的不均衡，但不引入 Q/K flatten row split。两个 AIV 写入同一个 A 矩阵的 disjoint row 区间时必须按 `aivBlockTokenOffset` 和实际 token 数计算 offset，不能按固定 `tokenTile / 2` 划定半区导致中间空洞：

```text
AIV0 token range: [tileStart, tileStart + actualAiv0)
AIV1 token range: [tileStart + cubeHalfTokenSize, tileStart + actualTokenTile)

AIV0 valid A rows: [0, actualAiv0 * qkN)
AIV1 valid A rows: [cubeHalfTokenSize * qkN, actualTokenTile * qkN)

validM_tile = actualTokenTile * qkN
physicalM capacity = cubeTokenSize * qkN
```

因此 AIV 生成的 Q/K 有效逻辑 row 覆盖始终连续无重复。写入 L1 时每个 AIV 按 `copyRows = headSize * cubeHalfTokenSize` 搬出本地 capacity 行；odd tail 中 pad 行也会进入 L1。AIC 按 Q segment base 和 K segment base 读取 `cubeTokenSize` 对应的物理行，pad 行参与 Cube 计算；Fixpipe 使用 dual M split 将完整物理 segment 搬到两个 AIV UB。本地 UB 中 `cubeHalfTokenSize` 是 capacity，`aivTokenSize` 是有效 token 数，Vec 后处理只按 `aivTokenSize` 量化和写回。若 tail/fallback 中 `actualAiv1 == 0`，AIV1 不写有效数据、不处理 pad 行，但仍需按同步协议参与必要的 ready/consumed 通知，避免 AIC 等待缺失。

## 3. L0C 约束

arch35 当前按 256 KiB L0C 估算。为了保留 C buffer double buffer：

```text
L0C_TOTAL      = 256 KiB
L0C_BUFFER_NUM = 2
L0C_ONE_BUFFER = 128 KiB
```

Cube 输出为 FP32。由于 Q/K 在 L0C 中采用 side-by-side 布局，单个 C buffer 的物理占用为：

```text
C_one_buffer = qkAlignedRows * 128 * sizeof(float32)
```

要求：

```text
C_one_buffer <= 128 KiB
```

等价于：

```text
qkAlignedRows <= 256
```

首期 host 侧按 full tile 容量直接保证：

```text
rowTile = tokenTile * qkN
fullTileQkAlignedRows = align_up(tokenTile * Nq, 16) + align_up(tokenTile * Nk, 16)
rowTile <= 256
fullTileQkAlignedRows <= 256
```

对当前完整支持的 8 组 head tuple，`rowTile` 约束先给出：

```text
token_by_l0c = floor(256 / qkN)
```

由于任意运行时 `M_tile <= rowTile` 且 `qkAlignedRows <= fullTileQkAlignedRows`，该约束可保证所有 tile 的 Cube M 和 side-by-side L0C 物理行数都不超过 256。`rowTile > 256` 或 `fullTileQkAlignedRows > 256` 的方案不进入首期实现，只作为后续扩展。

## 4. UB 静态划分

资源公式按单个 AIV 的最大 token 份额建模。记：

```text
Ta = ceil(tokenTile / 2)
```

主路径偶数 `tokenTile` 下 `Ta == tokenTile / 2`；`tokenTile = 1` fallback 和 tail 中，kernel 运行时使用 `tile.aivTokenSize <= Ta`。

首期实现不按 shape 动态拼 UB 空间，而是由 kernel 固定每个 AIV 的 UB layout。Tiling 只使用相同的静态预算做容量约束，不把 UB 总量、buffer 数或其他固定 buffer 的大小写入 tiling data。以下 UB 大小均是 **单个 AIV 的本地 UB 容量**，不是两个 AIV 合计容量：

```text
UB usable = 248 KiB

other fixed buffers = 40 KiB
input pool     = 80 KiB    // RoPE: 2 buffers, each 40 KiB
output DB pool = 128 KiB   // 2 buffers, each 64 KiB

40 KiB + 80 KiB + 128 KiB = 248 KiB
```

每个 AIV 的输入 double buffer 覆盖本 AIV 的 Q/K/V 输入：

```text
input_one_buffer = Ta * qkvN * 128 * sizeof(bf16)
input_one_buffer <= 40 KiB
```

其中 `input_one_buffer` 是单 AIV 实际使用量。整个 token tile 的两个 AIV 合计输入量为：

```text
input_used_per_tile = tokenTile * qkvN * 128 * sizeof(bf16)
```

每个 AIV 的输出 double buffer 覆盖本 AIV 的 Q/K Cube 输出：

```text
output_one_buffer = Ta * qkN * 128 * sizeof(float32)
output_one_buffer <= 64 KiB
```

该 64 KiB one buffer 是单 AIV 静态容量。AIV 在 Cube fix 输出占用对应 buffer 之前，可以复用 output pool 做 RMS/RoPE 的 fp32 中间量；实现侧必须保证该复用与 AIC fix 到 UB output、AIV quant 消费 UB output 的生命周期不重叠。

因此 UB 直接给出：

```text
Ta <= floor(40 KiB / (qkvN * 128 * sizeof(bf16)))
Ta <= floor(64 KiB / (qkN  * 128 * sizeof(float32)))
```

即：

```text
Ta <= floor(160 / qkvN)
Ta <= floor(128 / qkN)
```

除 input/output pool 外，普通 RoPE kernel 还静态分配 40 KiB，用于承载 `gamma_ub`、`cos_sin_ub_pool`、
`slot_mapping_ub_pool`、`v_scale_ub`、`qk_nz_scatter_index_ub`、`v_out_ub_pool`。其中 `cos_sin` 与
`slot_mapping` 由 `metadataBufferUseId_` 管理，`v_out` 由 `vOutBufferUseId_` 管理；物理 slot 均用资源
counter 对场景专用 buffer 数取模生成，不依赖 `TileParam::bufferId` 或 tile id。RoPE `[Nv]`、M-RoPE
`[Nv,D]` 的 `v_scale` 都在循环前完整读入各自 UB 常驻区并复用，
不开 DB。各 buffer 的容量、offset 和对齐规则由 kernel 按静态布局定义，tiling 不参与分配，也不写入
tiling data。

Q/K dynamic quant 的 FP8 staging 不分配独立 buffer；它复用 `output DB pool` 中当前 `qk_after_cube_ub` 所在 buffer，在按 row 消费完 FP32 C 后原地复用或直接写 GM/cache。RmsNormRopeNz 阶段已经按 `layout_q_out` 统一 Q/K 内部 row order：`NTD` 使用 `[head, local_t, d]` / `[head, local_t]`，`TND` 使用 `[local_t, head, d]` / `[local_t, head]`。新增 `layout_qkv=TND,layout_q_out=NTD` 时，Q/K RoPE 写 BF16 NZ 阶段已经按 NTD row order 写 `qRopeNz/kRopeNz`，Cube/Fixpipe 输出的 Q/K FP32 也是 `[head,local_t,d]`，因此 Q/K dynamic quant 直接复用 NTD 内部布局，不需要额外 `qFp8NtdChunk`。K cache 的对外写回仍根据 `slot_mapping`、K head 和 cache stride scatter，不受 `layout_q_out` 改变 shape。前处理阶段的中间临时量不单独静态分配 buffer；大块 Q/K fp32 中间量仍按生命周期复用 output DB pool。

V output DB 还需要满足单 AIV 当前 token 份额内的 V head 行数容量。按当前静态 layout，单 buffer 支持的 V output 上限为 80 行：

```text
Ta * Nv <= 80
```

Q/K FP8 与 scale staging 复用 64 KiB output DB pool，容量由上面的 `Ta * qkN <= 128` 约束覆盖。上述约束与 input/output UB 约束一起参与 host tiling 选候选 `tokenTile`。

V cache 写回属于后处理语义，但不依赖 Cube 输出。当前实现采用 V-first 调度：AIV 在 `Vec::ComputeTile` 内先执行 V vector quant，再执行 Q/K RMSNorm/RoPE；Q/K vec 到达安全点后写 `v_cache`，随后将 Q/K BF16 NZ 搬入 `a_rot_l1_nz`，最后由 BasicBlock 发送 `A_READY`：

```text
QuantVToFp8
  -> BuildRopeBf16NzUb
  -> ScatterVCache
  -> CopyRopeBf16NzUbToL1Nz
  -> A_READY
```

这里的 V-first 表示 V vector quant 先于 Q/K preprocess 发起，不表示 `v_cache` 写回先于 Q/K preprocess。V 分支不等待 Cube 输出，并在 Q/K AL1 handoff 和 `A_READY` 前完成 cache 写回。首期仍让同一个 AIV 处理自己 token 范围内的 Q/K/V，避免引入 V ownership 规则。

## 5. L1 静态划分

`rotation` 固定常驻 L1。为保持首期实现改动最小，L1 分配暂不收缩，仍静态分配两个内部 resident slot；但实际 ND2NZ 只填充一个 slot，MTE1 也只把这一份 rotation 加载到 L0B：

```text
single_rotation_l1_bytes = 128 * 128 * sizeof(bf16) = 32 KiB
rotation_l1_internal_bytes = 2 * single_rotation_l1_bytes = 64 KiB
```

首期静态分配 4 个 L1 A buffer，每个 buffer 按最大 `qkAlignedRows = 256` 确定容量：

```text
single A_L1 buffer = 256 * 128 * sizeof(bf16) = 64 KiB
A_L1 buffer count  = 4
```

L1 静态 layout：

```text
rotation_l1_internal_slots     = 64 KiB
A_L1 pool                    = 4 * 64 KiB = 256 KiB
unused L1 capacity           = 512 KiB - 64 KiB - 256 KiB = 192 KiB
```

该策略只收敛接口、ND2NZ/MTE1 指令和计算语义，不改变 L1 静态排布。

Q/K RoPE 输出写入 L1 A buffer，供 Cube 使用：

```text
qRowsFull            = tokenTile * Nq
kRowsFull            = tokenTile * Nk
fullTileQkAlignedRows = align_up(qRowsFull, 16) + align_up(kRowsFull, 16)
A_actual_bytes        = fullTileQkAlignedRows * 128 * sizeof(bf16)
```

host 只需保证：

```text
fullTileQkAlignedRows <= 256
A_actual_bytes <= 64 KiB
```

由于首期强制 `fullTileQkAlignedRows <= 256`，自然满足：

```text
A_actual_bytes <= 64 KiB
```

L1 静态总占用为：

```text
64 KiB + 4 * 64 KiB = 320 KiB <= 512 KiB
```

典型 shape 下 L1 不再是首要瓶颈，但 tiling UT 必须校验静态 layout 和 `fullTileQkAlignedRows <= 256`。

## 6. Host Tiling 算法

首期只接受 `D = 128`。若 `D != 128`，不进入当前 tiling。

算法：

```text
if Nq <= 0 or Nk <= 0 or Nv <= 0:
    unsupported_current_path

if Nq > 64 or Nq != 8 * Nk or Nk != Nv:
    unsupported_current_path

if D != 128:
    unsupported_current_path

qkN  = Nq + Nk
qkvN = Nq + Nk + Nv  // = Nq + 2 * Nk

UB_INPUT_ONE_BUFFER_BYTES  = 40 KiB
UB_OUTPUT_ONE_BUFFER_BYTES = 64 KiB

MAX_TOKEN_TILE = 16
TOKEN_TILE_PER_AIV_CAP = 4

resourceTokenTilePerAiv = min(
    MAX_TOKEN_TILE / 2,       // resource hard limit
    floor(128 / qkN),         // output UB
    floor(160 / qkvN),        // input UB
    floor(80 / Nv)            // V output DB
)

initialTokenTilePerAiv = min(
    resourceTokenTilePerAiv,
    TOKEN_TILE_PER_AIV_CAP    // performance-selected cap
)

if initialTokenTilePerAiv < 1:
    unsupported_current_path

for searchTokenTilePerAiv from initialTokenTilePerAiv down to 1:
    candidateTokenTile = 2 * searchTokenTilePerAiv
    tokenTilePerAiv = ceil(candidateTokenTile / 2)

    rowTile = candidateTokenTile * qkN
    rowTileAligned = align_up(rowTile, 16)
    fullTileQkAlignedRows = align_up(candidateTokenTile * Nq, 16) + align_up(candidateTokenTile * Nk, 16)

    fitsUb =
        tokenTilePerAiv * qkN  <= 128 and
        tokenTilePerAiv * qkvN <= 160 and
        tokenTilePerAiv * Nv   <= 80

    fitsL0c =
        rowTile <= 256 and
        rowTileAligned <= 256 and
        fullTileQkAlignedRows <= 256

    if fitsUb and fitsL0c:
        tokenTile = candidateTokenTile
        break

if no even candidate is accepted:
    tokenTile = 1  // prevalidated fallback
```

首期主路径使用偶数 `tokenTile = 2 * searchTokenTilePerAiv`。`MAX_TOKEN_TILE = 16` 是资源硬上限；当前性能策略设置 `TOKEN_TILE_PER_AIV_CAP = 4`，因此主路径实际 `tokenTile <= 8`。若偶数候选不满足资源约束，host 保留 `tokenTile = 1` 的兼容 fallback；进入 fallback 前已经由 `ValidateMinimumTokenTileFeasible()` 证明单 token 的 UB、L1、L0C 和 V output 约束满足。kernel 侧统一按 `cubeTokenSize/cubeHalfTokenSize/aivTokenSize` 处理偶数主路径、tail 和 fallback。

旧的 `align_up(tokenTile * Nq,16) + align_up(tokenTile * Nk,16) <= 128` 检查只适用于 Q/K 同时驻留 L0A 的实现。Q/K segment 分开循环后，该检查不再作为 host 拒绝条件；L0A 是否需要切 D 由 kernel 在单个 segment 内判断。

## 7. 完整结果

在 `0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv` 约束下，完整 tiling 结果如下：

| Nq | Nk | Nv | qkN | qkvN | Ta | tokenTile | rowTile | vRows/AIV |
|----|----|----|-----|------|----|-----------|---------|-----------|
| 8 | 1 | 1 | 9 | 10 | 4 | 8 | 72 | 4 |
| 16 | 2 | 2 | 18 | 20 | 4 | 8 | 144 | 8 |
| 24 | 3 | 3 | 27 | 30 | 4 | 8 | 216 | 12 |
| 32 | 4 | 4 | 36 | 40 | 3 | 6 | 216 | 12 |
| 40 | 5 | 5 | 45 | 50 | 2 | 4 | 180 | 10 |
| 48 | 6 | 6 | 54 | 60 | 2 | 4 | 216 | 12 |
| 56 | 7 | 7 | 63 | 70 | 2 | 4 | 252 | 14 |
| 64 | 8 | 8 | 72 | 80 | 1 | 2 | 144 | 8 |

表中的 `rowTile` 是非 tail tile 的 Q/K row 容量。tail tile 的有效行数 `validM_tile = actualTokenTile * qkN` 可能小于表中值；Cube/Fixpipe 物理 capacity 使用 `cubeTokenSize * qkN`，odd tail pad 行参与 Cube/Fixpipe 但不进入 Vec 有效写回。`v_out` 的静态容量证明归属 kernel 设计，不作为 tiling 表格字段。

## 8. Tiling Data 字段

当前 kernel tiling data 已收敛为：

```cpp
struct QkvRmsNormRopeCacheWithKScaleTilingData {
    uint64_t totalTokens;
    uint64_t batch;
    uint64_t qHeadNum;
    uint64_t kvHeadNum;
    uint64_t headDim;
    uint64_t blockSize;
    uint64_t coreTokenTile;
    uint64_t coreGroupNum;
    uint64_t kvCacheStrideBlock;
    uint64_t kvCacheStrideHead;
    uint64_t kvCacheStrideToken;
    uint64_t kScaleCacheStrideBlock;
    uint64_t kScaleCacheStrideHead;
    uint64_t kScaleCacheStrideToken;
    uint64_t tokenTile;
    float epsilon;
};
```

说明：

1. `qHeadNum` 来自必选属性 `head_nums[0]`，`kvHeadNum` 来自 `head_nums[1]`。host 已校验 `head_nums[2] == head_nums[1]`，kernel 使用同一个 `kvHeadNum` 表达 K/V head 数。
2. 当前没有 `numVHeads` tiling 字段，也没有历史 `flags` 字段。
3. `qkv` 支持 `NTD` 和 `TND` 排布；`layout_qkv` 不进入 tiling data，而是进入 tiling key。`layout_q_out` 也不进入 tiling data，但进入 tiling key 作为 `Q_OUT_LAYOUT` 模板常量。当前 layout 常量为 `QKV_K_SCALE_LAYOUT_NTD = 0`、`QKV_K_SCALE_LAYOUT_TND = 1`。
4. qkv GM stride 由 tiling key 中的 layout 常量决定：`NTD` 下 `qkvGmStrideT = headDim`、`qkvGmStrideN = totalTokens * headDim`；`TND` 下 `qkvGmStrideT = qkvN * headDim`、`qkvGmStrideN = headDim`。kernel input UB 保留 layout-native staging：`NTD` 使用 `qkvUbStrideN = actualTokenTile * headDim`、`qkvUbStrideT = headDim`；`TND` 使用 `qkvUbStrideT = qkvN * headDim`、`qkvUbStrideN = headDim`。VF 通过 `n_qkv * qkvUbStrideN + localToken * qkvUbStrideT` 访问逻辑 `(localToken,n_qkv,d)`。
5. `batch = query_start_loc.shape[0] - 1`。
6. `seq_lens.shape == [batch]`，dtype 为 int32；它只作为 GM 输入传入 kernel，不写入 kernel tiling data。
7. `maxSeqLen`、`blockNum` 只用于 host 侧 shape/checker 语义，不写入 kernel tiling data。
8. `coreTokenTile/coreGroupNum` 用于 controller 给 AIC/AIV 核组划分连续 token range。
9. `dimTile` 首期固定为 `128`，由 `headDim` 和 kernel `static_assert` 约束。
10. `qkHeadTile` 不作为字段，kernel 使用 `qHeadNum + kvHeadNum`。
11. `vHeadTile` 不作为字段，kernel 使用 `kvHeadNum`。
12. `rowTile = tokenTile * (qHeadNum + kvHeadNum)` 是资源公式中的 full tile Q/K row 容量，不作为字段。
13. `rowTileAligned = align_up(rowTile, 16)` 是资源校验推导量，不作为字段；L1 A buffer 和 side-by-side L0C 物理容量还需要使用 `align_up(tokenTile*Nq,16)+align_up(tokenTile*Nk,16)` 校验。
14. `validM_tile = actualTokenTile * (qHeadNum + kvHeadNum)` 是 Vec 后处理有效行数；Cube/Fixpipe 使用 `cubeTokenSize` 推导物理 Q/K 行数，odd tail 的 pad 行参与 Cube/Fixpipe 但不进入 Vec 有效写回；这些值都不是独立 tiling data 字段。
15. UB/L1 buffer 数、字节预算和各固定 buffer 的布局均为 kernel 静态常量；host tiling 使用相同常量计算 `tokenTile` 和校验资源上界，但不写入 tiling data。
16. `cos_sin` DB、`slot_mapping` DB、`v_scale_ub`、`v_out` DB 均由 kernel 按静态布局分配，不作为 tiling data 字段写入。
17. V cache 写回属于后处理子阶段，但不等待 Cube 输出。当前 `Vec::ComputeTile` 先发起 V quant，再完成 Q/K BF16 NZ staging，随后写 `v_cache`、搬运 Q/K AL1；V cache 写回仍早于 Q/K AL1 handoff 和 `A_READY`，该策略不需要 tiling flags 字段。
18. `k_cache`、`v_cache`、`k_scale_cache` 使用 stride 字段。当前代码中 K/V cache 共用 `kvCacheStrideBlock/Head/Token`，因为 host 已校验 K/V cache layout 和 head 数一致；K scale cache 使用独立 `kScaleCacheStride*`。对外 shape 固定为 block-head-token 布局：`k_cache [BlockNum, Nk, BlockSize, D]`、`v_cache [BlockNum, Nv, BlockSize, D]`、`k_scale_cache [BlockNum, Nk, BlockSize, 1]`。字段名保持 block/head/token 三元组，对应 dim0/dim1/dim2。连续 cache 由 host tiling 合成默认 stride；非连续 cache 由 tiling 读取真实 stride。K/V cache 要求 4 维正 stride、尾轴 stride 为 1，且 head/token stride 均不小于 `D=128`；两者前三维 stride 必须一致。`k_scale_cache` 要求 4 维正 stride且尾轴 stride 为 1，其前三维 stride 作为 block/head/token 写回 stride 使用。该合同保证 kernel 可以复用一次 `slot_mapping` 解析得到的 `block_id/block_offset`，并避免 2D cache 写回的目标 stride 下溢或相邻逻辑行重叠。

连续 cache 默认 stride：

```text
k/v cache:
  strideBlock = N * BlockSize * D
  strideHead  = BlockSize * D
  strideToken = D

k_scale_cache:
  strideBlock = Nk * BlockSize
  strideHead  = BlockSize
  strideToken = 1
```

kernel cache base offset 应由 `block_id * strideBlock + block_offset * strideToken` 组成，再叠加 head/dim 偏移。

## 9. Tiling Key

首期 tiling key 表达平台、D128 实现、`layout_qkv` 和 `layout_q_out`：

```text
arch35 only
D128 实现
layout_qkv: QKV_K_SCALE_LAYOUT_NTD = 0 / QKV_K_SCALE_LAYOUT_TND = 1
layout_q_out: QKV_K_SCALE_LAYOUT_NTD = 0 / QKV_K_SCALE_LAYOUT_TND = 1
qkv dtype fixed bf16
rotation dtype fixed bf16
gamma/cos_sin/scale dtype fixed fp32
out/cache dtype fixed fp8_e4m3fn
L0C double buffer
```

`layout_qkv` 和 `layout_q_out` 必须作为 tiling key 模板常量，而不是只写入 tiling data 后在 kernel 内 runtime 分支。建议：

```cpp
#define QKV_K_SCALE_LAYOUT_NTD 0
#define QKV_K_SCALE_LAYOUT_TND 1

ASCENDC_TPL_UINT_DECL(QKV_LAYOUT, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                      QKV_K_SCALE_LAYOUT_NTD, QKV_K_SCALE_LAYOUT_TND)
ASCENDC_TPL_UINT_DECL(Q_OUT_LAYOUT, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                      QKV_K_SCALE_LAYOUT_NTD, QKV_K_SCALE_LAYOUT_TND)
```

host tiling 根据属性选择 key：

```text
layout_qkv == "NTD", layout_q_out == "NTD" -> GET_TPL_TILING_KEY(D128, QKV_K_SCALE_LAYOUT_NTD, QKV_K_SCALE_LAYOUT_NTD)
layout_qkv == "TND", layout_q_out == "TND" -> GET_TPL_TILING_KEY(D128, QKV_K_SCALE_LAYOUT_TND, QKV_K_SCALE_LAYOUT_TND)
layout_qkv == "TND", layout_q_out == "NTD" -> GET_TPL_TILING_KEY(D128, QKV_K_SCALE_LAYOUT_TND, QKV_K_SCALE_LAYOUT_NTD)
```

kernel 使用 `QKV_LAYOUT` 模板常量选择 GM copy 和 input UB stride；使用 `Q_OUT_LAYOUT` 选择 RmsNormRopeNz 后 Q/K 内部 row order、Q/K dynamic quant source layout、Q output staging 和 Q output writeback，避免 VF 热循环中的 layout runtime branch。

不进入 tiling key：

1. `query_start_loc` 模式。当前固定为当前调用 local 前缀和语义，并结合 `seq_lens` 计算 RoPE 历史偏移。
2. `head_nums` 具体数值。
3. `rotation` 是否存在。当前必须非空，由 checker 强制校验。
4. `v_scale` 是否存在。当前必须非空，由 checker 强制校验；其精确 shape 由 scene resolver 选定的合同校验，
   不单独增加 tiling key 字段。
5. Hadamard/FWHT 模式。
6. `tokenTile` 具体数值。由 tiling data 承载。

## 10. 后续扩展路线

### 10.1 M 大于 256

首期主路径不允许 full tile `rowTile > 256` 或 side-by-side 物理行数 `fullTileQkAlignedRows > 256`，因此运行时 `M_tile` 和 `qkAlignedRows` 都不会超过 256。后续可评估两种扩展：

```text
Option A: AIC 内部 M_sub 切分
  host tokenTile 可允许 rowTile > 256
  AIC 内部按 M_sub <= 256 循环
  保持 L0C double buffer
  需要新增 sub-tile 级同步、UB output offset、FIX_OUTPUT_READY_SUB

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

当 `T` 很小且 `tileNum < 可用 AIC/AIV group 数` 时，很多核组没有任务。例如 `T = 1` 时，无论 `qkN` 多大，主路径只有一个 token tile。

后续可增加 small-T 二级 tiling：

```text
tokenTileSmall = min(tokenTile, T_rem)
qkHeadTileSmall < qkN
M_tile = tokenTileSmall * qkHeadTileSmall <= 256
```

触发条件建议：

```text
mainTileNum = ceil(T / tokenTile)

if mainTileNum < targetCoreGroupNum
   and T * qkN has enough rows:
       enable small-T head-split tiling
else:
       use token-only path
```

该模式需要新增 V ownership 规则，避免同一 token 的 V cache 被多个 Q/K head shard 重复写。建议约定：

```text
同一个 token tile 内：
  qkHeadOffset == 0 的 shard 负责 V quant + v_cache 写回
  其他 Q/K head shard 只处理自己的 Q/K
```

由于 Q/K/V 在 qkv N 轴中拼接，head split 会显著增加地址映射、V ownership、同步和测试复杂度，因此只作为后续扩展。

### 10.3 Core Wave 尾块均衡

首期 host tiling 在资源上限与 `TOKEN_TILE_PER_AIV_CAP = 4` 的共同约束下，从 `initialTokenTilePerAiv` 向下选择最大可接受偶数 `tokenTile`，不额外按 core 数重调 tile 大小。因此当：

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
row = tokenOffset * qkN + qkHeadOffset
```

但该路线会解耦 Q/K 与 V 的 token 生命周期，也会增加 row 到 token/head 的反解和 V ownership 复杂度。首期不采用。

## 11. Host UT

Host 侧验证按职责拆分：InferShape UT 只覆盖最小读取防御、`NTD->TND` 拒绝、`qkvN >= Nq` 和输出 shape/dtype 推导；checker/tiling UT 覆盖其余完整输入契约、tiling data 和资源公式。

### 11.1 正常用例

1. `D=128`，`query_start_loc=[0,T]`。
2. 多 batch：`query_start_loc=[0,2,9]`，`seq_lens=[2,7]`。
3. 多 batch 历史偏移：`query_start_loc=[0,17,18]`，`seq_lens=[8209,1]`。
4. `B in {1,2}`。
5. `head_nums` 覆盖从 `(8,1,1)` 到 `(64,8,8)` 的完整 8 组合法 tuple，其中包含 `(24,3,3)` 等
   非 2 的幂 head。
6. `T in {1,128,1024,8192,32768}`。
7. `rotation`、`v_scale` 均存在，且 RoPE 使用 `[Nv]`、M-RoPE 使用 `[Nv,D]`。
8. cache 输出与输入同 shape。

### 11.2 资源与 tile 校验

Tiling UT 必须校验：

1. 主路径 `tokenTile % 2 == 0`；若实现保留 `tokenTile = 1` fallback，需要单独覆盖 fallback case。
2. `tokenTile >= 1`，主路径 `tokenTile >= 2`。
3. `MAX_TOKEN_TILE == 16` 是资源硬上限；当前 `TOKEN_TILE_PER_AIV_CAP == 4`，因此主路径 `ceil(tokenTile / 2) <= 4` 且 `tokenTile <= 8`。
4. `rowTile == tokenTile * (Nq + Nk)`。
5. `rowTile <= 256`。
6. 非 tail tile `validM_tile == rowTile`，tail tile `validM_tile <= rowTile`；Cube/Fixpipe 物理行数按 `cubeTokenSize` 对齐。
7. `rowTileAligned == align_up(rowTile, 16)`。
8. `fullTileQkAlignedRows == align_up(tokenTile * Nq, 16) + align_up(tokenTile * Nk, 16)`，且 `fullTileQkAlignedRows <= 256`。
9. host tiling 先推导 `resourceTokenTilePerAiv = min(MAX_TOKEN_TILE / 2, floor(128 / qkN), floor(160 / qkvN), floor(80 / Nv))`，再计算 `initialTokenTilePerAiv = min(resourceTokenTilePerAiv, TOKEN_TILE_PER_AIV_CAP)`；当前 cap 为 4，之后从偶数候选递减搜索。
10. `ceil(tokenTile / 2) <= floor(160 / qkvN)`。
11. `ceil(tokenTile / 2) <= floor(128 / qkN)`。
12. `ceil(tokenTile / 2) <= floor(80 / Nv)`。
13. 不再要求 `align_up(tokenTile * Nq,16) + align_up(tokenTile * Nk,16) <= 128`；该条件属于旧的 Q/K 同驻 L0A 方案。
14. UB/L1 总量、buffer 数和各固定 buffer 的大小不写入 tiling data。
15. Q/K FP8 staging 不分配独立 buffer，复用 output DB pool。
16. L1 静态预算暂按两个内部 rotation resident slot 计算，满足 `64 KiB + 4 * 64 KiB <= 512 KiB`。
17. `dimTile` 的语义由 `headDim == 128` 和 kernel 静态约束共同保证，不作为 tiling data 字段。
18. `qkHeadTile` 和 `vHeadTile` 只作为资源推导量，分别等于 `qHeadNum + kvHeadNum` 和 `kvHeadNum`。
19. `qHeadNum`、`kvHeadNum` 与 host 的 `head_nums=[Nq,Nk,Nv]` 一致，且 host 已校验
    `0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`。
20. 全部 8 组合法 head tuple 的 `tokenTile` 与第 7 节表格一致。
21. tail tile 覆盖 `T < tokenTile` 和 `T % tokenTile != 0`。

### 11.3 错误用例

1. 缺失 `rotation`。
2. 缺失 `v_scale`。
3. `rotation.shape != [D,D]`。
4. `rotation.dtype != bf16`。
5. RoPE `v_scale.shape != [Nv]`，或 M-RoPE `v_scale.shape != [Nv,D]`。
6. `cos_sin.shape[1] != D`。
7. `q_gamma.shape != [D]` 或 `k_gamma.shape != [D]`。
8. `slot_mapping.shape != [T]`。
9. `query_start_loc` rank 不是 1 或长度小于 2。
10. `seq_lens` rank 不是 1、shape 不是 `[Batch]` 或 dtype 不是 int32。
11. 若 host 可读 `query_start_loc`，覆盖 `query_start_loc[0] != 0`、非单调、`query_start_loc[Batch] != T`。
12. 覆盖 `seq_lens[b] < query_start_loc[b + 1] - query_start_loc[b]` 或超出 `cos_sin.shape[0]` 的非法输入；前者属于调用方未满足合法输入约束，kernel 行为未定义。
13. `head_nums` 缺失、长度不是 3、任一值非正。
14. `qkv` rank 不是 3，`layout_qkv` 不是 `NTD/TND`，`layout_q_out` 不是 `NTD/TND`，`layout_qkv=NTD,layout_q_out=TND`，或按 `layout_qkv` 解释出的逻辑 N 轴长度不等于 `sum(head_nums)`。
15. cache 头维与 `head_nums` 不一致，或 `v_scale.shape[0]` 与 `kvHeadNum` 不一致；M-RoPE 还覆盖
    `v_scale.shape[1] != D`。
16. FP8 dtype 不是 `float8_e4m3fn`。
17. `Nq > 64` 时不进入当前实现；使用 `(72,9,9)` 隔离验证上限门禁。
18. `Nq != 8 * Nk` 时不进入当前实现；使用 `(16,4,4)` 隔离验证比例门禁。
19. `Nk != Nv` 时不进入当前实现；使用 `(16,2,1)` 隔离验证 K/V head 门禁。
20. `D != 128` 时不进入当前实现。
21. `initialTokenTilePerAiv < 1` 时不进入当前实现。

### 11.4 InferShape 最小集边界

1. InferShape 只读取 `head_nums[0]`、`layout_qkv/layout_q_out`、`qkv` 中的 T/D 以及三个 cache 输入 shape。
2. InferShape 只做避免非法读取和生成有效输出所需的最小指针、rank、ListInt 长度、正维度与布局检查；保留 `NTD->TND` 拒绝和逻辑 `qkvN >= Nq`。
3. `head_nums` 完整三项一致性、`0 < Nq <= 64`、`Nq == 8 * Nk`、`Nk == Nv`、
   `qkvN == sum(head_nums)`、cache/其他输入 shape/dtype、D 与资源约束由 ACLNN checker 和
   tiling/base tiling 验证，不要求 InferShape 重复校验。

## 12. Kernel 验证（DiscussionForum 用例）

`ops-transformer` 内不新增 `tests/ut/op_kernel` 或 ST target。Kernel 的数值、layout、tail、多 tile、cache stride 和端到端无死锁验证直接使用 DiscussionForum 资产：

```text
designFile/qkv_rms_norm_rope_cache_with_k_scale_tasks/assets/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py
designFile/qkv_rms_norm_rope_cache_with_k_scale_tasks/assets/qkv_rms_norm_rope_cache_with_k_scale_precision.py
GEN_CASE/qkv_rms_norm_rope_cache_with_k_scale/test_cases/*.csv
```

前两个入口依赖已安装算子、`torch_npu` 和可用 NPU；TTK CSV 由外部 TTK harness 执行。验证入口存在不等于当前 worktree 已执行通过。

DiscussionForum NPU golden/TTK 按阶段覆盖：

1. 前处理：Q/K/V split、RMSNorm、RoPE 位置映射。
2. AIV token 分配：主路径偶数非 tail tile 两个 AIV 各处理 `tokenTile / 2` 个 token；tail/fallback 按 `cubeHalfTokenSize/aivTokenSize` 分配。
3. Tail/fallback token 分配：`actualTokenTile` 为奇数时允许 AIV0 比 AIV1 多 1 个 token，且 AIV1 可能为 0 token。
4. Cube：Q/K segment 分开循环，分别执行 `q_rope_out @ rotation`、`k_rope_out @ rotation`；L0C 输出保持 `[Q aligned rows][K aligned rows]` side-by-side 布局，覆盖 full-D 和动态 D 方向 L0A split 两种路径。
5. V-first 调度：覆盖 `QuantVToFp8 -> BuildRopeBf16NzUb -> ScatterVCache -> Q/K AL1 copy -> A_READY`；确认 V 写回不等待 Cube，且在 Q/K AL1 handoff 和 `A_READY` 前完成。
6. 后处理：Q/K dynamic quant/cache。
7. 端到端：完整输出对比 golden。

必须覆盖：

1. `slot_mapping` 连续、非连续、跨 block。
2. 多 batch local 前缀和位置。
3. tile 跨 `query_start_loc` 边界时 RoPE 位置正确，并覆盖不同 batch 的 `seq_lens` 历史偏移不同时 position 不连续。
4. 独立用例覆盖 `v_scale` 含 0；M-RoPE 常规输入使用 channel-wise 非均匀 scale，不能只生成每个
   head 内相同的行，也不在常规输入中固定注入 0。
5. random `rotation`，覆盖非 Hadamard dense 矩阵。
6. cache alias。

同步协议另做静态设计审计：核对实际 event id、pipe pair、live buffer 冲突、循环/Init/End 中 Set/Wait 数量与顺序；不要求为 event id 范围增加 `static_assert`。NPU 长序列和多 tile 用例负责验证执行无超时、无死锁且输出正确。

## 13. 验收标准

1. InferShape UT 能证明默认/合法 layout、最小读取防御、`NTD->TND` 拒绝、`qkvN >= Nq` 和五个输出 shape/dtype 推导稳定，不把其余完整输入契约下沉到 InferShape。
2. Host checker/tiling UT 能证明当前接口、完整 contract 和实现选择行为稳定。
3. Tiling UT 能证明 UB/L1/L0C 公式全部满足。
4. Tiling UT 能证明主路径 `tokenTile` 为偶数，覆盖 `tokenTile = 1` fallback，且有效行数、Cube/Fixpipe 物理行数、`qkAlignedRows` 都不超过设计上界。
5. Tiling UT 能证明典型 shape 的 `tokenTile` 与表格一致。
6. DiscussionForum NPU golden/TTK 输出与 golden 对齐，长序列多 tile 执行无超时、无死锁。
7. 不出现 `SEQ_LEN_MODE` 双模式相关测试。
8. arch35 binary/config 能正确选择本 kernel。

## 14. M-RoPE 组合模式 Tiling 与验证

### 14.1 `ROPE_MODE/K_QUANT_MODE/Q_QUANT_MODE` 模板与 tiling data

在现有 D128、`QKV_LAYOUT`、`Q_OUT_LAYOUT` 后增加三个独立模板参数：

```cpp
ASCENDC_TPL_UINT_DECL(ROPE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
    QKV_K_SCALE_ROPE_MODE_ROPE, QKV_K_SCALE_ROPE_MODE_MROPE)
ASCENDC_TPL_UINT_DECL(K_QUANT_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
    QKV_K_SCALE_K_QUANT_MODE_FP8, QKV_K_SCALE_K_QUANT_MODE_INT8)
ASCENDC_TPL_UINT_DECL(Q_QUANT_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
    QKV_K_SCALE_Q_QUANT_MODE_PER_TOKEN_PER_HEAD, QKV_K_SCALE_Q_QUANT_MODE_NO_QUANT)
```

`query_start_loc/seq_lens/mrope_position/mrope_section` 在接口上均为 optional。令 `M` 表示
`mrope_section` 非空，而不是仅表示 ListInt 指针存在；框架物化出的空列表和显式 `[]` 都令 `M=0`。
host 先只检查 effective presence，不解引用 optional tensor 数据：`Q/S/P/M=1100` 识别 RoPE 候选，
`0011` 识别 M-RoPE 候选，其他组合直接失败；再校验候选
场景的量化模式、dtype/output/cache/layout 合同。只有 M-RoPE presence 加
`q_quant_mode=NoQuant`、BF16 q_out、qScale absent、INT8 k_cache、`v_scale.shape=[Nv,D]` 和 TND->TND 全部成立时选择新
key。RoPE 场景则要求 `q_quant_mode=PerTokenPerHead`。RoPE 场景的 3 个 key 保持不变，M-RoPE 只生成
1 个 TND->TND compact key，总计 4 个注册选择。host 只允许
`(ROPE, FP8, PER_TOKEN_PER_HEAD)` 和 `(MROPE, INT8, NO_QUANT)` 两组模板组合，不生成交叉组合；三个
mode 均不写入 tiling data，kernel 不做 runtime scene 判断。
tiling data 尾部只追加：

```cpp
uint64_t mropeSectionH;
uint64_t mropeSectionW;
```

令 $D_{\mathrm{half}}=D/2$，host 校验 `mrope_section` 三项非负、H/W 不超过第 14.3 节定义的
$(C_{\mathrm H},C_{\mathrm W})$，并校验三项之和不超过 $D_{\mathrm{half}}$；第 0 项不做独立 lane
上限校验，随后直接丢弃且不写入 tiling data。

既有 `batch` 字段改为 scene-dependent 派生量：RoPE 场景从
`query_start_loc.shape[0]-1` 计算且继续校验 `batch > 0`；M-RoPE 不需要 batch，写 0 sentinel，
M-RoPE kernel 模板不得读取该字段或空的 `query_start_loc/seq_lens` GM anchor。base tiling 读取 index
8/9 时改用 optional input API，presence resolver 必须先于 `FillShapeDerivedFields()` 和
`ValidateScalarInputs()`，判定完成后才访问对应 shape/dtype。BasicBlock 的 `InitAic/InitAiv` 将完整
tiling data 指针传给 Cube/Vec；Vec 在 Init 中通过编译期 `IS_MROPE` 条件令 M-RoPE 的 `batch_` 为 0，
只复制本场景字段并且不保存指针。M-RoPE 热循环不得读取 `batch` 或两个空位置 anchor。

### 14.2 UB 资源

RoPE 的普通 raw T/H/W buffer 绑定当前 host 的 `TOKEN_TILE_PER_AIV_CAP=4`；M-RoPE 实际只使用 compact
布局，`mrope_position` slot 按 kernel 通用 `MAX_TOKEN_TILE_PER_AIV=8` 分配，当前只使用其中最多 48 B：

```text
raw T/H/W one buffer = 3 * 4 * 128 * sizeof(float) = 6 KiB
raw DB = 12 KiB
mrope_position one buffer = 128 B (generic eight-token slot; current cap=4 uses 48 B)
mrope_position DB = 256 B
Gather id = 64 * sizeof(uint32) = 256 B
```

其他固定 buffers 原静态分配容量合计 31 KiB；以 12 KiB raw DB 替换原 8 KiB cos/sin DB，再加入 256 B
`mrope_position` DB 和 256 B Gather index 后，RoPE 共享布局的静态分配容量合计为 35.5 KiB，仍小于 40 KiB。`mrope_position` 以
`[tileT,3]` 单块连续搬入，token stride 为 3 个 int32，列 stride 为 1。`tileT=1/2/3/4` 的 GM 有效字节数
分别为 12/24/36/48 B；128 B 是单 slot UB 容量，必须通过 `DataCopyPad` 或等价安全尾搬运精确读取有效
字节并在 UB 补齐，禁止按 128 B 对 GM 扩读。RoPE 普通总 UB 保持 `80+128+40=248 KiB`；M-RoPE 的
`[Nv,D]` scale 在当前 `Nv<=8,D=128` 合同下最多占 `8*128*sizeof(float)=4 KiB`。host 继续保证
`ceil(tokenTile/2) <= 4`。

M-RoPE 只有 compact direct-Q 编译期路径，不使用 RoPE 的普通 reserve，也没有独立的第二条 M-RoPE
模板参数。选择条件为 M-RoPE、TND->TND、INT8 K、NoQuant BF16 Q，且当前 tile 的 `kRows <= 14`、
Q/K preprocess 容量不超过 40 KiB、输入行数不超过 80；不满足时 host 降低 token tile，最小 tile 仍不满足
则拒绝。compact 的 input/raw cos-sin 分别使用独立计数器；owner 根据各自物理 pool 与单 slot 容量
局部推导槽数，二者当前都为 4。目标路径使用 80 KiB input pool、80 KiB `outputDbPoolUb_`、14 KiB
K rotation DB、40 KiB compact reserve，并在其后放置 12 KiB 的 1024-token position window，静态总量为
`80+80+14+40+12=226 KiB`。compact reserve 中 `vScaleUb_` 从 64 B 扩为 4 KiB，后续索引、raw、V output
和 K quant 子区整体后移；不得覆盖这些 live range，也不得复用 position window。

M-RoPE q_out Fixpipe 使用默认 GM cache 行为。平台 compile info 不读取 L2 大小，GetTilingKey 不按
Q/K/V/K-scale 输出字节生成第二个模板。Q 的 F32->BF16 转换由该 Fixpipe 完成并直接写 `q_out` GM，
不经过 Vec QCast；旧的 `QCastBf16D128TndVfImpl` 路径已删除。

### 14.3 Host/API/InferShape UT

Host/Tiling/API UT 后续至少覆盖以下容量定义。令 $D_{\mathrm{half}}=D/2$，则

$$
\begin{aligned}
C_{\mathrm H}&=\left\lfloor\frac{D_{\mathrm{half}}+1}{3}\right\rfloor,\\
C_{\mathrm W}&=\left\lfloor\frac{D_{\mathrm{half}}}{3}\right\rfloor.
\end{aligned}
$$

1. OpDef input index 8/9 均为 OPTIONAL，ACLNN/L0 的 `queryStartLoc/seqLens` 固定位置不变、既有参数
   相对顺序不变；新增 `mrope_position` 会使 `headNums` 及后续参数的绝对序号顺延。presence 全矩阵只接受
   `Q/S/P/M=1100` 和 `0011`，其中 `M` 表示非空 section。覆盖 `Q xor S`、`P xor M`、两组同时存在、
   两组同时缺失，以及存在但 shape 缺失或零元素 tensor 的拒绝路径；显式空 `mrope_section=[]` 与未传
   section 等价，并覆盖其配合合法 `query_start_loc/seq_lens` 仍进入 RoPE 的成功路径。
2. 两个 presence signature 分别通过完整合法模式；presence 合法但量化模式、dtype/output/cache/layout
   不匹配时仍拒绝，不生成 key。覆盖非法 `q_quant_mode`、`PerTokenPerHead + BF16` 和
   `NoQuant + FP8`。
3. RoPE 场景保留 NTD->NTD、TND->TND、TND->NTD 三个 key；M-RoPE 只生成一个 TND->TND compact
   key，总计 4 个注册选择；tiling data 中不出现 scene flag。
4. RoPE 场景的 `batch=query_start_loc.shape[0]-1`；M-RoPE 的 `batch=0`，且 key/kernel 不读取两个空
   RoPE 位置输入。
5. OpDef 五个输出均为 REQUIRED。RoPE 的 Torch/ACLNN q_scale 为 FP32；M-RoPE 的 Torch
   运行值为 `None`，ACLNN `qScaleOptional` 为 `nullptr`。
6. `mrope_position` 的 int32、rank2、`[T,3]`；所有 M-RoPE 数值用例默认按 case seed 在
   `[0,MaxSeqLen)` 独立均匀随机生成三列位置，并用 `T=5` 覆盖 token/axis 维颠倒、末维不为 3 和
   第 0 维不等于 qkv token 数；`T=3` 的方阵不能作为维序校验的唯一用例。
7. `mrope_section` 长度为 3 且三项非负，H/W 上限分别为 $(C_{\mathrm H},C_{\mathrm W})$，三项之和不超过
   $D_{\mathrm{half}}$；覆盖 H/W 越界和总和越界拒绝。第 0 项不做独立 lane 上限校验，固定 H/W=20
   并遍历 `{0,1,11,24}`，均不影响 key、tiling data 或计算选择。
8. D128 明确覆盖 H/W 上限 `[21,21]`、总和 64 的合法边界和总和 65 的拒绝路径，并分别验证 cos/sin 两个 half。
9. `q_quant_mode` 只支持 `PerTokenPerHead`/`NoQuant`，并与候选场景、q_scale presence 及
   `q_out_dtype` 一致；`q_out_dtype` 只支持 FP8 E4M3FN/BF16，q_out dtype 必须一致。
10. InferShape 不读取 `query_start_loc/seq_lens/mrope_position/mrope_section/q_quant_mode/q_out_dtype` 的内容或 presence 做模式校验；
   五个 required output 均按固定 index 取得，q_scale 始终设置 shape，
   并继续保留 `NTD->TND`、`qkvN>=Nq` 和既有最小集边界。
11. M-RoPE 的 NTD->NTD、TND->NTD、NTD->TND 均由 checker/base tiling 拒绝，不要求 InferShape
   增加 scene-aware layout 校验。
12. `v_scale` shape 按候选场景严格校验：RoPE 仅 `[Nv]`，M-RoPE 仅 `[Nv,D]`；覆盖 M-RoPE 的
   `[Nv]`、`[Nv,D-1]`、`[Nv+1,D]` 和 RoPE 的 `[Nv,D]` 拒绝用例。

### 14.4 DiscussionForum NPU 验证

Kernel 验证仍不新增 `ops-transformer` kernel UT/ST。DiscussionForum golden/TTK 增加
`mrope_section[0]` 不变量、
`mrope_section[1] != mrope_section[2]`、token-major `[T,3]` 中三列非连续 mrope_position、BF16 Q/Torch q_scale `None`、INT8 K、逐 head/通道 `[Nv,D]` V scale、cap=4 跨 tile、tail、cache
scatter 和 RoPE 场景回归。同步协议继续以静态 Set/Wait 审计加 NPU 多 tile 无死锁验证为准。

大输出精度边界保留 `Nq/Nk/Nv=16/2/2`、`T=45000`：实际 Q+K+V+K-scale 写入约
198.10 MiB；性能压力矩阵在相同 head 配置下增加 `T=65536/131072` 的 d-pad case，并与同 shape
的 RoPE TND->TND case 配对。四路输出均通过精度比较；全量 DiscussionForum 回归为 `185/185`，
section-0 不变量比较为 `3/3`。

这里的 M-RoPE NPU golden 矩阵按同一个 compact 模板的输出规模组织，不构造第二个 M-RoPE key：

| 分组 | 触发路径 | 重点覆盖 |
|------|----------|----------|
| compact 常规输出 | M-RoPE、TND->TND、INT8 K、NoQuant BF16 Q | input/raw 各四槽、独立 K rotation/K quant、metadata window、`Stage -> K_CONSUMED -> Scatter` 同控制迭代、K quant reusable event、direct-Q、tail 和多 tile |
| compact 大输出 | 同一合同，覆盖 `T=45000/65536/131072` | 与上组相同；重点验证长序列、多 tile 和 Q/K/V/K-scale 写回 |

整个矩阵沿 token 长度（单 tile、跨 tile、tail）、head 组合、非连续位置、section 边界和 cache stride
交叉取样；compact 额外构造单核 token span 超过 1024 的 metadata refill。CPU/Python golden 对
`q_out`、`k_cache`、`v_cache`、`k_scale_cache` 逐元素比较；分组语义只区分输出规模，不对应不同
kernel 模板或 cache hint。
