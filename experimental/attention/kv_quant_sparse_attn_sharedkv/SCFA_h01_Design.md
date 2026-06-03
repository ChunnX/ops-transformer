# KvQuantSparseAttnSharedkv SCFA — h01 Lead-by-1 Pipeline Design

> Best-kept optimization. Single functional file changed:
> `op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel.h`.
> Result: **131.463us → 109.067us (-17.0%)**, accuracy **PASS 99.9948%**.
> No L1/UB/GM growth; no tiling/metadata change.

---

## English

### 1. Background

The SCFA kernel for `KvQuantSparseAttnSharedkv` runs on Ascend 950 as a CV-fused
producer/consumer pipeline across the **AIC (cube)** and **AIV (vector)** cores.

- Query: `bf16`. KV: `fp8_e4m3fn`. Case shape (`my_test`, layout BSND / KV PA_ND / SCFA):
  `B=48, N1=128, N2=1, S1=2, S2=8192, D=512, K(sparse blocks)=1024, rope_head_dim=64`
  → `gSize = N1/N2 = 128` drives the `IS_SPLIT_G` path.
- The KV must be **dequantized fp8→bf16 on the AIV** and round-tripped through GM
  (`v0ResGm`) before the cube can consume it — the Ascend 950 cube cannot dequant during
  its K-load, so this GM round-trip is structural (see `hw_no_cube_inline_dequant`).
- `IS_SPLIT_G` path: `gSize=128` splits M across two AIC cores that share the dequant KV
  through a per-pair GM channel.

The pipeline is driven by one outer loop over S2 tiles (`taskId`). Each iteration the AIV
produces a dequant tile and the cube consumes earlier tiles via matmuls.

### 2. The bottleneck h01 fixes

Original schedule (pre-h01, "a05"): the cube's `IterateLoadQK(tile T)` blocked on
`v0ResGm.WaitCrossCore()` for the **same tile T** that the AIV had *just* produced in the
**current** iteration. The cube therefore stalled on the producer every iteration —
`aic_mac_ratio` sat at **0.53** (cube ~47% idle waiting on dequant).

**Key insight:** the per-iteration group barrier `flag-15`
(`CrossCoreSetFlag/WaitFlag<0,PIPE_MTE3>(15)`) is **AIV-group-only — the cube does not
participate in it**. So nothing prevents the cube from running a full round *behind* the
producer, consuming a tile the producer finished in a *previous* iteration. The earlier
"a05 is the producer-bound floor" belief was wrong because it assumed the cube was pinned
to the producer by this barrier.

### 3. The h01 design: lead the producer by one round

Shift the **entire cube consumer chain** (`LoadQK / Bmm1 / Bmm2`) **and** the AIV
`Vec1 / Vec2` one global iteration deeper. **`Vec0` (the dequant producer) stays put.**
Net effect: `LoadQK(tile T)` now reads the `v0ResGm` tile that `Vec0` produced in the
*previous* iteration, so it no longer waits on the current iteration's dequant.

Pipeline depth grows from 3-deep to 4-deep. Per-tile stage assignment:

| Stage (role) | a05 tile read | h01 tile read |
|---|---|---|
| Vec0 (AIV produce) | T | **T** (unchanged) |
| LoadQK (cube load) | T | **T-1** |
| Bmm1 (cube QK^T) | T-1 | **T-2** |
| Bmm2 (cube PV) | T-2 | **T-3** |
| Vec1 (AIV softmax) | T-1 | **T-2** |
| Vec2 (AIV update) | T-2 | **T-3** |

The whole chain shifts uniformly by one, so every *internal* producer↔consumer spacing is
preserved; only the producer→first-consumer gap opens by one round (the round we now
overlap).

### 4. Code changes (kernel.h only)

Three edits, **none of which is a tensor / on-chip buffer**:

1. `PRELOAD_NUM 2 → 3` (line 70) — one extra drain iteration at the tail (`gS1LoopEnd +=
   PRELOAD_NUM`) so the deepest tile flushes Vec0→LoadQK→Bmm1→Bmm2.
2. `RunInfo runInfo[3] → runInfo[4]` (line 360) — the **metadata** ring grows to 4 so the
   shifted consumers can still index their tile's scalar params. `RunInfo` is plain
   scalars (s2 indices, sizes, offsets), living in registers/stack — **not UB/L1**.
3. `bool loadDrainOff` (new) — turns LoadQK off one drain step before Bmm1 so the tail
   drains in the correct order. A single `bool` in a register.

Plus the index updates inside the inner loop: array lookups become `taskId % 4`,
consumers move to `(taskId+2)%4` / `(taskId+1)%4` / `(taskId+3)%4`, and the `taskId>N`
guards shift by one (`>0→>1`, `>1→>2`) to match the deeper warm-up.

The drain `switch` gains a `case 2` so the last-block tail runs three extra steps
(`notLastTwoLoop` / `loadDrainOff` / `notLast` staged off across cases 0,1,2).

### 5. Why on-chip memory (UB / L1) is still enough — unchanged footprint

**The decisive detail: metadata index and buffer-slot index are decoupled.**

- `runInfo[taskId % 4]` — array index of the `RunInfo` **struct** (metadata). Grew 3→4.
- `runInfo.taskIdMod3 = taskId % 3` (line 520) — the **physical buffer-slot** index.
  Every on-chip `Get()` still uses **mod-3**:
  `l1RightBuffers.Get(runInfo.taskIdMod3)`, `v0ResGmBuffers.Get(runInfo.taskIdMod3)`.

So the rings stay **3-deep**; the on-chip byte footprint is identical before and after h01.

**Live-set proof the 3-deep ring suffices** (binding buffer = `l1Right`):
- LoadQK writes `l1Right[T-1]`; Bmm1 reads `l1Right[T-2]`; Bmm2 reads `l1Right[T-3]`.
- Simultaneously-live tiles = {T-1, T-2, T-3} = **3** → exactly fills the 3-deep ring.
- Pre-h01 the live set was {T, T-1, T-2} = also 3. Lead-by-1 shifts every stage uniformly,
  so the producer↔furthest-consumer spacing (2 stages) is unchanged → same count → fits.

### 6. Buffer allocation inventory

Compile-time tile constants: `s1BaseSize=64`, `s2BaseSize=128`, `D=512`, `dBaseMatmulSize=128`,
`CV_RATIO=2`, KV(fp8)=1B, Q/bf16=2B, T(fp32)=4B, `dVTemplateTypeInput=640`.

#### UB (AIV)

`ubBufferManager.Init(pipe, mm1ResultSize*2 + mm2ResultSize)` = **96 KB** matmul-result ring:

| Tensor | Pool / depth | Size each | Total |
|---|---|---|---|
| `bmm1Buffers` (QK^T result) | DB ×2 | `64/2·128·4` = 16 KB | 32 KB |
| `bmm2Buffers` (PV result) | single ×1 | `64/2·512·4` = 64 KB | 64 KB |

Vector working buffers (sized by tile dims, **independent of pipeline depth**):

| Tensor | Size |
|---|---|
| `stage0InBuf[2]` (fp8 KV in) | 2 × 640·16·1 = 20 KB |
| `stage0OutBuf[2]` (dequant bf16 out) | 2 × 640·17·2 ≈ 42.5 KB |
| `stage1OutQue[2]` (softmax cast out) | 2 × 33·128·2 ≈ 16.5 KB |
| `stage2OutBuf` (flash-update out) | 32·512·4 = 64 KB |
| `dequantScaleBuff` | 64·16·2·4 = 8 KB |
| `softmaxMax/Sum/Exp[2]` (6 bufs) | 6 × 256 B = 1.5 KB |
| `commonTBuf` + `sinksBuf` | 0.5 + 0.5 KB |
| `blkTableBuf` / `sparseIdxBuf` / `kvPhyAddrBuf` | 128 / 4 / 8 KB — phys-addr prologue (`GetKVPhyAddr`), separate pipe phase, not co-resident with the main loop |

#### L1 (AIC)

`l1BufferManager.Init(pipe, 416K)` + 96 KB l1Q = **512 KB (full L1)**:

| Tensor | Depth | Size each | Total |
|---|---|---|---|
| `l1PBuffers` (P, Bmm2-left) | DB ×2 | `64·128·2` = 16 KB | 32 KB |
| `l1RightBuffers` (dequant KV) | 3buff ×3 | `128·512·2` = 128 KB | **384 KB** |
| `l1QBuffers` (query) | single | 96 KB | 96 KB |

L0 (fixed): L0A 32 KB, L0B 64 KB, L0C 256 KB.

#### GM channel (not on-chip)

`v0ResGmBuffers` — `BuffersPolicy3buff` in workspace, `128·512·2 = 128 KB` each ×3 =
**384 KB** per AIC-pair. The fp8→bf16 dequant round-trip lives here (structural, per
`hw_no_cube_inline_dequant`); only consumer is `IterateLoadQK`.

> L1 is **exactly full** at 3-deep `l1Right` (384+32+96 = 512 KB). This is both why h01 is
> safe (kept depth = 3) and why a larger block (g03, s2=256) faulted: `l1Right` alone would
> need 3×256 KB = 768 KB, overflowing L1.

### 7. Result

| Metric | a05 (before) | h01 (after) |
|---|---|---|
| Task Duration (avg) | 131.463 us | **109.067 us (-17.0%)** |
| `aic_mac_ratio` | 0.53 | **0.69** |
| Accuracy (PctRlt, thd 99.50%) | PASS | **PASS 99.9948%** |

The gain is pure scheduling/overlap — the producer's work is unchanged; the cube simply
stops waiting on it. The residual ~15us gap to the pure-cube floor is the irreducible
per-tile dependency (AIV dequant path length), not a hideable stall; deeper leads (h02
lead-by-2) do not compound the gain.

---

## 中文

### 1. 背景

`KvQuantSparseAttnSharedkv` 的 SCFA kernel 运行在 Ascend 950 上，是 **AIC(cube)** 与
**AIV(vector)** 两类核之间的 CV 融合生产者/消费者流水线。

- Query 为 `bf16`，KV 为 `fp8_e4m3fn`。用例 shape(`my_test`，layout BSND / KV PA_ND / SCFA)：
  `B=48, N1=128, N2=1, S1=2, S2=8192, D=512, K(稀疏块数)=1024, rope_head_dim=64`
  → `gSize = N1/N2 = 128`，走 `IS_SPLIT_G` 路径。
- KV 必须在 **AIV 上做 fp8→bf16 反量化**，并经 GM(`v0ResGm`) 中转后 cube 才能消费——Ascend 950
  的 cube 在取 K 时无法做反量化，因此这条 GM 往返是结构性的（见 `hw_no_cube_inline_dequant`）。
- `IS_SPLIT_G` 路径：`gSize=128`，M 维拆到两个 AIC 核上，二者通过每对核的 GM 通道共享反量化 KV。

流水线由一层 S2 tile 外循环(`taskId`)驱动：每次迭代 AIV 生产一个反量化 tile，cube 通过矩阵乘消费较早的 tile。

### 2. h01 解决的瓶颈

原始调度(h01 之前，"a05")：cube 的 `IterateLoadQK(tile T)` 会在 `v0ResGm.WaitCrossCore()`
上阻塞，等待 AIV 在 **当前迭代刚刚** 产出的 **同一个 tile T**。于是 cube 每次迭代都卡在生产者上，
`aic_mac_ratio` 仅 **0.53**（cube 约 47% 时间空等反量化）。

**关键洞察：** 每次迭代的组屏障 `flag-15`
(`CrossCoreSetFlag/WaitFlag<0,PIPE_MTE3>(15)`) 是 **AIV 组内专用——cube 不参与**。因此没有任何
机制把 cube 钉在生产者上，cube 完全可以整整落后生产者一轮，去消费生产者在 **上一轮迭代** 就已完成
的 tile。此前"a05 是生产者瓶颈下界"的结论是错的，因为它误以为该屏障把 cube 与生产者绑定。

### 3. h01 设计：让生产者领先一轮

把 **整条 cube 消费链**(`LoadQK / Bmm1 / Bmm2`) **以及** AIV 的 `Vec1 / Vec2` 整体后移一个全局
迭代，**`Vec0`(反量化生产者)位置不变**。最终效果：`LoadQK(tile T)` 改读 `Vec0` 在 *上一轮* 产出
的 `v0ResGm`，从而不再等待本轮的反量化。

流水线深度由 3 段加深为 4 段。各阶段 tile 分配：

| 阶段(角色) | a05 读取 tile | h01 读取 tile |
|---|---|---|
| Vec0 (AIV 生产) | T | **T**(不变) |
| LoadQK (cube 取数) | T | **T-1** |
| Bmm1 (cube QK^T) | T-1 | **T-2** |
| Bmm2 (cube PV) | T-2 | **T-3** |
| Vec1 (AIV softmax) | T-1 | **T-2** |
| Vec2 (AIV 更新) | T-2 | **T-3** |

整条链统一后移一格，所有 *内部* 生产者↔消费者间距都保持不变；仅生产者→首个消费者的间距张开一轮
（也就是我们现在用来做重叠的那一轮）。

### 4. 代码改动（仅 kernel.h）

三处改动，**没有任何一处是 tensor / 片上 buffer**：

1. `PRELOAD_NUM 2 → 3`（第 70 行）——尾部多一轮 drain（`gS1LoopEnd += PRELOAD_NUM`），让最深的
   tile 完成 Vec0→LoadQK→Bmm1→Bmm2 的冲刷。
2. `RunInfo runInfo[3] → runInfo[4]`（第 360 行）——**元数据** 环扩到 4，让后移的消费者仍能索引到自己
   tile 的标量参数。`RunInfo` 是纯标量（s2 索引、尺寸、偏移），位于寄存器/栈，**不占 UB/L1**。
3. `bool loadDrainOff`（新增）——在 Bmm1 之前一个 drain 步关掉 LoadQK，使尾部按正确顺序排空。
   仅一个寄存器里的 `bool`。

以及内循环里的索引更新：数组下标改为 `taskId % 4`，消费者改到 `(taskId+2)%4` / `(taskId+1)%4` /
`(taskId+3)%4`，`taskId>N` 的护栏整体后移一格(`>0→>1`, `>1→>2`) 以匹配更深的预热。

drain 的 `switch` 新增 `case 2`，使末块尾部多跑三步
(`notLastTwoLoop` / `loadDrainOff` / `notLast` 在 case 0/1/2 上分级关闭)。

### 5. 为什么片上内存(UB/L1)依然够用——占用未变

**决定性细节：元数据下标与 buffer 槽位下标是解耦的。**

- `runInfo[taskId % 4]` —— `RunInfo` **结构体**(元数据)的数组下标，由 3 扩到 4。
- `runInfo.taskIdMod3 = taskId % 3`（第 520 行）—— **物理 buffer 槽位** 下标。所有片上 `Get()` 仍用
  **mod-3**：`l1RightBuffers.Get(runInfo.taskIdMod3)`、`v0ResGmBuffers.Get(runInfo.taskIdMod3)`。

因此各环仍是 **3 深**；h01 前后片上字节占用完全一致。

**3 深环仍然够用的存活集证明**（关键 buffer = `l1Right`）：
- LoadQK 写 `l1Right[T-1]`；Bmm1 读 `l1Right[T-2]`；Bmm2 读 `l1Right[T-3]`。
- 同时存活的 tile = {T-1, T-2, T-3} = **3** → 恰好填满 3 深环。
- h01 前存活集为 {T, T-1, T-2} = 同样是 3。lead-by-1 把每个阶段统一平移，生产者↔最远消费者的间距
  (2 段)不变 → 数量相同 → 装得下。

### 6. Buffer 分配清单

编译期 tile 常量：`s1BaseSize=64`、`s2BaseSize=128`、`D=512`、`dBaseMatmulSize=128`、
`CV_RATIO=2`，KV(fp8)=1B、Q/bf16=2B、T(fp32)=4B、`dVTemplateTypeInput=640`。

#### UB (AIV)

`ubBufferManager.Init(pipe, mm1ResultSize*2 + mm2ResultSize)` = **96 KB** 矩阵乘结果环：

| Tensor | 池/深度 | 单份大小 | 合计 |
|---|---|---|---|
| `bmm1Buffers` (QK^T 结果) | DB ×2 | `64/2·128·4` = 16 KB | 32 KB |
| `bmm2Buffers` (PV 结果) | single ×1 | `64/2·512·4` = 64 KB | 64 KB |

Vector 工作 buffer（按 tile 维度定大小，**与流水深度无关**）：

| Tensor | 大小 |
|---|---|
| `stage0InBuf[2]` (fp8 KV 输入) | 2 × 640·16·1 = 20 KB |
| `stage0OutBuf[2]` (反量化 bf16 输出) | 2 × 640·17·2 ≈ 42.5 KB |
| `stage1OutQue[2]` (softmax cast 输出) | 2 × 33·128·2 ≈ 16.5 KB |
| `stage2OutBuf` (flash-update 输出) | 32·512·4 = 64 KB |
| `dequantScaleBuff` | 64·16·2·4 = 8 KB |
| `softmaxMax/Sum/Exp[2]` (6 个) | 6 × 256 B = 1.5 KB |
| `commonTBuf` + `sinksBuf` | 0.5 + 0.5 KB |
| `blkTableBuf` / `sparseIdxBuf` / `kvPhyAddrBuf` | 128 / 4 / 8 KB——物理地址前导阶段(`GetKVPhyAddr`)，独立流水段，不与主循环同时驻留 |

#### L1 (AIC)

`l1BufferManager.Init(pipe, 416K)` + 96 KB l1Q = **512 KB(占满 L1)**：

| Tensor | 深度 | 单份大小 | 合计 |
|---|---|---|---|
| `l1PBuffers` (P, Bmm2 左矩阵) | DB ×2 | `64·128·2` = 16 KB | 32 KB |
| `l1RightBuffers` (反量化 KV) | 3buff ×3 | `128·512·2` = 128 KB | **384 KB** |
| `l1QBuffers` (query) | single | 96 KB | 96 KB |

L0(固定)：L0A 32 KB、L0B 64 KB、L0C 256 KB。

#### GM 通道（非片上）

`v0ResGmBuffers` —— workspace 中的 `BuffersPolicy3buff`，每份 `128·512·2 = 128 KB`，×3 =
每对 AIC **384 KB**。fp8→bf16 反量化往返就发生在这里（结构性，见 `hw_no_cube_inline_dequant`），
唯一消费者是 `IterateLoadQK`。

> 在 3 深 `l1Right` 下 L1 **恰好占满**（384+32+96 = 512 KB）。这既是 h01 安全的原因（深度保持 3），
> 也是更大块(g03, s2=256)运行时失败的原因：仅 `l1Right` 就需要 3×256 KB = 768 KB，超出 L1。

### 7. 结果

| 指标 | a05(改前) | h01(改后) |
|---|---|---|
| Task Duration(均值) | 131.463 us | **109.067 us (-17.0%)** |
| `aic_mac_ratio` | 0.53 | **0.69** |
| 精度(PctRlt，阈值 99.50%) | PASS | **PASS 99.9948%** |

收益纯粹来自调度/重叠——生产者的工作量不变，只是 cube 不再空等。距纯 cube 下界残留的约 15us 是不可
消除的逐 tile 依赖（AIV 反量化路径长度），并非可隐藏的停顿；更深的领先(h02 lead-by-2)不会让收益叠加。
