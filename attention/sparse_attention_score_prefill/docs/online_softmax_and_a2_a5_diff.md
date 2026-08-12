# Online Softmax 原理与 A2/A5 实现差异分析

## 1. Online Softmax 原理

### 1.1 核心问题

标准 softmax: `P[i] = exp(S[i]) / sum(exp(S[j]))`

当 S 分块处理（分多个 KV block）时，每个 block 只看到部分 S 列，**无法直接计算完整的 softmax**，因为分母 `sum(exp(S[j]))` 需要所有列。

### 1.2 朴素方案的缺陷

先算完所有 QK，存到 GM，再统一做 softmax。

**问题**: S 矩阵 `[M, totalN]` 太大，放不进 UB，且需要大量 GM 读写。

### 1.3 Online Softmax 核心：增量更新

**关键观察**: softmax 的分母可以**分块累加**。

对第 i 行，遍历多个 KV block (b=0,1,...,B):

```
维护两个累加量:
  m_b = max(m_{b-1}, max(S[i, block_b]))                                    // 全局行最大值
  l_b = l_{b-1} * exp(m_{b-1} - m_b) + sum(exp(S[i, block_b] - m_b))        // 全局行和
```

最终: `P[i, j] = exp(S[i,j] - m_B) / l_B`

### 1.4 逐步推导

**Block 0** (第一个 KV block):
```
m_0 = max(S[i, block_0])
l_0 = sum(exp(S[i, block_0] - m_0))
P_0[i] = exp(S[i, block_0] - m_0)     // 未归一化的 P
```

**Block 1** (第二个 KV block):
```
m_1 = max(m_0, max(S[i, block_1]))     // 更新全局 max

// 关键: 修正之前的 l_0 (因为 max 变了)
l_1 = l_0 * exp(m_0 - m_1) + sum(exp(S[i, block_1] - m_1))

// 关键: 修正之前的 P_0 (因为 max 变了)
P_1[i] = [P_0[i] * exp(m_0 - m_1),  exp(S[i, block_1] - m_1)]
```

**Block B** (最后一个):
```
最终 P[i] = P_B[i] / l_B    // 一次性归一化
```

### 1.5 数据流示意

```
Block 0:  S[0] → m_0, l_0, P_0(未归一化)
             ↓ rescale P_0 by exp(m_0 - m_1)
Block 1:  S[1] → m_1, l_1, P_1(未归一化)
             ↓ rescale P_1 by exp(m_1 - m_2)
Block 2:  S[2] → m_2, l_2, P_2(未归一化)
             ↓
...
Block B:  S[B] → m_B, l_B, P_B(未归一化)
             ↓ 最终归一化
Output:   P = P_B / l_B
```

### 1.6 代码中的对应关系

以 A2 prefill 实现为例:

| 步骤 | 函数 | 作用 |
|------|------|------|
| 1 | `CalcRowMax` | `lm = max(S[当前block])` — 局部 max |
| 2 | `UpdateGlobalRowMax` | `hm = max(lm, gm); dm = exp(gm - hm); gm = hm` — 更新全局 max, 算修正因子 |
| 3 | `SubtractRowMaxAndExp` | `S = exp(S - hm)` — 减去新 max, 做 exp |
| 4 | `ZeroOutCausalTail` | `S[tailN:] = 0` — mask 无效列 |
| 5 | `CalcRowSum` | `ll = sum(S)` — 局部 sum |
| 6 | `UpdateGlobalRowSum` | `gl = dm * gl + ll` — 修正旧 sum + 累加新 sum |
| 7 | `DownCastP` | `P = cast(S)` — 输出未归一化 P |

### 1.7 与 PV 的配合

**关键**: P 是**未归一化**的 (没除以 l)。PV matmul 用这个未归一化 P:

```
O_partial[b] = P_b × V_b    // 每个 block 的部分输出
```

最终输出需要修正:

```
O_final = sum(O_partial[b] * rescale_b) / l_final
```

其中 `rescale_b = exp(m_{b-1} - m_b)` 是每个 block 的缩放因子。

这就是 **Phase2CombineScale** (`EpilogueRescaleO.FlashDecodeCompute`) 的作用: 合并多个 topK partial 的 O 和 stats，做最终归一化。

### 1.8 为什么比朴素方案高效

| 方面 | 朴素方案 | Online Softmax |
|------|----------|----------------|
| GM 读写 | S 全量存 GM + 读回做 softmax | S 分块处理，无需全量存储 |
| UB 需求 | O(M×totalN) | O(M×blockSize) |
| 延迟 | 全部 QK 算完才能 softmax | QK/softmax/PV 流水线并行 |
| 精度 | 一次性 max+exp | 每 block 重新 max，数值更稳定 |

### 1.9 数学等价性证明

标准 softmax:
```
P[i,j] = exp(S[i,j]) / sum_k(exp(S[i,k]))
       = exp(S[i,j] - m) / sum_k(exp(S[i,k] - m))    // 减去 max, 数值稳定
```

Online softmax 分块累加:
```
sum_k(exp(S[i,k] - m)) = sum_b(sum_{k∈block_b}(exp(S[i,k] - m)))
                        = sum_b(exp(m_b - m) * sum_{k∈block_b}(exp(S[i,k] - m_b)))
                        = sum_b(exp(m_b - m) * l_b)
```

其中 `l_b = sum_{k∈block_b}(exp(S[i,k] - m_b))` 就是每个 block 的局部 rowSum。全局 rowSum 通过 `l = l_prev * exp(m_prev - m_current) + l_current` 增量更新，数学上完全等价。

---

## 2. A2 vs A5 整体架构差异

### 2.1 核心数据流对比

```
A5 数据流 (直通):
  QK: L0C ──Fixpipe──→ UB(S) ──→ VEC softmax
  PV: VEC ──UB→L1──→ L1(P) ──→ CUBE PV

A2 数据流 (GM 中转):
  QK: L0C ──Fixpipe──→ GM(S) ──DataCopyPad──→ UB(S) ──→ VEC softmax
  PV: VEC ──DataCopyPad──→ GM(P) ──Nd2Nz──→ L1(P) ──→ CUBE PV
```

A2 的所有差异本质上都源于两个硬件限制:
1. **A2 CUBE 无 L0C→UB 直接 Fixpipe 路径** → 需要 L0C→GM→UB 两步中转
2. **A2 不支持 VEC UB→L1 DataCopy** (A5-only 特性) → 需要 UB→GM→L1 两步中转

### 2.2 GM Workspace 布局 (A2 独有)

```
workspace: [accumOut][softmaxMax][softmaxSum][all_cores_gmS][all_cores_gmP]
```

- `all_cores_gmS` = `numAicCores * gmSWorkspaceSize_` — 每 AIC 核的 S 中转空间
- `all_cores_gmP` = `numAicCores * gmPWorkspaceSize_` — 每 AIC 核的 P 中转空间
- CUBE 和 VEC 共享同一 AIC 的 workspace slot (VEC 通过 `GetBlockIdx() / GetSubBlockNum()` 映射到 AIC 索引)
- Nd2Nz GM→L1 要求 32 字节对齐源地址

---

## 3. QK Block 差异 (`block_mmad_qk_prefill_a2.hpp` vs `block_mmad_qk_prefill_arch35.hpp`)

### 3.1 类定义与常量

| 项 | A2 | A5 |
|----|----|----|
| 类名 | `BlockMmadQKPrefillA2` | `BlockMmadQKPrefillArch35` |
| DispatchPolicy | `MmadAtlasA2PrefillQK` | `MmadAtlasA5PrefillQK` |
| TileCopy | `PackedTileCopyTla<..., void>` (L0C→**GM**) | `PackedTileCopyTlaToUB<..., NO_SPLIT>` (L0C→**UB**) |
| `L0_TILE_M` | **64** | **128** |
| GM workspace | `gmSWorkspace_` + `gmSStageElems_` | 无 |

### 3.2 Init 函数

```cpp
// A2: 多一个 gmSWorkspace 参数
void Init(..., AscendC::GlobalTensor<ElementS> &gmSWorkspace) {
    ...
    gmSWorkspace_ = gmSWorkspace;
    gmSStageElems_ = L0_TILE_M * blockSize;
}

// A5: 无 GM workspace
void Init(...) { ... }
```

### 3.3 跨核同步

```cpp
// A2: MODE_2, 单 flag, 非模板 Wait
void SetCrossCoreSync(Arch::CrossCoreFlag &flag) {
    if constexpr (MODE == 0x2U)
        Arch::CrossCoreSetFlag<MODE, PIPE>(flag);      // 单 flag
}
void WaitCrossCoreSync(Arch::CrossCoreFlag &flag) {
    Arch::CrossCoreWaitFlag(flag);                      // 非模板, MODE=0/PIPE_S
}

// A5: MODE_4, 双 flag (V0+V1), 模板 Wait
void SetCrossCoreSync(Arch::CrossCoreFlag &flag) {
    if constexpr (MODE == 4U) {
        Arch::CrossCoreSetFlag<MODE, PIPE>(flag);       // V0
        Arch::CrossCoreSetFlag<MODE, PIPE>(flagV1);     // V1 (id+16)
    }
}
void WaitCrossCoreSync(Arch::CrossCoreFlag &flag) {
    if constexpr (MODE == 4U) {
        Arch::CrossCoreWaitFlag<MODE, PIPE>(flag);      // V0
        Arch::CrossCoreWaitFlag<MODE, PIPE>(flagV1);    // V1
    }
}
```

### 3.4 Fixpipe 段 (最大差异)

**A2 (L0C → GM workspace):**
```cpp
(void)ubSTensor;  // CUBE 不写 UB

// GM S layout (A2 独有)
uint32_t nRound = RoundUp(N, 16U);
auto gmSLayoutEx = tla::MakeLayout<ElementS, LayoutTagS>(L0_TILE_M, nRound);
using CopyL0CToGm = typename TileCopy::template CopyL0CToDst<decltype(gmSEx)>;

// L0C layout — RoundUp(M, 16)
auto l0CLayout = tla::MakeLayoutL0C(RoundUp(M, 16U), nAct);

// Fixpipe:
WaitCrossCoreSync(mm1ToSmFlag);                    // 非模板
SetFlag<M_FIX>(l0CEventId);
WaitFlag<M_FIX>(l0CEventId);

uint32_t gmStageOffset = (bi % 2U) * gmSStageElems_;
uint32_t mFixAligned = RoundUp(M, 16U);
copyL0CToGmSub0(gmSTile, l0CFullTile);             // L0C → GM

PipeBarrier<PIPE_FIX>();                            // 确保 GM 可见性 (A2 独有)
SetFlag<FIX_M>(l0CEventId);

// 循环结束后:
SetCrossCoreSync<0x2U, PIPE_FIX>(mm1ToSmFlag);      // MODE_2
```

**A5 (L0C → UB, 按 group 分割到 2 个 AIV):**
```cpp
// 使用 ubSTensor 作为 fixpipe 目标
auto l0CLayout = tla::MakeLayoutL0C(M, nAct);

// Fixpipe:
WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);       // 模板 MODE_4
SetFlag<M_FIX>(l0CEventId);
WaitFlag<M_FIX>(l0CEventId);

// 按 group 分割到 2 个 AIV
uint32_t gSplit = CeilDiv(groupCount, 2U);
uint32_t mSub0 = gSplit * groupRows;
uint32_t mSub1 = (groupCount - gSplit) * groupRows;

copyL0CToDstSub0(ubSTileSub0, l0CTileSub0, false);  // L0C → UB (AIV0)
if (mSub1 > 0U) {
    copyL0CToDstSub1(ubSTileSub1, l0CTileSub1, true); // L0C → UB (AIV1)
}
SetFlag<FIX_M>(l0CEventId);

// 循环结束后:
SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);          // MODE_4
```

### 3.5 完全相同的部分

- Q gather 逻辑 (`COPY_GRANULARITY=2`, `ndNum=2` 批量搬运)
- K resident 加载 (`LoadKResident`)
- MMAD 三重循环 (nL0Itr × mL0Itr × kL0Itr)
- L0C 乒乓 (`bi % L0_STAGES`)
- L0A/L0B flag 编号
- 首末 batch gate/drain 逻辑
- 末尾 MTE1_MTE2 释放

---

## 4. PV Block 差异 (`block_mmad_pv_prefill_a2.hpp` vs `block_mmad_pv_prefill_arch35.hpp`)

| 项 | A2 | A5 |
|----|----|----|
| `L0_TILE_M` | **64** | **128** |
| P 数据来源 | **GM→L1** via Nd2Nz (MTE2), 从 `gmPWorkspace_` 读取 | **L1 直接读取** (VEC 通过 UB→L1 DataCopy 写入) |
| P GM→L1 加载 | `copyGmToL1P(l1PTile, gmPTile)` — A2 独有 | 不存在 |
| EVENT_ID3 | 用于 P 的 MTE1_MTE2/MTE2_MTE1 同步 | 不使用 |
| `SetGmPWorkspace` | 有 | 无 |
| 跨核同步 | MODE_2 (同 QK) | MODE_4 (同 QK) |
| L0C→GM fixpipe | 与 A5 相同 (NZ2ND scatter 到 GM) | 相同 |

---

## 5. Epilogue 差异 (`block_epilogue_online_softmax_prefill_a2.hpp` vs `_arch35.hpp`)

### 5.1 S 加载与数据类型

| 项 | A2 | A5 |
|----|----|----|
| S 加载 | `DataCopyPad` **GM→UB** (MTE3), 从 `gmSWorkspace_` 读取 | S 已在 UB (CUBE 直接 fixpipe L0C→UB) |
| S 数据类型 | **bf16** → 需 `CastSToFloat` + `ScaleS` 两步 | **float** → 直接 `Muls(scale)` 一步 |

### 5.2 P 输出路径

| 项 | A2 | A5 |
|----|----|----|
| P 输出 | `DownCastP` (float→bf16 RowMajor) → `CopyPUbToGm` (**UB→GM** DataCopyPad) | `CastTransdataDeconflict` (float→bf16 **zN**) → `CopyUbZNToL1` (**UB→L1** DataCopy) |
| P 格式 | RowMajor in GM (CUBE 用 Nd2Nz 读回 L1) | zN in L1 (CUBE 直接读) |

### 5.3 Causal Mask 处理

| 项 | A2 | A5 |
|----|----|----|
| 方式 | `ZeroOutCausalTail` — Exp 后显式置零 [tailN, nRound) | 无显式置零 — RowMax/RowSum 只 reduce [0, validSize) |
| Reduction 范围 | RowMax: numElems=tailN, Sub/Exp: numElems=**nRound**, RowSum: numElems=**nRound** | RowMax: numElems=validSize, Sub/Exp: numElems=**validSize**, RowSum: numElems=**validSize** |

### 5.4 跨核同步

| 项 | A2 | A5 |
|----|----|----|
| CUBE→VEC | `SetCrossCoreSync<0x2, PIPE_FIX>` | `SetCrossCoreSync<4, PIPE_FIX>` |
| VEC→CUBE | `SetCrossCoreSync<0x2, PIPE_MTE3>` | `SetVecCrossCoreSync<4, PIPE_MTE3>` |
| Pipe 同步 | V_MTE3(ubSBufId+3) → DataCopyPad → MTE3_V(ubSBufId+5) → MTE3_V(ubSBufId+2) | MTE3_V(ubSBufId+2) (仅 P copy 后) |

### 5.5 operator() 签名

| 项 | A2 | A5 |
|----|----|----|
| 额外参数 | `bi` (GM workspace stage 偏移) | 无 |

### 5.6 Online Softmax 状态管理

| 项 | A2 | A5 |
|----|----|----|
| 全局 rowMax | `gmUbTensor[ubSBufId]` — **无 max(lm, gm) 更新** | 有 `hm = vmax(lm, gm)` |
| 全局 rowSum | `glUbTensor[ubSBufId]` — **无 dm*gl+ll 更新** | 有 `UpdateGlobalRowSum` |
| 跨 batch 累积 | **无** — 由 Phase2 `EpilogueRescaleO` 处理 | 有 — epilogue 内部 online 累积 |

**核心架构差异**: A2 prefill epilogue **不做 online softmax 跨 batch 累积**。每个 batch 独立计算 softmax + P + 局部 stats，然后由 `ScatterBatchStats` 写到 GM，最终由 `Phase2CombineScale` 合并。

---

## 6. 三个 A2 Epilogue 实现对比

### 6.1 涉及文件

| 实现 | 文件 | 来源 |
|------|------|------|
| **A2 prefill** | `block_epilogue_online_softmax_prefill_a2.hpp` | sparse_attention_score_prefill |
| **catlass A2 FA** | `catlass/.../block_epilogue_fa_softmax.hpp` | catlass 参考库 |
| **BSA A2** | `bsa_block_epilogue_online_softmax.hpp` | block_sparse_attention |

### 6.2 计算精度路径

| 项 | A2 prefill | catlass A2 FA | BSA A2 |
|----|------------|---------------|--------|
| S GM 类型 | **bf16** | **half** (fp16) | **float** |
| Scale | **float** (Cast后Muls) | **half** (直接Muls) | **float** (直接Muls) |
| RowMax | **float** | **half** | **float** |
| Exp | **float** | **float** (Cast half→float, Exp, Cast回half) | **float** |
| RowSum | **float** | **float** (Cast后ReduceSum) | **float** |
| P 输出 | Cast float→**bf16** | Cast float→**half** | Cast float→**bf16** |

### 6.3 P 写出方式

| 项 | A2 prefill | catlass A2 FA | BSA A2 |
|----|------------|---------------|--------|
| P 写出 | `DataCopyPad` (UB→GM, **3参数**) | `CopyUb2Gm` (Tile copy) | `DataCopy` (DataCopyParams) |
| P GM 格式 | **RowMajor** bf16 | RowMajor half | RowMajor bf16 |
| CUBE 读回 | Nd2Nz (GM→L1) | 不需要 | 不需要 |

### 6.4 Causal Mask 处理

| 项 | A2 prefill | catlass A2 FA | BSA A2 |
|----|------------|---------------|--------|
| 方式 | **Exp 后置零** (`ZeroOutCausalTail`) | **RowMax 前加 mask** (Add -inf) | **RowMax 前加 mask** (`ApplyMask`, Muls -3e38 + Add) |
| 时机 | Exp 之后, RowSum 之前 | Scale 之后, RowMax 之前 | Scale 之后, RowMax 之前 |
| 对 RowMax 影响 | RowMax 只 reduce `[0, tailN)` — 无影响 | RowMax 排除无效列 (-inf) | RowMax 排除无效列 (-3e38) |

### 6.5 Online Softmax 累积

| 项 | A2 prefill | catlass A2 FA | BSA A2 |
|----|------------|---------------|--------|
| 跨 batch 累积 | **无** — 由 Phase2 处理 | 有 — epilogue 内部 online 累积 | 有 — epilogue 内部 online 累积 |
| 全局 rowMax 更新 | 无 | `hm = vmax(lm, gm)` | `UpdateGlobalRowMax` |
| 全局 rowSum 更新 | 无 | `dm = exp(gm-hm)`, rescale | `UpdateGlobalRowSum` |

### 6.6 UB 布局对比

| Buffer | A2 prefill (192KB) | catlass A2 FA | BSA A2 |
|--------|---------------------|---------------|--------|
| ls (S input) | bf16, 2 stages × 16KB | half, 1 stage × 32KB | float, 2 stages × 32KB |
| lp (P output) | bf16, 2 stages × 16KB | half, 1 stage × 32KB | bf16, 1 stage × 32KB |
| fs/ls32 (float S) | float, 1 stage × 32KB | float (ls32), 1 stage × 32KB | 无 (ls 本身就是 float) |
| tv (temp) | float, ~16KB | float, ~8KB | float, ~8KB |
| gm/gl (stats) | float, 2 stages × 256B each | half (gm) + float (ll), 各 ~256B | float, 各 ~256B |
| mask | 无 (用 ZeroOutCausalTail) | half, 1 stage × 32KB | half + float, 各 32KB |

### 6.7 子核分割

| 项 | A2 prefill | catlass A2 FA | BSA A2 |
|----|------------|---------------|--------|
| 分割方式 | 按 **group** (`CeilDiv(groupCount,2) * groupRows`) | 按 **M** (`mActualPerSubBlock`) | 按 **qSBlockSize** 或 **qNBlockSize** |
| 空行处理 | m==0 时仍做 sync handshake | 无 (subM>0 才执行) | m==0 时 set flag 并 return |

### 6.8 RowMax/RowSum 实现

| 项 | A2 prefill | catlass A2 FA | BSA A2 |
|----|------------|---------------|--------|
| RowMax | 通用 `CalcRowMax` (TAILTILE pattern, 任意N) | `BlockReduceMax` + `qkN<=128` 快速路径 | 按 tile 大小分特化 (1024/512/256/tail) |
| RowSum | 通用 `CalcRowSum` (TAILTILE pattern) | `RepeatReduceSum` + `qkN<=128` 快速路径 | 按 tile 大小分特化 (1024/512/256/tail) |
| 多行处理 | 一次处理所有行 (numRows = m) | 一次处理所有行 (subM) | 分行循环 (rowNumTile, preLoad=1) |

---

## 7. Kernel 层差异 (`_kernel_a2.h` vs `_kernel_arch35.h`)

| 项 | A2 | A5 |
|----|----|----|
| GM S/P workspace | 新增 `gmSWorkspace_`, `gmPWorkspace_` 及 size/stageElems | 无 |
| workspace 布局 | `accumOut\|softmaxMax\|softmaxSum\|[gmS][gmP]` | `accumOut\|softmaxMax\|softmaxSum` |
| InitSyncFlags | `CrossCoreSetFlag<0x2, PIPE_MTE1>(2,3,4)` — MODE_2, 3 flag | `CrossCoreSetFlag<MODE_4, PIPE_MTE1>(2,18,3,19,4,20)` — MODE_4, 6 flag |
| InitWorkspaceStats 顺序 | 在 InitSyncFlags **之前** | 在 InitSyncFlags **之后** |
| QK Init 调用 | `blockMmadQK.Init(resource, blockSize_, embed_, gmSWorkspace_)` | `blockMmadQK.Init(resource, blockSize_, embed_)` |
| PV workspace | `blockMmadPV.SetGmPWorkspace(gmPWorkspace_, ...)` | 无 |
| Epilogue workspace | `epilogueSoftmax.SetGmSWorkspace(...)` + `SetGmPWorkspace(...)` | 无 |
| Epilogue 调用 | 多传 `bi` 参数 | 无 `bi` 参数 |
| ReleaseSyncFlags | `CrossCoreWaitFlag(flag)` — 非模板化 | `CrossCoreWaitFlag<MODE_4, PIPE_FIX>(0,1,16,17)` — 模板化 |

---

## 8. 硬件约束总结

### 8.1 A2 (AtlasA2 / Ascend 910B) 限制

| 约束 | 影响 | A2 解决方案 |
|------|------|-------------|
| CUBE 无 L0C→UB Fixpipe | S 无法直接从 CUBE 到 VEC UB | L0C→GM(Fixpipe) → UB(DataCopyPad) |
| 不支持 VEC UB→L1 DataCopy | P 无法从 VEC UB 到 CUBE L1 | UB→GM(DataCopyPad) → L1(Nd2Nz) |
| MIX_AIC_1_2: CUBE 无 MTE3 | CUBE 不能做 DataCopy GM→UB | VEC 负责 GM↔UB 搬运 |
| MODE_4 不稳定 | 跨核同步可能死锁 | 使用 MODE_2 (已验证) |

### 8.2 A5 (AtlasA5 / Ascend 950) 特性

| 特性 | 优势 |
|------|------|
| L0C→UB 直接 Fixpipe | S 无需 GM 中转 |
| VEC UB→L1 DataCopy | P 无需 GM 中转 |
| MODE_4 双 AIV 同步 | 一次通知 2 个 AIV |
| RegBase MicroAPI | 更高效的 softmax 计算 |
| LoadData2DParamsV2 | 更灵活的 L0A/L0B 加载 |
