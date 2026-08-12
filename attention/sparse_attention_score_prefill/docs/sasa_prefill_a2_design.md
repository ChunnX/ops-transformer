# sparse_attention_score_prefill 算子 A2 (AtlasA2) 实现方案

## 1. 背景与目标

基于已迁移的 TLA (Tensor Layout Abstraction) 基础设施和 A2 tile copy 业务逻辑，实现 sparse_attention_score_prefill 算子的 AtlasA2 版本完整实现。

**核心约束**: 只对 A2 进行处理，不影响 A5 逻辑。

**A2 vs A5 架构差异**:

| 特性 | AtlasA2 (`__CCE_AICORE__==220`) | AtlasA5 (`__CCE_AICORE__==310`) |
|------|------|------|
| L0C 容量 | 128KB | 256KB |
| UB 容量 | 192KB | 256KB |
| L0C→UB 路径 | **无直接 fixpipe** (需 L0C→GM→UB) | 有直接 fixpipe (`CopyL0CToUBTla`) |
| Regbase MicroAPI (`__simd_vf__`) | **不支持** | 支持 |
| `CastTransdataDeconflict` | **不支持** | 支持 |
| `ComputeScaleValue_VF` / `ReduceFinalRes_VF` | **不支持** | 支持 |
| Fixpipe API | `FixpipeParamsV220` | `SetFixpipeNz2ndFlag` / `DataCopyCO12DstParams` |
| L0A Layout | `zZ` | `zN` |
| FIX→MTE3 同步 | 无 `HardEvent::FIX_MTE3`，用 `PipeBarrier<PIPE_FIX>()` | 有 `HardEvent::FIX_MTE3` |

## 2. 已完成工作

### 2.1 TLA 基础设施迁移 (Phase 1)
- 将 `d:\code3\catlass` 的 TLA 模板库迁移至 `op_kernel/tla/`
- 包含 `tensor.hpp`, `layout.hpp`, `coord.hpp` 等核心文件

### 2.2 A2 Tile Copy 迁移 (Phase 2)
- `copy_l0c_to_gm_a2.hpp` — A2 L0C→GM fixpipe tile copy (`CopyL0CToGmTla<AtlasA2>`)
- `copy_gm_to_l1_a2.hpp` — A2 GM→L1 tile copy
- `copy_l1_to_l0a_a2.hpp` — A2 L1→L0A tile copy
- `copy_l1_to_l0b_a2.hpp` — A2 L1→L0B tile copy
- `tile_copy.hpp` 中添加 `PackedTileCopyTla<AtlasA2>` 和 `TileCopyTlaToUB` 聚合器

### 2.3 A2 Prefill GEMM Block (Phase 3 — 已完成)

#### `block_mmad_qk_prefill_a2.hpp` (415 行，已完成)
- **类**: `BlockMmadQKPrefillA2`，DispatchPolicy = `MmadAtlasA2PrefillQK`
- **核心设计**: A2 无 L0C→UB 直接路径，采用 L0C→GM(workspace)→UB(DataCopyPad) 三步流
- **GM S Workspace**: 双缓冲，按 `bi%2` 切换，每 stage 容量 = `L0_TILE_M(128) × blockSize(128)` elements = 32KB (bf16)，2 stages = 64KB
- **Fixpipe 流**: `copyL0CToGm(gmSTile, l0CTile)` → `PipeBarrier<PIPE_FIX>()` → `DataCopyPad(ubSTile, gmSTile, params)`
- **跨核同步**: CUBE 完成后在 `PIPE_MTE3` 上 SetFlag (A5 用 `PIPE_FIX`)，因为 A2 最后一个 CUBE 操作是 MTE3 DataCopy GM→UB
- **Sub 核切分**: 按组切分 (`gSplit = CeilDiv(groupCount, 2)`)，sub0 得 `mSub0 = gSplit * groupRows` 行，sub1 得剩余
- **Init 签名**: `Init(resource, blockSize, D, gmSWorkspace)` — 比 A5 多一个 GM workspace 参数

#### `block_mmad_pv_prefill_a2.hpp` (283 行，已完成)
- **类**: `BlockMmadPVPrefillA2`，DispatchPolicy = `MmadAtlasA2PrefillPV`
- A2 PV 与 A5 PV 逻辑高度一致 (两者都用 `CopyL0CToGmTla` fixpipe 到 GM)
- `LayoutTagP = layout::zN`，`LayoutTagL1A` 解析为 `zN`

### 2.4 Dispatch Policy (已完成)
- `gemm/dispatch_policy.hpp`: 添加 `MmadAtlasA2PrefillQK` / `MmadAtlasA2PrefillPV` (L0_STAGES=2)
- `epilogue/dispatch_policy.hpp`: 已声明 `EpilogueOnlineSoftmaxPrefillA2` / `EpilogueRescaleOPrefillA2` (ArchTag=AtlasA2)

## 3. 待完成工作

### 3.1 A2 Softmax Epilogue (任务 b4)

**文件**: `attn_infra/epilogue/block/block_epilogue_online_softmax_prefill_a2.hpp`

**状态**: 文件已存在但内容损坏，需完全重写。

#### 3.1.1 模板特化

```cpp
template <class OutputType_, class LayoutS_>
class BlockEpilogue<
    EpilogueOnlineSoftmaxPrefillA2,
    OutputType_,
    Gemm::GemmType<bfloat16_t, LayoutS_>>
```

与 A5 BSA softmax (`EpilogueOnlineSoftmaxBsa`) 的模板签名一致，但 DispatchPolicy 改为 `EpilogueOnlineSoftmaxPrefillA2` (ArchTag=AtlasA2)。

#### 3.1.2 关键常量

```cpp
static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;   // 与 BSA/内核 SM_UB_BLOCK 一致
static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;      // bf16 元素数 (32768B / 2B)
static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
static constexpr uint32_t ELE_NUM_PER_C0 = 16;
static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
static constexpr uint32_t SM_UB_STAGES = 2;
```

#### 3.1.3 UB 布局 (与内核 SM_UB_GM_OFFSET / SM_UB_GL_OFFSET 对齐)

```
Offset 0         : lsUbTensor (ElementInput=bf16, S 输入)      [2 * 32768B = 65536B]
Offset 65536     : lpUbTensor (ElementOutput=bf16, P 输出)     [2 * 32768B = 65536B]
Offset 229376    : gmUbTensor[0..1] (float, rowMax)            [2 * 256B = 512B]  = 7 * 32768
Offset 229888    : glUbTensor[0..1] (float, rowSum)            [2 * 256B = 512B]  = 229376 + 512
```

与内核定义完全一致:
```cpp
SM_UB_BLOCK = 32768U;
SM_UB_GM_OFFSET = 7U * SM_UB_BLOCK;          // 229376
SM_UB_GL_OFFSET = SM_UB_GM_OFFSET + 2U * 64U * sizeof(float);  // 229888
SM_UB_STAGE_BYTES = 64U * sizeof(float);      // 256
```

#### 3.1.4 构造函数

```cpp
BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
```

与 BSA 构造函数签名完全一致。初始化:
- `lsUbTensor` / `lpUbTensor` (从 resource.ubBuf 按字节偏移获取)
- `gmUbTensor[2]` / `glUbTensor[2]` (2-deep by ubSBufId)
- `scaleValue` (bf16), `MIN_VALUE` (bf16, -3.39e+38)

#### 3.1.5 operator() 签名 (与 BSA 一致)

```cpp
template <class TensorP>
void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape,
    uint32_t ubSBufId, uint32_t l1PBufId,
    Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag,
    const uint32_t *causalValidLens, uint32_t groupCount, uint32_t groupRows)
```

#### 3.1.6 operator() 执行流程

```
1. mCopyOffset = CeilDiv(groupCount, 2) * groupRows  (按组切分)
2. m = subBlockIdx_==0 ? mHalf : (M - mHalf)
3. 若 m==0: 仅做 sync handshake 后 return
4. WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag)     — 等 QK 完成
5. WaitFlag<MTE3_V>(ubSBufId + 2)                — 等 S 在 UB 就绪

6. 逐组循环 (gStart..gEnd):
   a. Cast bf16 S → float (到临时 UB 区域，用 lsUbTensor 做 in-place float 化)
   b. Muls float S *= scaleValue
   c. CausalMask: [tailN, nAligned) 填 MIN_VALUE_FP32
   d. CalcRowMax: BlockReduceMax 两级归约 (→ tvUb → rowmaxUb)
   e. SubtractRowMaxAndExp: Brcb rowmax → Sub → Exp
   f. CalcRowSum: BlockReduceSum 两级归约
   g. DownCastP: Cast float→bf16 (写入 lpUbTensor，zN 布局)

7. SetFlag<V_MTE3>(ubSBufId) → WaitFlag<V_MTE3>(ubSBufId)
8. SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag)
9. CopyPUbToPL1(l1PTile, ubPTile, m)             — UB P → L1 P (供 PV 使用)
10. SetFlag<MTE3_V>(ubSBufId + 2)
11. SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag)
12. PipeBarrier<PIPE_V>()
```

#### 3.1.7 A5 API 替换映射

| BSA (A5 regbase) | A2 标准向量 API |
|------|------|
| `ComputeScaleAndMax` (LoadAlign + Mul + Select + Cast + ReduceMax) | `Cast<bf16→float>` + `Muls` + `CausalMask`(Duplicate+Max) + `BlockReduceMax` |
| `ComputeExpSubSum16` (FusedExpSub + Add + Cast + Select + StoreAlign + ReduceSum) | `Brcb` + `Sub` + `Exp` + `BlockReduceSum` + `Cast<float→bf16>` |
| `CastTransdataDeconflict` (A5 zN transdata) | `Cast` + `DataCopy` (A2 无 transdata 硬件) |
| `__simd_vf__` / MicroAPI | 标准 `AscendC::` 向量 API |

#### 3.1.8 参考文件

- **主参考** (BSA operator() 签名/流程): `block_epilogue_online_softmax_arch35_reg_low_prec_bf16.hpp`
- **标准向量 API 模式** (CalcRowMaxTail/CalcRowSumTail/SubtractRowMaxAndExp): `block_epilogue_online_softmax_prefill_arch35.hpp`
- **A2 Cast 模式** (DownCastP): `block_epilogue_online_softmax.hpp` (A2 infer softmax)

### 3.2 A2 Rescale-O Epilogue (任务 b5)

**文件**: `attn_infra/epilogue/block/block_epilogue_rescale_o_prefill_a2.hpp`

**需新建**。镜像 A5 的 `block_epilogue_rescale_o_prefill_arch35.hpp`，替换 A5 专属 API。

#### 3.2.1 模板特化

```cpp
template <class InDtype, class Tws>
class BlockEpilogue<EpilogueRescaleOPrefillA2, InDtype, Tws>
```

#### 3.2.2 接口 (与 A5 完全一致)

```cpp
void InitFDBuffers(AscendC::LocalTensor<T> &ubBase, uint32_t D, uint32_t groupSize, uint32_t topK, uint32_t kvHeads)
void FlashDecodeCompute(uint32_t tmpBlockIdx, uint32_t totalTaskNumP2,
    AscendC::GlobalTensor<ElementO> &gmO,
    AscendC::GlobalTensor<TwsO> &gmAccumOut,
    AscendC::GlobalTensor<T> &gmSoftmaxMax,
    AscendC::GlobalTensor<T> &gmSoftmaxSum,
    uint32_t numHeads, uint32_t kvHeads)
```

#### 3.2.3 A5 API 替换映射

| A5 (regbase VF) | A2 标准 API |
|------|------|
| `ComputeScaleValue_VF` (Broadcast + Max + Sub + Exp + Mul + Reciprocal + Add) | `Brcb` + `Sub` + `Exp` + `Mul` + `BlockReduceSum` + `Reciprocal` + `Mul` |
| `ReduceFinalRes_VF<T>` / `ReduceFinalRes_VF_BF16` (fused reduce + Div) | `Cast` + `Mul` (scale * O_norm) + `Add` (累加) + `Cast`(输出类型) |

#### 3.2.4 可直接复用的 A5 逻辑 (标准 API)

- `CopyLseIn`: `DataCopyPad` + `Broadcast` + `Reciprocal` — 全部标准 API
- `CopyAccumOutIn`: `DataCopyPad` — 标准 API
- `CopyFinalResOut`: `Cast` + `DataCopyPad` — 标准 API

### 3.3 Dispatcher 更新 (任务 b6)

#### `block_mmad.hpp` — 添加 A2 prefill includes

```cpp
#if (__CCE_AICORE__ == 220)
#include "block_mmad_qk.hpp"           // (已有) A2 infer
#include "block_mmad_pv.hpp"           // (已有) A2 infer
#include "block_mmad_qk_prefill_a2.hpp"  // (新增) A2 prefill
#include "block_mmad_pv_prefill_a2.hpp"  // (新增) A2 prefill
#endif
```

#### `block_epilogue.hpp` — 添加 A2 prefill includes

```cpp
#if (__CCE_AICORE__ == 220)
#include "block_epilogue_online_softmax.hpp"           // (已有) A2 infer
#include "block_epilogue_online_softmax_low_prec.hpp"  // (已有) A2 infer
#include "block_epilogue_rescale_o.hpp"                // (已有) A2 infer
#include "block_epilogue_rescale_o_low_prec.hpp"       // (已有) A2 infer
#include "block_epilogue_online_softmax_prefill_a2.hpp"  // (新增) A2 prefill
#include "block_epilogue_rescale_o_prefill_a2.hpp"       // (新增) A2 prefill
#endif
```

### 3.4 Kernel 修改 (任务 b7)

**文件**: `arch35/sparse_attention_score_prefill_kernel_arch35.h`

#### 3.4.1 GM S Workspace 分配

A2 QK block 需要一个 GM workspace 用于 L0C→GM→UB 中转。需在 kernel 的 GM 分配区域添加:

```cpp
// A2 only: GM S workspace for QK fixpipe L0C→GM→UB
#if (__CCE_AICORE__ == 220)
    // 双缓冲: 2 * L0_TILE_M(128) * blockSize(128) * sizeof(bf16) = 64KB
    AscendC::GlobalTensor<ElementS> gmSWorkspace_;
    constexpr uint32_t GM_S_WORKSPACE_BYTES = 2U * 128U * 128U * sizeof(bfloat16_t);
#endif
```

在 kernel Init 中分配 (从 userWorkspace 尾部或独立区域):
```cpp
#if (__CCE_AICORE__ == 220)
    gmSWorkspace_ = gmSWorkspace;  // 由 Tiling 传入
#endif
```

#### 3.4.2 QK Init 调用扩展

```cpp
// A5 (现有):
blockMmadQK.Init(resource, blockSize_, embed_);

// A2 (改为):
#if (__CCE_AICORE__ == 220)
    blockMmadQK.Init(resource, blockSize_, embed_, gmSWorkspace_);
#elif (__CCE_AICORE__ == 310)
    blockMmadQK.Init(resource, blockSize_, embed_);
#endif
```

#### 3.4.3 ReleaseSyncFlags 修改

A5 CUBE 侧 drain `PIPE_FIX` 跨核 flag (因为 A5 fixpipe 在 FIX pipe 上):
```cpp
// A5 现有:
CrossCoreWaitFlag<MODE_4, PIPE_FIX>(0);
CrossCoreWaitFlag<MODE_4, PIPE_FIX>(1);
CrossCoreWaitFlag<MODE_4, PIPE_FIX>(16);
CrossCoreWaitFlag<MODE_4, PIPE_FIX>(17);
```

A2 CUBE 侧最后操作在 MTE3 pipe (DataCopy GM→UB)，需 drain `PIPE_MTE3`:
```cpp
// A2 改为:
#if (__CCE_AICORE__ == 220)
    CrossCoreWaitFlag<MODE_4, PIPE_MTE3>(0);
    CrossCoreWaitFlag<MODE_4, PIPE_MTE3>(1);
    CrossCoreWaitFlag<MODE_4, PIPE_MTE3>(16);
    CrossCoreWaitFlag<MODE_4, PIPE_MTE3>(17);
#elif (__CCE_AICORE__ == 310)
    CrossCoreWaitFlag<MODE_4, PIPE_FIX>(0);
    CrossCoreWaitFlag<MODE_4, PIPE_FIX>(1);
    CrossCoreWaitFlag<MODE_4, PIPE_FIX>(16);
    CrossCoreWaitFlag<MODE_4, PIPE_FIX>(17);
#endif
```

#### 3.4.4 InitSyncFlags 补充

A2 CUBE 侧需额外 prime `PIPE_MTE3` 跨核 flag (QK block 的 `WaitCrossCoreSync<4, PIPE_MTE3>` 在第一个 batch 需要):

```cpp
#if (__CCE_AICORE__ == 220)
    // A2: QK block waits on PIPE_MTE3 for VEC to finish reading S from UB
    CrossCoreSetFlag<MODE_4, PIPE_MTE3>(0);
    CrossCoreSetFlag<MODE_4, PIPE_MTE3>(1);
    CrossCoreSetFlag<MODE_4, PIPE_MTE3>(16);
    CrossCoreSetFlag<MODE_4, PIPE_MTE3>(17);
#endif
```

#### 3.4.5 Tiling 参数传递

需在 Tiling 结构中添加 GM S workspace 指针/大小，通过 kernel interface 传入:

```cpp
// kernel_interface.cpp A2 分支:
// 在 userWorkspace 布局中预留 GM S workspace 区域
// 或使用独立的 GM 分配
```

## 4. 数据流总览

```
Phase1 (三级流水: QK(bi) || softmax(bi-2) || PV(bi-4)):

CUBE:
  GM Q ──gather──> L1 Q ──> L0A ──┐
  GM K ──> L1 K ──> L0B ──┤
                          ↓
                        MMA (L0C)
                          ↓
                    fixpipe L0C→GM(ws)     ← A2 独有中间步
                          ↓
                    PipeBarrier<PIPE_FIX>
                          ↓
                    DataCopyPad GM→UB S
                          ↓
                    CrossCoreSetFlag<PIPE_MTE3>  → VEC

VEC:
  WaitCrossCoreSync<PIPE_V>  ← 等 CUBE QK 完成
  WaitFlag<MTE3_V>           ← 等 S 在 UB 就绪
      ↓
  Cast bf16→float, Muls scale, CausalMask
  BlockReduceMax → rowMax (gmUb)
  Brcb + Sub + Exp
  BlockReduceSum → rowSum (glUb)
  Cast float→bf16 → P (lpUb)
      ↓
  CopyPUbToPL1 (UB P → L1 P)
  CrossCoreSetFlag<PIPE_MTE3>  → CUBE PV

CUBE PV:
  WaitCrossCoreSync<PIPE_MTE3>  ← 等 VEC softmax 完成
  L1 P ──> L0A ──┐
  L1 V ──> L0B ──┤
                 ↓
               MMA (L0C)
                 ↓
           fixpipe L0C→GM (accumOut workspace)

Phase2 (仅 VEC):
  GM softmaxMax/Sum ──> UB
  Broadcast + Exp + Reciprocal → scale[k]
  GM accumOut ──> UB O_norm
  scale[k] * O_norm[k] 累加 → O
  Cast → GM attentionOut
```

## 5. UB 内存布局 (A2 Prefill)

```
UB (192KB = 196608B):

[0, 65536)        lsUbTensor (bf16 S, 2 stages × 32768B)        ← QK fixpipe 写入
[65536, 131072)   lpUbTensor (bf16 P, 2 stages × 32768B)        ← softmax 输出
[131072, 229376)  tvUb / tvScratch / 临时计算区域                  ← BlockReduceMax/Sum 临时
[229376, 229888)  gmUbTensor[0..1] (float rowMax, 2 × 256B)     ← softmax rowMax
[229888, 230400)  glUbTensor[0..1] (float rowSum, 2 × 256B)     ← softmax rowSum
```

注: A2 UB 192KB，当前布局使用到 230400B ≈ 225KB，超出了 192KB 限制。需确认实际 UB 可用空间，或缩减 tvUb 区域。实际 A2 UB 为 192KB = 196608B，而 `7 * 32768 = 229376 > 196608`。

**关键问题**: A2 UB 只有 192KB，而 BSA softmax 布局需要 7×32768=229376B 起 + 1024B stats = 230400B，超出 A2 UB 限制。

**解决方案**: A2 softmax epilogue 使用更紧凑的 UB 布局:
- `lsUbTensor`: offset 0, 2 stages × MAX_UB_S_ELEM_NUM(8192) × 2B = 32768B (而非 65536B)
- `lpUbTensor`: offset 32768, 同上 32768B
- `gmUbTensor`: offset 65536, 2 × 256B = 512B
- `glUbTensor`: offset 66048, 2 × 256B = 512B
- `tvUb`: offset 66560, 剩余 ~125KB 用于临时计算

对应内核常量也需调整:
```cpp
// A2:
static constexpr uint32_t SM_UB_BLOCK = 16384U;  // A2: 16KB (A5: 32KB)
static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192U;  // A2: 8192 (A5: 16384)
```

但这会影响 QK block 写入 UB S 的布局 (QK 写 lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM])。需确保 QK block 和 softmax epilogue 使用一致的 UB 布局常量。

**最终方案**: 使用 A2 infer softmax 的常量 (`UB_UINT8_BLOCK_SIZE=16384`, `MAX_UB_S_ELEM_NUM=8192`)，同时调整内核 `SM_UB_BLOCK` / `SM_UB_GM_OFFSET` / `SM_UB_GL_OFFSET` 为 A2 条件编译值:

```cpp
#if (__CCE_AICORE__ == 220)
    static constexpr uint32_t SM_UB_BLOCK = 16384U;
    static constexpr uint32_t SM_UB_GM_OFFSET = 4U * SM_UB_BLOCK;  // 65536
    static constexpr uint32_t SM_UB_GL_OFFSET = SM_UB_GM_OFFSET + 2U * 64U * sizeof(float);  // 66048
#elif (__CCE_AICORE__ == 310)
    static constexpr uint32_t SM_UB_BLOCK = 32768U;
    static constexpr uint32_t SM_UB_GM_OFFSET = 7U * SM_UB_BLOCK;  // 229376
    static constexpr uint32_t SM_UB_GL_OFFSET = SM_UB_GM_OFFSET + 2U * 64U * sizeof(float);  // 229888
#endif
```

## 6. GM Workspace 布局

```
userWorkspace:
  [0, accumOutSize * sizeof(ElementWorkspaceO))    — accumOut (O_partial)
  [accumOutSize * sizeof, +lseStatSize * sizeof(float))  — softmaxMax
  [+lseStatSize, +lseStatSize * sizeof(float))     — softmaxSum
  [尾部, +64KB)                                      — A2 GM S workspace (双缓冲)
```

A2 GM S workspace: `2 × 128 × 128 × sizeof(bf16) = 64KB`，从 userWorkspace 尾部分配。

## 7. 实现顺序

| 步骤 | 文件 | 状态 | 依赖 |
|------|------|------|------|
| 1 | `block_epilogue_online_softmax_prefill_a2.hpp` | 待重写 | QK block (已完成) |
| 2 | `block_epilogue_rescale_o_prefill_a2.hpp` | 待新建 | 无 |
| 3 | `block_mmad.hpp` + `block_epilogue.hpp` | 待修改 | 步骤 1, 2 |
| 4 | `sparse_attention_score_prefill_kernel_arch35.h` | 待修改 | 步骤 1, 2, 3 |
| 5 | `sparse_attention_score_prefill_kernel_interface.cpp` | 待检查 | 步骤 4 |
| 6 | Tiling 参数 (GM S workspace) | 待添加 | 步骤 5 |

## 8. 风险与注意事项

1. **UB 容量**: A2 UB 仅 192KB，必须使用紧凑布局 (16KB/stage 而非 A5 的 32KB/stage)
2. **S 元素上限**: `MAX_UB_S_ELEM_NUM=8192` (bf16)，即 `128×128=16384 > 8192`。需确认 QK 输出的 S 矩阵实际尺寸。若 batchM×validSize 超过 8192，需分批处理或使用更大 UB 区域
3. **FIX→MTE3 同步**: A2 无 `HardEvent::FIX_MTE3`，用 `PipeBarrier<PIPE_FIX>()` 确保 fixpipe 写入 GM 后 DataCopy 才能读取
4. **跨核同步 pipe 变更**: A2 CUBE→VEC 同步在 `PIPE_MTE3` (非 A5 的 `PIPE_FIX`)，内核 InitSyncFlags/ReleaseSyncFlags 需条件编译
5. **BlockReduceMax/Sum mask**: 标准 reduce API 的 mask 设置需精确匹配行/列对齐，参考 A5 prefill softmax 的 `SetVecMask` / `SetBlockReduceMask`
6. **无 CastTransdataDeconflict**: A2 P 输出到 L1 需用 Cast(float→bf16) + DataCopy(UB→L1 zN) 两步替代 A5 的 CastTransdataDeconflict 一步
