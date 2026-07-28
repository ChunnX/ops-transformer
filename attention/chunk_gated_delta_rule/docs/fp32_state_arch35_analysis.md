# ChunkGatedDeltaRule arch35 FP32 State 路径全面分析

## 1. 分析范围

本文档针对 `chunk_gated_delta_rule` 算子 arch35（Ascend950）在 state 输入为 FP32 时的完整代码路径进行全面检视，覆盖以下维度：

- Workspace 分配与使用
- UB 划分与复用
- 三个 Stage 的参数传递
- AIC/AIV 跨核同步
- 核内流水线同步
- 类型转换（FP32 <-> BF16）

涉及文件：
- `op_kernel/arch35/chunk_gated_delta_rule.h` — 主 kernel
- `op_kernel/arch35/chunk_gated_delta_rule_stage1.h` — Stage1
- `op_kernel/arch35/chunk_gated_delta_rule_stage2.h` — Stage2（FP32 路径核心）
- `op_kernel/arch35/chunk_gated_delta_rule_stage3.h` — Stage3
- `op_kernel/arch35/chunk_gated_delta_rule_utils.h` — 常量定义
- `op_host/chunk_gated_delta_rule_tiling.cpp` — Tiling 与 workspace 计算

---

## 2. Workspace 分配与使用

### 2.1 Tiling 侧 workspace 计算（tiling.cpp:114-155）

FP32 路径分配的 interWorkspaceSz 各段：

| 序号 | 用途 | 大小（字节） | 数据类型 |
|------|------|-------------|---------|
| 1 | gCumExp | 4 * nv * s | FP32 |
| 2 | kCumDecay | 2 * nv * s * dk | BF16 |
| 3 | vInner (FP32) | 4 * nv * s * dv | FP32 |
| 4 | vInnerBf16 | 2 * nv * s * dv | BF16 |
| 5 | qPrime | 2 * nv * s * dk | BF16 |
| 6 | attnInter | 2 * nv * s * dv | BF16 |
| 7 | kg | 2 * nv * s * dk | BF16 |
| 8 | qkt | 2 * nv * s * c | BF16 |
| 9 | stateBf16Wk | 2 * nv * dv * dk | BF16 |
| 10 | mask (stage1+stage3) | 4 * c * c * aiCoreNum * 4 | FP32 |

其中 `s = maxGroupLength = p * aiCoreNum * chunkSize`，`c = chunkSize`。

### 2.2 Kernel 侧 workspace 绑定（chunk_gated_delta_rule.h:131-170）

Kernel 侧通过 `user + offset` 依次绑定各 GM tensor：

| 序号 | Kernel 变量 | 偏移计算 | 对应 Tiling 序号 |
|------|-----------|---------|-----------------|
| 1 | gCum_ | sizeof(float) * nv * s | 1 ✓ |
| 2 | kCumDecay_ | sizeof(bf16) * nv * s * dk | 2 ✓ |
| 3 | vInner_ | sizeof(float) * nv * s * dv | 3 ✓ |
| 4 | vInnerBf16_ | sizeof(bf16) * nv * s * dv | 4 ✓ |
| 5 | qPrime_ | sizeof(bf16) * nv * s * dk | 5 ✓ |
| 6 | attnInter_ | sizeof(bf16) * nv * s * dv | 6 ✓ |
| 7 | kg_ | sizeof(bf16) * nv * s * dk | 7 ✓ |
| 8 | qkt_ | sizeof(bf16) * nv * s * c | 8 ✓ |
| 9 | stateBf16Wk_ | sizeof(bf16) * nv * dv * dk | 9 ✓ |
| 10 | stageOneMask_ | sizeof(float) * c * c * aiCoreNum * TASK_RATIO | 10 的前半 ✓ |
| 11 | stageThreeMask_ | 同上 | 10 的后半 ✓ |

> `MASK_NUM=4`（tiling）与 `TASK_RATIO=2`（kernel），两个 mask 合计 `4*c*c*aiCoreNum*4 = 16*c*c*aiCoreNum` 字节，kernel 使用 `2*4*c*c*aiCoreNum*2 = 16*c*c*aiCoreNum` 字节。**匹配** ✓

### 2.3 Stage workspace（stageWsAddr_）

Stage1 使用 stageWsAddr_ 作为临时空间（gBKWsGm_, queryConGm_, keyConGm_, kkWsGm_, attnWsGm_, vBetaWsGm_），总大小 `sizeLow * c * (2*c + 3*dk + dv) * paraNum * aiCoreNum`。

Stage3 使用 `stageWsAddr_ + coreNum * c * c * sizeof(float)` 作为 tmpGM_。由于 Stage1/Stage3 之间有 `SyncAll<false>()` 隔离，复用安全。✓

### 2.4 结论

**Workspace 分配与使用无 bug。** Tiling 侧计算与 kernel 侧偏移完全匹配，FP32 路径额外分配的 vInnerBf16_ 和 stateBf16Wk_ 均有对应。

---

## 3. UB 划分与复用

### 3.1 Stage2 FP32 路径 UB 分配（stage2.h:88-100）

```
inQueue_  = max(max(inSize, deltaSize), vInnerCastInSize)
outQueue_ = max(max(inSize, deltaSize), vInnerCastOutSize)
tmpBuff_  = (Dv * curDk_ + BLOCK_FLOAT_NUM) * sizeof(float)
  ├── stateFp32Ub_  : Dv * curDk_ 个 float
  └── lastGCum_     : 1 个 float
```

其中：
- `inSize = max(chunkSize, Dv) * curDk * sizeof(float)` — FP32 state 搬入
- `deltaSize = Dv * curDk * sizeof(bfloat16_t)` — BF16 delta
- `vInnerCastInSize = chunkSize * Dv * sizeof(float)` — CastGmToGm 输入
- `vInnerCastOutSize = chunkSize * Dv * sizeof(bfloat16_t)` — CastGmToGm 输出

### 3.2 各操作 UB 使用量验证

| 操作 | 使用的 buffer | 需要大小 | buffer 大小 | 安全 |
|------|-------------|---------|-----------|------|
| CopyIn\<float\> (state) | inQueue_ | Dv * curDk * 4 | ≥ inSize | ✓ |
| CopyIn\<float\> Duplicate | inQueue_ | Dv * curDk * 4 | ≥ inSize | ✓ |
| CastGmToGm 输入 | inQueue_ | paddedCount * 4 | ≥ inSize | ✓ |
| CastGmToGm Duplicate | inQueue_ | paddedCount * 4 | ≥ inSize | ✓ |
| CastGmToGm Cast 输出 | outQueue_ | paddedCount * 2 | ≥ vInnerCastOutSize | ✓ |
| CopyOut\<bfloat16_t\> | outQueue_ | Dv * curDk * 2 | ≥ outSize | ✓ |
| CopyOut\<float\> | outQueue_ | Dv * curDk * 4 | ≥ inSize | ✓ |
| DataCopy(stateFp32Ub_) | tmpBuff_ | Dv * curDk * 4 | Dv * curDk * 4 | ✓ |
| Muls(stateFp32Ub_) | tmpBuff_ | Dv * curDk * 4 | Dv * curDk * 4 | ✓ |
| Add(stateFp32Ub_) | tmpBuff_ | Dv * curDk * 4 | Dv * curDk * 4 | ✓ |

> `paddedCount = Ceil(count, 16) * 16`（BF16 对齐），`count = curChunkSize * Dv`。对于典型值 Dv=128, curChunkSize=64: paddedCount=8192，需要 8192*4=32768 字节 < inSize=65536。✓

### 3.3 inQueue_/outQueue_ 跨类型复用

FP32 路径中 inQueue_ 和 outQueue_ 被多种类型复用：
- `inQueue_.AllocTensor<float>()` — CopyIn\<float\>, CastGmToGm\<float\>
- `inQueue_.AllocTensor<bfloat16_t>()` — BF16 路径（不执行）
- `outQueue_.AllocTensor<bfloat16_t>()` — Cast BF16 输出, CastGmToGm 输出
- `outQueue_.AllocTensor<float>()` — CopyOut FP32

由于 BUFFER_NUM_ONE=1，同一时刻只有一个 tensor 在队列中。跨类型复用在单 buffer 下是安全的，因为每次 AllocTensor 会获取整个 buffer。✓

### 3.4 结论

**UB 划分与复用无 bug。** 各操作所需空间均在 buffer 分配范围内。

---

## 4. 三 Stage 参数传递

### 4.1 Stage1 → Stage2 数据流

Stage1 输出（GM）→ Stage2 输入：

| Stage1 输出 | 数据类型 | Stage2 参数 | 用途 |
|------------|---------|------------|------|
| vInner_ | float (FP32 path) | sTP_->vInner | 被 CalVPrime 覆写为 v_new |
| gCum_ | float | sTP_->gCum | CalGCumExp 读取 |
| kCumDecay_ | bfloat16_t | sTP_->kCumdecay | CalVPrime 的 A 矩阵 |
| qPrime_ | bfloat16_t | sTP_->qPrime | CalAttnInter 的 A 矩阵 |
| kg_ | bfloat16_t | sTP_->kg | CalStateNew 的 B 矩阵 |

Stage1 和 Stage2 之间有 `SyncAll<false>()` 隔离。✓

### 4.2 Stage2 → Stage3 数据流

Stage2 输出（GM）→ Stage3 输入：

| Stage2 输出 | 数据类型 | Stage3 参数 | 用途 |
|------------|---------|------------|------|
| vInnerBf16_ | bfloat16_t | GetVInnerBf16() | Stage3 matmul 的 B 矩阵 |
| out_ (attnInter) | bfloat16_t | sTP_->attnOut | Stage3 accum=1 累加 |
| stateOut_ | float | (下一 group) | 最终 state 输出 |

Stage2 和 Stage3 之间有 `SyncAll<false>()` 隔离。✓

### 4.3 Stage2 内部 AIC/AIV 数据流

FP32 路径的 AIC/AIV 数据流（以 cId=0 为例）：

```
AIV:
  stateIn(FP32 GM) → CopyIn → stateFp32Ub_(UB)
  stateFp32Ub_ → Cast → curStateBf16(BF16 GM)
  stateFp32Ub_ → Muls(gCum) → stateFp32Ub_
  stateFp32Ub_ → CopyOut → stateOut(FP32 GM)

AIC:
  curStateBf16 → CalVPrime(kCumdecay @ state^T) → vInner(FP32 GM) [accum=1: vInner = value + vPrime]
  curStateBf16 → CalAttnInter(qPrime @ state^T) → out(BF16 GM)
  vInnerBf16 → CalStateNew(vInnerBf16^T @ kg) → stateOut(FP32 GM) [accum=0: overwrite]

AIV:
  vInner(FP32 GM) → CastGmToGm → vInnerBf16(BF16 GM)
  stateOut(FP32 GM) → CopyIn → deltaIn(UB)
  stateFp32Ub_ += deltaIn → stateFp32Ub_
  stateFp32Ub_ → CopyOut → stateOut(FP32 GM)
```

### 4.4 算法验证

Golden 算法：
```python
v_prime = k_cumdecay @ state
v_new = v_i - v_prime        # v_i = stage1's vInner = value
attn_inter = q_prime @ state
core_attn_out = attn_inter + attn @ v_new
new_state = state * exp(g[-1]) + kg^T @ v_new
```

Kernel 实现：
- Stage1 的 kCumDecay 包含 `-1` 因子（stage1.h:595 `Muls(gBUbFloat_, gBUbFloat_, -1, ...)`），所以 kernel 的 k_cumdecay = -golden 的 k_cumdecay
- CalVPrime 用 accum=1: `vInner = value + (-vPrime_golden) = value - vPrime_golden = v_new` ✓
- CalStateNew 用 accum=0: `stateOut = v_new^T @ kg` ✓
- AIV Add: `stateFp32Ub_ = gCum * state + delta = state * exp(g[-1]) + v_new^T @ kg` ✓

**算法正确。** ✓

### 4.5 结论

**参数传递无 bug。** Stage 间通过 GM 和 SyncAll 正确传递数据，算法实现与 golden 一致。

---

## 5. AIC/AIV 跨核同步

### 5.1 FP32 路径 CV 同步协议

```
AIV                              AIC
────                              ────
[chunk 0: write curStateBf16]
SetFlag(0x3, MTE3) ──────────→   WaitFlag(0x3)
                                 CalVPrime(kCumdecay, curStateBf16, vInner)
                                 CalAttnInter(qPrime, curStateBf16, out)
WaitFlag(0x2) ←──────────────   SetFlag(0x2, FIX)
CastGmToGm(vInner FP32 → vInnerBf16)
SetFlag(0x5, MTE3) ──────────→   WaitFlag(0x5)
                                 CalStateNew(vInnerBf16, kg, stateOut)
WaitFlag(0x4) ←──────────────   SetFlag(0x4, FIX)
CopyIn(stateOut), Add, CopyOut(stateOut)
```

### 5.2 同步正确性验证

| 步骤 | 写操作 | 写方 pipe | Set flag | 读操作 | 读方 pipe | Wait flag | 正确 |
|------|--------|----------|----------|--------|----------|-----------|------|
| 1 | AIV CopyOut curStateBf16 | MTE3 | 0x3 | AIC CalVPrime | FIX | 0x3 | ✓ |
| 2 | AIC CalVPrime→vInner | FIX | 0x2 | AIV CastGmToGm | MTE2 | 0x2 | ✓ |
| 3 | AIV CastGmToGm→vInnerBf16 | MTE3 | 0x5 | AIC CalStateNew | FIX | 0x5 | ✓ |
| 4 | AIC CalStateNew→stateOut | FIX | 0x4 | AIV CopyIn stateOut | MTE2 | 0x4 | ✓ |

**CV 同步无 bug。** 每个 flag 的 set/wait 配对正确，pipe 类型匹配。

---

## 6. 核内流水线同步

### 6.1 Stage2 AIV FP32 路径流水线事件

| 步骤 | 操作 | pipe | 同步事件 | 正确 |
|------|------|------|---------|------|
| CopyIn\<float\> Duplicate | V→MTE2 | V_MTE2_EVENT(7) | ✓ |
| DeQue → DataCopy(stateFp32Ub_) | MTE2→V | queue 内置 | ✓ |
| Cast(bf16←fp32) | V | — | ✓ |
| EnQue → CopyOut BF16 | V→MTE3 | queue 内置 | ✓ |
| CalGCumExp: GetValue | S | DataCacheClean | ✓ |
| CalGCumExp: Exp/Muls | S→V→S→V | S_V_EVENT(3) | ✓ |
| DataCopy(fp32Local←stateFp32Ub_) | V | V_MTE2_EVENT(7) | ✓ |
| CopyOut\<float\> | V→MTE3 | queue 内置 | ✓ |
| CastGmToGm Duplicate | V→MTE2 | V_MTE2_EVENT(7) | ✓ |
| CastGmToGm DataCopyPad | MTE2 | queue 内置 | ✓ |
| CastGmToGm Cast | V | queue 内置 | ✓ |
| CastGmToGm DataCopyPad out | MTE3 | queue 内置 | ✓ |
| CopyIn\<float\> stateOut | MTE2 | queue 内置 | ✓ |
| Add | V | — | ✓ |
| DataCopy + CopyOut | V→MTE3 | queue 内置 | ✓ |
| MTE3→MTE2 sync | MTE3→MTE2 | MTE3_MTE2_EVENT(4) | ✓ |

### 6.2 V_MTE2_EVENT 复用

`V_MTE2_EVENT = 7` 被多处复用：
1. CopyIn\<float\> 的 Duplicate 后
2. CalGCumExp 的 Muls 后（V_MTE2 sync for DataCopy）
3. CastGmToGm 的 Duplicate 后

这些操作在同一个 chunk 内顺序执行，每次 Set/Wait 配对完成后才进入下一步，无交叉。✓

### 6.3 stateFp32Ub_ 的 RAW 风险

`stateFp32Ub_` 位于 `tmpBuff_`（TBuf），不经过 queue 机制，所有 V 操作对其的读写依赖 `PipeBarrier<PIPE_V>()` 或隐式 V 排序：

1. `DataCopy(stateFp32Ub_, stateIn, ...)` — V 写
2. `Cast(bf16State, stateFp32Ub_, ...)` — V 读（紧接其后，隐式排序）✓
3. `Muls(stateFp32Ub_, stateFp32Ub_, ...)` — V 读写（CalGCumExp 内，有 S_V sync 前置）✓
4. `DataCopy(fp32Local, stateFp32Ub_, ...)` — V 读（CalGCumExp 后，有 PipeBarrier\<PIPE_V\>）✓
5. `Add(stateFp32Ub_, stateFp32Ub_, deltaIn, ...)` — V 读写（有 PipeBarrier\<PIPE_V\>）✓

**核内流水线同步无 bug。** ✓

---

## 7. 类型转换

### 7.1 FP32 → BF16 转换路径

| 转换 | 位置 | 方法 | RoundMode | 正确 |
|------|------|------|-----------|------|
| stateIn → curStateBf16 | stage2.h:147-150 | Cast + CopyOut | CAST_RINT | ✓ |
| vInner → vInnerBf16 | stage2.h:165 CastGmToGm | DataCopyPad + Cast + DataCopyPad | CAST_RINT | ✓ |

### 7.2 BF16 → FP32 转换路径

FP32 路径中无显式 BF16→FP32 转换。所有 FP32 数据要么来自用户输入（stateIn），要么来自 matmul 的 FP32 输出（mm1Fp32C）。✓

### 7.3 Matmul 类型配置

| Matmul | A 类型 | B 类型 | C 类型 | 用途 | 正确 |
|--------|--------|--------|--------|------|------|
| mm1Fp32C (CalVPrime) | BF16 | BF16 | FP32 | kCumdecay @ state^T → vInner | ✓ |
| mm1Fp32C (CalStateNew) | BF16 | BF16 | FP32 | vInnerBf16^T @ kg → stateOut | ✓ |
| mm1Fp32C (Stage1 attn@v_beta) | BF16 | BF16 | FP32 | attn @ v_beta → vInner | ✓ |

> 注意：CalVPrime 用 `mm1Fp32C`（C=FP32），accum=1，输出到 FP32 的 vInner。CalStateNew 也用 `mm1Fp32C`，accum=0，输出到 FP32 的 stateOut。

### 7.4 CastGmToGm 的 padding 处理

修复后的 CastGmToGm（stage2.h:343-374）：
- `paddedCount` 基于 dstType（BF16）对齐：`Ceil(count, 16) * 16`
- `srcPadded` 基于 srcType（float）对齐：`Ceil(count, 8) * 8`
- `padding = srcPadded - count`（float rightPadding，max=7）✓
- 当 `paddedCount > srcPadded` 时，先 Duplicate 清零再 DataCopyPad ✓
- Cast 处理 `paddedCount` 个元素（多出的为零）✓
- CopyOut 只写 `count` 个元素 ✓

### 7.5 CopyIn\<float\> 的 padding 处理

修复后的 CopyIn\<float\>（stage2.h:376-403）：
- `floatAligned = Ceil(col, 8) * 8`（float 对齐，max padding=7）✓
- `padding = floatAligned - col` ✓
- 当 `floatAligned < curDk_` 时：
  - `dstStride = (curDk_ - floatAligned) / 8`（32 字节块单位）✓
  - Duplicate 清零整个 buffer，V→MTE2 sync ✓
  - DataCopyPad 覆写有效数据，rightPadding 填充 [col, floatAligned) ✓
  - [floatAligned, curDk_) 保持为零（来自 Duplicate）✓

### 7.6 结论

**类型转换无 bug（修复后）。** 所有 FP32↔BF16 转换路径正确，padding 计算基于正确的类型对齐。

---

## 8. 多 chunk / 多 group state 处理分析

### 8.1 代码结构

FP32 路径 AIV 侧（stage2.h:136-159）的控制流：

```
if (GetSubBlockIdx() == 0) {
    if (isFirstGroup_ && cId == 0) {
        CopyIn<float>(stateIn)           ← 仅选择数据源
    } else {
        CopyIn<float>(stateOut)          ← cId>0 或 !isFirstGroup 时从 stateOut 读
    }
    // ↓ 以下对所有 chunk 都执行（在 if/else 之外）
    DataCopy(stateFp32Ub_, stateIn)      ← 更新 stateFp32Ub_
    Cast → CopyOut(curStateBf16)         ← 每个 chunk 都更新 curStateBf16
    CalGCumExp(gCum)                     ← 每个 chunk 都应用 gCum
    CopyOut(stateOut)                    ← 每个 chunk 都写入 stateOut
}
```

### 8.2 验证

| 场景 | CopyIn 数据源 | stateFp32Ub_ 更新 | curStateBf16 更新 | gCum 应用 | stateOut 写入 | 正确 |
|------|-------------|-------------------|-------------------|----------|-------------|------|
| first group, cId=0 | stateIn | ✓ | ✓ | ✓ | ✓ | ✓ |
| first group, cId>0 | stateOut | ✓ | ✓ | ✓ | ✓ | ✓ |
| !first group, cId=0 | stateOut | ✓ | ✓ | ✓ | ✓ | ✓ |
| !first group, cId>0 | stateOut | ✓ | ✓ | ✓ | ✓ | ✓ |

每个 chunk 的完整流程：
1. AIV 从 stateIn/stateOut 读取 FP32 state → stateFp32Ub_
2. AIV Cast 为 BF16 → curStateBf16（供 AIC 使用）
3. AIV 应用 gCum 衰减 → stateFp32Ub_ *= exp(gCum[-1])
4. AIV 写入 stateOut（gCum'd state，供 AIC CalStateNew 后的 Add 使用）
5. AIC 用 curStateBf16 做 CalVPrime 和 CalAttnInter
6. AIC CalStateNew → stateOut（覆写为 delta = v_new^T @ kg）
7. AIV 读取 stateOut（delta），Add 到 stateFp32Ub_
8. AIV 写入 stateOut（最终 state = gCum'd state + delta）

**多 chunk / 多 group state 处理无 bug。** ✓

---

## 9. 潜在问题（非 bug）

### ISSUE-1: curDk_ 始终使用 BF16 对齐

`curDk_ = Ceil(Dk, 16) * 16`（stage2.h:76），无论 stateType 是 float 还是 bfloat16_t。对于 FP32 state，理想情况下应使用 `Ceil(Dk, 8) * 8`（float 对齐），减少 UB 占用。但当前实现通过 CopyIn 修复中的 dstStride 和 Duplicate 正确处理了这一差异，不会导致错误，只是略有 UB 空间浪费。

### ISSUE-2: stateBf16Wk_ 地址对齐

`stateBf16Wk_` 的 GM 偏移取决于前面所有 workspace 分配的总和。对于某些 nv、s、dk、dv 组合，偏移可能不是 32 字节对齐。但由于 matmul 框架通常能处理非对齐地址（通过 tiling 配置），这不一定是 bug。需要在不同 shape 下验证。

### ISSUE-3: attnInter_ workspace 在 arch35 上未使用

Tiling 为 arch35 分配了 `attnInter_` 空间（tiling.cpp:138），但 stage2 的 CalAttnInter 直接写入 `sTP_->out`（用户输出），不经过 attnInter_。这是 arch22 兼容性遗留，不影响正确性。

---

## 10. 已修复的 Bug 总结

### 已修复: DataCopyPad rightPadding 溢出（commit fe63466bd）

**问题**: `CopyIn<float>` 和 `CastGmToGm<float, bfloat16_t>` 使用 BF16 对齐（max 15）计算 float 的 rightPadding，超出 float 的最大值 7（32B/4B=8 元素/块）。

**修复**:
- `CopyIn<float>`: 改用 `Ceil(col, BLOCK_FLOAT_NUM) * BLOCK_FLOAT_NUM` 计算 padding（max=7），通过 dstStride 维持 curDk_ 行布局，Duplicate 填充间隙
- `CastGmToGm`: 改用 srcType 对齐计算 padding，当 paddedCount > srcPadded 时 Duplicate 清零

---

## 11. 总结

| 维度 | 状态 | 说明 |
|------|------|------|
| Workspace 分配与使用 | ✓ 无 bug | Tiling 与 kernel 完全匹配 |
| UB 划分与复用 | ✓ 无 bug | 各操作所需空间充足 |
| 三 Stage 参数传递 | ✓ 无 bug | 数据流正确，算法与 golden 一致 |
| AIC/AIV 跨核同步 | ✓ 无 bug | flag 配对正确，pipe 类型匹配 |
| 核内流水线同步 | ✓ 无 bug | 事件同步正确，无 RAW 风险 |
| 类型转换 | ✓ 无 bug（修复后） | padding 基于正确类型对齐 |
| 多 chunk state 更新 | ✓ 无 bug | if/else 仅选数据源，后续操作每 chunk 都执行 |
| 多 group state 读取 | ✓ 无 bug | !isFirstGroup 的 cId=0 从 stateOut 读取 |

**结论**: 修复的 DataCopyPad rightPadding 溢出问题已解决 Dk=1/Dv=1 场景的 kernel 崩溃。FP32 state 路径在所有分析维度上均无 bug。
