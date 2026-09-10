# QFA 算子 MXFP8 模板 Kernel 技术文档

> 适用算子：`quant_flash_attn`（QFA），量化模板：MXFP8（`A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32`）
> 适用平台：Ascend 310（arch35，C310，Ascend 950 类）
> 本文档目的：完整梳理 MXFP8 模板 kernel 侧 PREFILL / DECODE 两套模板的所有计算流（C1/V1/C2/V2）、内存占用与 buffer 规划、跨核同步机制，供日后精度问题定位与技术复用。

---

## 1. 算子概述

QFA 的 MXFP8 模板做"全量化" Flash Attention：

- **输入**：Q / K / V 均为 `fp8_e4m3fn_t`（`INPUT_T`），配 `fp8_e8m0_t`（`SCALE_T`）逐张量/逐块缩放因子（dequantScale）。
- **计算精度**：MM1（QK^T）与 MM2（P·V）的累加类型 `T = float`（`MM_T`）。
- **中间结果 P**：以 `fp8_e4m3fn_t` 形态跨 AIC→AIV 传输，并携带 `fp8_e8m0_t` 的 PScale（per-32 元素一个 scale，即 MXFP8 格式）。
- **输出**：`OUT_T = bfloat16_t`（softmax 后 LSE 输出为 fp32）。

核心计算组件（4 段流水）：

| 组件 | 计算内容 | 执行核 | 函数 |
|------|----------|--------|------|
| C1 | S = Q·K^T · scale（MM1） | AIC（Cube） | `IterateBmm1` → `IterateBmm1Dn/Nd` |
| V1 | softmax(S) → P（含 mask、max/sum 在线更新） | AIV（Vector） | `ProcessVec1` → `ProcessVec1Dn/Nd` |
| C2 | O = P·V（MM2） | AIC（Cube） | `IterateBmm2` → `IterateBmm2Dn/Nd` |
| V2 | O 加权累加（flash 在线更新）→ 除以 sum → 输出 | AIV（Vector） | `ProcessVec2` → `ProcessVec2OnUb` |

**AIC:AIV = 1:2**（`CV_RATIO = 2`），一个 AIC 对应两个 AIV 子核（`subBlockIdx = 0/1`），M 维度按半块切分给两个 AIV 子核。

---

## 2. PREFILL 与 DECODE 两种模式

### 2.1 模式判定链路

```
Host 侧 (quant_flash_attn_tiling_mxfp8.cpp):
  decodeS1GMerge_ = (layoutQDescale == QfaLayout::N2TGD)   // Q descale 排布为 N2TGD 即按 N2 合并 → decode
  quantMode = decodeS1GMerge_ ? QFA_MXFP8_FP32_DECODE(2) : QFA_MXFP8_FP32_PREFILL(1)

Kernel 侧 (quant_flash_attn.cpp):
  useDn = (quantMode == QFA_MXFP8_FP32_PREFILL)   // 编译期常量，决定整套 DN/ND 模板
```

即：**PREFILL ? DN 模板；DECODE ? ND 模板**。`USE_DN` 为 `true` 时全链路走 `*Dn` 变体，`false` 时走 `*Nd` 变体。

### 2.2 核心差异总览

| 维度 | PREFILL（DN） | DECODE（ND） |
|------|--------------|--------------|
| quantMode | `QFA_MXFP8_FP32_PREFILL = 1` | `QFA_MXFP8_FP32_DECODE = 2` |
| USE_DN | `true` | `false` |
| 轴合并（InitConstInfo） | 不合轴：`realN2Size = n2Size*gSize`，`realGSize = 1` | 合轴：`realN2Size = n2Size`，`realGSize = gSize` |
| MM1 矩阵乘序 | K^T × Q：A=K(M=s2)，B=Q^T(N=m)，**Q 驻留 L1** | Q × K^T：A=Q(M=m)，B=K^T(N=s2)，**K 驻留 L1** |
| 典型场景 | 长序列 S1 大、KV 切外循环（GS1>80） | S1 小（decode 单 token），S2 大，按 S2 切 |
| L0C→UB Fixpipe 拆分 | `dualDstCtl = 2`（按 N 拆分，M×N/2 写入每个 UB） | `dualDstCtl = 1`（按 M 拆分，M/2×N 写入每个 UB） |
| MM2 的 PScale 来源 | **伪 scale**：`InitConstValue` 填 `0x7f7f`（= fp8_e8m0 的 1.0） | **真 PScale**：V1 阶段写入 L1P，C2 直接读取 |
| V1 软硬件交互 | `ProcessVec1VfDnMxfp8`（专用 DN 版本） | `ProcessVec1VfMxfp8`（按 s2CalcSize 分支 EQ_128 / ≤64 / 64~128 / 128~256） |
| actVecMSize（AIV 半块） | `actMSize <= 16 ? actMSize : actMSizeAlign32 >> 1` | `(actMSize + 1) >> 1` |
| 数据驻留 | Q 驻留 L1（多 S2 外循环复用） | K 驻留 L1（多 S1G 复用） |

### 2.3 模板参数与典型 Config

```
S1BaseSize = 128 (s1TemplateType)
S2BaseSize = 512（d≤128 时）或 256（d≤256 时）
dBaseSize / dVBaseSize ∈ {64, 72, 128, 256, 512, 576}
s2SplitSize = (dBaseSize == 256) ? 128 : 256    // C1/C2 的 K 方向子切分
```

注意：`s2BaseSizeCur = s2BaseSize >> 1`（256 或 128）是 V1 中单次处理的 S2 半块大小，C1 按 `s2SplitSize` 再切 subLoop（`c1v1Loop = ceil(actSingleLoopS2Size / s2SplitSize)`）。

---

## 3. Kernel 主执行流

文件：`op_kernel/arch35/quant_flash_attn_kernel_mxfp8.h`（类 `QuantFlashAttnKernelMxfp8`）

### 3.1 Init 阶段

`Init(...)` 顺序：

1. `InitConstInfo()`：解析 `QuantFlashAttnTilingData` → `constInfo_`；按 `USE_DN` 计算 `realN2Size/realGSize`。
2. 设置 Q/KV 实际序列长度 parser（`ActualSeqLensParser`：TND 用 cuSeqLens，BSH/BNSD 用 seqused）。
3. `InitMMResBuf(workspace)`：初始化跨核共享 buffer（见第 4 节）。
4. AIV 侧：`vecFaBlock_.InitVecBlock(...)` + `ClearOutput()`（用 `matmul::InitOutput` 把 attenOut / LSE 无效行清 0 / 置 inf）。
5. AIC 侧：`cubeBlock_.InitCubeBlock(...)`（绑定 GM tensor 与 offset calculator）。
6. 若 `FLASH_DECODE`：AIV 初始化 `vecFdBlock_`（FlashDecode 后处理，本模板 `isFdConst = false`，不启用）。

### 3.2 Process() 主循环

```cpp
if (aicIdx < coreNum) { CrossCoreBufferInit(); /* AIV: vecFaBlock_.InitBuffers()+AllocEventID; AIC: cubeBlock_.InitBuffers()+AllocEventID */ }
for (sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
    FlashAttention(sectionIdx);   // 主 FA 计算（AIC+AIV 协作）
    if (FLASH_DECODE) { if ASCEND_IS_AIV { FlashDecode(sectionIdx); } }
}
/* 收尾：FreeEventID + CrossCoreBufferUnInit */
```

- **sectionNum_** 来自 metadata[0]；每个 section 是一次完整的任务分片。
- `FlashAttention` 内通过 FA metadata（GM 中每核 16 个 uint32）拿到本核的 `bN2Start/End, gS1OStart/End, s2OStart/End, coreFirstTmpOutWsPos_`。

### 3.3 FlashAttention 任务调度（PRELOAD_N = 2 流水）

采用 **C1→C1→C1→C2** 的 2 级预载流水：

```
while (shouldDispatchTask || validTaskCount) {
    shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
    if (shouldDispatchTask) {
        mode = GetTaskDealMode(...);          // CREATE_TASK / DEAL_ZERO / SKIP_ZERO / NOT_START / SKIP_REMAINING_S2
        if (mode == CREATE_TASK) {
            CreateTask(createdTaskCount_, ...);   // 只填 RunInfoX（CalcParams），不执行
            validTaskCount++;
        }
        UpdateAxisInfo(mode, bN2Cur, gS1Cur, s2Cur);  // 快进 / 换行 / 换块
    }
    if (validTaskCount) {
        ExecuteTask(executedTaskCount, taskRunInfo);  // 真正执行
        if (executedTaskCount > PRELOAD_N) validTaskCount--;
    }
}
```

- `taskRunInfo[PRELOAD_TASK_CACHE_SIZE=3]` 环形缓存任务参数，`loop % 3` 取当前任务、`(loop - 2) % 3` 取 2 轮前的任务（供 C2/V2 消费）。
- **ExecuteTask 逻辑**：
  - 对当前任务 `runInfo0`：按 `c1v1Loop` 个 subLoop，AIC 执行 `ComputeMm1`（C1），AIV 执行 `ComputeVec1`（V1）。
  - 当 `loop >= PRELOAD_N` 时：消费 `runInfoNegN`（2 轮前的任务），AIC 执行 `ComputeMm2`（C2），AIV 执行 `ComputeVec2`（V2），并置 `isValid = false`。
- **任务轴序**：bN2（外层）→ gS1（中层）→ s2（内层）；`UpdateAxisInfo` 负责跨行/跨块快进。
- **Sparse 支持**：`GetTaskDealMode` 中 `CalcCurS2StartEndWithSparse`（BAND/CAUSAL/TREE 等）计算每行有效 S2 区间 `[curS2Start_, curS2End_)`，mask 模式下跳过无效块。
- **RunInfoX 关键字段**（`quant_flash_attn_common_def.h`）：`bIdx/n2Idx/realN2Idx/gS1Idx/s1Idx/s2Idx`、`actMSize/actMSizeAlign32/actVecMSize/vecMbaseIdx`、`actSingleLoopS2Size/actSingleLoopS2SizeAlign`、`isFirstS2Loop/isLastS2Loop/isS2SplitCore`、`faTmpOutWsPos`、`mloop/loop`。

### 3.4 C1/V1/C2/V2 调用链

```
ComputeMm1  (AIC) ──→ cubeBlock_.IterateBmm1(bmm1Buffers_.Get(), runInfo, subLoop)        [C1]
ComputeVec1 (AIV) ──→ vecFaBlock_.ProcessVec1(l1PBuffers_.Get/GetPre(), bmm1Buffers_.Get(), ...) [V1]
ComputeMm2  (AIC) ──→ cubeBlock_.IterateBmm2(bmm2Buffers_.Get(), l1PBuffers_, runInfo)   [C2]
ComputeVec2 (AIV) ──→ vecFaBlock_.ProcessVec2(bmm2Buffers_.Get(), runInfo)               [V2]
```

数据流（跨核共享内存）：

```
C1(AIC): GM Q/K → L1 → L0A/L0B → L0C → Fixpipe → UB(bmm1) ──→ V1(AIV): UB(bmm1) → softmax → P
V1(AIV): P + PScale ──→ L1(l1PBuffers_, 3buff) ──→ C2(AIC): L1(P) + L1(KV) → L0A/L0B → L0C → Fixpipe → UB(bmm2) ──→ V2(AIV)
V2(AIV): UB(bmm2) → 在线更新 → GM(attnOut / LSE)
```

---

## 4. 内存规划与 Buffer 布局

### 4.1 跨核共享 UB（InitMMResBuf，kernel 层）

| Buffer | Policy | 大小（字节） | 说明 |
|--------|--------|--------------|------|
| `ubBufferManager_` | UB | `mm1ResultSize*2 + mm2ResultSize` | 总管理器 |
| `bmm1Buffers_` | `BuffersPolicyDB<UB, CROSS_CORE_SYNC_BOTH>`（双缓冲） | 每块 `mm1ResultSize = (mBaseSize/2) * s2BaseSize * 4 / 2` | C1 结果（S 矩阵，fp32），AIC 写/AIV 读 |
| `bmm2Buffers_` | `BuffersPolicySingleBuffer<UB, CROSS_CORE_SYNC_BOTH>`（单缓冲） | `mm2ResultSize = (mBaseSize/2) * dVBaseSize * 4` | C2 结果（O 矩阵，fp32），AIC 写/AIV 读 |

典型数值（mBaseSize=128，T=fp32=4B，CV_RATIO=2）：

| Config | mm1ResultSize/块 | mm1 双缓冲 | mm2ResultSize | UB 合计 |
|--------|------------------|-----------|---------------|---------|
| S2=512, dV=128 | 64KB | 128KB | 32KB | 160KB |
| S2=256, dV=256 | 32KB | 64KB | 64KB | 128KB |

> 注：AIV 侧 `QuantFlashAttnBlockVecMxfp8::InitBuffers` 中 `mm1ResultSize = (mBaseSize/2)*s2BaseSize*4 = 128KB` 是"两个半块合计"口径（= kernel 单块 64KB × 2 个 dualDst 目标），并不重复申请内存，bmm1/bmm2 buffer 统一由 kernel 层分配。

### 4.2 跨核共享 L1（P 结果）

| Buffer | Policy | 大小（字节） | 内容 |
|--------|--------|--------------|------|
| `l1BufferManager_` | L1 | `524288`（512KB） | 总管理器 |
| `l1PBuffers_` | `BuffersPolicy3buff<L1, CROSS_CORE_SYNC_FORWARD>`（三缓冲） | `mm2LeftSize = mBaseSize*s2BaseSize*1 + mBaseSize*s2BaseSize/32` | V1 写 P(fp8) + PScale(fp8_e8m0)，C2 读 |

布局（单块）：`[0, mBaseSize*s2BaseSize)` 为 P（fp8，M×S2）；`[mBaseSize*s2BaseSize, ...)` 为 PScale（每 32 个 P 元素一个 fp8_e8m0）。

> 代码注释强调：保存 P 的 L1 内存必须放在第一个 L1 policy 上，保证与 vec 申请的地址相同（TODO 说明共享 buffer 初始化待下沉到 block 层）。

典型数值：S2=512 → 128*512 + 128*512/32 = 65536 + 2048 = **67584B（66KB）/块**；S2=256 → 33792B（33KB）/块。

### 4.3 AIC 侧 L1 / L0（cube block 层，InitBuffers）

```
l1QBuffers_  = mBaseSize*dBaseSize*1 + mBaseSize*dBaseSize/32          // Q + QScale（fp8）
l1KVBuffers_ = dBaseSize*s2BaseSize*1 + s2BaseSize*dBaseSize/32        // d=256 时全量（decode 方案）
             = dBaseSize*s2BaseSizeCur*1 + s2BaseSizeCur*dBaseSize/32  // d≠256 时按半块
l0aBufferManager_.Init(tPipe_, 65536);    // 64KB；mmL0ABuffers_ 每块 32KB
l0bBufferManager_.Init(tPipe_, 65536);    // 64KB；mmL0BBuffers_ 每块 32KB
l0cBufferManager_.Init(tPipe_, 262144);   // 256KB
mmL0CBuffers_.Init = (mBaseSize*s2BaseSize*4 <= 256KB/4 && mBaseSize*dVBaseSize*4 <= 256KB/4)
                     ? 64KB : 128KB
```

典型数值（S2=512, d=128）：l1Q ≈ 16384+512 = 16896B；l1KV ≈ 128*256 + 256*128/32 = 32768+1024 = 33792B。

### 4.4 AIV 侧 UB 布局（vec block 层，InitBuffers）

| Buffer | 大小（字节） | 内容 |
|--------|--------------|------|
| `stage1OutQue_[2]` | 每块 `16640` | softmax 结果（P，fp8）暂存 + **PScale 复用区**（`pScaleByteOffset = 16640`，见第 9 节） |
| `stage2OutBuf_` | `64 * dTemplateAlign64 * 4`（dV=128 → 32KB） | V2 结果暂存 |
| `softmaxSumBuf_[3]` | 256B×3 | sum（fp32 [64,1]） |
| `softmaxMaxBuf_[3]` | 256B×3 | max（fp32 [64,1]） |
| `softmaxExpBuf_[3]` | 256B×3 | exp 结果 |
| `preLoopMaxBuf_ / preLoopSumBuf_ / firstLoopSumBuf_` | 各 256B | 在线 softmax 状态 |
| `vselrIndexesBuf_[4]` | DN: 256B；ND: 128B + 64B | vselr 索引表（数据重排） |
| `attenMaskInQue_[0]` | `(mBaseSize/2) * (s2BaseSize/2)` = 16384B | mask 输入（HAS_MASK 时） |
| `commonTBuf_` | 512B | 临时区 |
| `softmaxLseQueue_` | `(mBaseSize/2)*4*8` = 2048B | LSE 输出（LSE 使能时） |
| `maxBrdcst_ / sumBrdcst_` | 2048B×2 | FD 模式 broadcast（FLASH_DECODE 时） |

### 4.5 GM 侧

- **metadata**：FA 部分 `QFA_AIC_CORE_NUM(36) * 16 * sectionNum`；FD 部分偏移 `FA_METADATA_HEADER_OFFSET + QFA_METADATA_SIZE(16)*36*sectionNum*4B` 之后，`QFA_AIV_CORE_NUM(72) * 16 * sectionNum`。
- **workspace**（FLASH_DECODE 时）：`faTmpAttenGmSize = numBlocks_*2*mSize*dSize`，`logSumExpSize = numBlocks_*2*mSize*8`，`accumOut` 在 workspace 首部，`softmaxFDSum/Max` 紧随其后（InitVecInput）。

---

## 5. C1（MM1）实现详解

文件：`op_kernel/arch35/quant_flash_attn_block_cube_mxfp8.h`（`QuantFlashAttnBlockCubeMxfp8`）

入口：`IterateBmm1(outputBuf=UB(bmm1), runInfo, subLoop)` → 按 `USE_DN` 分发。

### 5.1 ND（decode）：`IterateBmm1Nd`

计算：**Q × K^T**（M=mBaseSize, N=s2，K=d）。K 每次搬入，Q 首轮搬入后驻留。

1. **s2CurSize 计算**：默认 `s2SplitSize`；尾部 `actSingleLoopS2Size < s2BaseSize` 时，subLoop0 取 `min(s2SplitSize, actSize)`，否则取剩余。
2. **搬 Q**（仅 `isFirstS2Loop && subLoop==0`）：
   - `mm1A = l1QBuffers_.Get()`，`Wait<MTE1_MTE2>`；
   - `CopyQueryTile`（GM→L1，NZ 布局，dstStride = actMSize 对齐 32）＋ `CopyQueryScaleTile`（QScale，偏移 `qScaleOffset = actMSizeAlign32 * dBaseSize`）；
   - `Set<MTE2_MTE1>`。其余轮次用 `GetPre()`（ping-pong）。
3. **搬 K**（每 subLoop）：
   - `mm1B = l1KVBuffers_.Get()`，`Wait<MTE1_MTE2>`；
   - `CopyKeyTile(mm1BTensor, runInfo, s2CurSize, subLoop)`：`s2Offset = subLoop%2==0 ? s2Idx : s2Idx + s2BaseSize/2`（K 半块 ping-pong 搬入）；
   - `CopyKeyScaleTile`（偏移 `kScaleOffset = s2CurSizeAlign32 * dBaseSize`）；
   - `Set<MTE2_MTE1>`；然后 `mm1A.Wait<MTE2_MTE1>(); mm1B.Wait<MTE2_MTE1>();`
4. **Matmul**：
   ```
   MatmulFullMX<Q_T, KV_T, T, 128, s2BaseSize>>1, dBaseSize, ABLayout::MK, ABLayout::KN,
                L0AType, L0BType, SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
       mm1A(Q, L1), mm1B(K, L1), mmL0ABuffers_, mmL0BBuffers_, mm1ResL0C,
       MakeMMParam(actMSize, s2CurSize, dBaseSize, false, true),
       mm1A+[qScaleOffset], mm1B+[kScaleOffset]);
   ```
   （mmL0ABuffers_ 的 scale 参数为 `false,true`：A 无 trans、B 有 trans）
5. **同步与搬出**：`mm1ResL0C.Set<M_FIX>() → Wait<M_FIX>() → outputBuf.WaitCrossCore() → FixpipeMm1 → Set<FIX_M>() → outputBuf.SetCrossCore()`。

### 5.2 DN（prefill）：`IterateBmm1Dn`

计算：**K^T × Q**（M=s2, N=mBaseSize，K=d）。**Q 驻留 L1**，K 每次搬入。

- 矩阵乘角色对调：**B 矩阵（`mm1B = l1QBuffers_`）驻留**，A 矩阵（`mm1A = l1KVBuffers_`）每轮搬入。
- Matmul 参数：
  ```
  MatmulFullMX<Q_T, KV_T, T, s2BaseSize>>1, 128, dBaseSize, ABLayout::MK, ABLayout::KN,
              L0AType, L0BType, SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
      mm1A(K), mm1B(Q), ..., MakeMMParam(s2CalcSize, actMSize, dBaseSize, false, true),
      mm1A+[kScaleOffset], mm1B+[qScaleOffset]);
  ```
- `s2CalcSize` 计算同 ND（尾部修正）。
- 释放策略差异：DN 在 `isLastS2Loop` 时释放的是 **mm1B（Q 驻留缓冲）**。

### 5.3 Fixpipe（L0C→UB）差异

```cpp
// ND (FixpipeMm1): dualDstCtl = 1，按 M 拆分
fixpipeParams.nSize       = (s2RealSize + 7) >> 3 << 3;            // N = S2
fixpipeParams.mSize       = (actMSize + 31) >> 5 << 5;             // M = 行数(对齐32)
fixpipeParams.srcStride   = ((actMSize + 15) / 16) * 16;
fixpipeParams.dstStride   = s2SplitSize;                           // UB 两行间隔
fixpipeParams.dualDstCtl  = 1;                                     // M/2×N 写每个 UB

// DN (FixpipeMm1Dn): dualDstCtl = 2，按 N 拆分
fixpipeParams.nSize       = (actMSize + 31) >> 5 << 5;             // M（角色互换）
fixpipeParams.mSize       = (s2RealSize + 7) >> 3 << 3;            // S2
fixpipeParams.srcStride   = ((fixpipeParams.mSize + 15) / 16) * 16;
fixpipeParams.dstStride   = 64;                                    // UB 两行间隔
fixpipeParams.dualDstCtl  = 2;                                     // M×N/2 写每个 UB
```

关键点：**Fixpipe 的结果排布决定 AIV 侧读取方式**。ND 按 M 拆成上下两块（subBlockIdx 0 取上半、1 取下半）；DN 按 N 拆成左右两块（每个 subBlock 处理全部 64 行 × N/2）。

---

## 6. V1（softmax → P）实现详解

文件：`op_kernel/arch35/quant_flash_attn_block_vec_mxfp8.h`（`QuantFlashAttnBlockVecMxfp8`）

入口：`ProcessVec1(outputBuf=L1P, bmm1ResBuf=UB(bmm1), runInfo, subLoop)`：

```cpp
bmm1ResBuf.WaitCrossCore();
if (actVecMSize == 0) { bmm1ResBuf.SetCrossCore(); ...; return; }
USE_DN ? ProcessVec1Dn(...) : ProcessVec1Nd(...);
```

`actVecMSize`：AIV 子核实际处理行数（M 半块），`vecMbaseIdx` 为该子核行偏移（见 2.2 节公式，`subBlockIdx==1` 时取后半个 M）。

### 6.1 公共输入

- softmax 状态 buffer：`sumUb/maxUb = softmaxSumBuf_/MaxBuf_[mloop % 3]`、`expUb = softmaxExpBuf_[loop % 3]`、`preLoopMax/Sum`、`firstLoopSum`。
- `mmRes = bmm1ResBuf.GetTensor<T>()`（C1 结果 fp32）。
- `stage1CastTensor = stage1OutQue_[subLoop % 2].AllocTensor<INPUT_T>()`（输出 P，fp8）。
- **PScale 区复用**：`pScaleSubLoop0Tensor = stage1CastTensor.ReinterpretCast<fp8_e8m0_t>()[16640]` —— 16640 即 softmax 结果区的字节大小，PScale 数据紧跟在同一个 stage1 buffer 之后（同一块 UB 空间复用）。
- `s2CalcSize`：`actSingleLoopS2Size > s2SplitSize` 时 subLoop0 = `s2BaseSizeCur`，否则为剩余。
- `descaleQK = 1.0`（scale 已由 cube 侧通过 dequantScale 处理）；`scaleValue` 传入 `ProcessVec1Vf*` 作为 QK 缩放。

### 6.2 DN：`ProcessVec1Dn`

- `maskLine = ComputeMaskLineDN(runInfo, subLoop)`（HAS_MASK 时）；`isFullMask_` 时 `maskLine = -256`。
- 调用 **`ProcessVec1VfDnMxfp8<T, INPUT_T, isFirstS2Loop, hasAtten, s2BaseSizeCur>`**（`vf_mul_sel_softmaxflashv2_cast_nz_dn.h`），参数：
  `(stage1CastTensor, sumUb, maxUb, mmRes, expUb, vselrIndexesBuf_[DN_INDEX], pScaleSubLoop0Tensor, vecMSizeAlign, actS2Align/2, s2CalcSize, scaleValue, descaleQK, pScaleValue_, negativeFloatScalar_, 0.0F, preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, maskLine)`。
- 完成：`bmm1ResBuf.SetCrossCore()`（释放 UB 供下一轮 C1 写）。
- **P 写入 L1P**（`DataCopy` 循环）：
  - 目标：`mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>()`；
  - `subLoopOffset = s2BaseSizeCur * mBaseSize * subLoop`；
  - 循环 `i ∈ [0, ceil(mBaseSize/2/32))`：`dstOffset = 32*s2BaseSizeCur*i + subLoopOffset`（subBlockIdx==1 再加 `s2BaseSizeCur*actVec0Align32`），`srcOffset = i*(((s2BaseSizeCur>>2)+1)<<5)`，`DataCopy(dst, src, {4, s2BaseSizeCur/4, s2BaseSizeCur/4+2, 0})`。
- **PScale 写入 L1P**（仅当第二个 subLoop 或单 subLoop，即整块 S2 处理完）：
  - 目标偏移 `pScaleL1Offset = mBaseSize*s2BaseSize`（元素单位），`vecOffset = subBlockIdx * pScaleDataLen`，`pScaleDataLen = (mBaseSize>>1)*s2BaseSizeCur/32`；
  - `DataCopy(mm2AScaleL1Tensor[vecOffset+i*32], pScaleSubLoop0Tensor, {4, 1, 0, pScaleDstStride})`，`pScaleDstStride = s2BaseSizeCur/32/2 - 1`，循环 `copyTimes = s2BaseSizeCur/32/2` 次（16×2 分形重复 4 次）。
- `outputBuf.SetCrossCore()`（整块 S2 处理完才放行 C2）。
- `isLastS2Loop` 时 `SoftmaxDataCopyOut`（LSE/FD sum-max 写出，见 6.4）。

### 6.3 ND：`ProcessVec1Nd`

- 先 `AttenMaskCopyIn`（HAS_MASK 时全量拷贝 mask 到 `attenMaskInQue_[0]`）。
- 调用 **`ProcessVec1VfMxfp8<T, INPUT_T, pseShiftType, isFirstS2Loop, mBaseSize, s2BaseSizeCur, S2RANGE_TAG, HAS_MASK, PSE_NONE_TYPE, ...>`**（`vf_mul_sel_softmaxflashv2_cast_nz.h`），按 `s2CalcSize` 分支：
  - `s2CalcSize == 128` → `EQ_128`；
  - `<= 64` → `GT_0_AND_LTE_64`；
  - `< 128` → `GT_64_AND_LTE_128`；
  - 否则（s2BaseSizeCur==256）→ `GT_128_AND_LTE_256`。
- 传入 `attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb(=空), preLoopMaxUb, preLoopSumUb, firstLoopSumUb, subLoop, actVecMSizeAlign16, s2CalcSize, pseStride, slopes, posShift, scaleValue, descaleQK, negativeFloatScalar_, 0.0F, queryScaleUb, deSCaleKValue, pScaleValue_`。
- **P 写入 L1P**：
  ```
  dstOffset = subBlockIdx * 32 * actVecMSizeAlign16 + s2BaseSizeCur*mBaseSize*subLoop;
  DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor,
           {s2BaseSizeCur/32, actVecMSizeAlign16, vec1Srcstride - actVecMSizeAlign16, mBaseSize - actVecMSizeAlign16});
  ```
  其中 `vec1Srcstride = (mBaseSize>>1) + 1 = 65`。
- **PScale 写入 L1P**：`copyCount = actVecMSizeAlign16/16` 行；
  - 双 subLoop 的第二个（`actSize > s2SplitSize && subLoop%2==1`）：从 `pScaleSubLoop0Tensor` 与 `pScaleSubLoop0Tensor[128]` 两段各拷一份 `{copyCount,1,0,pScaleDstStride}`；
  - 单 subLoop（`actSize <= s2SplitSize`）：仅拷第一段。
  - `pScaleSubLoopOffset = pScaleDataLen * 2` 为第二段目标偏移。

### 6.4 softmax 状态维护与输出

- `SoftmaxDataCopyOut`：FLASH_DECODE 且 `isS2SplitCore` → `ComputeLogSumExpAndCopyToGm`（把 sum/max 经 `BroadcastMaxSum`（[64,1]→[64,8]）写到 workspace 的 `softmaxFDSumGm_/softmaxFDMaxGm_`，供 FD 阶段用）；否则 LSE 使能 → `SoftmaxLseCopyOut`（`ComputeLseOutputVF` 生成 LSE，按 TND/NTD/BSH/BNSD 不同布局 `DataCopySoftmaxLse*Arch35` 写出）。

---

## 7. C2（MM2）实现详解

入口：`IterateBmm2(outputBuf=UB(bmm2), inputBuf=l1PBuffers_, runInfo)` → 按 `USE_DN` 分发。A 为 L1P 中的 P（含 PScale），B 为 V。

### 7.1 公共流程

1. `mm2A = inputBuf.Get()`；`mm2A.WaitCrossCore()`（等 V1 写完 P）。
2. **PScale 偏移**：`pScaleOffset = mBaseSize * s2BaseSize`（元素单位），`mm2AScaleFakeTensor = mm2A.GetTensor<SCALE_T>(pScaleOffset)`。
3. 按 `baseK = s2SplitSize` 切 K 循环：`kLoops = ceil(actSingleLoopS2Size / baseK)`；每轮：
   - `mm2B = l1KVBuffers_.Get()`，`Wait<MTE1_MTE2>`；
   - `CopyValueSlice(mm2BTensor, s2Idx + k*baseK, realK, 0, dSizeV)`（V 半块）；
   - `vScaleOffset = align64(realK) * dSizeV`；`CopyValueScaleSlice(..., (s2Idx+k*baseK)/64, ..., 0, dSizeV*2)`（VScale，每 64 个 S2 一个、d 方向每 2 个一个）；
   - `Set<MTE2_MTE1>() → Wait<MTE2_MTE1>()`；
   - Matmul：
     ```
     MatmulFullMX<INPUT_T, KV_T, T, 128, dVBaseSize, baseK, ABLayout::MK, ABLayout::KN,
                  L0AType, L0BType, SCALE_T, SCALE_T, mx_fp8_e4m3_t, mx_fp8_e4m3_t>(
         mm2A[kIdx*l1BaseKOffset], mm2BTensor, mmL0ABuffers_, mmL0BBuffers_, mm2ResL0C,
         MakeMMParam(mBaseSize, dVBaseSize, align64(realK), isTransA?, false, kIdx==0, kIdx==0),
         mm2AScaleFakeTensor[kIdx*l1ScaleOffset], mm2BScaleTensor);
     ```
     - ND：`param.realM = actMSize`；`l1ScaleOffset = baseK/32 * actMSizeAlign32`；`MakeMMParam(..., false, false, k==0, k==0)`。
     - DN：`l1ScaleOffset = l1BaseKOffset / 32`；`MakeMMParam(..., USE_DN, false, k==0, k==0)`。
   - `l1BaseKOffset = baseK * mBaseSize`（P 在 L1 的 K 方向步进，fp8 元素）。
   - `mm2B.Set<MTE1_MTE2>()`（释放 L1B）。
4. 尾部处理：`isLastS2Loop` 时先 `InitValueL1BufferNoTrans`（nd2nz 补 64 对齐 padding 为 0）。
5. 结果搬出：`mm2ResL0C.Set<M_FIX>() → Wait<M_FIX>() → outputBuf.WaitCrossCore() → FixpipeMm2 → Set<FIX_M>() → outputBuf.SetCrossCore()`。

### 7.2 PScale 差异（关键）

- **ND（decode，真 PScale）**：直接读取 V1 写入的 PScale，`mm2AScaleFakeTensor[kIdx*l1ScaleOffset]` 即为真实缩放。
- **DN（prefill，伪 scale）**：
  ```cpp
  if (actSingleLoopS2Size > s2SplitSize)
      InitConstValue(mm2AScaleTensor, {1, mBaseSize*s2SplitSize/32*1/32, 0, 0x7f7f});  // 偏移 pScaleOffset/2 + mBaseSize*s2SplitSize/32/2
  else
      InitConstValue(mm2AScaleTensor, {1, mBaseSize*s2SplitSize/32*1/32, 0, 0x7f7f});  // 偏移 pScaleOffset/2
  ```
  即把 PScale 区填成 `0x7f7f`（int16 视角），对应 fp8_e8m0 的 **1.0（伪 scale）** —— 因为 DN 路径中 softmax 结果 P 本身已通过 `ProcessVec1VfDnMxfp8` 将 scale 折算（descaleQK/scaleValue 在 V1 内部处理），V 侧 dequantScale 由 cube 的 dequantScaleValue 处理，故 C2 无需再乘真实 PScale。

### 7.3 FixpipeMm2

```cpp
nSize       = (dSizeV + 7) >> 3 << 3;              // N = dV
mSize       = ND: (actMSize+31)>>5<<5   DN: mBaseSize;
srcStride   = ND: ((actMSize+15)/16)*16  DN: ((mBaseSize+15)/16)*16;
dstStride   = (dVTemplateType + 15) >> 4 << 4;
dualDstCtl  = 1;                                   // 按 M 拆分（DN/ND 均为 1）
```

---

## 8. V2（O 加权累加 + 输出）实现详解

入口：`ProcessVec2(bmm2ResBuf=UB(bmm2), runInfo)` → `WaitCrossCore()` → `ProcessVec2OnUb`。

### 8.1 加权累加（flash 在线更新）

```cpp
vecMSize    = USE_DN ? actVecMSize : actMSizeAlign32 / 2;
vec2CalcSize = vecMSize * dTemplateAlign64;
vec2ResUb    = stage2OutBuf_.Get<T>();
mmRes        = bmm2ResBuf.GetTensor<T>();
WaitFlag<MTE3_V>(mte3ToVId_[0]);   // 等上一轮输出搬完

if (isFirstS2Loop)
    DataCopy(vec2ResUb, mmRes, vec2CalcSize);        // 第一块：直接拷入（相当于 ×1）
else if (!isLastS2Loop)
    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);
    // O = O_prev*exp(max_prev - max) + P*V
else
    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, vecMSize, dTemplateAlign64, 1.0f, 1.0f);

bmm2ResBuf.SetCrossCore();
```

- `expUb = softmaxExpBuf_[loop % 3]`：上一轮 exp(max - max_prev) 结果（由 V1 写入）。
- 最后一个 S2 循环：`LastDivNew`（O /= sum）后输出；若同时是第一个 S2 循环（单块场景），先 `LastDivNew` 再输出。

### 8.2 输出路径

```
CopyOutAttentionOut(runInfo, vec2ResUb, 0, vecMSize, DealRowCount)
 ├─ FLASH_DECODE && isS2SplitCore: Bmm2ResForFDCopyOut → DataCopyPad 写 workspace(accumOutGm_)
 └─ 否则: Bmm2ResCastAndCopyOut
        ├─ RowInvalid (HAS_MASK 且稀疏窗口边界行越界时，用 max + minValue_ 把无效行刷为极小值)
        ├─ Cast(attenOut, vec2ResUb, CAST_ROUND, mDealSize*dSizeAligned64)   // fp32 → bf16（POST_QUANT 时走 PostQuant）
        └─ Bmm2DataCopyOutTrans → CopyAttentionOut（TND→TNGD / NTD→NGTD / BSH→BSNGD / BNSD→BNGSD 布局转换写 GM）
```

- `SetFlag/WaitFlag<V_MTE3>(vToMte3Id_[0])` 保证 Cast 完成后再搬出。
- `DealRowCount`：DN = `actVecMSize`；ND = `subBlockIdx==0 ? min(actMSize, groupsOf32*16) : actMSize - 16*groupsOf32`（上/下半块各自的有效行数）。

---

## 9. PScale 全链路

PScale（fp8_e8m0）的生成与消费是 MXFP8 模板的核心：

```
[V1 阶段] ProcessVec1Vf*Mxfp8 内部计算出每 32 个 P 元素的 scale
    └─ 写入 stage1CastTensor 偏移 16640 处（同一 UB 块复用）
    └─ DataCopy 循环写入 L1P: offset = mBaseSize*s2BaseSize + subBlockIdx*(M/2*S2/32) + i*32
[L1P 布局] | P(fp8, M×S2) | PScale(fp8_e8m0, M×S2/32) |
[C2 阶段]  ND: 直接读 L1P 中的真 PScale 作为 A 矩阵 scale
           DN: 将 L1P 的 PScale 区 InitConstValue 为 0x7f7f（伪 scale = 1.0），
               因为 DN 路径 V1 已把 descale/scale 折算进 P
```

> 关键陷阱（精度问题排查点）：DN 模板下若误读 PScale 区未初始化数据，或 ND 模板下 V1 未写完 PScale 就 SetCrossCore，都会导致 C2 结果异常。DN 的伪 scale 为 `0x7f7f`（int16），对应 fp8_e8m0 的 1.0。

---

## 10. 同步机制汇总

| 同步对象 | 机制 | 位置 |
|----------|------|------|
| C1 写 UB(bmm1) → V1 读 | `bmm1Buffers_.Set/WaitCrossCore()`（CROSS_CORE_SYNC_BOTH） | `IterateBmm1*` / `ProcessVec1` |
| C2 写 UB(bmm2) → V2 读 | `bmm2Buffers_.Set/WaitCrossCore()` | `IterateBmm2*` / `ProcessVec2` |
| V1 写 L1P → C2 读 | `l1PBuffers_.Set/WaitCrossCore()`（CROSS_CORE_SYNC_FORWARD，3buff） | `ProcessVec1*` / `IterateBmm2*` |
| MTE3 搬出 vs Vector 计算 | `mte3ToVId_[2]/vToMte3Id_[2]` event + `SetFlag/WaitFlag<MTE3_V/V_MTE3>` | `ProcessVec2OnUb` / `Bmm2ResCastAndCopyOut` |
| L0C 与 Fixpipe | `Set/Wait<M_FIX>`、`Set/Wait<FIX_M>` | `IterateBmm1* / IterateBmm2*` |
| L1 与 Cube | `Wait/Set<MTE1_MTE2>`、`Set/Wait<MTE2_MTE1>` | cube block 各 Copy 函数 |
| 输出清零 | `initOutputEventId` + `Set/WaitFlag<MTE3_V>` + `SyncAll()` | `ClearOutput` |

---

## 11. 关键常量表

| 常量 | 值 | 说明 |
|------|----|------|
| `CV_RATIO` | 2 | AIC:AIV = 1:2（`attention/flash_attn/op_host/fa_tiling_info.h`） |
| `PRELOAD_N` | 2 | C1/C2 预载深度（流水级数） |
| `PRELOAD_TASK_CACHE_SIZE` | 3 | RunInfoX 环形缓冲数 |
| `MXFP_GROUP_SIZE` | 32 | PScale 粒度（每 32 元素 1 scale） |
| `MXFP_DIVISOR_SIZE` | 64 | VScale 的 S2 粒度 |
| `MXFP_MULTI_BASE_SIZE` | 2 | VScale 的 d 方向粒度 |
| `s2SplitSize` | d=256→128，否则 256 | C1/C2 的 S2/K 子切分 |
| `DB` | 2 | stage1OutQue_ ping-pong 数 |
| `pScaleByteOffset` | 16640 | stage1 缓冲内 PScale 复用偏移 |
| `pScaleL1Offset` | `mBaseSize*s2BaseSize` | L1P 中 PScale 偏移（元素） |
| `QFA_AIC_CORE_NUM / AIV_CORE_NUM` | 36 / 72 | metadata 核数常量 |
| `QFA_MXFP8_FP32_PREFILL / DECODE` | 1 / 2 | quantMode 枚举 |
| 平台内存 | L1=512KB, L0C=256KB, L0A/L0B=64KB | 当前写死（`l1BufferManager_.Init(pipe_, 524288)` 等） |

---

## 12. 精度问题排查速查表（复用要点）

1. **模式判定**：先确认 tilingKey 的 quantMode 是 PREFILL(1)/DECODE(2)，对应 DN/ND；`layoutQDescale == N2TGD` 才会走 DECODE。
2. **scale 是否正确折算**：
   - DN：C2 用伪 scale（0x7f7f），真实 scale 在 V1 内部折算；若把 pScaleValue_（GM 读入）误当 1.0 会导致 DN 结果偏小。
   - ND：C2 直接读 L1P 的 PScale；若 V1 未写 PScale 或写入顺序错（subBlockIdx 偏移），C2 结果异常。
3. **UB 复用陷阱**：stage1 缓冲 16640B 之后存 PScale，需确认 `pScaleSubLoop0Tensor` 与 P 数据的写读时序（EnQue/DeQue/FreeTensor 配对）。
4. **actVecMSize / vecMbaseIdx**：AIV 子核 M 半块划分公式 DN 与 ND 不同（2.2 节），错配会导致只算一半行或重复计算。
5. **Fixpipe 排布**：dualDstCtl=1（M 拆）/2（N 拆），UB 读取 stride 必须与之一致（`dstStride` 参数），否则结果错位。
6. **尾部非整块**：`actSingleLoopS2Size < s2BaseSize` 时 s2CurSize/s2CalcSize 的尾部修正分支（subLoop0 取 min），勿直接使用完整 s2SplitSize。
7. **mask 相关**：`ComputeMaskLineDN` 维护 `isSkipMask_/isFullMask_` 状态，直接影响 VF 模板分支；mask 时 `maskLine=-256` 表示全 mask。
8. **LSE 输出**：`3e+99` 初始化无效行；TND 布局 LSE 为 N-major 排布，写 GM 用 `DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge`。
9. **在线 softmax 状态**：`preLoopMax/Sum`（两轮前）、`firstLoopSum`（首轮 sum）由 V1 维护、V2 依赖 `expUb` 做加权，顺序错乱会引入 flash 更新误差。
10. **workspace（FD）**：`accumOut/logSumExp` 的偏移按 `faTmpAttenGmSize` 对齐，`isS2SplitCore` 决定 O 写 workspace 还是直接输出。

---

## 13. 涉及源文件清单

| 文件 | 作用 |
|------|------|
| `op_kernel/quant_flash_attn.cpp` | 模板实例化入口，`useDn = (quantMode==PREFILL)` |
| `op_kernel/arch35/quant_flash_attn_kernel_mxfp8.h` | Kernel 编排：调度、C1/V1/C2/V2 分发、内存管理 |
| `op_kernel/arch35/quant_flash_attn_block_cube_mxfp8.h` | C1/C2（IterateBmm1/2 + Fixpipe + 数据搬入） |
| `op_kernel/arch35/quant_flash_attn_block_vec_mxfp8.h` | V1/V2（ProcessVec1/2 + softmax 状态 + 输出） |
| `op_kernel/arch35/quant_flash_attn_common_def.h` | `ConstInfo_t`、`RunInfoX`、quantMode 宏 |
| `op_kernel/arch35/quant_flash_attn_tiling_data.h` | tiling 数据结构与 metadata 索引常量 |
| `op_kernel/arch35/quant_flash_attn_template_tiling_key.h` | 模板 key 组合（含 quantMode） |
| `op_host/arch35/quant_flash_attn_tiling_mxfp8.cpp` | Host tiling：模式判定、config 选择、workspace |
| `common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h` | V1 DN 的 VF 内核函数 `ProcessVec1VfDnMxfp8` |
| `common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h` | V1 ND 的 VF 内核函数 `ProcessVec1VfMxfp8` |
