# FAG Kernel侧计算流水与Cube/Vec详解

---

## 第一部分：Kernel侧计算流水总结

### 1.1 FAG反向传播数学公式

Flash Attention Score Grad（FAG）是Flash Attention的反向传播算子，计算dq/dk/dv三个梯度。数学推导如下：

给定前向传播结果：
- `S = Q·K^T · scale`  （attention score）
- `P = softmax(S)`  （attention probability）
- `O = P·V`  （attention output）

反向传播输入：`dy`（对O的梯度）

梯度公式：
```
dV = P^T · dy           → mm5
dS = dropout(P) · dy - softmaxGradSum · P   → Vec3
dQ = dS · Q^T / scale   → mm3 + Vec5 muls
dK = dS · K^T / scale   → mm4 + Vec6 muls
```

其中 `softmaxGradSum = Σ(dy·O)` 沿s2轴的归约结果。

### 1.2 整体流水架构

FAG采用 **Cube-Vector三级流水线**，每级包含Cube计算和Vector计算的并行协作：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        三级流水线总览                                │
│                                                                     │
│  ┌─── Pre级 (当前block) ───┐  ┌─── Base级 (前一个block) ───┐      │
│  │  C1: mm1 = dy × V       │  │  V1: softmaxGrad(dy, y)    │      │
│  │  C2: mm2 = Q × K        │  │  V2: pse+mask+softmax      │      │
│  │  V_pre: CopyMaxSum      │  │  V3: ds = drop(P)*dy-sf*P  │      │
│  │                          │  │  V4: p量化 → L1            │      │
│  │                          │  │                             │      │
│  │  [Cube算mm1/mm2时]       │  │  [Vec算V1~V4时]           │      │
│  │  [Vec算V1~V4(前一轮)]    │  │  [Cube算mm3/mm4/mm5]       │      │
│  │                          │  │                             │      │
│  │                          │  │  C3: mm3 = ds × Q → dq    │      │
│  │                          │  │  C4: mm4 = ds × K → dk    │      │
│  │                          │  │  C5: mm5 = p × dy → dv    │      │
│  │                          │  │                             │      │
│  │                          │  │  V5: dq muls+cast → GM     │      │
│  │                          │  │  V6: dk/dv muls+cast → GM  │      │
│  └──────────────────────────┘  └─────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 PingPong双buffer机制

核心思想：**当前轮Cube在算mm1/mm2时，Vector在处理上一轮的mm1/mm2结果**。

```cpp
// kernel_base.h:126-127
TBuf<> mm1ResBuf[2];  // PingPong双buffer
TBuf<> mm2ResBuf[2];  // PingPong双buffer

// 使用方式：
mm1ResTensor = mm1ResBuf[taskId & 1];  // 当前轮用taskId%2索引
prevRunInfo = runInfos[(taskId + 1) & 1];  // 前一轮用(taskId+1)%2索引
```

时间线展开：

```
taskId=0:  Cube: mm1→mm1ResBuf[0], mm2→mm2ResBuf[0]
           Vector: 无前一轮数据（跳过V1~V6）

taskId=1:  Cube: mm1→mm1ResBuf[1], mm2→mm2ResBuf[1]
           Vector: V1~V4 使用 mm1ResBuf[0]/mm2ResBuf[0]（taskId=0的数据）
           Cube: C3~C5 使用 mm1ResBuf[0]/mm2ResBuf[0]（BN2写UB模式）
           Vector: V5~V6 输出dq/dk/dv

taskId=2:  Cube: mm1→mm1ResBuf[0], mm2→mm2ResBuf[0]
           Vector: V1~V4 使用 mm1ResBuf[1]/mm2ResBuf[1]（taskId=1的数据）
           ...

taskId=N:  无新block → isLastLoop=true
           Vector: V1~V4 使用 mm1ResBuf[(N-1)%2]/mm2ResBuf[(N-1)%2]
           Cube: C3~C5 使用前一轮数据
           Vector: V5~V6 输出dq/dk/dv → 结束
```

### 1.4 同步Flag时序图

```
时间 ─────────────────────────────────────────────────────────────────►

Loop N (Cube当前轮):
  AIC: C1_mm1 ──Set──► SYNC_C1_TO_V2_FLAG[N&1] + 16+同步
  AIC: C2_mm2 ──Set──► SYNC_C2_TO_V2_FLAG[N&1] + 16+同步

Loop N+1 (Vector处理前一轮+N+1的Cube):
  AIV: V1_softmaxGrad ──Wait──► SYNC_C1_TO_V2_FLAG[(N)&1] (mm1就绪)
  AIV:                  ──Wait──► SYNC_C2_TO_V2_FLAG[(N)&1] (mm2就绪)
  AIV: V2/V3/V4 完成后:
  AIV: ──Set──► SYNC_V3_TO_C3_FLAG  (ds已到L1)
  AIV: ──Set──► SYNC_V4_TO_C5_FLAG  (p已到L1)

  AIC: ──Wait──► SYNC_V3_TO_C3_FLAG + 16+同步  (等待ds)
  AIC: ──Wait──► SYNC_V4_TO_C5_FLAG + 16+同步  (等待p)

  AIC: C3_mm3(dq) ──Set──► SYNC_C3_TO_V5_FLAG + 16+
  AIC: C4_mm4(dk) ──Set──► SYNC_C4_TO_V6_FLAG + 16+
  AIC: C5_mm5(dv) ──Set──► SYNC_C5_TO_V4_FLAG + 16+

  AIV: ──Wait──► SYNC_C3_TO_V5_FLAG  (dq mm结果就绪)
  AIV: ──Wait──► SYNC_C4_TO_V6_FLAG  (dk mm结果就绪)
  AIV: V5/V6: dq/dk/dv muls+cast → GM输出

  AIC: ──Set──► SYNC_C4_TO_V3_FLAG + 16+  (dk完成，下一轮可复用L0C)
  AIV: ──Set──► SYNC_DETER_FIX_FLAG  (确定性场景标记)
```

### 1.5 1:2核比下的halfS1分割

Regbase架构采用1 Cube + 2 Vector核比。每个基本块`s1×s2`被两个Vector核各分一半：

```cpp
// kernel_base.h:1016-1021 (SetRunInfo中的分割逻辑)
runInfo.commonRunInfo.halfS1RealSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1;
runInfo.commonRunInfo.firstHalfS1RealSize = runInfo.commonRunInfo.halfS1RealSize;
if (vSubBlockIdx == 1) {
    runInfo.commonRunInfo.halfS1RealSize =
        runInfo.commonRunInfo.s1RealSize - runInfo.commonRunInfo.halfS1RealSize;
}

// 例：s1RealSize=128
// V0(vSubBlockIdx=0): halfS1RealSize=64, firstHalfS1RealSize=64
// V1(vSubBlockIdx=1): halfS1RealSize=64, firstHalfS1RealSize=64

// 例：s1RealSize=33
// V0: halfS1RealSize=17, firstHalfS1RealSize=17
// V1: halfS1RealSize=16, firstHalfS1RealSize=17
```

**Fixpipe的dualDstCtl=1机制**：Cube的mm1/mm2 Fixpipe时使用`dualDstCtl=1`，将L0C结果按M维度拆成两半（M/2×N），分别写入mm1ResBuf/mm2ResBuf的UB空间中V0和V1各自的区域。

```cpp
// block_cube.h:519 (mm1 Fixpipe)
fixpipeParams.mSize = (s1RealSize + 1) >> 1 << 1;  // 必须为偶数
fixpipeParams.dstStride = CUBE_BASEN;               // UB行间距128
fixpipeParams.dualDstCtl = 1;                        // 双目标模式
```

---

## 第二部分：Cube侧5次MatMul详细解释

### 2.1 mm1: Dy × V → P_part（计算dy与V的乘积）

**数学公式**：`P_part[s1, s2] = dy[s1, d_v] × V[s2, d_v]^T`

**矩阵维度**：
| 维度 | 含义 | 大小 |
|------|------|------|
| M | s1（query序列长度） | s1RealSize（最大CUBE_BASEM=128） |
| N | s2（key序列长度） | s2RealSize（最大CUBE_BASEN=128） |
| K | d_v（value维度） | dSizeV（HEAD_DIM_ALIGN对齐） |

**代码位置**：`block_cube.h:345-530 IterateMmDyV()`

**详细流程**：

#### 步骤1：加载左矩阵dy到L1

```
左矩阵 = dy（shape: s1 × d_v）

L1 Buffer选择策略：
├── IS_L1_PRELOAD: 
│   ├── 首轮(preloadArgs.copyCurrent=true): dYL1Buf.Get() 
│   ├── 非首轮: dYL1Buf.GetPre()（使用preload的前一轮buffer）
│   └── 有下一轮: dYL1Buf.Get()（用于preload下一轮dy）
│
├── IS_L1_REUSE: dYL1Buf.Get()（buffer固定，仅首次拷入）
│
├── IS_FP32_D_EXCEED_256: fp32L1Buf1.Get()（FP32+D>256时专用）
│
└── 普通模式: commonL1Buf.Get()（共享L1 buffer，按需复用）

DataCopy: dyGm[dyOffset] → dyL1Tensor (Nd2Nz格式)
  nd2NzParams.nValue = s1RealSize       // 实际行数
  nd2NzParams.dValue = dSizeV           // 实际列数
  nd2NzParams.srcDValue = mm1Ka         // GM stride
  nd2NzParams.dstNzC0Stride = AlignTo16(s1RealSize)  // L1上NZ C0 stride
```

**L1 Preload机制**（IS_L1_PRELOAD=true时）：
- 当前轮拷入dy的同时，下一轮的dy也在拷入另一个L1 buffer
- 使用Mutex（`vL1BufMutexId`）保护V buffer的并发访问
- `preloadArgs.copyCurrent/copyNext/nextMOrN/nextDyOffset` 控制preload逻辑

#### 步骤2：循环加载右矩阵V到L1 + MMad + Fixpipe

```
// FP32+D>512时，s2需要切两轮（baseN=CUBE_BASEN/2=64）
// 其他情况一轮（baseN=CUBE_BASEN=128）
nLoops = (s2RealSize + baseN - 1) / baseN;

for (n = 0; n < nLoops; ++n) {
    realN = (n==nLoops-1) ? (s2RealSize % baseN || baseN) : baseN;
    
    ┌─── 加载V到L1 ───┐
    │ IS_L1_REUSE/IS_L1_PRELOAD: 
    │   isCopyRight = !runInfo.isS2IdxNoChange  // s2块未变则不重拷
    │   vL1Buffer = vL1Buf.Get()
    │ IS_FP8_INPUT: Mutex Lock保护
    │
    │ if (isCopyRight):
    │   DataCopy: valueGm[valueOffset + gmNOffset] → vL1Tensor
    │   nd2NzParams.nValue = realN          // 当前s2实际大小
    │   nd2NzParams.dstNzC0Stride = AlignTo16(realN) // L1 NZ stride
    │ IS_FP8_INPUT: Mutex Unlock
    
    ┌─── L1→L0A/L0B + MMad ───┐
    │ mm1L0CBuffer.Wait(FIX_M)  // 反向同步：等待L0C可用
    │ MMParam:
    │   singleM = s1RealSize
    │   singleN = realN
    │   singleK = dSizeV
    │   isLeftTranspose = false
    │   isRightTranspose = true  // V按KN布局
    │   UNITFLAG_ENABLE = true   // 单元标志使能
    │
    │ MatmulBase(dyL1, vL1, l0aBuf, l0bBuf, mm1L0C, param)
    │   // 左矩阵从dyL1Buffer→L0A，右矩阵从vL1Buffer→L0B
    │   // 结果写入mm1L0CBuffer（float类型）
    │
    │ mm1L0CBuffer.Set(M_FIX) → Wait(M_FIX)  // 等待MMad完成
    
    ┌─── Fixpipe: L0C → UB ───┐
    │ FixpipeParams:
    │   nSize = realN
    │   mSize = (s1RealSize+1)>>1<<1  // 必须偶数，dualDstCtl需要
    │   srcStride = AlignTo16(mSize)   // L0C NZ stride
    │   dstStride = CUBE_BASEN         // UB行间距（128列对齐）
    │   dualDstCtl = 1                 // 双目标：M/2×N各写入V0/V1的UB区域
    │
    │ Fixpipe(mm1ResTensor[ubOffset], mm1L0C, fixpipeParams)
    │   // mm1ResTensor来自mm1ResBuf[taskId&1]
    │   // L0C结果搬运到UB的mm1ResBuf中
    │   // dualDstCtl=1：L0C的M方向拆成两半，
    │   //   前 halfS1 行写入 mm1ResTensor[0:CUBE_BASEN*halfS1]（V0区域）
    │   //   后 halfS1 行写入 mm1ResTensor[CUBE_BASEN*halfS1:]（V1区域）
    
    ┌─── L1 Preload下一轮dy ───┐
    │ IS_L1_PRELOAD && preloadArgs.copyNext:
    │   DataCopy: dyGm[nextDyOffset] → dyL1NextBuffer
    │   // 当前循环不wait，下轮循环开始时wait
    
    mm1L0CBuffer.Set(FIX_M)  // 释放L0C供下一轮使用
}
```

**Quant场景（IterateMmDyVQuant）差异**：
- dy在L1上类型为`OUTDTYPE`（不是`INPUT_TYPE`），因为quant场景下dy有独立的量化scale
- V的L1 buffer也使用`OUTDTYPE`
- 其他流程与非Quant版本相同

---

### 2.2 mm2: Q × K → S（重算attention score）

**数学公式**：`S[s1, s2] = Q[s1, d] × K[s2, d]^T · scale`

**矩阵维度**：
| 维度 | 含义 | 大小 |
|------|------|------|
| M | s1 | s1RealSize |
| N | s2 | s2RealSize |
| K | d（head维度） | dSize（HEAD_DIM_ALIGN对齐） |

**代码位置**：`block_cube.h:668-893 IterateMmQK()`

**详细流程**与mm1类似，关键差异：

#### 差异1：左矩阵为Q，右矩阵为K

```
左矩阵 = Q（shape: s1 × d）
  queryGm[queryOffset] → qL1Tensor
  IS_L1_REUSE: qL1Buf.Get()
  IS_L1_PRELOAD: qL1Buf.Get()/qL1Buf.GetPre()

右矩阵 = K（shape: s2 × d）
  keyGm[keyOffset + gmNOffset] → kL1Tensor
  IS_L1_REUSE: kL1Buf.GetReused(runInfo.isS2IdxNoChange)
    // K在s2不变时可复用上一轮的L1数据
```

#### 差异2：Rope场景的特殊处理

Rope（旋转位置编码）场景下，Q和K各分为两段拼接：

```cpp
// block_cube.h:702-720 (Rope Q拷入)
if constexpr (IS_ROPE) {
    // 第一段：基础Q（前128列，d=ROPE_D_128）
    nd2NzParams.dValue = ROPE_D_128;     // 128
    nd2NzParams.srcDValue = mm2Ka;       // 全stride（192对齐）
    nd2NzParams.dstNzC0Stride = AlignTo16(s1RealSize);
    DataCopy(qL1Tensor, queryGm[queryOffsetWithRopeForMm12], nd2NzParams);
    
    // 第二段：Rope部分（后64列，d=ROPE_D_64）
    nd2NzParams.dValue = ROPE_D_64;      // 64
    nd2NzParams.srcDValue = mm2Ka / NUM_TWO;  // 半stride
    // 拼接到第一段后面
    DataCopy(qL1Tensor[dstNzC0Stride * ROPE_D_128], queryRopeGm[qRopeOffset], nd2NzParams);
}

// Rope场景mm2Ka/mm2Kb的计算（kernel_base.h:674-676）
// mm2Ka = mm2Ka / 3 << 1 = 原Ka * 2/3
// 例：原始Ka=192，mm2Ka = 192/3*2 = 128（基础Q占128列，Rope占64列）
// 实际效果：L1上的Q布局 = [128列基础Q | 64列Rope]，总Ka=192对齐stride
```

K的拷入同理，分为基础K（128列）和RopeK（64列）。

#### 差异3：MMad参数

```
MMParam:
  singleK = dSize   // 非Rope时用dSize，Rope时已通过L1拼接处理
  isRightTranspose = true  // K按KN布局
```

#### 差异4：K的L1复用策略

```cpp
// block_cube.h:754-767 (K L1 buffer选择)
IS_L1_REUSE || IS_L1_PRELOAD:
  isCopyRight = !runInfo.isS2IdxNoChange
  if (isCopyRight):
    kL1Buffer = kL1Buf.Get()     // 需要重新拷入
  else:
    kL1Buffer = kL1Buf.GetPre()  // 复用上一轮的K数据（s2块相同）
```

**isS2IdxNoChange**的判断逻辑：当连续两个block的`s2oIdx`相同（且b/n2维度也相同），K矩阵在GM上的位置不变，可以跳过L1拷入。

---

### 2.3 mm3: dS × Q → dq（计算query梯度）

**数学公式**：`dq[s2, d] = dS[s2, s1] × Q[s1, d]^T`

注意：这里dS是`s2×s1`形状（因为Vec3中已经做了转置排列），Q是`s1×d`形状。

**矩阵维度**：
| 维度 | 含义 | 大小 |
|------|------|------|
| M | s2 | s2RealSize |
| N | d（head维度） | dSize |
| K | s1 | s1RealSize |

**代码位置**：`block_cube.h:973-... IterateMmDsK()`

**详细流程**：

#### 步骤1：等待FP8 ssbuf数据（仅FP8场景）

```cpp
// block_cube.h:986-987
if constexpr (IS_FP8_INPUT) {
    CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);  // 等待ssbuf中的qScaleDs
}
```

FP8场景下，Vector侧在ProcessVec3中计算了ds的量化scale（qScaleDs），并通过ssbuf传递给Cube侧。Cube需要先Wait这个信号才能开始mm3。

#### 步骤2：循环切N轴（D维度切分）

```
// D维度可能超过L1单buffer容量，需要切分
baseN = l1BaseD;  // L1一次能容纳的D维度大小
nLoops = (dSize + baseN - 1) / baseN;

for (n = 0; n < nLoops; ++n) {
    realN = (n==nLoops-1) ? (dSize % baseN || baseN) : baseN;
```

#### 步骤3：加载左矩阵dS和右矩阵Q到L1

```
左矩阵 = dS（shape: s2 × s1，NZ格式）
  来源：dSL1Buf（Vector侧ProcessVec3已经将ds从UB拷入L1）
  dSL1Buffer = dSL1Buf.Get()
  
  注意：dS已经在L1中了！不需要从GM拷入。
  Vector侧在ProcessVec3→CopyUB2L1Vec3中已完成 UB→L1 的拷入。
  Cube侧只需Wait SYNC_V3_TO_C3_FLAG 确认ds已就绪。

右矩阵 = Q（shape: s1 × d）
  IS_L1_REUSE/IS_L1_PRELOAD:
    qL1Buffer = qL1Buf.Get()（复用mm2的Q L1 buffer！）
    // 关键优化：mm3的Q就是mm2的Q，可以复用L1数据
  普通模式:
    qL1Buffer = commonL1Buf.Get()
  
  DataCopy: queryGm[queryOffset + gmNOffset] → qL1Tensor
```

#### 步骤4：MMad + Fixpipe

```
mm3L0CBuffer选择：
├── IS_DKV_RESIDENT_L0C: mm1Mm2Mm3L0CBuf.Get()  // dk/dv常驻L0C
├── isDkvL0CResidentForD192Dv128: mm1Mm2Mm3L0CSpecialBuf.Get()
└── 其他: commonl0CBuf.Get()

MMParam:
  singleM = s2RealSize  // M轴是s2！
  singleN = realN        // N轴是D的切分段
  singleK = s1RealSize   // K轴是s1
  isLeftTranspose = false  // dS按MK布局
  isRightTranspose = true  // Q按KN布局

MatmulBase(dSL1, qL1, l0aBuf, l0bBuf, mm3L0C, param)
```

#### 步骤5：Fixpipe输出（IS_WRITE_UB vs !IS_WRITE_UB）

**IS_WRITE_UB=true（BN2+非多block模式）**：

```
// dq结果Fixpipe到UB（mm1ResBuf复用）
FixpipeParams:
  nSize = (realN + 7) >> 3 << 3  // 8元素对齐（32B）
  mSize = (s1RealSize+1)>>1<<1   // dualDstCtl需要偶数
  dstStride = AlignTo16(dSize)    // UB行间距
  dualDstCtl = 1                  // 双目标模式

Fixpipe(mm1ResTensor[gmNOffset], mm3L0C, fixpipeParams)
  // mm1ResTensor此时已被V3消耗完，可以复用给dq
  // dq的L0C结果写入UB，后续V5做muls+cast
```

**IS_WRITE_UB=false（其他模式）**：

```
// dq结果Fixpipe直接到GM
FixpipeParams:
  nSize = realN
  mSize = s1RealSize       // 不需要dualDstCtl
  dstStride = mm3Ka        // GM stride
  dualDstCtl = 0           // 单目标模式

// 需要AtomicAdd的场景（BN2/BN2S2多核重叠）
needAtomic = (SPLIT_AXIS != BN2) || (IS_BN2_MULTIBLK && !isFirstS1Outer)
if (needAtomic):
  SetAtomicAdd<CALC_TYPE>()  // 开启原子加

// FP8场景额外处理
if IS_FP8_INPUT:
  // 从ssbuf读取sharedParams（qScaleDs）
  // fixpipeParams.quantPre = QuantMode_t::QF322F32_PRE
  // fixpipeParams.deqScalar = qScaleDs的量化参数

Fixpipe(dqWorkSpaceGm[offset + gmNOffset], mm3L0C, fixpipeParams)

if (needAtomic):
  SetAtomicNone()  // 关闭原子加
```

**BN2模式dq的GM offset计算**：

```cpp
// block_cube.h:959 (非BN2模式)
offset = SPLIT_AXIS == BN2 ? GetBlockIdx() * CUBE_BASEM * HEAD_DIM_ALIGN : queryOffset;

// block_cube.h:961 (BN2_MULTIBLK模式)
offset = GetBlockIdx() * AlignTo128(s1Size) * HEAD_DIM_ALIGN;
offset += s1oIdx * CUBE_BASEM * HEAD_DIM_ALIGN;
```

---

### 2.4 mm4: dS × K → dk（计算key梯度）

**数学公式**：`dk[s2, d] = dS[s2, s1] × K[s1, d]^T`

**矩阵维度**：
| 维度 | 含义 | 大小 |
|------|------|------|
| M | s2 | s2RealSize |
| N | d | dSize |
| K | s1 | s1RealSize |

**代码位置**：`block_cube.h: IterateMmDsQ()`

**与mm3的差异**：

1. **右矩阵为K而非Q**：
   - `kL1Buffer = kL1Buf.GetReused(runInfo.isNextS2IdxNoChange)`
   - K可以复用mm2的K L1 buffer，且下一轮s2不变时可以跳过拷入

2. **MM参数中K轴stride不同**：
   - `mm4Kb = mm2Kb`（K的stride与mm2相同）

3. **IS_DK_WRITE_UB模式**：
   - BN2+非多block模式：dk Fixpipe到UB（mm2ResBuf复用）
   - 其他模式：dk Fixpipe到GM（dkWorkSpaceGm）

4. **BN2S2+IS_DK_WRITE_UB模式**：
   - dk可以写入mm1ResBuf（因为此时mm1已被V3消耗，mm1ResBuf可复用）
   - 需要额外的SYNC_C4_TO_V3_FLAG同步（dk mm完成，下一轮需要使用dkL0C）

5. **Rope场景差异**：
   - mm4中K不需要分两段（Rope部分已在mm2中处理）
   - 但IS_ROPE时mm4Kb也需要 `mm2Kb / 3 << 1` 的stride调整

6. **L0C Resident策略**：
   - IS_DKV_RESIDENT_L0C模式：dk结果写入dkL0CBuf（常驻L0C），跨s2o累加
   - 累加完成后才Fixpipe到GM

---

### 2.5 mm5: P × dy → dv（计算value梯度）

**数学公式**：`dv[s2, d_v] = P[s1, s2]^T × dy[s1, d_v]`

**矩阵维度**：
| 维度 | 含义 | 大小 |
|------|------|------|
| M | s2 | s2RealSize |
| N | d_v | dSizeV |
| K | s1 | s1RealSize |

**代码位置**：`block_cube.h: IterateMmPDy()`

**与mm3/mm4的差异**：

1. **左矩阵为P而非dS**：
   - P来自pL1Buf（Vector侧ProcessVec4已将p从UB拷入L1）
   - Cube需要Wait SYNC_V4_TO_C5_FLAG确认p已就绪

2. **右矩阵为dy而非Q/K**：
   - dyGm[dyOffset] → dyL1Buffer
   - IS_L1_REUSE: dYL1Buf.Get()（复用mm1的dy L1 buffer）

3. **N轴使用dSizeV而非dSize**：
   - dv的输出维度是d_v（可能与d不等，IS_D_NO_EQUAL场景）

4. **IS_DV_WRITE_UB模式**（仅BN2S2+非确定性）：
   - dv Fixpipe到UB（mm2ResBuf复用）
   - 后续V6做muls+cast

5. **非IS_DV_WRITE_UB模式**：
   - dv Fixpipe直接到GM（dvGm/dvWorkSpaceGm）
   - BN2模式需要AtomicAdd（多核重叠写dv）

6. **L0C Resident策略**：
   - IS_DKV_RESIDENT_L0C: dv结果写入dvL0CBuf（常驻L0C），跨s2o累加

---

### 2.6 Cube侧Buffer管理总结

```
                    ┌─── L1 Buffer生命周期 ───┐
                    │                          │
mm1(mm2共用):       │  dYL1Buf ──── dy        │  IS_L1_REUSE/PRELOAD时固定
mm2:                │  qL1Buf ──── Q          │  IS_L1_REUSE/PRELOAD时固定
mm2:                │  kL1Buf ──── K          │  IS_L1_REUSE/PRELOAD时固定
mm1:                │  vL1Buf ──── V          │  IS_L1_REUSE/PRELOAD时固定
mm3/mm4:            │  dSL1Buf ─── ds         │  Vector侧CopyUB2L1写入
mm5:                │  pL1Buf ──── p          │  Vector侧CopyUB2L1写入
mm3/mm4:            │  qL1Buf ──── Q（复用mm2）│  L1_REUSE优化
mm4:                │  kL1Buf ──── K（复用mm2）│  L1_REUSE优化
mm5:                │  dYL1Buf ── dy（复用mm1）│  L1_REUSE优化

                    ┌─── L0C Buffer生命周期 ───┐
                    │                           │
IS_DKV_RESIDENT:    │  mm1Mm2Mm3L0CBuf ── mm1/mm2/mm3  │  三mm共享
                    │  dkL0CBuf ──────── mm4(dk)        │  dk常驻累加
                    │  dvL0CBuf ──────── mm5(dv)        │  dv常驻累加
普通模式:            │  commonl0CBuf ──── 所有mm轮流使用 │  每mm用完释放

                    ┌─── UB Buffer（CV共享）───┐
                    │                          │
PingPong:           │  mm1ResBuf[0]/[1] ── mm1结果/dq │  双buffer交替
PingPong:           │  mm2ResBuf[0]/[1] ── mm2结果/dk │  双buffer交替
```

---

## 第三部分：Vector侧6个阶段详细解释

### 3.1 VF1: ProcessVec1 — SoftmaxGrad前向

**数学公式**：
```
softmaxGradRes = Σ(dy × y)  沿s2轴求和
即：对每个s1位置，计算 dy[i] × y[i] 的s2方向归约和
```

**代码位置**：`block_vec.h:312-342 ProcessVec1()`

**详细流程**：

```
ProcessVec1(constInfo, runInfo)
│
├── if (halfS1RealSize == 0) return;  // V1核无数据则跳过
│   // 例：s1RealSize=1时，V0处理1行，V1处理0行
│
├── softmaxGradResTensor = softmaxGradResBuf.Get<CALC_TYPE>()
│   // 大小：VECTOR_BASEM × sizeof(CALC_TYPE) = 64×4B = 256B
│
├── if (HEAD_DIM_ALIGN ≤ VECTOR_BASEN):  // D ≤ 128，一次处理
│   │
│   ├── CopyInSoftmaxGrad(dy, y → attenMaskOrYInQue, pseOrDyInQue)
│   │   // 拷入dy（从dyGm）和y（从yGm）到UB
│   │   // 只拷入 halfS1RealSize 行（每个V核只处理自己的s1半区）
│   │   // dyOffset包含vSubBlockIdx的偏移（vecCoreOffset）
│   │
│   ├── CalculateCastSoftmaxGrad(dy, y → softmaxGradResTensor)
│   │   // Step 1: Cast dy从OUTDTYPE→CALC_TYPE
│   │   // Step 2: Cast y从OUTDTYPE→CALC_TYPE  
│   │   // Step 3: Mul: dy_cast × y_cast = dy_mul_y
│   │   // Step 4: SoftmaxGradFront: 沿s2轴ReduceSum
│   │   //   softmaxGradRes[s1_idx] = Σ(dy_mul_y[s1_idx, :])
│   │   //   输出shape: [halfS1RealSize]
│
├── else:  // D > 128，需要切分循环
│   │
│   ├── loopNum = Ceil(halfS1RealSize, sfmgMaxLoopSize)
│   │   // sfmgMaxLoopSize = VECTOR_BASEM * VECTOR_BASEN / HEAD_DIM_ALIGN
│   │   // 例：64×128/256 = 32，每次最多处理32行
│   │
│   ├── for (loopIdx = 0; loopIdx < loopNum; loopIdx++):
│   │   CopyInSoftmaxGrad(curLoopSize行) 
│   │   CalculateCastSoftmaxGrad(curLoopSize行 → softmaxGradResTensor[loopIdx*loopSize])
```

**softmaxGradResTensor的使用**：
- 后续在VF3（ProcessVec3）中被用来计算 `ds = dropout(P) × dy - softmaxGradRes × softmax(P)`
- softmaxGradRes是`s1维度的向量`，在VF3中会广播到`s1×s2`的矩阵

---

### 3.2 VF2: ProcessVec2 — pse + attenMask + SimpleSoftmax

**数学公式**：
```
S_masked = S + pse + attenMask  （加偏置和mask）
P = softmax(S_masked)           （重算attention probability）
```

**代码位置**：`block_vec.h:455-472 ProcessVec2()`

**详细流程**：

```
ProcessVec2(mm2ResTensor, constInfo, runInfo)
│
├── IS_FP8_INPUT: 
│   Muls(mm2ResTensor, mm2ResTensor, deqScaleQ * deqScaleK, VECTOR_BASEM*VECTOR_BASEN)
│   // FP8场景：mm2结果需要乘以量化scale（deqScaleQ × deqScaleK）
│   // 因为Q和K各有一个dequantization scale，mm2结果是两者乘积的量化
│
├── CopyInAttenMask(attenMaskU8Gm → attenMaskOrYInQue)
│   // 从GM拷入attenMask到UB
│   // attenMask是uint8类型（压缩格式）或float类型
│   // 只拷入当前block的s1×s2区域
│   // 根据sparseMode计算mask的有效范围（s2SparseLeft/s2SparseRight）
│
├── CopyInPse(pseGm → pseOrDyInQue)
│   // 从GM拷入pse（位置编码偏置）到UB
│   // pse有多种shape：BNHS/1NSS/1NHS/BNSS等
│   // 根据pseShapeType和pseLayoutType确定GM offset
│
├── CalculatePseMulsSelSimpleSoftMax(mm2Res → mm2Res)
│   │
│   ├── if IS_PSE: 加pse偏置
│   │   // mm2Res = mm2Res + pse
│   │
│   ├── if IS_ATTEN_MASK: 应用mask
│   │   // 无效区域填充 -∞（attenMaskMinValue = 0xFF7FFFFF ≈ -3.4e38）
│   │   // Select操作：mask=0的位置保留原值，mask=1的位置填-∞
│   │
│   ├── Muls(mm2Res, mm2Res, scaleValue)
│   │   // mm2Res = mm2Res × scale（softmax的温度缩放）
│   │   // scaleValue = 1/sqrt(d) 或用户指定值
│   │
│   ├── SimpleSoftmax计算：
│   │   // 使用前向传播保存的softmaxMax和softmaxSum
│   │   // softmaxMax: 每个s1位置的s2方向最大值（用于数值稳定）
│   │   // softmaxSum: 每个s1位置的s2方向softmax归约和
│   │   │
│   │   // Step 1: 从maxSumQue获取max和sum
│   │   //   max = softmaxMaxGm[s1_offset]（前向传播保存的）
│   │   //   sum = softmaxSumGm[s1_offset]
│   │   │
│   │   // Step 2: Sub: score - max = score_shifted
│   │   //   mm2Res = mm2Res - max  （减去最大值防止exp溢出）
│   │   │
│   │   // Step 3: Exp: exp(score_shifted)
│   │   //   mm2Res = exp(mm2Res)
│   │   │
│   │   // Step 4: Div: exp_result / sum = softmax_result
│   │   //   mm2Res = mm2Res / sum
│   │   │
│   │   // 输出：mm2Res = P = softmax(S + pse + mask) × scale
│   │   //   shape: [halfS1RealSize × s2RealSize]
│   │   //   存储在mm2ResBuf[taskIdMod2]中
```

**SimpleSoftmax vs 标准Softmax**：
- 标准Softmax需要先ReduceMax再Exp再ReduceSum再Div，四步串行
- SimpleSoftmax利用前向传播已保存的max和sum，直接做 `exp(score-max)/sum`，三步即可
- 优势：减少Vector侧计算量，避免二次ReduceMax/ReduceSum

---

### 3.3 VF3: ProcessVec3 — 计算 ds（核心梯度）

**数学公式**：
```
ds = dropout(P) × dy - softmaxGradSum × softmax(P)

展开：
ds[s1,s2] = drop_mask[s1,s2] × P[s1,s2] × dy[s1,s2] 
           - softmaxGradRes[s1] × P[s1,s2]
```

这是FAG最关键的Vector计算，产生ds梯度供后续mm3/mm4使用。

**代码位置**：`block_vec.h:544-645 ProcessVec3()`

**详细流程**：

```
ProcessVec3(dSL1Buffer, mm1Res, mm2Res, constInfo, runInfo, isDqNeedDeter)
│
├── Step 1: Dropout处理
│   ├── if dropMaskOuter: 
│   │   CopyInDropOuter(dropMaskGm → dropMaskBuf)  // 从GM拷入
│   ├── else:
│   │   GenDropMask(dropMaskBuf)  // 在线生成（根据seed/offset/keepProb）
│   │
│   ├── CalculateDropout(mm1Res → mm1Res)
│   │   // mm1Res = mm1Res × dropoutMask
│   │   // dropoutMask为1的位置保留，为0的位置置零
│   │   // IS_DROP=false时，dropoutMask全为1（无dropout）
│
├── Step 1b: Sink处理（IS_DROP场景）
│   ├── CopyDpToTmpBuf(mm1Res → attenMaskOrYInQue)
│   │   // 保存dropout(P)×dy的中间结果，供ProcessVecSink使用
│
├── Step 2: 核心 ds 计算
│   │
│   ├── softmaxGradResTensor = softmaxGradResBuf.Get()
│   │   // 来自VF1的softmaxGradSum结果
│   │
│   ├── vecOutBuffer = dSOutQue.AllocTensor<INPUT_TYPE>()
│   │   // ds的输出buffer
│   │
│   ├── BroadcastSubMul(mm1Res, softmaxGradRes, mm2Res → mm1Res)
│   │   // mm1Res = mm1Res - softmaxGradRes × mm2Res
│   │   // = dropout(P)×dy - softmaxGradSum × softmax(P)
│   │   // = ds
│   │   │
│   │   // 模板化s2大小（性能优化）：
│   │   // IS_FP8_INPUT: 固定VECTOR_BASEN模板
│   │   // s2RealSize > 64: Aligned128模板（更宽的向量指令）
│   │   // s2RealSize ≤ 64 且 noNeedDeter: Aligned64模板
│   │   // s2RealSize ≤ 64 且 需deter: Aligned64模板 + 清零64~128区域
│   │
│   │   // BroadcastSubMul内部：
│   │   //   Sub: mm1Res - softmaxGradRes（广播s1维度到s1×s2）
│   │   //   Mul: result × mm2Res（不需要，已经在Sub中完成）
│   │   //   实际实现：mm1Res = mm1Res - softmaxGradRes * mm2Res
│
├── Step 2b: 确定性尾块补零
│   ├── IS_DETER_OLD && !noNeedDeter:
│   │   if (halfS1RealSize != VECTOR_BASEM):
│   │     Duplicate(mm1Res[halfS1RealSize*VECTOR_BASEN:], 0, 
│   │               (VECTOR_BASEM-halfS1RealSize)*VECTOR_BASEN)
│   │     // 尾块s1不足64行时，补零到64行
│   │     // 防止后续mm3/mm4有脏数据参与计算
│
├── Step 3: FP8量化（仅FP8场景）
│   │
│   ├── DsAbsReduceMax(dsAmaxTensor, mm1Res, halfS1RealSize, s2RealSize)
│   │   // 计算ds矩阵的绝对值最大值
│   │   // 输出：dsAmaxTensor[0] = max(|ds[i,j]|)
│   │
│   ├── V→S同步 + 写GM:
│   │   SetFlag(V→S) → WaitFlag(V→S)
│   │   curDsMax = dsAmaxTensor.GetValue(0)
│   │   DataCopyPad(dsAmaxWorkSpaceGm[vBlockIdx*128], dsAmaxTensor)
│   │
│   ├── V0/V1交换amax:
│   │   SetCrossCore(SYNC_V0_V1_DS_A_MAX_DONE_FLAG)  // 通知对方
│   │   WaitCrossCore(SYNC_V0_V1_DS_A_MAX_DONE_FLAG)  // 等待对方
│   │   SetFlag(MTE3→S) → WaitFlag(MTE3→S)
│   │   DataCacheCleanAndInvalid(dsAmaxWorkSpaceGm[anotherVBlockIdx*128])
│   │   dsAmax = Max(curDsMax, dsAmaxWorkSpaceGm[anotherVBlockIdx*128])
│   │
│   ├── 计算量化scale:
│   │   if (dsAmax > 1e-6f):
│   │     qScaleDs = FP8_MAX / dsAmax  // 例：fp8_e4m3: 448/dsAmax
│   │     Muls(mm1Res, mm1Res, qScaleDs)  // 量化：ds = ds × qScaleDs
│   │
│   ├── 传递qScaleDs给Cube:
│   │   InitCubeVecSharedParams(sharedParams, cBlockIdx, vSubBlockIdx, qScaleDs)
│   │   // 写入ssbuf（Cube侧通过WaitCrossCore(15)读取）
│   │   // sharedParams.qScaleDs = qScaleDs
│   │
│   ├── CastTransdataDeconflict(mm1Res → vecOutBuffer)
│   │   // Cast: CALC_TYPE → INPUT_TYPE（fp32 → fp16/bf16/fp8）
│   │   // TransdataDeconflict: NZ格式转换（s1×s2 → NZ fractal布局）
│   │   // vselrIndexesTensor: FP8场景的索引向量
│
├── Step 4: FP32场景额外处理
│   ├── IS_FP32_INPUT:
│   │   Muls(mm1Res, mm1Res, scaleValue)  // FP32不做mm内scale，需在Vec侧补乘
│
├── Step 5: CopyUB2L1 — ds从UB拷入L1
│   │
│   ├── CopyUBToL1Vec3(dSL1Buffer, vecOutBuffer, runInfo, isDqNeedDeter)
│   │   │
│   │   ├── dSOutQue.EnQue(vecOutBuffer) → DeQue
│   │   │
│   │   ├── CopyUB2L1(dsL1Tensor, vecOutBuffer)
│   │   │   // DataCopy: UB → L1
│   │   │   // NZ格式的特殊stride处理
│   │   │   // scmOffset = vSubBlockIdx==0 ? 0 : firstHalfS1RealSize * C0_SIZE
│   │   │   // V0的数据放在L1前半区，V1的数据放在L1后半区
│   │   │   // 这样Cube侧mm3/mm4可以同时读到V0和V1的ds数据
│   │   │
│   │   ├── IS_DETER_OLD && isDqNeedDeter:
│   │   │   CopyUB2L1Deter(runInfo, dsL1Tensor, vecOutBuffer)
│   │   │   // 确定性场景：ds需要包含完整的CUBE_BASEM行（包括补零行）
│   │
│   │   ├── dSOutQue.FreeTensor(vecOutBuffer)
│
├── Step 6: Sink输出（可选）
│   ├── IS_SINK && IS_DROP: ProcessVecSink(mm1Res, mm2Res)
│   │   // 计算dsink梯度
```

**CopyUB2L1的关键细节**：

```cpp
// block_vec.h:352-380 CopyUB2L1<IS_DQ>()
// ds从UB拷入L1的NZ格式处理

scmOffset = vSubBlockIdx==0 ? 0 : firstHalfS1RealSize * FRACTAL_NZ_C0_SIZE;
// V0放在L1偏移0处，V1放在L1偏移firstHalfS1RealSize*C0处
// 这样Cube侧读L1时，V0和V1的数据已经拼在一起

dataCopyParams.blockCount = VECTOR_BASEN / FRACTAL_NZ_C0_SIZE;  // s2方向块数
dataCopyParams.blockLen = halfS1RealSize * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM;
dataCopyParams.srcStride = (VECTOR_BASEM+1-halfS1RealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM;
dataCopyParams.dstStride = (AlignTo16(s1RealSize)-halfS1RealSize) * FRACTAL_NZ_C0_SIZE / INPUT_BLOCK_NUM;
```

---

### 3.4 VF4: ProcessVec4 — P量化/格式转换

**数学公式**：将softmax结果P从float转为输入类型（fp16/bf16/fp8），并转为NZ格式供mm5使用。

**代码位置**：`block_vec.h:726-748 ProcessVec4()`

**详细流程**：

```
ProcessVec4(pL1Buffer, mm2Res, constInfo, runInfo)
│
├── CalculateDropout(mm2Res → mm2Res)
│   // P也需要做dropout（quant场景下p被量化）
│   // IS_DROP=false时无实际效果
│
├── CastTransdataDeconflict(mm2Res → vecOutBuffer1)
│   // Cast: CALC_TYPE → OUTDTYPE（float → fp16/bf16/fp8）
│   // TransdataDeconflict: NZ格式转换
│
├── pOutQue.EnQue(vecOutBuffer1) → DeQue
│
├── CopyUB2L1Quant<false>(pL1Tensor, vecOutBuffer1)
│   // P从UB拷入L1（Quant场景的专用拷入函数）
│   // shape = S1 × S2（IS_MM1=false表示不是mm1的格式）
│   // scmOffset同样区分V0/V1
│
├── pOutQue.FreeTensor(vecOutBuffer1)
```

**非Quant场景（ProcessVec4普通版）**：与Quant版类似，但使用`CopyUB2L1<false>`而非`CopyUB2L1Quant`。

**P的L1用途**：mm5 = P^T × dy，P需要从L1读入L0B参与MatMul。

---

### 3.5 VF5: ProcessMulsAndCast<DQ> — dq最终输出

**数学公式**：
```
dq_final = dq_mm_result × scale  → cast to output dtype → write GM
```

mm3的结果（dq）需要乘以scaleValue（1/√d或用户指定），然后从CALC_TYPE转为OUTDTYPE，最后写入GM。

**代码位置**：`block_vec.h:750-814 ProcessMulsAndCast<DQ>()` / `DqkvMulsAndCastFromUB/DqkvMulsAndCastFromGM`

**两种模式**：

#### 模式A：IS_DQ_WRITE_UB=true（BN2+非多block）

```
DqkvMulsAndCastFromUB(constInfo, runInfo, inputTensor, outQue)
│
├── if (halfS1RealSize == 0) return;  // V核无数据
│
├── Muls(inputTensor, inputTensor, scaleValue, dataSize)
│   // dq = dq × scale
│   // dataSize = halfS1RealSize × AlignTo16(dSize)
│   // 注意：这里dataSize包含对齐填充的数据，muls会对齐区域也乘scale
│   // 但对齐区域后续不会写入GM（DataCopyPad只写有效数据）
│
├── dqkvCastTensor = outQue.AllocTensor<OUTDTYPE>()
│
├── Cast(dqkvCastTensor, inputTensor, CAST_ROUND, dataSize)
│   // CALC_TYPE → OUTDTYPE（float → fp16/bf16）
│
├── outQue.EnQue(dqkvCastTensor) → DeQue
│
├── DataCopyPad(dqGm[dqkvGmOffset], dqkvCastTensor, intriParamsOut)
│   // intriParamsOut.blockCount = halfS1RealSize  // 写入行数
│   // intriParamsOut.blockLen = dSize * sizeof(OUTDTYPE)  // 每行长度
│   // intriParamsOut.dstStride = 根据layout计算
│   │
│   // GM offset计算（BN2模式）：
│   // BNGSD: dqkvGmOffset += vSubBlockIdx * firstHalfS1RealSize * dSize
│   //   V0从offset开始写halfS1行，V1从offset+firstHalfS1*d开始写
│   // SBNGD: dqkvGmOffset += vSubBlockIdx * firstHalfS1RealSize * n2G * bSize * dSize
│   // BSNGD: dqkvGmOffset += vSubBlockIdx * firstHalfS1RealSize * n2G * dSize
│   // TND:   dqkvGmOffset += vSubBlockIdx * firstHalfS1RealSize * dSize * n2G
│
├── outQue.FreeTensor(dqkvCastTensor)
```

#### 模式B：IS_DQ_WRITE_UB=false（其他模式）

```
DqkvMulsAndCastFromGM(constInfo, runInfo, inputTensor, inQue, outQue)
│
├── dqkvGmOffset = dqWorkSpaceGm的offset
│   // BN2模式: cBlockIdx * CUBE_BASEN * HEAD_DIM_ALIGN + vSubBlockIdx * firstHalfS2RealSize * curDAlign
│   // BN2_MULTIBLK: 额外包含s1oIdx * CUBE_BASEM * HEAD_DIM_ALIGN偏移
│
├── 循环切分（D维度可能超过单次Cast容量）
│   loopNum = Ceil(halfS2RealSize, maxLoopSize)
│   // maxLoopSize = VECTOR_BASEM * VECTOR_BASEN / curDAlign
│   // 例：64×128/128 = 64，每次最多处理64行
│
│   for (loopIdx = 0; loopIdx < loopNum; loopIdx++):
│     ├── dqkvTensor = inQue.AllocTensor<CALC_TYPE>()
│     ├── DataCopy(dqkvTensor, inputTensor[dkvWorkSpaceOffet + loopIdx*loopSize*curDAlign])
│     │   // 从GM（dqWorkSpaceGm）读入UB
│     │
│     ├── Muls(dqkvTensor, dqkvTensor, scaleValue, data_size)
│     │   // dq = dq × scale
│     │
│     ├── dqkvCastTensor = outQue.AllocTensor<OUTDTYPE>()
│     ├── Cast(dqkvCastTensor, dqkvTensor, CAST_ROUND, data_size)
│     │
│     ├── DataCopyPad(dqGm[dqkvGmOffset], dqkvCastTensor, intriParamsOut)
│     │   // 写入最终GM输出
│     │
│     ├── dqkvGmOffset += loopSize * dSize * stride  // 根据layout更新offset
```

---

### 3.6 VF6: ProcessMulsAndCast<DK/DV> — dk/dv最终输出

**数学公式**：
```
dk_final = dk_mm_result × scale  → cast → write GM
dv_final = dv_mm_result          → cast → write GM  （dv不需要乘scale！）
```

**代码位置**：与VF5共用`DqkvMulsAndCastFromUB/DqkvMulsAndCastFromGM`函数，通过`MM_IDX`参数区分。

**与VF5的关键差异**：

1. **DV不乘scale**：
```cpp
// block_vec.h:803-804 (FromUB模式)
if constexpr (MM_IDX != DV_IDX) {
    Muls(inputTensor, inputTensor, scaleValue, dataSize);
}
// dv不需要乘scaleValue，因为dv = P^T × dy，没有Q/K的1/√d缩放
```

2. **halfS1 vs halfS2**：
```cpp
// block_vec.h:769-771 (FromUB模式)
if ((MM_IDX == DQ_IDX && halfS1RealSize == 0) ||
    (MM_IDX != DQ_IDX && halfS2RealSize == 0)) {
    return;
}
// DQ使用halfS1RealSize（s1方向半区）
// DK/DV使用halfS2RealSize（s2方向半区）
```

3. **GM offset不同**：
- DK: `keyOffset`（s2方向的stride）
- DV: `valueOffset`（s2方向 + dSizeV stride）
- BN2模式DK/DV写入workspace（需要AtomicAdd累加）

4. **BN2模式特殊处理**：
```cpp
// block_vec.h:872-877 (BN2_MULTIBLK的dkvWorkSpaceOffet)
if (IS_BN2_MULTIBLK && MM_IDX == DQ_IDX) {
    dkvWorkSpaceOffet = cBlockIdx * AlignTo128(s1Size) * HEAD_DIM_ALIGN
                      + s1oIdx * CUBE_BASEM * HEAD_DIM_ALIGN
                      + vSubBlockIdx * firstHalfS1RealSize * curDAlign;
}
// BN2模式DK/DV: dkvWorkSpaceOffet = cBlockIdx * CUBE_BASEN * HEAD_DIM_ALIGN + vSubBlockIdx * firstHalfS2RealSize * curDAlign
```

5. **dSizeV vs dSize**：
```cpp
// block_vec.h:773 (FromUB模式)
uint64_t dSize = (MM_IDX == DV_IDX) ? constInfo.commonConstInfo.dSizeV : constInfo.commonConstInfo.dSize;
// DV使用dSizeV（可能与dSize不同，IS_D_NO_EQUAL场景）
```

---

### 3.7 Vector侧UB Buffer生命周期图

```
VF1 (ProcessVec1):
  pseOrDyInQue ← dyGm         （拷入dy）
  attenMaskOrYInQue ← yGm     （拷入y）
  → softmaxGradResBuf          （输出softmaxGradRes）

VF2 (ProcessVec2):
  attenMaskOrYInQue ← attenMaskU8Gm  （拷入mask，复用VF1的buffer）
  pseOrDyInQue ← pseGm               （拷入pse，复用VF1的buffer）
  maxSumQue[taskIdMod2] ← softmaxMaxGm/softmaxSumGm  （拷入max/sum）
  → mm2ResTensor = softmax(P)        （结果仍在mm2ResBuf中）

VF3 (ProcessVec3):
  dropMaskBuf ← dropMaskGm 或 GenDropMask   （dropout mask）
  mm1ResTensor = dropout(P)×dy - sfRes×softmax(P)  （计算ds）
  IS_FP8: dsAmaxOutQue ← dsAmaxTensor      （FP8量化amax）
  → dSOutQue ← vecOutBuffer (ds的Cast+NZ转换)
  → CopyUB2L1: ds → dSL1Buf              （ds拷入L1）
  → CopyUB2L1: p → pL1Buf (仅VF4)        （p拷入L1）

VF4 (ProcessVec4):
  mm2ResTensor = dropout(P)             （P也做dropout）
  → pOutQue ← vecOutBuffer1 (p的Cast+NZ转换)
  → CopyUB2L1: p → pL1Buf              （p拷入L1）

VF5 (ProcessMulsAndCast<DQ>):
  IS_WRITE_UB: mm1ResBuf中读dq → Muls*scale → Cast → dqGm
  !IS_WRITE_UB: dqWorkSpaceGm中读dq → Muls*scale → Cast → dqGm

VF6 (ProcessMulsAndCast<DK/DV>):
  IS_WRITE_UB: mm2ResBuf中读dk/dv → Muls*scale(Cast → dkGm/dvGm
  !IS_WRITE_UB: dkWorkSpaceGm/dvWorkSpaceGm → Muls*scale(仅dk) → Cast → dkGm/dvGm
```

**UB Buffer复用关系**：

```
attenMaskOrYInQue: VF1(y) → VF2(attenMask) → VF3(dropout临时) → VF5/VF6(dq/dk读)
pseOrDyInQue:      VF1(dy) → VF2(pse) → VF3(dy复用)
dSOutQue:           VF3(ds输出) → CopyUB2L1 → 释放
pOutQue:            VF4(p输出) → CopyUB2L1 → 释放
softmaxGradResBuf:  VF1(输出) → VF3(输入) → 释放
maxSumQue[2]:       VF2(输入) → PingPong交替使用
mm1ResBuf[2]:       C1(mm1结果) → VF3(输入) → VF5(dq输出,IS_WRITE_UB时)
mm2ResBuf[2]:       C2(mm2结果) → VF2/VF3/VF4(输入) → VF6(dk输出,IS_WRITE_UB时)
```

---

## 附录：完整数据流图

```
GM输入                  Cube侧                  Vector侧               GM输出
───────                ────────                ────────               ───────

dy ──GM→ L1 → L0A ─┐
                    │ mm1(Dy×V) → L0C → Fixpipe → UB:mm1ResBuf
V  ──GM→ L1 → L0B ─┘                          │
                                                ↓
Q ──GM→ L1 → L0A ─┐                     Vec1: softmaxGrad(dy,y) → softmaxGradResBuf
                    │ mm2(Q×K) → L0C → Fixpipe → UB:mm2ResBuf     │
K  ──GM→ L1 → L0B ─┘                          ↓
                                         Vec2: pse+mask+softmax → mm2ResBuf=P
softmaxMax ──GM→ UB:maxSumQue                │
softmaxSum ──GM→ UB:maxSumQue                ↓
                                         Vec3: ds = drop(P)*dy - sf*softmax(P)
                                                │
                                         Cast+NZ → UB:dSOutQue → L1:dSL1Buf
                                         Cast+NZ → UB:pOutQue → L1:pL1Buf

                                         Vec4: p量化 → UB:pOutQue → L1:pL1Buf

dSL1Buf → L0B ─┐
               │ mm3(ds×Q) → L0C → Fixpipe → UB(IS_WRITE_UB) 或 GM(dqWorkSpace)
Q(复用) → L0A ─┘                          │
                                           ↓
                                   Vec5: dq × scale → Cast → dqGm ←── dq输出

dSL1Buf → L0B ─┐
               │ mm4(ds×K) → L0C → Fixpipe → UB(IS_WRITE_UB) 或 GM(dkWorkSpace)
K(复用) → L0A ─┘                          │
                                           ↓
                                   Vec6: dk × scale → Cast → dkGm ←── dk输出

pL1Buf → L0B ─┐
              │ mm5(P×dy) → L0C → Fixpipe → GM(dvWorkSpace/dvGm)
dy(复用) → L0A ─┘                          │
                                           ↓
                                   Vec6: dv(Cast only) → dvGm ←── dv输出
```