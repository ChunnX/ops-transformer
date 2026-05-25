# FAG Kernel侧代码总结

## 1. 整体架构

### 1.1 文件结构

| 文件 | 行数 | 核心职责 |
|------|------|----------|
| `flash_attention_score_grad_common.h` | 165 | 全局常量定义、同步Flag宏、枚举类型 |
| `flash_attention_score_grad_tiling_data_regbase.h` | 628 | TilingData结构体解析（GetTilingData） |
| `flash_attention_score_grad_entry_regbase.h` | 452 | 入口函数：Init + Process路由 |
| `flash_attention_score_grad_kernel_base.h` | 2392 | KernelBase基类：Init/SetConstInfo/SetRunInfo/IsValid/GetNextValidIdx |
| `flash_attention_score_grad_kernel.h` | 410 | 非确定性Kernel：Process()主循环 |
| `flash_attention_score_grad_kernel_deter.h` | 1227 | 确定性Kernel：Process_OLD_DETER/Process_NEW_DETER |
| `flash_attention_score_grad_block_cube.h` | 1796 | Cube侧：5个MM计算（mm1~mm5）+ L1/L0C Buffer管理 |
| `flash_attention_score_grad_block_vec.h` | 1490 | Vector侧：Vec1~Vec6计算 + UB Buffer管理 |

### 1.2 类继承关系

```
FlashAttentionScoreGradKernelBase<ChildClass, CubeBlockType, VecBlockType>
    ├── FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>  (非确定性)
    │       └── Process()  (普通BN2GS1S2/BN2S2/BN2分核)
    └── FlashAttentionScoreGradKernelDeter<CubeBlockType, VecBlockType>  (确定性)
            ├── Process_OLD_DETER()  (旧确定性：dqIsNeedDeter[4] bitmap)
            └── Process_NEW_DETER()  (新确定性：CoordinateInfo索引)
```

CRTP模式：`ChildClass`通过模板参数传入，用于调用`SetUniqueConstInfo()`和`SetUniqueRunInfo()`，实现模板差异化。

### 1.3 CV协作模型

Regbase架构采用 **1 Cube + 2 Vector** 核比（CVRatio=1:2），即：
- 1个Cube核（AIC）负责5次MatMul计算
- 2个Vector核（AIV0/AIV1）共享1个Cube核的输出
- `vSubBlockIdx=0`处理上半s1（halfS1RealSize），`vSubBlockIdx=1`处理下半s1（s1RealSize - halfS1RealSize）
- 同步Flag使用 `+16` 偏移区分V0和V1

---

## 2. 初始化流程

### 2.1 Init() (entry_regbase.h)

```
Init()
├── GetTilingData()            # 从GM解析TilingData到tilingData结构体
├── InitGlobalBuffer()         # 设置GM地址（query/key/value/dy/dq/dk/dv等）
├── SetConstInfo()             # 从tilingData提取常量到constInfo结构体
├── SetSwizzleConstInfo()      # 计算swizzle分块轮数（leftUp/leftDown/rightUp/rightDown）
├── InitBuffer()               # 分配UB/L1/L0C Buffer
├── cubeBlock.InitCubeBuffer() # Cube侧L1/L0A/L0B/L0C Buffer分配
├── vecBlock.InitUbBuffer()    # Vector侧UB Buffer分配
└── 其他初始化（dropMask/attenMask/pse/deter等）
```

### 2.2 constInfo结构体（核心常量）

```cpp
FagConstInfo {
    commonConstInfo {
        s1Size, s2Size, dSize, dSizeV   // 输入shape
        s1BaseSize = CUBE_BASEM          // Cube基本块M维度（128）
        s2BaseSize = CUBE_BASEN          // Cube基本块N维度（128）
        layoutType                        // BNGSD/SBNGD/BSNGD
        mm1Ka, mm1Kb                     // mm1的K轴stride
        mm2Ka, mm2Kb                     // mm2的K轴stride
        gS1D, n2GS1D, s2D, gS2, n2D, n2GD  // stride预计算
    }
    s1Outer, s1CvTail, s2Outer, s2Tail  // 分块切分信息
    bSize, n2Size, gSize                  // BN2G维度
    scaleValue                            // softmax缩放因子
    deterConstInfo {                      // 确定性专用常量
        noNeedDeter, usedCubeCoreNum, usedVectorCoreNum
        eachVecCoreS1Offset, eachVecCoreS2Offset
        dqEachVectorSize, dkvEachVectorSize
        deterBStride, deterN2Stride, deterGStride, deterS1oStride
    }
    swizzleConstInfo {                    // swizzle分块常量
        leftUpTotalRound, leftDownTotalRound, rightUpTotalRound, rightDownTotalRound
        leftTotalRound, batchTotalRound
    }
}
```

### 2.3 Buffer分配策略

**UB（Vector侧）：**
| Buffer | 大小 | 用途 |
|--------|------|------|
| attenMaskOrYInQue | VECTOR_BASEM * VECTOR_BASEN * sizeof(CALC_TYPE) | y/attenMask/softmaxGradRes |
| pseOrDyInQue | VECTOR_BASEM * VECTOR_BASEN * sizeof(OUTDTYPE) | dy/pse |
| dSOutQue | (VECTOR_BASEM+1) * VECTOR_BASEN * sizeof(OUTDTYPE) | ds输出 |
| pOutQue | (VECTOR_BASEM+1) * VECTOR_BASEN * sizeof(OUTDTYPE) | p输出 |
| softmaxGradResBuf | VECTOR_BASEM * sizeof(CALC_TYPE) | softmaxGrad中间结果 |
| maxSumQue[2] | VECTOR_BASEM * MAX_SUM_REDUCE_AXIS_SIZE * 2 | softmax max/sum 双buffer |
| dropMaskBuf | VECTOR_BASEM * VECTOR_BASEN / 8 | dropMask |
| deterInOutQue | DETER_DQ_UB_SIZE | 确定性dq中间结果 |
| deterOffsetBuf | DETER_OFFSET_UB_SIZE | 确定性offset |

**L0C（Cube侧）：**
| 模式 | 分配策略 |
|------|----------|
| IS_DKV_RESIDENT_L0C | mm1Mm2Mm3L0CBuf + dkL0CBuf + dvL0CBuf（DK/DV常驻L0C） |
| D192/Dv128特殊 | mm1Mm2Mm3L0CSpecialBuf + dkL0CSpecialBuf + dvL0CSpecialBuf |
| HEAD_DIM≤256 | commonl0CBuf = L0C_MAX_SIZE / 2 |
| HEAD_DIM>256 | commonl0CBuf = L0C_MAX_SIZE |

**L1（Cube侧）：**
| 模式 | 分配策略 |
|------|----------|
| IS_L1_REUSE/IS_L1_PRELOAD | dYL1Buf + vL1Buf + qL1Buf + kL1Buf（各独立） |
| 普通模式 | commonL1Buf（共享，按需复用） |
| IS_FP32_D_EXCEED_256 | fp32L1Buf1 + fp32L1Buf2 |
| IS_FP8_INPUT + IS_L1_REUSE | dYL1Buf + qL1Buf + kL1Buf |

---

## 3. Process()主循环（非确定性）

### 3.1 主循环伪代码

```cpp
Process() {
    if (blockEnds[cBlockIdx] == 0) return;  // 无任务则退出
    
    taskId = 0;
    runInfos[2];  // PingPong: runInfos[taskId&1]交替使用
    
    // 获取第一个有效block
    nextValidBlockInnerIdx = GetNextValidIdx(runInfos[0], 0, blockStarts[cBlockIdx]);
    blockInnerIdx = nextValidBlockInnerIdx;
    
    while (true) {
        isLastLoop = (blockInnerIdx == -1);
        
        // ─── Pipeline Stage 1: 处理前一轮的Vector计算 ───
        if (taskId > 0) {
            prevRunInfo = runInfos[(taskId+1) & 1];
            
            // V1: SoftmaxGrad（Cast + SoftmaxGradFront）
            vecBlock.ProcessVec1(constInfo, prevRunInfo);
            
            // Wait mm1/mm2结果（Cube→Vector同步）
            if AIV {
                WaitCrossCore(SYNC_C1_TO_V2_FLAG[(taskId+1)&1]);  // mm1完成
                WaitCrossCore(SYNC_C2_TO_V2_FLAG[(taskId+1)&1]);  // mm2完成
            }
        }
        
        // ─── Pipeline Stage 2: 当前轮Cube计算 ───
        if (!isLastLoop) {
            // 获取下一个有效block
            nextValidBlockInnerIdx = GetNextValidIdx(runInfos[(taskId+1)&1], taskId+1, blockInnerIdx+1);
            SetRunInfo(runInfos[taskId&1], runInfos[(taskId+1)&1], taskId, blockInnerIdx, nextValidBlockInnerIdx);
            
            // C1: mm1 = dy × V（计算P的第一部分）
            mm1ResTensor = mm1ResBuf[taskId&1];
            cubeBlock.IterateMmDyV(mm1ResTensor, constInfo, runInfos[taskId&1], preloadArgs);
            if AIC { SetCrossCore(SYNC_C1_TO_V2_FLAG[taskId&1]); SetCrossCore(+16+...) }
            
            // C2: mm2 = Q × K（重算attention score）
            mm2ResTensor = mm2ResBuf[taskId&1];
            cubeBlock.IterateMmQK(mm2ResTensor, constInfo, runInfos[taskId&1], preloadArgs);
            if AIC { SetCrossCore(SYNC_C2_TO_V2_FLAG[taskId&1]); SetCrossCore(+16+...) }
            
            // V_pre: CopyMaxSum（拷入softmax max/sum）
            vecBlock.CopyMaxSum(constInfo, runInfos[taskId&1], taskId);
        }
        
        // ─── Pipeline Stage 3: 前一轮的后续Vector+Cube计算 ───
        if (taskId > 0) {
            mm1ResTensor = mm1ResBuf[prevRunInfo.taskIdMod2];
            mm2ResTensor = mm2ResBuf[prevRunInfo.taskIdMod2];
            
            // V2: pse + attenMask + muls + simpleSoftmax
            vecBlock.ProcessVec2(mm2ResTensor, constInfo, prevRunInfo);
            
            // V3: dropout + sub + mul + cast + nd2nz → ds
            vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, constInfo, prevRunInfo);
            
            // V4: cast + nd2nz → p
            vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, constInfo, prevRunInfo);
            
            // Vector→Cube同步：ds/p已拷入L1
            if AIV {
                SetCrossCore(SYNC_V3_TO_C3_FLAG);  // ds已到L1
                SetCrossCore(SYNC_V4_TO_C5_FLAG);  // p已到L1
            }
            if AIC {
                WaitCrossCore(SYNC_V3_TO_C3_FLAG);  WaitCrossCore(+16+...)
                WaitCrossCore(SYNC_V4_TO_C5_FLAG);  WaitCrossCore(+16+...)
            }
            
            // ─── 根据SPLIT_AXIS分支 ───
            if (SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK) {
                // C3: mm3 = ds × Q → dq
                // C4: mm4 = ds × K → dk
                // C5: mm5 = p × dy → dv
                // V5: dq muls + cast → GM
                // V6: dk muls + cast → GM
                needSyncDkMM = true;
            } else if (IS_BN2_MULTIBLK) {
                // C3/C4/C5: dq/dk/dv → workspace
                // V5/V6/V6: 仅在特定条件输出GM（isLastS1Outer / !isNextS2IdxNoChange）
                needSyncDkMM = true;
            } else if (SPLIT_AXIS == BN2S2) {
                // BN2S2模式：dq/dk/dv可能写UB或写GM
                // needSyncDkDvFixUb = true（L0C累加后需搬运）
                needSyncDkMM = true;
            } else {
                // BN2GS1S2标准模式：dq/dk/dv全写GM
                needSyncDkMM = true;
            }
        }
        
        if (blockInnerIdx == -1) break;
        taskId++;
        blockInnerIdx = nextValidBlockInnerIdx;
    }
}
```

### 3.2 Pipeline时间线示意

```
时间 ──────────────────────────────────────────────────────────►

Loop0:  [C1:mm1] [C2:mm2]
          │       │
          ├───────┤ SetFlag C1/C2_TO_V2

Loop1:  [V1:softmaxGrad(prev)]  [Wait C1/C2_TO_V2(prev)]
        [C1:mm1(curr)]  [C2:mm2(curr)]
        [V2:simpleSoftmax(prev)]  [V3:ds(prev)]  [V4:p(prev)]
          │ SetFlag V3_TO_C3, V4_TO_C5
        [Wait V3_TO_C3(prev), V4_TO_C5(prev)]
        [C3:mm3→dq(prev)]  [C4:mm4→dk(prev)]  [C5:mm5→dv(prev)]
        [V5:dq_cast(prev)]  [V6:dk_cast(prev)]

Loop2:  [V1(prev)] [Wait C1/C2_TO_V2(prev)]
        [C1:mm1] [C2:mm2] ...
        [V2/V3/V4(prev)] [C3/C4/C5(prev)] [V5/V6(prev)]
```

---

## 4. 同步机制详解

### 4.1 同步Flag定义（common.h）

```cpp
// Cube→Vector同步（mm1/mm2结果就绪）
SYNC_C1_TO_V2_FLAG[2]  // mm1(Dy×V)完成，PingPong双buffer
SYNC_C2_TO_V2_FLAG[2]  // mm2(Q×K)完成，PingPong双buffer

// Vector→Cube同步（ds/p已拷入L1）
SYNC_V3_TO_C3_FLAG     // ds已从UB拷入L1，Cube可开始mm3/mm4
SYNC_V4_TO_C5_FLAG     // p已从UB拷入L1，Cube可开始mm5

// Cube→Vector同步（dq/dk/dv mm结果就绪）
SYNC_C3_TO_V5_FLAG     // mm3(dq)完成
SYNC_C4_TO_V6_FLAG     // mm4(dk)完成
SYNC_C5_TO_V4_FLAG     // mm5(dv)完成（写GM后Vector可muls+cast）

// 确定性专用同步
SYNC_DETER_FIX_FLAG    // 确定性L0C累加完成
SYNC_C4_TO_V3_FLAG     // dk mm完成，下一轮可使用dkL0C结果
SYNC_V0_V1_DS_A_MAX_DONE_FLAG  // FP8场景V0/V1交换ds amax值

// 特殊同步
SYNC_V3_TO_V2_FLAG     // softmaxGrad中间结果传递（内部V→V）
```

### 4.2 +16偏移机制

1:2核比下，同一个同步Flag需要区分V0和V1：
- `SetCrossCore(SYNC_C1_TO_V2_FLAG[taskId&1])` → V0收到信号
- `SetCrossCore(SYNC_C1_TO_V2_FLAG[taskId&1] + 16)` → V1收到信号
- AIC侧SetFlag时同时发送两组（无+16和+16）
- AIV侧WaitFlag时只Wait自己对应的那组

**代码示例（Process()中的典型同步）：**

```cpp
// Cube侧：同时通知V0和V1
if ASCEND_IS_AIC {
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[taskId & 1]);
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[taskId & 1]);
}

// Vector侧：只等待自己的信号
if ASCEND_IS_AIV {
    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[(taskId + 1) & 1]);
}
```

### 4.3 FP8 ds Amax同步

FP8场景下，ds量化需要全局最大值：
1. V0/V1各自计算局部ds amax
2. 写入workspace GM（`dsAmaxWorkSpaceGm[vBlockIdx * 128]`）
3. `SetCrossCore(SYNC_V0_V1_DS_A_MAX_DONE_FLAG)` 通知对方
4. `WaitCrossCore(SYNC_V0_V1_DS_A_MAX_DONE_FLAG)` 等待对方
5. 读取对方的amx值，取Max作为全局qScaleDs
6. 写入ssbuf（`InitCubeVecSharedParams`），通知Cube侧读取

---

## 5. Cube侧计算详解（block_cube.h）

### 5.1 五次MatMul

| MM | 计算 | 公式 | M轴 | N轴 | K轴 |
|----|------|------|------|------|------|
| mm1 | Dy × V | P_part = dy × V^T | s1RealSize | s2RealSize | dSizeV |
| mm2 | Q × K | S = Q × K^T | s1RealSize | s2RealSize | dSize |
| mm3 | ds × Q | dq = ds × Q^T | s2RealSize | dSize | s1RealSize |
| mm4 | ds × K | dk = ds × K^T | s2RealSize | dSize | s1RealSize |
| mm5 | p × dy | dv = p^T × dy | s2RealSize | dSizeV | s1RealSize |

> 注意：mm3/mm4/mm5中，M轴变成了s2维度（因为ds/p是s1×s2矩阵，转置后s2变M轴）

### 5.2 IterateMmDyV (mm1) 核心流程

```
IterateMmDyV(mm1ResTensor, constInfo, runInfo, preloadArgs)
├── 加载左矩阵(dy)到L1
│   ├── IS_L1_PRELOAD → dYL1Buf.Get() / dYL1Buf.GetPre()
│   ├── IS_L1_REUSE → dYL1Buf.Get()
│   └── 普通模式 → commonL1Buf.Get()
│   └── Nd2Nz DataCopy: dyGm → dyL1Tensor
│
├── 循环切N轴（FP32+D>512时切2次，否则1次）
│   ├── 加载右矩阵(V)到L1
│   │   ├── IS_L1_REUSE/IS_L1_PRELOAD → vL1Buf.Get()（仅s2变化时重拷）
│   │   ├── Nd2Nz DataCopy: valueGm → vL1Tensor
│   │
│   ├── L1→L0A/L0B + MMad
│   │   ├── mm1L0CBuffer.Wait(FIX_M)
│   │   ├── MatmulBase(dyL1, vL1, l0aBuf, l0bBuf, mm1L0C, param)
│   │   ├── mm1L0CBuffer.Set(M_FIX) → mm1L0CBuffer.Wait(M_FIX)
│   │
│   ├── Fixpipe: L0C → UB
│   │   ├── dualDstCtl=1（双目标模式，M/2×N写入每个UB）
│   │   ├── dstStride=CUBE_BASEN（UB行间距128）
│   │   ├── Fixpipe(mm1ResTensor[ubOffset], mm1L0C, fixpipeParams)
│   │
│   ├── L1 preload: 拷入下一个dy到L1
│   │   ├── IS_L1_PRELOAD → DataCopy(nextDyGm → dyL1NextBuffer)
│   │
│   └── mm1L0CBuffer.Set(FIX_M)  // 释放L0C
```

### 5.3 IterateMmQK (mm2) 核心流程

与mm1类似，但左矩阵为Q（`queryGm`），右矩阵为K（`keyGm`）。

**Rope特殊处理：**
```cpp
if constexpr (IS_ROPE) {
    // Q拷入两段：前128列(queryGm) + 后64列(queryRopeGm)
    DataCopy(qL1Tensor, queryGm[queryOffsetWithRopeForMm12], nd2NzParams);
    DataCopy(qL1Tensor[dstNzC0Stride * ROPE_D_128], queryRopeGm[qRopeOffset], nd2NzParams_d64);
    // K同理
}
```

### 5.4 IterateMmDsK (mm3 → dq) / IterateMmDsQ (mm4 → dk)

```
IterateMmDsK(outTensor, dSL1Buf, constInfo, runInfo)
├── IS_FP8_INPUT → WaitCrossCore(15)  // 等待ssbuf中的qScaleDs
├── 循环切N轴（D维度切分，baseN=l1BaseD）
│   ├── 加载Q到L1（IS_L1_REUSE → kL1Buf.GetReused）
│   ├── Nd2Nz DataCopy: queryGm → qL1Tensor
│   ├── 加载ds到L0B（从dSL1Buf获取）
│   ├── MMad: ds × Q → L0C
│   ├── IS_WRITE_UB → Fixpipe到UB（dualDstCtl=1）
│   │   └── Cast + muls → dq写入GM
│   └── !IS_WRITE_UB → Fixpipe直接到GM
│       ├── needAtomic → SetAtomicAdd（BN2/BN2S2多核重叠）
│       ├── Fixpipe(outTensor[offset], mm3L0C, fixpipeParams)
│       └── SetAtomicNone
```

### 5.5 IterateMmPDy (mm5 → dv)

与mm3/mm4类似，左矩阵为p（从pL1Buf获取），右矩阵为dy。

---

## 6. Vector侧计算详解（block_vec.h）

### 6.1 六个Vec阶段

| 阶段 | 函数 | 核心计算 | 输入 | 输出 |
|------|------|----------|------|------|
| VF1 | ProcessVec1 | Cast + SoftmaxGradFront | dy, y | softmaxGradResTensor |
| VF2 | ProcessVec2 | pse + attenMask + muls + simpleSoftmax | mm2Res(Q×K), max/sum | mm2Res(softmax结果) |
| VF3 | ProcessVec3 | dropout + sub + mul + cast + nd2nz | mm1Res, mm2Res, softmaxGradRes | ds(L1), p(L1) |
| VF4 | ProcessVec4 | cast + nd2nz（p量化） | mm2Res(softmax后) | p(L1) |
| VF5 | ProcessMulsAndCast<DQ> | dq muls(scale) + cast(fp32→fp16) | mm3Res(dq) | dqGm |
| VF6 | ProcessMulsAndCast<DK/DV> | dk/dv muls + cast | mm4/mm5Res | dkGm/dvGm |

### 6.2 ProcessVec1（SoftmaxGrad）

```
ProcessVec1(constInfo, runInfo)
├── if (halfS1RealSize == 0) return;  // V1核若无数据则跳过
├── CopyInSoftmaxGrad(dy, y → attenMaskOrYInQue, pseOrDyInQue)
├── CalculateCastSoftmaxGrad(dy, y → softmaxGradResTensor)
│   ├── Cast: OUTDTYPE → CALC_TYPE
│   ├── SoftmaxGradFront: dy * y → sfmgRes
│   └── ReduceSum: sfmgRes沿s2轴求和 → softmaxGradRes
```

### 6.3 ProcessVec2（SimpleSoftmax）

```
ProcessVec2(mm2ResTensor, constInfo, runInfo)
├── IS_FP8_INPUT → Muls(mm2Res, deqScaleQ * deqScaleK)
├── CopyInAttenMask(attenMaskU8Gm → attenMaskOrYInQue)
├── CopyInPse(pseGm → pseOrDyInQue)
├── CalculatePseMulsSelSimpleSoftMax(mm2Res → mm2Res)
│   ├── pse处理（IS_PSE → 加pse偏置）
│   ├── attenMask处理（IS_ATTEN_MASK → mask无效区域）
│   ├── Muls(scaleValue缩放)
│   ├── SimpleSoftmax: exp(max - score) / sum → softmax结果
│   └── 输出: mm2ResTensor = softmax(P)
```

### 6.4 ProcessVec3（ds = dropout(P) * dy - softmaxGradSum * P）

```
ProcessVec3(dSL1Buffer, mm1Res, mm2Res, constInfo, runInfo)
├── Dropout处理
│   ├── GenDropMask / CopyInDropOuter
│   ├── CalculateDropout(mm1Res → mm1Res)  // mm1Res = P * dropoutMask
│
├── Sink处理（IS_DROP → CopyDpToTmpBuf）
│
├── BroadcastSubMul(mm1Res, softmaxGradRes, mm2Res → ds)
│   ├── ds = P_drop * dy - softmaxGradSum * P_softmax
│   ├── 模板化S2大小（Aligned64/Aligned128）
│   ├── IS_DETER → 补零尾块脏数据（Duplicate 0）
│
├── IS_FP8_INPUT → ds量化
│   ├── DsAbsReduceMax → dsAmax
│   ├── V→S同步（eventIDVToS）
│   ├── 写workspace GM → SetCrossCore(SYNC_V0_V1_DS_A_MAX_DONE_FLAG)
│   ├── WaitCrossCore → 读对方amax → Max(curDsMax, anotherDsMax)
│   ├── qScaleDs = FP8_MAX / dsAmax
│   ├── Muls(mm1Res, qScaleDs)
│   ├── InitCubeVecSharedParams(sharedParams → ssbuf)
│
├── CastTransdataDeconflict(ds → INPUT_TYPE, nd2nz格式)
├── CopyUBToL1Vec3(ds → dSL1Buffer)  // ds拷入L1供mm3/mm4使用
│
└── Sink输出（IS_SINK → ProcessVecSink）
```

### 6.5 ProcessVec4（p量化输出）

```
ProcessVec4(pL1Buffer, mm2Res, constInfo, runInfo)
├── CalculateDropout(mm2Res → mm2Res)  // p也做dropout（quant场景）
├── CastTransdataDeconflict(mm2Res → OUTDTYPE, nd2nz)
├── CopyUB2L1Quant(p → pL1Buffer)  // p拷入L1供mm5使用
```

### 6.6 ProcessMulsAndCast（dq/dk/dv最终输出）

```
ProcessMulsAndCast<MM_IDX>(inputTensor, constInfo, runInfo)
├── IS_WRITE_UB → DqkvMulsAndCastFromUB
│   ├── Muls(inputTensor, scaleValue)
│   ├── Cast(CALC_TYPE → OUTDTYPE)
│   ├── DataCopyPad(dqkvGm[offset])
│
├── !IS_WRITE_UB → DqkvMulsAndCastFromGM
│   ├── 循环切分（maxLoopSize = VECTOR_BASEM*VECTOR_BASEN/curDAlign）
│   ├── DataCopy(GM → LocalTensor)
│   ├── Muls + Cast
│   ├── DataCopyPad(LocalTensor → GM)
```

**GM offset计算（根据layoutType）：**

| Layout | dq offset | dk offset | dv offset |
|--------|-----------|-----------|-----------|
| BNGSD | b*n2GS1D + n2*gS1D + g*s1D + s1o*d + vSubBlock*firstHalfS1*d | 类似 | 类似(dSizeV) |
| SBNGD | b*n2GD*s1 + n2*gD*s1 + g*d*s1 + s1o*d + vSubBlock*firstHalfS1*n2G*b*d | 类似 | 类似 |
| BSNGD | b*n2GS1D + n2*gS1D + g*s1D + s1o*d + vSubBlock*firstHalfS1*n2G*d | 类似 | 类似 |
| TND | prefix*n2GD + n2*gD*s1o + g*d + vSubBlock*firstHalfS1*n2G*d | 类似 | 类似 |

---

## 7. SetRunInfo详解（block内坐标解码）

### 7.1 基本流程

```
SetRunInfo(runInfo, nextRunInfo, taskId, index, nextIndex)
├── 从index解码出(b, n2, g, s1o, s2o)五维坐标
│   ├── boIdx = index / n2GS1oS2o
│   ├── n2oIdx = bDimTail / gS1oS2o
│   ├── goIdx = n2DimTail / s1oS2o
│   ├── s1oIdx = gDimTail % s1Outer  (or from s1oS2o分解)
│   ├── s2oIdx = gDimTail / s1Outer
│
├── 计算s1RealSize和s2RealSize
│   ├── s1RealSize = (s1oIdx == s1Outer-1) ? s1CvTail : CUBE_BASEM
│   ├── s2RealSize = s2CvEnd - s2CvBegin
│
├── halfS1/halfS2分割（1:2核比）
│   ├── halfS1RealSize = (s1RealSize+1) >> 1
│   ├── firstHalfS1RealSize = halfS1RealSize
│   ├── if vSubBlockIdx==1:
│   │     halfS1RealSize = s1RealSize - halfS1RealSize
│   │     halfS2RealSize = s2RealSize - halfS2RealSize
│
├── 计算GM offset
│   ├── queryOffset = b*n2GS1D + n2*gS1D + g*s1D + s1o*CUBE_BASEM*d
│   ├── keyOffset = b*n2GS2D + n2*gS2D + g*s2D + s2o*CUBE_BASEN*d
│   ├── valueOffset = 类似keyOffset但用dSizeV
│   ├── dyOffset = queryOffset (IS_D_NO_EQUAL时单独计算)
│
├── vecCoreOffset = vSubBlockIdx * firstHalfS1RealSize
│
├── L1 preload offset计算
│   ├── GetNextDxAndQueryOffset → preloadArgs.nextQueryOffset/nextDyOffset/nextMorN
│
├── Rope场景额外offset
│   ├── queryOffsetWithRope / keyOffsetWithRope
│   ├── qRopeOffset / kRopeOffset
```

### 7.2 TND场景特殊处理

TND模式下，batch维度有变长序列，需要动态查找当前block属于哪个batch：

```cpp
for (bIdx = curBatchIdx; bIdx < bSize; bIdx++) {
    GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
    totalBaseIdx = n2G * s1Outer * s2Outer;
    if (resbaseIdx < totalBaseIdx) {
        // 当前block在此batch内
        // 解码n2o/go/s1o/s2o
        break;
    } else {
        // 累加offset跳过此batch
        curBatchTotalS1BOffset += actualS1Len * n2GD;
        curBatchTotalS2BOffset += actualS2Len * n2D;
        resbaseIdx -= totalBaseIdx;
    }
}
```

---

## 8. GetNextValidIdx（稀疏跳块）

### 8.1 流程

```
GetNextValidIdx(runInfo, taskId, blockInnerIdx, curLoopIdx)
├── IS_TND_SWIZZLE → 按sparseType分派
│   ├── DENSE → GetNextValidIdxForTndSwizzleDense
│   ├── CASUAL → GetNextValidIdxForTndSwizzleCasual
│   └── BAND → GetNextValidIdxForTndSwizzleBand
│
├── !isSplitByBlockIdx → 逐块检查IsValid
│   ├── while (!IsValid(runInfo, taskId, nextIdx))
│   │     nextIdx++
│   ├── return nextIdx or -1
│
├── isSplitByBlockIdx → GetNextValidIdxForSwizzle
│   └── 使用swizzle分块索引直接定位
```

### 8.2 IsValid判断逻辑

根据`sparseMode`判断block是否在有效区域内：
- **无mask**：所有block有效
- **LEFT_UP_CAUSAL**：s2Idx < s1Idx对应的范围
- **RIGHT_DOWN_CAUSAL**：s2EndLen根据s1位置计算
- **BAND**：s2SparseLeft和s2SparseRight限定有效s2范围
- **PREFIX**：s2EndLen受prefixN限制

---

## 9. 确定性模式（kernel_deter.h）

### 9.1 OLD_DETER模式

使用`dqIsNeedDeter[4]` bitmap（256bit）标记每个s2o是否需要确定性同步：

```cpp
GetIsNeedDeter(computeLoopIdx) {
    // dqIsNeedDeter[4]中每个元素对应64个s2o
    // bitmap标记哪些s2o的dq需要多核同步累加
    // dk/dv始终需要确定性累加
}

DeterSync(loopIdx) {
    // 确定性dq/dk/dv的GM输出需要：
    // 1. 先写入workspace（deterGm）
    // 2. 所有核完成一轮后，由特定核汇总写入最终GM
}
```

### 9.2 NEW_DETER模式

使用`CoordinateInfo`预计算索引，支持TND确定性：

```cpp
CoordinateInfo {
    batchId, n2Idx, gIdx, s1Idx, s2Idx
    actualS1Len, actualS2Len
    s1Outer, s2Outer
    isDense, isCasual, isBand  // sparse标记
}

CalDeterIndex(roundId, maxLoopNum, nextValidRoundId, nextValidIndex) {
    // 按sparseType分派：
    // CalDenseDeterIndex → 全密集遍历
    // CalCausalDeterIndex → causal三角形遍历
    // CalBandDeterIndex → band对角带遍历
}
```

### 9.3 确定性输出流程

```
Process_OLD_DETER() {
    // 主循环与非确定性类似，但：
    // 1. mm3/mm4/mm5结果先写入deterGm（workspace）
    // 2. 每轮完成后，ProcessPostDeter将workspace数据cast+写入最终GM
    // 3. 使用dqIsNeedDeter判断哪些dq需要同步
    // 4. dk/dv始终通过workspace中转
    // 5. Fixpipe结果写UB（IS_DQ_WRITE_UB/IS_DK_WRITE_UB）
    //    → ProcessMulsAndCast从UB muls+cast+写GM
    //    或写GM（!IS_WRITE_UB）→ ProcessMulsAndCast从GM读+muls+cast+写GM
}
```

---

## 10. 关键数据流图

### 10.1 非确定性BN2模式数据流

```
GM: dy ──→ [L1] ──→ [L0A] ──┐
                              │ MMad → [L0C] → Fixpipe → [UB:mm1ResBuf]
GM: V  ──→ [L1] ──→ [L0B] ──┘                                     │
                                                                    ↓
GM: Q ──→ [L1] ──→ [L0A] ──┐                              Vec1: softmaxGrad(dy,y)
                              │ MMad → [L0C] → Fixpipe → [UB:mm2ResBuf]    │
GM: K ──→ [L1] ──→ [L0B] ──┘                                     ↓
                                                            Vec2: pse+mask+softmax
                                                                    │
GM: softmaxMax/Sum ──→ [UB:maxSumQue] ──────────────────────────── ↓
                                                                    │
                                                            Vec3: ds = dropout(P)*dy - sfmgSum*softmax(P)
                                                                    │
                                                            Cast+nd2nz → [UB:dSOutQue] → CopyUB2L1 → [L1:dSL1Buf]
                                                            Cast+nd2nz → [UB:pOutQue]  → CopyUB2L1 → [L1:pL1Buf]

[L1:ds] ──→ [L0B] ──┐
GM: Q ──→ [L1] ──→ [L0A] ──┘ │ MMad → [L0C] → Fixpipe → [UB:mm1Res] or [GM:dqWorkSpace]
                                                       │
                                              Vec5: dq muls+cast → GM: dq

[L1:ds] ──→ [L0B] ──┐
GM: K ──→ [L1] ──→ [L0A] ──┘ │ MMad → [L0C] → Fixpipe → [UB:mm2Res] or [GM:dkWorkSpace]
                                                       │
                                              Vec6: dk muls+cast → GM: dk

[L1:p]  ──→ [L0B] ──┐
GM: dy ──→ [L1] ──→ [L0A] ──┘ │ MMad → [L0C] → Fixpipe → GM: dv
```

### 10.2 Buffer生命周期（PingPong）

```
taskId=0:  mm1ResBuf[0] ← C1(mm1), mm2ResBuf[0] ← C2(mm2)
taskId=1:  mm1ResBuf[1] ← C1(mm1), mm2ResBuf[1] ← C2(mm2)
           Vec1~Vec4 使用 mm1ResBuf[0], mm2ResBuf[0]
taskId=2:  mm1ResBuf[0] ← C1(mm1), mm2ResBuf[0] ← C2(mm2)
           Vec1~Vec4 使用 mm1ResBuf[1], mm2ResBuf[1]
           C3~C5 使用 mm1ResBuf[1]/mm2ResBuf[1] (BN2+IS_WRITE_UB)
           或 dqWorkSpaceGm/dkWorkSpaceGm/dvGm (其他模式)
```

---

## 11. 特殊场景处理

### 11.1 FP8量化

- mm1/mm2输入FP8，L0C输出fp32
- ds需要量化回FP8：先计算全局amax → qScaleDs = FP8_MAX / amax → Muls量化
- 通过ssbuf传递qScaleDs给Cube侧（mm3/mm4/mm5的Fixpipe需要deqScalar）
- `FixpipeParams.quantPre = QuantMode_t::QF322F32_PRE`

### 11.2 Rope旋转位置编码

- mm2的Q/K分为两段：前128列（基础Q/K）+ 后64列（rope部分）
- `mm2Ka/mm2Kb = 原始Ka/3 << 1`（2/3比例：128基础+64rope=192，rope部分占1/3）
- mm3/mm4/mm5的offset需要单独计算（`queryOffsetWithRope / keyOffsetWithRope`）

### 11.3 IS_D_NO_EQUAL（D_v ≠ D_q）

- dSizeV ≠ dSize，mm1使用dSizeV，mm2/mm3/mm4使用dSize
- valueOffset/dyOffset使用dSizeV stride
- dv输出使用dSizeV维度

### 11.4 BN2_MULTIBLK

- 同一个(b,n2,g)下，s1轴分多个block处理
- dq仅在`isLastS1Outer`时输出GM（先累加在workspace）
- dk/dv仅在`!isNextS2IdxNoChange`时输出GM
- dq offset使用 `cBlockIdx * AlignTo128(s1Size) * HEAD_DIM_ALIGN + s1oIdx * CUBE_BASEM * HEAD_DIM_ALIGN`

### 11.5 Sink注意力

- `sinkOptional`标记是否启用sink模式
- `sinkGm`存储sink缩放因子
- ProcessVecSink: 计算 dsink = softmaxGrad结果 × sinkScale
- dsink写入workspace GM后再由特定逻辑汇总

---

## 12. L0C Resident策略（IS_DKV_RESIDENT_L0C）

当dk/dv维度较小（如D≤128），L0C空间足够让dk/dv常驻：

```
mm1Mm2Mm3L0CBuf: CUBE_BASEM × min(CUBE_BASEN, HEAD_DIM_ALIGN) × sizeof(float)
dkL0CBuf:         CUBE_BASEN × HEAD_DIM_ALIGN × sizeof(float)
dvL0CBuf:         CUBE_BASEN × HEAD_DIM_ALIGN × sizeof(float)

优势：
- dk/dv在L0C中跨s2o累加，无需每轮Fixpipe到GM再AtomicAdd
- 减少GM写入次数和Atomic开销
- 适用于小D场景（D≤128或D≤192/Dv≤128）
```

D192/Dv128特殊场景：
```
mm1Mm2Mm3L0CSpecialBuf: CUBE_BASEM × 192 × sizeof(float)  // 98304B
dkL0CSpecialBuf:         CUBE_BASEN × 192 × sizeof(float)  // 98304B
dvL0CSpecialBuf:         CUBE_BASEN × 128 × sizeof(float)  // 65536B
```