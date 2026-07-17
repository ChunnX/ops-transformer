# lightning_indexer RadixTopK 迁移 — 详细修改清单

## 一、新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `op_kernel/arch22/radix_topk_inline.h` | ~250行 | **新增**: 内联 RadixTopK 实现。包含 `InlineRadixTopK()` 函数,实现 8 轮 2-bit 基数排序 + 多 tile 直方图累积 + GatherMask 提取 TopK。无 SyncAll,直方图在 UB 内累积。 |

### `radix_topk_inline.h` 核心逻辑

```
1. 从 GM 读入 bf16 score tile (DataCopyPad)
2. TwiddleInB16: bf16 → int16 可比较格式 (ShiftRight + Or + Xor)
3. DoAndMask: 提取当前轮的 2 bit (And)
4. Cast: int16 → half → float → int32 (中转)
5. CalcCumsumHistogram: Compares(EQ) + GatherMask 统计 4 个 bin
6. 多 tile 直方图累积到 globalHist (UB 内,无 SyncAll)
7. FindBoundaryBin: 找累积量 >= remainK 的 bin
8. 更新 involvedMask + remainK
9. 重复 1-8 共 8 轮 (round 7→0)
10. 逆 Twiddle 得到边界值
11. TileTopK: Compares(GT) + GatherMask 提取 > 边界值的元素
12. Compares(EQ) + GatherMask 补齐 = 边界值的元素
13. CopyOut indices 到 GM
```

---

## 二、`op_kernel/lightning_indexer_common.h` (修改 3 处)

### 2.1 移除

```diff
-    bool isLDOpen = false;
```

LD 功能不再需要,移除 `isLDOpen` 字段。

### 2.2 新增 RadixTopK 参数

```diff
     bool isSparseCountOver2K = false;
-    bool isLDOpen = false;
     bool returnValueFlag = false;
     bool splitMFlag = false;
+
+    // RadixTopK inline params
+    uint32_t radixSortLen = 0;
+    uint32_t radixSortLenAlign = 0;
+    uint64_t scoreB16GmOffset = 0;
```

| 字段 | 类型 | 用途 |
|------|------|------|
| `radixSortLen` | uint32_t | RadixTopK 排序长度 (= S2) |
| `radixSortLenAlign` | uint32_t | 对齐后的排序长度 (Align(S2, 256)) |
| `scoreB16GmOffset` | uint64_t | scoreB16Gm 在 workspace 中的偏移量 |

---

## 三、`op_host/lightning_indexer_tiling.h` (修改 1 处)

### 3.1 TilingData 新增 3 个字段

```diff
 TILING_DATA_FIELD_DEF(uint32_t, returnValue)
+TILING_DATA_FIELD_DEF(uint32_t, radixSortLen)
+TILING_DATA_FIELD_DEF(uint32_t, radixSortLenAlign)
+TILING_DATA_FIELD_DEF(uint64_t, scoreB16GmOffset)
 END_TILING_DATA_DEF
```

---

## 四、`op_host/lightning_indexer_tiling.cpp` (修改 3 处)

### 4.1 `LiCalcWorkspaceSize()` 函数签名和实现

**参数变更:**

```diff
-static uint64_t LiCalcWorkspaceSize(const platform_ascendc::PlatformAscendC &liPlatform,
-                                    int64_t liS2Size, uint32_t liAicNum)
+static uint64_t LiCalcWorkspaceSize(const platform_ascendc::PlatformAscendC &liPlatform,
+                                    int64_t liS2Size, uint32_t liBlockDim,
+                                    uint32_t liBSize, uint32_t liS1Size, uint32_t liN2Size)
```

**移除 LD workspace 计算:**

```diff
-    constexpr uint32_t liV1ResElemSize = 4;
-    constexpr uint32_t liV1ResElemType = 2;
-    constexpr uint32_t liV1DecodeParamElemSize = 8;
-    constexpr uint32_t liV1DecodeParamNum = 16;
-    constexpr uint32_t liV1DecodeDataNum = 2;
-    constexpr uint32_t liS1BaseSize = 8;
-    constexpr uint32_t liTopkMaxSize = 2048;
```

**aicNum → blockDim:**

```diff
-            sizeof(uint32_t) * liAicNum;
+            sizeof(uint32_t) * liBlockDim;
```

```diff
-        liWorkspaceSize += liMm1ResSize * liMm1ResElemSize * liDoubleBuffer * liAicNum;
-        liWorkspaceSize +=
-            liV1DecodeDataNum * liS1BaseSize * liV1ResElemType * liTopkMaxSize * liV1ResElemSize * liAicNum;
-        liWorkspaceSize +=
-            liV1DecodeDataNum * liS1BaseSize * liV1DecodeParamNum * liV1DecodeParamElemSize * liAicNum;
+        liWorkspaceSize += liMm1ResSize * liMm1ResElemSize * liDoubleBuffer * liBlockDim;
```

**新增 scoreB16Gm 大小:**

```diff
+    // scoreB16Gm: [totalBatch, sortLenAlign] of bf16
+    uint32_t sortLen = static_cast<uint32_t>(liS2Size);
+    uint32_t sortLenAlign = ((sortLen + 255) / 256) * 256;
+    uint32_t totalBatch = liBSize * liS1Size * liN2Size;
+    liWorkspaceSize += static_cast<uint64_t>(totalBatch) * sortLenAlign * sizeof(uint16_t);
```

### 4.2 `DoTiling()` 新增 scoreB16Gm offset 计算

```diff
+    // -------------compute scoreB16Gm offset-----------------
+    uint32_t sortLen = static_cast<uint32_t>(tilingInfo->s2Size);
+    uint32_t sortLenAlign = ((sortLen + 255) / 256) * 256;
+    uint64_t libApiWs = ascendcPlatform.GetLibApiWorkSpaceSize();
+    uint64_t mm1ResTotal = 0;
+    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
+        constexpr uint32_t li3510S1Base = 4;
+        constexpr uint32_t li3510S2Base = 128;
+        mm1ResTotal = li3510S1Base * ((tilingInfo->s2Size + li3510S2Base - 1) / li3510S2Base) *
+                      li3510S2Base * sizeof(uint32_t) * blockDim;
+    } else {
+        constexpr uint32_t liMBaseSize = 512;
+        constexpr uint32_t liS2BaseSize = 512;
+        mm1ResTotal = liMBaseSize * liS2BaseSize * 4 * 2 * blockDim;
+    }
+    uint64_t scoreB16GmOffset = libApiWs + mm1ResTotal;
```

### 4.3 `DoTiling()` 新增 TilingData 赋值

```diff
+    tilingData_.set_radixSortLen(sortLen);
+    tilingData_.set_radixSortLenAlign(sortLenAlign);
+    tilingData_.set_scoreB16GmOffset(scoreB16GmOffset);
```

---

## 五、`op_kernel/arch22/lightning_indexer_service_vector.h` (重大重写)

### 5.1 移除的内容

**移除的方法:**

| 方法 | 行数 | 说明 |
|------|------|------|
| `ProcessLD()` | ~170行 | 跨核 TopK 归约(MrgSort4 合并各核局部 TopK) |
| `InitLDBuffers()` | ~6行 | LD 阶段 UB buffer 初始化 |

**移除的成员变量:**

| 变量 | 类型 | 说明 |
|------|------|------|
| `vec1ResGm` | `GlobalTensor<float>` | LD TopK 中间结果 GM |
| `vec1ParamGm` | `GlobalTensor<int64_t>` | LD 参数 GM |
| `valueOutGm` | `GlobalTensor<K_T>` | value 输出 GM |
| `sortOutBuf_` | `TBuf<VECCALC>` | globalTopkUb (64KB) |
| `indexBuf_` | `TBuf<VECCALC>` | 有序索引 (2KB) |
| `paramBuf_` | `TBuf<VECCALC>` | LD 参数 (128B) |
| `ldToBeMrgBuf_` | `TBuf<>` | LD 待归并 (64KB) |
| `ldTmpBuf_` | `TBuf<>` | LD 临时 (64KB) |
| `ldOutValueBuf_` | `TBuf<>` | LD value 输出 (8KB) |
| `ldOutIdxBuf_` | `TBuf<>` | LD index 输出 (8KB) |
| `globalTopkIndice_` | `LocalTensor<int32_t>` | TopK 索引 |
| `globalTopkUb_` | `LocalTensor<float>` | TopK 结果 |
| `SortedBasicBlock_` | `LocalTensor<float>` | 排序基本块 |
| `globalTopkNum_` | `int32_t` | TopK 计数 |
| `blockS2StartIdx_` | `int64_t` | S2 起始索引 |
| `mrgListNum_` | `uint32_t` | MrgSort 路数 |
| `paramNum_` | `uint32_t` | LD 参数数量 |
| `virTopK` | `int32_t` | 虚拟 TopK 大小 |

**移除的常量:**

```diff
-constexpr uint32_t LD_PARAM_NUM = 16;
```

**移除的 InitBuffers 初始化:**

```diff
-    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float));    // 64KB
-    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                // 2KB
-    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));
-    // ... globalTopkIndice_, globalTopkUb_, SortedBasicBlock_ 初始化
-    // ... ArithProgression, InitSortOutBuf, vec1ParamGm 清零
```

**移除的 ProcessVec 中的 TopK 逻辑 (~200行):**

```diff
-    // Sort/MergeSort 逻辑
-    LIServiceVec::SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign);
-    LIServiceVec::MergeSort(globalTopkUb_[...], virTopK, reduceOutBuff, cuS2LenVecAlign, UbTmpSort);
-    // SparseTopK 逻辑
-    SparseTopK(globalTopkUb_[...], SortedBasicBlock_[...], tmpSortBuf, BASE_TOPK, ...);
-    // Extract + CopyOut 逻辑
-    Extract(outValueUb, outIdxUb, globalTopkUb_[...], (offset / 32));
-    // 写 workspace LD 参数
-    CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
-    CopyOut(vec1ResGm[wsOffset], globalTopkUb_[...], 2 * BASE_TOPK);
-    // 无效 S1 清理
-    CleanInvalidOutput(...);
```

**移除的 ProcessLD 完整实现 (~170行):**

```diff
-    // MrgSort4 跨核归约
-    MrgSort(tmpUb, srcList, params);
-    DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
-    // Extract + CopyOut
-    Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
-    DataCopyPad(indiceOutGm[outOffset], idxULocal1, ...);
-    DataCopyPad(valueOutGm[outOffset], valueULocal1, ...);
```

### 5.2 新增的内容

**新增成员变量:**

| 变量 | 类型 | 说明 |
|------|------|------|
| `scoreB16Gm` | `GlobalTensor<bfloat16_t>` | 存储 bf16 score 的 GM tensor |

**修改 `InitVec1GlobalTensor` 接口:**

```diff
-    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
-                                                GlobalTensor<float> vec1ResGm,
-                                                GlobalTensor<int64_t> vec1ParamGm,
-                                                GlobalTensor<W_T> weightsGm,
-                                                GlobalTensor<int32_t> indiceOutGm,
-                                                GlobalTensor<K_T> valueOutGm);
+    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
+                                                GlobalTensor<W_T> weightsGm,
+                                                GlobalTensor<int32_t> indiceOutGm,
+                                                GlobalTensor<bfloat16_t> scoreB16Gm);
```

**修改 `InitBuffers`:**

```diff
-    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
-    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
-    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
-    virTopK = constInfo_.isSparseCountOver2K ? constInfo_.sparseCount : BASE_TOPK;
+    uint32_t outNeedBufSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
+    uint32_t reduceCacheSize = outNeedBufSize;

-    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);
-    pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));
-    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float));    // 64KB
-    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                // 2KB
-    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 2 * sizeof(float));                          // 4KB
-    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
-    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));
+    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);
+    pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));
+    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 2 * sizeof(float));
+    pipe->InitBuffer(brcBuf_, (groupInner_ * 8 + s2BaseSize_) * sizeof(float));  // 扩大,容纳 Cast 中间结果
```

**新增 `ProcessVec` 中的 Cast + CopyOut 逻辑:**

```cpp
// 因果掩码: 无效 S2 填 -inf
if (cuS2Len < s2BaseSize_) {
    Duplicate(reduceOutInner[cuS2Len].template ReinterpretCast<int32_t>(),
              LIServiceVec::NEG_INF, s2BaseSize_ - cuS2Len);
    PipeBarrier<PIPE_V>();
}

// Cast float score to bfloat16, 写入 brcBuf_ 第二部分
LocalTensor<bfloat16_t> scoreB16Ub = brcBuf[groupInner_ * 8].template ReinterpretCast<bfloat16_t>();
Cast(scoreB16Ub, reduceOutInner, RoundMode::CAST_ROUND, s2BaseSize_);
PipeBarrier<PIPE_V>();

// CopyOut score to scoreB16Gm
uint64_t globalS1Idx = info.bIdx * constInfo_.qSeqSize + cuS1Idx;
uint64_t scoreOffset = globalS1Idx * constInfo_.radixSortLenAlign + cuBaseS2Idx;
SetFlag<HardEvent::V_MTE3>(EVENTID_V_TO_MTE2_TMPUB);
WaitFlag<HardEvent::V_MTE3>(EVENTID_V_TO_MTE2_TMPUB);
LIServiceVec::CopyOut(scoreB16Gm[scoreOffset], scoreB16Ub, s2BaseSize_);
SetFlag<HardEvent::MTE3_V>(EVENTID_V_TO_MTE2_TMPUB);
```

### 5.3 ProcessVec 核心变化

```
原始: CopyIn → DoScale → DoReduce → Sort/MergeSort → CopyOut TopK / 写 workspace LD
修改后: CopyIn → DoScale → DoReduce → 填 -inf(因果掩码) → Cast(float→bf16) → CopyOut(scoreB16Gm)
```

---

## 六、`op_kernel/arch22/lightning_indexer_kernel.h` (修改 5 处)

### 6.1 新增 include

```diff
 #include "lightning_indexer_service_cube.h"
+#include "radix_topk_inline.h"
```

### 6.2 成员变量变更

```diff
     // workspace
     GlobalTensor<MM1_OUT_T> mm1ResGm;  // 存放S
-    GlobalTensor<float> vec1ResGm;     // 存放TopK计算中间结果
-    GlobalTensor<int64_t> vec1ParamGm; // 存放LD参数信息
+    GlobalTensor<bfloat16_t> scoreB16Gm; // 存放bf16 score
+    __gm__ uint8_t *workspaceBase = nullptr;
+    __gm__ uint8_t *sparseIndicesBase = nullptr;
```

### 6.3 方法声明变更

```diff
-    __aicore__ inline void ProcessDecode();
+    __aicore__ inline void ProcessRadixTopK();
```

### 6.4 InitTilingData 新增

```diff
     constInfo.returnValue = tilingData->returnValue;
+    constInfo.radixSortLen = tilingData->radixSortLen;
+    constInfo.radixSortLenAlign = tilingData->radixSortLenAlign;
+    constInfo.scoreB16GmOffset = tilingData->scoreB16GmOffset;
```

### 6.5 Init() 变更

```diff
-    // ld流程需要ws大小: [aicnum, 2, ...]
-    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
-    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * WS_DOUBLE * BASE_TOPK * sizeof(float);
-    vec1ParamGm.SetGlobalBuffer((__gm__ int64_t *)(workspace + offset));
-    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * LD_PARAM_NUM * sizeof(int64_t);
+    // scoreB16Gm: [totalBatch, sortLenAlign] of bf16
+    workspaceBase = workspace;
+    scoreB16Gm.SetGlobalBuffer((__gm__ bfloat16_t *)(workspace + constInfo.scoreB16GmOffset));
```

```diff
     if ASCEND_IS_AIV {
         vectorService.InitParams(constInfo, tiling);
+        sparseIndicesBase = sparseIndices;
         indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
-        valueOutGm.SetGlobalBuffer((__gm__ K_T *)sparseValues);
         weightsGm.SetGlobalBuffer((__gm__ W_T *)weights);
-        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, vec1ParamGm, weightsGm, indiceOutGm, valueOutGm);
+        vectorService.InitVec1GlobalTensor(mm1ResGm, weightsGm, indiceOutGm, scoreB16Gm);
```

### 6.6 Process() 变更

```diff
     ProcessMain();
-    ProcessDecode();
+    ProcessRadixTopK();
```

### 6.7 新增 ProcessRadixTopK() (~60行)

```cpp
template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessRadixTopK()
{
    if ASCEND_IS_AIV {
        SyncAll();           // 确保 Phase 1 所有 score 写入完成
        pipe->Reset();       // 释放 Phase 1 UB

        // 计算 S1 token 分配
        uint32_t aivIdx = tmpBlockIdx;
        uint32_t aivTotal = GetBlockNum() * 2;
        uint32_t batchPerCore = (totalBatch + aivTotal - 1) / aivTotal;
        uint32_t batchStart = aivIdx * batchPerCore;
        uint32_t batchEnd = batchStart + batchPerCore;

        __gm__ uint8_t *scoreBase = workspaceBase + constInfo.scoreB16GmOffset;

        for (uint32_t batchId = batchStart; batchId < batchEnd; batchId++) {
            // 因果掩码检查
            uint32_t actS1, actS2;
            GetS1S2ActualSeqLen(bIdx, actS1, actS2);
            int32_t cuRealAcSeq = attenMaskFlag ?
                (actS2 - (actS1 - s1Idx)) : actS2;

            if (cuRealAcSeq <= 0) {
                // 无效 token: 填 -1
                __gm__ int32_t *outIdx = ...;
                for (uint32_t i = 0; i < kValue; i++) { outIdx[i] = -1; }
                continue;
            }

            // 调用内联 RadixTopK
            __gm__ uint8_t *scorePtr = scoreBase + batchId * sortLenAlign * sizeof(bfloat16_t);
            __gm__ uint8_t *idxPtr = sparseIndicesBase + batchId * kValue * sizeof(int32_t);
            InlineRadixTopK(pipe, scorePtr, idxPtr, sortLen, sortLenAlign,
                           kValue, tileLen, false, dummyValPtr);
        }
        SyncAll();           // 确保所有 AIV 完成
    } else {
        SyncAll();           // AIC 等待 Phase 1
        SyncAll();           // AIC 等待 Phase 2
    }
}
```

### 6.8 移除 ProcessDecode() (~10行)

```diff
-template <typename LIT>
-__aicore__ inline void LightningIndexerKernel<LIT>::ProcessDecode()
-{
-    if ASCEND_IS_AIV {
-        vectorService.InitLDBuffers(pipe);
-        ICachePreLoad(LD_PREFETCH_LEN);
-        SyncAll();
-        if (splitCoreInfo.isLD) {
-            vectorService.ProcessLD();
-        }
-    }
-}
```

---

## 七、架构变化总结

### 数据流对比

```
原始:
  Cube: Q@K^T+ReLU → mm1ResGm (float)
  Vector: CopyIn → DoScale → DoReduce → Sort/MergeSort → CopyOut TopK / 写 workspace
  ProcessLD: MrgSort4 跨核归约 → CopyOut 最终结果

修改后:
  Phase 1 (ProcessMain):
    Cube: Q@K^T+ReLU → mm1ResGm (float) [不变]
    Vector: CopyIn → DoScale → DoReduce → 填-inf → Cast(float→bf16) → CopyOut scoreB16Gm
  SyncAll
  Phase 2 (ProcessRadixTopK):
    pipe->Reset → InlineRadixTopK (8轮 radix + GatherMask) → CopyOut indices
  SyncAll
```

### UB 分配对比

| Phase 1 UB | 原始 | 修改后 |
|------------|------|--------|
| outQueue_ | 32KB (Extract buffer) | ~8KB (缩小) |
| tmpBuf_ | 68KB (DoScale DB) | 68KB (不变) |
| sortOutBuf_ | 64KB (globalTopkUb) | 移除 |
| indexBuf_ | 2KB (有序索引) | 移除 |
| reduceOutBuf_ | 4KB | 4KB (不变) |
| brcBuf_ | 512B | ~6KB (扩大,容纳 Cast) |
| paramBuf_ | 128B | 移除 |
| LD buffers | 144KB | 移除 |
| **Phase 1 总计** | ~171KB | ~86KB |

| Phase 2 UB (pipe->Reset 后) | 大小 |
|------------------------------|------|
| scoreBuf | 8KB |
| twiddleBuf | 8KB |
| histBuf | 32B |
| cmpMaskBuf | 512B |
| indexBuf | 16KB |
| outIdxBuf | 8KB |
| tempBuf | 16KB |
| **Phase 2 总计** | ~57KB |

### Workspace 对比

```
原始:
  |mm1ResGm|vec1ResGm|vec1ParamGm|

修改后:
  |mm1ResGm|scoreB16Gm|
```

---

## 八、精度和性能结果

### 精度对比

| 场景 | 原始实现 | RadixTopK 迁移后 |
|------|---------|-----------------|
| S2=3072 匹配率 | 98.50% | 98.03% |

### 性能对比

| S2 | 原始实现 | RadixTopK 迁移后 | 变化 |
|----|---------|-----------------|------|
| 3072 | 0.13 ms | 0.13 ms | 持平 |
| 4096 | 0.13 ms | 0.14 ms | +8% |
| 6144 | 0.14 ms | 0.16 ms | +14% |
| 8192 | 0.16 ms | 0.17 ms | +6% |
| 128K (B=1,S1=1) | 0.14 ms | 0.21 ms | +50% |
| 128K (B=1,S1=3) | 0.17 ms | 0.24 ms | +41% |
