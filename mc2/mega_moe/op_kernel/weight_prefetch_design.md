# 权重前移方案设计

> 基于 `arch35` 源码分析，将 topkWeights 加权乘从 Unpermute 尾阶段前移到激活量化阶段

## 1. 当前流程

```
GMM1 → SwiGLU → Quant → GMM2 → Combine归还 → Unpermute(Muls+Add加权求和) → Output
```

权重乘与求和在 `Unpermute`（`mega_moe.h:1126`）的串行循环里耦合执行：

```cpp
for (expId = 0; expId < topK_; ++expId) {
    float expScale = topKWeightsTensor_.GetValue(tokenIdx * topK_ + expId);
    Muls(dataInFp32, dataInFp32, expScale, k_);   // 权重乘
    Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);  // 求和
}
```

---

## 2. 权重前移到激活量化阶段

### 2.1 新流程

```
GMM1 → SwiGLU → w*Y → Quant(w*Y) → GMM2 → Combine归还 → Unpermute(纯Add求和) → Output
```

将 per-row 标量乘 `Muls(gluRes_, gluRes_, weight, nSize)` 插入 `VFDoSwigluAndQuantForMX`（`block_epilogue_swiglu_mx_quant.h:629`），SwiGLU 之后、quant 之前。

### 2.2 精度分析：MXFP scale 吸收

MXFP 量化是 per-group（32元素一组），weight 是 per-row 标量。设某 group 内 SwiGLU 输出为 `Y`，weight 为 `w`：

| 量 | 无权重 | 有权重 |
|---|---|---|
| max_abs | `M = max(|Y|)` | `|w| * M` |
| scale | `S = M / 448` | `S' = |w| * S` |
| quantized | `round(Y / S)` | `round(w*Y / (|w|*S)) = sign(w) * round(Y/S)` |

量化值只差 `sign(w)`，scale 吸收了 `|w|`。在 GMM2 的 MMAD 中：

```
sum(quantized' * quantized_W2 * S' * S_W2)
= sum(sign(w) * quantized * quantized_W2 * |w| * S * S_W2)
= w * sum(quantized * quantized_W2 * S * S_W2)
= w * 原GMM2结果
```

MXFP scale 吸收机制保证了权重前移操作本身不会造成精度劣化（前提：topkweights 来自 softmax，`|w| ≤ 1`，不会导致 max_abs 溢出）。但权重乘改变了后续量化、GMM2 等多轮操作的输入数据分布，整体数值会发生变化，后续精度对比结果可能不同于原流程，这不属于前移操作引入的精度劣化。

### 2.3 收益

1. **融入已有 element-wise 流水**：line 629 处 `gluRes_` 已在 UB，SwiGLU 刚做完一串 element-wise 操作，再加一个 per-row `Muls` 完全融合，零额外 pipeline stage
2. **Unpermute 退化为纯加法**：移除 `topKWeightsTensor_` 加载和 `Muls`，保留 `Add`
3. **负载均衡**：权重乘从串行尾阶段移到并行中阶段（M_tile×N_tile 块多核并行）

---

## 3. 使能控制：属性与 TilingData

### 3.1 算子属性 `topk_weights_mode`

在 `mega_moe_def.cpp` 中新增算子属性 `topk_weights_mode`，控制是否开启权重前移：

```cpp
// mega_moe_def.cpp，在现有 Attr 之后新增
this->Attr("topk_weights_mode").AttrType(OPTIONAL).Int(0);
// 0: 不开启权重前移（默认，保持原流程）
// 1: 开启权重前移
```

- 属性类型：`OPTIONAL`，默认值 `0`（不开启），保证向后兼容
- 属性位置：追加到现有属性列表末尾（`topo_type` 之后）

### 3.2 TilingData 字段 `is_weight_prefetch`

在 `MegaMoeTilingData`（`mega_moe_tiling.h`）中新增 `is_weight_prefetch` 字段，由 Tiling 阶段从属性读取并下发给 Kernel：

```cpp
// mega_moe_tiling.h
struct MegaMoeTilingData {
    uint32_t expertPerRank;
    uint32_t bs;
    uint32_t h;
    uint32_t hiddenDim;
    uint32_t epWorldSize;
    uint32_t blockNumPerEP;
    uint32_t maxOutputSize;
    uint32_t topK;
    uint32_t aicNum;
    uint32_t blockAivNum;
    int64_t combineQuantMode;
    float clampLimit;
    int64_t topoType;
    uint8_t groupedMatmulMode;
    // 新增：权重前移开关
    uint8_t isWeightPrefetch;       // 1B，0=关闭，1=开启
    uint8_t reserved[2];
};
```

### 3.3 Kernel 侧使用

Kernel 侧通过 `tilingData->isWeightPrefetch` 判断是否走权重前移路径：

```cpp
// mega_moe.h Init 中
isWeightPrefetch_ = (params_.tilingData->isWeightPrefetch == 1);

// 各改动点用 if constexpr 或 if (isWeightPrefetch_) 分支
if (isWeightPrefetch_) {
    // 权重前移路径：QuantProcessInRank 追加 weight 搬运、CopyGMToGMPerToken 拉取 weight、
    //               激活量化插入 Muls、Unpermute 纯 Add
} else {
    // 原路径：保持不变
}
```

---

## 4. 权重数据流设计

### 3.1 核心约束

- 源卡 `topkWeights` 在 GM 的 `probsGmAddr`，**其他卡无法直接读源卡 GM**
- 跨卡只能读源卡 peermem win 区
- 因此必须把权重从 GM 搬到 win 区，dispatch 拉取 token 时一并拉回

### 3.2 数据流总览

```
源卡 GM                  源卡 pemem win 区              专家卡本地 GM
┌──────────────┐         ┌────────────────────┐        ┌─────────────────────┐
│ token (bf16)  │→量化→   │ [data|scale|weights]│←拉取─ │ dispatchRevDataPtr  │ 行0
│ topkWeights   │→搬出→   │ (per-token 追加)    │       │ dispatchRevScalePtr │ 行0
│ (float)       │         │                    │        │ weightRecvPtr       │ 行0
└──────────────┘         └────────────────────┘        └─────────────────────┘
                                                          ↓ GMM1 + SwiGLU
                                                          ↓ 激活量化: weight[mLoc+mIdx] * gluRes
                                                          ↓ GMM2 + Combine + Unpermute(纯Add)
```

---

## 5. 改动点详解

### 4.1 改动点 1：源卡 peermem 布局（PeermemInfo）

**文件**：`mega_moe_base.h:44-90`

**当前**：`quantTokenScalePtr` 每 token 条目 = `[data | scale]`

**改动**：每 token 条目追加 `topK` 个 float weight，对齐 32B

```
当前 per-token:
┌──────────────────────────┬─────────────────────┐
│ 量化token数据             │ e8m0 scale           │
│ (mxQuantTokenAlignBytes_)│ (mxQuantScaleAlignB) │
└──────────────────────────┴─────────────────────┘

新增后:
┌──────────────────────────┬─────────────────────┬──────────────────────┐
│ 量化token数据             │ e8m0 scale           │ w0,w1,...,w_{topK-1} │
│ (mxQuantTokenAlignBytes_)│ (mxQuantScaleAlignB) │ (weightAlignBytes_)  │
└──────────────────────────┴─────────────────────┴──────────────────────┘
```

`weightAlignBytes_ = CeilAlign(topK * sizeof(float), 32B)`

以 topK=8 为例：`8*4=32B`，恰好 32B 对齐，每 token 仅多 32 字节。

**PeermemInfo 构造函数修改**：

```cpp
// mega_moe_base.h:75-76 当前
uint32_t tokenBytes = CeilAlign(dataBytes + scaleBytes, ALIGN_32);

// 修改后
uint32_t weightBytes = CeilAlign(topK * sizeof(float), ALIGN_32);
uint32_t tokenBytes = CeilAlign(dataBytes + scaleBytes + weightBytes, ALIGN_32);
```

后续 `combineSendPtr` 起址自动后移。

### 5.2 改动点 2：源卡权重搬运（QuantProcessInRank）

**文件**：`mega_moe.h:540-597`

**当前**：逐 token 量化，数据路径为 GM(token) → UB(量化) → peermem，写 `[data|scale]` 到 peermem

**改动**：weight 与 token/scale 一样走 GM→UB→peermem 路径，在同一循环中追加 weight 的搬运

源码现状的双 buffer 流水（`mega_moe.h:560-594`）：

```
MTE2: GM token → UB (xInTensor)
V:   UB token → 量化 → UB (xOutTensor: [data|scale])
MTE3: UB (xOutTensor) → peermem
```

改造后，weight 走相同路径，融入现有双 buffer 流水：

```cpp
// 新增：weight 相关常量（Init 中）
weightAlignBytes_ = CeilAlign(topK_ * sizeof(float), ALIGN_32);

// QuantProcessInRank 改造
for (index = 0; index < curentNum; index++) {
    auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
    auto xInTensor = (index % DOUBLE_BUFFER == 0) ? xInTensor1_ : xInTensor2_;
    auto xOutTensor = (index % DOUBLE_BUFFER == 0) ? xOutTensor1_ : xOutTensor2_;

    WaitFlag<MTE3_MTE2>(event);

    // === MTE2: GM → UB ===
    // (1) 现有：读 token
    DataCopyPad(xInTensor, srcGlobalTensor[aGmAddr + index*H], ...);
    // (2) 新增：读 weight（与 token 的 MTE2 顺序发出，复用同一 event）
    GlobalTensor<float> weightGm;
    weightGm.SetGlobalBuffer((__gm__ float*)(probsGmAddr + index * topK_ * sizeof(float)));
    LocalTensor<float> weightUb = xOutTensor[weightOffset].ReinterpretCast<float>();
    // weightOffset = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_
    DataCopyPad(weightUb, weightGm, {1, topK_ * sizeof(float), 0, 0, 0});

    SetFlag<MTE2_V>(event);
    WaitFlag<MTE2_V>(event);

    // === V: 量化 ===
    // 现有：token 在 xInTensor → 量化 → 结果写 xOutTensor 的 [data|scale] 区
    // weight 已在 xOutTensor 的 weight 区（MTE2 直接写入），V 不触碰 weight 区
    quant::ComputeMaxExp / ComputeScale / ComputeFp8Data;

    SetFlag<V_MTE3>(event);
    WaitFlag<V_MTE3>(event);

    // === MTE3: UB → peermem ===
    // 一次写出 [data|scale|weights]，xCopyOutParams.blockLen 增大
    auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
    DataCopyPad(dstGlobalTensor, xOutBytesTensor, xCopyOutParams);

    SetFlag<MTE3_MTE2>(event);
}
```

**数据路径**：

```
GM token     ──MTE2──→ UB xInTensor  ──V(量化)──→ UB xOutTensor[data|scale]
GM weight    ──MTE2──→ UB xOutTensor[weight]                     │
                                                               MTE3
                                                                 ↓
                                                          peermem [data|scale|weight]
```

**关键点**：
- weight 走 GM→UB→peermem 三级路径，与 token/scale 一致，不能直接 GM→peermem
- weight 的 MTE2 读入 `xOutTensor` 的 weight 区（量化产物之后），与 token 的 MTE2（读入 `xInTensor`）复用同一 event，不增加额外流水线 stage
- V 量化只写 `xOutTensor` 的 `[data|scale]` 区，不触碰 weight 区，两者在 UB 内分区共存
- MTE3 一次写出完整的 `[data|scale|weight]`，`blockLen` 增大 `weightAlignBytes_`

### 5.3 改动点 3：dispatch 拉取（CopyGMToGMPerToken）

**文件**：`mega_moe.h:668-749`

**当前流程**：6-buffer 软流水，MTE2 把 `[data|scale]` 一起读入 UB 的 `copyTmpTensors_[outIdx]`，MTE3 阶段在 UB 内按偏移分离，用两次 `DataCopyPad` 分别写到本地 GM 的 `tokenRevGlobalTensor` 和 `scaleRevGlobalTensor`：

```cpp
// 现有 MTE3 分离写入 (mega_moe.h:722-728)
LocalTensor<ActivationType> tokenScalebuf = copyTmpTensors_[outIdx];
LocalTensor<QuantScaleOutType> bufScale =
    tokenScalebuf[mxQuantTokenAlignBytes_].ReinterpretCast<QuantScaleOutType>();
DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScalebuf, ...);   // token 数据
DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], bufScale, ...);   // scale
```

**改动**：`copyInNum` 增大 `weightAlignBytes_`，MTE2 一次读入 `[data|scale|weights]`。token 和 scale 仍用 `DataCopyPad` 分离搬出；weight 只有 1 个 float（从 topK 个中取第 `topkIdx` 个），用 `DataCopyPad` 单独搬 4 字节效率太低，改用 `GetValue`/`SetValue` 标量操作：

```cpp
// copyInNum 增大
copyInNum = CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_ + weightAlignBytes_, ALIGN_32);

// MTE3 拆分写入（steady 阶段，mega_moe.h:722-728 改造）
LocalTensor<ActivationType> tokenScalebuf = copyTmpTensors_[outIdx];
LocalTensor<QuantScaleOutType> bufScale =
    tokenScalebuf[mxQuantTokenAlignBytes_].ReinterpretCast<QuantScaleOutType>();

// 现有：token 和 scale 用 DataCopyPad 分离搬出
DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScalebuf,
    {1, widthA * sizeof(ActivationType), 0, 0, 0});          // token 数据 → 行 copyIdx
DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], bufScale,
    {1, widthAScale * sizeof(QuantScaleOutType), 0, 0, 0});  // scale → 行 copyIdx

// 新增：weight 用 GetValue/SetValue 标量取值存值
// weightOffset = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_
LocalTensor<float> bufWeights = tokenScalebuf[weightOffset].ReinterpretCast<float>();
int32_t topkIdx = tripleTensor_[copyIdx * INT32_PER_256B].GetValue(TOPK_INDEX);  // 从三元组取 topkIdx
float weight = bufWeights.GetValue(topkIdx);      // 从 topK 个 weight 中取第 topkIdx 个
weightRecvGlobalTensor.SetValue(copyIdx, weight);  // 标量写入本地 GM 行 copyIdx
```

**数据路径**：

```
源卡 peermem [data|scale|weights]
       │
       │ MTE2 (一次 DataCopy 读入 copyInNum 字节)
       ▼
UB copyTmpTensors_[outIdx]:
  ┌────────────────┬──────────────┬──────────────────────┐
  │ token 数据      │ scale         │ w0,w1,...,w_{topK-1} │
  │ (tokenAlignB)  │ (scaleAlignB) │ (weightAlignBytes_)  │
  └────────────────┴──────────────┴──────────────────────┘
       │                  │                  │
       │ DataCopyPad      │ DataCopyPad      │ GetValue(topkIdx) → SetValue
       ▼                  ▼                  ▼
  tokenRevGlobal    scaleRevGlobal    weightRecvGlobalTensor[copyIdx]
     [copyIdx]         [copyIdx]         (float 标量)
```

**关键点**：

- MTE2 一次读入 `[data|scale|weights]`，6-buffer 结构不变，每 buffer 多读 `weightAlignBytes_`（32B）
- token 和 scale 仍用 `DataCopyPad` 分离搬出（批量搬运，高效）
- weight 是 per-token 单个 float，用 `GetValue`/`SetValue` 标量操作（避免 4 字节 `DataCopyPad` 低效）
- `topkIdx` 从三元组获取（`mega_moe.h:702` 已写入 `tripleTensor_`），直接索引第 `topkIdx` 个 weight
- `copyIdx` 是本地连续行号（0,1,2,...,totalToken-1），weight 与 token/scale 按相同行号对齐存储
- `GetValue`/`SetValue` 需要 `MTE2_S`/`S_MTE3` flag 同步确保数据可见性
- 6-buffer 结构完全不变，只是每个 buffer 多读 `weightAlignBytes_`（32B）

**关于冗余**：每个被拉取的 token 会带回全部 `topK` 个 weight，但只用第 `topkIdx` 个。topK=8 时冗余 28B/token，相对 4KB token 数据可忽略。这避免了为 4 字节 weight 单独发起 MTE2 的低效问题。

### 5.4 改动点 4：本地 weight 存储（workspace）

**文件**：`mega_moe_workspace_info.h`

本卡所有暂存数据（`dispatchRevDataPtr`、`dispatchRevScalePtr`、`swigluQuantDataPtr`、`tripleInfoPtr` 等）都开辟在 workspace 上，通过 `base + workspaceSize` 偏移线性分配，每块按 `ALIGN_512` 对齐。`weightRecvPtr` 应与现有暂存数据一样，在 `WorkspaceInfo` 中新增字段并分配 workspace 空间。

**WorkspaceInfo 新增字段**：

```cpp
struct WorkspaceInfo {
    // ... 现有字段 ...
    GM_ADDR weightRecvPtr;  // 新增：权重前移的本地权重存储
};
```

**构造函数中新增分配**（条件分配，仅 `isWeightPrefetch` 时）：

```cpp
// mega_moe_workspace_info.h 构造函数中，现有分区之后
weightRecvPtr = nullptr;
if (tilingData->isWeightPrefetch) {
    weightRecvPtr = base + workspaceSize;
    // 大小 = maxOutputSize * sizeof(float)，与 dispatchRevDataPtr 行数一致
    workspaceSize += Ops::Base::CeilAlign(
        SIZE_INT_32 * tilingData->maxOutputSize, ALIGN_512);
}
```

**布局**：与 `dispatchRevDataPtr` 行号完全一致，按 totalToken 行号排列的一维 float 数组：

```
dispatchRevDataPtr:    行0: token数据   ← GMM1 消费
dispatchRevScalePtr:   行0: scale       ← GMM1 消费
weightRecvPtr:         行0: weight      ← 激活量化消费
                       行1: weight
                       ...
                       行(totalToken-1): weight
```

**关键点**：
- 与其他暂存数据一样开在 workspace 上，通过 `base + workspaceSize` 分配，不临时开辟 GM Tensor
- 大小 `maxOutputSize * sizeof(float)`，与 `dispatchRevDataPtr` 的行数对齐
- 条件分配：仅 `isWeightPrefetch == 1` 时开辟，关闭时 `weightRecvPtr = nullptr`，不影响原有 workspace 布局

### 4.5 改动点 5：激活量化插入权重乘（VFDoSwigluAndQuantForMX）

**文件**：`block_epilogue_swiglu_mx_quant.h:558-648`

**当前**：SwiGLU → Quant

**改动**：SwiGLU → **Muls(weight)** → Quant

```cpp
// block_epilogue_swiglu_mx_quant.h:629 附近
// 当前：SwiGLU 结果在 gluRes_ (bf16, UB)，准备进入 quant

// 新增：权重乘
// mLoc 是当前 tile 在该专家 token 中的起始行号（从 scheduler.GetBlockCoord 获取）
// mIdx 是 tile 内行号 (0..mSize-1)
for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
    float weight = weightRecvPtr[mLoc + mIdx];
    // gluRes_ 是 bf16，weight 是 fp32，需先 Cast 或用 fp32 中间
    // 方案：在 SwiGLU 的 fp32 中间结果上乘，再 Cast 回 bf16
    Muls(swishData, swishData, weight, mask);  // 在 fp32 域做
}
// 然后进入原有 quant 逻辑
```

**`mLoc` 的来源**：`AivGmm1PostGeneric`（`mega_moe_impl.h:418`）从 `scheduler.GetBlockCoord(loopIdx)` 获取 `mLoc`，传入 epilogue。需要将其透传到 `VFDoSwigluAndQuantForMX`。

**weight 取法**：`weightRecvPtr[mLoc + mIdx]`

```
该专家 totalToken = 1000, tileM = 256

tile 0: mLoc=0,   mIdx=0..255  → weight[0..255]
tile 1: mLoc=256, mIdx=0..255  → weight[256..511]
tile 2: mLoc=512, mIdx=0..255  → weight[512..767]
tile 3: mLoc=768, mIdx=0..231  → weight[768..999]  (尾块)
```

### 4.6 改动点 6：Unpermute 退化为纯加法

**文件**：`mega_moe.h:1058-1186`

**当前**：

```cpp
// UnpermuteBuffInit: 加载 topKWeightsTensor_ (m*topK*sizeof(float))
// Unpermute: Muls + Add
for (expId = 0; expId < topK_; ++expId) {
    float expScale = topKWeightsTensor_.GetValue(tokenIdx * topK_ + expId);
    Muls(dataInFp32, dataInFp32, expScale, k_);
    Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
}
```

**改动后**：

```cpp
// UnpermuteBuffInit: 移除 topKWeightsTensor_ 相关代码
// Unpermute: 纯 Add
for (expId = 0; expId < topK_; ++expId) {
    // 移除 Muls
    Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
}
```

移除内容：
- `topKWeightsTensor_` 成员变量及 UB 分配
- `UnpermuteBuffInit` 中加载权重的 DataCopyPad/Cast（`mega_moe.h:1102-1119`）
- `Unpermute` 中的 `Muls`（`mega_moe.h:1170-1173`）

---

## 6. 数据流完整时序

```
Stage 1: QuantProcessInRank (源卡)
  GM token → 量化 → [data|scale|weights] → peermem quantTokenScalePtr
                              ↑ 新增 weights

Stage 2: SendMaskCal (源卡)
  topkIds → [mask|count] → peermem maskRecvPtr  (不变)

Stage 3: CopyGMToGMPerToken (专家卡 dispatch)
  6-buffer 软流水跨卡读:
    MTE2: peermem[tokenIdx] = [data|scale|weights] → UB
    MTE3: data → dispatchRevDataPtr[copyIdx]
          scale → dispatchRevScalePtr[copyIdx]
          weight[topkIdx] → weightRecvPtr[copyIdx]   ← 新增

Stage 4: GMM1 (专家卡)
  AIC: MMAD(dispatchRevDataPtr[行] × weight1) → L0C
  AIV0: L0C → UB → SwiGLU → w*Y → Quant → swigluQuantDataPtr
                                    ↑ 新增 w*Y (用 weightRecvPtr[mLoc+mIdx])

Stage 5: GMM2 + Combine + Unpermute (专家卡)
  GMM2: MMAD(swigluQuantDataPtr × weight2) → 结果
  Combine: 结果归还到源卡 combineSendPtr
  Unpermute: 纯 Add 求和 → output  ← 移除 Muls
```

---

## 7. 空间代价分析

以 `bs=4096, topK=8, hiddenDim=4096, worldSize=8, expertPerRank=8` 为例：

| 区域 | 当前 | 前移后 | 增量 |
|------|------|--------|------|
| peermem 每 token | ~4KB (data+scale) | ~4KB + 32B | +32B (0.8%) |
| peermem 量化区总量 | 16MB | 16MB + 128KB | +128KB |
| dispatch 每 token MTE2 | ~4KB | ~4KB + 32B | +32B (0.8%) |
| 本地 weightRecvPtr | 0 | maxOutputSize*4B | +16KB |
| UB topKWeightsTensor_ | 128KB | 0 | -128KB |

**净增**：peermem +128KB，本地 GM +16KB，UB -128KB。总体空间代价可忽略。

---

## 8. 关键设计决策

### 7.1 为什么 weight 随 token 数据发送，不拼在 mask+cnt 后

- mask+cnt 是**控制面**（固定长度二维表），weight 是**数据面**（per-token）
- 拼在 mask+cnt 后：固定长度空间爆炸（maxCount 预留膨胀 32 倍），变长破坏二维表寻址
- 随 token 数据发送：搭便车已有的 6-buffer 跨卡读路径，每 token 仅 +32B

### 7.2 为什么 per-token 追加全部 topK 个 weight，不按 topkIndex 独立排列

- per-token 追加：随 token 数据一次 DataCopy 读回，零额外 MTE2，6-buffer 不变
- 按 topkIndex 独立：每 token 多一次 4 字节 MTE2，利用率极低，破坏流水线
- 冗余代价：topK=8 时 28B/token，可忽略

### 7.3 为什么 weightRecvPtr 按行号存，不按 (tokenIdx, topkIdx) 索引

- dispatch 后 token 在本地 GM 按行号连续排列，GMM1 按行号消费
- weight 按相同行号存，激活量化按 `weightRecvPtr[mLoc + mIdx]` 直接取
- 无需 (tokenIdx, topkIdx) 反查，行号就是索引

### 7.4 为什么在 fp32 域做 Muls

- SwiGLU 中间结果 `swishData` 已是 fp32（`block_epilogue_swiglu_mx_quant.h:580`）
- weight 是 fp32，直接 `Muls(swishData, swishData, weight, mask)` 在 fp32 域做
- 乘完再 Cast 回 bf16 进入 quant，与现有流程一致

---

## 9. 改动文件清单

| 文件 | 改动内容 |
|------|---------|
| `op_host/mega_moe_def.cpp` | 新增 `topk_weights_mode` 属性（OPTIONAL Int，默认 0） |
| `op_host/op_tiling/arch35/mega_moe_tiling.cpp` | `SetAttrParams` 读取 `topk_weights_mode` 赋值 `tilingData->isWeightPrefetch` |
| `op_kernel/arch35/mega_moe_tiling.h` | `MegaMoeTilingData` 新增 `isWeightPrefetch`（uint8_t）+ `reserved[3]` 保证 4 字节对齐 |
| `op_kernel/arch35/mega_moe_base.h` | `PeermemInfo`：`tokenBytes` 增加 `weightAlignBytes` |
| `op_kernel/arch35/mega_moe.h` `Init` | 新增 `weightAlignBytes_`、`isWeightPrefetch_` 成员变量初始化 |
| `op_kernel/arch35/mega_moe.h` `QuantProcessInRank` | 新增 weight GM→peermem 搬运 |
| `op_kernel/arch35/mega_moe.h` `CopyGMToGMPerToken` | `copyInNum` 增大，MTE3 拆出 weight 存 `weightRecvPtr` |
| `op_kernel/arch35/mega_moe.h` `UnpermuteBuffInit` | 移除 `topKWeightsTensor_` 相关代码 |
| `op_kernel/arch35/mega_moe.h` `Unpermute` | 移除 `Muls`，保留 `Add` |
| `op_kernel/arch35/mega_moe_workspace_info.h` | 新增 `weightRecvPtr` GM 区域 |
| `op_kernel/arch35/block_epilogue_swiglu_mx_quant.h` | `VFDoSwigluAndQuantForMX` 插入 `Muls(weight)`，透传 `mLoc` |
| `op_kernel/arch35/mega_moe_impl.h` `AivGmm1PostGeneric` | 透传 `mLoc` 到 epilogue |

---

## 10. 风险与验证

### 9.1 精度验证

- **scale 吸收**：MXFP scale 吸收 `|w|`，quantized 差 `sign(w)`，MMAD 结果数学等价
- **边界**：`|w| ≤ 1`（softmax 输出），不会导致 max_abs 溢出
- **精度影响**：权重前移操作本身不会造成精度劣化，但改变了后续多轮操作的输入数据分布，整体精度对比结果可能变化，需实测验证是否符合精度标准

### 9.2 功能验证

- dispatch 拉取的 weight 与 token 行号对齐（`copyIdx` 一致）
- 激活量化 `mLoc + mIdx` 索引正确（与 GMM1 消费的行号范围一致）
- Unpermute 纯 Add 结果与原 Muls+Add 一致

### 9.3 性能验证

- peermem 空间增量是否影响后续区域（combineSendPtr 起址后移）
- 6-buffer 流水线是否因 `copyInNum` 增大而退化
- Unpermute 省去 Muls 和权重加载的加速效果
