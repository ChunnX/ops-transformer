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

### 3.2 TilingData 字段 `topk_weights_mode`

在 `MegaMoeTilingData`（`mega_moe_tiling.h`）中新增 `topk_weights_mode` 字段，int32 类型，由 Tiling 阶段从属性读取并下发给 Kernel：

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
    int64_t combineQuantMode;       // 8B
    float clampLimit;               // 4B
    int64_t topoType;               // 8B
    uint8_t groupedMatmulMode;      // 1B
    // 新增：权重前移开关，int32 类型，4 字节对齐
    int32_t topkWeightsMode;        // 0=不前移，1=前移
};
```

**4 字节对齐说明**：`topkWeightsMode` 用 `int32_t`（4 字节），天然满足 `MegaMoeTilingData` 的 4 字节对齐要求，无需额外 padding。

**Tiling 阶段赋值**（`mega_moe_tiling.cpp`）：

```cpp
// SetAttrParams 中新增
auto topkWeightsModePtr = attrs->GetAttrPointer<int64_t>(config.attrTopkWeightsModeIndex);
tilingData->topkWeightsMode = (topkWeightsModePtr != nullptr && *topkWeightsModePtr == 1) ? 1 : 0;
```

### 3.3 Kernel 侧使用

Kernel 侧通过 `tilingData->topkWeightsMode` 判断是否走权重前移路径：

```cpp
// mega_moe.h Init 中
topkWeightsMode_ = params_.tilingData->topkWeightsMode;

// 各改动点用 if (topkWeightsMode_) 分支
if (topkWeightsMode_) {
    // 权重前移路径：QuantProcessInRank 追加 weight 搬运、CopyGMToGMPerToken 拉取 weight 并存入三元组、
    //               激活量化插入 Muls、Unpermute 纯 Add
} else {
    // 原路径：保持不变
}
```

---

## 4. 权重数据流设计

### 4.1 核心约束

- 源卡 `topkWeights` 在 GM 的 `probsGmAddr`，**其他卡无法直接读源卡 GM**
- 跨卡只能读源卡 peermem win 区
- 因此必须把权重从 GM 搬到 win 区，dispatch 拉取 token 时一并拉回

### 4.2 数据流总览

权重随 token 数据一起从源卡 peermem 拉到专家卡 UB，在 MTE3 拆分阶段取出对应 weight，**复用三元组的未使用位槽**存入 `tripleInfoPtr`，激活量化阶段从 `tripleInfoPtr` 读取 weight。

```
源卡 GM                  源卡 pemem win 区              专家卡本地 GM
┌──────────────┐         ┌────────────────────┐        ┌─────────────────────────┐
│ token (bf16)  │→量化→   │ [data|scale|weights]│←拉取─ │ dispatchRevDataPtr      │ 行0
│ topkWeights   │→搬出→   │ (per-token 追加)    │       │ dispatchRevScalePtr     │ 行0
│ (float/bf16)  │         │                    │        │ tripleInfoPtr           │ 行0
└──────────────┘         └────────────────────┘        │  [rank|token|topk|w|int32│
                                                        └─────────────────────────┘
                                                          ↓ GMM1 + SwiGLU
                                                          ↓ 激活量化: tripleInfoPtr[mLoc+mIdx].weight * gluRes
                                                          ↓ GMM2 + Combine + Unpermute(纯Add)
```

### 4.3 权重类型统一为 float32

topkWeights 的输入类型可能是 float32 或 bfloat16。在源卡 `QuantProcessInRank` 搬运到 peermem 时统一转为 float32 存储：

| 输入类型 | 转换方式 | 精度 |
|---------|---------|------|
| float32 | 无需转换 | 无损 |
| bfloat16 | `Cast(fp32, bf16, CAST_NONE, topK)` | 无损（bf16 可精确提升为 fp32） |

后续全链路（peermem 存储、dispatch 拉取、三元组存储、激活量化使用）均按 float32 处理。

---

## 5. 改动点详解

### 5.1 改动点 1：源卡 peermem 布局（PeermemInfo）

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

**改动**：weight 与 token/scale 一样走 GM→UB→peermem 路径，在同一循环中追加 weight 的搬运。若 topkWeights 是 bfloat16，先 Cast 成 float32 再存入 peermem。

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
    // 若 topkWeights 是 bfloat16，先读 bf16 到临时 UB 再 Cast 成 float32 存入 weight 区
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
- 若 topkWeights 是 bfloat16，在 MTE2 后、V 前插入 `Cast(fp32, bf16, CAST_NONE, topK)` 转为 float32

### 5.3 改动点 3：dispatch 拉取 + weight 存入三元组（CopyGMToGMPerToken）

**文件**：`mega_moe.h:668-749`

**当前流程**：6-buffer 软流水，MTE2 把 `[data|scale]` 一起读入 UB 的 `copyTmpTensors_[outIdx]`，MTE3 阶段在 UB 内按偏移分离，用两次 `DataCopyPad` 分别写到本地 GM 的 `tokenRevGlobalTensor` 和 `scaleRevGlobalTensor`。

**三元组空间复用**：三元组每行开辟 32 字节（`ALIGN_32`，即 8 个 int32），但只填了 3 个 int32（RANK_ID=0, TOKEN_ID=1, TOPK_INDEX=2），剩余 5 个 int32（位槽 3-7）未使用。权重前移方案复用位槽 [3] 存储权重。

```
tripleTensor_[i * 8]:
  [0] RANK_ID     ← 已用
  [1] TOKEN_ID    ← 已用
  [2] TOPK_INDEX  ← 已用
  [3] weight      ← 新增（float 位重解释为 int32 存储）
  [4-7] 未用
```

**改动**：`copyInNum` 增大 `weightAlignBytes_`，MTE2 一次读入 `[data|scale|weights]`。token 和 scale 仍用 `DataCopyPad` 分离搬出；weight 只有 1 个 float（从 topK 个中取第 `topkIdx` 个），用 `GetValue` 标量取出，再通过 `reinterpret_cast` 位重解释为 int32，用 `SetValue` 写入三元组位槽 [3]：

```cpp
// copyInNum 增大
copyInNum = CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_ + weightAlignBytes_, ALIGN_32);

// Phase 2 steady: MTE3 拆分写入（mega_moe.h:722-728 改造）
LocalTensor<ActivationType> tokenScalebuf = copyTmpTensors_[outIdx];
LocalTensor<QuantScaleOutType> bufScale =
    tokenScalebuf[mxQuantTokenAlignBytes_].ReinterpretCast<QuantScaleOutType>();

// 现有：token 和 scale 用 DataCopyPad 分离搬出
DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScalebuf,
    {1, widthA * sizeof(ActivationType), 0, 0, 0});          // token 数据 → 行 copyIdx
DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], bufScale,
    {1, widthAScale * sizeof(QuantScaleOutType), 0, 0, 0});  // scale → 行 copyIdx

// 新增：从 buffer 取 weight，位重解释为 int32，存入三元组位槽 [3]
// weightOffset = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_
LocalTensor<float> bufWeights = tokenScalebuf[weightOffset].ReinterpretCast<float>();
int32_t topkIdx = tripleTensor_[copyIdx * INT32_PER_256B].GetValue(TOPK_INDEX);
float weight = bufWeights.GetValue(topkIdx);  // 从 topK 个 weight 中取第 topkIdx 个
int32_t weightBits = *reinterpret_cast<int32_t*>(&weight);  // float → int32 位重解释
tripleTensor_[copyIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);  // 写入位槽 [3]
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
       │ DataCopyPad      │ DataCopyPad      │ GetValue(topkIdx) → reinterpret_cast → SetValue
       ▼                  ▼                  ▼
  tokenRevGlobal    scaleRevGlobal    tripleTensor_[copyIdx*8][3] (int32, 位重解释)
     [copyIdx]         [copyIdx]              │
                                              │ Phase 3: DataCopy 搬出
                                              ▼
                                        tripleInfoPtr[copyIdx*8][3] (GM)
```

**关键点**：

- MTE2 一次读入 `[data|scale|weights]`，6-buffer 结构不变，每 buffer 多读 `weightAlignBytes_`（32B）
- token 和 scale 仍用 `DataCopyPad` 分离搬出（批量搬运，高效）
- weight 是 per-token 单个 float，用 `GetValue` 取出后通过 `reinterpret_cast` 位重解释为 int32，用 `SetValue` 写入三元组位槽 [3]（避免 4 字节 `DataCopyPad` 低效，且复用已有三元组空间）
- `topkIdx` 从三元组获取（`mega_moe.h:702` 已写入 `tripleTensor_`），直接索引第 `topkIdx` 个 weight
- `copyIdx` 是本地连续行号（0,1,2,...,totalToken-1），weight 与三元组同行存储
- `GetValue`/`SetValue` 需要 `MTE2_S`/`S_MTE3` flag 同步确保数据可见性
- Phase 3 三元组搬出（`mega_moe.h:748`）无需修改，weight 随三元组一起 DataCopy 到 `tripleInfoPtr`

**关于冗余**：每个被拉取的 token 会带回全部 `topK` 个 weight，但只用第 `topkIdx` 个。topK=8 时冗余 28B/token，相对 4KB token 数据可忽略。这避免了为 4 字节 weight 单独发起 MTE2 的低效问题。

### 5.4 改动点 4：激活量化读取权重（VFDoSwigluAndQuantForMX）

**文件**：`block_epilogue_swiglu_mx_quant.h:558-648`

**当前**：SwiGLU → Quant

**改动**：SwiGLU → **Muls(weight)** → Quant

激活量化阶段从 `tripleInfoPtr` 读取权重，通过 `reinterpret_cast` 将 int32 位重解释回 float32：

```cpp
// block_epilogue_swiglu_mx_quant.h:629 附近
// 当前：SwiGLU 结果在 gluRes_ (bf16, UB)，准备进入 quant

// 新增：权重乘
// mLoc 是当前 tile 在该专家 token 中的起始行号（从 scheduler.GetBlockCoord 获取）
// mIdx 是 tile 内行号 (0..mSize-1)
// weight 从 tripleInfoPtr 读取，int32 位重解释回 float
GlobalTensor<int32_t> tripleGm;
tripleGm.SetGlobalBuffer((__gm__ int32_t*)tripleInfoPtr);

for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
    // 从三元组位槽 [3] 读取 int32，位重解释回 float
    int32_t weightBits = tripleGm.GetValue((mLoc + mIdx) * INT32_PER_256B + WEIGHT_INDEX);
    float weight = *reinterpret_cast<float*>(&weightBits);  // int32 → float 位重解释
    // 在 SwiGLU 的 fp32 中间结果上乘
    Muls(swishData, swishData, weight, mask);  // swishData 是 fp32
}
// 然后进入原有 quant 逻辑
```

**`mLoc` 的来源**：`AivGmm1PostGeneric`（`mega_moe_impl.h:418`）从 `scheduler.GetBlockCoord(loopIdx)` 获取 `mLoc`，传入 epilogue。需要将其透传到 `VFDoSwigluAndQuantForMX`。

**weight 取法**：`tripleInfoPtr[(mLoc + mIdx) * INT32_PER_256B + WEIGHT_INDEX]`

```
该专家 totalToken = 1000, tileM = 256

tile 0: mLoc=0,   mIdx=0..255  → tripleInfoPtr[0..255][3]
tile 1: mLoc=256, mIdx=0..255  → tripleInfoPtr[256..511][3]
tile 2: mLoc=512, mIdx=0..255  → tripleInfoPtr[512..767][3]
tile 3: mLoc=768, mIdx=0..231  → tripleInfoPtr[768..999][3]  (尾块)
```

### 5.5 改动点 5：Unpermute 退化为纯加法

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

## 6. 类型转换方案：位重解释

### 6.1 方案说明

权重存入三元组时，三元组是 `LocalTensor<int32_t>`，`SetValue` 接受 int32 类型。权重是 float32，需要类型转换。有两种转换方式：

| 方式 | 接口 | 语义 | 精度 |
|------|------|------|------|
| **数值转换（不可用）** | `Cast<float, int32_t, CAST_ROUND>(2.5f)` | 带舍入的值转换，返回 3 | **有丢失** |
| **位重解释（推荐）** | `*reinterpret_cast<int32_t*>(&f)` | 比特位不变，返回 0x40400000 | **无丢失** |

必须使用**位重解释**（reinterpret_cast 指针别名），不能使用数值转换（Cast）。

### 6.2 写入：float → int32

```cpp
float weight = bufWeights.GetValue(topkIdx);
int32_t weightBits = *reinterpret_cast<int32_t*>(&weight);
tripleTensor_[copyIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
```

- 比特位完全保留，无精度丢失
- float32 和 int32 都是 4 字节，位重解释安全

### 6.3 读取：int32 → float

```cpp
int32_t weightBits = tripleGm.GetValue((mLoc + mIdx) * INT32_PER_256B + WEIGHT_INDEX);
float weight = *reinterpret_cast<float*>(&weightBits);
```

- 比特位原样还原为 float32，无精度丢失

### 6.4 bfloat16 处理

bfloat16 是 2 字节，不能直接位重解释为 int32（4 字节）。需要先 Cast 成 float32（无损提升），再位重解释：

```cpp
// 源卡搬运阶段：bf16 → fp32（Cast，无损）→ 存入 peermem
Cast(weightFp32Ub, weightBf16Ub, AscendC::RoundMode::CAST_NONE, topK_);
```

Cast 后的 float32 存入 peermem，后续全链路按 float32 处理，位重解释链路无精度丢失。

### 6.5 接口规范性

`reinterpret_cast` 指针别名是 **Ascend C 官方文档推荐的标准用法**，在以下场景中明确给出：

- **SetQuantScalar API 文档**：`*reinterpret_cast<int32_t*>(&tmp)` 做 float→int32 位重解释
- **ReduceMin/ReduceMax API 文档**：`*reinterpret_cast<uint32_t*>(&minIndex)` 做 float→uint32 位重解释
- **examples**：`matmul_quant.asc`、`fixpipe_l0c2gm.asc` 等多处使用
- **impl 内部库**：`asc_fp16_impl.h`、`asc_bf16_impl.h`、`math_functions_impl.h`、`softmax_common_impl.h` 等超过 25 处使用

注意：SIMT 模式有专用接口 `__float_as_int`/`__int_as_float`，但**仅 950PR/950DT 支持**，A3 环境（CANN 8.5.0）不可用，所以使用 SIMD 模式的 `reinterpret_cast` 指针别名。

---

## 7. 数据流完整时序

```
Stage 1: QuantProcessInRank (源卡)
  GM token → 量化 → [data|scale|weights] → peermem quantTokenScalePtr
                              ↑ 新增 weights（统一 float32）

Stage 2: SendMaskCal (源卡)
  topkIds → [mask|count] → peermem maskRecvPtr  (不变)

Stage 3: CopyGMToGMPerToken (专家卡 dispatch)
  6-buffer 软流水跨卡读:
    MTE2: peermem[tokenIdx] = [data|scale|weights] → UB
    MTE3: data → dispatchRevDataPtr[copyIdx]
          scale → dispatchRevScalePtr[copyIdx]
          weight[topkIdx] → tripleTensor_[copyIdx][3]  ← 位重解释存入三元组
    Phase 3: triple(含weight) → tripleInfoPtr[copyIdx]  (随三元组一起搬出)

Stage 4: GMM1 (专家卡)
  AIC: MMAD(dispatchRevDataPtr[行] × weight1) → L0C
  AIV0: L0C → UB → SwiGLU → w*Y → Quant → swigluQuantDataPtr
                                    ↑ 新增 w*Y (从 tripleInfoPtr[mLoc+mIdx][3] 读 weight)

Stage 5: GMM2 + Combine + Unpermute (专家卡)
  GMM2: MMAD(swigluQuantDataPtr × weight2) → 结果
  Combine: 结果归还到源卡 combineSendPtr
  Unpermute: 纯 Add 求和 → output  ← 移除 Muls
```

---

## 8. 空间代价分析

以 `bs=4096, topK=8, hiddenDim=4096, worldSize=8, expertPerRank=8` 为例：

| 区域 | 当前 | 前移后 | 增量 |
|------|------|--------|------|
| peermem 每 token | ~4KB (data+scale) | ~4KB + 32B | +32B (0.8%) |
| peermem 量化区总量 | 16MB | 16MB + 128KB | +128KB |
| dispatch 每 token MTE2 | ~4KB | ~4KB + 32B | +32B (0.8%) |
| 本地 tripleInfoPtr | 已有（32B/行，用3个int32） | 复用位槽[3]，无新增 | 0 |
| UB topKWeightsTensor_ | 128KB | 0 | -128KB |

**净增**：peermem +128KB，本地 GM 0（复用三元组），UB -128KB。总体空间代价可忽略，且省去了独立权重存储区域。

---

## 9. 关键设计决策

### 9.1 为什么 weight 随 token 数据发送，不拼在 mask+cnt 后

- mask+cnt 是**控制面**（固定长度二维表），weight 是**数据面**（per-token）
- 拼在 mask+cnt 后：固定长度空间爆炸（maxCount 预留膨胀 32 倍），变长破坏二维表寻址
- 随 token 数据发送：搭便车已有的 6-buffer 跨卡读路径，每 token 仅 +32B

### 9.2 为什么 per-token 追加全部 topK 个 weight，不按 topkIndex 独立排列

- per-token 追加：随 token 数据一次 DataCopy 读回，零额外 MTE2，6-buffer 不变
- 按 topkIndex 独立：每 token 多一次 4 字节 MTE2，利用率极低，破坏流水线
- 冗余代价：topK=8 时 28B/token，可忽略

### 9.3 为什么复用三元组空间存 weight，不新增 workspace

- 三元组每行 32 字节（8 个 int32），实际只用 3 个，剩余 5 个 int32 空间充足
- 复用后 weight 与三元组同行存储，天然对齐，激活量化按 `(mLoc+mIdx) * INT32_PER_256B + WEIGHT_INDEX` 直接取
- 无需新增 `weightRecvPtr` workspace 区域，零额外 GM 空间
- 三元组已有 Phase 3 的 DataCopy 搬出逻辑（`mega_moe.h:748`），weight 随之搬到 `tripleInfoPtr`，无需额外搬运

### 9.4 为什么用位重解释而非数值转换存 weight

- 三元组是 `LocalTensor<int32_t>`，`SetValue` 接受 int32；weight 是 float32，类型不匹配
- 数值转换（`Cast<float, int32_t, CAST_ROUND>`）带舍入，`2.5f → 3`，**有精度丢失**
- 位重解释（`*reinterpret_cast<int32_t*>(&f)`）比特位不变，`2.5f → 0x40400000`，**无精度丢失**
- `reinterpret_cast` 指针别名是 Ascend C 官方文档推荐的标准用法（SetQuantScalar、ReduceMin/ReduceMax 等多处）
- float32 和 int32 都是 4 字节，位重解释安全

### 9.5 为什么在 fp32 域做 Muls

- SwiGLU 中间结果 `swishData` 已是 fp32（`block_epilogue_swiglu_mx_quant.h:580`）
- weight 从三元组位重解释回 fp32，直接 `Muls(swishData, swishData, weight, mask)` 在 fp32 域做
- 乘完再 Cast 回 bf16 进入 quant，与现有流程一致

---

## 10. 改动文件清单

### 10.1 算子侧（op_host / op_kernel）

| 文件 | 改动内容 |
|------|---------|
| `op_host/mega_moe_def.cpp` | 新增 `topk_weights_mode` 属性（OPTIONAL Int，默认 0） |
| `op_host/mega_moe_infershape.cpp` | 无需修改（新属性不影响输出 shape 推导） |
| `op_host/op_tiling/arch35/mega_moe_tiling.cpp` | `SetAttrParams` 读取 `topk_weights_mode` 赋值 `tilingData->topkWeightsMode`；`MegaMoeConfig` 新增 `attrTopkWeightsModeIndex` |
| `op_kernel/arch35/mega_moe_tiling.h` | `MegaMoeTilingData` 新增 `topkWeightsMode`（int32_t），天然 4 字节对齐 |
| `op_kernel/arch35/mega_moe_base.h` | `PeermemInfo`：`tokenBytes` 增加 `weightAlignBytes` |
| `op_kernel/arch35/mega_moe.h` `Init` | 新增 `weightAlignBytes_`、`topkWeightsMode_` 成员变量初始化 |
| `op_kernel/arch35/mega_moe.h` `QuantProcessInRank` | 新增 weight GM→UB→peermem 搬运（bf16 先 Cast 成 fp32） |
| `op_kernel/arch35/mega_moe.h` `CopyGMToGMPerToken` | `copyInNum` 增大；MTE3 拆出 weight，位重解释为 int32 存入三元组位槽 [3] |
| `op_kernel/arch35/mega_moe.h` `UnpermuteBuffInit` | 移除 `topKWeightsTensor_` 相关代码 |
| `op_kernel/arch35/mega_moe.h` `Unpermute` | 移除 `Muls`，保留 `Add` |
| `op_kernel/arch35/block_epilogue_swiglu_mx_quant.h` | `VFDoSwigluAndQuantForMX` 从 `tripleInfoPtr` 读 weight（int32 位重解释回 float），插入 `Muls`；透传 `mLoc` |
| `op_kernel/arch35/mega_moe_impl.h` `AivGmm1PostGeneric` | 透传 `mLoc` 到 epilogue |

注：`mega_moe_workspace_info.h` **无需修改**，复用现有 `tripleInfoPtr` 空间，不新增 workspace 区域。

### 10.2 aclnn 接口侧（op_api）

新增属性需要透传到 aclnn 接口层。`aclnnMegaMoe` 调用 `aclnnInnerMegaMoe`（由 CANN SDK 提供，头文件 `aclnnInner_mega_moe.h` 不在代码仓内），需要将 `topkWeightsMode` 作为新参数透传。

| 文件 | 改动内容 |
|------|---------|
| `op_api/aclnn_mega_moe.cpp` | `aclnnMegaMoeGetWorkspaceSize` 函数签名新增 `int64_t topkWeightsMode` 参数；调用 `aclnnInnerMegaMoeGetWorkspaceSize` 时透传该参数 |
| `tests/ut/op_api/aclnn_mega_moe.h` | `aclnnMegaMoeGetWorkspaceSize` 声明新增 `int64_t topkWeightsMode` 参数 |

注：`aclnnInner_mega_moe.h` 由 CANN SDK 提供，不在代码仓内。若 `aclnnInnerMegaMoeGetWorkspaceSize` 的签名由 SDK 控制，则 `op_api/aclnn_mega_moe.cpp` 仅在调用处透传，SDK 侧需同步更新接口签名。若 SDK 暂不支持透传，可作为已知限制，后续 SDK 版本同步。

### 10.3 torch_extension 侧（Python 接口透传）

新增可选属性 `topk_weights_mode` 需要从用户调用入口（`mega_moe()` 函数）经 PyTorch dispatch 一路透传到 C++ 侧的 `ACLNN_CMD`。涉及 3 个文件：

| 文件 | 改动内容 |
|------|---------|
| `torch_extension/cann_ops_transformer/ops/mega_moe.py` | `mega_moe()` 函数新增 `topk_weights_mode` 可选参数；`SymmBuffer` 新增 `topk_weights_mode` 属性；`_npu_mega_moe` dispatch 函数和 `register_meta` 新增参数；`schema()` 新增参数声明 |
| `torch_extension/cann_ops_transformer/ops/csrc/mega_moe.cpp` | `NpuMegaMoe` 函数新增 `topk_weights_mode` 参数；`ACLNN_CMD(aclnnMegaMoe, ...)` 调用新增透传 |
| `torch_extension/cann_ops_transformer/ops/graph_convert/graph_convert_mega_moe.py` | `MegaMoe` GE 算子定义和 `convert_npu_mega_moe` 转换函数新增 `topk_weights_mode` 参数和 attr |

**完整透传链路**：

```
用户脚本调用
  → mega_moe(..., topk_weights_mode=1)              # mega_moe.py: mega_moe()
    → torch.ops.cann_ops_transformer.npu_mega_moe()  # mega_moe.py: _npu_mega_moe()
      → _op_module.npu_mega_moe()                     # mega_moe.cpp: NpuMegaMoe()
        → ACLNN_CMD(aclnnMegaMoe, ..., topkWeightsMode)  # 调用 aclnn 接口
          → aclnnMegaMoeGetWorkspaceSize(..., topkWeightsMode)  # op_api/aclnn_mega_moe.cpp
            → aclnnInnerMegaMoeGetWorkspaceSize(..., topkWeightsMode)  # CANN SDK
```

**mega_moe.py 具体改动点**：

1. `schema()`：末尾新增 `int? topk_weights_mode=None`
2. `register_meta()`：函数签名新增 `topk_weights_mode=None`
3. `_npu_mega_moe()`：函数签名新增 `topk_weights_mode=None`，调用 `_op_module.npu_mega_moe` 时透传
4. `SymmBuffer.__init__()`：新增 `topk_weights_mode: int = 0` 参数
5. `get_symm_buffer_for_mega_moe()`：新增 `topk_weights_mode: int = 0` 参数
6. `mega_moe()`：新增 `topk_weights_mode: Optional[int] = None` 参数，调用 `torch.ops` 时透传 `topk_weights_mode=sym_buffer.topk_weights_mode`

**torch_extension mega_moe.cpp 具体改动点**：

1. `NpuMegaMoe` 函数签名新增 `c10::optional<int64_t> topkWeightsMode`
2. 函数体内取值：`int64_t topkWeightsModeValue = topkWeightsMode.value_or(0);`
3. `ACLNN_CMD(aclnnMegaMoe, ...)` 调用末尾新增 `topkWeightsModeValue`

**op_api aclnn_mega_moe.cpp 具体改动点**：

1. `aclnnMegaMoeGetWorkspaceSize` 函数签名在 `topoType` 之后新增 `int64_t topkWeightsMode` 参数
2. 调用 `aclnnInnerMegaMoeGetWorkspaceSize` 时在 `topoType` 之后透传 `topkWeightsMode`

**graph_convert_mega_moe.py 具体改动点**：

1. `MegaMoe` 函数签名新增 `topk_weights_mode: int = 0`
2. `attrs` 字典新增 `"topk_weights_mode": attr.Int(topk_weights_mode)`
3. `IrDef` 链式调用新增 `.attr("topk_weights_mode", attr.Int(0))`
4. `convert_npu_mega_moe` 函数签名新增 `topk_weights_mode: int = 0`，调用 `MegaMoe` 时透传
4. `convert_npu_mega_moe` 函数签名新增 `topk_weights_mode: int = 0`，调用 `MegaMoe` 时透传

---

## 11. 风险与验证

### 11.1 精度验证

- **scale 吸收**：MXFP scale 吸收 `|w|`，quantized 差 `sign(w)`，MMAD 结果数学等价
- **位重解释**：float32 ↔ int32 位重解释无精度丢失（比特位完全保留）
- **bf16 提升**：bfloat16 → float32 的 Cast 无损（bf16 可精确表示为 fp32）
- **边界**：`|w| ≤ 1`（softmax 输出），不会导致 max_abs 溢出
- **精度影响**：权重前移操作本身不会造成精度劣化，但改变了后续多轮操作的输入数据分布，整体精度对比结果可能变化，需实测验证是否符合精度标准

### 11.2 功能验证

- dispatch 拉取的 weight 与三元组行号对齐（`copyIdx` 一致）
- 位重解释写入/读取一致性（float → int32 → float 往返无损）
- 激活量化 `(mLoc + mIdx) * INT32_PER_256B + WEIGHT_INDEX` 索引正确（与 GMM1 消费的行号范围一致）
- 三元组位槽 [3] 不与其他用途冲突
- Unpermute 纯 Add 结果与原 Muls+Add 一致

### 11.3 性能验证

- peermem 空间增量是否影响后续区域（combineSendPtr 起址后移）
- 6-buffer 流水线是否因 `copyInNum` 增大而退化
- 激活量化阶段从 GM 读 weight 标量的开销（ GetValue 走 DCache）
- Unpermute 省去 Muls 和权重加载的加速效果
