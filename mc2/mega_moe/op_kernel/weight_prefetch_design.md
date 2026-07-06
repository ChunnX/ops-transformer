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

### 3.3 Kernel 侧使用：TilingKey 编译期使能

权重前移涉及 GMM1 tile 大小变化（256→128）和 epilogue UB 布局变化，这些是**编译期决策**，不能用运行时 `if` 分支。因此通过 TilingKey 将 `topkWeightsMode` 提升为编译期常量，走模板特化。

#### 3.3.1 TilingKey 新增字段

在 `mega_moe_tiling_key.h` 中新增第 5 个字段 `TILINGKEY_TOPK_WEIGHTS_MODE`（bool，0/1）：

```cpp
ASCENDC_TPL_ARGS_DECL(MegaMoe,
    ... // 现有 4 个字段
    ASCENDC_TPL_BOOL_DECL(TILINGKEY_TOPK_WEIGHTS_MODE, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ... // 现有 4 个选择器
        ASCENDC_TPL_BOOL_SEL(TILINGKEY_TOPK_WEIGHTS_MODE, 0, 1),
        ASCENDC_TPL_TILING_STRUCT_SEL(MegaMoeTilingData)
    ),
);
```

#### 3.3.2 Tiling 阶段计算 TilingKey

`mega_moe_tiling.cpp` 的 `CalTilingKey` 从属性读取 `topkWeightsMode`，传入 `GET_TPL_TILING_KEY`：

```cpp
auto topkWeightsModePtr = attrs->GetAttrPointer<int64_t>(config.attrTopkWeightsModeIndex);
int64_t topkWeightsMode = (topkWeightsModePtr != nullptr && *topkWeightsModePtr == 1) ? 1 : 0;
return GET_TPL_TILING_KEY(..., topkWeightsMode);
```

#### 3.3.3 Kernel 模板参数贯穿

```
apt.cpp:  template<..., bool TopkWeightsMode>  kernel 模板
  ↓
MegaMoe<..., TopkWeightsMode>:  static constexpr GMM1_TILE_M = TopkWeightsMode ? 128 : 256
  ├── BlockEpilogueSwigluMxQuant<..., TileM=GMM1_TILE_M>  (epilogue UB 布局编译期确定)
  └── GroupMatmulSwigluQuant<..., Gmm1TileM=GMM1_TILE_M>
        └── Config<..., Gmm1TileM> → config.tileM = Gmm1TileM  (GMM1 各处从 config.tileM 取值)
```

#### 3.3.4 优势

- **零运行时开销**：`TopkWeightsMode=0` 编译出 256×256 的 kernel，与改动前完全一致
- **`__VEC_SCOPE__` 内无分支**：epilogue 的权重乘始终执行，mode=0 时 UB 填 1.0f，乘以 1.0 等价不乘
- **遵循现有模式**：与 `CombineQuantMode`、`CommModeType` 等 TilingKey 字段完全同构

#### 3.3.5 layered 模板

`mega_moe_layered.h`（URMA 拓扑）**不适配**权重前移功能。apt.cpp 中 `MegaMoeLayered` 实例化不传 `TopkWeightsMode`，layered 始终用默认 256×256。若需在 layered 上启用，需另行适配。

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

**文件**：`block_epilogue_swiglu_mx_quant.h`

**当前**：SwiGLU → Quant

**改动**：SwiGLU → **w\*Y（权重乘）** → Quant

#### 5.4.1 问题：`__VEC_SCOPE__` 内不能用标量 Muls

初版设计计划在 `__VEC_SCOPE__` 内用 `Muls(outFReg, outFReg, weight, mask)`（vec-scalar），但 `weight` 是从 GM 读取的运行时标量。`__VEC_SCOPE__` 内只允许寄存器间操作，运行时标量变量和 GM 访问（`GetValue`）都会导致编译器报"遇到无法优化的指令"。

#### 5.4.2 方案：UB weight buffer + DIST_BRC_B32 + Mul（vec-vec）

参考 `ops-nn/activation/swiglu_group_quant/op_kernel/arch35/swiglu_group_quant_base.h:234-251` 的官方实现，分三步：

**Step 1：UB weight buffer 分配（Init 中）**

在 `Init()` 中，`halfScale_` 之后追加 `weightUb_`，放在 VECIN 区之后的安全区域：

```cpp
// block_epilogue_swiglu_mx_quant.h Init()
constexpr uint32_t vecinEnd = MAX_SINGLE_MN * sizeof(DataTypeIn) * 2U;
weightUb_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, vecinEnd, MAX_TILE_M);
```

UB 布局（以 TileM=128, TileN=256, DataTypeIn=bf16 为例）：

```
物理 UB (256KB)
├── VECIN 区 [0, 128KB)
│   ├── l0cOutUbFirst_   [0,     64KB)   128×256×2B
│   └── l0cOutUbSecond_  [64KB, 128KB)   128×256×2B
├── VECCALC/VECOUT 区 [0, ~101KB)  (与 VECIN 重叠，流水线时序复用)
│   ├── gluRes_           [0,     64KB)
│   ├── quantOutput_      [64KB,  96KB)
│   ├── quantScaleOutput_ [96KB,  97KB)
│   ├── maxExp_           [97KB,  99KB)
│   └── halfScale_        [99KB, 101KB)
└── weightUb_ [128KB, 128KB+512B)   128×4B = 512B  ★ 新增，独立区域
```

TileM=128 时 VECIN 区只占 128KB，`[128KB, 256KB)` 完全空闲，weightUb_ 仅需 512B。

**Step 2：`__VEC_SCOPE__` 前——GM→UB gather**

`tripleInfoGm_` 中 weight 按 `INT32_PER_256B=8` 间隔 strided 存储，需逐个 gather 成连续 UB 数组。mode=0 时填 1.0f，mode=1 时取真实权重：

```cpp
// __VEC_SCOPE__ 之前
for (uint16_t i = 0; i < mSize; i++) {
    float w = (topkWeightsMode_ == 1) ? tripleInfoGm_.GetValue((mLoc + i) * INT32_PER_256B + WEIGHT_INDEX) : 1.0f;
    weightUb_.SetValue(i, w);
}
__ubuf__ float* weightUbAddr = (__ubuf__ float*)weightUb_.GetPhyAddr();
```

**关键点**：
- `GetValue`/`SetValue` 是标量操作，在 scope 外合法
- mode=0 填 1.0f，使 scope 内的 `Mul` 乘以 1.0 等价于不乘，**消除 scope 内 if 分支**
- `GetPhyAddr` 取地址在 scope 外，scope 内仅做指针偏移 `+ dim0vfLoopIdx`

**Step 3：`__VEC_SCOPE__` 内——DIST_BRC_B32 加载广播 + Mul（vec-vec）**

```cpp
__VEC_SCOPE__
{
    ...
    AscendC::MicroAPI::RegTensor<float> weightReg;
    ...
    for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
        // 从 UB 加载 1 个 float，广播到 weightReg 所有 lane
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
            weightReg, weightUbAddr + dim0vfLoopIdx);
        for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
            ...
            AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, mask);  // swish(x)*y
            AscendC::MicroAPI::Mul(outFReg, outFReg, weightReg, mask);   // w * swish(x)*y
            ...
        }
        // tail 循环同理
    }
}
```

**`DIST_BRC_B32` 原理**：从 UB 地址读取恰好 1 个 B32（32bit = 1 个 float），广播到寄存器所有 lane。`dim0vfLoopIdx` 既是 GM 中 token 的行内偏移，也是 UB 连续数组中的下标。

**数据流**：

```
GM (strided)                    UB (contiguous)              Register (broadcast)
┌─────────────┐                ┌──────────────┐             ┌─────────────────┐
│token0: W@3  │──GetValue──→  │weightUb[0]   │──DataCopy──→│weightReg[0..63]  │
│token1: W@3  │──GetValue──→  │weightUb[1]   │   BRC_B32   │  全部 = weight[0]│
│...          │                │...           │             └─────────────────┘
│stride=8     │                │连续float数组 │
└─────────────┘                └──────────────┘
  scope 外                       scope 内
  SetValue/GetValue              DataCopy + Mul (vec-vec)
```

#### 5.4.3 `mLoc` 透传

`mLoc` 是当前 tile 在该专家 token 中的起始行号，从 `scheduler.GetBlockCoord(loopIdx)` 获取。PR 已将其通过 `BlockCoord` 透传到 `VFDoSwigluAndQuantForMX` 的 `mLoc` 参数。

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

### 5.6 改动点 6：GMM1 tile 缩小 256→128（腾出 UB 空间）

#### 5.6.1 问题

epilogue 是 GMM1 的尾端，`l0cOutUbFirst_`/`l0cOutUbSecond_` 是 GMM1 L0C→UB 的输出，固定占用 VECIN 区。原 TileM=256 时两个 buffer 各 128KB，VECIN 区占满 256KB UB，无处放置 weightUb_。

#### 5.6.2 方案

`topkWeightsMode=1` 时将 GMM1 tile M 从 256 缩小到 128（128=8×16，满足 Cube MMAD 的 16 对齐）：

| Buffer | TileM=256 (原) | TileM=128 (新) |
|--------|----------------|-----------------|
| `l0cOutUbFirst_` | 128KB | 64KB |
| `l0cOutUbSecond_` | 128KB | 64KB |
| VECIN 区合计 | 256KB（满） | 128KB |
| **空闲区** | **0** | **128KB** |
| `weightUb_` | 无处放 | 512B（绰绰有余） |

#### 5.6.3 实现方式

通过 TilingKey 编译期选择，`TopkWeightsMode=1` 时 `GMM1_TILE_M = 128`：

```cpp
// mega_moe.h
static constexpr uint32_t GMM1_TILE_M = TopkWeightsMode ? MegaMoeImpl::L1_TILE_M_128 : MegaMoeImpl::L1_TILE_M_256;
```

GMM1 各处通过 `config.tileM`（= `Gmm1TileM` 模板参数）取值：
- `BuildProblemConfig`：`config.tileM = Gmm1TileM`
- `BuildLayouts`：`layouts.c = MakeLayoutC{}(Gmm1TileM, L1_TILE_N)`
- `AicComputeGeneric`：`ubBufSize` 按 `config.tileM` 选择 128/256 对应的 UB 大小
- `AicComputeA8W4`：`l0TileShape{config.tileM, ...}`、scheduler 用 `config.tileM`
- `WaitForUpstreamReady`：`waveIdx = mLoc / config.tileM`
- dispatch wave 粒度：`CeilDiv(maxOutputSize, GMM1_TILE_M)`

**GMM2 路径不受影响**：GMM2 始终用 `L1_TILE_M_128`（Generic NO_QUANT）或 `L1_TILE_M_256`（量化/A8W4），与 `TopkWeightsMode` 无关。

#### 5.6.4 性能影响

- M 从 256→128，单 tile 计算量减半，tile 数量翻倍
- 权重乘融入已有 element-wise 流水，零额外 pipeline stage
- Unpermute 退化为纯加法，负载降低
- GMM2 已在用 128，说明 128×256 的 Cube 效率在工程上可接受

### 5.7 改动点 7：变量统一与 epilogue 参数化重构

#### 5.7.1 问题

原代码中同一数值（256）在 4 处用不同名字定义，且 epilogue 与 GMM1 之间无参数传递、靠硬编码巧合一致：

| 变量 | 定义位置 | 值 |
|------|---------|-----|
| `L1_TILE_M_256` | `mega_moe_impl_base.h` | 256 |
| `MAX_TILE_M` | epilogue 匿名 namespace | 256 |
| `DISPATCH_WAVE_TILE_M` | `mega_moe_workspace_info.h` | 256 |
| `MAX_SINGLE_MN` | epilogue 匿名 namespace | 256×256 |

改 GMM1 tile M，epilogue 的 UB 布局不会跟着变，静默出错。

#### 5.7.2 重构内容

**1. epilogue 模板参数化**：删除匿名 namespace 的 `MAX_SINGLE_MN`/`MAX_TILE_M`，改为类模板参数：

```cpp
template <..., uint32_t TileM = 256, uint32_t TileN = 256>
class BlockEpilogueSwigluMxQuant {
    static constexpr uint32_t MAX_TILE_M = TileM;
    static constexpr uint32_t MAX_SINGLE_MN = TileM * TileN;
    ...
};
```

实例化时按需选择：`BlockEpilogueSwigluMxQuant<..., GMM1_TILE_M, L1_TILE_N>`

**2. 去重**：删除 `mega_moe_workspace_info.h` 中重复的 `DISPATCH_WAVE_TILE_M` 和 `L1_TILE_M_128`（int64_t），引用改为 `MegaMoeImpl::L1_TILE_M_256`。

**3. 删除死参数**：epilogue `Arguments` 删除未使用的 `baseM`/`baseN` 字段，`Init` 调用删除 `ALIGN_256, ALIGN_256` 传参。

**4. epilogue tile 与 GMM1 解耦**：epilogue 的 tile 大小由自己的模板参数决定，GMM1 的 tile 由自己的常量决定，在实例化点独立选择，不强制相等。

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

### 9.6 为什么用 TilingKey 而非运行时 if

- GMM1 tile 大小变化（256→128）和 epilogue UB 布局变化是**编译期决策**（`constexpr` offset、模板类型），运行时 `if` 无法分支
- TilingKey 将 `topkWeightsMode` 提升为编译期常量，走模板特化，零运行时开销
- `TopkWeightsMode=0` 编译出 256×256 的 kernel，与改动前完全一致，零影响回退
- 遵循现有模式（`CombineQuantMode`、`CommModeType` 等 TilingKey 字段完全同构）

### 9.7 为什么 GMM1 tile 缩小到 128 而非其他值

- 128=8×16，满足 Cube MMAD 的 16 对齐要求
- GMM2 Generic NO_QUANT 已在用 128，工程上已被验证可接受
- VECIN 区从 256KB 降到 128KB，省出 128KB，weightUb_ 仅需 512B，余量充足
- 240（15×16）虽也满足对齐，但 GMM2 已有 128 的基础设施（常量、UB 布局），复用成本更低

### 9.8 为什么 `__VEC_SCOPE__` 内不设 if 分支

- `__VEC_SCOPE__` 内的 `if` 分支会破坏编译器的寄存器优化，导致性能下降
- 方案：scope 前给 UB weight 数组填 1.0f 初值（mode=0）或真实权重（mode=1），scope 内 `DataCopy<DIST_BRC_B32>` + `Mul` 始终执行
- mode=0 时乘以 1.0 等价于不乘，无额外计算结果差异，仅多一条 `Mul` 指令（可被编译器优化或流水线隐藏）

### 9.9 为什么 epilogue tile 与 GMM1 tile 解耦

- 方案二（备选，未实施）需要 GMM1 保持 256 但 epilogue 独立切 128（Cube 算 256×256 写 GM，epilogue 从 GM 分批读 128×256）
- epilogue 通过模板参数 `TileM` 独立指定 UB 容量，不引用 GMM1 的 `L1_TILE_M_256`，在实例化点独立选择
- 当前方案一两者都选 128，但解耦后方案二只需改实例化点一处即可切换

---

## 10. 改动文件清单

### 10.1 算子侧（op_host / op_kernel）

| 文件 | 改动内容 |
|------|---------|
| `op_host/mega_moe_def.cpp` | 新增 `topk_weights_mode` 属性（OPTIONAL Int，默认 0） |
| `op_host/mega_moe_infershape.cpp` | 无需修改（新属性不影响输出 shape 推导） |
| `op_host/op_tiling/arch35/mega_moe_tiling.cpp` | `SetAttrParams` 读取 `topk_weights_mode` 赋值 `tilingData->topkWeightsMode`；`CalTilingKey` 传入 `topkWeightsMode`；`MegaMoeConfig` 新增 `attrTopkWeightsModeIndex` |
| `op_kernel/arch35/mega_moe_tiling.h` | `MegaMoeTilingData` 新增 `topkWeightsMode`（int32_t），天然 4 字节对齐 |
| `op_kernel/arch35/mega_moe_tiling_key.h` | 新增 `TILINGKEY_TOPK_WEIGHTS_MODE` 字段（bool，0/1）到 ARGS_DECL 和 SEL |
| `op_kernel/arch35/mega_moe_apt.cpp` | kernel 模板新增 `bool TopkWeightsMode`，传入 `MegaMoe`（`MegaMoeLayered` 不传，不适配） |
| `op_kernel/arch35/mega_moe_base.h` | `PeermemInfo`：`tokenBytes` 增加 `weightAlignBytes` |
| `op_kernel/arch35/mega_moe.h` | 宏新增 `TopkWeightsMode`；`static constexpr GMM1_TILE_M = TopkWeightsMode ? 128 : 256`；`BlockEpilogue` 实例化传 `GMM1_TILE_M`；`GroupMatmulWithSwigluQuant` 传 `GMM1_TILE_M`；dispatch wave 用 `GMM1_TILE_M`；`Init` 新增 `weightAlignBytes_`、`topkWeightsMode_`；`QuantProcessInRank` 新增 weight 搬运；`CopyGMToGMPerToken` weight 存入三元组；`UnpermuteBuffInit` 移除 `topKWeightsTensor_`；`Unpermute` 移除 `Muls` |
| `op_kernel/arch35/mega_moe_impl_base.h` | 无修改（`L1_TILE_M_256`/`L1_TILE_M_128` 保留原名） |
| `op_kernel/arch35/mega_moe_impl.h` | `Config` 新增 `Gmm1TileM` 模板参；`BuildProblemConfig` GMM1 用 `Gmm1TileM`（A8W4 GMM2 强制 256）；`BuildLayouts` GMM1 用 `Gmm1TileM`；`AicComputeGeneric`/`AicComputeA8W4` `ubBufSize`/`waveIdx`/`l0TileShape` 用 `config.tileM`；`GroupMatmulSwigluQuant`/`GroupMatmulImplGeneric`/`GroupMatmulSwigluQuantA8W4`/`GroupMatmulImplA8W4` 新增 `Gmm1TileM` 模板参；函数参数 epilogue 类型带 `Gmm1TileM` |
| `op_kernel/arch35/mega_moe_workspace_info.h` | 删除重复的 `DISPATCH_WAVE_TILE_M` 和 `L1_TILE_M_128`(int64_t)，引用改为 `MegaMoeImpl::L1_TILE_M_256` |
| `op_kernel/arch35/block_epilogue_swiglu_mx_quant.h` | 类模板新增 `TileM`/`TileN` 参数；删除匿名 namespace 的 `MAX_SINGLE_MN`/`MAX_TILE_M`；类内 `static constexpr` 派生；删除 `Arguments::baseM`/`baseN` 死参数；`Init` 分配 `weightUb_`；`VFDoSwigluAndQuantForMX` 用 `SetValue` gather weight 到 UB + `DIST_BRC_B32` + `Mul`(vec-vec) 替代 `Muls` |
| `op_kernel/arch35/mega_moe_layered.h` | **不适配**，零改动（URMA 拓扑不支持权重前移） |

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
- 激活量化 `(mLoc + i) * INT32_PER_256B + WEIGHT_INDEX` 索引正确（与 GMM1 消费的行号范围一致）
- 三元组位槽 [3] 不与其他用途冲突
- Unpermute 纯 Add 结果与原 Muls+Add 一致
- TilingKey 编译期特化：`TopkWeightsMode=0` 编译产物与改动前完全一致
- GMM2 路径不受 `TopkWeightsMode` 影响：Generic NO_QUANT 仍用 128，量化/A8W4 仍用 256
- layered 模板（URMA 拓扑）不受影响，始终走 256×256 原路径
- epilogue `weightUb_` UB 地址不与现有 buffer 重叠
- `__VEC_SCOPE__` 内无 `if` 分支：mode=0 时 UB 填 1.0f，`Mul` 乘以 1.0 等价不乘

### 11.3 性能验证

- peermem 空间增量是否影响后续区域（combineSendPtr 起址后移）
- 6-buffer 流水线是否因 `copyInNum` 增大而退化
- GMM1 tile 从 256→128 的 Cube 效率影响（tile 数量翻倍、单 tile 计算量减半）
- 激活量化阶段 GM→UB gather weight 的开销（`SetValue` 逐元素，128 次）
- `DIST_BRC_B32` + `Mul`(vec-vec) 相比原 `Muls`(vec-scalar) 的额外开销
- Unpermute 省去 Muls 和权重加载的加速效果
- 整体性能：权重前移收益（Unpermute 退化纯加法、负载从串行尾段移到并行中段）是否能抵消 GMM1 基本块缩小的损失

---

## 12. 遗留问题

### 12.1 权重读取未使用 reinterpret_cast 位重解释

**现状**：§6 设计了 `int32 → float` 的 `reinterpret_cast` 位重解释方案，但实际代码实现中 `tripleInfoGm_` 直接声明为 `GlobalTensor<float>`（`block_epilogue_swiglu_mx_quant.h:230`），`GetValue` 按 float 直接读取 4 字节，未经过 `reinterpret_cast<int32_t*>` → `reinterpret_cast<float*>` 的显式位重解释。

```cpp
// 实际实现（简化了位重解释）
AscendC::GlobalTensor<float> tripleInfoGm_;  // 直接声明为 float
tripleInfoGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.tripleInfoGmAddr));
float w = tripleInfoGm_.GetValue(...);  // 直接按 float 读
```

```cpp
// 设计方案（显式位重解释）
AscendC::GlobalTensor<int32_t> tripleInfoGm_;  // 声明为 int32
int32_t weightBits = tripleInfoGm_.GetValue(...);
float w = *reinterpret_cast<float*>(&weightBits);  // 显式位重解释
```

**潜在风险**：
- 当前依赖 `int32` 和 `float` 的内存布局一致（同为 4 字节 little-endian），在 Ascend NPU 上实际可行
- 但这属于**隐式类型双关**，不是 C++ 标准保证的行为，存在可移植性隐患
- 若未来编译器对 `GlobalTensor<float>` 和 `GlobalTensor<int32_t>` 的底层访存指令做不同优化（如对齐、缓存策略差异），可能导致权重值错误

**建议**：后续统一为显式 `reinterpret_cast` 方案，与文档 §6 设计保持一致，消除隐患。

### 12.2 方案二（GMM1 保持 256，epilogue 拆 128）未实现

**背景**：权重前移方案有两种实现路径：

| | 方案一（已实施） | 方案二（未实施） |
|---|---|---|
| GMM1 tile M | 128（缩小） | 256（保持不变） |
| epilogue tile M | 128 | 128（独立缩小） |
| Cube 效率 | tile 数量翻倍 | 保持 256 高效 |
| 数据流 | L0C→UB 128×256 直连 | L0C→GM 256×256，epilogue 分两次从 GM 读 128×256 |
| 适用路径 | Generic + A8W4 均可 | 仅 A8W4 可行（Generic 是 L0C→UB 直连，无法拆分） |

**现状**：已切出分支 `feat/topk-weights-epilogue-split-128`，但代码未实现。

**待办**：
- 在 `feat/topk-weights-epilogue-split-128` 分支上实现方案二
- A8W4 路径：`AivGmm1PostA8W4` 中将一次 256×256 的 GM→UB 搬运 + epilogue 改为两次 128×256 循环
- Generic 路径：方案二不可行（L0C→UB 直连无 GM 中转），需另行评估或放弃
- 两个方案分别上板测性能，对比数据决定最终采用哪个

**注意**：方案二只覆盖 A8W4 路径，Generic 路径无法配套，导致两套机制并存。且 A8W4 路径中 epilogue 分两次执行，中间有 `MTE2_V` 同步开销，性能不一定优于方案一。
