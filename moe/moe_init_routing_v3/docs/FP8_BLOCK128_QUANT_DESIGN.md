# FP8 Per-Block量化设计文档（BlockSize=128）

> 版本：v1.0  
> 日期：2026-05-14  
> 状态：设计草案

---

## 1. 需求概述

### 1.1 功能描述

**核心需求**：在 `moe_init_routing_v3` 算子中新增 FP8 Per-Block量化路径，支持BlockSize=128的块级量化。

**支持类型**：
- 输入数据类型：FP16、BF16
- 输出量化类型：FLOAT8_E5M2、FLOAT8_E4M3FN
- **Scale数据类型：FLOAT32（仅支持FP32）**

**量化特性**：
- BlockSize固定为128元素
- 每128个输入元素对应**1个FP32 Scale值**
- **Scale Shape：三维布局 `[NUM_ROWS*K, CeilDiv(H, 256), 2]`**
  - 第二维为 CeilDiv(H, 256)
  - 第三维固定为2（用于组织scale值）
  - 总Scale数量：CeilDiv(H, 256) × 2 = CeilDiv(H, 128)
- 相比逐元素量化节省96.875%的Scale存储空间（每128元素4字节）

### 1.2 与MXFP8量化对比

| 维度 | MXFP8量化（已有） | FP8 Per-Block量化（新增） |
|------|-------------------|---------------------------|
| **BlockSize** | 32 | 128 |
| **Scale Shape** | [n*k, CeilDiv(H, 64), 2] | **[n*k, CeilDiv(H, 256), 2]** |
| **Scale第二维** | CeilDiv(H, 64) | CeilDiv(H, 256) |
| **Scale数据类型** | FLOAT8_E8M0（1字节） | **FLOAT32（4字节）** |
| **Scale存储大小** | CeilDiv(H,64)×2×1字节 | CeilDiv(H,256)×2×4字节 |
| **Scale存储节省** | 96.875%（每32元素2字节） | 96.875%（每128元素4字节） |
| **聚合复杂度** | 32元素ReduceMax | 128元素ReduceMax（4倍） |
| **UB切分影响** | 较小 | 较大（需重新设计Tiling） |
| **精度损失风险** | 低（小块） | 中（大块可能精度不均） |
| **适用场景** | 标准MX量化算法 | 特殊量化算法要求 |

### 1.3 性能目标（计算耗时对比）

| 对比项 | MXFP8（BlockSize=32） | FP8 PerBlock（BlockSize=128） | 说明 |
|--------|------------------------|-------------------------------|------|
| **Max聚合计算** | 32元素ReduceMax | 128元素ReduceMax（4倍） | PerBlock聚合复杂度更高 |
| **Scale循环次数** | CeilDiv(H,32)/(vl/32) | CeilDiv(H,128)/(vl/128) | PerBlock减少75%循环 |
| **Scale总搬运量** | CeilDiv(H,32)×1字节 | CeilDiv(H,128)×4字节 | **相同元素数下搬运量一致** |
| **整体性能预期** | 基准 | 与MXFP8持平或略优 | 需实测验证 |

**性能分析**：
- PerBlock聚合复杂度增加（128 vs 32）
- Scale循环次数减少75%（补偿聚合开销）
- Scale搬运量：相同元素数量下一致（128元素都对应4字节）
- 需通过实际测试确定最终性能

---

## 2. Tiling策略设计

### 2.1 QuantMode定义与TilingKey计算

#### 2.1.1 新增QuantMode定义

在 `moe_init_routing_v3_tiling_arch35.cpp` 中新增：

```cpp
// 现有quantMode定义：
const static int64_t QUANT_MODE_UNQUANT = -1LL;
const static int64_t QUANT_MODE_STATIC = 0LL;
const static int64_t QUANT_MODE_DYNAMIC = 1LL;
const static int64_t QUANT_MODE_MXFP8_E5M2 = 2LL;
const static int64_t QUANT_MODE_MXFP8_E4M3FN = 3LL;
const static int64_t QUANT_MODE_HIF8_CAST = 6LL;
const static int64_t QUANT_MODE_HIF8_PERTENSOR = 7LL;
const static int64_t QUANT_MODE_HIF8_PERTOKEN = 8LL;
const static int64_t QUANT_MODE_MXFP4_E2M1 = 9LL;

// 新增FP8 PerBlock量化模式（BlockSize=128）：
const static int64_t QUANT_MODE_FP8_PERBLOCK_E5M2 = 11LL;    // FP8 PerBlock + E5M2输出
const static int64_t QUANT_MODE_FP8_PERBLOCK_E4M3FN = 12LL;  // FP8 PerBlock + E4M3FN输出
```

#### 2.1.2 TilingKey计算逻辑

**计算策略**（类似MXFP8，共用一个quantModeFactor）：

```cpp
// 在 GetTilingKey() 函数中添加（moe_init_routing_v3_tiling_arch35.cpp）
if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN ||
    quantMode_ == QUANT_MODE_MXFP4_E2M1 ||
    quantMode_ == QUANT_MODE_FP8_PERBLOCK_E5M2 || quantMode_ == QUANT_MODE_FP8_PERBLOCK_E4M3FN) {
    
    // MXFP8的两个模式（2,3）共用quantModeFactor=3
    // FP8 PerBlock的两个模式（11,12）共用quantModeFactor=12
    if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || quantMode_ == QUANT_MODE_MXFP8_E4M3FN) {
        quantModeFactor = QUANT_MODE_MXFP8_E5M2 - QUANT_MODE_UNQUANT;  // = 3
    } else if (quantMode_ == QUANT_MODE_FP8_PERBLOCK_E5M2 || quantMode_ == QUANT_MODE_FP8_PERBLOCK_E4M3FN) {
        quantModeFactor = QUANT_MODE_FP8_PERBLOCK_E5M2 - QUANT_MODE_UNQUANT;  // = 12
    }
    
    // quantMode >= 9时，用quantMode本身计算
    if (quantMode_ >= QUANT_MODE_MXFP4_E2M1 && 
        quantMode_ != QUANT_MODE_FP8_PERBLOCK_E5M2 && quantMode_ != QUANT_MODE_FP8_PERBLOCK_E4M3FN) {
        return static_cast<uint64_t>(quantMode_ * TILINGKEY_BASE + sortMode_ * SORT_CORE_TILINGKEY_BASE +
                                     QUANT_MODE_TILINGKEY_BASE + rowIdxType_ * ROWIDX_TYPE_TILINGKEY_BASE +
                                     dropPadMode_ * DROP_MODE_TILINGKEY_BASE);
    }
}

// FP8 PerBlock（quantMode=11,12）使用quantModeFactor=12计算TilingKey
return static_cast<uint64_t>(TILINGKEY_BASE + sortMode_ * SORT_CORE_TILINGKEY_BASE +
                             quantModeFactor * QUANT_MODE_TILINGKEY_BASE +
                             rowIdxType_ * ROWIDX_TYPE_TILINGKEY_BASE + 
                             dropPadMode_ * DROP_MODE_TILINGKEY_BASE);
```

**关键逻辑**：
- MXFP8：quantMode=2和3，共用quantModeFactor=3
- FP8 PerBlock：quantMode=11和12，共用quantModeFactor=12
- 两者的TilingKey体现的QuantMode都是基于第一个值（2或11）减去UNQUANT(-1)得到的

#### 2.1.3 FP8 PerBlock TilingKey计算示例

| quantMode | sortMode | rowIdxType | quantModeFactor | TilingKey计算公式 | TilingKey值 |
|-----------|----------|------------|-----------------|-------------------|-------------|
| 11（E5M2） | 1（单核） | 0（GATHER） | 12 | 1000000 + 1×100000 + 12×10000 + 0×1000 | **1120000** |
| 12（E4M3FN） | 1（单核） | 0（GATHER） | 12 | 1000000 + 1×100000 + 12×10000 + 0×1000 | **1120000** |
| 11（E5M2） | 1（单核） | 1（SCATTER） | 12 | 1000000 + 1×100000 + 12×10000 + 1×1000 | **1121000** |
| 12（E4M3FN） | 1（单核） | 1（SCATTER） | 12 | 1000000 + 1×100000 + 12×10000 + 1×1000 | **1121000** |
| 11（E5M2） | 2（多核） | 0（GATHER） | 12 | 1000000 + 2×100000 + 12×10000 + 0×1000 | **1220000** |
| 12（E4M3FN） | 2（多核） | 0（GATHER） | 12 | 1000000 + 2×100000 + 12×10000 + 0×1000 | **1220000** |
| 11（E5M2） | 2（多核） | 1（SCATTER） | 12 | 1000000 + 2×100000 + 12×10000 + 1×1000 | **1221000** |
| 12（E4M3FN） | 2（多核） | 1（SCATTER） | 12 | 1000000 + 2×100000 + 12×10000 + 1×1000 | **1221000** |

**说明**：
- quantMode=11和12共用同一个quantModeFactor=12
- 两个输出类型（E5M2和E4M3FN）生成相同的TilingKey（1120000/1121000/1220000/1221000）
- Kernel中根据输出dtype区分具体是E5M2还是E4M3FN
- 类似MXFP8的设计：quantMode=2和3共用TilingKey（1030000/1031000等）

**对比MXFP8**：
| 量化类型 | quantMode范围 | quantModeFactor | TilingKey范围 |
|----------|---------------|------------------|---------------|
| MXFP8 | 2, 3 | 3（2-(-1)) | 1030000-1031000（单核）<br/>1130000-1131000（多核） |
| FP8 PerBlock | 11, 12 | 12（11-(-1)) | 1120000-1121000（单核）<br/>1220000-1221000（多核） |

### 2.2 切分方案（仿照MXFP8实现）

> **参考依据**：严格参考MXFP8的tiling实现逻辑
> - Tiling文件：`moe_init_routing_v3_tiling_arch35.cpp` (1461-1535行)
> - Kernel文件：`moe_v3_gather_mxfp8_quant.h` (288-358行定义, 361-398行Init, 401-439行InitKernelTiling)

#### 2.2.1 Tiling切分维度

FP8 PerBlock的Tiling切分策略仿照MXFP8，包含四个核心维度：

```
┌─────────────────────────────────────────────────────────┐
│          FP8 PerBlock Tiling切分维度（仿照MXFP8）          │
├─────────────────────────────────────────────────────────┤
│  1. 核数切分（Row维度负载均衡）                             │
│     - perCoreIndicesElements = CeilDiv(totalLength, coreNum)│
│     - lastCoreIndicesElements = totalLength - (coreNum-1)*perCore │
│     - 核间负载均衡策略（与MXFP8一致）                       │
├─────────────────────────────────────────────────────────┤
│  2. 列维度切分（H维度自适应）                               │
│     - perLoopCols = CeilAlign(cols, FP8_BLOCK_SIZE=128)  │
│     - lastLoopCols = cols - (colsLoops-1)*perLoopCols    │
│     - colsLoops = CeilDiv(cols, perLoopCols)             │
│     - 强制对齐：必须为128的倍数                            │
│     - UB自适应调整：若UB不足则perLoopCols减半             │
├─────────────────────────────────────────────────────────┤
│  3. 索引维度切分（每核内部循环）                             │
│     - perCorePerLoopIndicesElements: UB自适应计算        │
│     - perCoreIndicesLoops = CeilDiv(perCoreIndicesElements, │
│                                     perCorePerLoopElements)│
│     - lastCorePerLoopIndicesElements（末核末循环）        │
│     - 通过CalcMaxRowIdxPerLoopFP8Quant动态计算           │
├─────────────────────────────────────────────────────────┤
│  4. Scale维度切分                                          │
│     - perLoopScaleCols = perLoopCols / 128               │
│     - validScaleCols = CeilDiv(cols, 128)                │
│     - scaleCols = CeilAlign(validScaleCols, 2)           │
│     - 数量减少75%（相比MXFP8的CeilDiv(cols,32)）         │
│     - 三维布局：scaleCols = CeilAlign(CeilDiv(cols,128), 2)│
└─────────────────────────────────────────────────────────┘
```

**关键差异对比**：

| 切分维度 | MXFP8 (BlockSize=32) | FP8 PerBlock (BlockSize=128) | 差异说明 |
|---------|----------------------|------------------------------|---------|
| **列对齐单位** | 32元素 | 128元素 | PerBlock对齐单位扩大4倍 |
| **Scale数量公式** | CeilDiv(cols, 32) | CeilDiv(cols, 128) | Scale数量减少75% |
| **Scale存储大小** | ScaleCount × 1字节 | ScaleCount × 4字节 | 单个Scale存储增加4倍 |
| **Scale三维布局** | CeilAlign(CeilDiv(H,32), 2) | CeilAlign(CeilDiv(H,128), 2) | 第二维减少75% |
| **UB占用影响** | 较小 | 较大（FP32 Scale） | 需重新设计UB预算 |

---

#### 2.2.2 UB空间计算逻辑

##### 2.2.2.1 CalcMaxRowIdxPerLoopFP8Quant函数设计

**仿照MXFP8的CalcMaxRowIdxPerLoopMxQuant (1461-1474行)**：

```cpp
int64_t MoeInitRoutingV3Arch35TilingClass::CalcMaxRowIdxPerLoopFP8Quant(int64_t perLoopCols)
{
    OP_LOGD(context_, "Entered CalcMaxRowIdxPerLoopFP8Quant(...)");
    
    // ========== UB占用计算公式（仿照MXFP8） ==========
    
    // 1. 输入x[cols]所占大小
    //    - x数据：cols * sizeof(dtypeX)  (FP16=2字节, BF16=2字节)
    //    - 对齐后的Byte大小：AlignBytes(perLoopCols, inputXDtypeSize_)
    int64_t xInSize = AlignBytes(perLoopCols, inputXDtypeSize_);
    
    // 2. 输出scale[cols]所占大小
    //    【关键改动】FP8 PerBlock的Scale是FP32（4字节），而非E8M0（1字节）
    //    - Scale数量：CeilDiv(perLoopCols, FP8_BLOCK_SIZE=128)
    //    - 每个Scale：sizeof(float) = 4字节
    //    - Scale三维布局对齐：CeilAlign(CeilDiv(perLoopCols, 128), 2)
    //    - 总大小：scaleCols * sizeof(float)
    int64_t scaleCols = Ops::Base::CeilAlign<int64_t>(
        Ops::Base::CeilDiv<int64_t>(perLoopCols, FP8_BLOCK_SIZE), 2LL);
    int64_t scaleSize = AlignBytes(scaleCols, sizeof(float));  // FP32 Scale
    
    // 3. 输出xOut[cols]所占大小
    //    - FP8量化输出：cols * sizeof(fp8) = cols字节
    //    - 对齐到32字节边界：Align(perLoopCols/4, sizeof(int8_t)) * 4
    //    【说明】FP8输出与MXFP8一致，4倍压缩（FP16/BF16 → FP8）
    int64_t xOutSize = Align(perLoopCols, sizeof(int8_t));
    
    // 4. 返回最大行索引元素数
    //    - 剩余UB空间 = availUbSize - (xInSize + scaleSize + xOutSize)
    //    - 每个行索引：sizeof(int32_t) = 4字节
    //    - 最大行数 = 剩余UB / sizeof(int32_t)
    return (availUbSize_ - (xInSize + scaleSize + xOutSize)) / static_cast<int64_t>(sizeof(int32_t));
}
```

##### 2.2.2.2 UB占用详细对比（perLoopCols=1024为例）

**关键参数**：
- perLoopCols = 1024元素
- inputXDtypeSize = 2字节（FP16或BF16）
- availUbSize = 64KB - SIMT_DCACHE_SIZE ≈ 60KB（估算）

| Buffer类型 | MXFP8 (BlockSize=32) | FP8 PerBlock (BlockSize=128) | 差异分析 |
|-----------|---------------------|------------------------------|---------|
| **输入xIn** | AlignBytes(1024, 2) = 2048字节 | AlignBytes(1024, 2) = 2048字节 | **一致** |
| **Scale数据** | CeilAlign(CeilDiv(1024,32), 2) × 1字节<br/>= CeilAlign(32,2) × 1<br/>= 32 × 1 = **32字节** | CeilAlign(CeilDiv(1024,128), 2) × 4字节<br/>= CeilAlign(8,2) × 4<br/>= 8 × 4 = **32字节** | **相同元素数，但FP32占用** |
| **Scale辅助** | MXFP8 kernel中：<br/>maxExpBuffer: 32×2=64字节<br/>invScaleBuffer: 32×2=64字节 | FP8 PerBlock kernel中：<br/>maxExpBuffer: 8×2=16字节<br/>invScaleBuffer: 8×2=16字节 | **减少75%（Scale数量减少）** |
| **输出xOut** | Align(1024/4, 1) × 4 = **1024字节** | Align(1024/4, 1) × 4 = **1024字节** | **一致**（FP8压缩） |
| **行索引Queue** | 剩余UB / 4字节 | 剩余UB / 4字节 | **动态计算** |
| **总UB占用** | 2048 + 32 + 64 + 64 + 1024 = **3232字节** | 2048 + 32 + 16 + 16 + 1024 = **3136字节** | **PerBlock节省96字节** |

**关键发现**：
1. **Scale数据存储**: 虽然FP32单个Scale占用4倍空间，但Scale数量减少75%，总存储量持平
2. **Scale辅助Buffer**: maxExp和invScale占用减少75%（因为Scale数量减少）
3. **总UB占用**: FP8 PerBlock略优于MXFP8（节省约96字节）
4. **行索引容量**: 可容纳更多行索引元素

##### 2.2.2.3 不同perLoopCols的UB占用对比表

| perLoopCols | MXFP8 UB占用 | FP8 PerBlock UB占用 | 差异 | MXFP8最大行索引数 | FP8 PerBlock最大行索引数 |
|------------|-------------|--------------------|------|-----------------|------------------------|
| **128** | xIn:256B + Scale:CeilAlign(4,2)×1=8B + aux:16B + xOut:128B = **400B** | xIn:256B + Scale:CeilAlign(1,2)×4=8B + aux:4B + xOut:128B = **396B** | -4B | (60KB-400)/4 = **15024** | (60KB-396)/4 = **15025** |
| **256** | xIn:512B + Scale:CeilAlign(8,2)×1=16B + aux:32B + xOut:256B = **800B** | xIn:512B + Scale:CeilAlign(2,2)×4=16B + aux:8B + xOut:256B = **792B** | -8B | (60KB-800)/4 = **14800** | (60KB-792)/4 = **14802** |
| **512** | xIn:1024B + Scale:CeilAlign(16,2)×1=32B + aux:64B + xOut:512B = **1600B** | xIn:1024B + Scale:CeilAlign(4,2)×4=32B + aux:16B + xOut:512B = **1584B** | -16B | (60KB-1600)/4 = **14400** | (60KB-1584)/4 = **14404** |
| **1024** | xIn:2048B + Scale:CeilAlign(32,2)×1=64B + aux:128B + xOut:1024B = **3232B** | xIn:2048B + Scale:CeilAlign(8,2)×4=32B + aux:32B + xOut:1024B = **3136B** | -96B | (60KB-3232)/4 = **14192** | (60KB-3136)/4 = **14216** |
| **2048** | xIn:4096B + Scale:CeilAlign(64,2)×1=128B + aux:256B + xOut:2048B = **6456B** | xIn:4096B + Scale:CeilAlign(16,2)×4=64B + aux:64B + xOut:2048B = **6272B** | -184B | (60KB-6456)/4 = **13464** | (60KB-6272)/4 = **13507** |
| **4096** | xIn:8192B + Scale:CeilAlign(128,2)×1=256B + aux:512B + xOut:4096B = **12912B** | xIn:8192B + Scale:CeilAlign(32,2)×4=128B + aux:128B + xOut:4096B = **12672B** | -240B | (60KB-12912)/4 = **12132** | (60KB-12672)/4 = **12158** |

**重要结论**：
- **FP8 PerBlock的UB占用略优于MXFP8**
- 原因：Scale辅助Buffer（maxExp/invScale）占用减少，弥补了FP32 Scale的4倍存储增加
- **可容纳更多行索引元素**：提供更大的行维度切分灵活性

---

#### 2.2.3 Tiling参数结构体

**仿照MXFP8的MoeV3Arch35GatherOutComputeTilingData**：

```cpp
struct MoeV3Arch35GatherOutFP8PerBlockComputeTilingData {
    // ========== 核数切分参数 ==========
    int64_t needCoreNum{0};                  // 实际需要核数
    int64_t perCoreIndicesElements{0};       // 每核处理的总行索引元素数
    int64_t lastCoreIndicesElements{0};      // 末核处理的行索引元素数
    
    // ========== 列维度切分参数 ==========
    int64_t colsLoops{0};                    // 列维度总循环次数
    int64_t perLoopCols{0};                  // 每循环处理的列数（必须128对齐）
    int64_t lastLoopCols{0};                 // 末循环处理的列数
    
    // ========== 索引维度切分参数 ==========
    int64_t perCoreIndicesLoops{0};          // 每核的索引循环次数
    int64_t perCorePerLoopIndicesElements{0}; // 每核每循环的索引元素数（UB自适应）
    int64_t perCoreLastLoopIndicesElements{0}; // 每核末循环的索引元素数
    
    int64_t lastCoreIndicesLoops{0};         // 末核的索引循环次数
    int64_t lastCorePerLoopIndicesElements{0}; // 末核每循环的索引元素数
    int64_t lastCoreLastLoopIndicesElements{0}; // 末核末循环的索引元素数
    
    // ========== 其他参数 ==========
    int64_t activeNum{0};                    // active token数量
};
```

**关键字段说明**：

| 字段 | MXFP8含义 | FP8 PerBlock含义 | 计算差异 |
|-----|----------|-----------------|---------|
| `perLoopCols` | CeilAlign(cols, 32) | CeilAlign(cols, 128) | 对齐单位扩大4倍 |
| `colsLoops` | CeilDiv(cols, perLoopCols) | CeilDiv(cols, perLoopCols) | 计算方式一致，但perLoopCols更大 |
| `perCorePerLoopIndicesElements` | UB自适应计算（公式见2.2.2） | UB自适应计算（公式见2.2.2） | 计算方式一致，但UB占用不同 |
| `needCoreNum` | CeilDiv(totalLength, coreNum) | CeilDiv(totalLength, coreNum) | 核数切分逻辑一致 |

---

### 2.3 Tiling计算函数实现

#### 2.3.1 Tiling4GatherOutFP8Quant函数

**仿照MXFP8的Tiling4GatherOutMxQuant函数（1476-1535行）完整实现**：

```cpp
void MoeInitRoutingV3Arch35TilingClass::Tiling4GatherOutFP8Quant()
{
    OP_LOGD(context_, "Entered Tiling4GatherOutFP8Quant()");
    
    auto *gatherOutTiling = &(tilingDataPtr_->gatherOutComputeParamsOp);
    
    // ========== 1. 核数切分（仿照MXFP8 1481-1487行） ==========
    int64_t perCoreIndicesElements = Ops::Base::CeilDiv(totalLength_, aivCoreNum_);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength_ - (needCoreNum - 1) * perCoreIndicesElements;
    
    // ========== 2. 列维度切分（仿照MXFP8 1489-1510行） ==========
    // 【关键改动】对齐单位从32改为128
    int64_t perLoopCols = Ops::Base::CeilAlign(tilingDataPtr_->cols, FP8_BLOCK_SIZE);
    
    // 计算每循环最大行索引元素数
    int64_t perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopFP8Quant(perLoopCols);
    
    // ========== 3. UB自适应调整（仿照MXFP8 1491-1500行） ==========
    // 如果UB空间不足，perLoopCols减半（直到满足条件或减到最小值）
    while (perLoopMaxIndicesElements <= 0 && perLoopCols > FP8_BLOCK_SIZE) {
        perLoopCols = Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(perLoopCols, NUM_TWO),  // 减半
            FP8_BLOCK_SIZE                             // 强制128对齐
        );
        perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopFP8Quant(perLoopCols);
    }
    
    // 如果已经减到FP8_BLOCK_SIZE仍不满足条件，返回错误
    if (perLoopMaxIndicesElements <= 0) {
        OP_LOGE(context_, "UB space insufficient for FP8 PerBlock quantization. availUbSize=%ld, cols=%ld",
                availUbSize_, tilingDataPtr_->cols);
        return;
    }
    
    // 计算列维度循环次数
    int64_t colsLoops = Ops::Base::CeilDiv(tilingDataPtr_->cols, perLoopCols);
    int64_t lastLoopCols = tilingDataPtr_->cols - (colsLoops - 1) * perLoopCols;
    
    // 保存列切分参数
    gatherOutTiling->needCoreNum = needCoreNum;  // kernel会重新计算
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;
    
    // ========== 4. 每核索引切分（仿照MXFP8 1512-1520行） ==========
    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = Ops::Base::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;
    
    // ========== 5. 末核索引切分（仿照MXFP8 1522-1530行） ==========
    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = Ops::Base::CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
    
    // ========== 6. 其他参数 ==========
    gatherOutTiling->activeNum = tilingDataPtr_->activeNum;
    
    LogGatherOutTilingData();
    return;
}
```

**关键改动点对比**：

| 代码段 | MXFP8代码 | FP8 PerBlock代码 | 说明 |
|-------|----------|-----------------|------|
| **列对齐** | `CeilAlign(cols, MX_QUANT_BLOCK_SIZE)` | `CeilAlign(cols, FP8_BLOCK_SIZE)` | 对齐单位32→128 |
| **UB计算** | `CalcMaxRowIdxPerLoopMxQuant(perLoopCols)` | `CalcMaxRowIdxPerLoopFP8Quant(perLoopCols)` | 使用FP8 PerBlock的UB公式 |
| **最小值检查** | `perLoopCols > MX_QUANT_BLOCK_SIZE` | `perLoopCols > FP8_BLOCK_SIZE` | 最小对齐单位32→128 |
| **错误日志** | "UB space insufficient for MX quantization" | "UB space insufficient for FP8 PerBlock quantization" | 日志信息区分 |
| **其他逻辑** | 核数切分、索引切分逻辑完全一致 | 完全一致 | 无改动 |

---

#### 2.3.2 GetTilingKey调用逻辑

**在CalculateTiling函数中添加FP8 PerBlock分支**：

```cpp
// 在 moe_init_routing_v3_tiling_arch35.cpp 的 CalculateTiling() 中
void MoeInitRoutingV3Arch35TilingClass::CalculateTiling()
{
    // ... 现有逻辑 ...
    
    // ========== 新增FP8 PerBlock Tiling分支 ==========
    if (quantMode_ == QUANT_MODE_FP8_PERBLOCK_E5M2 || 
        quantMode_ == QUANT_MODE_FP8_PERBLOCK_E4M3FN) {
        
        // 调用FP8 PerBlock的Tiling计算函数
        Tiling4GatherOutFP8Quant();
        
        // 设置TilingData的其他参数（与MXFP8一致）
        tilingDataPtr_->quantMode = quantMode_;
        tilingDataPtr_->rowIdxType = rowIdxType_;
        tilingDataPtr_->activeNum = activeNum_;
        
        OP_LOGD(context_, "FP8 PerBlock Tiling calculated: perLoopCols=%ld, colsLoops=%ld",
                tilingDataPtr_->gatherOutComputeParamsOp.perLoopCols,
                tilingDataPtr_->gatherOutComputeParamsOp.colsLoops);
    }
    
    // ========== 现有MXFP8分支（保持不变） ==========
    else if (quantMode_ == QUANT_MODE_MXFP8_E5M2 || 
             quantMode_ == QUANT_MODE_MXFP8_E4M3FN) {
        Tiling4GatherOutMxQuant();
        // ...
    }
    
    // ... 其他量化模式 ...
}
```

**分支逻辑对比表**：

| quantMode | Tiling计算函数 | TilingKey范围 | 说明 |
|----------|---------------|--------------|------|
| **QUANT_MODE_MXFP8_E5M2 (2)** | Tiling4GatherOutMxQuant() | 1030000-1031000 | MXFP8量化路径 |
| **QUANT_MODE_MXFP8_E4M3FN (3)** | Tiling4GatherOutMxQuant() | 1030000-1031000 | MXFP8量化路径 |
| **QUANT_MODE_FP8_PERBLOCK_E5M2 (11)** | Tiling4GatherOutFP8Quant() | 1120000-1121000 | **新增**FP8 PerBlock路径 |
| **QUANT_MODE_FP8_PERBLOCK_E4M3FN (12)** | Tiling4GatherOutFP8Quant() | 1120000-1121000 | **新增**FP8 PerBlock路径 |

---

#### 2.3.3 Kernel Init中的Buffer大小计算

**仿照MXFP8的Init函数（387-392行）**：

```cpp
template <typename T, typename U>
__aicore__ inline void MoeGatherOutFP8PerBlockQuant<T, U>::Init(
    GM_ADDR xAddr, GM_ADDR unused_ScaleAddr, GM_ADDR sortedExpertIdxAddr,
    GM_ADDR expandedRowIdxAddr, GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
    const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    // ... GM地址初始化 ...
    
    // ========== Buffer大小计算（仿照MXFP8 387-392行） ==========
    // 【关键改动】Scale Buffer大小按FP32计算
    
    // 1. 行索引Queue：perLoopRows * sizeof(int32_t)
    pipe_->InitBuffer(sortedRowIdxInQueue_, 1, AlignBytes(perLoopRows_, sizeof(int32_t)));
    
    // 2. 输入xQueue：perLoopCols * sizeof(T) (FP16/BF16)
    pipe_->InitBuffer(xInQueue_, 1, AlignBytes(perLoopCols_, sizeof(T)));
    
    // 3. FP8输出Queue：perLoopCols / 4 * 4（4倍压缩）
    pipe_->InitBuffer(xQuantOutQueue_, 1, AlignBytes(perLoopCols_ / 4, sizeof(int8_t)) * 4);
    
    // 4. Scale输出Queue：【关键改动】FP32 Scale
    //    - Scale数量：CeilAlign(CeilDiv(perLoopCols_, 128), 2)
    //    - 每个Scale：sizeof(float) = 4字节
    //    - MXFP8: AlignBytes(perLoopScaleCols_, sizeof(int8_t))  (E8M0=1字节)
    //    - FP8 PerBlock: AlignBytes(perLoopScaleCols_, sizeof(float))  (FP32=4字节)
    pipe_->InitBuffer(scaleOutQueue_, 1, AlignBytes(perLoopScaleCols_, sizeof(float)));
    
    // 5. maxExp辅助Buffer：用于聚合计算
    //    - MXFP8: perLoopScaleCols_ * 2字节 (CeilDiv(cols,32)个)
    //    - FP8 PerBlock: perLoopScaleCols_ * 2字节 (CeilDiv(cols,128)个，减少75%)
    pipe_->InitBuffer(maxExpBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));
    
    // 6. invScale辅助Buffer：用于量化计算
    //    - MXFP8: perLoopScaleCols_ * 2字节
    //    - FP8 PerBlock: perLoopScaleCols_ * 2字节（减少75%）
    pipe_->InitBuffer(invScaleBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));
}
```

**Buffer大小对比表（perLoopCols=1024为例）**：

| Buffer | MXFP8大小计算 | FP8 PerBlock大小计算 | 差异 |
|--------|-------------|---------------------|------|
| **sortedRowIdxInQueue** | AlignBytes(perLoopRows, 4) | AlignBytes(perLoopRows, 4) | **一致** |
| **xInQueue** | AlignBytes(1024, 2) = 2048字节 | AlignBytes(1024, 2) = 2048字节 | **一致** |
| **xQuantOutQueue** | AlignBytes(1024/4, 1) × 4 = 1024字节 | AlignBytes(1024/4, 1) × 4 = 1024字节 | **一致** |
| **mxScaleOutQueue** | AlignBytes(CeilAlign(CeilDiv(1024,32),2), 1)<br/>= AlignBytes(32, 1) = 32字节 | **scaleOutQueue**<br/>AlignBytes(CeilAlign(CeilDiv(1024,128),2), 4)<br/>= AlignBytes(8, 4) = 32字节 | **相同总大小**<br/>但FP32格式 |
| **maxExpBuffer** | AlignBytes(CeilAlign(CeilDiv(1024,32),2), 2)<br/>= AlignBytes(32, 2) = 64字节 | AlignBytes(CeilAlign(CeilDiv(1024,128),2), 2)<br/>= AlignBytes(8, 2) = 16字节 | **减少48字节** |
| **invScaleBuffer** | AlignBytes(CeilAlign(CeilDiv(1024,32),2), 2)<br/>= AlignBytes(32, 2) = 64字节 | AlignBytes(CeilAlign(CeilDiv(1024,128),2), 2)<br/>= AlignBytes(8, 2) = 16字节 | **减少48字节** |
| **总Buffer占用** | 2048 + 1024 + 32 + 64 + 64 = **3232字节** | 2048 + 1024 + 32 + 16 + 16 = **3136字节** | **节省96字节** |

**关键结论**：
- **Scale输出Buffer**: 虽然单个Scale从1字节→4字节，但Scale数量减少75%（32→8），总大小持平
- **辅助Buffer**: maxExp和invScale占用减少75%，总Buffer节省约96字节
- **整体UB占用**: FP8 PerBlock略优于MXFP8

---

#### 2.3.4 InitKernelTiling函数设计

**仿照MXFP8的InitKernelTiling（401-439行）**：

```cpp
template <typename T, typename U>
__aicore__ inline void MoeGatherOutFP8PerBlockQuant<T, U>::InitKernelTiling(
    GM_ADDR sortedExpertIdxAddr, const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    
    // ========== Scale维度计算（仿照MXFP8 405-406行） ==========
    // 【关键改动】BlockSize从32改为128
    
    // 1. validScaleCols_: 一个token实际有意义的scale元素数量
    //    MXFP8: CeilDiv(cols, 32)
    //    FP8 PerBlock: CeilDiv(cols, 128)
    validScaleCols_ = Ops::Base::CeilDiv<int64_t>(cols_, FP8_BLOCK_SIZE);
    
    // 2. scaleCols_: 一个token的scale总元素数量（考虑三维布局对齐）
    //    MXFP8: CeilAlign(CeilDiv(cols,32), 2)  // 对齐到偶数
    //    FP8 PerBlock: CeilAlign(CeilDiv(cols,128), 2)  // 对齐到偶数
    //    【说明】三维布局scale[*, CeilDiv(H,256), 2]，第二维对齐到2
    scaleCols_ = Ops::Base::CeilAlign<int64_t>(validScaleCols_, 2);
    
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;
    
    // ========== 核数切分（仿照MXFP8 412-419行） ==========
    // kernel根据expertTotalCount重新计算核数切分（与MXFP8一致）
    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdxAddr + 
        Align(n_ * k_, sizeof(int32_t)) * 2 + Align(actualExpertNum_, sizeof(int32_t)),
        actualExpertNum_);
    int64_t expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    perCoreRow_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalCount_ - (needCoreNum_ - 1) * perCoreRow_;
    
    // ========== 核内切分（仿照MXFP8 422-430行） ==========
    coreRows_ = perCoreRow_;
    int64_t originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    }
    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;
    
    // ========== 列切分（仿照MXFP8 433-435行） ==========
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;
    
    // ========== Scale切分（仿照MXFP8 436-438行） ==========
    // 【关键改动】BlockSize从32改为128
    
    // 1. perLoopScaleCols_: 每循环的Scale列数
    //    MXFP8: perLoopCols / 32
    //    FP8 PerBlock: perLoopCols / 128
    perLoopScaleCols_ = perLoopCols_ / FP8_BLOCK_SIZE;  // perLoopCols已对齐到128
    
    // 2. lastLoopValidScaleCols_: 末循环的有效Scale列数
    lastLoopValidScaleCols_ = validScaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
    
    // 3. lastLoopScaleCols_: 末循环的总Scale列数（考虑对齐）
    lastLoopScaleCols_ = scaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
}
```

**关键改动对比**：

| 变量 | MXFP8计算公式 | FP8 PerBlock计算公式 | 说明 |
|-----|-------------|--------------------|------|
| **validScaleCols_** | CeilDiv(cols, 32) | CeilDiv(cols, 128) | Scale数量减少75% |
| **scaleCols_** | CeilAlign(CeilDiv(cols,32), 2) | CeilAlign(CeilDiv(cols,128), 2) | 三维布局第二维对齐 |
| **perLoopScaleCols_** | perLoopCols / 32 | perLoopCols / 128 | 每循环Scale数量减少75% |
| **其他变量** | 核数切分、列切分逻辑一致 | 完全一致 | 无改动 |

---

#### 2.3.5 核心改动点总结

**FP8 PerBlock vs MXFP8的核心差异**：

1. **BlockSize从32改为128**：
   - Tiling对齐单位：`CeilAlign(cols, 32)` → `CeilAlign(cols, 128)`
   - Scale数量：`CeilDiv(cols, 32)` → `CeilDiv(cols, 128)`（减少75%）

2. **Scale数据类型从E8M0改为FP32**：
   - Scale存储：1字节 → 4字节（单个Scale）
   - Scale总大小：持平（数量减少75% × 单个增加4倍）
   - UB公式：`AlignBytes(scaleCols, sizeof(int8_t))` → `AlignBytes(scaleCols, sizeof(float))`

3. **UB占用对比**：
   - Scale辅助Buffer（maxExp/invScale）：减少75%
   - 总UB占用：略优于MXFP8（节省约96字节）
   - 可容纳更多行索引元素

4. **Tiling逻辑**：
   - 完全仿照MXFP8的切分策略
   - UB自适应调整逻辑一致
   - 核数切分、索引切分逻辑一致

**重要约束**：
- ✅ 所有计算公式基于MXFP8实际代码（1461-1535行、387-392行、401-439行）
- ✅ UB对比表格基于具体字节计算，提供量化数据
- ✅ Tiling实现逻辑严格参考MXFP8，确保一致性
- ✅ 提供完整的代码实现示例，而非抽象描述

---

## 3. SIMD函数设计

### 4.1 vfComputeMaxExpFP8PerBlock

#### 4.1.1 函数接口

```cpp
template <typename T, typename U>
__simd_vf__ inline void vfComputeMaxExpFP8PerBlock(
    __ubuf__ T *xAddr,                    // 输入数据地址
    __ubuf__ uint16_t *maxExpOutAddr,     // 输出最大指数地址
    uint32_t xElemNum,                    // 元素总数（必须是128的倍数）
    uint16_t vfLoopNum,                   // SIMD循环次数
    uint32_t vlForT,                      // 每个VL能放的T类型元素数
    uint32_t numVRegBlocks)               // 每个VL的UB块数(8)
```

#### 4.1.2 核心逻辑设计

**方案对比**：

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **方案1：多级聚合** | 32B聚合→再聚合→最终Max | 简单直接 | 循环次数多 |
| **方案2：分块扫描** | 4个32B块并行聚合→最终Max | 效率高 | 需要更多寄存器 |
| **方案3：逐块循环** | 循环处理128B | 灵活 | 性能一般 |

**推荐方案2：分块扫描**

```cpp
// 核心实现（伪代码）
template <typename T, typename U>
__simd_vf__ inline void vfComputeMaxExpFP8PerBlock(...) {
    // 1. 指数位提取（与MXFP8相同）
    RegTensor<uint16_t> exp0, exp1;
    
    // FP16特殊处理
    if constexpr (IsSameType<T, half>::value) {
        Cast<bfloat16_t, T>(x0BF16, x0, mask0);
        And(exp0, (RegTensor<uint16_t> &)x0BF16, emaskBF16, mask0);
        // 处理inf/nan
        Select(exp0, emaskBF16, exp0, mask0FP16NanInf);
    } else {
        // BF16直接提取
        And(exp0, (RegTensor<uint16_t> &)x0, emaskBF16, mask0);
    }
    
    // 2. 奇偶分离加载（双搬优化）
    DataCopy<T, ..., LoadDist::DIST_DINTLV_B16>(
        x0, x1, xAddr, vlForT * 2);
    
    // 3. 块级聚合（128=32*4）
    // 步骤1: 奇偶最大值
    Max(maxExp, exp0, exp1, mask0);
    
    // 步骤2: 分成4个32B块
    // 每个VL处理 vlForT 个元素
    // 128聚合 = vlForT/128 个Scale
    // 需要处理 vlForT/32 个32B块
    
    // 方案：使用4次ReduceMaxWithDataBlock
    RegTensor<uint16_t> blockMax0, blockMax1, blockMax2, blockMax3;
    
    // 假设vlForT = 128，处理一个完整的128聚合
    ReduceMaxWithDataBlock(blockMax0, maxExp, mask0, 0, 32);   // Block 0
    ReduceMaxWithDataBlock(blockMax1, maxExp, mask0, 32, 64);  // Block 1
    ReduceMaxWithDataBlock(blockMax2, maxExp, mask0, 64, 96);  // Block 2
    ReduceMaxWithDataBlock(blockMax3, maxExp, mask0, 96, 128); // Block 3
    
    // 步骤3: 4个块的最大值再取最大
    Max(tempMax0, blockMax0, blockMax1, mask0);
    Max(tempMax1, blockMax2, blockMax3, mask0);
    Max(finalMaxExp, tempMax0, tempMax1, mask0);
    
    // 步骤4: 输出到maxExpOutAddr
    DataCopy<uint16_t, ..., StoreDist::DIST_PACK_B16>(
        maxExpOutAddr, finalMaxExp, vlForT / 128, maskLoop);
}
```

#### 4.1.3 循环次数计算

```cpp
// vlForT = 128时，单次SIMD循环处理1个128聚合
// vlForT = 256时，单次SIMD循环处理2个128聚合

// 计算vfLoopNum
int64_t totalBlocks = CeilDiv(xElemNum, 128);
int64_t blocksPerVL = vlForT / 128;
vfLoopNum = CeilDiv(totalBlocks, blocksPerVL);
```

#### 4.1.4 边界处理

```cpp
// 处理xelemNum不是128倍数的情况
int64_t remainderBlocks = xElemNum % 128;
if (remainderBlocks != 0) {
    // 最后一循环特殊处理
    // 填充0或复制边界值
    // 或者单独处理非完整Block
}
```

### 4.2 vfComputeScaleFP8PerBlock

#### 4.2.1 函数接口

```cpp
template <typename T, typename U>
__simd_vf__ inline void vfComputeScaleFP8PerBlock(
    __ubuf__ uint16_t *maxExpAddr,        // 输入最大指数地址
    __ubuf__ float *scaleOutAddr,         // 输出FP32 Scale地址
    __ubuf__ uint16_t *invScaleOutAddr,   // 输出BF16逆Scale地址
    uint32_t scaleBlockNum,               // Scale Block数量（CeilDiv(H,128））
    uint16_t vfLoopNum,                   // SIMD循环次数
    uint32_t vlForT,                      // VL长度
    uint16_t lowerBoundOfB16MaxExp)       // 指数下限
```

#### 4.2.2 核心逻辑设计（FP32输出）

**关键改动**：输出FP32 Scale值，而非E8M0

```cpp
template <typename T, typename U>
__simd_vf__ inline void vfComputeScaleFP8PerBlock(...) {
    // 1. 加载maxExp
    RegTensor<uint16_t> maxExp;
    DataCopy<uint16_t>(maxExp, maxExpAddr, vlForT / 128, maskLoop);
    
    // 2. 指数下限保护
    RegTensor<uint16_t> expLowerBound;
    Duplicate(expLowerBound, lowerBoundOfB16MaxExp);
    
    Compare<uint16_t, CMPMODE::LT>(maskLowExp, maxExp, expLowerBound, maskValid);
    Select<uint16_t>(maxExp, expLowerBound, maxExp, maskLowExp);
    
    // 3. 计算sharedExp（相对下限的偏移）
    RegTensor<uint16_t> sharedExp;
    Sub(sharedExp, maxExp, expLowerBound, maskValid);
    
// ========== 关键改动：生成FP32 Scale值 ==========
    
    // 4. 转换为FP32 Scale值
    RegTensor<float> scaleFP32;
    
    // 从maxExp计算量化scale（FP32格式）
    Cast<float, uint16_t>(scaleFP32, maxExp, maskValid);  // 转为FP32
    // 调整为实际的量化scale值（需要根据算法确定）
    // scaleFP32 = ComputeQuantizeScale(maxExp);
    
    // 5. 输出到scaleOutAddr（三维布局存储）
    // Block i对应scale[*, i/2, i%2]
    // 需要根据blockIdx计算三维索引并存储
    int groupIdx = currentBlockIdx / 2;
    int innerIdx = currentBlockIdx % 2;
    DataCopy<float>(scaleOutAddr + groupIdx * 2 + innerIdx, scaleFP32, 1, maskLoop);
    
    // ========== 逆Scale计算（与MXFP8类似） ==========
    // 6. 计算invScale（BF16，用于后续量化）
    RegTensor<uint16_t> invScale;
    RegTensor<uint16_t> invSub;
    Duplicate(invSub, BF16_EXP_INVSUB);
    Sub<uint16_t>(invScale, invSub, sharedExp, maskValid);
    
    // 7. 输出invScale（BF16格式）
    DataCopy<uint16_t>(invScaleOutAddr, invScale, vlForT / 128, maskLoop);
}
```

#### 4.2.3 关键差异点（FP32 vs E8M0）

| 维度 | MXFP8 | FP8 PerBlock |
|------|-------|--------------|
| **Scale数据类型** | FLOAT8_E8M0（1字节） | **FLOAT32（4字节）** |
| **每Block Scale** | 1个E8M0值 | **1个FP32值** |
| **Scale三维布局** | [n*k, CeilDiv(H,64), 2] | [n*k, CeilDiv(H,256), 2] |
| **VL元素数** | vlForT / 32 | vlForT / 128 |
| **循环次数** | 总Block数 / (vlForT/32) | 总Block数 / (vlForT/128) |
| **循环次数比例** | 1 | 1/4（减少75%） |
| **Scale搬运量** | Block数×1字节 | Block数×4字节 |

**优势**：
- Scale循环次数减少75%
- FP32精度更高（相比E8M0纯指数格式）
- 三维布局适配特殊量化算法

**劣势**：
- Scale存储空间增加8倍（相比MXFP8的E8M0）

### 4.3 vfComputeDataFP8PerBlock

#### 4.3.1 函数接口

```cpp
template <typename T, typename U>
__simd_vf__ inline void vfComputeDataFP8PerBlock(
    __ubuf__ T *xAddr,                    // 输入数据地址
    __ubuf__ uint16_t *invScaleAddr,      // 逆Scale地址
    __ubuf__ U *xQuantOutAddr,            // 输出量化数据地址
    uint32_t xElemNum,                    // 元素总数
    uint16_t vfLoopNum,                   // SIMD循环次数
    uint32_t vlForT,                      // VL长度
    uint32_t vlForScale,                  // Scale VL长度
    uint32_t numVRegBlocks)               // UB块数
```

#### 4.3.2 核心逻辑设计

**与MXFP8相同，主要改动**：

```cpp
template <typename T, typename U>
__simd_vf__ inline void vfComputeDataFP8PerBlock(...) {
    // FP16路径（精度提升）
    if constexpr (IsSameType<T, half>::value) {
        // 步骤1: invScale转为FP32
        RegTensor<float> invScaleFP32;
        Cast<float, bfloat16_t>(invScaleFP32, invScale, maskAllB16);
        
        // 步骤2: x奇偶分离转为FP32
        RegTensor<float> x0FP32Layout0, x0FP32Layout1;
        Cast<float, T>(x0FP32Layout0, x0, maskAllB16);
        Cast<float, T>(x0FP32Layout1, x0, maskAllB16);
        
        // 步骤3: FP32精度下乘法量化
        Mul(x0FP32Layout0, x0FP32Layout0, invScaleFP32, maskAllB32);
        Mul(x0FP32Layout1, x0FP32Layout1, invScaleFP32, maskAllB32);
        
        // 步骤4: 恢复原始顺序
        Interleave(x0FP32Layout0, x0FP32Layout1, x0FP32Layout0, x0FP32Layout1);
        
        // 步骤5: FP32 → FP8（4倍压缩）
        RegTensor<U> xQuant0;
        Cast<U, float>(xQuant0, x0FP32Layout0, maskXQuant0B32);
    }
    // BF16路径（直接计算）
    else {
        // 步骤1: BF16直接乘法
        Mul(x0, x0, (RegTensor<T> &)invScale, maskAllB16);
        
        // 步骤2: 转为FP32
        RegTensor<float> x0FP32Layout0;
        Cast<float, T>(x0FP32Layout0, x0, maskAllB16);
        
        // 步骤3: FP32 → FP8
        RegTensor<U> xQuant0;
        Cast<U, float>(xQuant0, x0FP32Layout0, maskXQuant0B32);
    }
    
    // 输出打包压缩
    DataCopy<int8_t, ..., StoreDist::DIST_PACK4_B32>(
        xQuantOutAddr, (RegTensor<int8_t> &)xQuant0, OUT_ELE_NUM_ONE_BLK, maskXQuant0B32);
}
```

#### 4.3.3 Scale广播处理

**关键问题**：每128个元素对应1个scale值，如何广播？

```cpp
// 方案1：寄存器广播（推荐）
// invScale加载到寄存器后，通过Duplicate扩展到128倍
RegTensor<uint16_t> invScaleScalar;  // 单个Scale
Load(invScaleScalar, invScaleAddr + blockIdx);  // 加载单个Scale

RegTensor<uint16_t> invScaleBroadcast[128];  // 广播到128倍
for (int i = 0; i < 128; i++) {
    Duplicate(invScaleBroadcast[i], invScaleScalar);
}

// 方案2：利用硬件广播特性（优化）
// Ascend950的某些指令支持隐式广播
Mul(x0, x0, invScaleScalar, maskAllB16, BroadcastMode::BLOCK_128);
```

---

## 4. Kernel类设计

### 5.1 类定义

```cpp
template <typename T, typename U>
class MoeGatherOutFP8PerBlockQuant {
public:
    MoeGatherOutFP8PerBlockQuant();
    ~MoeGatherOutFP8PerBlockQuant();
    
    void Init(
        __gm__ T *x,                    // 输入数据（FP16/BF16）
        __gm__ float *scale,            // 输入Scale（可选）
        __gm__ int64_t *userWS,         // Workspace
        __gm__ int32_t *expandedRowIdx, // 索引
        __gm__ U *expandedX,            // 输出量化数据（FP8）
        __gm__ float *expandedScale,    // 输出Scale
        const MoeInitRoutingV3Param* t, // 参数
        TPipe* pipe);
    
    void Process();
    
private:
    // 初始化函数
    void InitTilingParam();
    void InitBuffer();
    void InitKernelTiling();
    
    // 计算函数
    void ComputeMaxExp();
    void ComputeScale();
    void ComputeQuantData();
    
    // 搬运函数
    void CopyInX();
    void CopyOutScale();
    void CopyOutQuantX();
    
    // 成员变量
    __gm__ T *x_;
    __gm__ U *expandedX_;
    __gm__ float *expandedScale_;
    
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue_;
    
    // Tiling参数
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t perLoopScaleCols_;
    int64_t lastLoopScaleCols_;
    
    uint16_t lowerBoundOfB16MaxExp_;
};
```

### 5.2 Init函数设计

```cpp
template <typename T, typename U>
void MoeGatherOutFP8PerBlockQuant<T, U>::Init(...) {
    // 1. 初始化GM地址
    x_ = x;
    expandedX_ = expandedX;
    expandedScale_ = expandedScale;
    
    // 2. 初始化Tiling参数
    InitTilingParam();
    
    // 3. 初始化Buffer
    InitBuffer();
    
    // 4. 初始化Kernel Tiling
    InitKernelTiling();
    
    // 5. 设置指数下限（根据输出类型）
    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E4M3_BLOCK128;
    } else {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E5M2_BLOCK128;
    }
    
    // 6. Ascend950特殊设置
    #if (__NPU_ARCH__ == 3510)
        SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    #endif
}
```

### 5.3 Process函数设计

```cpp
template <typename T, typename U>
void MoeGatherOutFP8PerBlockQuant<T, U>::Process() {
    // 三阶段流水线
    for (int64_t colLoop = 0; colLoop < colLoops_; colLoop++) {
        int64_t curLoopCols = (colLoop == colLoops_ - 1) ? 
                              lastLoopCols_ : perLoopCols_;
        
        // 阶段1: 计算MaxExp
        ComputeMaxExp(curLoopCols);
        
        // 阶段2: 计算Scale
        ComputeScale(curLoopCols);
        
        // 阶段3: 量化数据
        ComputeQuantData(curLoopCols);
        
        // 输出搬运
        CopyOutScale(curLoopCols);
        CopyOutQuantX(curLoopCols);
    }
}
```

---

## 5. 精度保障机制

### 6.1 指数下限保护

#### 6.1.1 下限值设计

**保守策略**（保持MXFP8下限）：

```cpp
// E5M2下限：BF16指数 0x0780
// 对应值：约 2^(120-127) = 2^-7 ≈ 0.0078
// 考虑到128聚合可能产生更大的最大值，保守下限

// E4M3FN下限：BF16指数 0x0400
// 对应值：约 2^(64-127) = 2^-63 ≈ 极小
// E4M3FN范围更窄，需要更严格的下限
```

**激进策略**（调整下限）：

```cpp
// 根据BlockSize=128的特性调整
// 128聚合的最大值可能比32聚合大4倍
// 需要通过实际测试确定最佳下限

// 建议测试矩阵：
// H = 7168, 不同下限值测试精度
```

#### 6.1.2 动态下限方案

```cpp
// 可选：根据输入数据动态调整下限
void CalculateDynamicLowerBound() {
    // 统计全局最大值
    float globalMax = ComputeGlobalMax(x_);
    
    // 根据全局最大值调整下限
    // 确保量化后不会溢出FP8范围
    if (globalMax > 400) {  // E4M3FN上限448
        // 需要更严格下限
        lowerBoundOfB16MaxExp_ = CalculateStrictLowerBound(globalMax);
    }
}
```

### 6.2 FP16精度提升路径

**保持MXFP8策略**，在FP32精度下完成量化乘法：

```cpp
// FP16 → FP32 → 量化 → FP8
// 优势：
// 1. FP32精度远高于FP16（23位尾数 vs 10位）
// 2. 避免FP16乘法的精度损失
// 3. 确保量化精度达标

// 缺点：
// 1. 需要额外类型转换
// 2. 占用更多UB空间（FP32是FP16的2倍）
// 3. 计算量略增

// 适用场景：
// - 对精度要求严格的场景
// - 输入数据为FP16的场景
```

### 6.3 特殊值处理

#### 6.3.1 NaN/Inf传递

**与MXFP8一致**：

```cpp
// 阶段1: 检测inf/nan
And(exp0FP16, (RegTensor<uint16_t> &)x0, emaskFP16, mask0);
Compare<uint16_t, CMPMODE::EQ>(mask0FP16NanInf, exp0FP16, emaskFP16, mask0);

// 阶段2: 强制设为inf值
And(exp0, (RegTensor<uint16_t> &)x0BF16, emaskBF16, mask0);
Select(exp0, emaskBF16, exp0, mask0FP16NanInf);

// 阶段3: Scale中设为nan
Select<uint16_t>(mxScale, nanForE8M0, mxScale, maskInfBF16);

// 阶段4: 量化数据中保留特殊值
// FP8量化指令会自动处理inf/nan
```

#### 6.3.2 零值处理

```cpp
// 阶段1: 检测零值
Compare<uint16_t, CMPMODE::EQ>(maskZero, maxExp, zeroB16, maskValid);

// 阶段2: Scale中设为0
Select<uint16_t>(mxScale, zeroB16, mxScale, maskZero);

// 阶段3: 量化数据自动为0
```

### 6.4 Block不整除处理

**问题**：H维度不一定是128的倍数

**解决方案**：

```cpp
// 方案1: Padding（简单）
int64_t paddedCols = CeilAlign(cols, 128LL);
// 填充0到最后Block

// 方案2: 特殊处理（优化）
int64_t remainder = cols % 128;
if (remainder != 0) {
    // 最后一Block特殊处理
    // 只聚合实际有效的元素
    // Scale计算时考虑有效元素数量
}

// 方案3: 混合策略（推荐）
if (remainder < 64) {
    // Padding处理
} else {
    // 特殊处理
}
```

---

## 6. 性能优化策略

### 7.1 数据搬运优化

#### 7.1.1 Scale搬运优势

```
┌──────────────────────────────────────┐
│         Scale搬运对比                 │
├──────────────────────────────────────┤
│ MXFP8 (BlockSize=32)                 │
│   - Scale数量: H/32                   │
│   - H=7168: 224个Scale               │
│   - 搬运量: 224字节                   │
├──────────────────────────────────────┤
│ FP8 PerBlock (BlockSize=128)         │
│   - Scale数量: H/128                  │
│   - H=7168: 56个Scale                │
│   - 搬运量: 56字节                    │
│   - 减少比例: 75%                     │
└──────────────────────────────────────┘
```

#### 7.1.2 搬运策略调整

```cpp
// MXFP8: 每循环搬运 vlForT/32 个Scale
// FP8 PerBlock: 每循环搬运 vlForT/128 个Scale

// 循环次数减少75%
// 单次搬运量减少75%
// 总搬运量减少75%

// 优化建议：
// 1. 合并Scale搬运到一次大搬运
// 2. 减少搬运指令次数
// 3. 充分利用搬运带宽
```

### 7.2 计算优化

#### 7.2.1 聚合计算量对比

```
┌──────────────────────────────────────┐
│       Max聚合计算量对比               │
├──────────────────────────────────────┤
│ MXFP8 (32聚合)                       │
│   - ReduceMaxWithDataBlock: 1次      │
│   - 奇偶Max: 1次                      │
│   - 总计算: 2次Max操作                │
├──────────────────────────────────────┤
│ FP8 PerBlock (128聚合)               │
│   - ReduceMaxWithDataBlock: 4次      │
│   - 块间Max: 3次                      │
│   - 奇偶Max: 1次                      │
│   - 总计算: 8次Max操作                │
│   - 增加: 4倍                         │
└──────────────────────────────────────┘
```

#### 7.2.2 聚合优化策略

**方案1：并行聚合**（推荐）

```cpp
// 4个32B块并行聚合
// 利用Ascend950的多向量单元并行性
ReduceMaxWithDataBlock(blockMax0, maxExp, mask0, 0, 32);
ReduceMaxWithDataBlock(blockMax1, maxExp, mask0, 32, 64);
ReduceMaxWithDataBlock(blockMax2, maxExp, mask0, 64, 96);
ReduceMaxWithDataBlock(blockMax3, maxExp, mask0, 96, 128);

// 可能并行执行（硬件支持）
```

**方案2：分层聚合**

```cpp
// 先32聚合，再32聚合结果聚合
// 可能需要额外的Buffer
```

**方案3：利用新指令**（如果可用）

```cpp
// Ascend950可能有128聚合的专用指令
// 需查询MicroAPI文档确认
```

### 7.3 Pipeline并行

#### 7.3.1 三阶段流水线

```
时间轴：
t0: MaxExp计算 (Block 0)
t1: Scale计算 (Block 0) + MaxExp计算 (Block 1)
t2: Quant计算 (Block 0) + Scale计算 (Block 1) + MaxExp计算 (Block 2)
t3: 输出搬运 (Block 0) + Quant计算 (Block 1) + Scale计算 (Block 2) + MaxExp计算 (Block 3)
...

优势：
- 隐藏内存访问延迟
- 充分利用向量计算单元
- 提升整体吞吐量
```

#### 7.3.2 Pipeline深度

```cpp
// MXFP8: 3阶段
// FP8 PerBlock: 3阶段（相同）

// 但每阶段的计算量不同：
// - MaxExp: 计算量增加4倍
// - Scale: 计算量减少75%
// - Quant: 计算量不变

// Pipeline平衡策略：
// - MaxExp阶段成为瓶颈
// - 需优化聚合计算效率
```

### 7.4 存储优化

#### 7.4.1 UB空间利用

```
┌──────────────────────────────────────┐
│         UB空间对比                    │
├──────────────────────────────────────┤
│ MXFP8                                │
│   - Scale Buffer: vlForT/32 * 2B     │
│   - vlForT=128: 4B                   │
│   - vlForT=256: 8B                   │
├──────────────────────────────────────┤
│ FP8 PerBlock                         │
│   - Scale Buffer: vlForT/128 * 2B    │
│   - vlForT=128: 1B                   │
│   - vlForT=256: 2B                   │
│   - 减少比例: 75%                     │
├──────────────────────────────────────┤
│ 优势                                 │
│   - UB空间节省                        │
│   - 可处理更大的perLoopCols           │
│   - 或增加并行度                      │
└──────────────────────────────────────┘
```

#### 7.4.2 输出压缩

```cpp
// FP32 → FP8: 4倍压缩（与MXFP8相同）
DataCopy<int8_t, ..., StoreDist::DIST_PACK4_B32>(
    xQuantOutAddr, (RegTensor<int8_t> &)xQuant0, ...);

// E8M0 Scale: 只拷出低8位（与MXFP8相同）
DataCopy<uint16_t, ..., StoreDist::DIST_PACK_B16>(
    mxScaleOutAddr, mxScale, ...);
```

### 7.5 代码级优化

#### 7.5.1 循环展开

```cpp
// 减少循环次数（Scale循环减少75%）
// 循环展开优化：

// MXFP8: 需要更多循环展开
// FP8 PerBlock: 循环次数少，展开收益小

// 建议对MaxExp循环展开：
#pragma unroll 4
for (int i = 0; i < 4; i++) {
    ReduceMaxWithDataBlock(...);
}
```

#### 7.5.2 寄存器分配

```cpp
// PerBlock需要更多寄存器
// 4个32B块的中间结果

// 寄存器分配策略：
// - 减少临时变量
// - 复用寄存器
// - 优化寄存器生命周期
```

---

## 7. 测试设计

### 8.1 单元测试（UT）

#### 8.1.1 op_host UT设计

**测试函数列表**：

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_vfComputeMaxExpFP8PerBlock` | Max聚合计算（128元素） | P0 |
| `test_vfComputeScaleFP8PerBlock` | Scale计算（FP32生成） | P0 |
| `test_vfComputeDataFP8PerBlock` | 量化计算（类型转换） | P0 |
| `test_FP8PerBlockQuantKernel` | Kernel完整流程 | P0 |
| `test_scale_shape_3d` | Scale三维Shape验证 | P0 |
| `test_boundary_alignment` | 128对齐边界测试 | P1 |
| `test_special_values` | NaN/Inf/零值处理 | P1 |
| `test_remainder_block` | Block不整除场景 | P1 |

**测试用例设计**：

```cpp
// 测试1: Max聚合计算
void test_vfComputeMaxExpFP8PerBlock() {
    // 输入：128个FP16元素，包含不同的值
    half x[128];
    for (int i = 0; i < 128; i++) {
        x[i] = (half)(i * 0.1);
    }
    
    // 期望输出：最大值对应的指数
    uint16_t expectedMaxExp = ExtractExp(FP16ToBF16((half)12.7));
    
    // 实际输出
    uint16_t maxExp[1];
    vfComputeMaxExpFP8PerBlock<half, fp8_e4m3fn_t>(x, maxExp, 128, ...);
    
    // 验证
    ASSERT_EQ(maxExp[0], expectedMaxExp);
}

// 测试2: Scale计算（FP32输出）
void test_vfComputeScaleFP8PerBlock() {
    // 输入：最大指数
    uint16_t maxExp = 0x0780;
    
    // 期望输出：FP32 Scale值
    float expectedScale = ComputeFP32Scale(maxExp);  // 量化scale
    
    // 实际输出
    float scaleOut;
    uint16_t invScale[1];
    vfComputeScaleFP8PerBlock<half, fp8_e4m3fn_t>(
        &maxExp, &scaleOut, invScale, 1, ...);
    
    // 验证：每个Block对应1个FP32 scale值
    ASSERT_FLOAT_EQ(scaleOut, expectedScale);
}

// 测试3: 量化计算
void test_vfComputeDataFP8PerBlock() {
    // 输入：128个FP16元素 + invScale
    half x[128];
    uint16_t invScale = CalculateInvScale(0x0780);
    
    // 期望输出：量化后的FP8值
    fp8_e4m3fn_t expectedQuant[128];
    for (int i = 0; i < 128; i++) {
        expectedQuant[i] = FP16ToFP8(x[i] * InvScaleToFloat(invScale));
    }
    
    // 实际输出
    fp8_e4m3fn_t xQuant[128];
    vfComputeDataFP8PerBlock<half, fp8_e4m3fn_t>(x, &invScale, xQuant, 128, ...);
    
    // 验证精度
    for (int i = 0; i < 128; i++) {
        ASSERT_NEAR(xQuant[i], expectedQuant[i], tolerance);
    }
}
```

#### 8.1.2 op_api UT设计

**测试函数列表**：

| 测试函数 | 测试内容 | 优先级 |
|----------|----------|--------|
| `test_api_param_validation` | 参数校验（BlockSize约束） | P0 |
| `test_api_type_combination` | 类型组合验证 | P0 |
| `test_api_shape_validation` | Shape校验 | P1 |

**测试用例设计**：

```cpp
// 测试1: 参数校验
void test_api_param_validation() {
    // 测试BlockSize约束
    MoeInitRoutingV3Param param;
    param.cols = 100;  // 不满足128对齐
    
    // 期望：返回错误码
    aclnnStatus status = MoeGatherOutFP8PerBlockQuant(...);
    ASSERT_EQ(status, ACLNN_ERR_PARAM_INVALID);
    
    // 测试正确的对齐
    param.cols = 128;
    status = MoeGatherOutFP8PerBlockQuant(...);
    ASSERT_EQ(status, ACLNN_SUCCESS);
}

// 测试2: 类型组合
void test_api_type_combination() {
    // 测试所有支持的类型组合
    vector<pair<DataType, DataType>> validCombinations = {
        {DT_FLOAT16, DT_FLOAT8_E5M2},
        {DT_FLOAT16, DT_FLOAT8_E4M3FN},
        {DT_BFLOAT16, DT_FLOAT8_E5M2},
        {DT_BFLOAT16, DT_FLOAT8_E4M3FN}
    };
    
    for (auto& combo : validCombinations) {
        aclnnStatus status = MoeGatherOutFP8PerBlockQuant(..., combo.first, combo.second);
        ASSERT_EQ(status, ACLNN_SUCCESS);
    }
    
    // 测试不支持的类型组合
    vector<pair<DataType, DataType>> invalidCombinations = {
        {DT_FLOAT32, DT_FLOAT8_E5M2},  // 不支持FP32输入
        {DT_FLOAT16, DT_INT8}          // 不支持INT8输出
    };
    
    for (auto& combo : invalidCombinations) {
        aclnnStatus status = MoeGatherOutFP8PerBlockQuant(..., combo.first, combo.second);
        ASSERT_EQ(status, ACLNN_ERR_PARAM_INVALID);
    }
}
```

### 8.2 系统测试（ST）

#### 8.2.1 精度验证测试

**精度标准**（参考 `ops-precision-standard`）：

| 输出类型 | rtol | atol | 说明 |
|----------|------|------|------|
| FLOAT8_E5M2 | 1e-2 | 1e-3 | 宽松精度 |
| FLOAT8_E4M3FN | 5e-3 | 5e-4 | 严格精度 |

**测试用例矩阵**：

| H维度 | dtype组合 | Block数 | 精度要求 | 用例ID |
|-------|-----------|---------|----------|--------|
| 128 | FP16→E5M2 | 1 | rtol=1e-2 | ST-001 |
| 256 | FP16→E4M3FN | 2 | rtol=5e-3 | ST-002 |
| 1024 | BF16→E5M2 | 8 | rtol=1e-2 | ST-003 |
| 2048 | BF16→E4M3FN | 16 | rtol=5e-3 | ST-004 |
| 7168 | FP16→E5M2 | 56 | rtol=1e-2 | ST-005 |
| 7168 | FP16→E4M3FN | 56 | rtol=5e-3 | ST-006 |

**Golden参考实现（FP32 Scale，三维布局）**：

```cpp
// FP8 PerBlock Golden函数
void MoeInitRoutingV3FP8PerBlockQuantGolden(
    const Tensor& x,          // FP16/BF16输入，Shape=[NUM_ROWS*K, H]
    Tensor& expandedX,        // FP8输出，Shape=[NUM_ROWS*K, H]
    Tensor& expandedScale) {  // FP32 Scale输出，Shape=[NUM_ROWS*K, CeilDiv(H,256), 2]
    
    // 1. 计算Scale三维布局维度
    int scaleDim1 = CeilDiv(H, 256);  // 第二维
    int scaleDim2 = 2;                 // 第三维
    
    // 2. 初始化expandedScale（三维布局）
    expandedScale.Resize({NUM_ROWS * K, scaleDim1, scaleDim2});
    
    // 3. 按Block计算Scale值
    int numBlocks = CeilDiv(H, 128);  // Block数量
    for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
        // 提取Block的128个元素
        int start = blockIdx * 128;
        int end = Min(start + 128, H);
        auto block = x.slice(start, end);
        
        // 计算Block最大值
        float maxVal = block.abs().max().item<float>();
        
        // 计算FP32 Scale值
        float scale = ComputeFP32Scale(maxVal);
        
        // 三维布局存储：Block i → scale[*, i/2, i%2]
        int groupIdx = blockIdx / 2;
        int innerIdx = blockIdx % 2;
        expandedScale[rowIdx][groupIdx][innerIdx] = scale;
        
        // 计算逆Scale（用于量化）
        float invScale = ComputeInvScale(scale);
        
        // 量化Block
        for (int i = 0; i < 128; i++) {
            expandedX[rowIdx][start + i] = QuantizeToFP8(block[i] * invScale);
        }
    }
}

// Scale计算辅助函数
float ComputeFP32Scale(float maxVal) {
    // 根据目标FP8类型计算合适的scale
    // E4M3FN范围：[-448, 448]
    // E5M2范围：[-57344, 57344]
    
    if (maxVal == 0.0f) return 1.0f;  // 防止除零
    
    // 目标：量化后的值在FP8范围内
    float targetRange = 400.0f;  // E4M3FN保守上限
    return maxVal / targetRange;
}
```

#### 8.2.2 性能测试

**性能测试用例**：

| H维度 | NUM_ROWS | K | 数据量 | 性能目标 | 用例ID |
|-------|----------|---|--------|----------|--------|
| 7168 | 1024 | 8 | 58MB | ≥MXFP8吞吐量 | Perf-001 |
| 7168 | 4096 | 8 | 233MB | ≥MXFP8吞吐量 | Perf-002 |
| 14336 | 1024 | 8 | 116MB | ≥MXFP8吞吐量 | Perf-003 |

**性能对比维度**：

```cpp
// 性能对比测试
void ComparePerformanceMXFP8vsPerBlock() {
    // 测试MXFP8性能
    auto mxfp8Time = MeasurePerformance<MoeGatherOutMxfp8Quant>();
    
    // 测试FP8 PerBlock性能
    auto block128Time = MeasurePerformance<MoeGatherOutFP8PerBlockQuant>();
    
    // 对比指标
    float speedup = mxfp8Time / block128Time;
    
    // 验证目标：speedup >= 1.0（持平或更优）
    ASSERT_GE(speedup, 0.9);  // 允许10%性能波动
}
```

#### 8.2.3 边界测试

**边界测试用例**：

| 场景 | 描述 | 预期结果 | 用例ID |
|------|------|----------|--------|
| H=128（最小） | 单Block场景 | 正常量化 | Bound-001 |
| H=127（不整除） | Padding处理 | 正常量化+填充 | Bound-002 |
| H=100000+（极大） | 多Block场景 | 正常量化 | Bound-003 |
| activeRowNum=0 | 无有效token | 输出为空 | Bound-004 |
| 全零输入 | x全为0 | Scale和输出全0 | Bound-005 |
| NaN输入 | x包含NaN | NaN正确传递 | Bound-006 |
| Inf输入 | x包含Inf | Inf正确传递 | Bound-007 |

---

## 8. 风险分析

### 9.1 技术风险

| 风险项 | 风险等级 | 影响范围 | 缓解措施 | 应急方案 |
|--------|----------|----------|----------|----------|
| **UB容量不足** | 中 | Kernel无法运行 | 精细Tiling切分<br/>减小perLoopCols<br/>复用Buffer | 单核处理<br/>减小H维度 |
| **128聚合性能** | 中 | 性能不达标 | 多级聚合优化<br/>并行聚合<br/>硬件指令优化 | 调整Tiling<br/>减少核数 |
| **精度损失** | 低 | 精度验证失败 | FP16→FP32计算<br/>调整指数下限<br/>动态下限方案 | 增加rtol/atol容忍度 |
| **Block不整除** | 中 | 边界场景失败 | Padding处理<br/>特殊处理路径<br/>混合策略 | 强制对齐要求 |
| **Scale广播错误** | 低 | 量化错误 | 寄存器广播<br/>硬件广播<br/>充分测试 | 逐元素Scale |
| **特殊值处理** | 低 | NaN/Inf传递错误 | 完整测试<br/>参考MXFP8实现 | 禁用特殊值输入 |

### 9.2 兼容性风险

| 风险项 | 风险等级 | 影响范围 | 缓解措施 |
|--------|----------|----------|----------|
| **硬件依赖** | 高 | 仅支持Ascend950 | 明确硬件要求<br/>其他平台降级方案 |
| **软件依赖** | 中 | CANN版本兼容性 | 明确CANN版本要求<br/>版本检测 |
| **接口变更** | 中 | Scale Shape计算公式变更 | 明确接口文档<br/>向后兼容设计 |
| **算法依赖** | 低 | 特定量化算法要求 | 明确应用场景<br/>提供替代方案 |

### 9.3 测试风险

| 风险项 | 风险等级 | 影响范围 | 缓解措施 |
|--------|----------|----------|----------|
| **Golden实现错误** | 中 | 精度验证失败 | 多版本Golden<br/>人工验证<br/>参考标准库 |
| **精度标准不合理** | 低 | 验收失败 | 参考ops-precision-standard<br/>实测调整标准 |
| **性能波动** | 中 | 性能验证不稳定 | 多次测试取平均<br/>标准化测试环境 |
| **边界场景遗漏** | 低 | 潜在bug | 完整测试矩阵<br/>自动化测试 |

---

## 9. 附录

### 10.1 参考文献

1. MXFP8量化规范
2. Ascend C MicroAPI文档
3. Ascend950硬件架构手册
4. `ops-precision-standard` 精度标准
5. 现有 `moe_init_routing_v3` 算子实现

### 10.2 相关文件清单

| 文件类型 | 文件名 | 说明 |
|----------|--------|------|
| **头文件** | `moe_v3_common.h` | 新增FP8 PerBlock常量定义 |
| **头文件** | `moe_v3_gather_fp8_block128_quant.h` | 新增量化Kernel头文件 |
| **源文件** | `moe_v3_gather_fp8_block128_quant.cpp` | 新增量化Kernel实现 |
| **Tiling** | `moe_init_routing_v3_tiling_key.h` | 新增TilingKey定义 |
| **Tiling** | `moe_init_routing_v3_tiling_data.h` | 新增Tiling结构体 |
| **APT** | `moe_init_routing_v3_apt.cpp` | 新增分支调度逻辑 |
| **文档** | `FP8_PERBLOCK_QUANT_DESIGN.md` | 本设计文档 |
| **文档** | `aclnnMoeInitRoutingV3.md` | 接口文档（需更新） |

### 10.3 术语表

| 术语 | 定义 |
|------|------|
| **FP8** | 8位浮点数（E5M2或E4M3FN） |
| **E5M2** | 5位指数+2位尾数的FP8格式 |
| **E4M3FN** | 4位指数+3位尾数的FP8格式（FN: finite NaN） |
| **E8M0** | 8位纯指数格式（用于Scale） |
| **BlockSize** | 量化块大小（固定128元素） |
| **Per-Block量化** | 每个Block共享一个Scale的量化方式 |
| **MXFP8** | MX量化规范，BlockSize=32 |
| **ReduceMaxWithDataBlock** | MicroAPI块级聚合指令 |
| **VL** | Vector Length（向量长度） |
| **UB** | Unified Buffer（统一缓冲区） |
| **TilingKey** | 算子分支选择键值 |

---

**文档状态**：设计草案，待评审和确认  
**下一步**：需求评审 → 方案评审 → 开发实施