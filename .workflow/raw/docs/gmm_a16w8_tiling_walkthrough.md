# GMM A16W8 Tiling 走读文档

> 走读对象：`grouped_weight_quant_batch_matmul_tiling.{h,cpp}` + `grouped_matmul_tiling_data_apt.h`
> 目标：理解 A16W8 多多多的 tiling 计算流程，为 A16W4 pergroup ND 适配提供依据。

---

## 1. 代码结构概览

### 1.1 文件关系

```
grouped_weight_quant_batch_matmul_tiling.h   ← 类定义、枚举、常量
grouped_weight_quant_batch_matmul_tiling.cpp ← 实现逻辑
grouped_matmul_tiling_data_apt.h             ← tiling 数据结构（kernel 侧读取）
```

### 1.2 核心类

| 类/结构 | 职责 |
|---------|------|
| `GroupedWeightQuantBatchMatmulTiling` | tiling 主类，负责参数校验、tiling 计算、结果写入 |
| `TilingKeyConfigure` | tiling key 配置，用于模板分发 |
| `GMMWeightQuantParam` | tiling 数据结构（传递到 kernel） |
| `GMMWeightQuantTilingData` | 完整 tiling 数据包（= GMMWeightQuantParam + GMMArray + TCubeTiling） |
| `TailBlockResplitParam` | N 轴尾块重切分参数 |

### 1.3 关键枚举

```cpp
enum class GroupType : int64_t {
    NO_SPLIT = -1,   // 多多多
    SPLIT_M = 0,     // 单单单（M轴分组）
    SPLIT_N = 1,
    SPLIT_K = 2,
};

enum class QuantType : uint8_t {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4
};

enum class WeightFormat : uint8_t {
    ND = 0,
    FRACTAL_NZ = 1,
};
```

---

## 2. GMMWeightQuantParam 结构详解

```cpp
// grouped_matmul_tiling_data_apt.h
struct GMMWeightQuantParam {
    uint32_t groupNum = 0;           // matmul 分组数
    uint32_t coreNum = 0;            // 使用的 AIC 核数
    uint64_t kSize = 0;              // K 轴大小
    uint64_t nSize = 0;              // N 轴大小（原始值，非 NZ 对齐后）
    uint8_t singleX = 0;             // x 是否为单 tensor
    uint8_t singleWeight = 0;        // weight 是否为单 tensor
    uint8_t singleY = 0;             // y 是否为单 tensor
    int8_t groupType = 0;            // -1=多多多, 0=单单单
    uint8_t groupListType = 0;       // 0=累积式, 1=增量式
    uint8_t hasBias = 0;             // 是否有 bias
    uint8_t cubeNumBlocksN = 0;      // N 方向 cube 基本块数（用于尾块重切分）
    uint8_t reserved = 0;
    uint32_t groupSize = 0;          // ★ 反量化 groupSize（pergroup 场景关键参数）
    uint32_t mainBlockSize = 0;      // 主块大小（尾块重切分用）
    uint64_t mainBlockCount = 0;     // 主块数量
    uint16_t firstTailBlockSize = 0; // 第一类尾块大小
    uint16_t secondTailBlockSize = 0;// 第二类尾块大小
    uint16_t firstTailBlockCount = 0;// 第一类尾块数量
    uint16_t secondTailBlockCount = 0;// 第二类尾块数量
};
```

### 2.1 groupSize 字段

- **当前 A16W8 场景**：`groupSize` 默认为 0（perchannel 场景不使用此字段）
- **A16W4 pergroup 场景**：`groupSize` 需要设置为 32/64/128/256
- **设置逻辑**：在 `SetAntiquantGroupSize()` 中根据 `antiquantScale` 的 shape 推导

### 2.2 GMMArray 结构

```cpp
struct GMMArray {
    int32_t mList[128] = {0};  // 每组的 M 值
    int32_t kList[128] = {0};  // 每组的 K 值
    int32_t nList[128] = {0};  // 每组的 N 值
};
```

### 2.3 完整 Tiling 数据包

```cpp
struct GMMWeightQuantTilingData {
    GMMWeightQuantParam gmmWeightQuantParam;  // 自定义参数
    GMMArray gmmArray;                         // MKN 列表
    TCubeTiling mmTilingData;                  // matmul tiling 参数
};
```

---

## 3. Tiling 主流程

### 3.1 SetTiling() 入口

```
SetTiling(context)
  ├── AnalyzeAttr(context)        // 解析属性 + 参数校验
  │     ├── 读取 transA/transB/groupType/splitItem/groupListType
  │     ├── 判断 isSingleX/isSingleWeight/isSingleY
  │     ├── GetNumOfInputs()      // 获取各输入 tensor 数量
  │     ├── AnalyzeInput()        // 获取 dtype/format
  │     ├── CheckCoreNum()        // 校验 aicNum/aivNum 比例（1:2）
  │     ├── CheckUnsupportDataFlow()  // 校验数据流合法性
  │     ├── CheckTransposeStatus()    // A16W4ND 不支持 transA
  │     ├── CheckGroupTypeAndSplitItem()  // groupType=-1/0, splitItem=0/1/2/3
  │     ├── CheckTensorListSize()   // tensor list 大小校验
  │     ├── CheckEmptyTensor()     // 空 tensor 校验
  │     ├── CheckAntiQuantDtype()  // antiquantScale dtype 校验
  │     ├── CheckBiasDtype()       // bias dtype 校验
  │     ├── CheckGroupList()       // groupList 校验
  │     ├── CheckEveryTensor()     // 逐 tensor 全量校验
  │     ├── SetShapeList()         // ★ 填充 mList/kList/nList
  │     ├── SetAntiquantGroupSize()// ★ 设置 groupSize
  │     └── CheckPerTokenScale()   // MX 场景 perTokenScale 校验
  ├── CalcResplitTiling(context)  // N 轴尾块重切分计算
  ├── SetBaseTiling()             // ★ 填充 GMMWeightQuantParam
  ├── SetMatMulTiling()           // ★ 填充 TCubeTiling
  ├── SetTilingKey(context)       // ★ 设置 tiling key（模板分发）
  ├── SetCustomParam(context)     // 写入 workspace/blockDim/rawTilingData
  └── PrintTilingResult(context)  // 日志输出
```

---

## 4. 关键函数详解

### 4.1 SetShapeList() — MKN 列表填充

根据 `singleX/singleWeight/singleY` 分发到两条路径：

#### 4.1.1 单单单路径：`SetShapeListSplitMSingleXSingleWeightSingleY()`

```cpp
// groupType == 0 (SPLIT_M)
groupNum_ = wShape.GetDim(0);  // weight 第0维 = group 数
mSize_ = xShape.GetDim(0);     // x 第0维 = M（transA=false）
kSize_ = xShape.GetDim(xDimNum - 1);  // x 最后一维 = K
nSize_ = wShape.GetDim(wDimNum - 1);  // weight 最后一维 = N（transB=false）

// int4 weight 特殊处理：
if (weightDtype_ == DT_FLOAT || weightDtype_ == DT_INT32) {
    weightDtype_ = DT_INT4;  // float32/int32 打包表示 int4
    if (!transB_) nSize_ = 8 * nSize_;  // N 轴扩大 8 倍
}

// 填充列表
kList_[0] = kSize_;
nList_[0] = nSizeOri_;
mList_[0] = -1;  // 单单单场景 mList 不使用
```

#### 4.1.2 多多多路径：`SetShapeListMultiXMultiWeightMultiY()`

```cpp
// groupType == -1 (NO_SPLIT)
for (i = 0; i < MAX_TENSOR_CONT; i++) {
    m = xShape.GetDim(xDimNum - 2);  // batch 轴累乘
    for (xDim = 0; xDim < xDimNum - 2; xDim++) m *= xShape.GetDim(xDim);
    k = transB_ ? wShape.GetDim(1) : wShape.GetDim(0);
    n = transB_ ? wShape.GetDim(0) : wShape.GetDim(1);
    mList_[i] = m;
    kList_[i] = k;
    nList_[i] = n;
    mSize_ = max(mSize_, m);  // 取最大值用于 tiling 参数
    kSize_ = max(kSize_, k);
    nSize_ = max(nSize_, n);
}
```

### 4.2 SetAntiquantGroupSize() — groupSize 设置

```cpp
// antiquantScale shape 维度判断
if (antiquantScaleDimNum == 3) {
    // 3维: (g, K, N) 或 (g, N, K) → pergroup 场景
    groupNum = transB_ ? shape.GetDim(dim-1) : shape.GetDim(dim-2);
    groupSize_ = kSize_ / groupNum;  // ★ 核心公式
} else if (antiquantScaleDimNum == 4) {
    // 4维: MX 格式 (g, n, k/64, 2) 或 (g, k/64, n, 2)
    groupNum = ... * MX_GROUP_FACTOR;
    groupSize_ = kSize_ / groupNum;
}
// 若 antiquantScaleDimNum == 2 → perchannel，groupSize_ 保持 0
```

**⚠️ A16W4 pergroup 适配点**：
- 当前 A16W8 perchannel 场景 `groupSize_` = 0
- A16W4 pergroup 场景需要正确设置 `groupSize_` = 32/64/128/256
- `SetAntiquantGroupSize()` 中 3 维分支已能处理 pergroup，但需确认 antiquantScale shape 为 `(g, k/gs, n)` 时的维度解析是否正确
- `CheckGroupSize()` 当前仅校验 S8S4 场景（128/192/256/512），**A16W4 pergroup 需要新增 32/64 的校验**

### 4.3 SetBaseTiling() — 填充 GMMWeightQuantParam

```cpp
tilingData_.gmmWeightQuantParam.groupNum = groupNum_;
tilingData_.gmmWeightQuantParam.coreNum = coreNum_;
tilingData_.gmmWeightQuantParam.kSize = kSize_;
tilingData_.gmmWeightQuantParam.nSize = nSizeOri_;
tilingData_.gmmWeightQuantParam.singleX = isSingleX_;
tilingData_.gmmWeightQuantParam.singleWeight = isSingleWeight_;
tilingData_.gmmWeightQuantParam.singleY = isSingleY_;
tilingData_.gmmWeightQuantParam.groupType = groupType_;
tilingData_.gmmWeightQuantParam.groupListType = groupListType_;
tilingData_.gmmWeightQuantParam.hasBias = hasBias_;
tilingData_.gmmWeightQuantParam.cubeNumBlocksN = cubeNumBlocksN_;
tilingData_.gmmWeightQuantParam.groupSize = groupSize_;  // ★
// + resplit 参数 + mList/kList/nList memcpy
```

### 4.4 SetMatMulTiling() — TCubeTiling 参数

```cpp
// 基础参数（A16W8 默认）
baseM = 256; singleCoreM = 256;
baseN = 256; singleCoreN = 256;
baseK = 64;  singleCoreK = kSize_;
stepKa = 4;  stepKb = 4;
depthA1 = 8; depthB1 = 8;
dbL0A = 2;   dbL0B = 2;   dbL0C = 1;

// A16W8 有 bias 时
if (hasBias_) {
    baseM = 240; singleCoreM = 240;  // 256→240 留出 bias 空间
}

// S8S4 场景
if (xDType == INT8 && weightDtype == INT4) {
    baseK = 128;  // 64*2
    isBias = 0;   // A8W4 在 UB 中处理 bias
}
```

**⚠️ A16W4 pergroup 适配点**：
- baseM/baseN/baseK 可复用 A16W8 默认值（256/256/64）
- 需确认 A16W4 pergroup 场景下 `stepKb` 是否需要调整（与 `ubMte2InnerSize` 相关）

### 4.5 SetTilingKey() — 模板分发

```cpp
tilingKeyConfig_.socVersionType = SUPPORT_L1_TO_BT_BF16 * 10;  // 平台类型
tilingKeyConfig_.algorithm = VECTOR_ANTIQUANT * 10 + N_FIRST_BASIC_BLOCK;  // 或 N_FIRST_TAIL_RESPLIT
tilingKeyConfig_.transposeSituation = (transA << 1) | transB;
tilingKeyConfig_.antiquantType = PER_CHANNEL;  // ★ 当前 A16W8 固定为 PER_CHANNEL
tilingKeyConfig_.weightFormat = ND;            // A16W8 固定 ND
tilingKeyConfig_.templateCustom = MTE2_INNER_SIZE_256_BUF_NUM_4;  // 默认 MTE2 配置
```

**⚠️ A16W4 pergroup 适配点**：
- `antiquantType` 需要设置为 `PER_GROUP`（当前固定为 `PER_CHANNEL`）
- 可能需要新增 `Mte2Configuration` 枚举值或复用现有配置
- tiling key 的分发将影响 kernel 侧模板实例化路径

### 4.6 CalcResplitTiling() — N 轴尾块重切分

```
EnableTailResplit() 判断：
  - 多多多 → 不使能
  - 单单单 + weight ND + B非转置 → 不使能
  - 其余 → 使能

A16W4 pergroup ND + B非转置 → 不使能尾块重切分（与 A16W8 ND 一致）
```

---

## 5. IsA16W4ND() 判断逻辑

```cpp
bool IsA16W4ND() const {
    return ge::GetSizeByDataType(xDType_) == 2 &&  // x 为 16bit (fp16/bf16)
           (weightDtype_ == DT_INT4 || weightDtype_ == DT_INT32) &&
           !weightNzFlag_;  // ND 格式
}
```

此函数在 tiling 中被多处调用来区分 A16W4ND 场景的特殊处理：
- `CheckTensorDtype()` — 强制 dtype 一致性
- `CheckXAndWeightFormat()` — 强制 weight ND
- `CheckTensorDimSingleXSingleWeightSingleY()` — antiquantScale 维度校验
- `CheckTensorShape()` — shape 校验
- `CheckDimValue()` — weight N/K 维度校验
- `CheckGroupTypeAndSplitItem()` — groupType 限制为 -1/0
- `CheckAntiQuantDtype()` — antiquantScale dtype 必须与 x 一致
- `CheckBiasDtype()` — bias dtype 约束

---

## 6. Tiling 参数传递到 Kernel

```
Tiling 侧：
  context->GetRawTilingData()->GetData() ← memcpy(&tilingData_, sizeof(tilingData_))
  context->SetBlockDim(coreNum_)

Kernel 侧：
  GMMWeightQuantBasicController::Init(
      ..., baseTiling, mmTiling, tiling, gmmArrayAddrIn, ...)
  其中：
    baseTiling = GMMWeightQuantParam*  ← 从 tiling 数据中解析
    mmTiling = TCubeTiling*            ← 从 tiling 数据中解析
    gmmArrayAddrIn = TILING_TYPE*       ← mList/kList/nList 地址
```

---

## 7. A16W4 pergroup ND 适配清单

### 7.1 需要修改的代码

| 位置 | 修改内容 | 原因 |
|------|----------|------|
| `SetAntiquantGroupSize()` | 确认 3 维分支对 `(g, k/gs, n)` shape 的解析 | pergroup 的 antiquantScale shape 为 3 维 |
| `CheckGroupSize()` | 新增 A16W4 pergroup 的 groupSize 校验（32/64/128/256） | 当前仅校验 S8S4 场景 |
| `SetTilingKey()` | `antiquantType` 设为 `PER_GROUP` | 当前固定为 `PER_CHANNEL` |
| `SetTilingKey()` | 确认 `templateCustom` 配置 | 可能需要不同的 MTE2 配置 |
| `CheckTensorDimSingleXSingleWeightSingleY()` | antiquantScale 维度校验适配 | pergroup 为 3 维 `(g, k/gs, n)` vs perchannel 为 2 维 `(g, n)` |
| `CheckTensorShape()` | antiquantScale N 轴校验适配 | pergroup shape 为 `(g, k/gs, n)` |

### 7.2 可复用的代码

| 位置 | 说明 |
|------|------|
| `SetShapeList()` | 单单单/多多多的 MKN 填充逻辑完全可复用 |
| `SetBaseTiling()` | GMMWeightQuantParam 填充逻辑可复用（groupSize 字段已有） |
| `SetMatMulTiling()` | baseM/baseN/baseK 等参数可复用 |
| `CalcResplitTiling()` | A16W4 ND 非转置场景不使能，无需修改 |
| `SetCustomParam()` | workspace/blockDim 设置可复用 |
| `AnalyzeAttr()` 主流程 | 校验框架可复用，仅需在分支中增加 A16W4 pergroup 处理 |

### 7.3 需要新增的代码

| 内容 | 说明 |
|------|------|
| A16W4 pergroup 的 `IsA16W4NDPergroup()` 判断函数 | 区分 perchannel 和 pergroup |
| pergroup 场景的 antiquantScale/antiquantOffset shape 校验 | 3 维 `(g, k/gs, n)` |
| pergroup groupSize 合法性校验 | 32/64/128/256，且 K % groupSize == 0 |

---

## 8. 常量速查表

| 常量 | 值 | 用途 |
|------|-----|------|
| `BASIC_BLOCK_BASE_M` | 256 | baseM 默认值 |
| `BASIC_BLOCK_BASE_M_WITH_BIAS` | 240 | 有 bias 时的 baseM |
| `BASIC_BLOCK_BASE_N` | 256 | baseN 默认值 |
| `BASIC_BLOCK_BASE_K` | 64 | baseK 默认值 |
| `BASIC_BLOCK_BASE_N_MIN` | 128 | 尾块最小 N |
| `STEP_K_4` | 4 | stepK 默认值 |
| `DEPTH_8` | 8 | L0 depth |
| `BUFFER_NUM_2` | 2 | 双缓冲 |
| `AIC_AIV_CORE_RATIO` | 2 | AIC:AIV = 1:2 |
| `MAX_TENSOR_CONT` | 128 | 最大 tensor 数 |
| `MX_GROUP_SIZE` | 32 | MX 量化 group size |
