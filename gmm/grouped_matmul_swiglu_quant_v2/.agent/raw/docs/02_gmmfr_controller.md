# GMMFR Controller 层代码走读

## 1. 概述

`GMMFRWeightQuantResplitController` 是 GMMFR（GroupedMatmulFinalizeRouting）算子的 Controller 层，负责**多核分块调度**和**按 group 遍历**。它不直接执行矩阵乘计算，而是将每个 group 的 M/N/K 维度切分为多个 BasicBlock，分发给 AIC（Cube 计算核）和 AIV（Vector 计算核）流水线执行。

核心职责：
- 按 groupList 遍历每个 group，获取 M 轴大小
- 将 M 轴在 L1 上分块，N 轴多核并行切分
- 动态选择 K 轴切分大小（256/512），平衡搬运与计算效率
- 跨 group 更新各 GM 地址指针

---

## 2. 类定义分析

### 2.1 模板参数

```cpp
template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType,
          typename perTokenScaleType, typename biasType, typename yType, typename sharedInputDType,
          typename logitsType, typename rowIndexType, bool hasBias, const WqmmConfig &wqmmConfig,
          const VecAntiQuantConfig &vecConfig>
```

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `xType` | 左矩阵 A 的数据类型 | `fp8_e4m3fn_t`（MxA8W4 场景） |
| `wType` | 右矩阵 Weight 的数据类型 | `int4b_t` / `fp4x2_e2m1_t` / `fp4x2_e1m2_t` |
| `antiQuantScaleType` | 反量化 scale 类型 | `float` |
| `scaleType` | 量化 scale 类型 | `int64_t` |
| `perTokenScaleType` | per-token 反量化 scale 类型 | `float` |
| `biasType` | Bias 数据类型 | `float` |
| `yType` | 输出数据类型 | `float` |
| `sharedInputDType` | 共享输入（MoE shared expert）类型 | `bfloat16_t` |
| `logitsType` | MoE logits 类型 | `float` |
| `rowIndexType` | MoE routing 索引类型 | `int64_t` |
| `hasBias` | 是否有 Bias | `true` / `false` |
| `wqmmConfig` | Weight Quant MatMul 配置 | `MXA8W4_NZNK`（NZ 格式，K 轴为主序） |
| `vecConfig` | Vector 反量化配置 | `VEC_ANTIQUANT_CONFIG_DYNAMIC` |

### 2.2 成员变量

```cpp
const GMMFinalizeRoutingWeightQuantTilingData *tiling_;  // Tiling 数据（groupNum, kSize, nSize 等）

// GM 地址指针 —— 跨 group 遍历时动态偏移
__gm__ xType *xGm_;                    // 左矩阵 A 的 GM 地址
__gm__ wType *weightGm_;               // 右矩阵 Weight 的 GM 地址
__gm__ antiQuantScaleType *antiquantScaleGm_;  // 反量化 scale 的 GM 地址
__gm__ biasType *biasGm_;              // Bias 的 GM 地址
__gm__ yType *yGm_;                    // 输出 Y 的 GM 地址
__gm__ perTokenScaleType *perTokenScaleGm_;    // per-token scale 的 GM 地址
__gm__ sharedInputDType *shareInputAddr_;      // 共享输入（MoE shared expert）地址
__gm__ logitsType *logitsAddr_;        // MoE logits 地址
__gm__ rowIndexType *rowIndexAddr_;    // MoE routing row index 地址
GlobalTensor<int64_t> groupListGm_;    // groupList（cumsum 或 count 模式）

GMMFRWeightQuantVcvBasicBlock<...> basicBlock_;  // BasicBlock 实例（Cube + Vector 流水线）

uint64_t preOffset_ = 0;  // cumsum 模式下记录前一个 group 的累积偏移

// 常量定义
static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256;   // K 轴 L1 切分大小（小 M/N 场景）
static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512;   // K 轴 L1 切分大小（大 M/N 场景）
static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;  // N 轴动态阈值
static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256 = 256;  // M 轴动态阈值（无 Bias）
static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 = 240;  // M 轴动态阈值（有 Bias）
static constexpr uint64_t M_L1 = 256;    // M 轴 L1 分块大小
static constexpr uint64_t KB_L1_SIZE = 256;  // K 轴 L1 默认分块大小
static constexpr uint64_t N_L1 = 256;    // N 轴 L1 分块大小
uint64_t mxA8W4L1KDynamicConfigMThreshold_;  // 运行时确定的 M 轴动态阈值
```

### 2.3 Tiling 数据结构

```cpp
struct GMMFinalizeRoutingWeightQuantTilingData {
    uint8_t groupListType;       // 0: cumsum 模式, 1: count 模式
    uint8_t hasBias;             // 是否有 Bias
    uint16_t reserve1;           // 保留
    uint32_t groupNum;           // Group 数量
    uint32_t coreNum;            // 使用的核数
    uint32_t outputBs;           // 输出 batch size
    uint32_t sharedInputOffset;  // 共享输入在输出中的偏移
    uint32_t sharedInputLen;     // 共享输入长度
    float sharedInputWeight;     // 共享输入权重系数
    uint32_t groupSize;          // 反量化 group size
    uint64_t kSize;              // K 轴总大小
    uint64_t nSize;              // N 轴总大小
    uint64_t initSize;           // 输出初始化大小
};
```

---

## 3. Init() 分析

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias,
                            GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR logitsAddr,
                            GM_ADDR rowIndexAddr, GM_ADDR y, GM_ADDR shareInput,
                            const GMMFinalizeRoutingWeightQuantTilingData *__restrict baseTiling)
```

### 3.1 GM 地址初始化

将所有传入的 `GM_ADDR` 强制转换为对应类型的 `__gm__` 指针：

```cpp
xGm_ = reinterpret_cast<__gm__ xType *>(x);
weightGm_ = reinterpret_cast<__gm__ wType *>(weight);
biasGm_ = reinterpret_cast<__gm__ biasType *>(bias);
antiquantScaleGm_ = reinterpret_cast<__gm__ antiQuantScaleType *>(antiquantScale);
perTokenScaleGm_ = reinterpret_cast<__gm__ perTokenScaleType *>(perTokenScale);
logitsAddr_ = reinterpret_cast<__gm__ logitsType *>(logitsAddr);
rowIndexAddr_ = reinterpret_cast<__gm__ rowIndexType *>(rowIndexAddr);
yGm_ = reinterpret_cast<__gm__ yType *>(y);
shareInputAddr_ = reinterpret_cast<__gm__ sharedInputDType *>(shareInput);
```

`groupListGm_` 使用 `SetGlobalBuffer` 绑定到 GlobalTensor：

```cpp
groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList));
```

### 3.2 BasicBlock 初始化

```cpp
basicBlock_.Init(tiling_->groupSize, yGm_, tiling_->sharedInputWeight);
```

`BasicBlock::Init()` 内部完成：
- **AIC 侧**：初始化 Cube 计算模块（`cubeCompute_.MxA8W4Init`），分配 L1 上的 Weight buffer（256×256 双缓冲）和 Bias buffer
- **AIV 侧**：初始化 Vector 计算模块（`vecCompute_.Init`），设置 sharedInputWeight 系数
- **同步**：AIC 通过 `SetAicToAiv<PIPE_MTE1>` 通知 AIV 初始化完成

### 3.3 动态阈值 `mxA8W4L1KDynamicConfigMThreshold_`

```cpp
if constexpr (hasBias) {
    mxA8W4L1KDynamicConfigMThreshold_ = MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240;  // 240
} else {
    mxA8W4L1KDynamicConfigMThreshold_ = MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256;  // 256
}
```

**作用**：在 MxA8W4 场景下，K 轴 L1 切分大小（`kbL1Size`）需要动态选择 256 或 512。当 M 和 N 的 L1 分块都较小时（`mL1Size <= threshold && nL1Size <= 128`），使用 512 以减少 K 轴循环次数；否则使用 256 以保证 L1 空间充足。

有 Bias 时阈值降低到 240，因为 Bias 占用了额外的 L1 空间（约 4KB × 双缓冲），留给 A 矩阵的空间更少。

---

## 4. Process() 分析

### 4.1 整体流程

```
Process()
  │
  ├─ GetBlockIdx() → 获取当前核 ID
  │   └─ AIV 核: blockIdx >> 1（AIC/AIV 配对，共享同一个逻辑 blockIdx）
  │
  ├─ InitOffsetParam() → 初始化 offsetParam[0] 和 [1]
  │   └─ 设置 kSize, nSize, kbL1Size=256, kaL1Size=256, kAlign, nAlign
  │
  ├─ 缓存对齐检查: isCacheLineUnaligned
  │   └─ 8bit: kSize % 128 != 0
  │   └─ 4bit: kSize % 256 != 0
  │
  ├─ basicBlock_.InitAtomicGm() → 初始化输出 GM 为零 + 拷贝 sharedInput
  ├─ SyncAll() → 所有核同步
  │
  └─ for each group (groupIdx = 0 .. groupNum-1):
       │
       ├─ GetSplitValueFromGroupList(groupIdx) → 获取当前 group 的 M 大小
       │
       ├─ M 轴 L1 分块: mBlkNum = CeilDivide(mSize, 256)
       │   └─ mL1Size = CeilDivide(mSize, mBlkNum)  // 尽量均匀分配
       │
       ├─ basicBlock_.UpdateGlobalAddr() → 更新 BasicBlock 的 GM 地址
       │   └─ 传入 weightL2Cacheable 标志（M 需要多轮 L1 加载时启用 L2 cache）
       │
       ├─ 计算 curBasicBlockId（当前核负责的 BasicBlock 起始 ID）
       │
       ├─ for each M 分块 (mOffset += mL1Size):
       │   │
       │   ├─ SplitNByMultiCore() → N 轴多核分块
       │   │   └─ 每个核处理不同的 N 分块
       │   │   └─ 动态选择 kbL1Size (256/512)
       │   │   └─ 计算 kaL1Size（A 矩阵分段全载策略）
       │   │   └─ basicBlock_.ComputeBasicBlock() → 触发 Cube+Vector 流水线
       │   │
       │   └─ basicBlockLimit += N 分块数
       │
       └─ UpdateGmAddr() → 跨 group 更新所有 GM 地址指针
```

### 4.2 GetBlockIdx() 和 AIV/AIC 的关系

```cpp
uint32_t cubeBlockIdx = GetBlockIdx();
if ASCEND_IS_AIV {
    cubeBlockIdx = cubeBlockIdx >> 1;
}
```

**关键设计**：每个逻辑核（cubeBlockIdx）对应一对物理核 —— 一个 AIC（Cube 核）和一个 AIV（Vector 核）。AIC 和 AIV 通过 `GetBlockIdx() >> 1` 映射到同一个逻辑 ID，共享相同的分块调度逻辑。

- AIC 的 `GetBlockIdx()` 返回偶数（0, 2, 4, ...）
- AIV 的 `GetBlockIdx()` 返回奇数（1, 3, 5, ...）
- 右移一位后两者得到相同的 `cubeBlockIdx`（0, 1, 2, ...）

### 4.3 InitOffsetParam()：M/N/K 初始偏移参数

```cpp
offsetParam[0].kbL1Size = KB_L1_SIZE;  // 256
offsetParam[0].kaL1Size = offsetParam[0].kbL1Size;  // 256（A 矩阵切分与 B 矩阵一致）
offsetParam[0].kSize = tiling_->kSize;
offsetParam[0].nSize = tiling_->nSize;
offsetParam[0].kAlign = CeilAlign(tiling_->kSize, BLOCK_CUBE);
offsetParam[1] = offsetParam[0];  // 双缓冲：两个 offsetParam 交替使用
offsetParam[0].nAlign = CeilAlign(tiling_->nSize, BLOCK_CUBE);
offsetParam[1].nAlign = offsetParam[0].nAlign;
```

`BASIC_BLOCK_PROCESS_NUM = 2`，即维护两个 `BasicBlockOffsetParam`，通过 `processId` 在 0 和 1 之间切换（`GetSwitchedProcessId` 返回 `1 - processId`），实现**乒乓双缓冲**：当前 BasicBlock 使用 `offsetParam[processId]`，上一个 BasicBlock 的结果在 `offsetParam[1-processId]` 中等待 Vector 侧完成反量化写出。

### 4.4 M 轴在 L1 上的分块策略

```cpp
uint64_t mBlkNum = CeilDivide(ctrlParam.mSize, M_L1);  // M_L1 = 256
ctrlParam.mL1Size = CeilDivide(ctrlParam.mSize, mBlkNum);
```

**策略**：尽量让每个 M 分块大小均匀。例如 `mSize = 300`：
- `mBlkNum = CeilDivide(300, 256) = 2`
- `mL1Size = CeilDivide(300, 2) = 150`
- 两个分块各 150，而非一个 256 + 一个 44

这样避免了最后一个分块过小导致的 Cube 利用率下降。

### 4.5 isCacheLineUnaligned 缓存对齐检查

```cpp
bool isCacheLineUnaligned = offsetParam[0].kSize % 128 != 0;  // 8bit: 128B 缓存行 = 128 个元素
if constexpr (IsSameType<wType, int4b_t>::value || ...) {
    isCacheLineUnaligned = offsetParam[0].kSize % 256 != 0;  // 4bit: 128B 缓存行 = 256 个元素
}
```

**作用**：当 K 轴大小不是缓存行（128B）的整数倍时，Weight 矩阵的 GM→L1 搬运无法对齐缓存行边界，导致 L2 Cache 效率下降。此时设置 `weightL2Cacheable = true`，让 BasicBlock 在 M 轴需要多轮 L1 加载时启用 L2 Cache 预取，弥补对齐损失。

传入 `UpdateGlobalAddr` 的逻辑：
```cpp
basicBlock_.UpdateGlobalAddr(..., ctrlParam.mL1Size < ctrlParam.mSize || isCacheLineUnaligned);
```
即：**M 轴需要多轮加载 或 K 轴缓存不对齐**时，启用 Weight 的 L2 Cache。

### 4.6 curBasicBlockId 的计算

```cpp
ctrlParam.curBasicBlockId =
    cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + tiling_->coreNum;
```

**含义**：每个 group 的 BasicBlock 按轮次分配给各核。`startBasicBlockId` 是上一 group 结束后剩余未分配的 BasicBlock 起始 ID。如果当前核的 ID 大于等于起始 ID，直接使用；否则加上 `coreNum`（跳到下一轮）。

例如 `coreNum=4`，上一 group 结束后 `startBasicBlockId=2`：
- 核 0 → `curBasicBlockId = 0 + 4 = 4`（下一轮）
- 核 2 → `curBasicBlockId = 2`（当前轮）
- 核 3 → `curBasicBlockId = 3`（当前轮）

---

## 5. SplitNByMultiCore() 分析

### 5.1 N 轴多核分块策略

```cpp
for (; ctrlParam.curBasicBlockId < ctrlParam.basicBlockLimit + basicBlockCount;
     ctrlParam.curBasicBlockId += tiling_->coreNum) {
```

**核心思想**：N 轴被切分为 `basicBlockCount` 个大小为 `basicBlockSize`（256）的分块。每个核按 `coreNum` 的步长跳跃处理不同的 N 分块。

```
N 轴: [0..255] [256..511] [512..767] [768..1023]  (basicBlockCount=4, basicBlockSize=256)
      核0处理    核1处理    核2处理    核3处理      (coreNum=4)
```

每个核的 N 偏移计算：
```cpp
offsetParam[processId].nOffset =
    ctrlParam.nOffset +
    ((ctrlParam.curBasicBlockId - ctrlParam.basicBlockLimit) % basicBlockCount) * basicBlockSize;
```

N 分块大小（处理尾部不对齐）：
```cpp
offsetParam[processId].nL1Size =
    nOffset + basicBlockSize > tiling_->nSize ? tiling_->nSize - nOffset : basicBlockSize;
```

### 5.2 动态 K 选择逻辑：kbL1Size 的 256/512 动态选择

```cpp
offsetParam[processId].kbL1Size =
    (mL1Size <= mxA8W4L1KDynamicConfigMThreshold_ && nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
        MX_A8W4_L1_K_CONFIG_512 : MX_A8W4_L1_K_CONFIG_256;
```

**决策逻辑**：

| 条件 | kbL1Size | 原因 |
|------|----------|------|
| `mL1Size <= 240/256` **且** `nL1Size <= 128` | **512** | M 和 N 都较小，L1 空间充足，增大 K 分块减少循环次数 |
| 其他情况 | **256** | M 或 N 较大，L1 空间紧张，使用较小的 K 分块 |

**为什么是 512 和 256？**
- `kbL1Size = 512`：K 轴每次搬运 512 个元素，减少 K 轴循环次数，降低指令开销
- `kbL1Size = 256`：K 轴每次搬运 256 个元素，L1 占用更小，适合大 M/N 场景

### 5.3 kaL1Size 的计算逻辑（A 矩阵分段全载策略）

```cpp
uint64_t mL1Align = CeilAlign(mL1Size, BLOCK_CUBE);
uint64_t kaDepth = CeilDivide(nL1Size, mL1Align * 2);
```

**核心思想**：A 矩阵通过 ND2NZ 搬运（效率较低），B 矩阵通过 MOVE_ALIGN 搬运（效率较高）。为了平衡两者的搬运效率，使用折算系数 2 简化计算。

- `kaDepth`：A 矩阵在 K 轴上需要分多少段才能覆盖整个 N 轴
- 公式含义：`nL1Size / (mL1Align * 2)` → N 轴大小除以（M 对齐大小 × 2）

**L1 空间约束**：
```cpp
uint64_t aL1Size = hasBias ? 124 * GetKBUnit<xType>() : 128 * GetKBUnit<xType>();
uint64_t maxKaDepth = aL1Size / (mL1Align * kbL1Size);
if (kaDepth > maxKaDepth) {
    kaDepth = maxKaDepth;
}
offsetParam[processId].kaL1Size = kaDepth * kbL1Size;
```

- 有 Bias 时 A 矩阵可用 L1 空间为 124KB，无 Bias 时为 128KB
- `maxKaDepth`：L1 空间能容纳的最大 K 轴分段数
- 最终 `kaL1Size = kaDepth * kbL1Size`：A 矩阵在 K 轴上每次加载的总大小

**策略含义**：A 矩阵采用"分段全载"策略 —— 将 A 矩阵在 K 轴上分为 `kaDepth` 段，每段大小为 `kbL1Size`，一次加载到 L1 后供多轮 N 轴计算复用，减少 A 矩阵的重复搬运。

### 5.4 basicBlock_.ComputeBasicBlock() 的调用

```cpp
basicBlock_.ComputeBasicBlock(offsetParam[processId], offsetParam[GetSwitchedProcessId(ctrlParam)]);
ctrlParam.processId = GetSwitchedProcessId(ctrlParam);  // 切换 0 ↔ 1
```

每次调用传入**当前**和**上一个** BasicBlock 的偏移参数：
- `curOffsetParam`：当前要计算的 BasicBlock
- `lastOffsetParam`：上一个 BasicBlock（用于 Vector 侧完成上一轮的 Y 反量化写出）

`ComputeBasicBlock` 内部按核类型分发：
- **AIC**：`IterateNzNkWithKAic` → 执行 Cube 矩阵乘（WaitAivToAic 同步 → CopyGmToL1 → LaunchMatmul → SetAicToAiv 通知）
- **AIV**：`IterateNzNkWithAiv` → 执行 Weight 反量化 + Y 反量化 + Routing（CopyGmToUb → WeightAntiQuant → SetAivToAic 通知 → MulLogits → RoutingYToGm）

---

## 6. UpdateGmAddr() 分析

跨 group 时更新所有 GM 地址指针，使其指向下一个 group 的数据起始位置。

```cpp
__aicore__ inline void UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize)
{
    xGm_ += mSize * kSize;                                    // ①
    weightGm_ += (nSize * kSize) >> 1;                        // ②
    uint64_t kAlignSize = CeilAlign(kSize, K_ALIGNMENT64);    // ③
    antiquantScaleGm_ += nSize * CeilDivide(kAlignSize, groupSize);  // ④
    perTokenScaleGm_ += mSize * CeilDivide(kAlignSize, groupSize);   // ⑤
    biasGm_ += nSize;                                         // ⑥
    yGm_ += mSize * nSize;                                    // ⑦
    logitsAddr_ += mSize;                                     // ⑧
    rowIndexAddr_ += mSize;                                   // ⑨
}
```

### 6.1 各地址更新详解

| 行号 | 变量 | 偏移量 | 解释 |
|------|------|--------|------|
| ① | `xGm_` | `mSize * kSize` | A 矩阵形状为 `[M, K]`，跳过当前 group 的 M×K 个元素 |
| ② | `weightGm_` | `(nSize * kSize) >> 1` | **关键**：Weight 为 4bit 类型，每个字节存 2 个元素，所以元素数除以 2 |
| ③ | `kAlignSize` | `CeilAlign(kSize, 64)` | K 轴对齐到 64（scale 的 group 对齐要求） |
| ④ | `antiquantScaleGm_` | `nSize * CeilDivide(kAlignSize, groupSize)` | 反量化 scale 形状为 `[N, K/groupSize]`，per-group 量化 |
| ⑤ | `perTokenScaleGm_` | `mSize * CeilDivide(kAlignSize, groupSize)` | per-token scale 形状为 `[M, K/groupSize]` |
| ⑥ | `biasGm_` | `nSize` | Bias 形状为 `[N]`，每个 group 跳过 N 个元素 |
| ⑦ | `yGm_` | `mSize * nSize` | 输出 Y 形状为 `[M, N]` |
| ⑧ | `logitsAddr_` | `mSize` | Logits 形状为 `[M]`，每个 token 一个 logit 值 |
| ⑨ | `rowIndexAddr_` | `mSize` | RowIndex 形状为 `[M]`，每个 token 一个路由索引 |

### 6.2 weightGm_ 的 `>> 1` 原因

Weight 矩阵在 MxA8W4 场景下使用 4bit 存储（`int4b_t` / `fp4x2_e2m1_t` / `fp4x2_e1m2_t`），每个字节打包 2 个 4bit 元素。因此：

```
weightGm_ 偏移（字节） = (nSize * kSize) / 2 = (nSize * kSize) >> 1
```

而 `xGm_` 使用 8bit 类型（`fp8_e4m3fn_t`），每个元素占 1 字节，所以偏移量为 `mSize * kSize`。

---

## 7. GetSplitValueFromGroupList() 分析

```cpp
__aicore__ inline uint64_t GetSplitValueFromGroupList(uint64_t groupIdx)
{
    uint64_t splitValue = 0;
    if (tiling_->groupListType == 0) {
        // cumsum 模式
        uint64_t offset = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        // count 模式
        splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
    }
    return splitValue;
}
```

### 7.1 Cumsum 模式（groupListType == 0）

`groupList` 存储的是**前缀和**（cumulative sum），每个 group 的 M 大小 = 当前前缀和 - 上一个前缀和。

```
groupList = [3, 7, 10]  →  M 大小依次为: 3, 4, 3
```

- `groupIdx=0`: `offset=3`, `splitValue = 3 - 0 = 3`, `preOffset_ = 3`
- `groupIdx=1`: `offset=7`, `splitValue = 7 - 3 = 4`, `preOffset_ = 7`
- `groupIdx=2`: `offset=10`, `splitValue = 10 - 7 = 3`, `preOffset_ = 10`

### 7.2 Count 模式（groupListType == 1）

`groupList` 直接存储每个 group 的 M 大小。

```
groupList = [3, 4, 3]  →  M 大小依次为: 3, 4, 3
```

---

## 8. GetSwitchedProcessId() 分析

```cpp
__aicore__ inline uint64_t GetSwitchedProcessId(const BasicBlockControlParam &ctrlParam)
{
    return 1 - ctrlParam.processId;
}
```

简单的乒乓切换：`0 → 1 → 0 → 1 ...`，用于在两个 `BasicBlockOffsetParam` 之间交替，实现双缓冲流水线。

---

## 9. 完整调用流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    GMMFRWeightQuantResplitController             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Init()                                                          │
│    ├─ 转换所有 GM_ADDR → __gm__ 类型指针                          │
│    ├─ groupListGm_.SetGlobalBuffer()                             │
│    ├─ basicBlock_.Init()                                         │
│    │    ├─ [AIC] cubeCompute_.MxA8W4Init() → L1 buffer 分配      │
│    │    ├─ [AIV] vecCompute_.Init()                              │
│    │    └─ SetAicToAiv 同步                                       │
│    └─ 设置 mxA8W4L1KDynamicConfigMThreshold_ (240/256)           │
│                                                                  │
│  Process()                                                       │
│    ├─ GetBlockIdx() → AIV 右移 1 位                               │
│    ├─ InitOffsetParam() → 初始化 offsetParam[0/1]                │
│    ├─ 缓存对齐检查 → isCacheLineUnaligned                         │
│    ├─ basicBlock_.InitAtomicGm() → 输出清零 + sharedInput 拷贝    │
│    ├─ SyncAll()                                                  │
│    │                                                             │
│    └─ for each group:                                            │
│         ├─ GetSplitValueFromGroupList() → M 大小                  │
│         ├─ M 轴 L1 分块 (mL1Size)                                 │
│         ├─ basicBlock_.UpdateGlobalAddr()                        │
│         │    ├─ [AIC] cubeCompute_.UpdateGlobalAddr()            │
│         │    └─ [AIV] vecCompute_.UpdateGlobalAddr()             │
│         │                                                        │
│         └─ for each M 分块:                                      │
│              └─ SplitNByMultiCore()                              │
│                   ├─ 计算 nOffset, nL1Size                        │
│                   ├─ 动态选择 kbL1Size (256/512)                  │
│                   ├─ 计算 kaL1Size (A 矩阵分段全载)               │
│                   └─ basicBlock_.ComputeBasicBlock()             │
│                        ├─ [AIC] IterateNzNkWithKAic              │
│                        │    ├─ WaitAivToAic (同步)                │
│                        │    ├─ cubeCompute_.GetTensorC (上一轮)   │
│                        │    ├─ SetAicToAiv (通知 AIV)             │
│                        │    └─ for kbL1Offset:                   │
│                        │         ├─ CopyMxScaleGmToL1            │
│                        │         ├─ CopyAAndBiasGmToL1           │
│                        │         ├─ PadL1IfNotAlign              │
│                        │         ├─ WaitAivToAic (同步 Weight)    │
│                        │         ├─ LaunchMatmul                 │
│                        │         └─ SetAicToAiv (通知 AIV)        │
│                        │                                         │
│                        └─ [AIV] IterateNzNkWithAiv              │
│                             ├─ VecComputeNzNkWithStartLimit      │
│                             │    └─ for kMte2Offset:             │
│                             │         ├─ CopyGmToUb (Weight+Bias)│
│                             │         ├─ WaitAicToAiv (同步)      │
│                             │         ├─ WeightAntiQuantCompute  │
│                             │         └─ SetAivToAic (通知 AIC)   │
│                             ├─ CopyRowIndexLogitsGmToUb          │
│                             ├─ SetAivToAic (通知 AIC 取 TensorC) │
│                             ├─ WaitAicToAiv (等待 TensorC)        │
│                             ├─ MulLogits (Y × logits)            │
│                             └─ RoutingYToGm (scatter add 写出)   │
│                                                                  │
│         └─ UpdateGmAddr() → 所有 GM 指针偏移到下一 group          │
│                                                                  │
│    └─ basicBlock_.End() → 处理最后一个 BasicBlock 的尾量          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 关键设计要点总结

| 设计要点 | 说明 |
|----------|------|
| **AIC/AIV 配对** | 每个逻辑核对应一对 AIC+AIV，通过 `GetBlockIdx() >> 1` 映射到同一逻辑 ID |
| **双缓冲乒乓** | `BASIC_BLOCK_PROCESS_NUM = 2`，两个 offsetParam 交替使用，实现 Cube 和 Vector 的流水线并行 |
| **M 轴均匀分块** | `mL1Size = CeilDivide(mSize, mBlkNum)`，避免尾部小块导致的利用率下降 |
| **N 轴多核并行** | 每个核按 `coreNum` 步长跳跃处理不同的 N 分块，实现负载均衡 |
| **K 轴动态选择** | 根据 M/N 的 L1 分块大小动态选择 kbL1Size=256 或 512，平衡 L1 空间和循环效率 |
| **A 矩阵分段全载** | kaL1Size 将 A 矩阵在 K 轴分段，一次加载后供多轮 N 轴计算复用 |
| **L2 Cache 优化** | 当 M 轴多轮加载或 K 轴缓存不对齐时，启用 Weight 的 L2 Cache 预取 |
| **4bit Weight 偏移** | `weightGm_ += (nSize * kSize) >> 1`，因为 4bit 元素每字节存 2 个 |
| **Group 遍历** | 通过 groupList（cumsum 或 count 模式）获取每个 group 的 M 大小，跨 group 更新所有 GM 指针 |