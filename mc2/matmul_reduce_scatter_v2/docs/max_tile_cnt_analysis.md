# maxTileCnt=16 与 AdjustHCCLLimit 函数冲突分析报告

## 1. maxTileCnt=16 的含义解析

### 1.1 定义位置

**代码位置**：`mc2/common/op_host/op_tiling/hccl_formulaic_tiling.h` 第 38 行

```cpp
constexpr uint64_t MAX_TILE_CNT = 16;
```

**使用位置**：`hccl_formulaic_tiling.h` 第 131 行

```cpp
explicit OneCalcOneCommBase(...) {
    ...
    tilingM_.SetMaxTileCnt(MAX_TILE_CNT);  // 受限ffts队列大小，最多切16轮
}
```

### 1.2 具体含义

**关键结论**：`maxTileCnt=16` 是指 **totalTileCnt（总切分块数）的最大值**，而不是单独指 tileCnt 或 tailCnt。

#### 数据结构分析

**CutResult 结构**（`formulaic_tiling_datatype.h` 第 140-147 行）：
```cpp
struct CutResult {
    bool shortTileAtBack;
    uint64_t totalTileCnt;    // 总切分块数 = numLongTile + numShortTile
    uint64_t numLongTile;     // 长块（主块）数量
    uint64_t numShortTile;    // 短块（尾块）数量，值为 0 或 1
    uint64_t longTileLen;     // 长块长度
    uint64_t shortTileLen;    // 短块长度
};
```

**数学关系**：
```
totalTileCnt = numLongTile + numShortTile
             = tileCnt + tailCnt (在 RCSTiling 中的对应关系)
```

**RCSTiling 结构**（`mc2_tiling_struct.h` 第 105-134 行）：
```cpp
struct RCSTiling {
    uint32_t tileCnt;   // 主块数量（对应 numLongTile）
    uint32_t tailCnt;   // 尾块数量（对应 numShortTile，值为 0 或 1）
    uint32_t tailM;     // 尾块的 M 轴长度
    ...
};
```

#### 约束条件验证

**MaxTileCntUniform 函数**（`hccl_formulaic_tiling.cpp` 第 317-333 行）：
```cpp
void FormPartition::MaxTileCntUniform() {
    // 检查是否超过限制
    if (cutRes.totalTileCnt <= tileArgs.maxTileCnt) {
        return;  // 未超限，直接返回
    }
    
    // 超限处理
    if (tileArgs.maxTileCnt <= 1 || totalLen < tileArgs.maxTileCnt) {
        NoCutTiling();  // 无法切分，不切
        return;
    }

    // 强制切分为 maxTileCnt - 1 块长块 + 1 块尾块
    cutRes.longTileLen = totalLen / (tileArgs.maxTileCnt - 1);
    cutRes.longTileLen = (cutRes.longTileLen + tileArgs.mAlignLen - 1) /
                         tileArgs.mAlignLen * tileArgs.mAlignLen;
    cutRes.longTileLen = std::min(cutRes.longTileLen, totalLen);
    cutRes.numLongTile = totalLen / cutRes.longTileLen;
    CalcBackTile();  // 计算尾块，最终 totalTileCnt = numLongTile + numShortTile ≤ maxTileCnt
}
```

**验证结论**：
- `cutRes.totalTileCnt <= tileArgs.maxTileCnt`：检查的是**总切分块数**
- 强制切分公式 `maxTileCnt - 1 + 1`：确保总数不超过 16

### 1.3 限制原因

**注释说明**（`hccl_formulaic_tiling.h` 第 131 行）：
```cpp
tilingM_.SetMaxTileCnt(MAX_TILE_CNT);  // 受限ffts队列大小，最多切16轮
```

**技术背景**：
- **FFTS（Fast Fourier Transform Scheduler）队列**：硬件任务调度队列
- **通信轮次限制**：每个通信轮次需要占用一个 FFTS 队列项
- **队列容量**：FFTS 队列最大支持 16 个并发任务

---

## 2. AdjustHCCLLimit 函数逻辑分析

### 2.1 函数定义

**代码位置**：`matmul_reduce_scatter_tiling_base.cpp` 第 127-157 行

```cpp
ge::graphStatus MatmulReduceScatterTilingBase::AdjustHCCLLimit(
    Mc2Tiling::RCSTiling &rcsCfg, mc2tiling::Mc2QuantMode quantMmMode)
{
    // 1. 检查是否满足 256MB 限制
    if (tileMValue_ * args_.nValue * ge::GetSizeByDataType(args_.geCType) <= 
        mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT) {
        return ge::GRAPH_SUCCESS;  // 满足限制，无需调整
    }
    
    // 2. 记录原始切分信息
    OPS_LOG_I(opName_, "The result of formulaic tiling result does not meet the hccl restriction,"
             " current splitting: tileM [%ld], tileCnt [%ld], tailM [%ld], tailCnt [%ld].",
        tileMValue_, rcsCfg.tileCnt, tailMValue_, rcsCfg.tailCnt);

    // 3. 检查量化模式限制
    OP_TILING_CHECK((quantMmMode == mc2tiling::Mc2QuantMode::PERBLOCK_MODE),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "quantMmMode",
            std::to_string(static_cast<int>(quantMmMode)).c_str(), 
            "The value of quantMmMode must not be PERBLOCK_MODE when size exceeds 256MB"),
        return ge::GRAPH_FAILED);

    // 4. 重新切分以满足 256MB 限制
    uint64_t minSplitPart = Ops::Base::CeilDiv(
        args_.mValue * args_.nValue * ge::GetSizeByDataType(args_.geCType),
        mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT);
    
    tileMValue_ = Ops::Base::CeilDiv(args_.mValue, minSplitPart);
    rcsCfg.tileCnt = Ops::Base::FloorDiv(args_.mValue, tileMValue_);
    rcsCfg.tailM = args_.mValue - rcsCfg.tileCnt * tileMValue_;
    tailMValue_ = rcsCfg.tailM;
    
    if (tailMValue_ == 0) {
        rcsCfg.tailCnt = 0;
    } else {
        rcsCfg.tailCnt = 1;
    }
    
    longTileLen_ = tileMValue_;
    
    // 5. 记录重新切分结果
    OPS_LOG_I(opName_, "Because the result of formulaic tiling result does not meet the hccl restriction,"
             " the re-splitM result: tileM [%ld], tileCnt [%ld], tailM [%ld], tailCnt [%ld]. end re-splitM.",
        tileMValue_, rcsCfg.tileCnt, tailMValue_, rcsCfg.tailCnt);
    
    return ge::GRAPH_SUCCESS;
}
```

### 2.2 核心逻辑解析

#### 输入参数
- `rcsCfg`：RCSTiling 结构体，包含切分结果
- `quantMmMode`：量化模式（PERBLOCK_MODE 不支持超过 256MB）

#### 返回值
- `ge::GRAPH_SUCCESS`：成功调整
- `ge::GRAPH_FAILED`：调整失败（量化模式不支持）

#### 核心逻辑步骤

**步骤1：检查 256MB 限制**
```cpp
单块数据量 = tileMValue_ * nValue * dtypeSize
限制条件：单块数据量 ≤ 256MB
```

**步骤2：计算最小切分份数**
```cpp
minSplitPart = 总数据量 / 256MB
             = (mValue * nValue * dtypeSize) / 256MB
             = ⌈总数据量 / 256MB⌉
```

**步骤3：重新计算切分**
```cpp
tileMValue_ = ⌈mValue / minSplitPart⌉  // 每块的 M 轴长度
tileCnt = ⌊mValue / tileMValue_⌋       // 主块数量
tailM = mValue - tileCnt * tileMValue_  // 尾块的 M 轴长度
tailCnt = (tailM == 0) ? 0 : 1          // 尾块数量
```

### 2.3 为什么要调整限制

#### HCCL 通信约束（`CheckHCCLSize` 函数，第 111-125 行）

**约束1：单块数据量不超过 256MB**
```cpp
uint64_t sizeOfSplitM = Ops::Base::CeilDiv(args_.mValue, mc2tiling::MC2_HCCL_NUM_LIMIT) * 
                        sizeOfSingleM;
OP_TILING_CHECK(sizeOfSplitM > mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT,
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x1size", ...,
        "The value of x1size must not exceed 256MB after splitting M into 16 parts"),
    return ge::GRAPH_FAILED);
```

**约束2：最大通信次数限制**
```cpp
constexpr static uint64_t MC2_HCCL_NUM_LIMIT = 63;  // 最大通信次数
constexpr static uint64_t ALL_GATHER_HCCL_NUM_LIMIT = 16;  // 另一个相关限制
```

#### 硬件资源限制

**HCCL 通信句柄限制**：
- 单次通信任务数据量上限：256MB
- 原因：HCCL 内部缓冲区大小限制

**FFTS 队列限制**：
- 最大并发通信任务：16
- 原因：硬件调度队列容量

---

## 3. 冲突分析

### 3.1 限制范围对比

| 维度 | maxTileCnt=16 | AdjustHCCLLimit | 关系 |
|------|---------------|-----------------|------|
| **限制对象** | totalTileCnt（总切分块数） | 单块数据量 | 不同维度 |
| **限制值** | ≤ 16 | ≤ 256MB | 不同单位 |
| **触发条件** | 公式化切分后检查 | 单块超 256MB 时触发 | 不同时机 |
| **调整方式** | 强制切分到 16 块 | 重新切分满足 256MB | 不同策略 |

### 3.2 是否存在冲突

**结论：不存在直接冲突，但存在约束叠加效应**

#### 理论分析

**公式化切分流程**：
```
1. DoFormulaticTiling() → 生成初始切分（可能超过 16 块）
2. MaxTileCntUniform() → 强制限制到 16 块
3. AdjustHCCLLimit() → 检查单块是否超过 256MB，必要时重新切分
```

**约束叠加**：
- **第一层约束**：totalTileCnt ≤ 16（FFTS 队列限制）
- **第二层约束**：单块数据量 ≤ 256MB（HCCL 缓冲区限制）

#### 数学验证

**场景1：数据量较小**
```
假设：
  mValue = 4096, nValue = 1024, dtypeSize = 2 (FP16)
  总数据量 = 4096 * 1024 * 2 = 8MB

公式化切分结果：
  totalTileCnt = 8（可能）
  tileM = 512, 单块数据量 = 512 * 1024 * 2 = 1MB

验证：
  totalTileCnt = 8 ≤ 16 ✓
  单块数据量 = 1MB ≤ 256MB ✓
  
结论：无需调整，两者都满足
```

**场景2：数据量中等**
```
假设：
  mValue = 16384, nValue = 2048, dtypeSize = 2 (FP16)
  总数据量 = 16384 * 2048 * 2 = 64MB

公式化切分结果：
  totalTileCnt = 16（可能）
  tileM = 1024, 单块数据量 = 1024 * 2048 * 2 = 4MB

验证：
  totalTileCnt = 16 ≤ 16 ✓
  单块数据量 = 4MB ≤ 256MB ✓
  
结论：满足限制，无需调整
```

**场景3：数据量极大**
```
假设：
  mValue = 65536, nValue = 4096, dtypeSize = 2 (FP16)
  总数据量 = 65536 * 4096 * 2 = 512MB

公式化切分初始结果：
  totalTileCnt = 32（可能）
  
MaxTileCntUniform() 强制调整：
  totalTileCnt = 16
  tileM = 4096, 单块数据量 = 4096 * 4096 * 2 = 32MB

AdjustHCCLLimit() 检查：
  单块数据量 = 32MB ≤ 256MB ✓
  
结论：两层约束都满足
```

**场景4：极端数据量**
```
假设：
  mValue = 131072, nValue = 4096, dtypeSize = 2 (FP16)
  总数据量 = 131072 * 4096 * 2 = 1GB

公式化切分初始结果：
  totalTileCnt = 64（可能）
  
MaxTileCntUniform() 强制调整：
  totalTileCnt = 16
  tileM = 8192, 单块数据量 = 8192 * 4096 * 2 = 64MB

AdjustHCCLLimit() 检查：
  单块数据量 = 64MB ≤ 256MB ✓
  
结论：仍满足限制
```

**场景5：更极端数据量（触发 AdjustHCCLLimit）**
```
假设：
  mValue = 262144, nValue = 8192, dtypeSize = 4 (FP32)
  总数据量 = 262144 * 8192 * 4 = 8GB

公式化切分初始结果：
  totalTileCnt = 128（可能）
  
MaxTileCntUniform() 强制调整：
  totalTileCnt = 16
  tileM = 16384, 单块数据量 = 16384 * 8192 * 4 = 512MB

AdjustHCCLLimit() 触发重新切分：
  minSplitPart = ⌈8GB / 256MB⌉ = ⌈32⌉ = 32
  tileM = ⌈262144 / 32⌉ = 8192
  tileCnt = ⌊262144 / 8192⌋ = 32
  tailCnt = 0
  
验证：
  tileCnt + tailCnt = 32 > 16  ⚠️
  单块数据量 = 8192 * 8192 * 4 = 256MB ✓
  
问题：重新切分后 totalTileCnt = 32，超过了 maxTileCnt = 16！
```

### 3.3 潜在冲突场景

**发现的关键问题**：AdjustHCCLLimit 重新切分后，可能出现 `tileCnt + tailCnt > 16` 的情况！

#### 问题根源

**AdjustHCCLLimit 的切分逻辑**：
```cpp
uint64_t minSplitPart = Ops::Base::CeilDiv(
    args_.mValue * args_.nValue * ge::GetSizeByDataType(args_.geCType),
    mc2tiling::ALL_GATHER_HCCL_MEM_LIMIT);

tileMValue_ = Ops::Base::CeilDiv(args_.mValue, minSplitPart);
rcsCfg.tileCnt = Ops::Base::FloorDiv(args_.mValue, tileMValue_);
rcsCfg.tailCnt = (tailMValue_ == 0) ? 0 : 1;
```

**数学关系**：
```
tileCnt = ⌊mValue / tileMValue_⌋
tileMValue_ = ⌈mValue / minSplitPart⌉

近似估算：
tileCnt ≈ mValue / (mValue / minSplitPart) = minSplitPart

最终：tileCnt + tailCnt ≈ minSplitPart + 1
```

**冲突条件**：
```
如果 minSplitPart > 16：
  tileCnt + tailCnt > 16 → 超过 maxTileCnt 限制！
```

**触发阈值**：
```
总数据量 > 16 * 256MB = 4GB
```

### 3.4 代码实际验证

**调用顺序**（`matmul_reduce_scatter_v2_tiling.cpp` 第 184 行）：
```cpp
MC2_CHECK_LOG_RET(opName_, AdjustHCCLLimit(matmulReduceScatterV2TilingData_->param, 
                                            mc2tiling::Mc2QuantMode::DEFAULT));
```

**调用时机**：在公式化切分之后

**潜在问题**：
- AdjustHCCLLimit 重新切分后，没有再次检查 totalTileCnt ≤ 16
- 可能违反 FFTS 队列限制

---

## 4. 实际限制验证

### 4.1 最终 tileCnt 最大值

**理论最大值**：
- **正常情况**：受 maxTileCnt 限制，tileCnt ≤ 16
- **AdjustHCCLLimit 触发后**：tileCnt 可能达到 `⌈总数据量/256MB⌉`

**实际限制**：
```cpp
// 没有显式的二次检查代码
// AdjustHCCLLimit 重新切分后，tileCnt 可能超过 16
```

### 4.2 最终 tailCnt 最大值

**固定限制**：tailCnt 值为 **0 或 1**

**代码验证**：
```cpp
if (tailMValue_ == 0) {
    rcsCfg.tailCnt = 0;
} else {
    rcsCfg.tailCnt = 1;
}
```

### 4.3 tileCnt + tailCnt 最大值

**正常情况**：≤ 16

**AdjustHCCLLimit 触发后**：
- **无二次检查**：可能超过 16
- **最大可能值**：`⌈总数据量/256MB⌉ + 1`

### 4.4 其他约束条件

#### ReduceScatterSpliteM 函数限制

**代码位置**：`matmul_reduce_scatter_tiling_base.cpp` 第 54-71 行

```cpp
uint32_t MatmulReduceScatterTilingBase::ReduceScatterSpliteM(
    mc2tiling::TilingArgs& args, uint32_t maxTileCnt = 64) const
{
    // 检查允许通信的最大次数
    if (args.commTurn >= maxTileCnt) {
        args.commTurn = maxTileCnt;  // 默认 maxTileCnt = 64
    }
    
    uint64_t tileLen = 1;
    if (args.mValue > args.commTurn) {
        tileLen = args.mValue / args.commTurn;
    }
    
    // 对齐处理
    if (args.outputDtypeSize == 2) {
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 64);
    } else if (args.outputDtypeSize == 4) {
        tileLen = mc2tiling::AlignUp<uint64_t>(tileLen, 32);
    }
    
    return args.mValue > tileLen ? tileLen : args.mValue;
}
```

**注意**：这里的 `maxTileCnt = 64`（默认值），与公式化切分的 `MAX_TILE_CNT = 16` 不同！

---

## 5. HCCL 资源限制分析

### 5.1 HCCL 通信句柄数量限制

**常量定义**（`mc2_tiling_utils.h` 第 90-92 行）：
```cpp
constexpr static uint64_t ALL_GATHER_HCCL_MEM_LIMIT = 256 * 1024 * 1024;  // 256MB
constexpr static uint64_t ALL_GATHER_HCCL_NUM_LIMIT = 16;
constexpr static uint64_t MC2_HCCL_NUM_LIMIT = 63;
```

**含义**：
- `ALL_GATHER_HCCL_NUM_LIMIT = 16`：AllGather 算子的通信次数限制
- `MC2_HCCL_NUM_LIMIT = 63`：MC2 系列算子的通用通信次数限制

### 5.2 FFTS 队列限制

**来源**：硬件架构限制

**容量**：16 个并发任务

**影响**：
- 公式化切分的 `MAX_TILE_CNT = 16` 直接来源于此
- 每个通信轮次占用一个 FFTS 队列项

### 5.3 其他硬件资源限制

#### HCCL 缓冲区限制

**大小**：256MB（单次通信数据量上限）

**原因**：
- HCCL 内部通信缓冲区大小限制
- 防止内存溢出

#### 通信带宽限制

**影响**：
- 影响通信时间拟合模型
- 不直接影响切分块数

---

## 6. 总结与建议

### 6.1 核心发现

#### maxTileCnt=16 的含义
- **限制对象**：totalTileCnt（总切分块数 = tileCnt + tailCnt）
- **限制原因**：FFTS 队列容量限制（16 个并发任务）
- **强制机制**：MaxTileCntUniform() 函数强制调整

#### AdjustHCCLLimit 的逻辑
- **触发条件**：单块数据量超过 256MB
- **调整方式**：重新切分以满足 256MB 限制
- **核心算法**：计算 minSplitPart，重新计算 tileM、tileCnt、tailCnt

#### 冲突分析结论
- **不存在直接冲突**：两个限制作用于不同维度
- **存在潜在问题**：AdjustHCCLLimit 重新切分后可能超过 maxTileCnt
- **触发条件**：总数据量 > 4GB（16 * 256MB）

### 6.2 实际限制总结

| 限制维度 | 正常情况 | AdjustHCCLLimit 触发后 | 验证状态 |
|---------|---------|----------------------|---------|
| **tileCnt** | ≤ 16 | 可能 > 16 | ⚠️ 无二次检查 |
| **tailCnt** | 0 或 1 | 0 或 1 | ✓ 固定限制 |
| **tileCnt + tailCnt** | ≤ 16 | 可能 > 16 | ⚠️ 潜在违反 |
| **单块数据量** | ≤ 256MB | ≤ 256MB | ✓ 强制保证 |

### 6.3 建议改进

#### 建议1：添加二次检查

在 AdjustHCCLLimit 重新切分后，添加 totalTileCnt 检查：

```cpp
ge::graphStatus MatmulReduceScatterTilingBase::AdjustHCCLLimit(...) {
    ...
    // 重新切分逻辑
    
    // 新增：检查是否超过 maxTileCnt
    uint32_t totalTileCnt = rcsCfg.tileCnt + rcsCfg.tailCnt;
    if (totalTileCnt > MAX_TILE_CNT) {
        OP_LOGE(opName_, "After AdjustHCCLLimit, totalTileCnt %u exceeds maxTileCnt %u",
                totalTileCnt, MAX_TILE_CNT);
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}
```

#### 建议2：统一限制常量

将多个相关的限制常量统一管理：

```cpp
// 建议：统一命名和注释
constexpr uint64_t MAX_TILE_CNT_FFTS = 16;      // FFTS 队列限制
constexpr uint64_t HCCL_MEM_LIMIT = 256 * 1024 * 1024;  // HCCL 缓冲区限制
constexpr uint64_t HCCL_NUM_LIMIT = 63;         // HCCL 通信次数限制
```

#### 建议3：增强日志记录

在 AdjustHCCLLimit 中记录 totalTileCnt：

```cpp
OPS_LOG_I(opName_, "After AdjustHCCLLimit: totalTileCnt [%u] = tileCnt [%u] + tailCnt [%u]",
          rcsCfg.tileCnt + rcsCfg.tailCnt, rcsCfg.tileCnt, rcsCfg.tailCnt);
```

### 6.4 相关代码位置索引

| 文件 | 关键代码 | 行号 | 说明 |
|------|---------|------|------|
| `hccl_formulaic_tiling.h` | MAX_TILE_CNT = 16 | 38 | 常量定义 |
| `hccl_formulaic_tiling.h` | SetMaxTileCnt() | 131 | 设置限制 |
| `hccl_formulaic_tiling.cpp` | MaxTileCntUniform() | 317-333 | 强制调整 |
| `formulaic_tiling_datatype.h` | CutResult | 140-147 | 数据结构 |
| `mc2_tiling_struct.h` | RCSTiling | 105-134 | 输出结构 |
| `mc2_tiling_utils.h` | HCCL 常量 | 90-92 | 限制常量 |
| `matmul_reduce_scatter_tiling_base.cpp` | AdjustHCCLLimit() | 127-157 | 重新切分 |
| `matmul_reduce_scatter_tiling_base.cpp` | ReduceScatterSpliteM() | 54-71 | 默认 maxTileCnt=64 |

---

## 附录：数学推导

### A.1 totalTileCnt 计算

**公式化切分**：
```
totalTileCnt = numLongTile + numShortTile
             = ⌊totalLen / longTileLen⌋ + (有尾块 ? 1 : 0)
```

**MaxTileCntUniform 调整**：
```
如果 totalTileCnt > maxTileCnt:
  longTileLen = ⌊totalLen / (maxTileCnt - 1)⌋_aligned
  numLongTile = ⌊totalLen / longTileLen⌋
  numShortTile = (有余数 ? 1 : 0)
  totalTileCnt ≤ maxTileCnt
```

### A.2 AdjustHCCLLimit 计算

**单块数据量**：
```
单块数据量 = tileM * n * dtypeSize
```

**最小切分份数**：
```
minSplitPart = ⌈总数据量 / 256MB⌉
             = ⌈(m * n * dtypeSize) / 256MB⌉
```

**重新切分**：
```
tileM = ⌈m / minSplitPart⌉
tileCnt = ⌊m / tileM⌋
tailCnt = (有尾块 ? 1 : 0)
```

**totalTileCnt**：
```
totalTileCnt = tileCnt + tailCnt
             ≈ minSplitPart + 1
```

**超过 maxTileCnt 的条件**：
```
minSplitPart > 16
→ 总数据量 > 16 * 256MB = 4GB
```

---

**文档版本**：v1.0  
**生成时间**：2026-06-26  
**分析深度**：完整代码追踪 + 数学推导 + 场景验证