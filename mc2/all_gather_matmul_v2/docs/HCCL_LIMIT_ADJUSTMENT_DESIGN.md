# HCCL通信限制调整设计文档

## 1. 问题背景

### 1.1 HCCL通信约束

HCCL集合通信存在以下硬件约束：

| 约束项 | 限制值 | 说明 |
|-------|-------|------|
| 单次通信量上限 | 256MB | 单次AllGather通信数据量不得超过256MB |
| 通信次数上限 | 63次 | 总通信次数（tileCnt + tailCnt）不得超过63次 |

### 1.2 触发场景

当公式化切分（Formulaic Tiling）计算出的切分结果不满足上述约束时，需要进行调整：

- **场景1**：单次通信量超过256MB
  - 原因：tileM × K × dtypeSize × rankDim > 256MB
  - 处理：调整tileM，使其满足限制

- **场景2**：总通信次数超过63次
  - 原因：M维过大，切分份数超过63
  - 处理：反向计算最小tileM，满足63次限制

- **场景3**：无法同时满足两个约束
  - 原因：极端大数据量，即使63次通信、每次256MB也处理不完
  - 处理：报错提示不支持

---

## 2. 设计方案

### 2.1 整体流程

```
CalcMaxTileMFromHcclLimit()      // 步骤1: 判断是否需要调整，计算maxTileM
    ↓
SelectOptimalCandidateTileM()    // 步骤2: 优选tileM候选值（2048优先）
    ↓
DetermineFinalTileMWith63Limit() // 步骤3: 确定满足63次限制的最终tileM
    ↓
ApplyTileSplit()                 // 步骤4: 应用切分结果
```

### 2.2 算法步骤详解

#### 步骤1：CalcMaxTileMFromHcclLimit

**功能**：计算满足256MB限制的最大tileM，并判断是否需要调整

**算法**：
```
1. 安全检查：
   - singleRowSize = K × dtypeSize × rankDim
   - 若 singleRowSize == 0，报错返回 HCCL_UNSUPPORTED（除零保护）
   - 分步计算 originalCommSize，每步检查溢出

2. 判断是否需要调整：
   - 若 originalCommSize ≤ 256MB，返回 HCCL_NO_ADJUSTMENT（不需要调整）
   - 否则，计算 maxTileM = 256MB / singleRowSize

3. 边界检查：
   - 若 maxTileM < 256，报错返回 HCCL_UNSUPPORTED（无法对齐）
   - 否则，返回 maxTileM
```

**安全防护**：
- 除零保护：检查 `singleRowSize == 0`
- 溢出保护：分步计算乘法，每步验证 `a × b` 是否溢出

#### 步骤2：SelectOptimalCandidateTileM

**功能**：从候选列表中优选满足256MB限制的tileM

**候选列表**：`[2048, 1024, 512, 256]`（优先2的幂次大值）

**优选策略**：
```
从大到小遍历候选列表
找第一个 ≤ maxTileM 的值
若候选值都超过maxTileM，返回向下256对齐的计算值
```

**优选原因**：
- 大tileM → 少通信次数 → 性能更好
- 2的幂次 → 整齐的block划分 → 计算效率高

#### 步骤3：DetermineFinalTileMWith63Limit

**功能**：确保总通信次数不超过63次

**算法**：
```
1. 预估总切分份数：
   totalTileCnt = ceil(M / candidateTileM)

2. 判断是否满足63次限制：
   若 totalTileCnt ≤ 63，直接返回 candidateTileM（优选值已满足）

3. 超过63次时反向计算：
   minTileM = ceil(M / 63)
   minTileM_align = 向上256对齐(minTileM)
   
4. 从候选列表找满足最小值的候选：
   从小到大遍历候选列表，找第一个 ≥ minTileM_align 的值
   
5. 验证256MB限制：
   若 finalTileM > maxTileM，报错返回 HCCL_UNSUPPORTED
   否则，返回 finalTileM
```

#### 步骤4：ApplyTileSplit

**功能**：应用切分结果，更新CutResult

**算法**：
```
1. 计算切分：
   numLongTile = floor(M / tileM)
   tailM = M - numLongTile × tileM

2. 边界处理：
   若 numLongTile == 0（M < tileM）：
     - 调整为单块：numLongTile=1, tileM=M, tailM=0
     - 验证单块256MB限制
     - 若超过，报错返回

3. 更新CutResult各字段：
   - longTileLen = tileM（主块长度）
   - numLongTile（主块数量）
   - shortTileLen = tailM（尾块长度）
   - numShortTile = (tailM > 0) ? 1 : 0
   - totalTileCnt = numLongTile + numShortTile
   - shortTileAtBack 保持原始值
```

---

## 3. 安全防护设计

### 3.1 除零保护（Critical）

**问题**：除法运算 `maxTileM = 256MB / (K × dtypeSize × rankDim)` 可能除数为零

**防护**：
```cpp
uint64_t singleRowSize = kValue * dtypeSize * rankDim;
if (singleRowSize == 0) {
    OP_LOGE(opName.c_str(),
            "Invalid parameters: kValue=%lu, dtypeSize=%lu, rankDim=%lu\n"
            "  Product is zero, division undefined\n"
            "  This shape is NOT SUPPORTED",
            kValue, dtypeSize, rankDim);
    return HCCL_UNSUPPORTED;
}
```

**触发场景**：
- kValue = 0（K维为零）
- dtypeSize = 0（数据类型无效）
- rankDim = 0（卡数为零）

### 3.2 溢出保护（Medium）

**问题**：计算 `originalCommSize = tileM × K × dtypeSize × rankDim` 时，4个uint64_t相乘可能溢出

**防护**：
```cpp
// 分步计算并检查溢出
uint64_t tileMKBytes = cutRes.longTileLen * kValue;
if (cutRes.longTileLen != 0 && tileMKBytes / cutRes.longTileLen != kValue) {
    OP_LOGE(opName.c_str(), "Multiplication overflow: longTileLen * kValue");
    return HCCL_UNSUPPORTED;
}

uint64_t tileBytesPerRank = tileMKBytes * dtypeSize;
if (tileMKBytes != 0 && tileBytesPerRank / tileMKBytes != dtypeSize) {
    OP_LOGE(opName.c_str(), "Multiplication overflow: tileMKBytes * dtypeSize");
    return HCCL_UNSUPPORTED;
}

uint64_t originalCommSize = tileBytesPerRank * rankDim;
if (tileBytesPerRank != 0 && originalCommSize / tileBytesPerRank != rankDim) {
    OP_LOGE(opName.c_str(), "Multiplication overflow: tileBytesPerRank * rankDim");
    return HCCL_UNSUPPORTED;
}
```

**溢出检测原理**：若 `a × b` 溢出，则 `result / a ≠ b`

---

## 4. 性能优化策略

### 4.1 候选值优选

| 优先级 | 候选值 | 适用场景 |
|-------|-------|---------|
| 1 | 2048 | 正常数据量，性能最优 |
| 2 | 1024 | 中等数据量 |
| 3 | 512 | 较大数据量 |
| 4 | 256 | 小数据量/极限场景 |

### 4.2 通信次数控制

| 总切分份数 | 处理策略 |
|-----------|---------|
| ≤ 16 | 正常模式：使用优选候选值，性能最优 |
| 17 ~ 63 | 扩展模式：仍使用优选候选值，通信次数增加 |
| > 63 | 反向计算：调整tileM满足63次限制 |

---

## 5. 错误处理

### 5.1 不支持的Shape

以下场景会报错返回 `HCCL_UNSUPPORTED`：

| 场景 | 错误信息 | 原因 |
|-----|---------|------|
| 参数为零 | "Product is zero, division undefined" | singleRowSize = 0 |
| 溢出 | "Multiplication overflow" | 中间结果超过UINT64_MAX |
| Shape太小 | "Shape too small: maxTileM < 256" | 无法256对齐 |
| Shape太大 | "Cannot satisfy HCCL 256MB limit with 63 tiles" | 极端大数据量 |

### 5.2 返回值定义

| 返回值 | 含义 |
|-------|------|
| HCCL_NO_ADJUSTMENT (0) | 不需要调整，原始切分已满足 |
| HCCL_UNSUPPORTED | 无法支持，报错退出 |
| 其他值 | 需要调整，返回调整后的maxTileM |

---

## 6. 代码检视修复记录

### 6.1 ISSUE-001（Critical级除零保护）

**检视意见**：`CalcMaxTileMFromHcclLimit` 函数除法运算 `maxTileM = HCCL_MEM_LIMIT / (kValue * dtypeSize * rankDim)` 除数可能为零。

**修复方案**：
- 新增 `singleRowSize` 计算，提前检查是否为零
- 若为零，返回 `HCCL_UNSUPPORTED` 并记录错误日志

**修复时间**：2026-05-22

### 6.2 ISSUE-002（Medium级溢出保护）

**检视意见**：`originalCommSize = cutRes.longTileLen * kValue * dtypeSize * rankDim` 4个uint64_t相乘可能溢出回绕。

**修复方案**：
- 分步计算乘法：tileMKBytes → tileBytesPerRank → originalCommSize
- 每步通过反向除法验证是否溢出
- 若溢出，返回 `HCCL_UNSUPPORTED` 并记录错误日志

**修复时间**：2026-05-22

---

## 7. 测试验证

### 7.1 正常场景验证

| 测试用例 | M | K | dtypeSize | rankDim | 预期结果 |
|---------|---|---|-----------|---------|---------|
| 小数据量 | 4096 | 12288 | 2 | 8 | tileM=1024, tileCnt=4 |
| 中数据量 | 8192 | 12288 | 2 | 8 | tileM=1024, tileCnt=8 |
| 大数据量 | 100000 | 1000 | 2 | 8 | 扩展到63次内 |

### 7.2 异常场景验证

| 测试用例 | 参数 | 预期结果 |
|---------|------|---------|
| K=0 | kValue=0 | 返回 HCCL_UNSUPPORTED |
| dtypeSize=0 | dtypeSize=0 | 返回 HCCL_UNSUPPORTED |
| rankDim=0 | rankDim=0 | 返回 HCCL_UNSUPPORTED |
| 溢出场景 | 极大M/K值 | 返回 HCCL_UNSUPPORTED |

---

## 8. 相关文件

| 文件路径 | 说明 |
|---------|------|
| `op_host/op_tiling/arch35/all_gather_hccl_utils.h` | HCCL限制调整工具函数头文件 |
| `op_host/op_tiling/arch35/all_gather_hccl_utils.cpp` | HCCL限制调整工具函数实现 |
| `op_host/op_tiling/all_gather_matmul_tiling_base.cpp` | CheckHCCLSize前置检查函数 |
| `op_kernel/common.h` | HCCL通信常量定义 |

---

## 9. 变更历史

| 版本 | 日期 | 变更内容 |
|-----|------|---------|
| v1.0 | 2026-05-22 | 初始设计：256MB限制调整，最多8次通信 |
| v2.0 | 2026-05-22 | 扩展设计：最多63次通信，支持更大数据量 |
| v2.1 | 2026-05-22 | 安全防护：新增除零保护和溢出保护 |