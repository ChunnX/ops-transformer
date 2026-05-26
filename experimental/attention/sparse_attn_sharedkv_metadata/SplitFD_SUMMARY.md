# SplitFD 函数总结

## 功能定位

`SplitFD` 函数负责将 **FlashDecode（FD）归约任务** 分配到多个 **AIV Vector 核** 上执行，实现负载均衡。

当稀疏注意力的 S2 维度被切分到多个 AIC 核时，每个核产出的部分 softmax 结果需要 AIV 核进行归约合并，此函数就是规划这些归约任务的分配。

---

## 核心计算公式

### 1. 计算 FD 总负载

**公式**：`totalFDLoad = Σ(fdS2SplitNum[i] × fdMSize[i])`

**含义**：遍历所有需要 FD 归约的 head，累加每个 head 的数据量。

| 变量 | 说明 |
|------|------|
| `numOfFdHead` | 需要 FD 归约的 head 数量 |
| `fdS2SplitNum[i]` | 第 i 个 head 的 S2 切分块数（被拆到多个 AIC 核） |
| `fdMSize[i]` | 第 i 个 head 的 M 维度总大小 |
| `totalFDLoad` | FD 归约的总数据量 |

### 2. 计算每核平均负载

**公式**：`averageLoad = totalFDLoad / aivCoreNum_`

**含义**：将总负载平均分配到所有可用的 AIV 核上。

| 变量 | 说明 |
|------|------|
| `aivCoreNum_` | AIV Vector 核的数量（Ascend A3 为 72） |
| `averageLoad` | 每个 AIV 核应处理的平均数据量 |

### 3. 计算单个 FD head 的核分配数

**公式**：`curFDVectorNum = (fdS2SplitNum[i] × fdMSize[i]) / averageLoad`

**含义**：根据单个 head 的数据量占比，计算需要分配的 AIV 核数量。

### 4. 计算每核处理的 M 维度大小

**公式**：`curAveMSize = fdMSize[i] / curFDVectorNum`

**含义**：将单个 head 的 M 维度均匀分配到对应数量的 AIV 核。

### 5. 分配 M 维度起止位置

**公式**：
- `fdMStart[core] = vid × curAveMSize`
- `fdMNum[core] = (最后一块) ? (fdMSize[i] - vid × curAveMSize) : curAveMSize`

**含义**：为每个 AIV 核分配 M 维度的连续片段，最后一块处理整除后的余数。

---

## 执行流程

1. **统计阶段**：遍历所有 FD head，计算总数据量
2. **规划阶段**：计算每核平均负载，确定每个 head 需要的核数
3. **分配阶段**：为每个 AIV 核设置任务边界（head 索引、M 起始位置、M 数量）
4. **收尾阶段**：记录实际使用的 AIV 核总数

---

## 设计意图

| 设计点 | 说明 |
|--------|------|
| **负载均衡** | 根据数据量按比例分配到多个 AIV 核，避免个别核过载 |
| **M 维度切分** | 将 M 维度均匀切分，每个核处理连续的一段，便于向量化处理 |
| **余数处理** | 最后一个核处理剩余的 M 维度，确保数据完整性 |
| **批量分配** | 一个 FD head 可分配到多个 AIV 核并行处理，提高归约效率 |

---

## 输入输出

### 输入

| 字段 | 类型 | 说明 |
|------|------|------|
| `numOfFdHead` | uint32_t | 需要 FD 归约的 head 数量 |
| `fdS2SplitNum[]` | uint32_t[] | 每个 head 的 S2 切分块数 |
| `fdMSize[]` | uint32_t[] | 每个 head 的 M 维度大小 |

### 输出

| 字段 | 类型 | 说明 |
|------|------|------|
| `fdIdx[]` | uint32_t[] | 每个 AIV 核处理的 head 索引 |
| `fdMStart[]` | uint32_t[] | 每个 AIV 核处理的 M 维度起始位置 |
| `fdMNum[]` | uint32_t[] | 每个 AIV 核处理的 M 维度数量 |
| `fdUsedVecNum` | uint32_t | 实际使用的 AIV 核总数 |

---

## 总结

`SplitFD` 实现了 FlashDecode 归约任务的多核并行分配策略：

- **核心目标**：将跨核切分的注意力结果均衡分配到 AIV 核进行归约
- **分配策略**：按数据量比例分配，M 维度连续切分
- **技术要点**：负载均衡、余数处理、批量并行