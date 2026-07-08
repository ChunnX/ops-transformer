# 白盒用例设计规则 — SparseFlashMla (arch35)

---

## 一、参数维度的边界值 / 极值覆盖

### 1.1 核心维度参数

| 参数 | 最小值 | 典型值 | 最大值 | 特殊边界 |
|------|--------|--------|--------|----------|
| `B` (batch) | 1 | 4, 16 | 大B (如256) | **0** — 空batch |
| `S1` (Q seq) | 1 | 128, 4096 | 大S1 | **0** — 空query（触发needInit） |
| `N1` (Q heads) | 1 | 64, 128 | 128 | 必须被N2整除；A2/A3要求2的幂 |
| `N2` (KV heads) | 1 | 1 | 1 | **固定为1** |
| `D` (head dim) | 512 | 512 | 512 | **固定为512** |
| `K` (top-k) | — | 512, 1024 | 1024 | CSA模式强制要求 |
| `actualS2OriSize` | 0 | 4096, 8192 | 大值 | **0** — 无原始KV |
| `actualS2CmpSize` | 0 | 1024 (=4096/4) | 大值 | **0** — 无压缩KV |
| `cmpRatio` | 1 | 4, 128 | 128 | CSA=4, HCA=128 |
| `gSize` (=N1/N2) | 1 | 64, 128 | 128 | |

### 1.2 Sliding Window 参数

| 参数 | 极小/特殊 | 典型值 | 极大/特殊 |
|------|-----------|--------|-----------|
| `oriWinLeft` | **-1** (无限) | 127 | 大值 |
| `oriWinRight` | **-1** (无限) | 0 | 大值 |
| `oriMaskMode` | 0 (NO_MASK) | 4 (BAND) | 3 (RIGHT_DOWN_CAUSAL) |
| `cmpMaskMode` | 0 (NO_MASK) | 3 (RIGHT_DOWN_CAUSAL) | |

**组合场景：**

- NO_MASK + full S2 range → 无裁剪
- BAND + WinLeft=-1 → `preTokensPerBatchOri = actualS1Size`（全因果）
- BAND + 有限窗口 → `preTokensPerBatchOri < actualS1Size`
- RIGHT_DOWN_CAUSAL → `preTokens=INT64_MAX, nextTokens=0`

### 1.3 nextTokensPerBatch 的负值场景

```
nextTokensPerBatchOri = actualS2OriSize - actualS1Size
```

当 `actualS2OriSize < actualS1Size` 时，该值为**负数** → 跳过前导GS1循环（`ComputeS1LoopInfo` 中 `gs1LoopStartIdx` 被调整）。

### 1.4 数据类型

| 类型 | 场景 |
|------|------|
| `FLOAT16` | 标准精度 |
| `BF16` | 另一种精度路径（tiling key不同） |

---

## 二、块切分与尾块覆盖

### 2.1 各轴的块大小与尾块条件

| 轴 | 块大小 | 尾块条件 | 尾块代码路径 |
|----|--------|----------|--------------|
| **S1 (Q)** | `qSNumInOneBlock` = gSize (SWA/HCA) 或 1 (CSA) | `s1RealSize < qSNumInOneBlock` | `ComputeSouterParam` 中 `s1RealSize = Min(qSNumInOneBlock, actualS1Size - cubeSOuterOffset)` |
| **S2 (KV)** | `s2BaseSize = 128` | `s2RealSize < 128` | `SetRunInfo` 中 `s2RealSize` 计算；softmax 分支 `s2RealSize <= 64` vs `> 64` |
| **M (M=Q*G)** | `mBaseSize` (SOC相关) | 非整除时的余数 | metadata `s1GTailSize` 触发 `TAIL_BLOCK` 代价 |
| **K (matmul)** | `dBaseMatmulSize = 128` | dSize=512 始终整除，**无尾块** | — |

### 2.2 S1 尾块详细场景

**SWA/HCA 模式（qSNumInOneBlock = gSize）：**

- `actualS1Size` 恰好被 `gSize` 整除 → 所有块满块
- `actualS1Size < gSize` → 唯一个尾块（s1RealSize < gSize）
- `actualS1Size > gSize` 且不整除 → 最后一个块为尾块
- `actualS1Size = 0` → s1RealSize=0, mRealSize=0，跳过计算

**CSA 模式（qSNumInOneBlock = 1）：**

- 每个GS1循环处理1个Q token → `s1RealSize` 始终为1（最后一个 GS1 也一样）
- 无需考虑S1轴尾块

### 2.3 S2 尾块详细场景

**s2RealSize 的3种 softmax 分支：**

1. `s2RealSize == 128 && s2RealSizeUpdate == 128` → `EQ_128_SFA` 快速路径
2. `s2RealSize <= 64` → `GT_0_AND_LTE_64_SFA` 分支
3. `s2RealSize > 64 && s2RealSize < 128` → `GT_64_AND_LTE_128_SFA` 分支

**尾块测试矩阵：**

| s2RealSize | softmax分支 | 场景说明 |
|------------|-------------|----------|
| 128 | EQ_128 | 满块 |
| 127 | GT_64_AND_LTE_128 | 接近满块的尾块 |
| 65 | GT_64_AND_LTE_128 | 分界线上方 |
| 64 | GT_0_AND_LTE_64 | 刚好分界线 |
| 63 | GT_0_AND_LTE_64 | 分界线下方 |
| 1 | GT_0_AND_LTE_64 | 极小尾块 |
| 0 | 跳过 | 空KV循环 |

### 2.4 subBlock 非对称分割

在 `ComputeSouterParam` 中：

- subBlock 0: `halfMRealSize = ceil(mRealSize / 2)`
- subBlock 1: `halfMRealSize = floor(mRealSize / 2)`

**当 mRealSize 为奇数时两个 subBlock 不均衡** → 需要分别为 subBlock 0 和 subBlock 1 设计用例。

### 2.5 IS_SPLIT_G 下的 M 轴切分

- `mRealSize` 被折半 → 产生新的尾块场景
- `halfMRealSize` / `firstHalfMRealSize` 计算进一步复杂化
- `cubeMOuterOffset` 不为0（第二个核心）

### 2.6 V0 稀疏KV gather 中16行分块

```cpp
calNum = Min(16, s2RealSize - startPos);
s2Loops = ceil_div(calNum, 2);
```

- `s2RealSize <= 16` → 单次处理
- `s2RealSize > 16` → 多次外循环（ping-pong）
- 最后一次的 `calNum < 16` → 尾块处理

---

## 三、分核场景覆盖（metadata / load balancing）

### 3.1 分核算法产生的主要场景

| # | 场景 | 触发条件 | 关键代码路径 |
|---|------|----------|--------------|
| 1 | **all-zero KV 短路** | 所有batch的 `s1GBaseNum==0` 或 `(oriS2BaseNum==0 && cmpS2BaseNum==0)` | `BalanceSchedule` 直接返回1核心 |
| 2 | **单核心分配** | 总代价 ≤ 平均代价 | `usedCoreNum == 1` |
| 3 | **batch级分配** | 整batch代价 ≤ costLimit | `AssignByBatch` |
| 4 | **row级分配** | 整GS1行代价 ≤ costLimit | `AssignByRow` |
| 5 | **block级分配** | supportFd=true，单S2块分配 | `AssignByBlock` |
| 6 | **强制分配** | supportFd=true, 某核心0块 | `ForceAssign` 强制1块 |
| 7 | **跨核心KV行分裂(FD)** | 一个KV行的S2块跨多个核心 | `curKvSplitPart > 1` → 记录FD信息 |

### 3.2 SOC版本差异

| 特性 | ASCEND910 | ASCEND950 |
|------|-----------|-----------|
| `s2BaseSize` | **512** | **128** |
| `mBaseSize` | isSparseCmpKv? gSize : (256/gSize)*gSize | **gSize** |
| `isSplitG` | 始终false | N1==128 时为true |
| `aicCoreNum` | 不变 | N1==128 时 **减半** |
| metadata输出 | 1逻辑核心/物理核心 | 2逻辑核心/物理核心(isSplitG) |

### 3.3 代价计算中的尾块

metadata中 `CalcCostTable` 构建4×4代价矩阵：

```
         ORI_NORMAL  ORI_TAIL  CMP_NORMAL  CMP_TAIL
ORI_NORMAL   [0,0]     [0,1]      0          0
ORI_TAIL     [1,0]     [1,1]      0          0
CMP_NORMAL     0         0       [2,2]     [2,3]
CMP_TAIL       0         0       [3,2]     [3,3]
```

**需要覆盖的尾块组合：**

- S1G尾块 + S2尾块 (双尾, `[TAIL_BLOCK][TAIL_BLOCK]`)
- S1G尾块 + S2正常块 (`[TAIL_BLOCK][NORMAL_BLOCK]`)
- S1G正常块 + S2尾块 (`[NORMAL_BLOCK][TAIL_BLOCK]`)
- S1G正常块 + S2正常块 (`[NORMAL_BLOCK][NORMAL_BLOCK]`)

### 3.4 分配容忍度

```
IsWithinTolerance(costLimit + lastBlockCost / FA_TOLERANCE_RATIO >= coreCache.cost + assignedCost)
```

- 刚好在容忍度边界上的 batch/row → 跨核心分配
- 刚好在容忍度边界下的 batch/row → 当前核心分配

### 3.5 FD (Flash Decode) 负载均衡

- FD 向量数 = `ceil(totalFDLoad / aivCoreNum)`
- 每个 reduction task 至少1个向量
- M轴按 `curAveMSize` 均分，重新计算 `curFDVectorNum`

**边界：** `fdMSize` 除以 `curFDVectorNum` 不能整除 → 尾向量处理少量M行。

### 3.6 极端不平衡场景

- 1个巨大batch + N个微小batch → `maxS1GCost` 作为 costLimit 下界
- 所有batch都很大，均匀分布 → 多核心均匀分配
- 部分batch的S2=0 → 跳过分核

---

## 四、Kernel 内部分支场景

### 4.1 Pipeline 状态

三阶段流水线 `ProcessMainLoop`：

| taskId | Stage0 (BMM1+Vec0) | Stage1 (Vec1+BMM2) | Stage2 (Vec2) |
|--------|--------------------|--------------------|---------------|
| 0 | ✓ | — | — |
| 1 | ✓ | — | Vec2 |
| 2 | ✓ | Vec1+BMM2 | Vec2 |
| 3+ | ✓ | Vec1+BMM2 | Vec2 |

**Drain阶段** (最后2个循环): `s2LoopLimit=0`, 仅运行 dummy 迭代。

- `extraGS1 = 0`: `notLastTwoLoop=false` (Drain stage 1)
- `extraGS1 = 1`: `notLast=false, notLastTwoLoop=false` (Drain stage 2)

**用例要点：**

- `s2LoopLimit = 0` → 无S2循环（空KV）
- `s2LoopLimit = 1` → 仅1个S2循环
- `s2LoopLimit = 2` → 刚好2个循环
- `s2LoopLimit >= 3` → 完整pipeline

### 4.2 S2循环双阶段

```
s2LoopCount < oriKvLoopEndIdx  → 原始KV阶段
s2LoopCount >= oriKvLoopEndIdx → 压缩KV阶段
```

**组合：**

- 仅有原始KV (SWA / oriKvLoopEndIdx>0, cmpKvLoopEndIdx=0)
- 仅有压缩KV (oriKvLoopEndIdx=0, cmpKvLoopEndIdx>0)
- 原始+压缩 (HCA / CSA) — 需覆盖ori→cmp过渡

### 4.3 Flash Attention 状态机

| s2LoopCount | s2RealSizeUpdate | 操作 |
|-------------|------------------|------|
| 0 (无sinks) | — | 初始化softmax |
| 0 (有sinks) | — | 从sinks初始化softmax |
| >0, interior | >0 | `FlashUpdateNew` |
| >0, last | >0 | `FlashUpdateLastNew` + CopyOut |
| >0, interior | 0 | 跳过更新 |
| >0, last | 0 | `LastDivNew` (仅除以sum) |

**用例要点：**

- `s2LoopLimit == 0` + `needInit` → 仅输出初始化
- `s2LoopLimit == 1` + sinks → 单S2块 + sinks初始化 → 最后块
- `s2LoopLimit == 1` 无sinks → 单S2块 + 新初始化 → 最后块
- 中间S2块 `s2RealSizeUpdate == 0` → 空更新跳过分支
- 最后块 `s2RealSizeUpdate == 0` → `LastDivNew`

### 4.4 CSA 稀疏KV处理

- `oriTopkLength` / `cmpTopkLength` 是否为null
- sparseBlockCount为0 → 无稀疏块
- 稀疏indices中 tokenIdx == -1 → 提前终止
- 2个token中仅1个有效 → 单行copy
- 2个token都有效 → 合并copy
- 2个token都无效 → 终止
- `keySrcStride >= INT32_MAX || <0` → 降级为分离copy

### 4.5 output初始化 (needInit)

```cpp
if (s1Size > s2Size || s1Size < expectQs) needInit = 1
```

- S1 > S2 → 部分output位置无S2数据 → 需初始化为0
- TND模式下 `expectQs` 来自 cuSeqlens 累计和 → 不匹配触发初始化

### 4.6 Layout 组合

| Q Layout | KV Layout | 用例 |
|----------|-----------|------|
| BSND | BSND | 标准4D layout |
| TND | TND | 打包TND layout |
| BSND | PA_BBND | BSND Q + PageAttention KV |
| TND | PA_BBND | TND Q + PageAttention KV |

每种layout组合影响：offset计算、cuSeqlens使用、block table使用

### 4.7 可选tensor的 null/非null

| 参数 | 为null时行为 | 非null时行为 |
|------|-------------|-------------|
| `cmp_kv` | SWA模式 | HCA/CSA模式 |
| `cmp_sparse_indices` | SWA/HCA模式 | CSA模式 |
| `cu_seqlens_q` | 使用maxSeqlen（BSND） | 使用累计长度（TND） |
| `seqused_q` | 使用cu_seqlens或maxSeqlen | 使用实际seq len |
| `seqused_ori_kv` | 使用全长度 | 使用实际长度 |
| `ori_block_table` | BSND/TND kv layout | PA_BBND kv layout |
| `cmp_residual_kv` | residual=0 | 使用实际residual |
| `sinks` | 不使用sinks初始化softmax | 从sinks初始化 |
| `return_softmax_lse` | 不输出LSE | 输出LSE |
| `ori_topk_length` | hasOriTopkLength=false | hasOriTopkLength=true |

---

## 五、推荐的测试场景矩阵

### 5.1 基本场景

| 场景编号 | 场景名称 | 关键参数 | 覆盖目标 |
|----------|----------|----------|----------|
| B-01 | 最小batch | B=1, S1=1, N1=1, N2=1 | 最小工作负载 |
| B-02 | 典型CSA decode | B=1, S1=1, N1=64, K=512, cmpRatio=4 | 标准CSA解码 |
| B-03 | 典型CSA prefill | B=1, S1=16, N1=128, K=512, cmpRatio=4 | G-split, prefill |
| B-04 | 大batch多核 | B=16, S1=1, N1=64 | 多batch并行 |
| B-05 | 非2的幂N1(ASCEND950) | B=1, S1=1, N1=32, N2=1 | ASCEND950非2幂头数 |

### 5.2 尾块场景

| 场景编号 | 尾块类型 | 关键参数 | 覆盖目标 |
|----------|----------|----------|----------|
| T-01 | S2尾块=127 | actualS2OriSize=4095 (128×31+127) | GT_64_AND_LTE_128分支 |
| T-02 | S2尾块=64 | actualS2OriSize=(128×N+64) | 刚好GT_0_AND_LTE_64边界 |
| T-03 | S2尾块=63 | actualS2OriSize=(128×N+63) | GT_0_AND_LTE_64分支 |
| T-04 | S2尾块=1 | actualS2OriSize=(128×N+1) | 极小尾块 |
| T-05 | S1尾块(SWA/HCA) | N1=64, gSize=64, actualS1Size=65 | s1RealSize=1 (< gSize) |
| T-06 | S1尾块(SWA/HCA) | N1=64, gSize=64, actualS1Size=63 | s1RealSize < gSize |
| T-07 | M尾块(subBlock奇数) | mRealSize为奇数 | subBlock 0/1不对称 |
| T-08 | 双尾块(S1G尾+S2尾) | mBaseSize非整除 + s2BaseSize非整除 | 4×4代价矩阵TAIL_TAIL |
| T-09 | V0稀疏16行尾块 | sparseCalSize=17 | calNum=16, 第二次=1 |

### 5.3 分核场景

| 场景编号 | 分核类型 | 关键条件 | 覆盖目标 |
|----------|----------|----------|----------|
| C-01 | all-zero KV | 所有batch的KV=0 | 单核心短路 |
| C-02 | 单核心 | 小负载，totalCost ≤ avgCost | usedCoreNum=1 |
| C-03 | batch级分核 | 多batch，整batch可分 | AssignByBatch |
| C-04 | row级分核 | 单batch内GS1行按核分配 | AssignByRow |
| C-05 | block级分核 | supportFd=true, S2块跨核心 | AssignByBlock, FD记录 |
| C-06 | 强制分配 | supportFd=true, 某核心0块 | ForceAssign |
| C-07 | 极端不平衡 | 1个大batch + 多小batch | maxS1GCost下界 |
| C-08 | ASCEND910路径 | s2BaseSize=512 | 不同块大小 |
| C-09 | ASCEND950 + N1=128 | isSplitG=true, aicCoreNum/=2 | 双metadata槽位 |
| C-10 | 仅ori无cmp | hasCmpKv=false (SWA) | cmp成本全0 |
| C-11 | 仅cmp无ori | hasOriKv=false | ori成本全0 |
| C-12 | NO_MASK | oriMaskMode=DEFAULT_MASK | 全S2范围，无掩码 |

### 5.4 Kernel 分支场景

| 场景编号 | 分支类型 | 关键条件 | 覆盖目标 |
|----------|----------|----------|----------|
| K-01 | SWA模式 | 仅ori_kv输入 | SWA kernel template |
| K-02 | HCA模式 | ori_kv + cmp_kv, 无sparse_indices | HCA kernel template |
| K-03 | CSA模式 | ori_kv + cmp_kv + sparse_indices | CSA kernel template |
| K-04 | Pipeline单S2循环 | s2LoopLimit=1 | 仅stage0执行 |
| K-05 | Pipeline双S2循环 | s2LoopLimit=2 | stage0+stage1 |
| K-06 | Pipeline完整 | s2LoopLimit≥3 | 三阶段全执行+drain |
| K-07 | ori→cmp过渡 | oriKvLoopEndIdx>0, cmpKvLoopEndIdx>0 | isCmp标志切换 |
| K-08 | 无sinks初始化 | sinks=null | 新softmax初始化 |
| K-09 | 有sinks初始化 | sinks非null | 从sinks初始化softmax |
| K-10 | needInit | S1 > S2 或 TND长度不匹配 | CleanOutput执行 |
| K-11 | return_softmax_lse | 需要LSE输出 | ComputeLse + CopyOut |
| K-12 | CSA sparse提前终止 | topkIndices=-1 | meetEnd break |
| K-13 | CSA sparse双token有效 | 两个tokenIdx都≥0 | 合并copy |
| K-14 | CSA sparse单token有效 | 仅1个tokenIdx≥0 | 单行copy |
| K-15 | CSA sparse跨步降级 | keySrcStride异常 | 分离copy |
| K-16 | nextTokens负值 | actualS2Ori < actualS1 | 跳过前导GS1循环 |
| K-17 | BSND layout | Q_BSND + KV_BSND | offset计算 |
| K-18 | TND layout | Q_TND + KV_TND | cuSeqlens计算 |
| K-19 | PA_BBND layout | KV_PA_BBND + blockTable | PA offset计算 |

### 5.5 综合场景

| 场景编号 | 组合描述 |
|----------|----------|
| Z-01 | 大B(16) + CSA decode + K=512 + TND/PA_BBND — 典型生产场景 |
| Z-02 | B=1 + CSA prefill + S1=16 + N1=128 + K=512 — G-split + prefill |
| Z-03 | B=1 + SWA + S1=64 + oriWinLeft=127 + oriWinRight=0 — 纯SWA窗口 |
| Z-04 | B=1 + HCA + cmpRatio=128 — 大压缩比 |
| Z-05 | 多batch + 不同长度 (padding) — 变长序列 |
| Z-06 | 单batch + S2=0 + needInit — 空KV输出初始化 |
| Z-07 | FP16 vs BF16 — 两种精度路径 |
| Z-08 | ASCEND910 vs ASCEND950 — 两种SOC路径 |

---

## 六、设计用例时的 Checklist

对每个用例，确认是否覆盖以下检查点：

- [ ] `actualS1Size` 的 **0 / 1 / < gSize / = gSize / > gSize**
- [ ] `actualS2OriSize` 的 **0 / < 128 / = 128 / > 128**（尾块覆盖）
- [ ] `actualS2CmpSize` 的 **0 / < 128 / = 128 / > 128**
- [ ] `mRealSize` 的 **奇数/偶数**（subBlock对称性）
- [ ] `nextTokensPerBatch` 的 **正/0/负**
- [ ] `s2LoopLimit` 的 **0 / 1 / 2 / ≥3**（pipeline深度）
- [ ] `cmpResidual` 的 **0 / 非0**
- [ ] **有无sinks**
- [ ] **是否需要return_softmax_lse**
- [ ] **是否需要needInit**
- [ ] **IS_SPLIT_G 的 true/false**
- [ ] **Layout组合** (BSND/BSND, TND/TND, BSND/PA_BBND, TND/PA_BBND)
- [ ] **Template模式** (SWA/HCA/CSA)
- [ ] **Sparse indices中-1的存在**（有/无）
- [ ] **coreNum分配** (1核 / 部分核 / 全部核)
- [ ] **跨核心KV行分裂** (有/无 FD记录)
