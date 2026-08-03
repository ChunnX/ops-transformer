# aclnnMoeInitRoutingV3 算子功能说明

> 源码位置：`moe/moe_init_routing_v3/`
> 接口文档：`docs/aclnnMoeInitRoutingV3.md`
> 关联接口：[aclnnMoeGatingTopKSoftmaxV2](../../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)（本算子的前置输入）

## 1. 算子定位

`aclnnMoeInitRoutingV3` 是 MoE（Mixture of Experts）模型中的 **routing 算子**，承接 `aclnnMoeGatingTopKSoftmaxV2` 的输出（每个 token 选出的 Top-K 个专家 ID），完成以下核心任务：

1. 按专家 ID 对 token 做稳定排序（key-value sort）。
2. 输出 token 与 expert 的索引映射关系（gather 或 scatter）。
3. 统计每个专家实际处理的 token 数量直方图。
4. 可选地将 token 特征按映射关系搬运为 expandedXOut，并支持多种量化模式。
5. 支持 DropPad 场景：每个专家处理的 token 数对齐到固定 capacity，超出丢弃、不足补零。

## 2. 产品支持

| 产品系列 | 是否支持 |
|----------|:------:|
| Ascend 950PR / Ascend 950DT | ✓ |
| Atlas A3 训练/推理 | ✓ |
| Atlas A2 训练/推理 | ✓ |
| Atlas 200I/500 A2 推理 | ✗ |
| Atlas 推理系列 | ✗ |
| Atlas 训练系列 | ✗ |

> 注：A2/A3 走 A3 实现路径（3 种量化模式）；950PR/950DT 走 A5 实现路径（14 种量化模式）。详见 `A3_implementation.md` / `A5_implementation.md` / `A3_vs_A5_diff.md`。

## 3. 函数原型（两段式接口）

```cpp
// 第一段：获取 workspace 大小与 executor
aclnnStatus aclnnMoeInitRoutingV3GetWorkspaceSize(
    const aclTensor   *x,
    const aclTensor   *expertIdx,
    const aclTensor   *scaleOptional,
    const aclTensor   *offsetOptional,
    int64_t            activeNum,
    int64_t            expertCapacity,
    int64_t            expertNum,
    int64_t            dropPadMode,
    int64_t            expertTokensNumType,
    bool               expertTokensNumFlag,
    int64_t            quantMode,
    const aclIntArray *activeExpertRangeOptional,
    int64_t            rowIdxType,
    const aclTensor   *expandedXOut,
    const aclTensor   *expandedRowIdxOut,
    const aclTensor   *expertTokensCountOrCumsumOut,
    const aclTensor   *expandedScaleOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);

// 第二段：执行计算
aclnnStatus aclnnMoeInitRoutingV3(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## 4. 输入输出

### 4.1 输入 Tensor

| 名称 | shape | dtype | 必选 | 含义 |
|------|-------|-------|:----:|------|
| `x` | `(NUM_ROWS, H)` | FP16/BF16/FP32/INT8/HIF8/FP8_E4M3FN/FP8_E5M2/FP4_E2M1 | ✓ | token 特征矩阵 |
| `expertIdx` | `(NUM_ROWS, K)` | INT32 | ✓ | 每行 token 的 K 个处理专家 ID |
| `scaleOptional` | 1D/2D/3D | FP32/FP8_E8M0 | ✗ | 量化 scale，按 `quantMode` 不同有不同 shape 约束 |
| `offsetOptional` | `(1,)` | FP32 | ✗ | 静态量化偏移；仅静态量化场景必选 |

### 4.2 输入属性

| 属性 | 类型 | 取值 | 含义 |
|------|------|------|------|
| `activeNum` | int64 | ≥0，0 表示 Dropless | 总的最大处理 row 数，约束 `expandedXOut` 的有效行数 |
| `expertCapacity` | int64 | (0, NUM_ROWS] | 每个专家最多处理的 token 数（DropPad 场景生效） |
| `expertNum` | int64 | [0, 10240]，key_value 模式上限 5120 | 总专家数 |
| `dropPadMode` | int64 | 0 / 1 | 0=Dropless，1=DropPad |
| `expertTokensNumType` | int64 | 0 / 1 / 2 | 直方图模式：cumsum / count / key_value |
| `expertTokensNumFlag` | bool | true/false | 是否输出 `expertTokensCountOrCumsumOut` |
| `quantMode` | int64 | -1, 0, 1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 16, 17（A5 全集）；-1, 0, 1（A3） | 量化模式 |
| `activeExpertRangeOptional` | int64 array | `[expertStart, expertEnd]`，左闭右开 | 活跃专家范围；DropPad 下必须 `[0, expertNum]` |
| `rowIdxType` | int64 | 0 / 1 | expandedRowIdxOut 索引类型：0=gather，1=scatter |

### 4.3 输出 Tensor

| 名称 | shape | dtype | 含义 |
|------|-------|-------|------|
| `expandedXOut` | Dropless: `[NUM_ROWS*K, H]` 或 `[min(activeNum, NUM_ROWS*K), H]`<br>DropPad: `[expertNum, expertCapacity, H]` | 同 x 或量化后类型 | 按 expertIdx 扩展后的特征矩阵 |
| `expandedRowIdxOut` | `(NUM_ROWS*K,)` | INT32 | x 与 expandedXOut 的索引映射；前 `availableIdxNum` 个元素有效 |
| `expertTokensCountOrCumsumOut` | count/cumsum: `(expertEnd-expertStart,)`<br>key_value: `(expertNum, 2)` | INT64 | 各专家处理 token 数的统计 |
| `expandedScaleOut` | 量化场景：expandedXOut shape 去掉最后一维的乘积<br>MXFP8/MXFP4/FP8 PerBlock 等有特定 shape | FP32/FP8_E8M0 | 量化过程中的 scale 中间值 |

## 5. 计算公式（按执行顺序）

### 步骤 1：排序

对 `expertIdx` 做 key-value sort，得到 `sortedExpertIdx` 与原序号 `sortedRowIdx`：

$$
sortedExpertIdx,\ sortedRowIdx = keyValueSort(expertIdx,\ rowIdx)
$$

### 步骤 2：索引映射

依据 `rowIdxType` 生成 `expandedRowIdxOut`：

- **scatter 索引（rowIdxType=1）**：直接输出原序号

  $$
  expandedRowIdxOut[i] = sortedRowIdx[i]
  $$

- **gather 索引（rowIdxType=0）**：输出逆映射，无效位置填 -1

  $$
  expandedRowIdxOut[sortedRowIdx[i]] = i
  $$

### 步骤 3：直方图统计

在 `sortedExpertIdx` 上按 `expertTokensNumType` 模式统计：

| expertTokensNumType | 模式 | 含义 |
|:-:|:-:|------|
| 0 | cumsum | 排序后各专家处理 token 数的前缀和 |
| 1 | count | 排序后各专家处理 token 的单独计数 |
| 2 | key_value | shape `[expertNum, 2]`：每个专家 + 该专家处理非零 token 数累计值 |

$$
expertTokensCountOrCumsumOut[i] = Histogram(sortedExpertIdx)
$$

### 步骤 4：量化（可选）

`quantMode != -1` 时，对 `x` 做量化。量化类型由 `quantMode` 决定，主要分类：

| 类别 | quantMode | 公式 |
|------|:-:|------|
| 静态量化 | 0 | $quant = round(x \cdot scale + offset)$ |
| 动态量化（无 scale 输入） | 1 | $dynScale = row\_max(\|x\|) / 127$；$quant = round(x / dynScale)$ |
| 动态量化（有 scale 输入） | 1 | $dynScale = row\_max(\|x \cdot scale\|) / 127$；$quant = round(x / dynScale)$ |
| INT4 动态量化 | 13 | 对称量化范围 [-8, 7]，分母为 7，H 维每两个 INT4 打包为 1 字节 |
| MXFP8 / MXFP4 / FP8 PerGroup / PerBlock / HIF8 | 2,3,4,5,6,7,8,9,11,12,14,15,16,17 | 各类低精度量化（仅 A5 支持） |

### 步骤 5：特征搬运（Gather/Scatter）

按 `activeExpertRangeOptional` 决定搬运方式：

- **全专家范围（`expertStart=0 && expertEnd=expertNum`）**：scatter 搬运
- **部分专家范围**：gather 搬运

DropPad 模式下，每个专家处理 token 数对齐到 `expertCapacity`：超出 drop，不足 pad 0。

**非量化场景**：

$$
expandedXOut[i] = x[scatterRowIdx[i] / K] \quad \text{(scatter)}
$$

$$
expandedXOut[gatherRowIdx[i]] = x[i / K] \quad \text{(gather)}
$$

**量化场景**：将 `x` 替换为 `quantResult` 即可。

### 步骤 6：有效元素计数

$$
availableIdxNum = |\{x \in expertIdx \mid expert\_start \le x < expert\_end\}|
$$

即 `expertIdx` 中落在 `activeExpertRangeOptional` 范围内的元素总数。`expandedRowIdxOut` 仅前 `availableIdxNum` 个元素有效。

## 6. 直方图输出模式详解

| DropPadMode | expertsTokensNumFlag | expertTokensNumType | 输出说明 |
|:-:|:-:|:-:|------|
| 0 | true | 0 | cumsum 模式：排序后各专家处理 token 数的前缀和直方图 |
| 0 | true | 1 | count 模式：各专家处理 token 的单独计数直方图 |
| 0 | true | 2 | key_value 模式：shape `[expertNum, 2]` |
| 1 | true | 1 | count 模式（DropPad 下仅支持 count） |
| 不开启 | false | 不开启 | 不输出 `expertTokensCountOrCumsumOut` |

## 7. 约束说明

### 7.1 通用约束

- 确定性计算：默认确定性实现。
- 不支持输入为 inf / -inf / nan。

### 7.2 DropPad 模式额外约束

`dropPadMode=1` 时：

- `rowIdxType` 仅支持 0（gather 索引）。
- `activeExpertRangeOptional` 必须为 `[0, expertNum]`。
- `expertTokensNumType` 仅支持 1（count 模式）。
- A3：`quantMode` 仅支持 -1（非量化）。
- A5（950PR/950DT）：`quantMode` 仅支持 -1，数据类型仅支持 FP16/BF16/FP32/INT8/HIFLOAT8。

### 7.3 空 tensor 约束

- `x.dim0 == 0`（即 `NUM_ROWS=0`）时：
  - `dropPadMode` 必须为 0。
  - 进入空 tensor 模板。
  - `expandedXOut / expandedRowIdxOut / expandedScaleOut` 返回空 tensor。
  - `expertTokensCountOrCumsumOut` 返回全 0 tensor。
- A5 额外处理 `cols==0 且 n*k>0` 的子场景：仍需初始化 `expandedRowIdx` 与部分模式的 `expandedScale`。

### 7.4 数据类型约束

- `expertIdx` 必须为 INT32。
- `expandedRowIdxOut` 必须为 INT32。
- `expertTokensCountOrCumsumOut` 必须为 INT64。
- `INT8 x` 不支持量化（A3 与 A5 共有约束）。
- 静态量化场景下 `scale / offset` 必须为 FLOAT。

## 8. 性能模板准入条件

> 仅 Atlas A2/A3 系列支持性能模板；950PR/950DT 走 A5 全载 + 通用模板。

| 性能模板 | 准入条件 |
|---------|---------|
| **低时延性能模板** | x=`(1,7168)` BF16、expertIdx=`(1,8)`、scale=`(256,7168)`、属性 `expert_range=[0,256]`、`quantMode=1`、`expertTokensNumType=2`、`expertNum=256` |
| **大 batch 性能模板** | `NUM_ROWS ∈ [384, 8192]`、`K=8`、属性 `expertNum=256`、`expertEnd-expertStart ≤ 32`、`quantMode=-1`、`rowIdxType=1`、`expertTokensNumType=1` |
| **全载性能模板** | `dropPadMode=0`（A5 非量化还支持 `dropPadMode=1`，此时 `quantMode=-1`、`rowIdxType=0`、`expert_range=[0,expertNum]`）；shape 较小、操作间多核同步占比较高时启用，搬入+排序+计算在同一 kernel 完成 |

## 9. 数据流总览

```
                    ┌──────────────────┐
                    │     expertIdx     │
                    │   shape (N, K)    │
                    └────────┬──────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   keyValueSort    │
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      sortedExpertIdx  sortedRowIdx   （可选）Gather
                              │           反映射
                              ▼
                    ┌──────────────────┐
                    │ expandedRowIdxOut │
                    │   shape (N*K,)    │
                    └──────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼
      ┌──────────────┐ ┌──────────────┐
      │  Histogram    │ │  RowIdxGather │
      │ (cumsum/count │ │ /SrcToDst     │
      │  /key_value)  │ │               │
      └──────┬───────┘ └──────┬────────┘
             │                │
             ▼                ▼
   expertTokensCount     x ─(可选 quant)─► expandedXOut
   OrCumsumOut                              │
                                            ▼
                                    expandedScaleOut
                                    (量化场景)
```

## 10. 典型调用流程

```cpp
// 1. 构造输入/输出 aclTensor
aclTensor *x = ...; aclTensor *expertIdx = ...;
aclTensor *scale = ...; aclTensor *offset = ...;
aclTensor *expandedXOut = ...; aclTensor *expandedRowIdxOut = ...;
aclTensor *expertTokensCountOut = ...; aclTensor *expandedScaleOut = ...;

// 2. 构造属性
int64_t activeNum = 12, expertCapacity = 4, expertNum = 256;
int64_t dropPadMode = 0, expertTokensNumType = 1;
bool expertTokensNumFlag = true;
int64_t quantMode = -1, rowIdxType = 1;
aclIntArray *activeExpertRange = aclCreateIntArray({0, 4});

// 3. 第一段：获取 workspaceSize
uint64_t workspaceSize;
aclOpExecutor *executor;
aclnnMoeInitRoutingV3GetWorkspaceSize(
    x, expertIdx, scale, offset,
    activeNum, expertCapacity, expertNum,
    dropPadMode, expertTokensNumType, expertTokensNumFlag,
    quantMode, activeExpertRange, rowIdxType,
    expandedXOut, expandedRowIdxOut,
    expertTokensCountOut, expandedScaleOut,
    &workspaceSize, &executor);

// 4. 申请 device workspace
void *workspaceAddr;
aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

// 5. 第二段：执行
aclnnMoeInitRoutingV3(workspaceAddr, workspaceSize, executor, stream);

// 6. 同步
aclrtSynchronizeStream(stream);
```

完整示例参见 `aclnnMoeInitRoutingV3.md` 调用示例章节。

## 11. 错误码

| 错误码 | 含义 | 触发场景 |
|--------|------|---------|
| 161001 (ACLNN_ERR_PARAM_NULLPTR) | 计算输入/输出是空指针 | 必选输入/输出为 null |
| 161002 (ACLNN_ERR_PARAM_NULLPTR) | 数据类型不在支持范围 | dtype 不在允许列表 |
| 561002 (ACLNN_ERR_INNER_TILING_ERROR) | shape/属性不在支持范围 | shape 维度错误、属性越界、组合约束冲突 |

## 12. 与 V2 接口的差异

V3 相对 V2 的功能演进：

| 项 | V2 | V3 |
|----|----|----|
| 量化模式 | 仅非量化 | 静态量化 + 动态量化（V3）+ 11 种低精度量化（A5） |
| 直方图输出 | `expertTokensBeforeCapacityFlag` (bool) | `expertsTokensNumFlag` (bool) + `expertTokensNumType` (int)，支持 cumsum/count/key_value 三种模式 |
| DropPad 输出 | `expertTokensBeforeCapacityOut` | `expertTokensCountOrCumsumOut`（语义更通用） |
| 兼容性 | — | V3 兼容 V2 原有输出模式 |
