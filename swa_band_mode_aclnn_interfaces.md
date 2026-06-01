# Flash Attention Score BAND模式（滑窗注意力/SWA）ACLNN接口完整文档

## 目录
1. [概述](#概述)
2. [接口版本总览](#接口版本总览)
3. [核心参数说明](#核心参数说明)
4. [BAND模式参数详解](#band模式参数详解)
5. [接口详细说明](#接口详细说明)
6. [使用示例](#使用示例)
7. [注意事项](#注意事项)

---

## 概述

Flash Attention Score BAND模式（Sliding Window Attention，滑窗注意力）是一种稀疏注意力机制，通过限制每个token只关注其周围一定窗口内的token，从而降低计算复杂度。本文档详细介绍了所有支持BAND模式的ACLNN接口。

### BAND模式特点
- **计算复杂度**：O(n×w)，其中n是序列长度，w是窗口大小
- **内存占用**：相比全注意力显著降低
- **适用场景**：长序列建模，局部依赖关系较强的任务

---

## 接口版本总览

### 接口分类

| 接口类型 | 接口名称 | 支持BAND模式 | 变长序列 | RoPE支持 | 量化支持 |
|---------|---------|------------|---------|---------|---------|
| 定长序列 | aclnnFlashAttentionScore | ✓ | ✗ | ✗ | ✗ |
| 定长序列V2 | aclnnFlashAttentionScoreV2 | ✓ | ✗ | ✗ | ✗ |
| 定长序列V3 | aclnnFlashAttentionScoreV3 | ✓ | ✗ | ✗ | ✗ |
| 定长序列V4 | aclnnFlashAttentionScoreV4 | ✓ | ✗ | ✓ | ✓ |
| 量化版本 | aclnnQuantFlashAttentionScore | ✓ | ✗ | ✗ | ✓ |
| 变长序列 | aclnnFlashAttentionVarLenScore | ✓ | ✓ | ✗ | ✗ |
| 变长序列V2 | aclnnFlashAttentionVarLenScoreV2 | ✓ | ✓ | ✗ | ✗ |
| 变长序列V3 | aclnnFlashAttentionVarLenScoreV3 | ✓ | ✓ | ✓ | ✗ |
| 变长序列V4 | aclnnFlashAttentionVarLenScoreV4 | ✓ | ✓ | ✗ | ✗ |
| 变长序列V5 | aclnnFlashAttentionVarLenScoreV5 | ✓ | ✓ | ✓ | ✗ |

### 版本演进

**V1 → V2**:
- 新增pseType参数支持
- 新增qStartIdx/kvStartIdx参数

**V2 → V3**:
- 新增sink参数支持
- 支持sink token机制

**V3 → V4**:
- 新增queryRope/keyRope参数
- 新增量化反量化支持（dScaleQ/dScaleK/dScaleV）
- 新增dropout seed/offset参数
- 新增outDtype参数
- 新增softmaxOutLayout参数

---

## 核心参数说明

### 输入张量

| 参数名 | 类型 | 必需 | 说明 | 形状示例 |
|-------|------|------|------|---------|
| query | aclTensor* | 是 | Query张量 | [B, N1, S1, D] 或 [B, S1, N1, D] |
| key | aclTensor* | 是 | Key张量 | [B, N2, S2, D] 或 [B, S2, N2, D] |
| value | aclTensor* | 是 | Value张量 | [B, N2, S2, D2] 或 [B, S2, N2, D2] |
| realShiftOptional | aclTensor* | 否 | PSE编码张量 | 根据pseType决定 |
| dropMaskOptional | aclTensor* | 否 | Dropout掩码 | - |
| paddingMaskOptional | aclTensor* | 否 | 填充掩码 | [B, S1] |
| attenMaskOptional | aclTensor* | 否 | 注意力掩码 | [B, N2, G, S1, S2] 等 |
| queryRopeOptional | aclTensor* | 否 | Query RoPE嵌入 | 与query同形状 |
| keyRopeOptional | aclTensor* | 否 | Key RoPE嵌入 | 与key同形状 |
| sinkOptional | aclTensor* | 否 | Sink token张量 | - |
| dScaleQOptional | aclTensor* | 否 | Query反量化scale | - |
| dScaleKOptional | aclTensor* | 否 | Key反量化scale | - |
| dScaleVOptional | aclTensor* | 否 | Value反量化scale | - |

### 输入数组

| 参数名 | 类型 | 必需 | 说明 |
|-------|------|------|------|
| prefixOptional | aclIntArray* | 否 | Prefix长度数组 |
| actualSeqQLenOptional | aclIntArray* | 否 | 实际Query序列长度（变长） |
| actualSeqKvLenOptional | aclIntArray* | 否 | 实际KV序列长度（变长） |
| qStartIdxOptional | aclIntArray* | 否 | Query起始索引 |
| kvStartIdxOptional | aclIntArray* | 否 | KV起始索引 |

### 输出张量

| 参数名 | 类型 | 说明 |
|-------|------|------|
| softmaxMaxOut | aclTensor* | Softmax最大值输出 |
| softmaxSumOut | aclTensor* | Softmax求和输出 |
| softmaxOutOut | aclTensor* | Softmax输出 |
| attentionOutOut | aclTensor* | 注意力输出 |

### 标量参数

| 参数名 | 类型 | 说明 | BAND模式相关 |
|-------|------|------|-------------|
| scaleValue | double | 缩放因子 | 通常为1/√D |
| keepProb | double | Dropout保留概率 | - |
| preTokens | int64_t | **BAND模式核心参数** | 向前看的token数 |
| nextTokens | int64_t | **BAND模式核心参数** | 向后看的token数 |
| headNum | int64_t | 注意力头数 | - |
| inputLayout | char* | 输入布局 | "BNSD"、"BSND"、"SBHD"等 |
| innerPrecise | int64_t | 内部计算精度 | 0:高精度 1:高性能 |
| **sparseMode** | int64_t | **稀疏模式** | **4=BAND模式** |
| pseType | int64_t | PSE编码类型 | - |
| outDtype | int64_t | 输出数据类型 | - |
| softmaxOutLayout | char* | Softmax输出布局 | - |
| seed | int64_t | Dropout随机种子 | - |
| offset | int64_t | Dropout偏移量 | - |

---

## BAND模式参数详解

### sparseMode参数

**参数值**：`sparseMode = 4` (BAND模式)

**稀疏模式枚举**：
```cpp
enum SparseMode {
    NO_MASK = 0,          // 无掩码
    ALL_MASK = 1,         // 全掩码
    LEFT_UP_CAUSAL = 2,   // 左上因果（下三角）
    RIGHT_DOWN_CAUSAL = 3,// 右下因果
    BAND = 4,            // BAND模式（滑窗）
    PREFIX = 5,          // Prefix模式
    BAND_COMPRESS = 6,   // BAND压缩模式
    RIGHT_DOWN_CAUSAL_BAND = 7,    // 右下因果+BAND
    BAND_LEFT_UP_CAUSAL = 8        // BAND+左上因果
};
```

### preTokens和nextTokens参数

#### 参数含义

- **preTokens**: 向前看的token数量（向左/向上）
- **nextTokens**: 向后看的token数量（向右/向下）

#### 窗口计算

对于位置`i`的query，其有效的key范围：
```cpp
startIdx = max(0, i - preTokens)
endIdx = min(S2, i + nextTokens)
```

#### 参数组合场景

**场景1：preTokens ≥ 0, nextTokens ≥ 0**
- 标准BAND模式
- 窗口大小 = preTokens + nextTokens + 1
- 当前位置 + 前preTokens个 + 后nextTokens个

**场景2：preTokens < 0, nextTokens ≥ 0**
- 偏向右侧的窗口
- 窗口向右下偏移

**场景3：preTokens ≥ 0, nextTokens < 0**
- 偏向左侧的窗口
- 窗口向左上偏移

**场景4：preTokens < 0, nextTokens < 0**
- 非法配置，返回错误

#### S1≠S2时的坐标变换

当query序列长度(S1)与key序列长度(S2)不等时，需要进行坐标变换：

```cpp
// 从右下顶点参考系转换到左上顶点参考系
if (s1Size != s2Size) {
    preTokens_left = s1Size - s2Size + preTokens_right;
    nextTokens_left = s2Size - s1Size + nextTokens_right;
}
```

**示例**：
- S1 = 100, S2 = 150
- 用户输入：preTokens = 10, nextTokens = 20（右下顶点参考）
- 变换后：preTokens = -40, nextTokens = 70（左上顶点参考）

#### 常用配置

**对称窗口**：
```cpp
preTokens = windowSize / 2
nextTokens = windowSize / 2
```

**因果窗口**：
```cpp
preTokens = windowSize
nextTokens = 0
```

**前瞻窗口**：
```cpp
preTokens = 0
nextTokens = windowSize
```

---

## 接口详细说明

### 1. aclnnFlashAttentionScore (V1 - 定长序列)

#### 接口原型

```cpp
// 第一阶段：获取workspace大小
aclnnStatus aclnnFlashAttentionScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional,
    double scaleValue,
    double keepProb,
    int64_t preTokens,           // BAND模式参数
    int64_t nextTokens,          // BAND模式参数
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,          // 设置为4启用BAND模式
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);

// 第二阶段：执行计算
aclnnStatus aclnnFlashAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream
);
```

#### 支持的特性
- ✓ BAND模式（sparseMode=4）
- ✗ 变长序列
- ✗ RoPE
- ✗ 量化
- ✓ Prefix
- ✓ Sink（V3+）

---

### 2. aclnnFlashAttentionScoreV2 (V2 - 定长序列增强版)

#### 接口原型

```cpp
aclnnStatus aclnnFlashAttentionScoreV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional,      // 新增
    const aclIntArray *kvStartIdxOptional,    // 新增
    double scaleValue,
    double keepProb,
    int64_t preTokens,                        // BAND模式参数
    int64_t nextTokens,                       // BAND模式参数
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,                       // 设置为4启用BAND模式
    int64_t pseType,                          // 新增
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);
```

#### V2新增特性
- ✓ pseType参数（位置编码类型）
- ✓ qStartIdx/kvStartIdx（起始索引）

---

### 3. aclnnFlashAttentionScoreV3 (V3 - 定长序列+Sink)

#### 接口原型

```cpp
aclnnStatus aclnnFlashAttentionScoreV3GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *sinkOptional,              // 新增sink支持
    const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double scaleValue,
    double keepProb,
    int64_t preTokens,                         // BAND模式参数
    int64_t nextTokens,                        // BAND模式参数
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,                        // 设置为4启用BAND模式
    int64_t pseType,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);
```

#### V3新增特性
- ✓ sink token支持

---

### 4. aclnnFlashAttentionScoreV4 (V4 - 定长序列完整版)

#### 接口原型

```cpp
aclnnStatus aclnnFlashAttentionScoreV4GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *queryRopeOptional,        // 新增RoPE支持
    const aclTensor *keyRopeOptional,          // 新增RoPE支持
    const aclTensor *dScaleQOptional,          // 新增反量化
    const aclTensor *dScaleKOptional,          // 新增反量化
    const aclTensor *dScaleVOptional,          // 新增反量化
    const aclTensor *sinkOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,  // 变长序列支持
    const aclIntArray *actualSeqKvLenOptional, // 变长序列支持
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double scaleValue,
    double keepProb,
    int64_t preTokens,                         // BAND模式参数
    int64_t nextTokens,                        // BAND模式参数
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,                        // 设置为4启用BAND模式
    int64_t outDtype,                          // 新增输出类型
    int64_t pseType,
    char *softmaxOutLayout,                    // 新增softmax输出布局
    int64_t seed,                              // 新增dropout参数
    int64_t offset,                            // 新增dropout参数
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);
```

#### V4新增特性
- ✓ RoPE支持（queryRope/keyRope）
- ✓ 量化反量化（dScaleQ/K/V）
- ✓ 变长序列支持（actualSeqQLen/KvLen）
- ✓ 输出数据类型控制（outDtype）
- ✓ Softmax输出布局（softmaxOutLayout）
- ✓ Dropout随机数控制（seed/offset）

---

### 5. aclnnQuantFlashAttentionScore (量化版本)

#### 接口原型

```cpp
aclnnStatus aclnnQuantFlashAttentionScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *attenMaskOptional,
    const aclTensor *dScaleQ,                  // 反量化scale
    const aclTensor *dScaleK,                  // 反量化scale
    const aclTensor *dScaleV,                  // 反量化scale
    const aclTensor *pScale,                   // 量化scale
    double scaleValue,
    int64_t preTokens,                         // BAND模式参数
    int64_t nextTokens,                        // BAND模式参数
    int64_t headNum,
    char *inputLayout,
    int64_t sparseMode,                        // 设置为4启用BAND模式
    aclTensor *softmaxMaxOut,
    aclTensor *softmaxSumOut,
    aclTensor *softmaxOutout,
    aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);
```

#### 量化特性
- ✓ 输入量化支持（需要dScaleQ/K/V反量化）
- ✓ 输出量化支持（需要pScale量化）
- ✓ BAND模式支持

---

### 6. aclnnFlashAttentionVarLenScore (变长序列V1)

#### 接口原型

```cpp
aclnnStatus aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,   // 变长Query长度
    const aclIntArray *actualSeqKvLenOptional,  // 变长KV长度
    double scaleValue,
    double keepProb,
    int64_t preTokens,                          // BAND模式参数
    int64_t nextTokens,                         // BAND模式参数
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,                         // 设置为4启用BAND模式
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);
```

#### 变长序列特性
- ✓ actualSeqQLen/KvLen：每个样本的实际序列长度
- ✓ 支持batch内不同长度的序列
- ✓ BAND模式自适应不同长度

---

### 7. aclnnFlashAttentionVarLenScoreV2 (变长序列V2)

#### 接口原型

```cpp
aclnnStatus aclnnFlashAttentionVarLenScoreV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional,       // 新增
    const aclIntArray *kvStartIdxOptional,      // 新增
    double scaleValue,
    double keepProb,
    int64_t preTokens,                          // BAND模式参数
    int64_t nextTokens,                         // BAND模式参数
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,                         // 设置为4启用BAND模式
    int64_t pseType,                            // 新增
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);
```

---

### 8. aclnnFlashAttentionVarLenScoreV3 (变长序列V3 - RoPE)

#### 接口原型

```cpp
aclnnStatus aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *queryRope,                 // 新增RoPE
    const aclTensor *key,
    const aclTensor *keyRope,                   // 新增RoPE
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double scaleValue,
    double keepProb,
    int64_t preTokens,                          // BAND模式参数
    int64_t nextTokens,                         // BAND模式参数
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,                         // 设置为4启用BAND模式
    int64_t pseType,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);
```

---

### 9. aclnnFlashAttentionVarLenScoreV4 (变长序列V4)

#### 接口原型

```cpp
aclnnStatus aclnnFlashAttentionVarLenScoreV4GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    double scaleValue,
    double keepProb,
    int64_t preTokens,                          // BAND模式参数
    int64_t nextTokens,                         // BAND模式参数
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,                         // 设置为4启用BAND模式
    char *softmaxOutLayout,                     // 新增
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);
```

---

### 10. aclnnFlashAttentionVarLenScoreV5 (变长序列V5 - 最新版)

#### 接口原型

```cpp
// 第一阶段：获取workspace大小
aclnnStatus aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *queryRope,
    const aclTensor *key,
    const aclTensor *keyRope,
    const aclTensor *value,
    const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *sinkOptional,
    const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    double scaleValue,
    double keepProb,
    int64_t preTokens,                          // BAND模式参数
    int64_t nextTokens,                         // BAND模式参数
    int64_t headNum,
    char *inputLayout,
    int64_t innerPrecise,
    int64_t sparseMode,                         // 设置为4启用BAND模式
    int64_t pseType,
    char *softmaxOutLayout,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);

// 获取最大workspace大小（用于动态shape场景）
aclnnStatus aclnnFlashAttentionVarLenScoreV5GetMaxWorkspaceSize(
    // 参数同GetWorkspaceSize
    ...
);

// 第二阶段：执行计算
aclnnStatus aclnnFlashAttentionVarLenScoreV5(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream
);
```

#### V5特性
- ✓ 支持动态shape（GetMaxWorkspaceSize）
- ✓ 完整的RoPE支持
- ✓ Sink token支持
- ✓ 最新的性能优化

---

## 使用示例

### 示例1：基础BAND模式（定长序列）

```cpp
#include "aclnn/aclnn_base.h"
#include "aclnn_flash_attention_score.h"

// 设置BAND模式参数
int64_t sparseMode = 4;          // BAND模式
int64_t preTokens = 64;          // 向前看64个token
int64_t nextTokens = 64;         // 向后看64个token
double scaleValue = 1.0 / sqrt(128.0);  // D=128
double keepProb = 1.0;           // 不使用dropout
int64_t headNum = 32;
char inputLayout[] = "BNSD";
int64_t innerPrecise = 0;        // 高精度模式

// 准备输入张量
aclTensor *query, *key, *value;  // [B, N, S, D]
aclTensor *softmaxMaxOut, *softmaxSumOut, *softmaxOutOut, *attentionOutOut;

// 第一阶段：获取workspace大小
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus ret = aclnnFlashAttentionScoreGetWorkspaceSize(
    query, key, value, nullptr, nullptr, nullptr, nullptr,
    nullptr, scaleValue, keepProb, preTokens, nextTokens,
    headNum, inputLayout, innerPrecise, sparseMode,
    softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut,
    &workspaceSize, &executor
);

// 第二阶段：分配workspace并执行
void *workspace = nullptr;
aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

ret = aclnnFlashAttentionScore(
    workspace, workspaceSize, executor, stream
);

// 同步等待
aclrtSynchronizeStream(stream);
```

### 示例2：变长序列BAND模式

```cpp
#include "aclnn_flash_attention_score.h"

// 变长序列参数
int64_t sparseMode = 4;          // BAND模式
int64_t preTokens = 128;         // 窗口大小128
int64_t nextTokens = 0;          // 因果窗口（只看前文）
std::vector<int64_t> actualSeqQLens = {100, 150, 80, 200};  // 各样本实际长度
std::vector<int64_t> actualSeqKvLens = {120, 160, 100, 200};

// 创建aclIntArray
aclIntArray *actualSeqQLenOptional = aclCreateIntArray(
    actualSeqQLens.data(), actualSeqQLens.size()
);
aclIntArray *actualSeqKvLenOptional = aclCreateIntArray(
    actualSeqKvLens.data(), actualSeqKvLens.size()
);

// 获取workspace大小
aclnnStatus ret = aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
    query, key, value, nullptr, nullptr, nullptr, nullptr,
    nullptr, actualSeqQLenOptional, actualSeqKvLenOptional,
    scaleValue, keepProb, preTokens, nextTokens,
    headNum, inputLayout, innerPrecise, sparseMode,
    softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut,
    &workspaceSize, &executor
);

// 执行计算
ret = aclnnFlashAttentionVarLenScore(
    workspace, workspaceSize, executor, stream
);
```

### 示例3：BAND模式 + RoPE（V3接口）

```cpp
// 使用V3接口支持RoPE
int64_t sparseMode = 4;
int64_t preTokens = 32;
int64_t nextTokens = 32;
int64_t pseType = 2;  // RoPE类型

// 准备RoPE张量
aclTensor *queryRope, *keyRope;  // [B, N, S, DRoPe]

// 使用V3接口
aclnnStatus ret = aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize(
    query, queryRope, key, keyRope, value,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, actualSeqQLenOptional, actualSeqKvLenOptional,
    nullptr, nullptr,
    scaleValue, keepProb, preTokens, nextTokens,
    headNum, inputLayout, innerPrecise, sparseMode, pseType,
    softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut,
    &workspaceSize, &executor
);

ret = aclnnFlashAttentionVarLenScoreV3(
    workspace, workspaceSize, executor, stream
);
```

### 示例4：量化BAND模式

```cpp
// 量化BAND模式
int64_t sparseMode = 4;
int64_t preTokens = 64;
int64_t nextTokens = 64;

// 准备反量化scale张量
aclTensor *dScaleQ, *dScaleK, *dScaleV;  // [1]或[B,N,1,1]
aclTensor *pScale;  // 输出量化scale

// 使用量化接口
aclnnStatus ret = aclnnQuantFlashAttentionScoreGetWorkspaceSize(
    query, key, value, nullptr,
    dScaleQ, dScaleK, dScaleV, pScale,
    scaleValue, preTokens, nextTokens,
    headNum, inputLayout, sparseMode,
    softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut,
    &workspaceSize, &executor
);

ret = aclnnQuantFlashAttentionScore(
    workspace, workspaceSize, executor, stream
);
```

### 示例5：S1≠S2场景的BAND模式

```cpp
// Query序列长度 != Key序列长度
// S1 = 1024, S2 = 2048

// 用户视角（右下顶点参考）
int64_t preTokens_user = 128;   // 相对于右下顶点
int64_t nextTokens_user = 256;  // 相对于右下顶点

// 注意：接口内部会自动进行坐标变换
// 无需手动调整，直接传入用户参数即可

aclnnStatus ret = aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(
    query, queryRope, key, keyRope, value,
    nullptr, nullptr, nullptr, nullptr, sinkOptional,
    nullptr, actualSeqQLenOptional, actualSeqKvLenOptional,
    nullptr, nullptr,
    scaleValue, keepProb, preTokens_user, nextTokens_user,
    headNum, inputLayout, innerPrecise, sparseMode, pseType, softmaxOutLayout,
    softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut,
    &workspaceSize, &executor
);
```

### 示例6：不同窗口配置

```cpp
// 1. 对称窗口（最常用）
int64_t windowSize = 128;
int64_t preTokens = windowSize / 2;   // 64
int64_t nextTokens = windowSize / 2;  // 64

// 2. 因果窗口（解码器常用）
int64_t preTokens = 128;   // 只看前文
int64_t nextTokens = 0;    // 不看后文

// 3. 前瞻窗口（编码器有时使用）
int64_t preTokens = 0;     // 不看前文
int64_t nextTokens = 128;  // 只看后文

// 4. 非对称窗口
int64_t preTokens = 96;    // 前窗更大
int64_t nextTokens = 32;   // 后窗较小

// 5. 大窗口（长距离依赖）
int64_t preTokens = 512;
int64_t nextTokens = 512;
```

---

## 注意事项

### 1. BAND模式参数约束

#### preTokens和nextTokens约束
- 不能同时为负数
- 不能同时大于序列长度
- 窗口大小建议：32、64、128、256、512等2的幂次

#### sparseMode设置
```cpp
int64_t sparseMode = 4;  // 必须设置为4启用BAND模式
```

### 2. 内存对齐要求

- 序列长度建议按64或128对齐
- 窗口大小建议按16对齐
- D（head_dim）必须能被16整除

### 3. 性能优化建议

#### 选择合适的窗口大小
- 小窗口（32-64）：计算快，适合局部依赖
- 中窗口（128-256）：平衡性能和效果
- 大窗口（512+）：接近全注意力，计算量大

#### 使用innerPrecise参数
```cpp
int64_t innerPrecise = 0;  // 高精度（推荐用于小窗口）
int64_t innerPrecise = 1;  // 高性能（推荐用于大窗口）
```

#### 推荐配置
```cpp
// 小窗口场景
windowSize = 64;
innerPrecise = 0;  // 高精度

// 大窗口场景
windowSize = 512;
innerPrecise = 1;  // 高性能
```

### 4. 变长序列注意事项

#### actualSeqQLen/KvLen要求
- 数组长度等于batch size
- 值必须为单调递增（累积长度）
- 最大值不能超过张量的S维度

#### 示例
```cpp
// Batch中有4个样本，实际长度分别为100, 150, 80, 200
std::vector<int64_t> lens = {100, 150, 80, 200};
std::vector<int64_t> cumLens = {100, 250, 330, 530};  // 累积长度
```

### 5. 与其他模式组合

BAND模式可以与其他稀疏模式组合：

```cpp
// BAND + Causal
int64_t sparseMode = 8;  // BAND_LEFT_UP_CAUSAL
int64_t preTokens = 128;
int64_t nextTokens = 0;  // Causal限制nextTokens=0

// BAND + Prefix
int64_t sparseMode = 5;  // PREFIX模式
// 先计算prefix，再计算band窗口
```

### 6. 错误处理

#### 常见错误及解决

**错误1：preTokens和nextTokens同时为负**
```cpp
// 错误配置
preTokens = -10;
nextTokens = -20;

// 解决：调整参数使其至少一个非负
preTokens = 0;
nextTokens = 20;
```

**错误2：窗口过大**
```cpp
// 窗口大于序列长度
preTokens = 10000;
nextTokens = 10000;
// S1 = 1024

// 解决：调整窗口大小
preTokens = 512;
nextTokens = 512;
```

**错误3：S1≠S2时参数未转换**
```cpp
// 注意：接口内部会自动转换，用户无需手动处理
// 直接传入相对于右下顶点的参数即可
```

### 7. 接口选择建议

#### 场景映射表

| 使用场景 | 推荐接口 | 理由 |
|---------|---------|------|
| 定长序列 + 基础BAND | V1/V2 | 简单高效 |
| 定长序列 + RoPE | V4 | 完整功能 |
| 定长序列 + 量化 | Quant | 量化优化 |
| 变长序列 + 基础BAND | V1/V2 | 变长支持 |
| 变长序列 + RoPE | V3/V5 | 完整功能 |
| 动态Shape | V5 GetMaxWorkspaceSize | 动态支持 |

### 8. 性能对比

#### 计算复杂度对比

| 模式 | 计算复杂度 | 内存占用 |
|------|-----------|---------|
| 全注意力 | O(S²) | 高 |
| BAND模式 | O(S×W) | 低 |
| 因果模式 | O(S²/2) | 中 |

其中：
- S：序列长度
- W：窗口大小（W << S）

#### 性能提升示例

```cpp
// S = 4096, D = 128, N = 32

// 全注意力
compute_time_full = 4096 * 4096 = 16,777,216

// BAND模式 (W = 128)
compute_time_band = 4096 * 128 = 524,288

// 加速比
speedup = 16,777,216 / 524,288 = 32x
```

---

## 总结

本文档详细介绍了Flash Attention Score BAND模式的所有ACLNN接口，包括：

1. **10个主要接口版本**：涵盖定长/变长、基础/增强、量化等场景
2. **完整的参数说明**：特别是preTokens/nextTokens的详细解释
3. **丰富的使用示例**：涵盖各种常见使用场景
4. **详细的注意事项**：包括参数约束、性能优化、错误处理等

### 推荐使用流程

1. **确定需求**：定长/变长、是否需要RoPE、是否需要量化
2. **选择接口**：根据需求选择合适的接口版本
3. **设置参数**：配置preTokens/nextTokens确定窗口大小
4. **调用接口**：先GetWorkspaceSize，再Execute
5. **优化调优**：根据实际效果调整窗口大小和精度模式

BAND模式是处理长序列的有效手段，在保持较好性能的同时显著降低计算复杂度，适用于各种大模型推理和训练场景。
