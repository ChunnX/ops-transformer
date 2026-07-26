# FlashAttention Score Grad 算子 Swizzle 优化分析

## 1. 概述

本文档分析 `FlashAttentionScoreGradKernel` 及其基类 `FlashAttentionScoreGradKernelBase` 中 Swizzle 优化的实现细节。Swizzle 是一种**分块空间重排策略**，通过改变基本的遍历顺序，提升 L1/L0C 缓存的命中率，减少 Cube（矩阵乘）与 Vector（向量计算）之间的数据搬运开销。

### 1.1 核心思想

传统的分核策略按 `B → N2 → G → S1_outer → S2_outer` 顺序线性遍历，相邻两次迭代在 S1 或 S2 维度上连续推进。这导致：
- 同一行（S1 块）的 Q/Dy 数据需要反复从 GM 加载到 L1
- 同一列（S2 块）的 K/V 数据也需要反复加载

Swizzle 优化将 S1×S2 空间划分为更大的 **Swizzle Block**（由多个基本块组成），在 Swizzle Block 内部以连续块方式遍历，使得：
- Q/Dy 数据在 L1 中被相邻的多个 S2 块复用
- K/V 数据在 L1 中被相邻的多个 S1 块复用

### 1.2 适用范围

```cpp
constexpr static uint32_t IS_NORMAL_SWIZZLE = !IS_TND && CUBE_BASEM == CUBE_BASEN && SPLIT_AXIS == BN2GS1S2 &&
                                              DETER_SPARSE_TYPE == NO_DETER;
```

| 条件 | 说明 |
|------|------|
| `!IS_TND` | 非 TND（变长序列）格式 |
| `CUBE_BASEM == CUBE_BASEN` | Cube 基本块 M == N（方形分块） |
| `SPLIT_AXIS == BN2GS1S2` | 分轴策略为 BN2GS1S2 |
| `DETER_SPARSE_TYPE == NO_DETER` | 非确定性计算模式 |

此外还有 TND 格式专用的 Swizzle（`IS_TND_SWIZZLE`），支持 Dense/Casual/Band 三种稀疏模式。

---

## 2. Swizzle 启用条件与配置

### 2.1 启用入口

```cpp
// SetSwizzleConstInfo() 中
if constexpr (IS_NORMAL_SWIZZLE) {
    if (!tilingData->s1s2BNGS1S2BaseParams.isSplitByBlockIdx) {
        return;  // 未启用按 blockIdx 分核时，不使用 swizzle
    }
    // ... 判断 enableDenseSwizzle / enableCasualSwizzle
}
```

Swizzle 仅在 `isSplitByBlockIdx == true` 时启用，即 Tiling 阶段已经按核分配了独立的 block 范围。

### 2.2 Dense Swizzle 启用条件

```cpp
enableDenseSwizzle = (!IS_ATTEN_MASK || sparseType == DENSE) &&
                     s1Size > M_SWIZZLE_SIZE &&
                     s2Size > N_SWIZZLE_SIZE;
```

| 条件 | 说明 |
|------|------|
| 无 Attention Mask 或 Dense 类型 | 稀疏模式为全密集 |
| `s1Size > M_SWIZZLE_SIZE` | S1 维度足够大，Swizzle 有收益 |
| `s2Size > N_SWIZZLE_SIZE` | S2 维度足够大，Swizzle 有收益 |

### 2.3 Casual Swizzle 启用条件

```cpp
enableCasualSwizzle = IS_ATTEN_MASK &&
                      sparseType == CASUAL &&
                      s1Size == s2Size &&
                      s1Size > M_SWIZZLE_SIZE;
```

| 条件 | 说明 |
|------|------|
| 启用了 Attention Mask | 需要 mask 处理 |
| 稀疏类型为 Casual（下三角） | 因果注意力 |
| `s1Size == s2Size` | S1 与 S2 等长 |
| `s1Size > M_SWIZZLE_SIZE` | S1 维度足够大 |

---

## 3. Swizzle 空间划分

### 3.1 基本块与 Swizzle Block

```
Swizzle Block 结构示意 (Dense 模式):

  S2 维度
  ┌─────────────────────────────────────────────┐
  │  Swizzle Block[0,0]  │  Swizzle Block[0,1]  │ ...  ← N_SWIZZLE_BLOCK_SIZE 个基本块宽
  │  ┌───┬───┬───┐       │  ┌───┬───┬───┐       │
  S1│  │   │   │   │       │  │   │   │   │       │
  维│  ├───┼───┼───┤       │  ├───┼───┼───┤       │
  度│  │   │   │   │  ...  │  │   │   │   │  ...  │ ← M_SWIZZLE_BLOCK_SIZE 个基本块高
  │  ├───┼───┼───┤       │  ├───┼───┼───┤       │
  │  │   │   │   │       │  │   │   │   │       │
  │  └───┴───┴───┘       │  └───┴───┴───┘       │
  ├──────────────────────┼──────────────────────┤
  │  Swizzle Block[1,0]  │  Swizzle Block[1,1]  │
  │  ...                 │  ...                 │
  └─────────────────────────────────────────────┘

  每个基本块: CUBE_BASEM × CUBE_BASEN (如 128×128 或 256×256)
  每个 Swizzle Block: M_SWIZZLE_BLOCK_SIZE × N_SWIZZLE_BLOCK_SIZE 个基本块
```

### 3.2 关键尺寸定义

```cpp
constexpr static uint32_t M_SWIZZLE_BLOCK_SIZE = M_SWIZZLE_SIZE / CUBE_BASEM;  // S1方向基本块数
constexpr static uint32_t N_SWIZZLE_BLOCK_SIZE = N_SWIZZLE_SIZE / CUBE_BASEN;  // S2方向基本块数
```

- `M_SWIZZLE_SIZE` / `N_SWIZZLE_SIZE`：Swizzle Block 的绝对大小（元素数）
- `M_SWIZZLE_BLOCK_SIZE` / `N_SWIZZLE_BLOCK_SIZE`：Swizzle Block 内基本块的数量

### 3.3 Swizzle Block 的数量计算

```cpp
constInfo.mSwizzleBlockNum = constInfo.s1Outer / M_SWIZZLE_BLOCK_SIZE;       // S1方向Swizzle Block数
constInfo.mSwizzleBlockNumTail = constInfo.s1Outer % M_SWIZZLE_BLOCK_SIZE;   // S1方向尾块
constInfo.nSwizzleBlockNum = constInfo.s2Outer / N_SWIZZLE_BLOCK_SIZE;       // S2方向Swizzle Block数
constInfo.nSwizzleBlockNumTail = constInfo.s2Outer % N_SWIZZLE_BLOCK_SIZE;   // S2方向尾块
```

---

## 4. Continuous Block（连续块数）配置

### 4.1 计算逻辑

```cpp
uint32_t maxContinuousBlockNum =
    (s1Size < MIN_SWIZZLE_S1) ? MAX_CONTINUOUS_BLOCK_NUM
                              : (s1Size / MIN_SWIZZLE_S1) * BASE_SWIZZLE_BLOCK_NUM;

if (enableDenseSwizzle || enableCasualSwizzle) {
    maxContinuousBlockNum = SWIZZLE_CONTINUOUS_BLOCK_NUM;
}

constInfo.continuousBlockNum = min(maxValidBBLen, maxContinuousBlockNum);
```

`continuousBlockNum` 表示每个核在一次连续分配中获取的基本块数量。Swizzle 启用后使用固定的 `SWIZZLE_CONTINUOUS_BLOCK_NUM`，确保在同一 Swizzle Block 内连续遍历。

### 4.2 每轮总块数

```cpp
uint64_t allCoreOnceBlockNum = constInfo.aicCoreNum * constInfo.continuousBlockNum;
```

所有核一轮联合处理的基本块总数。

---

## 5. Dense Swizzle 详细实现

### 5.1 四区域轮次计算

Dense Swizzle 将 Swizzle Block 划分为四个区域，分别计算各区域每核需处理的轮数：

```
  ┌─────────────────────┬─────────────────────┐
  │  LeftUp             │  RightUp            │
  │  (M_SWIZZLE × N_SWIZZLE) │ (M_SWIZZLE × nTail) │
  │  完整 Swizzle Block │  S2方向尾块          │
  ├─────────────────────┼─────────────────────┤
  │  LeftDown           │  RightDown          │
  │  (mTail × N_SWIZZLE)│  (mTail × nTail)    │
  │  S1方向尾块          │  双尾块             │
  └─────────────────────┴─────────────────────┘
```

```cpp
// 左上区域：M_SWIZZLE_BLOCK_SIZE × N_SWIZZLE_BLOCK_SIZE 个基本块
uint64_t c1Tmp = (M_SWIZZLE_BLOCK_SIZE * N_SWIZZLE_BLOCK_SIZE) % allCoreOnceBlockNum;
uint64_t c1 = min(c1Tmp, continuousBlockNum);
constInfo.leftUpTotalRound =
    (M_SWIZZLE_BLOCK_SIZE * N_SWIZZLE_BLOCK_SIZE / allCoreOnceBlockNum) * continuousBlockNum + c1;

// 左下区域：mSwizzleBlockNumTail × N_SWIZZLE_BLOCK_SIZE
// 右上区域：M_SWIZZLE_BLOCK_SIZE × nSwizzleBlockNumTail
// 右下区域：mSwizzleBlockNumTail × nSwizzleBlockNumTail
// （类似计算方式）
```

### 5.2 聚合轮次

```cpp
// 左半部分单列总轮数 = 左上 × 完整行数 + 左下
constInfo.leftSingleColTotalRound =
    leftUpTotalRound * mSwizzleBlockNum + leftDownTotalRound;

// 左半部分总轮数 = 单列 × 列数
constInfo.leftTotalRound = leftSingleColTotalRound * nSwizzleBlockNum;

// 每个 batch 总轮数 = 左半部分 + 右上 + 右下
constInfo.batchTotalRound = leftTotalRound +
                            rightUpTotalRound * mSwizzleBlockNum +
                            rightDownTotalRound;
```

### 5.3 索引计算 (`GetNextValidIdxForSwizzleDense`)

```cpp
int64_t bIdx = swizzleLoopIdx / batchTotalRound;   // 当前 batch/head 索引
loopIdx = swizzleLoopIdx % batchTotalRound;         // batch 内的轮次

if (loopIdx < leftTotalRound) {
    // 左半部分
    globalS2Idx = loopIdx / leftSingleColTotalRound;       // Swizzle Block 列索引
    int64_t tmp = loopIdx % leftSingleColTotalRound;
    globalS1Idx = tmp / leftUpTotalRound;                   // Swizzle Block 行索引
    int64_t rTmp = tmp - globalS1Idx * leftUpTotalRound;    // 行内偏移

    if (globalS1Idx == mSwizzleBlockNum) {
        // 尾块行
        isEnd = GetLocalS1S2Idx<true>(rTmp, mSwizzleBlockNumTail, N_SWIZZLE_BLOCK_SIZE, ...);
    } else {
        // 完整行
        isEnd = GetLocalS1S2Idx<true>(rTmp, M_SWIZZLE_BLOCK_SIZE, N_SWIZZLE_BLOCK_SIZE, ...);
    }
} else {
    // 右半部分（S2 尾块列）
    loopIdx -= leftTotalRound;
    globalS2Idx = nSwizzleBlockNum;
    // ... 类似逻辑
}

// 组合全局索引
int64_t s1Idx = M_SWIZZLE_BLOCK_SIZE * globalS1Idx + localS1Idx;
int64_t s2Idx = N_SWIZZLE_BLOCK_SIZE * globalS2Idx + localS2Idx;
```

### 5.4 局部索引计算 (`GetLocalS1S2Idx`)

```cpp
// Dense 模式
int64_t r1 = loopIdx / continuousBlockNum;   // 组索引
int64_t r2 = loopIdx % continuousBlockNum;   // 组内偏移

int64_t globalIdx = aicCoreNum * continuousBlockNum * r1 +
                    cBlockIdx * continuousBlockNum + r2;

if (globalIdx >= m * n) {
    return true;  // 超出范围，跳过
}
localS1Idx = globalIdx % m;   // 行索引
localS2Idx = globalIdx / m;   // 列索引
```

**遍历顺序示意**（以 2 核、continuousBlockNum=2 为例）：

```
  基本块在 Swizzle Block 内的遍历顺序 (Dense):
  ┌────┬────┬────┐
  │ 0  │ 2  │ 4  │   ← 核0处理 0,1 → 核0处理 2,3 → 核0处理 4,5
  ├────┼────┼────┤
  │ 1  │ 3  │ 5  │   ← 核1处理 0,1 → 核1处理 2,3 → 核1处理 4,5
  ├────┼────┼────┤
  │ 6  │ 8  │ 10 │   ← 下一轮
  │ 7  │ 9  │ 11 │
  └────┴────┴────┘
  S1方向↓  S2方向→
```

---

## 6. Casual Swizzle 详细实现

### 6.1 三角形区域划分

Casual 模式下，只有下三角（含对角线）的基本块需要计算：

```
  Casual Swizzle Block 示意 (m = n = 4):
  S2 →
  ┌────┬────┬────┬────┐
  │ ✓  │    │    │    │  ← 第0行: 只有 s2Idx <= s1Idx 有效
  ├────┼────┼────┼────┤
  │ ✓  │ ✓  │    │    │  ← 第1行
  ├────┼────┼────┼────┤
  │ ✓  │ ✓  │ ✓  │    │  ← 第2行
  ├────┼────┼────┼────┤
  │ ✓  │ ✓  │ ✓  │ ✓  │  ← 第3行
  └────┴────┴────┴────┘
  S1↓

  三角形块数 = m * (m + 1) / 2
```

### 6.2 三角块数计算

```cpp
uint64_t triangleBlockNum = M_SWIZZLE_BLOCK_SIZE * (M_SWIZZLE_BLOCK_SIZE + 1) >> 1;
// 即 m*(m+1)/2
```

### 6.3 四区域轮次计算

```cpp
// 左上：完整三角形
constInfo.leftUpTotalRound = triangleBlockNum / allCoreOnceBlockNum * continuousBlockNum + c1;

// 左下：完整矩形 (非三角区域)
constInfo.leftDownTotalRound =
    (M_SWIZZLE_BLOCK_SIZE * M_SWIZZLE_BLOCK_SIZE / allCoreOnceBlockNum) * continuousBlockNum + c2;

// 右上：尾块三角形
// 右下：尾块三角形
```

### 6.4 聚合轮次

```cpp
// 左半部分 = (左上 + 右上) × 完整行数 + (m-1)*m/2 × 左下
constInfo.leftTotalRound =
    (leftUpTotalRound + rightUpTotalRound) * mSwizzleBlockNum +
    ((mSwizzleBlockNum - 1) * mSwizzleBlockNum >> 1) * leftDownTotalRound;

// 总轮数 = 左半部分 + 右下三角形
constInfo.batchTotalRound = leftTotalRound + rightDownTotalRound;
```

### 6.5 索引计算 (`GetNextValidIdxForSwizzleCasual`)

Casual 模式需要将全局索引映射到三角形区域中的 (s1, s2) 位置：

```cpp
if (loopIdx < leftUpTotalRound) {
    // 对角线上的三角形块
    globalS1Idx = globalS2Idx;
    isEnd = GetLocalS1S2Idx<false>(loopIdx, M_SWIZZLE_BLOCK_SIZE, M_SWIZZLE_BLOCK_SIZE, ...);
} else if (leftUpTotalRound <= loopIdx < leftUpTotalRound + leftDownTotalRound * (m-1-globalS2Idx)) {
    // 对角线下方的完整矩形块
    loopIdx -= leftUpTotalRound;
    globalS1Idx = globalS2Idx + loopIdx / leftDownTotalRound + 1;
    loopIdx = loopIdx % leftDownTotalRound;
    isEnd = GetLocalS1S2Idx<true>(loopIdx, M_SWIZZLE_BLOCK_SIZE, M_SWIZZLE_BLOCK_SIZE, ...);
} else {
    // 尾块行
    // ...
}
```

### 6.6 三角形局部索引 (`GetLocalS1S2Idx<false>`)

```cpp
// Casual 模式：将线性索引映射到下三角区域
// 三角形前 k 行有 k*(k+1)/2 个元素
// 给定 globalIdx，求 (s1, s2) 使得 s2*(2m-s2-1)/2 <= globalIdx < (s2+1)*(2m-s2-2)/2

int64_t twoM_minus_1 = NUM_TWO * m - 1;
localS2Idx = Ceil(
    twoM_minus_1 - sqrt(twoM_minus_1 * twoM_minus_1 + MULTIPLY_COEF * (m - 1 - globalIdx)),
    NUM_TWO
);
localS1Idx = globalIdx - (twoM_minus_1 - localS2Idx) * localS2Idx / NUM_TWO;
```

**公式推导**：

设三角形有 `m` 行，第 `k` 行（从0开始）有 `m-k` 个元素。前 `k+1` 行共 `(k+1)*(2m-k)/2` 个元素。

给定 `globalIdx`，求最大的 `s2Idx` 使得前 `s2Idx` 行的元素数 `≤ globalIdx`：

```
前 s2Idx 行元素数 = s2Idx * (2m - s2Idx) / 2
s2Idx * (2m - s2Idx) / 2 ≤ globalIdx
解方程: s2Idx = ⌈(2m-1 - √((2m-1)² - 8(m-1-globalIdx))) / 2⌉
s1Idx = globalIdx - s2Idx * (2m - s2Idx - 1) / 2
```

---

## 7. TND Swizzle 详细实现

TND（变长序列）格式有三种 Swizzle 模式，由 `IS_TND_SWIZZLE` 控制。

### 7.1 TND Dense Swizzle (`GetNextValidIdxForTndSwizzleDense`)

```cpp
int64_t delta = loopIdx - tndS2BlockPrefixSum[bIdx] + deltaCnt;
if (delta < 0) {
    deltaCnt += (-delta);  // 修正跨 batch 的偏移
    delta = 0;
}
int64_t s1Idx = delta % s1OuterTmp;
int64_t s2IdxTmp = delta / s1OuterTmp * aicCoreNum + cBlockIdx;

// 解码 N1 索引和 S2 索引
int64_t n1Idx = s2IdxTmp / s2OuterTmp;
int64_t s2Idx = s2IdxTmp - s2OuterTmp * n1Idx;
```

**关键设计**：
- `deltaCnt` 用于修正跨 batch 边界时的索引偏移
- S2 方向按核数分核，每个核处理 S2 方向的 `s2OuterTmp / aicCoreNum` 个块
- S1 方向在核内顺序遍历

### 7.2 TND Casual Swizzle (`GetNextValidIdxForTndSwizzleCasual`)

```cpp
// 处理无效列场景
if (sparseMode == LEFT_UP_CAUSAL && s1Outer < s2Outer) {
    s2Outer = s1Outer;
} else if (sparseMode == RIGHT_DOWN_CAUSAL && s1Outer > s2Outer) {
    s1Outer = s2Outer;
}

// 三角形 → 矩形变换
if (sparseMode == RIGHT_DOWN_CAUSAL) {
    s2OuterTmp = (s2Outer << 1) - s1Outer + 1;
} else {
    s1OuterTmp = (s1Outer << 1) - s2Outer + 1;
}
```

**三角形到矩形的变换逻辑**：

```
  原始 Casual 三角形          变换后的矩形
  ┌──┬──┬──┬──┐             ┌──┬──┬──┬──┬──┬──┐
  │✓ │  │  │  │             │✓ │✓ │✓ │✓ │  │  │  ← 第0行映射
  ├──┼──┼──┼──┤             ├──┼──┼──┼──┼──┼──┤
  │✓ │✓ │  │  │    →        │✓ │✓ │✓ │  │  │  │  ← 第1行映射
  ├──┼──┼──┼──┤             ├──┼──┼──┼──┼──┼──┤
  │✓ │✓ │✓ │  │             │✓ │✓ │  │  │  │  │  ← 第2行映射
  ├──┼──┼──┼──┤             ├──┼──┼──┼──┼──┼──┤
  │✓ │✓ │✓ │✓ │             │✓ │  │  │  │  │  │  ← 第3行映射
  └──┴──┴──┴──┘             └──┴──┴──┴──┴──┴──┘
  s1Outer=4, s2Outer=4       s1OuterTmp=4, s2OuterTmp=7 (= 2*4 - 4 + 1)
```

变换后，三角形被展开为矩形，相邻的 N 头可以拼接：

```cpp
if (s2Idx >= s1Idx + s2Outer - s1Outer + 1) {
    // 超出三角形边界，需要翻转到下一个 N 头
    n1Idx = (n1Idx << 1) + 1;
    s2Idx = (s2Outer << 1) - s1Outer - s2Idx;
    s1Idx = s1Idx + s2Idx - s2Outer + s1Outer;
} else {
    n1Idx = (n1Idx << 1);
}
```

### 7.3 TND Band Swizzle (`GetNextValidIdxForTndSwizzleBand`)

Band 模式处理带状稀疏（preTokens / nextTokens 约束），是三种 TND Swizzle 中最复杂的。

#### 7.3.1 块数计算

```cpp
int64_t oriPreTokenBlock = (preTokens + CUBE_BASEM - 1) / CUBE_BASEM + 1;
int64_t oriNextTokenBlock = (nextTokens + CUBE_BASEN - 1) / CUBE_BASEN + 1;
```

将 token 级别的 pre/next 约束转换为块级别的约束。

#### 7.3.2 有效区域划分

```
  Band 模式有效区域示意 (preTokenBlock=3, nextTokenBlock=3):
  S2 →
  ┌───────────────────────────────┐
  │     递增区      等宽区    递减区  │
  │  ┌──┐         ┌──┬──┬──┐  ┌──┐│
  │  │✓ │         │✓ │✓ │✓ │  │✓ ││  ← s1Idx=0: s2 ∈ [0, nextTokenBlock-1]
  │  └──┘         └──┴──┴──┘  └──┘│
  │  ┌──┬──┐      ┌──┬──┬──┐  ┌──┐│
  S1│  │✓ │✓ │      │✓ │✓ │✓ │  │✓ ││  ← s1Idx=1: s2 ∈ [0, nextTokenBlock]
  1↓  └──┴──┘      └──┴──┴──┘  └──┘│
  │  ┌──┬──┬──┐   ┌──┬──┬──┐  ┌──┐│
  │  │✓ │✓ │✓ │   │✓ │✓ │✓ │  │✓ ││  ← s1Idx=2: s2 ∈ [0, nextTokenBlock+1]
  │  └──┴──┴──┘   └──┴──┴──┘  └──┘│
  │                ...              │
  └───────────────────────────────┘

  s2IncreaseValidBlockLen = nextTokenBlock - 1        (递增区宽度)
  s2EqualValidBlockLen = min(s2Outer - nextTokenBlock + 1,
                             s1Outer + 2 - preTokenBlock - nextTokenBlock)  (等宽区宽度)
  s2DecreaseValidBlockLen = max(0, min(preTokenBlock + s2Outer - s1Outer - 1,
                                       preTokenBlock + nextTokenBlock - 2))  (递减区宽度)
  s2MaxValidBlockLen = preTokenBlock + nextTokenBlock - 1  (最大列高)
```

#### 7.3.3 斜向拼接优化

当递减区域存在且 S2 外宽大于 S1 外宽时，采用斜向拼接：

```cpp
if (s2DecreaseValidBlockLen > 0 && s2OuterTmp - s1OuterTmp >= 1) {
    // 斜向拼接：将递减区与下一个 N 头的递增区拼接
    s2IdxTmp2 = delta % s2MaxValidBlockLen;
    s1IdxTmp = s2IdxTmp2 + s2IdxTmp1 + 1 - nextTokenBlock;

    if (s1IdxTmp < 0) {
        // 回绕到上一个 N 头的末尾
        n1Idx = n2G - 1;
        s2Idx = s1IdxTmp + s1OuterTmp;
        s1Idx = s2IdxTmp1 + s1OuterTmp;
    } else {
        n1Idx = s1IdxTmp / s1OuterTmp;
        s1Idx = s1IdxTmp % s1OuterTmp;
        s2Idx = s1Idx + nextTokenBlock - 1 - s2IdxTmp2;
    }
}
```

#### 7.3.4 有效性检查

```cpp
// Band 模式需要检查 s1/s2 是否在有效范围内
if (s1Idx < 0 || s2Idx < 0 || s1Idx >= s1Outer || s2Idx >= s2Outer ||
    s1Idx - s2Idx >= preTokenBlock || s2Idx - s1Idx >= nextTokenBlock) {
    continue;  // 跳过无效块
} else {
    break;  // 找到有效块
}
```

---

## 8. 非 Swizzle 模式的回退逻辑

当 Swizzle 未启用时，使用顺序遍历方式：

```cpp
// GetNextValidIdx() 中的 else 分支
nextValidBlockInnerIdx = blockInnerIdx;
while (!IsValid(runInfo, taskId, nextValidBlockInnerIdx)) {
    if (nextValidBlockInnerIdx >= blockEnds[cBlockIdx]) {
        return -1;
    }
    nextValidBlockInnerIdx++;
}
```

`GetNextValidIdxForSwizzle` 中的 else 分支也提供了非 Swizzle 的回退：

```cpp
// 非 Dense/Casual Swizzle 的默认遍历
int64_t blockGroupIdx = loopIdx / continuousBlockNum;
int64_t blockGroupInnerIdx = loopIdx % continuousBlockNum;

int64_t globalIdx = aicCoreNum * continuousBlockNum * blockGroupIdx +
                    cBlockIdx * continuousBlockNum + blockGroupInnerIdx;

// 根据 sparseType 计算 s1Idx 和 s2Idx
if (sparseType == DENSE) {
    s2Idx = gDimTail / s1Outer;
    s1Idx = gDimTail % s1Outer;
} else if (sparseType == CASUAL) {
    // 使用求根公式计算三角形位置
    sqrt_delta = sqrt(...);
    s2Idx = Ceil(...);
    s1Idx = gDimTail - ...;
} else {
    // BAND 模式：梯形/矩形/三角形组合计算
    // ...
}
```

---

## 9. BAND 模式（非 TND）的 Swizzle

### 9.1 有效块计算 (`UpdateMNPQ`)

```cpp
void UpdateMNPQ(int64_t &m, int64_t &n, int64_t &p, int64_t &q,
                int64_t &mGap, int64_t &nGap)
{
    m = s1Outer;  // S1方向总块数
    n = s2Outer;  // S2方向总块数
    p = Ceil(s1Token, CUBE_BASEM) + 1;  // preToken对应的块数
    q = Ceil(s2Token, CUBE_BASEN) + 1;  // nextToken对应的块数

    // 处理负 token 的情况
    if (p < 0) {
        // preToken 为负，需要调整
        actualN = n + p - 1;
        actualP = 1;
        actualQ = p + q - 1;
        nGap = -p + 1;
    } else if (q < 0) {
        // nextToken 为负
        actualM = m + q - 1;
        actualP = p + q - 1;
        mGap = -q + 1;
    } else {
        // 正常情况
        actualM = m;
        actualN = n;
        actualP = p;
        actualQ = q;
    }
}
```

### 9.2 有效块区域

```cpp
if (p + q > m) {
    // 梯形 + 矩形
    s2FirstWidth = m - p;                          // 梯形宽度
    s2SecondWidth = p + q - m - 1;                 // 矩形宽度
    s2FirstBlockNum = (p + m - 1) * s2FirstWidth / 2;  // 梯形块数
    s2SecondBlockNum = m * s2SecondWidth;           // 矩形块数
} else {
    // 梯形 + 矩形 + 梯形
    s2FirstWidth = q - 1;
    s2SecondWidth = min(n - q + 1, m + 2 - p - q);
    s2FirstBlockNum = ((p-1)*2 + q) * (q-1) / 2;   // 第一个梯形
    s2SecondBlockNum = (p + q - 1) * s2SecondWidth; // 矩形
}
```

### 9.3 BAND 索引计算

在 `GetNextValidIdxForSwizzle` 的 BAND 分支中：

```cpp
if (gDimTail < s2FirstBlockNum) {
    // 第一个梯形区域：使用求根公式
    tmp = (firstValidColBlockNum << 1) - 1;
    s2Idx = (sqrt(tmp * tmp + (gDimTail << 3)) - tmp) >> 1;
    s1Idx = gDimTail - ((tmp + s2Idx) * s2Idx >> 1);
} else if (s2FirstBlockNum <= gDimTail < s2FirstBlockNum + s2SecondBlockNum) {
    // 矩形区域：直接除法
    gDimTail -= s2FirstBlockNum;
    s2Idx = gDimTail / actualS1Outer + s2FirstWidth;
    s1Idx = gDimTail % actualS1Outer;
} else {
    // 第二个梯形区域：使用求根公式
    // ...
}

// 加上 gap 偏移
s1Idx += mGap;
s2Idx += nGap;
```

---

## 10. Swizzle 在主循环中的使用

### 10.1 Process 主循环

```cpp
void Process() {
    // ...
    while (true) {
        // ...
        if (!this->isLastLoop) {
            nextValidBlockInnerIdx = this->GetNextValidIdx(
                runInfos[(taskId + 1) & 1], taskId + 1,
                blockInnerIdx + 1, curLoopIdx + 1);
            // GetNextValidIdx 内部会根据 IS_NORMAL_SWIZZLE / IS_TND_SWIZZLE
            // 选择不同的 Swizzle 策略
        }
        // ... 执行 MM1, MM2, Vec 计算
        taskId++;
        blockInnerIdx = nextValidBlockInnerIdx;
    }
}
```

### 10.2 调用链

```
Process()
  └─ GetNextValidIdx()
       ├─ IS_TND_SWIZZLE?
       │    ├─ GetNextValidIdxForTndSwizzleDense()   // TND Dense
       │    ├─ GetNextValidIdxForTndSwizzleCasual()  // TND Casual
       │    └─ GetNextValidIdxForTndSwizzleBand()    // TND Band
       │
       └─ IS_NORMAL_SWIZZLE?
            ├─ isSplitByBlockIdx?
            │    └─ GetNextValidIdxForSwizzle()
            │         ├─ enableDenseSwizzle?
            │         │    └─ GetNextValidIdxForSwizzleDense()
            │         ├─ enableCasualSwizzle?
            │         │    └─ GetNextValidIdxForSwizzleCasual()
            │         └─ else: 默认顺序遍历 (Dense/Casual/Band)
            │
            └─ else: 顺序遍历 + IsValid() 过滤
```

---

## 11. Swizzle 优化的性能分析

### 11.1 数据复用分析

**非 Swizzle 模式**：
- 每个 S1 块的 Q/Dy 只被相邻的 1 个 S2 块使用
- 每个 S2 块的 K/V 只被相邻的 1 个 S1 块使用
- L1 缓存复用率低

**Swizzle 模式**：
- 在一个 Swizzle Block 内（M_SWIZZLE_BLOCK_SIZE × N_SWIZZLE_BLOCK_SIZE），Q/Dy 被 N_SWIZZLE_BLOCK_SIZE 个 S2 块复用
- K/V 被 M_SWIZZLE_BLOCK_SIZE 个 S1 块复用
- 复用率提升 M_SWIZZLE_BLOCK_SIZE × N_SWIZZLE_BLOCK_SIZE 倍

### 11.2 数据搬运量对比

```
假设: s1Outer = s2Outer = S, CUBE_BASEM = CUBE_BASEN = B
      M_SWIZZLE_BLOCK_SIZE = N_SWIZZLE_BLOCK_SIZE = W
      Q/Dy/K/V 每个基本块大小 = D 个元素

非 Swizzle:
  Q/Dy 加载次数 = S × S (每个 (s1, s2) 对加载一次 Q 和 Dy)
  K/V 加载次数 = S × S

Swizzle (Dense):
  Q/Dy 加载次数 = S × (S / W)  (每行只需加载 S/W 次，每次复用 W 个 S2 块)
  K/V 加载次数 = (S / W) × S  (每列只需加载 S/W 次，每次复用 W 个 S1 块)
  
  搬运量减少倍数 ≈ W
```

### 11.3 Casual 模式的额外收益

Casual Swizzle 避免了对上三角无效块的计算：
- 三角形有效块数 = S × (S+1) / 2
- 无效块跳过比例 ≈ 50%

---

## 12. 关键数据结构

### 12.1 FagConstInfo 中的 Swizzle 字段

```cpp
struct FagConstInfo {
    // Swizzle Block 数量
    uint32_t mSwizzleBlockNum;        // S1方向Swizzle Block数
    uint32_t mSwizzleBlockNumTail;    // S1方向尾块
    uint32_t nSwizzleBlockNum;        // S2方向Swizzle Block数
    uint32_t nSwizzleBlockNumTail;    // S2方向尾块

    // 四区域轮次
    uint64_t leftUpTotalRound;
    uint64_t leftDownTotalRound;
    uint64_t rightUpTotalRound;
    uint64_t rightDownTotalRound;

    // 聚合轮次
    uint64_t leftSingleColTotalRound;
    uint64_t leftTotalRound;
    uint64_t batchTotalRound;

    // 连续块配置
    uint32_t continuousBlockNum;

    // Band 模式
    int64_t actualS1Outer;
    int64_t actualS2Outer;
    int64_t firstValidColBlockNum;
    int64_t firstValidRowBlockNum;
    int64_t mGap;
    int64_t nGap;
    int64_t s2FirstWidth;
    int64_t s2SecondWidth;
    int64_t s2FirstBlockNum;
    int64_t s2SecondBlockNum;
};
```

### 12.2 运行时状态

```cpp
// 在 KernelBase 中
int64_t swizzleLoopIdx = 0;           // 全局 Swizzle 循环索引
bool enableDenseSwizzle = false;      // Dense Swizzle 启用标志
bool enableCasualSwizzle = false;     // Casual Swizzle 启用标志

// TND Swizzle 专用
int64_t deltaCnt{};                   // 跨 batch 偏移修正
int64_t bandLoopIdx{};                // Band 模式循环索引
```

---

## 13. 总结

### 13.1 Swizzle 优化策略总结

| 策略 | 适用场景 | 核心方法 | 索引计算方式 |
|------|----------|----------|-------------|
| Dense Swizzle | 无 Mask / Dense Mask | 矩形分块 + 四区域划分 | 除法取模 |
| Casual Swizzle | 下三角 Casual Mask | 三角形 → 矩形变换 + 求根公式 | 求根公式 |
| TND Dense Swizzle | TND 格式 Dense | 按 S2 分核 + S1 顺序遍历 | delta 计算 + 除法 |
| TND Casual Swizzle | TND 格式 Casual | 三角形展开 + N 头拼接 | 三角变换 + 翻转 |
| TND Band Swizzle | TND 格式 Band | 三区域划分 + 斜向拼接 | 多段函数 + 求根公式 |
| BAND Swizzle (非TND) | 非 TND Band | 梯形+矩形分块 | 求根公式 + 除法 |

### 13.2 设计要点

1. **空间局部性优先**：Swizzle 改变遍历顺序，优先在局部区域内完成所有计算，最大化 L1/L0C 缓存复用。

2. **三角形变换**：Casual 模式通过将三角形展开为矩形，避免了无效块的遍历，同时保持了均匀的分核。

3. **斜向拼接**：Band 模式通过将递减区与下一个 N 头的递增区拼接，进一步减少无效遍历。

4. **尾块处理**：四区域划分（左上/左下/右上/右下）确保 S1/S2 方向的尾块被正确处理。

5. **跨 batch 修正**：TND 模式通过 `deltaCnt` 修正跨 batch 边界时的索引偏移，保证连续性。

6. **回退机制**：当 Swizzle 不满足启用条件时，自动回退到顺序遍历 + `IsValid()` 过滤的方式，保证正确性。