# FlashAttention Tiling L2 Cache 优化详设

## 1. 优化目标

FlashAttention 算子在 Cube 和 Vector 之间反复搬运中间数据。当单次交换数据量超过 L2 Cache 容量时，数据无法被缓存，产生大量 GM 重复访问。

本优化在 Ascend910B4.1（L2 Cache=168MB）平台上，通过缩小 tiling 基本块，使 Cube/Vector 单次交换数据量落入 L2 Cache，提升 cache hit rate。

## 2. 生效条件

三个条件**同时满足**时生效：

| 条件 | 判断 | 原因 |
|------|------|------|
| 硬件 L2 Cache | `l2CacheSize == B4_1_L2_CACHESIZE`（168MB） | 容量足够容纳缩小后的 block，B4（96MB）收益不明显 |
| 数据类型 | `inputDtype == DT_BF16 \|\| DT_FLOAT16` | BF16/FP16 单元素 2 字节，FP32 数据量翻倍超出收益区间 |
| 输入排布 | `LAYOUT_SBH \|\| LAYOUT_BNSD \|\| LAYOUT_BSND` | TND（VarLen）tiling 路径独立，不在本次优化范围 |

任一条件不满足时，保持原值不变。

## 3. 实现方案

### 3.1 新增常量

`flash_attention_score_tiling_general.cpp`（arch22），紧跟 `B4_L2_CACHESIZE` 之后：

```cpp
static constexpr uint64_t B4_1_L2_CACHESIZE = static_cast<uint64_t>(168) * 1024 * 1024;
```

### 3.2 修改点 1：s1BasicBlock

**所属类**：`FlashAttentionScoreTilingS1Bn2gs1`
**函数**：`CalcS1S2BasicBlock()`

```cpp
void CalcS1S2BasicBlock([[maybe_unused]] const BufferNum &bufferNum) override
{
    auto optimCond = (l2CacheSize == B4_1_L2_CACHESIZE) &&
                     (inputDtype == ge::DT_BF16 || inputDtype == ge::DT_FLOAT16) &&
                     (tilingData->inputParams.get_layoutType() == LAYOUT_SBH ||
                      tilingData->inputParams.get_layoutType() == LAYOUT_BNSD ||
                      tilingData->inputParams.get_layoutType() == LAYOUT_BSND);
    s1BasicBlock = optimCond ? std::min(64L, alignedS1) : std::min(128L, alignedS1);
    s2BasicBlock = std::min(128L, alignedS2);
}
```

**变化**：`s1BasicBlock` 从 128 降至 64，BMM1 单次计算量减半，Cube 产出中间结果更易驻留 L2 Cache。

### 3.3 修改点 2：nRatio

**所属类**：`FlashAttentionScoreTilingS1s2Bn2gs1`
**函数**：`CalcNRatio()`

```cpp
void CalcNRatio() override
{
    auto optimCond = (l2CacheSize == B4_1_L2_CACHESIZE) &&
                     (inputDtype == ge::DT_BF16 || inputDtype == ge::DT_FLOAT16) &&
                     (tilingData->inputParams.get_layoutType() == LAYOUT_SBH ||
                      tilingData->inputParams.get_layoutType() == LAYOUT_BNSD ||
                      tilingData->inputParams.get_layoutType() == LAYOUT_BSND);
    nRatio = optimCond ? 4L : 8L;
    // 后续 s2Size 特殊值修正逻辑保持不变
    ...
}
```

**变化**：`nRatio` 从 8 降至 4，减少单次迭代中需在 L2 Cache 中保留的中间数据量。

## 4. 数据量分析

以 `BF16, N=1, D=128, S1=256, S2=256` 为例：

| | 原始（128+8） | 优化后（64+4） |
|--|--------------|---------------|
| bmm1Out 单 block | 128×128×4B = 64KB | 64×128×4B = 32KB |
| 单次迭代总量 | 64KB × 8 = 512KB | 32KB × 4 = 128KB |
| L2 Cache 占比（168MB） | ~0.3% | ~0.08% |

数据量缩小 4 倍，多轮迭代中间结果可同时在 L2 Cache 中保留，Vector 读取 softmax 输入时 cache hit rate 提升。

## 5. 风险评估

- **正确性**：s1BasicBlock/nRatio 仅影响 tiling 分块策略，不影响计算结果
- **兼容性**：通过 l2CacheSize/dtype/layout 三重条件判断，旧平台和非目标场景行为不变
- **trade-off**：更小的 block 增加 Cube 调度次数，但 L2 cache 命中带来的访存延迟降低远大于调度开销

## 6. 影响的 Tiling 类

```
FlashAttentionScoreTilingBase
├── FlashAttentionScoreTilingS1Bn2gs1      ← 修改 s1BasicBlock
├── FlashAttentionScoreTilingS1s2Bn2gs1    ← 修改 nRatio
├── FlashAttentionScoreTilingS1s2Bn2gs1SameAB  未修改
├── FlashAttentionVarLenScoreTiling            未修改
└── FlashAttentionScoreTilingDropMask          未修改
```
