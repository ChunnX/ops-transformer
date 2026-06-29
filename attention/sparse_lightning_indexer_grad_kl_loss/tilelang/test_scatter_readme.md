# T.tile.atomic_add Scatter 验证测试

## 目的

验证 TileLang 的 `T.tile.atomic_add` 是否支持动态索引 scatter，这对优化 0622.py 的 Python 后处理瓶颈至关重要。

## 当前瓶颈分析

根据 `planB_analysis.md`，当前性能分布：

| 阶段 | 耗时 | 占比 |
|------|-----------|------|
| gather | 0.044ms | **3.1%** - 不是瓶颈 |
| kernel1+2 | 0.686ms | 48.8% |
| **Python后处理** | 0.677ms | **48.1%** - 主要瓶颈 |
| 总计 | ~1.4ms | |

Python后处理包含：
```python
# _compute_d_key_index - 占主要时间
relu_grad_t @ query_index  # torch.bmm (~33M FLOPs)
scatter_add_(sparse_idx, ...)  # 随机索引写入
```

## 测试场景

### Test 1: Static atomic_add (baseline)
- 应该 PASS
- 验证基本功能

### Test 2-3: Dynamic atomic_add
- **关键测试**
- 如果 PASS：可以将 scatter 移入 kernel
- 如果 FAIL：需要替代方案

### Test 3b: Multi-core fixed scatter
- 类似 0622.py 当前实现
- 每个 core 有固定的 scatter 目标

### Test 5: 模拟 0622.py Matmul + Scatter
- **最关键测试**
- 直接验证优化方案的可行性

### Test 5b: Partial output (fallback)
- 如果动态 scatter 不支持
- kernel 只计算 matmul，Python 做 scatter
- 减少 Python 工作量但仍需 scatter

## 运行方法

```bash
# 在 NPU 环境中运行
cd F:\WorkSpace\Code\ops-transformer
python attention/sparse_lightning_indexer_grad_kl_loss/tilelang/test_atomic_add_scatter.py
```

## 预期结果分析

### 如果 Test 2-3 和 Test 5 PASS

**最优方案**：将 `_compute_d_key_index` 完全移入 kernel

```
新 Kernel 设计：
  C scope: Matmul5 = reluGrad^T @ QueryIndex → L0C
  V scope: T.tile.atomic_add scatter to GM using sparse_indices

收益：
  - Python 后处理从 48% → ~10% (只剩 dWeight/Loss sum)
  - 总耗时预期: 0.7ms → 0.4ms (接近官方 0.35ms)
```

### 如果 Test 2-3 和 Test 5 FAIL

**替代方案 A**：Partial kernelization

```
Kernel 设计：
  C scope: Matmul5 = reluGrad^T @ QueryIndex → GM partial output
  Python: scatter_add_(sparse_idx, matmul_result)

收益：
  - Python 后处理从 48% → ~30% (bmm 移出，scatter 保留)
  - 总耗时预期: 1.4ms → 1.0ms
```

**替代方案 B**：预计算静态 scatter 映射

```
Python preprocess:
  1. 分析 sparse_indices 的分布模式
  2. 将动态 scatter 转换为固定块的 scatter

Kernel:
  每个 core scatter 到预设的 GM block

问题：
  - 需要额外的预处理开销
  - sparse_indices 分布不规律时难以实现
```

**替代方案 C**：T.tile.gather 反向实现

```
原理：scatter(dst, src, idx) = gather(dst_dst, reverse_map)

问题：
  - 需要预计算 reverse_map
  - reverse_map 可能复杂
```

## 与 AscendC 版本对比

官方 AscendC kernel 的 ScatterAdd 实现 (sparse_lightning_indexer_grad_kl_loss_vector_block.h:906-1013):

```cpp
// V7 阶段: ScatterAdd
SetAtomicAdd<T>();  // 开启原子加
for (int32_t kLoopIdx = 0; kLoopIdx < kLoopTimes; ++kLoopIdx) {
    // 从 GM 加载 matmul 结果到 UB
    DataCopy(scatterUb, srcGm[kLoopIdx * kSplitSize * dSizeQueryIndex], ...);
    
    for (int32_t topKIdx = 0; topKIdx < topKLoopTimes; ++topKIdx) {
        // 动态获取 scatter 目标位置
        realS2Idx1 = topKGm.GetValue(runInfo.topkGmBaseOffset + s2GmOffset);
        keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit);
        
        // Scatter 到 GM (使用动态 keyOffset1)
        DataCopy(scatterAddGm[keyOffset1 * dSizeQueryIndex], srcTmpUb1, ...);
    }
}
SetAtomicNone();
```

**关键差异**：
- AscendC 直接使用 `DataCopy` + `SetAtomicAdd` 实现 scatter
- TileLang 需要验证 `T.tile.atomic_add` 是否支持相同功能

## 下一步

1. 运行验证脚本，确认哪些测试 PASS/FAIL
2. 根据结果选择优化方案
3. 如果支持动态 scatter，设计新 Kernel 架构
4. 如果不支持，评估替代方案的性能收益