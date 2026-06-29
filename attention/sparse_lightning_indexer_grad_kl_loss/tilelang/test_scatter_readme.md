# T.tile.atomic_add Scatter 验证测试 - 实际结果

## 测试结论（已完成）

**关键发现：TileLang不支持动态索引scatter**

```
error: T.tile.atomic_add V1 only supports local tensor -> GM atomic add.
Cannot deduce atomic_add extents.
```

这意味着：
- `T.tile.atomic_add`只能使用**静态切片表达式**（编译时可计算的索引）
- **不支持动态索引**（从buffer读取的索引值）
- 只支持 `local tensor (L0C/UB) -> GM` 的atomic_add

## 实际运行结果

| 测试 | 结果 | 说明 |
|------|------|------|
| Test 1 (static L0C) | 待验证 | 使用L0C→GM，类似0622.py |
| Test 2 (dynamic region) | **FAILED** | 编译错误：无法推导extent |
| Test 3 (dynamic buffer) | **FAILED** | 结果全0 |
| Test 3b (fixed K-parallel) | 待验证 | 类似0622.py Matmul4 |
| Test 5 (simulate scatter) | **FAILED** | 不支持动态scatter |
| Test 5b (partial output) | 待验证 | Fallback方案 |

**核心结论**：动态scatter在TileLang kernel内**不可实现**。

## 优化方案分析

### ❌ 方案A：完全移入Kernel（不可行）

```
原计划：
  C scope: Matmul + atomic_add scatter
问题：
  T.tile.atomic_add不支持动态索引
结论：
  无法在kernel内实现scatter_add_
```

### ✅ 方案B：Partial Kernelization（推荐）

**设计**：
```
Kernel新增部分：
  C scope: Matmul5 = reluGrad^T @ QueryIndex -> GM output
  Output: matmul_partial[B,S1,K,DQueryIndex]
  
Python简化：
  scatter_add_(d_key_index, sparse_indices, matmul_partial)
  （不再需要torch.bmm）
```

**收益估算**：
```
当前Python后处理（48%时间）：
  - torch.bmm: ~33M FLOPs（主要部分）
  - scatter_add: 随机写入（次要部分）

优化后：
  - Kernel: Matmul5（移出bmm）
  - Python: 仅scatter_add
  - 预期Python占比: 48% -> 15-20%
  - 总耗时: 1.4ms -> 0.85-0.9ms
```

### 方案C：Block-based Scatter（需分析sparse_indices分布）

如果`sparse_indices`有规律分布（如连续块），可尝试：
```
预处理：
  分析sparse_indices，转换为静态block映射
Kernel：
  多核K-parallel scatter到固定block
问题：
  当前sparse_indices是任意索引，可能无规律
```

### 方案D：T.tile.gather反向（复杂）

```
原理：
  scatter(dst, src, idx) ≈ dst = gather(full_src, reverse_idx)
问题：
  需要预计算reverse mapping
  reverse mapping可能很复杂
可行性：
  低，不建议优先考虑
```

## 最终推荐：方案B（Partial Kernelization）

**实施步骤**：
1. 在0622.py中新增Matmul5 kernel部分
2. 输出matmul_partial到GM
3. Python层仅执行scatter_add
4. 预期性能提升30-40%

**与官方AscendC对比**：
```
官方实现（单Kernel）：
  V7阶段: SetAtomicAdd + DataCopy scatter（支持动态索引）

TileLang限制：
  不支持动态索引的atomic_add

差异原因：
  AscendC是底层API，可直接操作地址
  TileLang是高层DSL，需要编译器推导extent
```