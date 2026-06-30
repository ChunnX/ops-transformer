# TileLang-Ascend 使用限制总结

> 基于 sparse_lightning_indexer_grad_kl_loss 算子重写过程中发现的问题。

## 目录

- [一、真实的技术限制](#一真实的技术限制)
- [二、对算子实现的阻碍](#二对算子实现的阻碍)
- [三、已验证可用的特性](#三已验证可用的特性)
- [四、已排除的伪问题](#四已排除的伪问题)
- [五、与官方 AscendC 的差距](#五与官方-ascendc-的差距)

---

## 一、真实的技术限制

### 1. `T.gemm_v0(transpose_A=True)` 疑似精度 bug

`transpose_B=True` 路径正常工作，但 `transpose_A=True` 在数学正确的情况下产出 ~100% 相对误差（max_diff ~14.5）。

```python
# 数学上等价的两种写法：
# ❌ 误差 ~100%
T.gemm_v0(RG(64,128), QI(64,128), C(128,128), transpose_A=True)
# ✅ 精度正常
RGT = alloc((128,64)); T.copy(..., RGT)  # 需预先转置
T.gemm_v0(RGT(128,64), QI(64,128), C(128,128), transpose_B=False)
```

**影响**：无法利用 `transpose_A` 省去 host 侧转置操作。

### 2. `T.tile.atomic_add` 不支持 kernel block 变量的符号切片

```python
n_idx_k = cid % n_num          # 这是一个 kernel block ID (TIR symbolic)
k_s = n_idx_k * 128

# ❌ 报错: int() argument must be a string, not 'Sub'
T.tile.atomic_add(tensor[..., k_s:k_s+128, ...], buf)

# ✅ loop 变量可以
for nidx_c in T.serial(4):
    nidx_s = nidx_c * 16
    T.tile.atomic_add(tensor[..., nidx_s:nidx_s+16, ...], buf)  # OK
```

`atomic_add` 的实现尝试对切片边界做 `int()` 转换，TIR 表达式 `n_idx_k * n_block` 无法转换。

**绕过**：改用 `T.copy` 直接覆写（仅当各 core 写入不重叠区域时可行）。

### 3. `T.copy` 不同 shape 间是 flat memcpy，不做矩阵转置

```python
A = alloc((16, 128), fp16)   # 2048 元素
B = alloc((128, 16), fp16)   # 2048 元素，同大小
T.copy(A, B)                  # 不报错，但数据按 row-major flat copy
# B[p][q] = A 的 flat 第 p*16+q 个元素
# 不等于 A[q][p] (真正的转置)
```

这是 DMA 的正常行为，不是 bug。但意味着 kernel 内无法零成本做矩阵转置。

**影响**：如需转置，必须 host 侧预 transpose（`relu_grad.transpose(-1,-2)`）或 kernel 内逐元素 copy。

### 4. V scope 中执行 gather 失败

尝试在 C1 scope 前加 V scope 做 DMA gather（将 key/key_index/key_rope 按 sparse_indices 索引读取），触发：

```
Illegal instruction, unaligned UUB addresses
```

> **注**：V→C cross_flag 本身是可用的（0629 的 V1→C2 正常工作），问题出在 gather 指令（随机 DMA 读取）的上下文限制。

**影响**：gather 被迫留在 host 侧（~0.04ms overhead）。

### 5. 无 DMA scatter 原语

官方 AscendC 用 V2 scope 的硬件 DMA scatter 做 `scatter_add_`。TileLang 没有对应原语。

```python
# Python 侧（快）：底层调 CANN 优化的 scatter DMA
d_key_index.scatter_add_(dim, indices, values)

# TileLang 内只能这样（极慢）：
for i in T.serial(2048):              # 2048 次串行
    idx = sparse_indices[i]
    if idx >= 0:
        T.tile.atomic_add(d_key_index[idx], val)  # 随机地址，无 burst
```

**影响**：dKeyIndex scatter 被迫留在 host 侧（~0.05ms overhead）。

### 6. CV scope 严格串行，无法交叉流水

```python
C1 → [flag] → V1 → [flag] → C2 → [flag] → V3
# 严格串行：V1 必须等 C1 全部完成才启动
```

官方 AscendC 用 14-stage 流水线让 C 和 V 同时处理不同 K-block，互相掩盖延迟。TileLang 的 `T.Scope` 模型不支持这种交叉调度。

**影响**：V1 scope 的 1024 次标量迭代 + 48 次 K-block 遍历完全暴露在关键路径上，是 **0629 性能的最大瓶颈**。

### 7. `T.tile.broadcast` 不支持 1D→2D 扩展

```python
# ❌ 不可行
T.tile.broadcast(vec_1d(N,), mat_2d(M, N), ...)
```

导致无法用 `T.tile` 向量化指令替代标量循环。

**影响**：V1 中 `for i in range(16): T.tile.mul(...)` 必须保持标量形式。

---

## 二、对算子实现的阻碍

| 想做的事 | 受哪个限制 | 被迫的替代方案 | 性能代价 |
|-----------|-----------|--------------|---------|
| gather 进 kernel | 限制4 (V scope gather失败) | Host 侧 `torch.gather` | ~0.04ms |
| dKeyIndex bmm 进 kernel | 限制1 (transpose_A bug) + 限制3 | Host 侧 `torch.bmm` (FP16) | ~0.15ms |
| dKeyIndex scatter 进 kernel | 限制5 (无 scatter 原语) | Host 侧 `scatter_add_` | ~0.05ms |
| V1 标量循环向量化 | 限制7 (broadcast 不支持) | 保持标量循环 | ~0.1ms |
| CV 流水掩盖 V1 | 限制6 (scope 串行) | 无法绕过 | **最大损失** |

---

## 三、已验证可用的特性

| 特性 | 说明 | 使用位置 |
|------|------|---------|
| C→V cross_flag | Cube 设 flag, Vector 等待 | 0629: C1→V1, C2→V3 |
| V→C cross_flag | Vector 设 flag, Cube 等待 | 0629: V1→C2 |
| `T.Pipelined` | C1 MM1 5-pass 流水掩盖 DMA | 0629 C1 scope |
| `T.tile.atomic_add` | Cube→GM 累加 (loop 变量切片) | 0629: dQueryIndex |
| `T.copy(L0C → GM)` | Cube 结果直接写 Global | 0629: bmm1Res |
| `T.gemm_v0(transpose_B=True)` | B 矩阵转置乘法 | 0629: MM1/MM2/MM4 |
| UB→UB 2D copy | Vector scope 内数据搬运 | 0629 V1 scope |

---

## 四、已排除的伪问题

以下是在开发过程中遇到、但最终确认**不是** TileLang 限制的问题：

| 条目 | 结论 |
|------|------|
| V→C cross_flag 不可用 | **不成立**。0629 的 V1→C2 (`set_flag("V",12)` → `wait_flag(12)`) 正常使用 |
| `out_idx` 负索引漂移 | 自己代码问题。插入新参数后 `[-5,-4,-3,-2]` 指向变化。**改用绝对索引如 `[15,16,17,18]` 即可** |
| `out_idx` 分配未初始化内存 | 自己代码问题。JIT 首次编译可能分配新 tensor（未初始化），`atomic_add` 到垃圾值上。**用 `T.copy` 替代 `atomic_add` 或做 warmup 可解** |
| `T.copy` 不同 shape 不做转置 | 正常行为，底层 DMA memcpy，不是 bug |
| `T.tile.broadcast` 完全不可用 | 1D→1D broadcast 可用，只是不支持 1D→2D |

---

## 五、与官方 AscendC 的差距

| 维度 | 官方 AscendC | TileLang (我们的实现) |
|------|:---:|:---:|
| 阶段数 | 14-stage 流水线 | 4-scope 串行 |
| Gather | V0 scope (设备侧 DMA gather) | Host 侧 torch.gather |
| Scatter | V2 scope (设备侧 DMA scatter) | Host 侧 scatter_add_ |
| CV 掩盖 | ✅ C 和 V 交叉执行 | ❌ 严格串行 |
| dKeyIndex bmm | C scope (Cube unit) | Host 侧 torch.bmm |
| 向量化 | 全向量化 | 大量标量循环 |

这些差距解释了 ~3x 的性能差异（官方 ~0.31ms vs 0629 ~0.9ms）。
