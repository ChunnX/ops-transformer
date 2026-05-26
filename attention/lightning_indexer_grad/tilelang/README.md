# Lightning Indexer Gradient - TileLang-Ascend 实现

使用 TileLang-Ascend 实现 lightning_indexer_grad 算子，用于计算 Lightning Attention 稀疏索引操作中的梯度。

## 概述

本实现使用 TileLang DSL 定义昇腾 NPU 的硬件优化内核，遵循原始 Ascend C 实现的流水线结构。

## 数学背景

### 前向传播
```
Indices = TopK{[1]_{1×g} @ [(W@[1]) ⊙ ReLU(Q_index @ K_index^T)]}
```

### 反向传播
给定上游梯度 `Dy` (形状 `[B, S1, K]`)，计算:

**dWeights** (Vec2):
```
dW = ReduceSum(relu_in ⊙ Dy, dim=-1)
```

**dQuery** (Cube3):
```
dQ = ReluGrad_out @ KeyGather
```
其中 `ReluGrad_out = (W ⊙ Dy) ⊙ mask`，`mask = (relu_in > 0)`。

**dKey** (Cube4 + Vec3):
```
dK[i] = Σ_{(s,k): indices[s,k]=i} ReluGrad_out[s,k] @ Query[s,:]
```

## 流水线结构

### 核心任务 (每个批次，每个头组)
```
┌─────────────────────────────────────────────────────────────────┐
│                         任务流水线                                │
├─────────────────────────────────────────────────────────────────┤
│  pre1    │ GatherTopk: 基于 TopK 索引加载 Key 数据               │
│  pre2    │ (保留用于未来优化)                                     │
│  pre3    │ Cube1: Query @ KeyGather^T -> attn_local              │
│  MID     │ Vec1: 应用来自 ReLU 输入的掩码                         │
│          │ Cube2: Weights @ Dy -> relu_grad_in                   │
│          │ Vec2: ReLU 反向 + ReduceSum 计算 dWeights             │
│  end3    │ Cube3: ReluGrad @ KeyGather -> dQuery                │
│  end2    │ Cube4: ReluGrad^T @ Query -> scatter_add             │
│  end1    │ Vec3: ScatterAdd 到 dKey                              │
└─────────────────────────────────────────────────────────────────┘
```

### 流水线阶段
对于 `loopTimes` 次迭代:
- **填充阶段**: 迭代 0-2 (pre1→pre2→pre3)
- **稳定阶段**: 迭代 3 到 loopTimes-1
- **排空阶段**: 迭代 loopTimes 到 loopTimes+2

## 文件结构

```
tilelang/
├── README.md                           # 本文件
├── lightning_indexer_grad_op.py        # 主算子类
├── lightning_indexer_grad_matmul.py    # MatMul 内核 (Cube1-4)
├── lightning_indexer_grad_vector.py    # Vector 内核 (Vec1-3)
└── tests/
    └── test_lightning_indexer_grad.py  # 单元测试
```

## 内核详解

### 1. MatMul 内核 (lightning_indexer_grad_matmul.py)

#### Cube1: Query @ KeyGather^T
```python
kernel_cube1 = matmul_relu_kernel(M, D, K, block_m, block_d, block_k)
# 输入: Query (M, D), KeyGather (K, D)
# 输出: attn_local (M, K) 带 ReLU
```

#### Cube2: Weights @ Dy (广播)
```python
kernel_cube2 = matmul_broadcast_kernel(M, 1, K, block_m, 1, block_k)
# 输入: Weights (M, 1), Dy (1, K) - 广播
# 输出: relu_grad_in (M, K)
```

#### Cube3: ReluGrad @ KeyGather
```python
kernel_cube3 = matmul_dq_kernel(M, K, D, block_m, block_k, block_d)
# 输入: ReluGrad (M, K), KeyGather (K, D)
# 输出: dQuery (M, D)
```

#### Cube4: ReluGrad^T @ Query
```python
kernel_cube4 = matmul_scatter_kernel(K, M, D, block_k, block_m, block_d)
# 输入: ReluGrad^T (K, M), Query (M, D)
# 输出: scatter_add (K, D)
```

### 2. Vector 内核 (lightning_indexer_grad_vector.py)

#### Vec2: ReLU 反向
```python
kernel_relu_grad = relu_grad_kernel(M, K, block_m, block_k)
# 输入: relu_grad_in (M, K), relu_in (M, K)
# 输出: relu_grad_out = relu_grad_in * (relu_in > 0)
```

#### Vec2: ReduceSum
```python
kernel_reduce = reduce_sum_kernel(M, K, block_m, block_k)
# 输入: relu_in * Dy (M, K)
# 输出: dWeights (M,)
```

#### Vec3: ScatterAdd
```python
scatter_add_python(d_key_cpu, scatter_add_cpu, indices_cpu, n2_idx, head_dim)
# 输入: scatter_add (K, D), indices (K,)
# 输出: dKey 在索引位置累加
```

### 3. Gather 操作 (混合 CPU-NPU)

```python
key_gather = gather_topk_python(key_cpu, indices_cpu, n2_idx, head_dim)
# 输入: Key (B, S2, head_num_k, D), Indices (B, S1, K)
# 输出: KeyGather (B, S1, K, D)
```

## 使用方法

### 基本用法
```python
from lightning_indexer_grad_op import LightningIndexerGrad

# 创建算子
op = LightningIndexerGrad(
    seqlen_q=1024,
    seqlen_k=2048,
    head_dim=128,
    topk=2048,
    head_num_q=32,
    head_num_k=8,
    group_num=4,
    block_m=64,
    block_k=128,
    block_d=128,
)

# 计算梯度
d_query, d_key, d_weights = op.forward(
    query=query_tensor,      # (batch, S1, D)
    key=key_tensor,          # (batch, S2, head_num_k, D)
    weights=weights_tensor,  # (batch, S1, N1)
    dy=dy_tensor,            # (batch, S1, K)
    indices=indices_tensor,  # (batch, S1, K)
    relu_in=relu_in_tensor,  # (batch, S1, K)
)
```

### 分块大小调优

分块大小影响内存层次使用，应根据具体硬件调优:

```python
# 用于昇腾 910B (更大的 L1/L0 缓冲区)
op = LightningIndexerGrad(..., block_m=128, block_k=256, block_d=256)

# 用于昇腾 310 (更小的缓冲区)
op = LightningIndexerGrad(..., block_m=32, block_k=64, block_d=64)
```

## 内存层次

```
┌────────────────────────────────────────────────────────────┐
│                    昇腾内存布局                              │
├────────────────────────────────────────────────────────────┤
│  HBM (全局内存)                                              │
│    ├── Query, Key, Weights, Dy (输入)                       │
│    ├── dQuery, dKey, dWeights (输出)                        │
│    └── Indices, Relu_in (侧输入)                             │
│                                                              │
│  L1 缓冲区 (AI Core 本地内存)                                │
│    ├── A_L1, B_L1 (MatMul 的输入分块)                        │
│    └── ~1MB 每个 AI Core                                     │
│                                                              │
│  L0A/L0B/L0C (Cube 单元缓冲区)                               │
│    ├── L0A: 矩阵 A 输入                                      │
│    ├── L0B: 矩阵 B 输入                                      │
│    ├── L0C: 矩阵 C 累加器                                    │
│    └── ~256KB 总计                                           │
│                                                              │
│  UB (Vector 单元的统一缓冲区)                                │
│    ├── Vector 操作的输入/输出分块                            │
│    └── ~64KB 每个 AI Core                                    │
└────────────────────────────────────────────────────────────┘
```

## 性能优化

### 1. 流水线执行
```python
# 使用 T.Pipelined 实现内存和计算重叠
for k_idx in T.Pipelined(k_num, num_stages=2):
    T.copy(A[...], A_L1)    # 阶段 N: 加载
    T.gemm_v0(...)         # 阶段 N+1: 计算
```

### 2. 多核并行
```python
# 每个核处理独立的分块
with T.Kernel(m_num * n_num, threads=VEC_NUM, is_npu=True) as (cid):
    m_idx = cid // n_num
    n_idx = cid % n_num
```

### 3. 双缓冲
```python
# T.Pipelined 中隐式使用 num_stages=2
# 允许并发加载和计算
```

## 与 Ascend C 实现对比

| 方面 | Ascend C | TileLang |
|------|----------|-----------|
| 内存管理 | 手动 GM->L1->L0 | 使用 T.alloc_* 自动 |
| 内核启动 | 多任务流水线 | 每个操作单内核 |
| Gather/Scatter | 自定义 CPU+AICPU | Python + NPU 混合 |
| 流水线 | 手动任务依赖 | T.Pipelined DSL |
| 代码行数 | ~2000+ | ~500 |

## 测试

```bash
# 运行单元测试
cd tilelang
python -m pytest tests/ -v

# 运行特定测试
python lightning_indexer_grad_op.py
```

## 已知限制

1. **Gather/Scatter**: 当前实现使用基于 CPU 的 Python 进行 gather 和 scatter 操作。这是因为 TileLang 专注于计算密集型操作。后续工作可集成 AICPU 以获得更好性能。

2. **动态形状**: 分块大小必须能整除张量维度。对于不能整除的大小可能需要填充。

3. **原子操作**: ScatterAdd 使用 CPU 以获得确定性结果。可添加 NPU 原子加以获得更好性能。

## 参考资料

- [TileLang-Ascend 编程指南](../../docs/TileLang-Ascend%20Programming%20Guide.md)
- [Lightning Indexer Grad 分析](../docs/lightning_indexer_grad_analysis.md)
- [Lightning Indexer V2 Metadata 分析](../../lightning_indexer_v2_metadata/docs/lightning_indexer_v2_metadata_analysis.md)

## 许可证

Copyright (c) 2024. All rights reserved.