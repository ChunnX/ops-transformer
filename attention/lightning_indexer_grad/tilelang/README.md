# Lightning Indexer Gradient - TileLang-Ascend 实现（CV 分离架构）

使用 TileLang-Ascend 实现 lightning_indexer_grad 算子，计算 Lightning Attention 稀疏索引操作中 Query、Key、Weights 的梯度。

## 概述

本实现采用 **CV 分离架构**（Cube-Vector 分离），在单一 fused kernel 中通过 `T.Scope("C")` / `T.Scope("V")` 划分 Cube 与 Vector 阶段，使用 cross-flag 进行 Cube↔Vector 同步。该架构对标 AscendC arch22 流水线，将 dQuery 的完整计算下推到 NPU，dKey 和 dWeights 因涉及不规则索引访问（Gather/ScatterAdd）在 Python 层补算。

## 数学背景

### 前向传播
```
Indices = TopK{[1]_{1×G} @ [(W@[1]) ⊙ ReLU(Q_index @ K_index^T)]}
```

### 反向传播
给定上游梯度 `Dy`（形状 `[B, S1, N1, sparse_count]`），计算：

**dWeights**（Python 层）:
```
dW = ReduceSum(relu_in ⊙ Dy, dim=-1)
```

**dQuery**（NPU fused kernel）:
```
dQ = ReluGrad_out @ Key
```
其中 `ReluGrad_out = (W ⊙ Dy) ⊙ mask`，`mask = (relu_in > 0)`。

**dKey**（Python 层 ScatterAdd）:
```
dK[i] = Σ_{(s,k): indices[s,k]=i} ReluGrad_out[s,k] @ Query[s,:]
```

## CV 分离架构流水线

### Fused Kernel 流水线（每个 `(batch, n2_idx, s1_tile)` 任务）

```
┌───────────────────────────────────────────────────────────────────┐
│                    tilelang_lightning_indexer_grad_bsnd           │
├───────────────────────────────────────────────────────────────────┤
│  Scope("C")  │  Cube1: Q_group @ Key^T -> reluIn_ws (FP32)      │
│               │  set_cross_flag(FIX, 0)                          │
├───────────────────────────────────────────────────────────────────┤
│  Scope("V")  │  Vec2:  weights * dy (FP32乘)                    │
│               │        ReLU反向: compare(GT) + select            │
│               │        cast FP32->FP16 -> reluGrad_ws            │
│               │  set_cross_flag(FIX, 1)                          │
├───────────────────────────────────────────────────────────────────┤
│  Scope("C")  │  Cube3: reluGrad_ws @ Key -> dQuery_ws (FP32)    │
│               │  set_cross_flag(FIX, 2)                          │
├───────────────────────────────────────────────────────────────────┤
│  Scope("V")  │  Vec4:  dQuery_ws FP32->FP16 cast -> dQuery      │
└───────────────────────────────────────────────────────────────────┘
```

### Cross-Flag 同步机制

| 阶段 | Flag | 含义 |
|------|------|------|
| Cube1 完成 | flag=0 | reluIn_ws 已就绪，Vec2 可读取 |
| Vec2 完成 | flag=1 | reluGrad_ws 已就绪，Cube3 可读取 |
| Cube3 完成 | flag=2 | dQuery_ws 已就绪，Vec4 可读取 |

### Python 层补算（dKey + dWeights）

fused kernel 仅计算 dQuery。dKey 和 dWeights 在 Python 层逐 `(b, s1, n2)` 遍历：

1. **GatherTopk**: 基于 `sparse_indices` 收集 Key 数据 → `key_gather`
2. **ReLU 输入**: `relu_in = Q_group @ key_gather^T`
3. **ReluGradOut**: `weights * dy * (relu_in > 0)`
4. **Cube4 + ScatterAdd**: `scatter_add = reluGradOut^T @ Q_group`，按索引累加到 `dKey`
5. **dWeights**: `ReduceSum(relu_in * dy, dim=-1)`

## 参数详解

### 入口函数 `lightning_indexer_grad()`

| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `query` | `torch.Tensor` | `(B, S1, N1, D)` | Query 输入，BSND 布局，FP16/BF16 |
| `key` | `torch.Tensor` | `(B, S2, N2, D)` | Key 输入，BSND 布局，FP16/BF16 |
| `dy` | `torch.Tensor` | `(B, S1, N1, sparse_count)` | 上游梯度，BSND 布局，与 query 同 dtype |
| `sparse_indices` | `torch.Tensor` | `(B, S1, sparse_count)` 或 `(B, S1, N1, sparse_count)` | TopK 稀疏索引，int32，前向传播输出 |
| `weights` | `torch.Tensor` | `(B, S1, N1)` | 前向传播权重，与 query 同 dtype（若为 FP32 自动转 FP16） |
| `actual_seq_lengths_query` | `Optional[torch.Tensor]` | `(B,)` | 每个 batch 的实际 Query 序列长度，int32，默认为 `S1` |
| `actual_seq_lengths_key` | `Optional[torch.Tensor]` | `(B,)` | 每个 batch 的实际 Key 序列长度，int32，默认为 `S2` |
| `layout_query` | `str` | — | 数据布局，当前仅支持 `"BSND"` |
| `sparse_count` | `int` | — | TopK 稀疏索引数量，即 `topK`，默认 `2048` |
| `sparse_mode` | `int` | — | 稀疏模式：`0`=无 mask，`3`=causal mask，默认 `0` |

**返回值**:

| 输出 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `d_query` | `torch.Tensor` | `(B, S1, N1, D)` | Query 梯度，NPU 计算 fused kernel 输出 |
| `d_key` | `torch.Tensor` | `(B, S2, N2, D)` | Key 梯度，Python 层 ScatterAdd 计算 |
| `d_weights` | `torch.Tensor` | `(B, S1, N1)` | Weights 梯度，Python 层 ReduceSum 计算 |

### TilingData 参数 `LightningIndexerGradTilingData`

| 字段 | 类型 | 说明 |
|------|------|------|
| `B` | `int` | 批次大小（batch size） |
| `N1` | `int` | Query 头数量（head_num_q），必须满足 `N1 = N2 * G` |
| `N2` | `int` | Key 头数量（head_num_k） |
| `G` | `int` | 组大小（group_num），`G = N1 // N2` |
| `D` | `int` | 头维度（head_dim），必须能被 `BLOCK_D` 整除 |
| `S1` | `int` | Query 序列长度（seqlen_q） |
| `S2` | `int` | Key 序列长度（seqlen_k） |
| `sparse_count` | `int` | TopK 索引数量，默认 `2048`，必须能被 `BLOCK_N` 整除 |
| `sparse_mode` | `int` | 稀疏模式，`0` 或 `3`，默认 `0` |

### 全局 Tiling 常量

| 常量 | 值 | 说明 |
|------|------|------|
| `BLOCK_M` | `64` | S1 维度分块大小（每个 AI Core 处理 64 行 Query） |
| `BLOCK_N` | `128` | S2/sparse_count 维度分块大小（Key 行分块） |
| `BLOCK_D` | `128` | 头维度分块大小（D 的切分步长） |
| `VECTOR_BASEG` | `16` | Vector 对齐基数（组维度对齐） |
| `VECTOR_BASEN` | `256` | Vector 对齐基数（S2 维度 padding 对齐） |

### Fused Kernel 内部 Tensor

| Tensor | 形状 | dtype | 阶段 | 说明 |
|--------|------|-------|------|------|
| `Query` | `(B, S1, N1, D)` | input_dtype | 输入 | Query 输入 |
| `Key` | `(B, S2, N2, D)` | input_dtype | 输入 | Key 输入 |
| `Dy` | `(B, S1, N1, sparse_count)` | input_dtype | 输入 | 上游梯度 |
| `Weights` | `(B, S1, N1)` | input_dtype | 输入 | 前向权重 |
| `actual_q_len` | `(B,)` | int32 | 输入 | 实际 Query 序列长度 |
| `actual_k_len` | `(B,)` | int32 | 输入 | 实际 Key 序列长度 |
| `reluIn_ws` | `(B, N2, S1, G, sparse_count)` | calc_dtype (FP32) | workspace[2] | Cube1 输出，Vec2 读取 |
| `reluGrad_ws` | `(B, N2, S1, G, sparse_count)` | input_dtype (FP16) | workspace[3] | Vec2 输出，Cube3 读取 |
| `dQuery_ws` | `(B, S1, N1, D)` | calc_dtype (FP32) | workspace[4] | Cube3 输出，Vec4 读取 |
| `dQuery` | `(B, S1, N1, D)` | input_dtype | 输出 | 最终输出 |

## 参数校验规则

`check_param_valid()` 执行以下校验（与 `aclnnLightningIndexerGrad` 文档一致）：

| 校验项 | 规则 |
|--------|------|
| query 维度 | 必须为 4（BSND 布局） |
| key 维度 | 必须为 4（BSND 布局） |
| query dtype | 必须为 `float16` 或 `bfloat16` |
| key dtype | 必须与 query dtype 一致 |
| dy dtype | 必须与 query dtype 一致 |
| weights dtype | 必须与 query dtype 一致 |
| sparse_mode | 必须为 `0` 或 `3` |
| N1 可整除 | `N1` 必须能被 `N2` 整除，`G = N1 // N2` |

## 文件结构

```
tilelang/
├── README.md                              # 本文件
├── lightning_indexer_grad_cv.py           # CV 分离架构主实现（fused kernel + Python 入口）
├── lightning_indexer_grad_op.py           # 多 kernel 架构实现（旧版，Cube1-4 + Vec1-3 分离调用）
├── lightning_indexer_grad_matmul.py       # MatMul 内核（Cube1-4，供 op.py 使用）
├── lightning_indexer_grad_vector.py       # Vector 内核（Vec1-3，供 op.py 使用）
├── lightning_indexer_grad_kernel.py       # 单 kernel 伪实现（早期原型）
├── precision_test_cv.py                   # CV 架构精度测试
├── precision_test_official.py             # 官方算子精度对比测试
└── tests/
    └── test_lightning_indexer_grad.py     # 单元测试
```

## Fused Kernel 详解

### 1. Cube1: Q_group @ Key^T → reluIn_ws

```python
# Scope("C")
for g_idx in T.serial(G):
    n1_idx = n2_id * G + g_idx
    for n_idx in T.serial(n_num):
        for k_idx in T.serial(d_num):
            T.copy(Query[...], Q_L1)          # 加载 Q 分块到 L1
            T.copy(Key[...], K_L1)            # 加载 K 分块到 L1
        T.gemm_v0(Q_L1, K_L1, C_L0C, transpose_B=True)  # MatMul
        T.copy(C_L0C, reluIn_ws[...])        # 写入 FP32 workspace
```

- **输入**: `Query(block_m, D)`, `Key(block_n, D)`
- **输出**: `reluIn_ws(B, N2, S1, G, sparse_count)`，FP32
- **多核并行**: `block_num = B * N2 * s1_tile_num`，每个核处理一个 `(batch, n2_idx, s1_tile)`

### 2. Vec2: weights * dy + ReLU 反向 → reluGrad_ws

```python
# Scope("V"), wait_cross_flag(0)
for g_idx in T.serial(G):
    for n_idx in T.serial(n_num):
        T.copy(Weights[...], weight_ub)                    # 加载权重
        T.tile.cast(weight_fp32_ub, weight_ub)             # FP16→FP32
        T.copy(Dy[...], dy_ub)                             # 加载 Dy
        T.tile.cast(dy_fp32_ub, dy_ub)                     # FP16→FP32
        T.tile.mul(dy_fp32_ub, dy_fp32_ub, weight_fp32_ub)# weights * dy
        T.tile.compare(mask_ub, relu_in_ub, zero_ub, "GT") # relu_in > 0
        T.tile.select(relu_grad_ub, mask_ub, dy_fp32_ub)   # ReLU 反向
        T.tile.cast(relu_grad_half_ub, relu_grad_ub)       # FP32→FP16
        T.copy(relu_grad_half_ub, reluGrad_ws[...])        # 写入 FP16 workspace
```

- **输入**: `Weights(block_m,)`, `Dy(block_m, block_n)`, `reluIn_ws(block_m, block_n)`
- **输出**: `reluGrad_ws(B, N2, S1, G, sparse_count)`，FP16
- **内部**: 全部在 FP32 下计算，最终 cast 回 FP16

### 3. Cube3: reluGrad_ws @ Key → dQuery_ws

```python
# Scope("C"), wait_cross_flag(1)
for g_idx in T.serial(G):
    for d_idx in T.serial(d_num):
        for k_idx in T.serial(n_num):
            T.copy(reluGrad_ws[...], RG_L1)                # 加载 reluGrad
            T.copy(Key[...], KG_L1)                         # 加载 Key
            T.gemm_v0(RG_L1, KG_L1, DQ_L0C)               # MatMul
        T.copy(DQ_L0C, dQuery_ws[...])                     # 写入 FP32 workspace
```

- **输入**: `reluGrad_ws(block_m, block_n)`, `Key(block_n, block_d)`
- **输出**: `dQuery_ws(B, S1, N1, D)`，FP32
- **注意**: 输出为 FP32 workspace，避免 L0C→UB 通路的类型转换问题

### 4. Vec4: dQuery_ws FP32→FP16 cast → dQuery

```python
# Scope("V"), wait_cross_flag(2)
for g_idx in T.serial(G):
    for d_idx in T.serial(d_num):
        T.copy(dQuery_ws[...], dq_fp32_ub)                 # 加载 FP32 dQuery
        T.tile.cast(dq_fp16_ub, dq_fp32_ub, "CAST_ROUND") # FP32→FP16
        T.copy(dq_fp16_ub, dQuery[...])                    # 写入最终输出
```

- **输入**: `dQuery_ws(block_m, block_d)`，FP32
- **输出**: `dQuery(B, S1, N1, D)`，FP16

## 数据类型处理

| 阶段 | 计算精度 | 存储精度 | 说明 |
|------|----------|----------|------|
| Cube1 | FP32 (L0C) | FP32 (reluIn_ws) | MatMul 累加在 FP32 |
| Vec2 | FP32 (UB) | FP16 (reluGrad_ws) | Vector 内部 FP32，输出 cast 回 FP16 |
| Cube3 | FP32 (L0C) | FP32 (dQuery_ws) | MatMul 累加在 FP32，避免 L0C→UB 转换 |
| Vec4 | FP32→FP16 | FP16 (dQuery) | 最终 cast 回输入精度 |

## 使用方法

### 基本用法

```python
from lightning_indexer_grad_cv import lightning_indexer_grad

d_query, d_key, d_weights = lightning_indexer_grad(
    query=query_tensor,                    # (B, S1, N1, D)  FP16/BF16
    key=key_tensor,                        # (B, S2, N2, D)  FP16/BF16
    dy=dy_tensor,                          # (B, S1, N1, sparse_count)  FP16/BF16
    sparse_indices=indices_tensor,         # (B, S1, sparse_count)  int32
    weights=weights_tensor,                # (B, S1, N1)  FP16/BF16
    actual_seq_lengths_query=q_len_tensor, # (B,) int32, 可选
    actual_seq_lengths_key=k_len_tensor,   # (B,) int32, 可选
    layout_query="BSND",                   # 当前仅支持 BSND
    sparse_count=2048,                     # TopK 索引数量
    sparse_mode=0,                         # 0=无mask, 3=causal mask
)
```

### S2 Padding

当 `S2` 不能被 `VECTOR_BASEN` (256) 整除时，自动对 Key 进行零填充：

```python
S2_pad = ceil(S2 / 256) * 256
if S2_pad > S2:
    key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, S2_pad - S2, 0, 0))
```

## 内存层次

```
┌────────────────────────────────────────────────────────────┐
│                    昇腾内存布局                              │
├────────────────────────────────────────────────────────────┤
│  HBM (全局内存)                                              │
│    ├── Query, Key, Dy, Weights (输入)                       │
│    ├── reluIn_ws, reluGrad_ws, dQuery_ws (workspace)       │
│    └── dQuery (输出)                                        │
│                                                              │
│  L1 缓冲区 (AI Core 本地内存)                                │
│    ├── Q_L1, K_L1, RG_L1, KG_L1 (MatMul 输入分块)          │
│    └── ~1MB 每个 AI Core                                     │
│                                                              │
│  L0A/L0B/L0C (Cube 单元缓冲区)                               │
│    ├── L0A: 矩阵 A 输入 (FP16)                              │
│    ├── L0B: 矩阵 B 输入 (FP16)                              │
│    ├── L0C: 矩阵 C 累加器 (FP32)                            │
│    └── ~256KB 总计                                           │
│                                                              │
│  UB (Vector 单元的统一缓冲区)                                │
│    ├── relu_in_ub, dy_ub, weight_ub 等 (Vector I/O)        │
│    ├── 内部 FP32 计算缓冲区                                  │
│    └── ~64KB 每个 AI Core                                    │
└────────────────────────────────────────────────────────────┘
```

## 性能优化

### 1. CV 分离架构
```python
# Cube 和 Vector 阶段通过 Scope 分离，cross-flag 同步
with T.Scope("C"):
    # ... Cube1 计算 ...
    T.set_cross_flag("FIX", 0)
with T.Scope("V"):
    T.wait_cross_flag(0)
    # ... Vec2 计算 ...
    T.set_cross_flag("FIX", 1)
```

### 2. 多核并行
```python
# block_num = B * N2 * s1_tile_num 个核并行
with T.Kernel(block_num, is_npu=True) as (cid, vid):
    bz = cid // (N2 * s1_tile_num)
    n2_id = cid // s1_tile_num % N2
    m = cid % s1_tile_num
```

### 3. FP32 中间计算
- Vec2 内部全部在 FP32 下运算（compare、select、mul），避免 FP16 精度损失
- Cube3 输出 FP32 到 `dQuery_ws`，避免 L0C→UB 通路的类型转换问题
- 最终通过 Vec4 的 `CAST_ROUND` 转回 FP16

### 4. Workspace 复用
- `reluIn_ws`、`reluGrad_ws`、`dQuery_ws` 通过 `workspace_idx=[2, 3, 4]` 声明
- 编译器可优化 workspace 内存分配与生命周期

## 与旧版多 kernel 架构对比

| 方面 | CV 分离架构 (cv.py) | 多 kernel 架构 (op.py) |
|------|---------------------|------------------------|
| 架构 | 单 fused kernel | 6 个独立 kernel |
| Cube/Vec 同步 | cross-flag | 无（每次 kernel 调用隐式同步） |
| dQuery 计算 | 全部在 NPU | Cube3 kernel |
| dKey/dWeights 计算 | Python 层 PyTorch | Python 层 PyTorch |
| 中间数据 | workspace (reluIn_ws, reluGrad_ws, dQuery_ws) | 每步返回 PyTorch Tensor |
| 数据类型 | FP32 workspace, FP16 传输 | 全程 FP16 |
| 启动开销 | 1 次 kernel 启动 | 6 次 kernel 启动 |

## 已知限制

1. **Gather/ScatterAdd**: dKey 和 dWeights 在 Python 层逐元素遍历，性能受限于 CPU。后续可集成 AICPU 或 NPU 原子加以提升。

2. **仅 BSND 布局**: 当前仅支持 4 维 BSND 布局，不支持 TND。

3. **sparse_mode 有限**: 仅支持 `sparse_mode=0`（无 mask）和 `sparse_mode=3`（causal mask）。

4. **整除约束**: `S1` 须能被 `BLOCK_M`(64) 整除，`sparse_count` 须能被 `BLOCK_N`(128) 整除，`D` 须能被 `BLOCK_D`(128) 整除。不满足时需 padding。

5. **Key padding**: S2 不满足 `VECTOR_BASEN`(256) 对齐时自动 padding，产生额外内存开销。

## 测试

```bash
# 运行单元测试
cd tilelang
python -m pytest tests/ -v

# 运行 CV 架构精度测试
python precision_test_cv.py
```

## 参考资料

- [TileLang-Ascend 编程指南](../../docs/TileLang-Ascend%20Programming%20Guide.md)
- [Lightning Indexer Grad 分析](../docs/lightning_indexer_grad_analysis.md)
- [aclnnLightningIndexerGrad 文档](../docs/aclnnLightningIndexerGrad.md)

## 许可证

Copyright (c) 2024. All rights reserved.
