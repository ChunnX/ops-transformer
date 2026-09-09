# Quant Sparse Lightning Indexer

## 功能说明

`quant_sparse_lightning_indexer` 是 QSLI 的融合 DSL sample。算子接收已量化的
MXFP4 Query、离散 PageAttention Key/KScale 和 candidate block 列表，完成：

1. 根据 `candidate_block_indices` 和 `block_table` 将离散 K/KScale 拼接成 Cube tile；
2. 在 MXFP4 Cube 上计算 Query-Key 相关性；
3. 执行 ReLU、Query head 权重归约和 mask；
4. 对 token 分数执行流式 TopK；
5. 将 compact position 恢复为逻辑 token index，并应用 `output_idx_offset`。

离散 KV gather、PageAttention 地址映射、QK、mask、TopK 和输出映射均在一个融合
Kernel 内完成。K 和 KScale 保持 MXFP4/E8M0 表示，不包含软件反量化流程，也不会对
输入 candidate block 重新排序。

## 函数原型

```python
def quant_sparse_lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    descale_k: torch.Tensor,
    candidate_block_indices: torch.Tensor,
    candidate_block_length: torch.Tensor,
    topk: int,
    quant_mode: int,
    candidate_block_size: int,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_kv: str = "PA_BBND",
    return_value: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]
```

该入口注册为
`torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer`；通过
`cann_ops_transformer.ops.ds41` 导入的同名对象就是该 Torch operator。

## 维度含义

- `T1`：所有 Batch 的 Query token 数之和。
- `B`：Batch 数量。
- `N1`：Query head 数，固定为 32。
- `N2`：Key head 数，固定为 1。
- `D`：Q/K 逻辑特征维度，固定为 128。
- `PA`：物理 PageAttention block 数量。
- `block_size`：每个 PA block 的 token 数，固定为 128。
- `C`：candidate block 容量，固定为 2048。
- 一个 candidate block 包含 8 个连续逻辑 K token。

MXFP4 每个字节打包两个逻辑元素，因此 `q` 和 `k` 的物理末维为 `D/2=64`。
每个 E8M0 scale 对应 32 个逻辑 D 元素；D=128 时每行 Q/K 使用 4 个 scale，接口
表示为 `(D/64,2)=(2,2)`。

## 参数说明

| 参数名 | 类型 | Shape | 说明 |
|---|---|---|---|
| `q` | NPU `torch.uint8` | `(T1,N1,D)` | Packed MXFP4 Query。必须是 TND 排布，不支持空 Tensor。 |
| `k` | NPU `torch.uint8` | `(block_num,block_size,N2,D)` | Packed MXFP4 PageAttention Key。支持仅第 0 轴非连续。 |
| `w` | NPU `torch.float32` | `(T1,N1)` | Query head 的归约权重。 |
| `descale_q` | NPU `torch.uint8` 或 `torch.float8_e8m0fnu` | `(T1,N1,Ceil(D/64),2)` | Query 的 4 个 E8M0 MX scale。 |
| `descale_k` | NPU `torch.uint8` 或 `torch.float8_e8m0fnu` | `(block_num,block_size,N2,D)` | Key 的 4 个 E8M0 MX scale。支持仅第 0 轴非连续。 |
| `candidate_block_indices` | NPU `torch.int32` | `(T1,1,C)` | 每行 Query 对应的逻辑 candidate block id。只读取有效前缀，后缀忽略。 |
| `candidate_block_length` | NPU `torch.int32` | 建议为 `(T1,1)` | 每行 candidate 的有效前缀长度，取值范围为 `[0,C]`。 |
| `cu_seqlens_q` | 可选，NPU `torch.int32` | `(B+1,)` | TND Query 的 Batch 前缀和。未传入时仅表示单 Batch。 |
| `seqused_q` | 可选，NPU `torch.int32` | `(B,)` | 每个 Batch 的有效 Query 长度，用于 TND padding。 |
| `seqused_k` | 可选，NPU `torch.int32` | `(B,)` | 每个 Batch 的实际 K 长度，用于 mask。 |
| `cmp_residual_k` | 条件输入，NPU `torch.int32` | `(B,)` | 压缩 causal mask 的残余长度。仅 `mask_mode=3` 且 `cmp_ratio!=1` 时传入。 |
| `block_table` | 可选，NPU `torch.int32` | `(B,max_blocks)` | 逻辑 PA page 到物理 page 的映射。`B>1` 时必须传入。 |
| `output_idx_offset` | 可选，NPU `torch.int32` | `(T1,1)` | 加到每行有效输出 token index 上的偏移量。 |

### 属性

| 属性名 | 范围/取值 | 说明 |
|---|---|---|
| `topk` | `[1,8192]` | 每行保留的 token 数量。 |
| `candidate_block_size` | `8` | 一个 candidate block 对应的 K token 数。 |
| `quant_mode` | `1` | 当前 sample 唯一支持的量化模式。 |
| `max_seqlen_q` | `-1` 或非负整数 | 接口兼容属性；`-1` 表示不指定。 |
| `mask_mode` | `0` 或 `3` | `0` 表示不使用 causal mask，`3` 表示右下 causal mask。 |
| `cmp_ratio` | `[1,128]` | K 的压缩比例。 |
| `layout_q` | `"TND"` | 当前唯一支持的 Query layout。 |
| `layout_kv` | `"PA_BBND"` | 当前唯一支持的 Key layout。 |

`metadata` 是保留参数，当前必须为 `None`；QSLI 使用融合 Kernel 内的手工分核。
`return_value` 是兼容参数，当前接口始终返回 `sparse_indices` 和 `sparse_values`。

## 输出说明

返回两个 Tensor 组成的 tuple：

| 位置 | 含义 | 数据类型 | Shape | 说明 |
|---|---|---|---|---|
| `0` | `sparse_indices` | NPU `torch.int32` | `(T1,1,topk)` | TopK 对应的逻辑 K token index。有效 index 已加 `output_idx_offset`；无效位置为 `-1`。 |
| `1` | `sparse_values` | NPU `torch.bfloat16` | `(T1,1,topk)` | 与 `sparse_indices` 配对的相关性值；无效位置为 0。 |

当有效 token 数少于 `topk` 时，输出尾部按上述无效值填充。TopK 顺序不承诺作为
稳定排序接口；精度比较应同时结合 index 对应的 value。

## PageAttention 与 candidate 语义

`candidate_block_indices[t,0,i]` 是由 8 个 token 组成的逻辑 block id。Kernel 中的
地址恢复过程为：

```text
logical_token_begin = candidate_block_id * 8
logical_page        = logical_token_begin // 128
offset_in_page      = logical_token_begin % 128
physical_page       = block_table[batch, logical_page]
```

未传入 `block_table` 时，物理 page 与逻辑 page 相同，且只支持单 Batch。输入
candidate 的有效前缀必须属于对应 Batch 的逻辑 K 范围。

## 非连续 Tensor 支持

`k` 和 `descale_k` 支持第 0 轴非连续，连续 Tensor 是该 stride 规则的特例。除第 0
轴外，其余轴必须保持标准 ND 连续 stride；不支持任意内轴转置或负 stride。
K 与 KScale 的第 0 轴 stride 独立读取，因此可以分别为连续或非连续。

## 动态 Shape 与编译缓存

以下维度或 stride 使用 `cannbotdsl.Dim` 动态编译：

- `T1`、`B`、`PA`；
- `block_table` 的 `max_blocks`；
- K 和 KScale 的第 0 轴 stride。

S2 对应的 `logical_k_capacity` 作为运行时标量传入 Kernel。编译结果由线程锁保护并
缓存在进程内，同一 constexpr 配置下改变上述 shape/stride 不会再次编译。

当前 DSL 的 Host ABI 要求 `fp4x2_e2m1` staging 使用静态 shape，因此以下项目仍是
有限编译特化项：

- `worker_count=min(T1,32)`；
- `topk`；
- `mask_mode`、`cmp_ratio`；
- 可选输入是否存在对应的 constexpr 分支。

## 约束说明

- 所有 Tensor 输入必须位于同一 NPU device。
- `T1>0`、`PA>0`。
- `N1=32`、`N2=1`、逻辑 `D=128`、PA `block_size=128`。
- candidate 容量固定为 2048，candidate block size 固定为 8。
- `B>1` 时必须提供 `cu_seqlens_q` 和 `block_table`。
- `mask_mode=3` 且 `cmp_ratio!=1` 时必须提供 `cmp_residual_k`；其他场景必须传
  `None`。
- `descale_k` 第 0 轴 stride 必须满足 E8M0 pair 对齐。

## 调用说明

依赖已配置的 CANN、PyTorch、torch_npu 和 CANNBotDSL 环境。通过仓库
`torch_extension` 构建流程安装 `cann_ops_transformer` 后使用：

| 调用方式 | 入口 | 说明 |
| :--- | :--- | :--- |
| PyTorch API | `cann_ops_transformer.ops.ds41.quant_sparse_lightning_indexer` | 分配输出并调用 DSL，返回 `(sparse_indices, sparse_values)` |
| torch.ops | `torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer` | 导入扩展完成注册后，可通过 PyTorch 自定义算子入口调用；参数和返回值与 PyTorch API 一致 |
| DSL 直调 | [`torch_extension/quant_sparse_lightning_indexer_dsl.py`](torch_extension/quant_sparse_lightning_indexer_dsl.py) | 返回包含两个输出的 `dict`，主要用于内部实现和调试 |

`ds41` 是实际 Python 模块名和 `torch.library` 注册名。正常业务代码推荐使用
PyTorch API；需要验证注册通路时可直接使用 `torch.ops`。

## 调用示例

```python
from cann_ops_transformer.ops.ds41 import (
    quant_sparse_lightning_indexer,
)

result = quant_sparse_lightning_indexer(
    q,
    k,
    weights,
    descale_q,
    descale_k,
    candidate_block_indices,
    candidate_block_length,
    topk,
    quant_mode,
    candidate_block_size,
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    cmp_residual_k=None,
    block_table=block_table,
    output_idx_offset=output_idx_offset,
    mask_mode=mask_mode,
    cmp_ratio=cmp_ratio,
    layout_q=layout_q,
    layout_kv=layout_kv,
)

sparse_indices, sparse_values = result
```

等价的 `torch.ops` 调用如下。先导入算子扩展以完成注册：

```python
import torch
import cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl

result = torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer(
    q,
    k,
    weights,
    descale_q,
    descale_k,
    candidate_block_indices,
    candidate_block_length,
    topk,
    quant_mode,
    candidate_block_size,
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    cmp_residual_k=None,
    block_table=block_table,
    output_idx_offset=output_idx_offset,
    mask_mode=mask_mode,
    cmp_ratio=cmp_ratio,
    layout_q=layout_q,
    layout_kv=layout_kv,
)
```

测试入口和转测用例说明见
`test/quant_sparse_lightning_indexer/README.md`。


## net Native wheel 调用

DSL 实现统一维护在 `cannbot-arena-ds41/net/ops/`，三个算子通过
`net/native_package/run-build.sh --group ds41` 编译到同一个
`cannbot-arena-net-ops` wheel。先安装该 wheel，再构建并安装本仓的
`cann_ops_transformer` wheel。torch_extension 仅保留 torch 注册和接口适配，
运行时从已安装的 `ops` 包导入 host 入口，不再打包本目录的 DSL 副本。
pytest 仍通过已安装的 torch wheel 调用 `torch.ops`。

Native 包预编译范围由导出配置决定；默认覆盖已有单算子 smoke 配置，
其他静态配置需通过 `CANNBOTDSL_DS41_PROFILES` 指定 JSON 后重新打包。
详见 net 工程 `native_package/README.md`。

2026-09-09 net 迁移 smoke：`pytest/test_quant_sparse_lightning_indexer_single.py
-k qsli_decode` 结果 `1 passed, 9 deselected`；
`CANNBOTDSL_NATIVE_BINARY_MODE=require` 下报告 `installed_hit=1`。
