# FusedQuantLightningIndexerManage

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：`fused_quant_lightning_indexer_manage` 面向大序列稀疏 KV Cache 卸载场景，在一次 NPU 调用中完成量化 Lightning Indexer TopK 检索、HBM 稀疏缓存的初始化或稳态淘汰、MTP 多路 Query 的 TopK Union 管理，以及 Cache Miss 搬运计划生成。同一个 Batch 可以混合非卸载、首次卸载和稳态卸载请求，每个请求支持 MTP0～3，即 1～4 路 Query。
- Lightning Indexer 计算公式：

  $$
  \mathrm{Indices}=\mathrm{TopK}\left(\mathbf{1}_{1\times N}\left[\left(W\mathbf{1}_{1\times S_k}\right)\odot\mathrm{ReLU}\left(Q_{\mathrm{index}}K_{\mathrm{index}}^{T}\right)\right]\right)
  $$

  其中，$Q_{\mathrm{index}}\in\mathbb{R}^{N\times128}$ 为某一路反量化后的 Index Query，$K_{\mathrm{index}}\in\mathbb{R}^{S_k\times128}$ 为该路 Query 因果可见的反量化 Index Key，$W\in\mathbb{R}^{N\times1}$ 为 Index Head 聚合权重。算子为每一路 Query 输出最多 2048 个 Source Token ID 及其 HBM Logical Slot。
- 稀疏缓存管理：算子通过 `request_state` 区分非卸载、首次卸载和稳态卸载状态，并通过 `cache_slots_pool` 维护 Source Token 到 HBM Logical Slot 的持久化映射。算子输出本轮需要从 CPU 侧搬入 HBM 的 Source Token ID、目标 Slot 及有效搬运数量；实际数据搬运由上层框架执行。

## 参数说明

```python
from cann_ops_transformer_custom.ops import fused_quant_lightning_indexer_manage

fused_quant_lightning_indexer_manage(
    index_weights,
    query_dequant_scale,
    query,
    index_key_dequant_scale,
    index_key_cache,
    index_block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    offload_seq_lengths_key,
    num_cache_tokens,
    request_state,
    req_pool_entries,
    cache_slots_pool,
    topk_src_ids,
    topk_dst_slots,
    topk_miss_counts,
    miss_src_ids,
    miss_dst_slots,
    miss_counts,
)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| `index_weights` | 输入 | 每条 Query、每个 Index Head 的聚合权重，shape 为 `(T,N)`，`N` 支持 32 或 64。 | BFLOAT16 | ND |
| `query_dequant_scale` | 输入 | Query 反量化 Scale，shape 为 `(T,N)`。 | FLOAT | ND |
| `query` | 输入 | TND 排布的量化 Index Query，shape 为 `(T,N,128)`；同一请求的多路 Query 在 T 维连续。 | FLOAT8_E4M3FN | ND |
| `index_key_dequant_scale` | 输入 | Index Key 反量化 Scale，shape 为 `(index_block_num,128,1)`。 | FLOAT | ND |
| `index_key_cache` | 输入 | PageAttention 排布的量化 Index Key Cache，shape 为 `(index_block_num,128,1,128)`。 | FLOAT8_E4M3FN | ND |
| `index_block_table` | 输入 | 每个请求的 Index Key 逻辑块到物理块映射表，shape 为 `(B,index_max_blocks)`；`index_max_blocks × 128` 不超过 262144。 | INT32 | ND |
| `actual_seq_lengths_query` | 输入 | 各请求 Query 在 T 维的累计结束位置，shape 为 `(B)`；元素严格递增，最后一个元素必须等于 `T`。 | INT32 | ND |
| `actual_seq_lengths_key` | 输入 | 每个请求最后一路 Query 对应的实际 KV 长度，shape 为 `(B)`；其他路 Query 的因果可见长度由该值和 Query 位置推导。 | INT32 | ND |
| `offload_seq_lengths_key` | 输入 | 每个请求可检索、可卸载的稳定 Source Prefix 长度，shape 为 `(B)`，仅 `request_state` 为 `-2` 或 `-1` 时使用。 | INT32 | ND |
| `num_cache_tokens` | 输入 | 每个请求的 HBM 稀疏缓存容量，shape 为 `(B)`，仅 `request_state` 为 `-2` 或 `-1` 时使用。 | INT32 | ND |
| `request_state` | 输入 | 请求状态，shape 为 `(B)`；仅支持 `-3`（非卸载）、`-2`（首次卸载）和 `-1`（稳态卸载）。 | INT32 | ND |
| `req_pool_entries` | 输入 | 每个请求占用的 `cache_slots_pool` 行号，shape 为 `(B)`；同一次调用中的有效请求必须使用不同的行号。 | INT32 | ND |
| `cache_slots_pool` | 输入/输出（原地更新） | 跨 Decode Step 持久化的 Source Token 到 HBM Logical Slot 映射，shape 为 `(pool_size,source_capacity)`，其中 `source_capacity=index_max_blocks×128`；`-1` 表示 Source 不常驻。 | INT32 | ND |
| `topk_src_ids` | 输出 | 每一路 Query 的 TopK Source Token ID，shape 为 `(T,1,2048)`；可见长度不足 2048 时使用 `-1` 填充。 | INT32 | ND |
| `topk_dst_slots` | 输出 | 与 `topk_src_ids` 逐项对应的 HBM Logical Slot，shape 为 `(T,1,2048)`；无有效 Slot 时为 `-1`。 | INT32 | ND |
| `topk_miss_counts` | 输出 | 每一路 Query 在 `topk_src_ids` 中的 Miss 前缀长度，shape 为 `(T)`；非卸载状态固定为 0。 | INT32 | ND |
| `miss_src_ids` | 输出 | 每个请求本轮需要搬入 HBM 的 Source Token ID，shape 为 `(B,32768)`；仅前 `miss_counts[b]` 项有效。 | INT32 | ND |
| `miss_dst_slots` | 输出 | 与 `miss_src_ids` 逐项对应的目标 HBM Logical Slot，shape 为 `(B,32768)`；仅前 `miss_counts[b]` 项有效。 | INT32 | ND |
| `miss_counts` | 输出 | 每个请求的有效搬运项数量，shape 为 `(B)`；`-1` 表示该请求处理失败。 | INT32 | ND |

六个输出 Tensor 均由调用方分配并传入，算子原地写入且不返回 Python 对象；`cache_slots_pool` 同样由算子原地更新。三个物理 stride 由 Torch 扩展从输入 Tensor 自动读取并传给 ACLNN。

其中，`B` 为 batch size，`T` 为 query 总路数，`S = index_block_table.shape[1] × 128 = cache_slots_pool.shape[1]`。每个请求的路数为：

```text
Q_b = actual_seq_lengths_query[b] - (b == 0 ? 0 : actual_seq_lengths_query[b-1])
```

支持 `Q_b=1..4`，分别对应 MTP0～3。32768 是 COPY buffer 的 ABI 宽度，不是缓存预算。

## 约束说明

- `S <= 262144`，所有请求的累计 query end 必须严格递增，且末项等于 `T`。
- 每请求 `Q_b` 为 1～4，`actual_seq_lengths_key[b] >= Q_b`，且不超过 `S`。
- key cache 与 block table 必须覆盖 `actual_seq_lengths_key[b]` 的完整可见序列，而非仅覆盖 `L`。
- `index_weights` 仅允许 T 轴 padding stride；`index_key_cache` 和 `index_key_dequant_scale` 仅允许 block 轴 padding stride；其他 Tensor 必须 contiguous。
- 所有 Tensor 必须位于同一 NPU 设备。
- `req_pool_entries[b]` 必须位于 `[0,POOL_SIZE)`，同 batch 的 active pool row 不得重复。
- 当前有效缓存预算 `C` 只允许 8192、10240、12288。
- `request_state` 不由算子修改，状态切换由调用方在下一次调用前完成。

`-2/-1` 状态还必须满足：

```text
L >= 2048，L % 128 == 0
C ∈ {8192,10240,12288}，C <= L
L <= actual_seq_lengths_key[b] - (Q_b-1)
0 <= actual_seq_lengths_key[b] - L <= 256
```

shape、dtype、stride、设备等静态 ABI 错误在调用时由 Torch/host 直接报错。对于设备端能够检测的逐请求非法 metadata、TopK/union 容量或 victim 不足，失败协议为：`miss_counts[b]=-1`，相关 `topk_miss_counts=-1`，该请求的 TopK src/dst 全部为 `-1`，pool row 保持调用前内容；该请求的 COPY buffer 不可使用，其他合法请求继续执行。

## Query 布局

对请求 `b` 的 query 行 `r`，令 `end = actual_seq_lengths_query[b]`、`Q_b` 为该请求路数，则该路的因果可见长度为：

```text
V_r = actual_seq_lengths_key[b] - (end - 1 - r)
```

同一请求的各路 query 在 T 轴连续。每路在完整 `[0,V_r)` 上执行 A5 C8 LI；`-2/-1` 的 `L` 不截断 LI 搜索域，只划分缓存管理域。

## LI 与输出约定

每路 LI 输出最多 2048 个 source。只有真实入选 Top2048 的 tail token 才出现在输出中。对 `-2/-1`，`token >= L` 的入选 tail 直接归为 hit，目标 slot 为 `C + token % 256`，不进入 pool、COPY union 或 victim scan。调用方若消费该 slot 范围，需按自身链路准备对应存储；本算子不负责下游配置。

`topk_miss_counts[r]` 划定该路 TopK 的 miss-prefix：`topk_src_ids[r,0,:count]` 为 miss，其余有效位置为 hit。两段内部均不承诺按 source ID 排序。`miss_counts[b]` 是请求级 COPY 列表有效长度，不必等于各路 miss count 之和。source/slot 之间的配对及 pool 映射必须一致。

`miss_src_ids[b,:miss_counts[b]]` 和 `miss_dst_slots[b,:miss_counts[b]]` 在 count 非负时有效，后缀内容不保证。

## 请求状态与生命周期

### `-3`：非卸载

每路按自己的 `V_r` 执行 LI；可见长度不足 2048 时仍走 LI，有效后缀外填 `-1`。有效输出满足 `topk_dst_slots == topk_src_ids`；该请求所有 `topk_miss_counts=0`、`miss_counts=0`。`offload_seq_lengths_key` 和 `num_cache_tokens` 不参与检索或缓存管理。

每次 `-3` 调用都把对应的完整 pool row 写为 identity：`pool[source]=source`，范围为 `[0,S)`。该状态不产生 COPY 列表，`miss_src_ids/miss_dst_slots` 内容无效。

### `-2`：首次填充

要求 `L=offload_seq_lengths_key[b]`、`C=num_cache_tokens[b]`。算子忽略旧 pool，包括先前 `-3` 留下的 identity row。完成各路 full-visible Top2048 后，只将入选且 `token < L` 的 source 加入请求级 union；先安置该 union，再从 `[0,L)` 中非 union source 补 filler，直到恰好有 C 个 resident，slot 为 `0..C-1`。入选 tail 是 hit，不进入 union。

成功时，每路 `topk_miss_counts` 等于该路入选 prefix 数；TopK 输出为 prefix miss 在前、tail hit 在后；`miss_counts[b]=C`，COPY 列表只含 C 个 prefix resident，当前实现的 `miss_dst_slots[b,:C]` 依次为 `0..C-1`。完整 pool row 清为 `-1` 后，仅这些 resident source 映射到各自 slot。算子先校验 metadata、TopK union 和 C 容量，再改动 pool，以保持可检测失败时的请求级原子性。

### `-1`：卸载稳态

入选的 `token < L` 按调用前 pool 判定 resident hit 或 miss；入选 tail 始终为 hit。对全部路的 prefix miss 做请求级去重 union，保护入选的 prefix hit，再进行 hashed victim scan、分配 slot 并更新 pool。每路仍保持 miss-prefix/hit-suffix；`topk_miss_counts` 只统计该路 prefix miss，`miss_counts` 只统计请求级 unique prefix miss，COPY 仅列出这些 miss。不保证 union 列表或路内两段按 ID 排序。

合法的 sparse pool 在 `[0,L)` 内恰好有 C 个 resident source，slot 集合恰好为 `0..C-1`；其他 source 为 `-1`，`[L,S)` 不常驻。

直接从 `-3` 切到 `-1` 时，如果 `C<S` 且 `pool[row,C] == C`，管理器验证 identity row，保留 `[0,C)` 的 identity mapping，把 `[C,S)` 清为 `-1`，再按稳态处理。普通 sparse row 不做整行重写。调用方可使用 `-2 → -1`、`-3 → -1` 或 `-3 → -2 → -1`；一个请求的 pool row 在生命周期内应保持稳定。

## 性能结果

以下结果为本仓 FusedQuantLightningIndexerManage 算子在 `request_state=-1`、`Q=4`、`C=12288` 场景下的实测结果。`union miss` 是请求级去重后的平均 miss 数，`query miss` 是每路 query 的平均 miss 数，LIM 时延单位为 µs。

| L | B | union miss 均值 | query miss 均值 | 本仓 LIM |
|---:|---:|---:|---:|---:|
| 65536 | 1 | 308.000 | 82.000 | 111.516 |
| 65536 | 8 | 394.625 | 104.406 | 121.212 |
| 65536 | 16 | 395.812 | 104.109 | 170.191 |
| 65536 | 24 | 391.125 | 102.865 | 203.228 |
| 65536 | 32 | 394.031 | 103.508 | 220.032 |
| 131072 | 1 | 308.000 | 78.750 | 208.294 |
| 131072 | 8 | 394.625 | 101.562 | 290.599 |
| 131072 | 16 | 395.812 | 101.656 | 322.264 |
| 131072 | 24 | 391.125 | 100.229 | 366.298 |
| 131072 | 32 | 394.031 | 101.086 | 421.810 |

正确性测试环境和执行命令见 [tests/pytest/README.md](tests/pytest/README.md)。
