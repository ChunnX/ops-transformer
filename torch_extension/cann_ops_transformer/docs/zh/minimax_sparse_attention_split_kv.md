# minimax\_sparse\_attention\_split\_kv

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

  `minimax_sparse_attention_split_kv` 是基于 `torch_npu` 的 `cann_ops_transformer` 扩展接口，用于调用 `MinimaxSparseAttentionSplitKv` 完成 Prefill 阶段的 KV-Gather-Q 稀疏注意力。同一个 KV block 只加载一次，按反向 CSR 收集选中该 block 的 Q token，做 QK / softmax / PV 得到未归一化 partial，再在 Phase2 用 FlashDecode 合并。

  `build_k2q_csr` 是配套的 host 侧工具：训练侧通常只有 indexer 产出的 `select_idx`，需要先转成算子要的 CSR 三件套。

- **计算公式**：

  对每个 Q token，只在其 topK 个 KV block 上计算注意力：

  $$
  S = \frac{QK_{sel}^{T}}{\sqrt{d}},\quad
  P = e^{S - \mathrm{rowMax}},\quad
  O = \frac{\sum_k P_k V_k}{\sum_k \mathrm{rowSum}_k}
  $$

  `inner_precise=0` 时 softmax 在 fp32 上做，P 再 round 到 bf16 做 PV；默认 `inner_precise=4` 时 softmax 走 bf16，partial O 仍为 fp32。

> [!NOTE]
>
> B 为 batch，S 为 padded 序列长，T 为 packed token 数（`sum(actual_seq_lengths)`），N / Nkv 为 Q / KV head 数，D 为 head dim（当前仅 128）。`group_size = N / Nkv` 必须在 `[1, 16]`。本接口只有前向，没有 backward。

## 函数原型

```python
cann_ops_transformer.build_k2q_csr(
    select_idx,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_size,
    *,
    select_num_idx=None,
    index_mode="batch_local",
    input_layout="TND",
    q_seq_pad=None,
) -> (Tensor, Tensor, Tensor)
```

```python
cann_ops_transformer.minimax_sparse_attention_split_kv(
    query,
    key,
    value,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_slot_indices,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    num_key_value_heads,
    scale_value,
    block_size,
    top_k,
    *,
    block_table=None,
    inner_precise=4,
    softmax_lse_flag=False,
    input_layout="TND",
) -> (Tensor, Tensor)
```

## 参数说明

### build_k2q_csr

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| select_idx | Tensor | 必选 | indexer 选出的 KV block id，无效位填 `-1`。TND 为 packed token 轴；BNSD/BSND 可为 `[Nkv, B, S, topK]` | int32 | `[Nkv, T, topK]` 或 `[Nkv, B, S, topK]` |
| actual_seq_lengths | Tensor | 必选 | 每个 batch 真实 Q 长度，非 cumsum，允许 0 | int32 | `[B]` |
| actual_seq_lengths_kv | Tensor | 必选 | 每个 batch 真实 KV 长度，非 cumsum，允许 0 | int32 | `[B]` |
| block_size | int | 必选 | 每个 KV block 的 token 数，生产配置 128 | int | - |
| select_num_idx | Tensor | 可选 | 每个 token 实际有效的 topK 个数，超出部分按 `-1` 处理 | int32 | `[Nkv, T]` |
| index_mode | str | 可选 | `"batch_local"`：每 batch 内的逻辑 block id（indexer / CUDA q2k 默认）；`"global"`：跨 batch packed 的全局 block id | string | - |
| input_layout | str | 可选 | 与主算子一致，决定 CSR 里 q id 是 packed T 还是 padded `b*S+t` | string | - |
| q_seq_pad | int | 可选 | BNSD/BSND 且 `select_idx` 为 3D 时的 padded S，默认 `T/B` | int | - |

### minimax_sparse_attention_split_kv

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | 必选 | Q。不支持空 Tensor | bfloat16 | TND `[T, N, D]`；BNSD `[B, N, S, D]`；BSND `[B, S, N, D]` |
| key | Tensor | 必选 | K。paged 或与 query 同 layout 的连续 KV | bfloat16 | paged `[num_blocks, block_size, Nkv, D]`；连续则与 `input_layout` 一致 |
| value | Tensor | 必选 | V，layout / dtype 与 key 相同 | bfloat16 | 同 key |
| k2q_row_ptr | Tensor | 必选 | CSR 行指针，由 `build_k2q_csr` 生成 | int32 | `[Nkv, total_kv_rows+1]` |
| k2q_q_indices | Tensor | 必选 | CSR 的 q token id。TND 为 packed；BNSD/BSND 为 `b*S+t` | int32 | `[Nkv, T*topK]` |
| k2q_slot_indices | Tensor | 必选 | 每条边对应的 topK slot，范围 `[0, topK)` | int32 | `[Nkv, T*topK]` |
| actual_seq_lengths | Tensor | 必选 | 每个 batch 真实 Q 长度，允许 0（dummy request） | int32 | `[B]` |
| actual_seq_lengths_kv | Tensor | 必选 | 每个 batch 真实 KV 长度，允许 0 | int32 | `[B]` |
| num_key_value_heads | int | 必选 | KV head 数 | int | - |
| scale_value | float | 必选 | softmax scale，通常 `1/sqrt(D)` | float | - |
| block_size | int | 必选 | KV block token 数，`(0, 128]`，生产 128 | int | - |
| top_k | int | 必选 | 每个 Q token 选中的 KV block 数 | int | - |
| block_table | Tensor | 可选 | paged KV 的逻辑→物理 block 表。训练连续 KV 传 `None`。paged 仅支持 TND | int32 | `[B, max_blocks_per_batch]` |
| inner_precise | int | 可选 | `0`：fp32 softmax + fp32 O_partial；`1`：bf16 softmax + bf16 O_partial；`4`（默认）：bf16 softmax + fp32 O_partial | int | - |
| softmax_lse_flag | bool | 可选 | `True` 时写出 fp32 LSE；`False` 时 LSE 为 `[0]` 占位 | bool | - |
| input_layout | str | 可选 | `"TND"` / `"BNSD"` / `"BSND"`。训练图常用 `"BNSD"`。paged KV 必须 `"TND"` | string | - |

## 返回值说明

### build_k2q_csr

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- |
| k2q_row_ptr | Tensor | packed-row CSR 行指针 | int32 | `[Nkv, total_kv_rows+1]` |
| k2q_q_indices | Tensor | CSR q token id，无效位为 `-1` | int32 | `[Nkv, T*topK]` |
| k2q_slot_indices | Tensor | CSR slot id，无效位为 `-1` | int32 | `[Nkv, T*topK]` |

### minimax_sparse_attention_split_kv

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 注意力输出，shape / dtype 与 query 相同 | bfloat16 | 同 query |
| softmax_lse | Tensor | `softmax_lse_flag=True` 时为 LSE；否则为 `[0]` 占位 | float32 | TND `[T, N, 1]`；BNSD `[B, N, S, 1]`；BSND `[B, S, N, 1]` |

## 约束说明

- 仅前向，无 autograd。
- 仅 bf16，D 必须为 128。
- `N` 必须能被 `num_key_value_heads` 整除，且 `group_size = N / Nkv ∈ [1, 16]`。
- `block_size ∈ (0, 128]`。
- `actual_seq_lengths` / `actual_seq_lengths_kv` / CSR / `block_table` 的数值正确性由调用方保证；tiling 只读 shape，不校验 CSR 内容。非法值会导致精度错误或 AICore 异常。
- BNSD/BSND 的 CSR q id 必须是 padded 展平 `b * S + t`，不要传 TND packed id。`build_k2q_csr(..., input_layout="BNSD")` 会按这个约定写。
- paged KV（传入 `block_table`）只支持 `input_layout="TND"`。训练连续 KV 不要传 `block_table`。
- 首次调用会 JIT 编译 C++ wrapper，需要已安装 custom opp，并设置 `ASCEND_CUSTOM_OPP_PATH`。

## 调用示例

训练侧推荐：连续 BNSD KV + indexer `select_idx`（batch 内逻辑 block id）。

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

batch, s_q, s_kv = 2, 32, 64
q_heads, kv_heads, head_dim = 64, 4, 128
block_size, top_k = 128, 4
group_size = q_heads // kv_heads  # 16，当前上限

query = torch.randn(batch, q_heads, s_q, head_dim, dtype=torch.bfloat16, device="npu")
key = torch.randn(batch, kv_heads, s_kv, head_dim, dtype=torch.bfloat16, device="npu")
value = torch.randn(batch, kv_heads, s_kv, head_dim, dtype=torch.bfloat16, device="npu")

actual_seq_lengths = torch.tensor([32, 24], dtype=torch.int32, device="npu")
actual_seq_lengths_kv = torch.tensor([64, 48], dtype=torch.int32, device="npu")

# indexer 输出：每个 Q token 的 batch-local KV block id，无效填 -1
# 本例 KV 长度 < block_size，每条序列只有 1 个逻辑 block
select_idx = torch.zeros(kv_heads, batch, s_q, top_k, dtype=torch.int32, device="npu")
select_idx[..., 1:] = -1

k2q_row_ptr, k2q_q_indices, k2q_slot_indices = cann_ops_transformer.build_k2q_csr(
    select_idx,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    block_size,
    input_layout="BNSD",
)

attn_out, softmax_lse = cann_ops_transformer.minimax_sparse_attention_split_kv(
    query,
    key,
    value,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_slot_indices,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    kv_heads,
    1.0 / math.sqrt(head_dim),
    block_size,
    top_k,
    inner_precise=4,
    softmax_lse_flag=False,
    input_layout="BNSD",
)
print(attn_out.shape, attn_out.dtype)
```

TND paged KV 时传入 `block_table`，且 `select_idx` 用 packed `[Nkv, T, topK]`、`input_layout="TND"`。

图模式：导入 `cann_ops_transformer` 后会注册 `MinimaxSparseAttentionSplitKv` 的 GE Converter。`build_k2q_csr` 是 host Python，不要放进 `torch.compile` 图里；先在 eager 里建好 CSR，再对主算子走图模式。
