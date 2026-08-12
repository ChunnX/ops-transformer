# Golden Dump .pt 文件说明

> 启用方式：运行测试时设置环境变量 `GOLDEN_DUMP=1`
>
> Dump 目录：默认 `{cwd}/golden_dumps/run_{pid}_{timestamp}/`，可通过 `GOLDEN_DUMP_DIR` 自定义

---

## 一、输入数据（make_case 阶段）

| 文件名 | 数据说明 | dtype | shape |
|---|---|---|---|
| `input_query.pt` | Query 张量 | bf16 | `[total_q, q_heads, head_dim]` |
| `input_key.pt` | Key 张量（paged KV cache） | bf16 | `[num_blocks, block_size, kv_heads, head_dim]` |
| `input_value.pt` | Value 张量（paged KV cache） | bf16 | `[num_blocks, block_size, kv_heads, head_dim]` |
| `input_block_table.pt` | 物理 block 映射表 | int32 | `[batch, max_blocks_per_batch]` |
| `input_k2q_row_ptr.pt` | CSR 行指针 | int32 | `[kv_heads, num_kv_blocks + 1]` |
| `input_k2q_q_indices.pt` | CSR 列索引（Q token id） | int32 | `[kv_heads, nnz_upper]` |
| `input_k2q_slot_indices.pt` | CSR slot 索引（topK 槽位） | int32 | `[kv_heads, nnz_upper]` |
| `input_actual_seq_lengths.pt` | Q 序列长度 | int32 | `[batch]` |
| `input_actual_seq_lengths_kv.pt` | KV 序列长度 | int32 | `[batch]` |
| `input_select_idx.pt` | block 选择索引 | int32 | `[total_q, top_k]` |
| `input_select_num_idx.pt` | 每 token 有效 block 数 | int32 | `[total_q]` |
| `input_scale_value.pt` | scale 标量 | float | 标量 |
| `input_block_size.pt` | block_size 标量 | int | 标量 |
| `input_top_k.pt` | top_k 标量 | int | 标量 |
| `input_kv_heads.pt` | kv_heads 标量 | int | 标量 |

---

## 二、Phase1 逐 block 中间结果

每个 KV block 对每个 `(q_token, kv_head, slot_k)` 组合 dump 5 个文件。
tag 格式：`q{q}_h{h}_t{t}`，其中 `q`=Q token id，`h`=kv_head index，`t`=slot_k。

| 文件名 | 数据说明 | dtype | shape |
|---|---|---|---|
| `scores_S_q{q}_h{h}_t{t}.pt` | S = Q @ K^T * scale | fp32 | `[group_size, valid_len]` |
| `softmax_P_q{q}_h{h}_t{t}.pt` | P = exp(S - rowMax) | bf16 | `[group_size, valid_len]` |
| `pv_Opartial_q{q}_h{h}_t{t}.pt` | O_partial = P @ V（未归一化） | fp32 | `[group_size, head_dim]` |
| `rowMax_q{q}_h{h}_t{t}.pt` | 每 row 的 max | fp32 | `[group_size]` |
| `rowSum_q{q}_h{h}_t{t}.pt` | 每 row 的 sum | fp32 | `[group_size]` |

> 无效行（row_sum <= 0）的 rowMax 设为 -inf，rowSum 设为 0。

---

## 三、Phase1 workspace 汇总

| 文件名 | 数据说明 | dtype | shape |
|---|---|---|---|
| `phase1_ws_o.pt` | 所有 O_partial 汇总 | fp32 | `[total_q, kv_heads, top_k, group_size, head_dim]` |
| `phase1_ws_max.pt` | 所有 rowMax 汇总 | fp32 | `[total_q, kv_heads, top_k, group_size]` |
| `phase1_ws_sum.pt` | 所有 rowSum 汇总 | fp32 | `[total_q, kv_heads, top_k, group_size]` |

---

## 四、最终输出

| 文件名 | 数据说明 | dtype | shape |
|---|---|---|---|
| `golden_output.pt` | Phase2 IFA 合并后的最终 attention 输出 | bf16 | `[total_q, q_heads, head_dim]` |

---

## 用法示例

```bash
cd /opt/rrh/code3/ops-transformer/attention/sparse_attention_score_prefill
GOLDEN_DUMP=1 python -m pytest -q tests/test_sparse_attention_score_prefill_npu.py -s -k test_bf16_minimal_debug_2
```

加载 .pt 文件：

```python
import torch
s = torch.load("scores_S_q0_h0_t0.pt")  # fp32 [group_size, valid_len]
```
