# NsaCompressAttention（NSA Compress + TopK）

## 功能

实现 NSA 中的 **compress attention** 与 **importance score / topk 索引** 路径：对 `query`/`key`/`value`（当前 Bench 验证路径为 **TND**、`FLOAT16`、无 `attenMask`/`topkMask`）计算 softmax 统计、`attention_out` 与 `topk_indices_out`。

## 主要参数

- `query`：`[N2, T, G, D1]`（TND 内部为 4 维；与 `head_num = N2*G` 一致）
- `key` / `value`：`[T2, N2, D1]` / `[T2, N2, D2]`
- `actual_seq_qlen` / `actual_cmp_seq_kvlen` / `actual_sel_seq_kvlen`：长度前缀和（本任务单段用例为长度 1、元素为序列总长）
- `scale_value`、`head_num`、`input_layout`（`"TND"`）、`sparse_mode`
- `compress_block_size`、`compress_stride`、`select_block_size`、`select_block_count`

## 输出

- `softmax_max` / `softmax_sum`：`[T, N1, 8]`（`N1 = N2*G`），`FLOAT32`
- `attention_out`：`[N2, T, G, D2]`，与 `query` dtype 一致
- `topk_indices_out`：`[N2, T, select_block_count]`，`INT32`

## aclnn

`aclnnNsaCompressAttentionGetWorkspaceSize` / `aclnnNsaCompressAttention`（两段式）。
