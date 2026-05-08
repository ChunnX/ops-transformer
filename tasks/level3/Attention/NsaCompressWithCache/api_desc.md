# NsaCompressWithCache（NSA KV 压缩写 Cache）

## 功能

推理阶段按 **compress_block_size / compress_stride** 将满足条件的 batch 尾部 KV 片段与 **weight** 做乘加归约，写入 **output_cache** 的 `slot_mapping[batch]` 行（in-place 更新同一张 cache 张量）。

## 主要参数

- `input`：`[blocks_num, page_block_size, N, D]`（与 `page_block_size` 属性一致）
- `weight`：`[compress_block_size, N]`
- `slot_mapping`：`[batch]`，`INT32`，写入行索引
- `output_cache`：输入/输出一体，`[result_len, N, D]`
- `act_seq_len`（可选）：`[batch]`，`INT64`，各 batch 当前实际 token 数（本 Bench 用例为必选）
- `block_table`（可选）：`[batch, page_num]`，`INT32`，页块到 `input` 第一维下标的映射
- `layout`：如 `"TND"`
- `compress_block_size`、`compress_stride`、`act_seq_len_type`（当前 tiling 仅支持 `1`）、`page_block_size`

## aclnn

`aclnnNsaCompressWithCacheGetWorkspaceSize` / `aclnnNsaCompressWithCache`（两段式）。
