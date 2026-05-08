# IncreFlashAttention

## 功能

增量场景下的 Flash Attention：Query 的序列长度通常为 1，Key/Value 为带 KV Cache 的填充张量，通过 `actual_seq_lengths` 描述各 batch 实际有效 KV 长度。

## aclnn 两段式接口

- `aclnnIncreFlashAttentionGetWorkspaceSize`
- `aclnnIncreFlashAttention`

主要参数（与算子定义一致）：`query`；动态输入 `key` / `value`（`aclTensorList`）；可选 `pse_shift`、`atten_mask`、`actual_seq_lengths`；属性 `num_heads`、`scale_value`、`input_layout`、`num_key_value_heads`；输出 `attention_out`。

## 约束（本任务 validation）

- 数据类型：FP16，`layout=BNSD`。
- `key`/`value` 各提供单张量列表（长度为 1 的 TensorList）。
- 单用例用于编译与精度冒烟；完整场景需扩展 `test_cases.csv` 与参考实现。
