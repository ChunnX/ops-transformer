# CausalConv1dDecode

面向 MTP 小 chunk 的 decode-only causal depthwise conv1d 算子。

本算子只保留 decode 路径。参数命名借鉴 `attention/fused_causal_conv1d` 的公开接口，但删除 prefill、prefill/decode 混合和 APC 相关字段。

## 支持范围

- BF16 first。
- 固定卷积宽度 `K=4`。
- `seq_len` 在 `[1, 4]`。
- `x`：`[batch, seq_len, dim]`。
- `weight`：`[4, dim]`。
- `conv_states`：`[num_cache_slots, state_len, dim]`，原地更新。
- `cache_indices`：`[batch]`。
- 可选 `bias`：`[dim]`。
- 可选 `num_accepted_tokens`：`[batch]`。
- 输出 `y`：`[batch, seq_len, dim]`。

## v1 不支持

- prefill。
- prefill/decode 混合。
- APC block mapping。
- `run_mode`。
- 可变卷积宽度。

## 文档

- 英文 API：`docs/aclnnCausalConv1dDecode.md`
- 中文 API：`docs/aclnnCausalConv1dDecode.zh.md`
- 可回溯记录：`reports/work_log.md`

