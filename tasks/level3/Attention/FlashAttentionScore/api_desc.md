# FlashAttentionScore

## 功能

融合实现的 Flash Attention 前向打分：根据 `query`/`key`/`value` 计算注意力，`softmax_max`、`softmax_sum`、`softmax_out`、`attention_out` 为输出。

## 主要约束

- `input_layout` 支持 `BSH`、`SBH`、`BSND`、`BNSD`、`TND`（大小写不敏感）；本基准默认验证 **`BNSD`**。
- 精度验证侧使用 `aclnnFlashAttentionScore` / `aclnnFlashAttentionScoreGetWorkspaceSize`，可选输入（mask、rope、prefix 等）传空。

## aclnn 接口

参见 CANN `aclnn_flash_attention_score.h`：`aclnnFlashAttentionScoreGetWorkspaceSize` 与 `aclnnFlashAttentionScore`。
