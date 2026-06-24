# aclnnCausalConv1dDecode

## 用途

`aclnnCausalConv1dDecode` 计算 decode-only causal depthwise conv1d，支持 width 4，并原地更新 `conv_states`。

## 接口

```cpp
aclnnStatus aclnnCausalConv1dDecodeGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    aclTensor *convStates,
    const aclTensor *queryStartLoc,
    const aclTensor *cacheIndices,
    const aclTensor *bias,
    const aclTensor *numAcceptedTokens,
    int64_t activationMode,
    int64_t padSlotId,
    int64_t convMode,
    aclTensor *y,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnCausalConv1dDecode(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 输入

| 名称 | 必选 | 类型 | Shape | 说明 |
|---|---:|---|---|---|
| `x` | 是 | BF16 | `[batch, dim]`、`[batch, seq_len, dim]` 或 `[total_tokens, dim]` | Decode chunk。`[batch,dim]` 为非 MTP 单 token decode；固定 batch 模式 `seq_len` 为 1..4；flatten 模式由 `queryStartLoc` 划分每个 batch。 |
| `weight` | 是 | BF16 | `[width, dim]` | causal conv1d 权重，`width` 固定为 4，顺序为 oldest 到 newest。 |
| `conv_states` | 是 | BF16 | `[num_cache_slots, state_len, dim]` | cache state，原地更新。 |
| `queryStartLoc` | 否 | INT32 | `[batch + 1]` | `x` 为 `[total_tokens, dim]` 时必选，用于描述每个 batch 的 token 范围。 |
| `cache_indices` | 否 | INT32 | `[batch]` | batch 行到 cache slot 的映射；缺省时 `slot = batch_id`。 |
| `bias` | 否 | BF16 | `[dim]` | 可选 bias。 |
| `num_accepted_tokens` | 否 | INT32 | `[batch]` | 可选 MTP 接受 token 数；存在时会偏移 `y` 使用的 state 历史窗口，`W=4` 时也选择 speculative state writeback 路径。 |

## 属性

| 名称 | 类型 | 默认 | 说明 |
|---|---|---:|---|
| `activationMode` | INT64 | 0 | v1 只支持 `0 = none`。 |
| `padSlotId` | INT64 | -1 | 命中该 cache slot 时跳过。 |
| `convMode` | INT64 | 0 | 预留字段，v1 只支持 `0`。 |

## 输出

| 名称 | 类型 | Shape | 说明 |
|---|---|---|---|
| `y` | BF16 | 与 `x` token 布局一致 | Decode 输出。 |

`conv_states` 会原地更新，但 ACLNN 公开输出只有 `y`。
