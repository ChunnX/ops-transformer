# aclnnCausalConv1dDecode

## Purpose

`aclnnCausalConv1dDecode` computes a decode-only causal depthwise conv1d with width 4 and updates
`conv_states` in-place.

## Signature

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

## Inputs

| Name | Required | Type | Shape | Description |
|---|---:|---|---|---|
| `x` | yes | BF16 | `[batch, seq_len, dim]` or `[total_tokens, dim]` | Decode chunk. Fixed-batch `seq_len` is 1..4; flattened mode is partitioned by `queryStartLoc`. |
| `weight` | yes | BF16 | `[width, dim]` | Causal conv1d weights, `width` is fixed to 4, ordered oldest to newest. |
| `conv_states` | yes | BF16 | `[num_cache_slots, state_len, dim]` | Cache state, updated in-place. |
| `queryStartLoc` | no | INT32 | `[batch + 1]` | Required when `x` is `[total_tokens, dim]`; maps each batch to a token range. |
| `cache_indices` | no | INT32 | `[batch]` | Maps batch rows to cache slots; default is `slot = batch_id`. |
| `bias` | no | BF16 | `[dim]` | Optional bias. |
| `num_accepted_tokens` | no | INT32 | `[batch]` | Optional MTP accepted-token count. When present, it offsets the state-history window used for `y`; for `W=4` it also selects the speculative state writeback path. |

## Attributes

| Name | Type | Default | Description |
|---|---|---:|---|
| `activationMode` | INT64 | 0 | v1 supports `0 = none` only. |
| `padSlotId` | INT64 | -1 | Cache slot id to skip. |
| `convMode` | INT64 | 0 | Reserved. v1 supports `0` only. |

## Output

| Name | Type | Shape | Description |
|---|---|---|---|
| `y` | BF16 | same token layout as `x` | Decode output. |

`conv_states` is updated in-place, but the ACLNN public output is `y` only.

## Semantics

For each `(b, s, d)`:

```text
slot = cache_indices[b] if cache_indices exists else b
window = width-4 causal window, oldest to newest
acc = bias[d] if bias exists else 0
for k in 0..3:
    acc += window[k] * weight[k, d]
y[b, s, d] = bf16(acc)
```

Normal decode state update preserves the latest `3` history rows in
`conv_states[slot, 0:3, :]`. In the W=4 speculative path,
`numAcceptedTokens[b] - 1` selects the history read offset, `conv_states[slot,
0:2, :]` stores the selected two history rows, and `conv_states[slot, 2:2+len,
:]` stores the current token chunk.
