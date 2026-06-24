# vLLM Ascend PR 8215 CausalConv1d Decode Alignment Audit

## 1. Reference

This audit uses the local checkout:

```text
external/vllm-ascend-pr8215
branch: pr-8215
```

Key files:

- `csrc/causal_conv1d_v310/op_host/causal_conv1d_v310_def.cpp`
- `csrc/causal_conv1d_v310/op_host/causal_conv1d_v310_tiling.cpp`
- `csrc/causal_conv1d_v310/op_kernel/causal_conv1d_v310.h`
- `vllm_ascend/_310p/ops/fla/gdn_310.py`

## 2. Decode Input Modes

The vLLM Ascend PR 8215 `runMode=1` decode/update path has three input interpretations:

| inputMode | Input | queryStartLoc | Meaning |
|---:|---|---|---|
| `2` | `[batch, dim]` | absent or empty | Non-MTP single-token decode. |
| `1` | `[batch, seqlen, dim]` | optional/empty | Fixed-length MTP/speculative decode. |
| `0` | `[cuSeqlen, dim]` | `[batch+1]` | Variable-length flattened decode. |

Host tiling first treats 2D `x` in `runMode=1` as `[batch, dim]`. If a valid
`queryStartLoc` exists and `queryStartLoc.size - 1 != x.shape[0]`, it switches
to `[cuSeqlen, dim]` varlen decode.

## 3. Reference Tiling

The reference does not split the flattened `B*S` token rows as the top-level task axis.
It uses:

```text
gridSize = batch * blocksPerSeq
task     = one batch segment + one dim tile
```

Each task derives the token range for one batch:

```text
inputMode=0: start=queryStartLoc[b], len=queryStartLoc[b+1]-queryStartLoc[b]
inputMode=1: start=b*seqLen,        len=seqLen
inputMode=2: start=b,               len=1
```

Therefore reference varlen decode is still a `batch x dim` split; tokens inside
one batch are processed inside the core task.

## 4. Risk In Batch-Group Flatten Copy

The reference processes one batch token range per task. Each `x` read is:

```text
x[start + t, c0 : c0 + dimTile]
```

So batch boundaries are explicit: a task only reads the current batch segment.

The current implementation copies a group of batches into one UB tile:

```text
rowStart = queryStartLoc[firstBatch]
rowEnd   = queryStartLoc[firstBatch + batchNumInLoop]
blockCount = rowEnd - rowStart
```

It then copies all flattened rows in `[rowStart, rowEnd)` at once. This is valid
only if:

- `queryStartLoc` is monotonic;
- flattened rows are continuous inside the batch group;
- every batch token count is legal;
- compute correctly reconstructs each batch's UB offset through `curRowStart - tileRowStart`.

When those conditions hold, the grouped copy is workable. It is riskier than the
reference because malformed `queryStartLoc`, empty batches, batch-group boundary
cases, or a later token-limit change can corrupt multiple batches in one tile.
Strict PR 8215 alignment may favor the reference-style per-batch streaming copy.

## 5. num_accepted_tokens Alignment Status

The reference speculative decode condition is:

```text
runMode == 1 && hasNumAcceptedTokens && width == 4
```

In that path:

```text
stateTokenOffset = clamp(numAcceptedTokens[b] - 1, 0, stateLen - (width - 1))
```

`InitRing` loads the convolution history window from
`convStates[slot, stateTokenOffset : stateTokenOffset + width - 1, :]`.
Therefore `numAcceptedTokens` can affect computed `y`.

The early implementation mainly used `num_accepted_tokens` for state update:

```text
new_state = last state_len rows of concat(old_state, x[0:accepted])
```

This was not fully aligned with PR 8215. The current implementation has been
corrected as follows:

- When `num_accepted_tokens` is present, `accepted - 1` selects the GM `W-1`
  history window, so it affects `y`.
- The W=4 speculative path writes `state[0:2] = selected_history[1:3]` and
  `state[2:2+token_count] = x chunk`.
- When `num_accepted_tokens` is absent, normal decode uses
  `conv_states[slot,0:W-1]` as compact history.

## 6. Current Gaps

| Item | vLLM Ascend PR 8215 | Current implementation | Result |
|---|---|---|---|
| Non-MTP decode | Supports `[batch, dim]` | Added 2D `x` without `query_start_loc` path | Pending CANN validation |
| Fixed MTP | Supports `[batch,seqlen,dim]` | Supports `[B,S,D]` | Mostly aligned |
| Varlen decode | Supports `[cuSeqlen,dim]+queryStartLoc` | Supports `[T,D]+query_start_loc` | Shape aligned, copy strategy differs |
| dtype | FP16 | BF16 | Custom |
| metadata dtype | INT64 | INT32 | Custom |
| width | 2..4; `numAcceptedTokens` speculative writeback only width=4 | W=4 only for the current baseline | Intentional scope reduction; W4 decode semantics aligned |
| SiLU | Supports activationMode 0/1 | Supports 0 only | Missing |
| initialStateMode | Supported | Not supported | User-confirmed exclusion |
| runMode | Supports 0/1 | Not exposed | Decode-only design |
| numAcceptedTokens affects y | Yes | Added history-window offset | Pending CANN validation |
| W=4 speculative state layout | `state[0:2] + x chunk` | Added `state[0:2] + x chunk` | Pending CANN validation |

## 7. Recommended Fix Order

1. Done: update the spec to define the three decode input modes.
2. Done: update the reference with `[B,D]` and vLLM-style `numAcceptedTokens` history-window offset semantics.
3. Done: update host tiling so 2D `x` without `query_start_loc` uses `[B,D]` with `seqLen=1`.
4. Done: update the kernel so `num_accepted_tokens` offsets state-history loading through `accepted-1`.
5. Done: add the PR 8215 width=4 speculative writeback path, keeping two history rows selected by `accepted-1` and writing current tokens from state row 2.
6. Current main path stays with BH tiling and does not switch to BSH. BSH remains a future performance experiment for long-token or small `B x dimBlocks` shapes.
7. Current implementation keeps batch-group copy. A stricter vLLM-style per-batch streaming copy should be evaluated separately with correctness and performance data.
