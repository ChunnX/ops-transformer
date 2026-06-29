# CausalConv1dDecode AscendC Operator Spec Summary

Full requirements:

- `docs/REQUIREMENTS.zh.md`
- `docs/REQUIREMENTS.en.md`

This file keeps only the compact spec-gate summary to avoid duplicated requirements drifting from the wiki documents.

## 1. Semantics

- Name: `CausalConv1dDecode`
- Backend: Ascend C, A5/VF-style implementation.
- Mode: decode/update only.
- Input/output dtype: BF16.
- Accumulator dtype: FP32.
- Public output: `y`.
- Side effect: in-place `conv_states` update.

## 2. Inputs And Attributes

- `x`: BF16, with three decode input modes:
  - `[B,D]`: non-MTP single-token decode, one token per batch.
  - `[B,S,D]`: fixed-length MTP/speculative decode, where `S` is the candidate-token count per batch.
  - `[T,D] + query_start_loc`: variable-length MTP/speculative decode, where `T` is the flattened token-row count.
- `weight`: BF16, `[W,D]`, where `W` is 2 to 4.
- `conv_states`: BF16, `[num_slots,state_len,D]`, updated in place.
- `query_start_loc`: optional INT32, `[B+1]`; required when `x` is `[T,D]`.
- `cache_indices`: optional INT32, `[B]`; default is `slot=b`.
- `bias`: optional BF16, `[D]`.
- `num_accepted_tokens`: optional INT32, `[B]`; default is that batch's token count.
- `activation_mode`: v1 supports `0` only.
- `pad_slot_id`: skip a batch when its selected slot matches this value.
- `conv_mode`: v1 supports `0` only.

Excluded: `initialStateMode`, `runMode`, and prefill/APC/block-mapping parameters.

## 3. Shape Domain

- `1 <= B <= 256`
- `1 <= S <= 4`
- In `[B,D]` mode, `token_count = 1`.
- In `[B,S,D]` mode, `1 <= S <= 4`.
- In `[T,D] + query_start_loc` variable storage mode, `token_count = query_start_loc[b+1] - query_start_loc[b]` should be in `1..4`.
- `64 <= D <= 16384` and `D % 16 == 0`
- `W == 4`
- Normal decode: `state_len >= W - 1`
- W=4 speculative layout with `num_accepted_tokens`: `state_len >= 2 + max_token_count`

## 4. Compute Definition

For batch `b`:

```text
row_start = b                             # x is [B,D]
row_start = b * S                         # x is [B,S,D]
row_start = query_start_loc[b]            # x is [T,D]
row_end   = row_start + token_count
slot      = cache_indices[b] or b
accepted  = num_accepted_tokens[b] or token_count
accepted  = clamp(accepted, 0, token_count)
```

For token `s` and channel `d`:

```text
acc = bias[d] if bias exists else 0
hist_base = 0 if num_accepted_tokens absent
hist_base = clamp(accepted - 1, 0, state_len - (W - 1)) if num_accepted_tokens exists
for k in 0..W-1:
    rel = s - (W - 1) + k
    v = conv_states[slot, hist_base + rel + W - 1, d] if rel < 0 else x[row_start + rel, d]
    acc += fp32(v) * fp32(weight[k, d])
y[row_start + s, d] = bf16(acc)
```

State update:

```text
default:
    hist = conv_states[slot, hist_base : hist_base + W - 1]
    conv_states[slot, 0 : W - 1] = last W - 1 rows of concat(hist, x[row_start : row_start + accepted])

vLLM-style speculative path when num_accepted_tokens exists and W == 4:
    hist_base = clamp(accepted - 1, 0, state_len - 3)
    conv_states[slot, 0:2] = old_state[hist_base + 1 : hist_base + 3]
    conv_states[slot, 2 : 2 + token_count] = x[row_start : row_end]
```

## 5. Tiling And Dataflow Summary

Overall dataflow:

```text
GM tensors
  x[b, s, d]
  conv_state[slot, h, d]
  weight[k, d]
  bias[d]
        |
        | DataCopy d0:d0+D0 owned by the current core / dimLoop
        v
UB buffers
  x_ub[s, d_local]
  state_ub[h, d_local]        # only W-1 history rows
  weight_ub[k, d_local]       # copied once per dimLoop, reused across batches
  bias_ub[d_local]            # copied once per dimLoop, reused across batches
        |
        | VF computes one 64-d_local chunk at a time
        v
Reg / VF
  acc_s0[64]
  acc_s1[64]
  ...
        |
        | Store
        v
GM outputs / side effects
  y[b, s, d]
  conv_state[slot, h, d] update
```

- Inter-core: split `batch x dim`, not token.
- Intra-core: `dimLoop -> batchLoop` forms UB tiles, reusing `weight/bias` across batchLoop iterations under the same dimLoop.
- GM to UB:
  - `x`: contiguous dim sub-slice per row, row stride `xStride=D`, compact in UB.
  - `weight`: `[W, dimTile]`, row stride `D`, compact in UB; copied once per dimLoop.
  - `conv_states`: per selected `slot`, copy only the `[W-1, dimTile]` history window; normal decode copies from `conv_states[slot,0:W-1]`, while `num_accepted_tokens` copies from the `accepted-1` offset.
  - `bias`: copy `[dimTile]`, then broadcast in compute; copied once per dimLoop.
- VF/register: when `tokenCount == 1`, the kernel uses the single-S VF path; when `tokenCount >= 2`, it uses the multi-S VF path, which processes up to 4 token rows from the same batch over a contiguous dim tile. Both paths loop internally by register width:
  - `reg_elems = 256 / sizeof(float) = 64` FP32 lanes.
  - `dimLoopNum = dimSizeInLoop / 64` uses the full-mask main loop.
  - `dimRem = dimSizeInLoop % 64` uses the `UpdateMask(dimRem)` tail loop.
  - VF does not compute one scalar channel/window at a time; for each window tap `k`, it processes 64 dim lanes together.
  - W is fixed to 4; the AICore wrapper calls only `Conv1dDecodeW4` or `Conv1dDecodeW4MultiS`.
  - The single-S path keeps one FP32 accumulator per VF sub-loop.
  - The multi-S path keeps up to four FP32 accumulators per VF sub-loop and reuses the same weights plus overlapping state/x timeline values.
  - Each VF sub-loop performs BF16 load, cast to FP32, multiply/add, cast to BF16, and store.
- Synchronization: `xQueue`, `stateQueueIn`, and `yQueue` use `EnQue/DeQue` as per-tile sync boundaries; `weightBiasBuf` and metadata use `TBuf` plus `SetFlag/WaitFlag` after GM copy.

### 5.1 Current Tiling Decision

The current main path always uses BH-style tiling:

```text
task space = batch x dim
token/S axis stays inside the core
```

All three decode input modes use the same BH kernel:

```text
[B,D]                   -> token_count = 1
[B,S,D]                 -> token_count = S
[T,D] + query_start_loc -> token_count = query_start_loc[b+1] - query_start_loc[b]
```

BSH is not the current main path. A BSH `flattened_token_rows x dim` split can
offer more parallelism for long sequences, prefill, mixed mode, or when `sum(S)`
is much larger than `B`. This operator is decode-only with 1..4 tokens per batch,
so `batch x dim` usually provides enough tasks. BSH would add row-to-batch/s
mapping, batch-boundary handling, cross-core state update merging, and sync
overhead. It is kept as a future performance experiment, not part of the current
functional-alignment path.

Continuity thumbnail:

```text
GM row: | d0 ... d0+dimTile-1 | skipped D-dimTile |
        |---- contiguous ----|<------- stride ---->|

UB:     | row0 dimTile | row1 dimTile | row2 dimTile |
        compact, no row gap

VF:     | 0..63 | 64..127 | ... | tail |
          full    full            UpdateMask(dimRem)
```

## 6. Correctness And Validation

Coverage must include:

- `[B,D]`, `[B,S,D]`, and `[T,D] + query_start_loc`
- `W=4`
- token counts `1..4`
- `cache_indices` absent/present
- `num_accepted_tokens` absent/zero/partial/full acceptance
- `bias` absent/present
- dim main/tail tiles
- `pad_slot_id` hit/miss

Current limits:

- CANN compile validation has not been completed in the current environment.
- SiLU is not implemented in v1.
- Host tiling cannot statically validate runtime `query_start_loc` differences.
- The current code now supports `[B,D]` non-MTP decode input; CANN compile
  validation is still pending.
- The current code now uses `num_accepted_tokens` to offset the state-history
  read window, so it can affect `y`. When `W == 4` and `num_accepted_tokens`
  exists, the code also implements the vLLM Ascend PR 8215-style speculative
  state writeback path. CANN compile validation is still pending.
