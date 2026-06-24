# AscendC Operator Dataflow Spec

This operator uses an A5-style dataflow: persistent tensors live in GM, tiles are copied into UB, VF/reg computes over contiguous dim slices, then results are written back from UB to GM.

Current scope:

- BF16 input/output, FP32 accumulation.
- Decode/update only.
- `weight` width is fixed to 4.
- `seqLen` or per-batch token count 1..4.
- `conv_states` is an in-place side effect. The public output is `y` only.

## 1. Overall Dataflow

```text
Host tiling
  Read shapes/attrs and derive B / token-count source / D / W / state_len.
  Generate one batch range and one dim range per core.

        |
        v

Inter-core split
  core owns:
    batch range [firstBatchIdx, firstBatchIdx + batchNum)
    dim range   [dimOffset, dimOffset + coreDimLen)
  token/s is not split across cores.

        |
        v

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
  acc_s(token_count-1)[64]

  for each k in 0..W-1:
    acc_s += input_for_s_k[64] * weight_ub[k, 64]

        |
        | Store
        v

GM outputs / side effects
  y[b, s, d]
  conv_state[slot, h, d] update
```

The variable `[T,D]` input mode is only a storage-row representation. Inter-core tiling does not use `B*T` as an independent task axis. Each core first owns a batch range, then derives per-batch token rows inside the core from fixed `S` or `query_start_loc[b+1]-query_start_loc[b]`.

## 2. Inter-Core Split

```text
Logical work space = batch x dim

                         dim / channel
              0        dimOffset               dimOffset+coreDimLen        D
              |------------|===========================|-------------------|
batch b0      | other core |        core k dim range   |     other core    |
batch b1      | other core |        core k dim range   |     other core    |
              |------------|===========================|-------------------|
              firstBatchIdx          batchNum rows

core k:
  batch interval is contiguous in logical batch id
  dim interval is contiguous in channel id
  token/seq axis is not split
```

`cache_indices` only maps a batch to a `conv_states` slot. It does not change the batch range owned by the core.

## 3. Intra-Core UB Tile

```text
The current core's ownership is further split into UB tiles:

        dimLoop=0          dimLoop=1                  dimLoop=N
     |===============| |===============|        |===============|
b0   | UB tile       | | UB tile       |  ...   | UB tail tile   |
b1   | batchLoop=0   | | batchLoop=0   |        | batchLoop=0   |
     |===============| |===============|        |===============|
b2   | UB tile       | | UB tile       |  ...   | UB tail tile   |
b3   | batchLoop=1   | | batchLoop=1   |        | batchLoop=1   |
     |===============| |===============|        |===============|

UB tile rows:
  fixed [B,S,D] mode: batchNumInLoop * S
  variable [T,D] storage mode: query_start_loc[first+batchNumInLoop] - query_start_loc[first]

UB tile cols:
  dimSizeInLoop, always contiguous in D/channel direction
```

## 4. GM To UB Movement Continuity

### 4.1 x

```text
GM x, row-major by D:

row r0: | d0 ... d0+dimTile-1 | skipped D-dimTile |
        |---- contiguous ----|<----- srcStride --->|
row r1: | d0 ... d0+dimTile-1 | skipped D-dimTile |
row r2: | d0 ... d0+dimTile-1 | skipped D-dimTile |

DataCopyPad:
  blockCount = row_count
  blockLen   = dimSizeInLoop * sizeof(bf16)
  srcStride  = (xStride - dimSizeInLoop) * sizeof(bf16)
  dstStride  = 0

UB xLocal:
  | r0 dimTile | r1 dimTile | r2 dimTile | ...
  compact, no row gap
```

Conclusion: `x` is contiguous within each row, strided between rows, and compact in UB.

### 4.2 weight And bias

```text
GM weight [W,D], W=4:

w0: | d0 ... d0+dimTile-1 | skipped D-dimTile |
w1: | d0 ... d0+dimTile-1 | skipped D-dimTile |
w2: | d0 ... d0+dimTile-1 | skipped D-dimTile |  # W>=3
w3: | d0 ... d0+dimTile-1 | skipped D-dimTile |  # W>=4

UB weightLocal:
  | w0 dimTile | w1 dimTile | w2 dimTile | w3 dimTile |

GM bias [D]:
  | d0 ... d0+dimTile-1 |

UB biasLocal:
  | bias dimTile |
```

Conclusion: each weight tap has a contiguous dim sub-slice, tap rows are separated by stride `D`, and bias is a single contiguous row.

### 4.3 conv_states

```text
cache_indices[b] = slot

GM conv_states [num_slots, state_len, D]

slot 5:
  h0: | d0 ... d0+dimTile-1 | skipped D-dimTile |
  h1: | d0 ... d0+dimTile-1 | skipped D-dimTile |
  ...
  hN: | d0 ... d0+dimTile-1 | skipped D-dimTile |

DataCopyPad:
  src base   = slot * cacheStride0 + historyBase * D + dimOffset + dimOffsetPerLoop
  blockCount = W - 1
  blockLen   = dimSizeInLoop * sizeof(bf16)
  srcStride  = (D - dimSizeInLoop) * sizeof(bf16)
  dstStride  = 0

UB stateLocal:
  | history0 dimTile | history1 dimTile | ... | history(W-2) dimTile |
```

Conclusion: state is copied per batch according to `slot`. Decode computation only needs the `W-1` history-row window, so the full `state_len` is not copied. Each row is contiguous, rows are strided by `D`, and UB storage is compact.

## 5. VF/Register Compute

VF sees only contiguous `dimTile` slices in UB. It does not compute one scalar channel/window at a time; for each window tap `k`, it processes 64 dim lanes together:

```text
Window for current token s:

src0 -> state/x row for oldest tap  | 0..63 | 64..127 | ... | tail |
src1 -> state/x row                 | 0..63 | 64..127 | ... | tail |
src2 -> state/x row, W>=3           | 0..63 | 64..127 | ... | tail |
src3 -> state/x row, W>=4           | 0..63 | 64..127 | ... | tail |
weight rows                         | w0    | w1      | ... |
bias                                | 0..63 | 64..127 | ... | tail |
y                                   | 0..63 | 64..127 | ... | tail |

VF loop:
  reg_elems = 64 FP32 lanes
  for offset in 0, 64, 128, ...
    acc_s0[64] = bias[offset:offset+64]
    acc_s1[64] = bias[offset:offset+64]   # if token_count >= 2
    ...

    for k in 0..W-1:
      load weight_ub[k, offset:offset+64]
      load input_for_s0_k[offset:offset+64]
      load input_for_s1_k[offset:offset+64]  # if token_count >= 2
      acc_s += input_for_s_k * weight_ub[k]

    Cast FP32 -> BF16
    StoreAlign y_ub[s, offset:offset+64]
  tail uses UpdateMask<float>(dimRem)
```

`input_for_s_k` is selected by the causal window:

```text
rel = s + k - (W - 1)
rel < 0  -> state_ub[rel + W - 1, d_local]
rel >= 0 -> x_ub[rel, d_local]
```

## 6. y And State Writeback

```text
UB yLocal:
  | row0 dimTile | row1 dimTile | row2 dimTile | ...

GM y:
  row0: | d0 ... d0+dimTile-1 | skipped D-dimTile |
  row1: | d0 ... d0+dimTile-1 | skipped D-dimTile |

DataCopyPad:
  blockCount = yBlockCount
  blockLen   = dimSizeInLoop * sizeof(bf16)
  srcStride  = 0
  dstStride  = (D - dimSizeInLoop) * sizeof(bf16)
```

State writeback:

```text
old state window:
  [h0][h1][h2]              # W=4 example, W-1=3

accepted x rows:
  [x0][x1]

default new state window:
  suffix([h0][h1][h2][x0][x1], W-1) = [h2][x0][x1]

store:
  conv_states[slot, 0:W-1, d0:d0+dimTile] = [h2][x0][x1]
```

Each box is the current dim tile. GM writeback is contiguous within each row and strided by D between rows.

For the W=4 vLLM-style speculative layout when `num_accepted_tokens` exists, there is a separate alignment path: keep two history rows and write the whole current token chunk into `conv_states[slot, 2:2+token_count]`, so the next decode step can select logical history by the new accepted offset.

## 7. Synchronization Boundaries

```text
GM -> UB CopyIn
  weight/bias: DataCopyPad(...) into TBuf once per dimLoop, then SetFlag/WaitFlag
  x/state: DataCopyPad(...) into TQue, then EnQue(queue)

Compute
  DeQue(queue)
  VF/register compute
  EnQue(yQueue)

UB -> GM CopyOut
  DeQue(yQueue)
  DataCopyPad(...)
```

Metadata `query_start_loc/cache_indices/num_accepted_tokens` lives in `TBuf`. After GM copy, `SetFlag/WaitFlag` ensures scalar reads happen after MTE2 completion.
