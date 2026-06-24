# AscendC 算子数据流规范

本算子按 A5 风格组织数据流：GM 持久张量进入 UB，VF/reg 在连续 dim 子片上计算，再从 UB 写回 GM。

当前范围：

- BF16 输入输出，FP32 累加。
- decode/update only。
- `weight` width 当前仅支持 4。
- `seqLen` 或每 batch token 数支持 1..4。
- `conv_states` 是原地更新副作用，公开输出只有 `y`。

## 1. 总体数据流

```text
Host tiling
  读取 shape/attr，确定 B / token_count source / D / W / state_len
  生成每个 core 的 batch range 和 dim range

        |
        v

Inter-core split
  core owns:
    batch range [firstBatchIdx, firstBatchIdx + batchNum)
    dim range   [dimOffset, dimOffset + coreDimLen)
  token/s 维不跨 core 拆分

        |
        v

GM tensors
  x[b, s, d]
  conv_state[slot, h, d]
  weight[k, d]
  bias[d]

        |
        | DataCopy 当前 core / 当前 dimLoop 负责的 d0:d0+D0
        v

UB buffers
  x_ub[s, d_local]
  state_ub[h, d_local]        # 只搬 W-1 行历史窗口
  weight_ub[k, d_local]       # 每个 dimLoop 搬一次，跨 batch 复用
  bias_ub[d_local]            # 每个 dimLoop 搬一次，跨 batch 复用

        |
        | VF 按 64 个 d_local 一组计算
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
  conv_state[slot, h, d] 更新
```

这里的 `[T,D]` 变长输入只是存储行表示；核间分核不把 `B*T` 当作独立任务轴。每个 core 先确定 batch 范围，再在核内根据固定 `S` 或 `query_start_loc[b+1]-query_start_loc[b]` 得到该 batch 的 token rows。

## 2. 核间分核图

```text
逻辑任务空间 = batch x dim

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

`cache_indices` 只改变 batch 到 `conv_states` slot 的映射，不改变当前 core 拥有的 batch 范围。

## 3. 核内 UB tile 图

```text
当前 core 的责任区域继续切成 UB tile：

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

## 4. GM 到 UB 搬运连续性

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

结论：`x` 每一行内部连续搬运，行间非连续，UB 内连续压实。

### 4.2 weight 和 bias

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

结论：weight 每个 tap 的 dim 子片连续，tap 行之间按 `D` stride；bias 是单行连续搬运。

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

结论：state 是每个 batch 根据 slot 单独搬；decode 计算只需要 `W-1` 行历史窗口，因此不搬完整 `state_len`。每行内部连续，行间按 `D` stride，UB 内 compact。

## 5. VF/register 计算图

VF 看到的输入都是 UB 中连续的 `dimTile`。VF 不是逐 channel 标量算一个 window，而是在同一个 window tap `k` 上一次处理 64 个 dim：

```text
当前 token s 的窗口：

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

`input_for_s_k` 的来源由 causal window 决定：

```text
rel = s + k - (W - 1)
rel < 0  -> state_ub[rel + W - 1, d_local]
rel >= 0 -> x_ub[rel, d_local]
```

## 6. y 和状态写回

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

状态写回：

```text
old state window:
  [h0][h1][h2]              # W=4 示例，W-1=3

accepted x rows:
  [x0][x1]

default new state window:
  suffix([h0][h1][h2][x0][x1], W-1) = [h2][x0][x1]

store:
  conv_states[slot, 0:W-1, d0:d0+dimTile] = [h2][x0][x1]
```

每个框都是当前 dim tile。写回 GM 时每行内部连续，行间按 D stride。

W=4 且存在 `num_accepted_tokens` 的 vLLM-style speculative layout 另有对齐路径：写回保留 2 行历史，并把当前 token chunk 写入 `conv_states[slot, 2:2+token_count]`，以便下一轮通过新的 accepted offset 切出逻辑历史。

## 7. 同步边界

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

metadata `query_start_loc/cache_indices/num_accepted_tokens` 使用 `TBuf`，GM 搬入后通过 `SetFlag/WaitFlag` 保证 scalar 读取发生在 MTE2 完成之后。
