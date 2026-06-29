# CausalConv1dDecode AscendC 算子规格摘要

完整需求文档见：

- `docs/REQUIREMENTS.zh.md`
- `docs/REQUIREMENTS.en.md`

本文件只保留 workflow/spec gate 需要快速读取的核心约束，避免和 wiki 需求文档重复漂移。

## 1. 算子语义

- 名称：`CausalConv1dDecode`
- 后端：Ascend C，A5/VF 风格实现。
- 场景：decode/update only。
- 输入输出 dtype：BF16。
- 累加 dtype：FP32。
- 公开输出：`y`。
- 副作用：`conv_states` 原地更新。

## 2. 输入和属性

- `x`：BF16，支持三种 decode 输入模式：
  - `[B,D]`：非 MTP 单 token decode，等价于每个 batch 一个 token。
  - `[B,S,D]`：定长 MTP/speculative decode，`S` 是每个 batch 的候选 token 数。
  - `[T,D] + query_start_loc`：变长 MTP/speculative decode，`T` 是所有 batch token 拼接后的总行数。
- `weight`：BF16，`[W,D]`，`W` 当前仅支持 4。
- `conv_states`：BF16，`[num_slots,state_len,D]`，原地更新。
- `query_start_loc`：可选 INT32，`[B+1]`；`x` 为 `[T,D]` 时必选。
- `cache_indices`：可选 INT32，`[B]`；缺省时 `slot=b`。
- `bias`：可选 BF16，`[D]`。
- `num_accepted_tokens`：可选 INT32，`[B]`；缺省时等于该 batch 的 token 数。
- `activation_mode`：v1 只支持 `0`。
- `pad_slot_id`：命中时跳过对应 batch。
- `conv_mode`：v1 只支持 `0`。

不包含：`initialStateMode`、`runMode`、prefill/APC/block mapping 相关参数。

## 3. Shape 范围

- `1 <= B <= 256`
- `1 <= S <= 4`
- `[B,D]` 模式下 `token_count = 1`。
- `[B,S,D]` 模式下 `1 <= S <= 4`。
- `[T,D] + query_start_loc` 变长存储模式下，每个 batch 的 `token_count = query_start_loc[b+1] - query_start_loc[b]` 应在 `1..4`。
- `64 <= D <= 16384`，且 `D % 16 == 0`
- `W == 4`
- 普通 decode：`state_len >= W - 1`
- W=4 且存在 `num_accepted_tokens` 的 speculative layout：`state_len >= 2 + max_token_count`

## 4. 计算定义

对 batch `b`：

```text
row_start = b                             # x is [B,D]
row_start = b * S                         # x is [B,S,D]
row_start = query_start_loc[b]            # x is [T,D]
row_end   = row_start + token_count
slot      = cache_indices[b] or b
accepted  = num_accepted_tokens[b] or token_count
accepted  = clamp(accepted, 0, token_count)
```

对 token `s`、channel `d`：

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

状态更新：

```text
default:
    hist = conv_states[slot, hist_base : hist_base + W - 1]
    conv_states[slot, 0 : W - 1] = last W - 1 rows of concat(hist, x[row_start : row_start + accepted])

vLLM-style speculative path when num_accepted_tokens exists and W == 4:
    hist_base = clamp(accepted - 1, 0, state_len - 3)
    conv_states[slot, 0:2] = old_state[hist_base + 1 : hist_base + 3]
    conv_states[slot, 2 : 2 + token_count] = x[row_start : row_end]
```

## 5. 分核和数据流摘要

总体数据流：

```text
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
        |
        | Store
        v
GM outputs / side effects
  y[b, s, d]
  conv_state[slot, h, d] 更新
```

- 核间：拆 `batch x dim`，不拆 token。
- 核内：`dimLoop -> batchLoop` 形成 UB tile，`weight/bias` 在同一个 dimLoop 内跨 batchLoop 复用。
- GM 到 UB：
  - `x`：每行 dim 子片连续，行间按 `xStride=D` 跳转，UB compact。
  - `weight`：`[W, dimTile]`，行间按 `D` 跳转，UB compact；每个 dimLoop 搬一次。
  - `conv_states`：按 `slot` 单 batch 只搬 `[W-1, dimTile]` 历史窗口；普通路径从 `conv_states[slot,0:W-1]` 搬，`num_accepted_tokens` 存在时从 `accepted-1` 偏移搬。
  - `bias`：搬 `[dimTile]` 后广播使用；每个 dimLoop 搬一次。
- VF/register：`tokenCount == 1` 时使用 single-S VF；`tokenCount >= 2` 时使用 multi-S VF，一次最多处理同一 batch 的 4 个 token row 的连续 dim tile。两条路径内部都按寄存器宽度循环：
  - `reg_elems = 256 / sizeof(float) = 64` 个 FP32 lane。
  - `dimLoopNum = dimSizeInLoop / 64` 走 full mask 主循环。
  - `dimRem = dimSizeInLoop % 64` 走 `UpdateMask(dimRem)` 尾循环。
  - VF 不是逐 channel 标量算一个 window；它在同一 window tap `k` 上一次处理 64 个 dim。
  - W 固定为 4，AICore wrapper 只调用 `Conv1dDecodeW4` 或 `Conv1dDecodeW4MultiS`。
  - single-S 每个 VF 子循环维护 1 个 FP32 accumulator。
  - multi-S 每个 VF 子循环维护最多 4 个 FP32 accumulator，复用同一组 weight 和重叠的 state/x timeline。
  - 每个 VF 子循环执行 BF16 load，cast FP32，乘加，cast BF16 store。
- 队列同步：`xQueue`、`stateQueueIn`、`yQueue` 通过 `EnQue/DeQue` 作为 tile 内同步边界；`weightBiasBuf` 和 metadata 用 `TBuf`，GM copy 后 `SetFlag/WaitFlag`。

### 5.1 当前分核路线决策

当前主路径固定采用 BH 风格分核：

```text
task space = batch x dim
token/S axis stays inside the core
```

三种 decode 输入都走同一套 BH kernel：

```text
[B,D]                 -> token_count = 1
[B,S,D]               -> token_count = S
[T,D] + query_start_loc -> token_count = query_start_loc[b+1] - query_start_loc[b]
```

不把 BSH 作为当前主路径。BSH 的 `flattened_token_rows x dim` 分核在长序列、
prefill、mixed 或 `sum(S)` 远大于 `B` 时有更高并行度潜力；但本算子目标是
decode-only 且每 batch token 数为 1..4，`batch x dim` 通常已能提供足够任务数。
BSH 会引入 row 到 batch/s 的反查、batch 边界、跨核 state update 合并和同步开销，
所以只作为后续性能实验候选，不进入当前功能对齐主线。

连续性缩略图：

```text
GM row: | d0 ... d0+dimTile-1 | skipped D-dimTile |
        |---- contiguous ----|<------- stride ---->|

UB:     | row0 dimTile | row1 dimTile | row2 dimTile |
        compact, no row gap

VF:     | 0..63 | 64..127 | ... | tail |
          full    full            UpdateMask(dimRem)
```

## 6. 正确性和验证

必须覆盖：

- `[B,D]`、`[B,S,D]` 和 `[T,D] + query_start_loc`
- `W=4`
- token 数 `1..4`
- `cache_indices` 缺省/存在
- `num_accepted_tokens` 缺省/0/部分接受/全部接受
- `bias` 缺省/存在
- dim main/tail tile
- `pad_slot_id` 命中/不命中

当前限制：

- 当前环境未完成 CANN 编译验证。
- v1 未实现 SiLU。
- host tiling 不能静态检查 `query_start_loc` 运行时差分是否都在 `1..4`。
- 当前代码已补 `[B,D]` 非 MTP decode 输入；仍需 CANN 编译验证。
- 当前代码已补 `num_accepted_tokens` 对 state 历史读取窗口的偏移影响，因此它会影响 `y`；W=4 且存在 `num_accepted_tokens` 时也已补 vLLM Ascend PR 8215 风格 speculative state writeback。仍需 CANN 编译验证。
