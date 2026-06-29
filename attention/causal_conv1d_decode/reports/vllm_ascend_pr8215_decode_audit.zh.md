# vLLM Ascend PR 8215 CausalConv1d Decode 对齐审计

## 1. 参考来源

本审计基于本地 checkout：

```text
external/vllm-ascend-pr8215
branch: pr-8215
```

重点文件：

- `csrc/causal_conv1d_v310/op_host/causal_conv1d_v310_def.cpp`
- `csrc/causal_conv1d_v310/op_host/causal_conv1d_v310_tiling.cpp`
- `csrc/causal_conv1d_v310/op_kernel/causal_conv1d_v310.h`
- `vllm_ascend/_310p/ops/fla/gdn_310.py`

## 2. 原实现 decode 输入模式

vLLM Ascend PR 8215 的 `runMode=1` decode/update 路径包含三种输入解释：

| inputMode | 输入 | queryStartLoc | 语义 |
|---:|---|---|---|
| `2` | `[batch, dim]` | 缺省或空 | 非 MTP 单 token decode。 |
| `1` | `[batch, seqlen, dim]` | 可空 | 定长 MTP/speculative decode。 |
| `0` | `[cuSeqlen, dim]` | `[batch+1]` | 变长 flatten decode。 |

host tiling 逻辑先把 `runMode=1` 的 2D `x` 当作 `[batch, dim]`。如果存在有效
`queryStartLoc` 且 `queryStartLoc.size - 1 != x.shape[0]`，才改判为
`[cuSeqlen, dim]` 变长 decode。

## 3. 原实现分核

原实现不是把 `B*S` token 行作为第一维直接分核，而是：

```text
gridSize = batch * blocksPerSeq
task     = one batch segment + one dim tile
```

每个 task 内部根据 `inputMode` 得到该 batch 的 token 范围：

```text
inputMode=0: start=queryStartLoc[b], len=queryStartLoc[b+1]-queryStartLoc[b]
inputMode=1: start=b*seqLen,        len=seqLen
inputMode=2: start=b,               len=1
```

所以原实现的 varlen decode 本质是 `batch x dim` 分核，batch 内 token 在核内循环。

## 4. “batch group 一次搬多 batch flatten rows”的风险

vLLM 原实现一次 task 只处理一个 batch 的 token range。每次读取 `x` 时使用：

```text
x[start + t, c0 : c0 + dimTile]
```

这意味着 batch 边界天然清楚：当前 task 永远只读当前 batch 的 token。

我们当前实现为了提高搬运效率，会在一个 UB tile 里合并多个 batch：

```text
rowStart = queryStartLoc[firstBatch]
rowEnd   = queryStartLoc[firstBatch + batchNumInLoop]
blockCount = rowEnd - rowStart
```

然后一次把 `[rowStart, rowEnd)` 之间所有 flatten rows 搬到 UB。这个方案要求：

- `queryStartLoc` 单调；
- batch group 内的 flatten rows 连续；
- 每个 batch 的 token 数合法；
- 后续 compute 能用 `curRowStart - tileRowStart` 正确还原每个 batch 在 UB 中的起点。

这些条件满足时，该方案是可行的；但它比 vLLM 原实现更依赖输入合法性，也更容易在
`queryStartLoc` 非法、空 batch、batch group 边界、或 token 上限变化时出错。后续如果要
严格对齐 PR 8215，可以考虑改成原实现那种“每个 task 只搬一个 batch segment”的流式方式。

## 5. num_accepted_tokens 语义对齐状态

原实现中 speculative decode 条件为：

```text
runMode == 1 && hasNumAcceptedTokens && width == 4
```

在该路径下：

```text
stateTokenOffset = clamp(numAcceptedTokens[b] - 1, 0, stateLen - (width - 1))
```

`InitRing` 会从 `convStates[slot, stateTokenOffset : stateTokenOffset + width - 1, :]`
读入卷积历史窗口。因此 `numAcceptedTokens` 会影响 `y` 的计算历史。

早期实现里 `num_accepted_tokens` 主要用于 state update：

```text
new_state = last state_len rows of concat(old_state, x[0:accepted])
```

这和 PR 8215 不完全一致。当前实现已修正为：

- `num_accepted_tokens` 存在时，用 `accepted - 1` 选择 GM 中的 `W-1` 历史窗口，因此会影响 `y`。
- W=4 speculative path 写回 `state[0:2] = selected_history[1:3]`，`state[2:2+token_count] = x chunk`。
- 无 `num_accepted_tokens` 时，普通 decode 使用 `conv_states[slot,0:W-1]` 作为 compact history。

## 6. 当前实现差距

| 项目 | vLLM Ascend PR 8215 | 当前实现 | 结论 |
|---|---|---|---|
| 非 MTP decode | 支持 `[batch, dim]` | 已补 2D `x` 无 `query_start_loc` 路径 | 待 CANN 验证 |
| 定长 MTP | 支持 `[batch,seqlen,dim]` | 支持 `[B,S,D]` | 基本对齐 |
| 变长 decode | 支持 `[cuSeqlen,dim]+queryStartLoc` | 支持 `[T,D]+query_start_loc` | 形态对齐，搬运策略不同 |
| dtype | FP16 | BF16 | 本算子自定义 |
| metadata dtype | INT64 | INT32 | 本算子自定义 |
| width | 2..4；但 `numAcceptedTokens` speculative writeback 仅 width=4 | 当前仅支持 W=4 | 有意收窄范围；W4 decode 语义对齐 |
| SiLU | 支持 activationMode 0/1 | 只支持 0 | 缺功能 |
| initialStateMode | 支持 | 不支持 | 用户已确认不加入 |
| runMode | 支持 0/1 | 不暴露 | decode-only 设计 |
| numAcceptedTokens 影响 y | 是 | 已补 history window offset | 待 CANN 验证 |
| W=4 speculative state layout | `state[0:2] + x chunk` | 已补 `state[0:2] + x chunk` | 待 CANN 验证 |

## 7. 建议整改顺序

1. 已修正 spec：明确三种 decode 输入模式。
2. 已修正 reference：加入 `[B,D]` 和 vLLM 风格 `numAcceptedTokens` 影响历史窗口的语义。
3. 已修正 host tiling：2D `x` 无 `query_start_loc` 时支持 `[B,D]`，并把 `seqLen=1`。
4. 已修正 kernel：`num_accepted_tokens` 存在时用 `accepted-1` 偏移 state 历史读取窗口。
5. 已补 PR 8215 width=4 speculative writeback 路径：保留两行由 `accepted-1` 决定的历史，并从 state row 2 开始写本次 token。
6. 当前主路径继续采用 BH 分核，不切换到 BSH。BSH 只作为未来性能实验候选，用于评估长 token 或小 `B x dimBlocks` 形状。
7. 当前保留 batch-group 搬运；如果后续为了更严格贴近 vLLM per-batch stream，需要单独做性能/正确性对照。
