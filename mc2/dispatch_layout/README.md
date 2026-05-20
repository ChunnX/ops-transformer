# DispatchLayout

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ (tiling key 123) |

## 功能说明

`DispatchLayout` 用于 MoE EP **dispatch 前的布局统计**：根据 `topkIdx` 统计本 rank 发往各 rank / 各 expert 的 token 数、token→rank 路由矩阵，并生成 notify 阶段所需的辅助缓冲。

典型链路：

```
dispatch_layout → notify_dispatch → dispatch_normal → expert → combine
```

定义 `round = ⌈num_tokens / per_round_tokens⌉`（长序列 / Chunked Prefill 分轮）。

## 参数说明

### 输入（Tensor）

| 参数名 | 类型 | Shape | 说明 |
| :--- | :--- | :--- | :--- |
| `topkIdx` | `DT_INT64` | `[num_tokens, num_topk]` | 每个 token 选择的专家索引，`-1` 表示无效。 |

调用方须保证 Attr `num_tokens`、`num_topk` 与 `topkIdx` 的 shape 一致。

### 属性（Attr）

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `num_tokens` | `int` | 本 rank 参与 dispatch 的 token 总数。 |
| `num_ranks` | `int` | 全局 EP 组 rank 数。 |
| `num_experts` | `int` | 专家总数，须能被 `num_ranks` 整除。 |
| `num_topk` | `int` | 每个 token 的 Top-K 专家数。 |
| `local_ranksize` | `int` | 单机 rank 数（拓扑参数，tiling 校验）。 |
| `per_round_tokens` | `int` | 每轮处理的 token 上限（Chunked Prefill）。 |
| `rank_id` | `int` | 当前 EP rank，`[0, num_ranks)`（拓扑参数，tiling 下发）。 |

### 输出（Tensor）

| 参数名 | 类型 | Shape（InferShape） | 说明 |
| :--- | :--- | :--- | :--- |
| `numTokensPerRank` | `DT_INT32` | `[num_ranks]` | 本 rank 发往各目标 rank 的 token 数。 |
| `numTokensPerExpert` | `DT_INT32` | `[round * num_experts]` | 按轮展平的每专家发送 token 数；下标 `r * num_experts + e` 为第 `r` 轮、专家 `e`。 |
| `isTokenInRank` | `DT_INT32` | `[round * per_round_tokens, num_ranks]` | Token→rank 路由，值为 `0/1`；第 0 维按轮对齐，末轮不足 `per_round_tokens` 的行为 padding 行。 |
| `ext_handle_data` | `DT_INT32`（DynamicOutput，固定 2 项） | 见下表 | 打包原 `notifySendData` + `sendTokenIdxSmall`。 |

`ext_handle_data` 子项：

| 下标 | 原名 | Shape | 说明 |
| :---: | :--- | :--- | :--- |
| 0 | `notifySendData` | `[round * num_local_experts]` | `num_local_experts = num_experts / num_ranks`；kernel scratch。 |
| 1 | `sendTokenIdxSmall` | `[num_tokens * num_topk]` | 每 (token, topk) 在 expert 上的序号等小表。 |

## tiling_key

| SoC | tiling_key |
| :--- | :---: |
| A3 / A5 默认 | 23 |
| A2 (910B) | 123 |

定义见 `op_kernel/dispatch_layout_tiling_key.h`，host 通过 `GetDispatchLayoutTilingKey(isA2)` 选择。

## 相关文件

| 路径 | 说明 |
| :--- | :--- |
| `op_host/dispatch_layout_def.cpp` | OpDef |
| `op_host/dispatch_layout_infershape.cpp` | 输出 shape |
| `op_api/aclnn_dispatch_layout.h` | ACLNN 两段式接口 |
| `op_graph/dispatch_layout_proto.h` | GE `REG_OP` |
