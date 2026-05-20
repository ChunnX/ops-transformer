# NotifyDispatch

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | √ (A5 kernel) |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ (tiling key 123) |

## 功能说明

`NotifyDispatch` 用于 MoE EP 流程中 **AllToAll 元数据交换**（notify 阶段），在 `dispatch_layout` 之后、`dispatch_normal` 之前执行。

- 采用 **Mc2 context** 通信域（`update_context` 预填充），不再使用 `comm_group` 字符串；
- 输出 offset 以 **Tensor[]** 形式在 Python 侧打包，直接对接 `npu_dispatch_normal` 的 `recv_offset` / `expert_output_offset` 输入。

典型链路：

```
dispatch_layout → notify_dispatch → dispatch_normal → expert → combine
```

## 参数说明

### 输入

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `context` | `Tensor(int32)` | 通信域上下文，1D，由 `update_context(...)` 填充。 |
| `sendData` | `Tensor(int32)` | 来自 `dispatch_layout` 的 `notify_send_data`（`ext_handle_data[0]`），长度 `send_count`。 |
| `tokenPerExpertData` | `Tensor(int32)` | 每专家 token 计数，通常来自 `dispatch_layout` 的 `num_tokens_per_expert`，长度 `round * moe_expert_num`。 |

### 属性

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `send_count` | `int` | `SEND_PER_GROUP(3) * moe_expert_num * round`。 |
| `num_tokens` | `int` | 本卡 token 数（batch size）。 |
| `ep_world_size` | `int` | EP 域大小。 |
| `ep_rank_id` | `int` | 当前 EP rank，`[0, ep_world_size)`。 |
| `local_rank_size` | `int` | 单机 rank 数。 |
| `local_rank_id` | `int` | 单机内 rank，`[0, local_rank_size)`。 |
| `ccl_buffer_size` | `int` | 通信窗口大小（字节），与 `dispatch_normal` 一致。 |
| `round` | `int` | 通信 round 数。 |
| `perRoundTokens` | `int` | 每 round token 上限。 |

### Python 返回值（对接 dispatch_normal）

| 返回值 | 说明 |
| :--- | :--- |
| `send_offset` | → `dispatch_normal.send_offset` |
| `recv_offset[0]` | → `dispatch_normal.recv_offset[0]` |
| `recv_offset[1]` | → `dispatch_normal.recv_offset[1]`（recv_count） |
| `expert_output_offset[0..2]` | → `dispatch_normal.expert_output_offset` |
| `total_recv_tokens` | → `num_recv_tokens`（标量 tensor） |
| `max_bs` | → `real_max_bs`（标量 tensor） |
| `recv_data` | AllToAll 接收缓冲（中间数据） |
| `recv_tokens_per_expert` | 每专家接收 token 统计 |

## tiling_key

| SoC | tiling_key |
| :--- | :---: |
| A3 默认 | 23 |
| A2 (910B) | 123 |
| A5 (950) | 223 |

定义见 `op_kernel/notify_dispatch_tiling_key.h`，host 通过 `GetNotifyDispatchTilingKey(isA2, isA5)` 选择。

## 调用示例

```python
from npu_ops_transformer.ops.update_context import update_context

# 1) layout
_, _, num_tokens_per_expert, _, ext_handle_data = npu_dispatch_layout(
    topk_idx, num_ranks=ep_world_size, num_experts=moe_expert_num,
    local_ranksize=local_rank_size, per_round_tokens=per_round_tokens, rank_id=ep_rank_id,
)
send_data, send_token_idx = ext_handle_data[0], ext_handle_data[1]

# 2) notify
send_count = 3 * moe_expert_num * round
local_rank_id = ep_rank_id % local_rank_size
(
    send_offset,
    recv_offset,
    expert_output_offset,
    total_recv_tokens,
    max_bs,
    recv_data,
    _,
) = npu_notify_dispatch(
    context, send_data, num_tokens_per_expert,
    send_count, num_tokens,
    ep_world_size, ep_rank_id, local_rank_size, local_rank_id,
    ccl_buffer_size, round, per_round_tokens,
)

# 3) dispatch
num_recv_tokens = int(total_recv_tokens.item())
real_max_bs = int(max_bs.item())
global_bs = min(per_round_tokens, real_max_bs) * ep_world_size
recv_x, x_scales, assist_info_for_combine, _ = npu_dispatch_normal(
    context, x, topk_idx,
    send_offset, send_token_idx,
    recv_offset, expert_output_offset,
    ep_world_size, ep_rank_id, moe_expert_num, ccl_buffer_size,
    num_recv_tokens, quant_mode=0, real_max_bs=real_max_bs, global_bs=global_bs,
    round=round, per_round_tokens=per_round_tokens,
)
```

也可使用融合封装 `npu_moe_dispatch_prefill`（内部调用 notify + dispatch_normal）。

## 目录结构

- `op_host/`: OpDef、infershape、tiling
- `op_kernel/`: kernel（A3/A5）
- `op_api/`: ACLNN 接口
- `tests/`: smoke 测试
