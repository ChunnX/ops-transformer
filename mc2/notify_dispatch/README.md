# NotifyDispatch

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | √ (A5 kernel) |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ (tiling key 123) |

## 功能说明

`NotifyDispatch` 用于 MoE EP 流程中 **AllToAll 元数据交换**（notify 阶段），在 `dispatch_layout` 之后、`dispatch_normal` 之前执行。

- 采用 **Mc2 context** + Attr **`ccl_buffer_size`** 表示通信域（不用 `comm_group` 字符串）；
- **`recv_offset`** / **`expert_output_offset`** 以 **DynamicOutput（TensorList）** 输出，与 `dispatch_normal` 的 DynamicInput 对齐；
- **`ext_handle_data`** 为 **OPTIONAL** 预留输出（`int32` 1D、长度 0），与 `dispatch_normal` 一致，kernel 不写。

典型链路：

```
dispatch_layout → notify_dispatch → dispatch_normal → expert → combine
```

## 参数说明

### 输入（Tensor）

| 参数名 | 类型 | Shape | 说明 |
| :--- | :--- | :--- | :--- |
| `context` | `DT_INT32` | 1D | `update_context(...)` 填充的 Mc2 通信上下文。 |
| `sendData` | `DT_INT32` | 1D | 来自 layout 的 `ext_handle_data[0]`（notifySendData）。 |
| `tokenPerExpertData` | `DT_INT32` | `[round * num_experts]` | 来自 layout 的 `numTokensPerExpert`。 |

### 属性（Attr）

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `sendCount` | `int` | `3 * num_experts * round`（`SEND_PER_GROUP=3`）。 |
| `num_tokens` | `int` | 本 rank 发送 token 总数。 |
| `ep_world_size` | `int` | EP 域 rank 数。 |
| `ep_rank_id` | `int` | 当前 EP rank。 |
| `local_rank_size` | `int` | 节点内 rank 数。 |
| `local_rank_id` | `int` | 节点内 rank 编号。 |
| `ccl_buffer_size` | `int` | 通信窗口（字节）。 |
| `round` | `int` | 长序列搬运轮数。 |
| `perRoundTokens` | `int` | 每轮 token 上限。 |

### 输出（Tensor）

| 参数名 | 类型 | Shape（InferShape） | 说明 |
| :--- | :--- | :--- | :--- |
| `sendDataOffset` | `DT_INT32` | `[round * num_experts]` | 各 expert 发送 offset。 |
| `recvData` | `DT_INT32` | `[sendCount]` | AllToAll 交换缓冲。 |
| `recv_offset` | `DT_INT32`（DynamicOutput，2 项） | 见下表 | 对接 `dispatch_normal.recv_offset`。 |
| `expert_output_offset` | `DT_INT32`（DynamicOutput，3 项） | 见下表 | 对接 `dispatch_normal.expert_output_offset`。 |
| `totalRecvTokens` | `DT_INT32` | `[1]` | 本 rank 接收 token 总数。 |
| `maxBs` | `DT_INT32` | `[1]` | EP 域内最大 batch size。 |
| `recvTokensPerExpert` | `DT_INT32` | `[round * num_local_experts]` | 本 rank 各 local expert 接收 token 数。 |
| `ext_handle_data` | `DT_INT32`（OPTIONAL） | `[0]` | 预留，可不连。 |

`recv_offset` 子项：

| 下标 | 原名 | Shape |
| :---: | :--- | :--- |
| 0 | `recvOffset` | `[round * num_experts]` |
| 1 | `recvCount` | `[round * num_experts]` |

`expert_output_offset` 子项：

| 下标 | 原名 | Shape |
| :---: | :--- | :--- |
| 0 | `expertGlobalOffset` | `[num_local_experts]` |
| 1 | `srcrankInExpertOffset` | `[num_local_experts * ep_world_size]` |
| 2 | `rInSrcrankOffset` | `[num_local_experts * ep_world_size * round]` |

其中 `num_experts = sendCount / 3 / round`，`num_local_experts = num_experts / ep_world_size`。

## tiling_key

与 `dispatch_layout` / `dispatch_normal` 一致，使用 **ASCENDC 模板 tiling**（`ASCENDC_TPL_*` + `GET_TPL_TILING_KEY`）。

| 模板 `ARCH_TAG` | SoC | 设备实现 |
| :--- | :--- | :--- |
| `TILINGKEY_TPL_A3` (=1) | Ascend910_93 等 | `NotifyDispatch` |
| `TILINGKEY_TPL_A2` (=0) | Ascend910B | `NotifyDispatch` |
| `TILINGKEY_TPL_A5` (=2) | Ascend950 | `NotifyDispatchA5` |

Host：`notify_dispatch_tiling.cpp` 按 `Short_SoC_version` 选 `archTag` 后 `GET_TPL_TILING_KEY(archTag)`。  
Device：`notify_dispatch.cpp` 为 `template <uint8_t ArchTag>` + `if constexpr`（定义见 `notify_dispatch_tiling_key.h`）。

## 相关文件

| 路径 | 说明 |
| :--- | :--- |
| `op_host/notify_dispatch_def.cpp` | OpDef |
| `op_host/notify_dispatch_infershape.cpp` | InferShape |
| `op_host/op_tiling/notify_dispatch_tiling.cpp` | Tiling |
| `op_api/aclnn_notify_dispatch.h` | ACLNN |
| `op_graph/notify_dispatch_proto.h` | GE REG_OP |

Torch 封装与 `moe_dispatch_prefill` 将在后续 PR 中对接新接口。
