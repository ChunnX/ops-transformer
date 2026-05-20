# CombineNormal

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

`CombineNormal` 用于 MoE 场景下 EP 域 token 合并（combine）流程，通过 `context` 描述 Mc2 通信域（与 `dispatch_normal` / `notify_dispatch` 一致）。

- 首个输入为 `context`（`int32` 一维张量），由 `update_context` 填充；
- 使用 `ccl_buffer_size` 校验通信窗口大小；
- 支持 `recv_x` 数据类型为 `float16/bfloat16`；
- 可选 TP 参数：`tp_world_size`、`tp_rank_id`、`tp_recv_counts`。

典型链路：`dispatch_layout` → `notify_dispatch` → `dispatch_normal` → expert → `combine_normal`。

## 参数说明

### 输入

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `context` | `Tensor(int32)` | Mc2 通信上下文。 |
| `recv_x` | `Tensor(float16/bfloat16)` | dispatch 后的 token 特征，shape 为 `(num_recv_tokens, H)`。 |
| `token_src_info` | `Tensor(int32)` | token 来源信息，shape 为 `(num_recv_tokens * 3,)`。 |
| `ep_recv_counts` | `Tensor(int32)` | EP 域接收计数，shape 为 `(moe_expert_num,)`。 |
| `recv_topk_weights` | `Tensor(float32)` | topK 权重，shape 为 `(BS, K)`。 |
| `token_idx` | `Tensor(int32)` | token 索引，shape 为 `(BS, K)`。 |
| `tp_recv_counts` | `Tensor?(int32)` | 可选，TP 域接收计数。 |

### 属性

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `ep_world_size` | `Int` | EP 域大小。 |
| `ep_rank_id` | `Int` | EP 域 rank。 |
| `moe_expert_num` | `Int` | MoE 专家总数。 |
| `ccl_buffer_size` | `Int` | 通信 buffer 大小（字节），须 > 0。 |
| `tp_world_size` | `Int` | 可选，TP 域大小，默认 0。 |
| `tp_rank_id` | `Int` | 可选，TP rank，默认 0。 |
| `real_max_bs` | `Int` | 可选，全局最大 BS，0 表示使用当前 BS。 |
| `round` | `Int` | 可选，多轮 combine 轮数，默认 4。 |
| `per_round_tokens` | `Int` | 可选，每轮 token 数，默认 1024。 |

### 输出

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `x` | `Tensor(float16/bfloat16)` | 合并后的 token 特征，shape 为 `(BS, H)`。 |
| `combine_send_cost_stats` | `Tensor?(int32)` | 可选，发送耗时统计。 |

## Tiling Key

| Key | 含义 |
| :--- | :--- |
| 13000 | A3 单轮 |
| 13001 | A3 多轮 |
| 15000 | A5 单轮 |
| 15001 | A5 多轮 |

## 测试

```bash
torchrun --nproc_per_node=2 mc2/combine_normal/tests/test_combine_normal.py
```

Python 接口：`torch.ops.npu_ops_transformer.npu_combine_normal`。
