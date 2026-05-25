# MoeCombineNormal

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

当前 `moe_combine_normal` 仅面向 A3 场景，其他产品当前版本不支持。

## 功能说明

`MoeCombineNormal` 用于 MoE 场景下 EP 域 token 合并（combine）流程，采用通信域名称描述方式，支持 EP/TP 通信域。

在当前实现中：

- 基于 `ep_group_name` 读取 EP 域通信上下文；
- 支持 `recv_x` 数据类型为 `float16/bfloat16`；
- 输出 `x`、可选输出 `combine_send_cost_stats`；
- 支持 TP 域通信（可选）。

与 `moe_distribute_combine_v2` 的差异重点：

- 本算子是 normal 路径，简化了部分复杂功能；
- 输入参数更简洁，移除了一些预留参数；
- 支持 `combine_send_cost_stats` 作为可选输出，用于记录发送耗时信息。

## 参数说明

### 输入

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `recv_x` | `Tensor(float16/bfloat16/int32)` | dispatch 后的 token 特征，shape 为 `(num_recv_tokens, H)`。 |
| `token_src_info` | `Tensor(int32)` | token 来源信息，shape 为 `(num_recv_tokens * 3,)`。 |
| `ep_recv_counts` | `Tensor(int32)` | EP 域接收计数，shape 为 `(moe_expert_num,)`。 |
| `recv_topk_weights` | `Tensor(float32)` | topK 权重，shape 为 `(BS, K)`。 |
| `tp_recv_counts` | `Tensor?(int32)` | 可选输入，TP 域接收计数。 |

### 属性

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `ep_group_name` | `String` | EP 域通信组名称。 |
| `ep_world_size` | `Int` | EP 域大小。 |
| `ep_rank_id` | `Int` | 当前卡在 EP 域内 rank，范围 `[0, ep_world_size)`。 |
| `moe_expert_num` | `Int` | MoE 专家总数，要求可被 `ep_world_size` 整除。 |
| `tp_group_name` | `String` | 可选，TP 域通信组名称。 |
| `tp_world_size` | `Int` | 可选，TP 域大小，默认 `0`。 |
| `tp_rank_id` | `Int` | 可选，TP 域 rank，默认 `0`。 |
| `global_bs` | `Int` | 可选，全局 BS，默认 `0`。 |

### 输出

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `x` | `Tensor(float16/bfloat16)` | combine 后特征，shape 为 `(BS, H)`。 |
| `combine_send_cost_stats` | `Tensor?(int32)` | 可选输出，发送耗时统计，shape 为 `(ep_world_size,)`。 |

## 约束说明

- `moe_combine_normal` 应与对应 dispatch 路径配套使用，`token_src_info` 来自 dispatch 输出的 `assist_info_for_combine`。
- `ep_group_name`、`ep_world_size`、`moe_expert_num` 等关键通信参数在同一 EP 域内各 rank 必须一致。
- 形状约束：
  - `recv_x` shape 为 `(num_recv_tokens, H)`，其中 `H` 取值范围 `[1024, 7168]`；
  - `token_src_info` shape 为 `(num_recv_tokens * 3,)`；
  - `ep_recv_counts` shape 为 `(moe_expert_num,)`；
  - `recv_topk_weights` shape 为 `(BS, K)`，`K` 范围 `(0, min(16, moeExpertNum)]`；
  - `x` shape 为 `(BS, H)`。
- 数值约束：
  - `ep_rank_id` 取值范围 `[0, ep_world_size)`；
  - `ep_world_size` 范围 `[2, 384]`；
  - `moe_expert_num` 范围 `(0, 512]` 且可被 `ep_world_size` 整除；
  - `BS` 范围 `[1, 8000]`。
- A3 约束（当前版本）：
  - 仅支持 A3 平台；
  - `recv_x` 数据类型不支持 `int32`（当前实现限制）。

## 调试说明

建议按"Python 参数校验 -> tiling 校验 -> kernel 执行 -> 通信域检查"顺序定位问题。

1. Python 侧快速自检  
   先检查 `recv_x`、`token_src_info`、`ep_recv_counts`、`recv_topk_weights` 的 shape、dtype 是否满足约束，再确认通信组已正确初始化。

2. tiling 失败定位  
   若报 shape/dtype/format/attr 失败，优先排查：
   - `ep_rank_id`、`ep_world_size`、`moe_expert_num` 是否匹配；
   - `global_bs` 是否为 0 或满足 `global_bs >= ep_world_size * bs` 且可被整除；
   - `tp_world_size > 1` 时，`moe_expert_num / ep_world_size` 必须为 1。

3. 可选输出 buffer 排查  
   - `combine_send_cost_stats` 传入时需为 `int32` 1D，长度为 `ep_world_size`。

4. 通信域问题排查  
   在多卡场景下，确认所有 rank 的 `ep_group_name`、EP 域参数一致；若不一致，通常会表现为通信阻塞或结果异常。

5. 最小化复现  
   可使用 `experimental/mc2/combine_normal/tests/test_moe_combine_normal.py` 做最小 smoke 测试，先验证单一配置，再逐步增加业务复杂度。

## 目录结构

- `op_host/`: OpDef 与 tiling 校验
- `op_kernel/`: kernel 实现
- `op_api/`: ACLNN 对外接口
- `op_graph/`: GE 图模式 proto 注册（plugin）
- `tests/`: 示例测试脚本