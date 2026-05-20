# DispatchNormal

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

当前 `dispatch_normal` 仅面向 A3 场景，其他产品当前版本不支持。

## 功能说明

`DispatchNormal` 用于 MoE 场景下 EP 域 token 分发（dispatch）流程，采用 context 通信域描述方式，不再使用 `group_ep/group_tp` 等字符串域名参数，也不包含 TP 路径。

在当前实现中：

- 基于 `context` 读取 EP 域通信上下文（通过 `update_context(...)` 预先写入）；
- 支持 `quant_mode=0/2`（非量化 / 动态量化）；
- 输出 `recv_x`、`x_scales`、`assist_info_for_combine`；
- 支持两个“可选输出 buffer”入参：`dispatch_wait_recv_cost_stats` 与 `ext_handle_data`。

与 `moe_distribute_dispatch_v3` 的差异重点：

- 本算子是 normal 路径，移除了 TP 相关属性与处理；
- 偏移类输入改为 `Tensor[]` 形式传入，内部在 kernel `Init` 阶段再拆分到原始处理字段；
- `ccl_buffer_size` 作为显式属性输入，用于窗口大小校验与通信资源约束。

## 参数说明

### 输入

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `context` | `Tensor(int32)` | 通信域上下文信息，调用前由 `update_context(...)` 填充。 |
| `x` | `Tensor(float16/bfloat16)` | 本卡待 dispatch 的 token 特征，shape 为 `(BS, H)`。 |
| `topk_idx` | `Tensor(int32)` | 每个 token 的 topK 专家索引，shape 为 `(BS, K)`。 |
| `send_offset` | `Tensor(int32)` | 发送 offset 信息，1D。 |
| `send_token_idx` | `Tensor(int32)` | 发送 token 索引，1D。 |
| `recv_offset` | `Tensor[]` | 固定 2 个元素：`[recv_offset, recv_count]`。 |
| `expert_output_offset` | `Tensor[]` | 固定 3 个元素：`[expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset]`。 |
| `dispatch_wait_recv_cost_stats` | `Tensor?(int32)` | 可选输出 buffer；用于记录等待接收耗时信息。 |
| `ext_handle_data` | `Tensor?(int32)` | 可选输出 buffer；当前版本约束为 1D 且长度为 0 的预留张量。 |

### 属性

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `ep_world_size` | `int` | EP 域大小。 |
| `ep_rank_id` | `int` | 当前卡在 EP 域内 rank，范围 `[0, ep_world_size)`。 |
| `moe_expert_num` | `int` | MoE 专家总数，要求可被 `ep_world_size` 整除。 |
| `ccl_buffer_size` | `int` | 通信 buffer 大小（MB），必须大于 0。 |
| `num_recv_tokens` | `int` | 输出接收 token 数，必须大于 0。 |
| `quant_mode` | `int` | 量化模式：`0`（非量化）或 `2`（动态量化）。 |
| `real_max_bs` | `int` | 单卡最大 BS（按业务传入）。 |
| `global_bs` | `int` | 全局 BS，必须大于 0。 |
| `round` | `int` | 通信 round 数，默认 `4`。 |
| `per_round_tokens` | `int` | 每轮 token 上限，默认 `1024`。 |

### 输出

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `recv_x` | `Tensor` | dispatch 后特征，shape 为 `(num_recv_tokens, H)`，`quant_mode=0` 时 dtype 与 `x` 一致，`quant_mode=2` 时为 `int8`。 |
| `x_scales` | `Tensor(float32)` | 量化相关 scale 输出，shape 为 `(num_recv_tokens,)`。 |
| `assist_info_for_combine` | `Tensor(int32)` | combine 辅助信息，shape 为 `(num_recv_tokens * 3,)`。 |
| `ext_handle_data` | `Tensor(int32)` | 预留输出，当前实现中为长度 0 的 1D 张量。 |

## 约束说明

- `dispatch_normal` 应与对应 combine 路径配套使用，`assist_info_for_combine` 仅作为配套算子间中间信息，不建议被业务逻辑直接解析依赖。
- `context`、`ep_world_size`、`moe_expert_num`、`ccl_buffer_size`、`global_bs` 等关键通信参数在同一 EP 域内各 rank 必须一致。
- 形状与数量约束：
  - `recv_offset` 必须恰好包含 2 个 1D int32 Tensor；
  - `expert_output_offset` 必须恰好包含 3 个 1D int32 Tensor；
  - `recv_offset[0].numel + recv_offset[1].numel = 2 * round * moe_expert_num`；
  - `expert_output_offset[0].numel + expert_output_offset[1].numel + expert_output_offset[2].numel = moe_expert_num / ep_world_size + moe_expert_num + round * moe_expert_num`。
- 数值约束：
  - `ep_rank_id` 取值范围 `[0, ep_world_size)`；
  - `quant_mode` 仅支持 `0` 或 `2`；
  - `num_recv_tokens > 0`，`global_bs > 0`，`ccl_buffer_size > 0`；
  - `ext_handle_data` 若传入，必须为 `int32`、1D、长度为 0。
- A3 约束（当前版本）：
  - `H` 取值范围要求与 A3 dispatch 路径一致，当前实现按 tiling 约束校验在 `[1024, 7168]`；
  - `ep_world_size` 当前版本支持范围 `[2, 384]`；
  - `moe_expert_num` 当前版本支持范围 `(0, 512]`。

## 调试说明

建议按“Python 参数校验 -> tiling 校验 -> kernel 执行 -> 通信域检查”顺序定位问题。

1. Python 侧快速自检  
   先检查 `recv_offset` / `expert_output_offset` 的元素个数、dtype、dim 和总 numel 是否满足约束，再确认 `context` 已通过 `update_context(...)` 初始化。

2. tiling 失败定位  
   若报 shape/dtype/format/attr 失败，优先排查：
   - `ep_rank_id`、`ep_world_size`、`moe_expert_num` 是否匹配；
   - `ccl_buffer_size`、`global_bs`、`num_recv_tokens` 是否为正；
   - 动态输入 `Tensor[]` 中每个元素的 shape 是否与 round/expert 配置一致。

3. 可选输出 buffer 排查  
   - `dispatch_wait_recv_cost_stats` 传入时需为 `int32` 1D，长度为 `ep_world_size`；
   - `ext_handle_data` 传入时需为 `int32` 1D 空张量（shape `[0]`）。

4. 通信域问题排查  
   在多卡场景下，确认所有 rank 的 `context`、EP 域参数、`ccl_buffer_size` 一致；若不一致，通常会表现为通信阻塞或结果异常。

5. 最小化复现  
   可使用 `mc2/dispatch_normal/tests/test_dispatch_normal.py` 做最小 smoke 测试，先验证单一配置，再逐步增加业务复杂度。

## 调用注意事项

- `update_context(...)` 的第 3 个参数需与 `deep_ep.py` 一致，传入可写对象（如 `IntWrapper`），由 C++ 侧回填 `ccl_buffer_size.value`。
- 不要把普通 `int` 直接传给 `update_context(...)` 第 3 个参数，否则无法承接回填值。

```python
class IntWrapper:
    def __init__(self, value=0):
        self.value = value


ccl_buffer_size = IntWrapper()
context = torch.zeros((context_tensor_size,), dtype=torch.int32, device="npu")
ok = update_context(group_name, ep_world_size, ccl_buffer_size, context)
assert ok and ccl_buffer_size.value > 0

recv_x, x_scales, assist_info_for_combine, ext_handle_data = torch.ops.npu_ops_transformer.npu_dispatch_normal(
    context=context,
    x=x,
    topk_idx=topk_idx,
    send_offset=send_offset,
    send_token_idx=send_token_idx,
    recv_offset=recv_offset,
    expert_output_offset=expert_output_offset,
    ep_world_size=ep_world_size,
    ep_rank_id=ep_rank_id,
    moe_expert_num=moe_expert_num,
    ccl_buffer_size=ccl_buffer_size.value,
    num_recv_tokens=num_recv_tokens,
)
```

## 目录结构

- `op_host/`: OpDef 与 tiling 校验
- `op_kernel/`: kernel 实现
- `op_api/`: ACLNN 对外接口
- `op_graph/`: GE 图模式 proto 注册（plugin）
- `tests/`: 示例测试脚本
