# moe_finalize_routing

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

`moe_finalize_routing`是基于`torch_npu`的`cann_ops_transformer`扩展接口。该接口用于MoE（Mixture of Experts）计算的最后阶段，将各专家FFN的输出结果按路由权重加权合并，还原为原始token序列。

计算公式：

$$
\mathrm{expertid} = \mathrm{expertIdx}[i,k]
$$

$$
\mathrm{out}(i, j) = x1_{i, j} + x2_{i, j} + \sum_{k=0}^{K}\mathrm{scales}_{i,k} \cdot \big(\mathrm{expandedX}_{\mathrm{expandedRowIdx}_{i+k \cdot num\_rows},j} + \mathrm{bias}_{\mathrm{expertid},j}\big)
$$

> 当`scales`为空时，K为1。

## 函数原型

```python
cann_ops_transformer.ops.moe_finalize_routing(
    expanded_x,
    expanded_row_idx,
    x1=None,
    x2=None,
    bias=None,
    scales=None,
    expert_idx=None,
    drop_pad_mode=0,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `expanded_x` | Tensor | 必选 | MoE FFN的输出，即公式中的expandedX。drop less场景shape为`(NUM_ROWS * K, H)`，drop pad场景shape为`(E, C, H)`。 | FLOAT16、BFLOAT16、FLOAT32 | `(NUM_ROWS * K, H)`或`(E, C, H)` |
| `expanded_row_idx` | Tensor | 必选 | token索引映射，即公式中的expandedRowIdx。drop_pad_mode为0或2时，值域为`[0, NUM_ROWS * K - 1]`；为1或3时，值域为`[-1, E * C - 1]`，其中-1表示该token被丢弃。 | INT32 | `(NUM_ROWS * K,)` |
| `x1` | Tensor | 可选 | 第一个共享专家输出，即公式中的x1。shape与输出一致。 | 与expanded_x一致 | `(NUM_ROWS, H)` |
| `x2` | Tensor | 可选 | 第二个共享专家输出，即公式中的x2。shape与输出一致。x1未传入时，x2也不能传入。 | 与expanded_x一致 | `(NUM_ROWS, H)` |
| `bias` | Tensor | 可选 | 偏置量。bias存在时，expert_idx必须同时存在。 | 与expanded_x一致 | `(E, H)` |
| `scales` | Tensor | 可选 | 路由权重系数。scales为空时，K为1。 | FLOAT16、BFLOAT16、FLOAT32 | `(NUM_ROWS, K)` |
| `expert_idx` | Tensor | 可选 | 专家索引，即公式中的expertIdx。bias存在时必须同时传入。值域为`[0, E-1]`。 | INT32 | `(NUM_ROWS, K)` |
| `drop_pad_mode` | int | 可选 | 丢弃/填充模式及expandedRowIdx的排列方式。`0`：drop less，按列排列；`1`：drop pad，按列排列；`2`：drop less，按行排列；`3`：drop pad，按行排列。默认值为0。 | int64 | - |

## 返回值说明

返回一个Tensor，shape为`(NUM_ROWS, H)`，数据类型与`expanded_x`一致。

## 约束说明

- `expanded_x`必须是2D或3D Tensor。
- `expanded_row_idx`必须是1D Tensor。
- `drop_pad_mode`取值范围为`[0, 3]`。
- `x1`未传入时，`x2`也不能传入。
- `bias`存在时，`expert_idx`必须同时存在。
- `scales`为空时，K为1。

## 调用示例

```python
import torch
import torch_npu
from cann_ops_transformer import moe_finalize_routing

NUM_ROWS = 4
K = 2
H = 8
E = 8

expanded_x = torch.randn(NUM_ROWS * K, H, dtype=torch.float32, device="npu")
row_idx = torch.arange(NUM_ROWS * K, dtype=torch.int32, device="npu")
scales = torch.randn(NUM_ROWS, K, dtype=torch.float32, device="npu")
x1 = torch.randn(NUM_ROWS, H, dtype=torch.float32, device="npu")
x2 = torch.randn(NUM_ROWS, H, dtype=torch.float32, device="npu")
bias = torch.randn(E, H, dtype=torch.float32, device="npu")
expert_idx = torch.tensor(
    [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32, device="npu"
)

out = moe_finalize_routing(
    expanded_x,
    row_idx,
    x1=x1,
    x2=x2,
    bias=bias,
    scales=scales,
    expert_idx=expert_idx,
    drop_pad_mode=0,
)
print(out.shape)  # torch.Size([4, 8])
```

## 确定性计算

默认支持确定性计算。
