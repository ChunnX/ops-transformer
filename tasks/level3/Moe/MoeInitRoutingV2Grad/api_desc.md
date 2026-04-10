# aclnnMoeInitRoutingV2Grad

## 功能描述

### 算子功能
该算子实现aclnnMoeInitRoutingV2的反向传播，完成tokens的加权求和。

### 计算公式

$$
gradX_i=\sum_{t=0}^{topK}gradExpandedX[expandedRowIdx[i * topK + t]]
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_init_routing_v2_grad()` 函数形式提供：

```python
def moe_init_routing_v2_grad(grad_expanded_x: Tensor, 
                            expanded_row_idx: Tensor, 
                            top_k: int, 
                            drop_pad_mode: int, 
                            active_num: int):
    """
    实现 MoE 中 Routing 的反向传播操作，完成 tokens 的加权求和。

    参数:
        grad_expanded_x (Tensor): Routing 过后的目标张量的梯度，形状为 2D 或 3D
        expanded_row_idx (Tensor): token 按照专家序排序索引，形状为 1D
        top_k (int): topK 值，必须大于 0
        drop_pad_mode (int): 表示是否为 Drop/Pad 场景，取值为 0 或 1
        active_num (int): 表示场景是否为 Active 场景，值范围大于等于 0

    返回:
        Tensor: Routing 反向输出，形状为 [B*S, H]
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 场景1：Dropless 模式
# 创建输入张量
tokens_num = 32
hidden_size = 512
topK = 4

# 2D 形状：[B*S*K, H]
grad_expanded_x = torch.randn(tokens_num * topK, hidden_size, dtype=torch.float16)
expanded_row_idx = torch.randint(0, tokens_num * topK, (tokens_num * topK,), dtype=torch.int32)

# 执行反向传播计算
grad_x = kernel_gen_ops.moe_init_routing_v2_grad(
    grad_expanded_x, expanded_row_idx, topK, 0, 0
)

print(f"grad_x shape: {grad_x.shape}")

# 场景2：Drop/Pad 模式
tokens_num = 64
hidden_size = 1024
topK = 8
expert_num = 32
expert_capacity = 16

# 3D 形状：[E, C, H]
grad_expanded_x = torch.randn(expert_num, expert_capacity, hidden_size, dtype=torch.float32)
expanded_row_idx = torch.randint(-1, expert_num * expert_capacity, (tokens_num * topK,), dtype=torch.int32)

# 执行反向传播计算
grad_x = kernel_gen_ops.moe_init_routing_v2_grad(
    grad_expanded_x, expanded_row_idx, topK, 1, 0
)

print(f"grad_x shape: {grad_x.shape}")
```

## 约束与限制

- 输入张量 grad_expanded_x 数据类型支持：FLOAT16、BFLOAT16、FLOAT32
- 输入张量 expanded_row_idx 数据类型支持：INT32
- topK 必须大于 0，且 expanded_row_idx 的 0 轴大小能被 topK 整除
- dropPadMode 的取值为 0 或 1
- activeNum 的值范围大于等于 0，当 dropPadMode 为 0 时生效
- grad_expanded_x 支持 2D 或 3D 形状：
  - 2D 形状：Dropless 场景的 [B*S*K, H] 或 Active 场景下的 [A, H]
  - 3D 形状：Drop/Pad 场景下的 [E, C, H]
- expanded_row_idx 要求是 1D 张量，形状为 [B*S*K]
- expanded_row_idx 元素值在 Drop/Pad 场景下范围为 [-1, E*C)，其他场景范围为 [0, B*S*K)
- 输出张量 grad_x 的数据类型与输入 grad_expanded_x 一致
- 输出张量 grad_x 是 2D 张量，形状为 [B*S, H]
- 确定性计算：aclnnMoeInitRoutingV2Grad 默认确定性实现
