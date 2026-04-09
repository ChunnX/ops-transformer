# aclnnMoeInitRouting

## 功能描述

### 算子功能
MoE的routing计算，根据aclnnMoeGatingTopKSoftmax的计算结果做routing处理。

### 计算公式

将输入shape为[NUM_ROWS, K]的expertIdx展平为一行做排序，其中NUM_ROWS为输入token个数，K为token选择的专家个数：

$$
expandedExpertIdxOut,sortedRowIdx=keyValueSort(expertIdx,rowIdx)
$$

$$
expandedRowIdxOut[sortedRowIdx[i]]=i
$$

$$
expandedXOut[i]=x[sortedRowIdx[i]\%NUM_ROWS]
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_init_routing()` 函数形式提供：

```python
def moe_init_routing(x: Tensor, 
                    row_idx: Tensor, 
                    expert_idx: Tensor, 
                    active_num: int = 0):
    """
    实现 MoE 中 Routing 的计算，生成路由矩阵供后续计算使用。

    参数:
        x (Tensor): MOE的输入，即token特征输入，形状为[NUM_ROWS, H]
        row_idx (Tensor): 指示每个位置对应的原始行位置，形状为[NUM_ROWS, K]
        expert_idx (Tensor): 每一行特征对应的K个处理专家，形状为[NUM_ROWS, K]
        active_num (int): 表示expandedXOut的有效行数，值范围大于等于0

    返回:
        Tuple[Tensor, Tensor, Tensor]: 包含三个张量的元组
            - expanded_x_out: 根据expert_idx进行扩展过的特征
            - expanded_row_idx_out: expanded_x_out和x的映射关系
            - expanded_expert_idx_out: 输出expert_idx排序后的结果
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 场景1：非Active模式
# 创建输入张量
num_rows = 32
k = 4
h = 512

x = torch.randn(num_rows, h, dtype=torch.float16)
row_idx = torch.arange(num_rows, dtype=torch.int32).unsqueeze(1).expand(-1, k)
expert_idx = torch.randint(0, 16, (num_rows, k), dtype=torch.int32)

# 执行计算
expanded_x_out, expanded_row_idx_out, expanded_expert_idx_out = kernel_gen_ops.moe_init_routing(
    x, row_idx, expert_idx, 0
)

print(f"expanded_x_out shape: {expanded_x_out.shape}")
print(f"expanded_row_idx_out shape: {expanded_row_idx_out.shape}")
print(f"expanded_expert_idx_out shape: {expanded_expert_idx_out.shape}")

# 场景2：Active模式
num_rows = 64
k = 8
h = 1024

x = torch.randn(num_rows, h, dtype=torch.float32)
row_idx = torch.arange(num_rows, dtype=torch.int32).unsqueeze(1).expand(-1, k)
expert_idx = torch.randint(0, 32, (num_rows, k), dtype=torch.int32)

# 执行计算
expanded_x_out, expanded_row_idx_out, expanded_expert_idx_out = kernel_gen_ops.moe_init_routing(
    x, row_idx, expert_idx, 2
)

print(f"expanded_x_out shape: {expanded_x_out.shape}")
print(f"expanded_row_idx_out shape: {expanded_row_idx_out.shape}")
print(f"expanded_expert_idx_out shape: {expanded_expert_idx_out.shape}")
```

## 约束与限制

- 输入张量x数据类型支持：FLOAT16、BFLOAT16、FLOAT32
- 输入张量row_idx和expert_idx数据类型支持：INT32
- 输出张量expanded_x_out数据类型与x一致
- 输出张量expanded_row_idx_out和expanded_expert_idx_out数据类型支持：INT32
- active_num的值范围大于等于0
- x的shape必须为2D，形状为(NUM_ROWS, H)
- row_idx的shape必须为2D，且与expert_idx的shape相等
- expanded_row_idx_out和expanded_expert_idx_out的shape必须相等，且等于(num_rows * k, )
- expanded_x_out的shape必须等于(min(num_rows, active_num) * k, H)
- 确定性计算：aclnnMoeInitRouting默认确定性实现
- expertIdx内的元素的值需要大于-2**24，不超过2**24，否则可能会存在精度问题
