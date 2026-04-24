# aclnnMoeReRouting

## 功能描述

### 算子功能
该算子实现MoE（Mixture of Experts）网络中的token重排操作。在AlltoAll操作从其他卡上拿到需要计算的token后，将token按照专家顺序重新排列。

### 计算公式

通过双重求和计算当前token在源位置的偏移量：

$$
SrcOffset = 
\sum_{i=0}^{cur\_rank} \left( \sum_{j=0}^{cur\_expert} {expert\_token\_num\_per\_rank}(i,j) \right)
$$

通过双重求和计算当前token在目标位置的偏移量：

$$
DstOffset = 
\sum_{j=0}^{cur\_expert} \left( \sum_{i=0}^{cur\_rank} {expert\_token\_num\_per\_rank}(i,j) \right)
$$

- SrcOffset指当前需要移动的token源偏移，根据输入expert_token_num_per_rank的值进行计算。
- DstOffset指当前需要移动的token目的偏移。
- cur_rank是expert_token_num_per_rank的纵轴索引，表示该token原本在的卡。
- cur_expert是expert_token_num_per_rank的横轴索引，表示该token由卡上专家cur_expert计算。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_re_routing()` 函数形式提供：

```python
def moe_re_routing(tokens: Tensor, 
                   expert_token_num_per_rank: Tensor, 
                   per_token_scales: Optional[Tensor] = None,
                   expert_token_num_type: int = 1,
                   idx_type: int = 0):
    """
    实现 MoE Token ReRouting 操作，将token按照专家顺序重新排列。

    参数:
        tokens (Tensor): 输入张量，表示待重新排布的token，形状为(A, H)
        expert_token_num_per_rank (Tensor): 输入张量，表示每张卡上各个专家处理的token数，形状为(N, E)
        per_token_scales (Optional[Tensor]): 可选输入张量，表示每个token对应的scale，形状为(A,)
        expert_token_num_type (int): 输出expert_token_num的模式，0为cumsum模式，1为count模式，默认值为1
        idx_type (int): 输出permute_token_idx的索引类型，0为gather索引，1为scatter索引，默认值为0

    返回:
        Tuple[Tensor, Optional[Tensor], Tensor, Tensor]: 包含四个张量的元组
            - permute_tokens: 重新排布后的token，形状与tokens相同
            - permute_per_token_scales: 重新排布后的per_token_scales，当per_token_scales不为None时返回
            - permute_token_idx: 每个token在原排布方式的索引，形状为(A,)
            - expert_token_num: 每个专家处理的token数，形状为(E,)
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 场景1：不带per_token_scales的情况
# 创建输入张量
A = 128  # token个数
H = 512  # token长度
N = 8    # 卡数
E = 4    # 专家数

tokens = torch.randn(A, H, dtype=torch.float16)
expert_token_num_per_rank = torch.randint(0, 20, (N, E), dtype=torch.int32)

# 执行token重排操作
permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num = kernel_gen_ops.moe_re_routing(
    tokens, expert_token_num_per_rank
)

print(f"permute_tokens shape: {permute_tokens.shape}")
print(f"permute_per_token_scales: {permute_per_token_scales}")  # 应为None
print(f"permute_token_idx shape: {permute_token_idx.shape}")
print(f"expert_token_num shape: {expert_token_num.shape}")

# 场景2：带per_token_scales的情况
tokens = torch.randn(A, H, dtype=torch.float16)
expert_token_num_per_rank = torch.randint(0, 20, (N, E), dtype=torch.int32)
per_token_scales = torch.randn(A, dtype=torch.float32)

# 执行token重排操作
permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num = kernel_gen_ops.moe_re_routing(
    tokens, expert_token_num_per_rank, per_token_scales
)

print(f"permute_tokens shape: {permute_tokens.shape}")
print(f"permute_per_token_scales shape: {permute_per_token_scales.shape}")
print(f"permute_token_idx shape: {permute_token_idx.shape}")
print(f"expert_token_num shape: {expert_token_num.shape}")

# 场景3：使用idx_type=1（scatter索引）
permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num = kernel_gen_ops.moe_re_routing(
    tokens, expert_token_num_per_rank, per_token_scales, idx_type=1
)

print(f"使用scatter索引模式")
print(f"permute_token_idx shape: {permute_token_idx.shape}")
```

## 约束与限制

### 数据类型支持
- tokens/permute_tokens：
  - 通用：FLOAT16、BF16、INT8
  - <term>Ascend 950PR/Ascend 950DT</term>：FLOAT16、BF16、INT8、FLOAT8_E5M2、FLOAT8_E4M3FN
- expert_token_num_per_rank：INT32、INT64
- per_token_scales/permute_per_token_scales：
  - 通用：FLOAT
  - <term>Ascend 950PR/Ascend 950DT</term>：FLOAT、FLOAT8_E8M0
- permute_token_idx：INT32
- expert_token_num：INT32、INT64

### Tensor形状约束
- A：表示token个数，取值要求Sum(expert_token_num_per_rank)=A
- H：表示token长度，取值要求 0 < H < 16384
- N：表示卡数，取值无限制
- E：表示卡上的专家数，取值无限制

### 输入值域限制
- expert_token_num_type：输出expert_token_num的模式。0为cumsum模式，1为count模式，默认值为1。当前只支持为1
- idx_type：输出permute_token_idx的索引类型。0为gather索引，1为scatter索引，默认值为0。<term>Ascend 950PR/Ascend 950DT</term>支持0或1，其余产品仅支持0

### 其他约束
- 输出张量均为连续张量
- 确定性计算：aclnnMoeReRouting默认确定性实现
