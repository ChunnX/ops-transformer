# aclnnGroupedBiasAddGrad

## 功能描述
分组偏置加法（GroupedBiasAdd）的反向计算。
### 算子功能
`aclnnGroupedBiasAddGrad` 实现了对输入数据根据groupIdx进行累加，返回累加后结果的功能。

### 计算公式

$$
out(G,H) = \begin{cases} \sum_{i=groupIdxOptional(j-1)}^{groupIdxOptional(j)}  gradY(i, H), & 1 \leq j \leq G-1 \\  \sum_{i=0}^{groupIdxOptional(j)}  gradY(i, H), & j = 0 \end{cases}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.grouped_bias_add_grad()` 函数形式提供：

```python
def grouped_bias_add_grad(grad_y, grouped_idx):
    """
    分组偏置加法（GroupedBiasAdd）的反向计算
    
    参数:
        grad_y（Tensor): 必选参数，反向传播梯度，公式中的gradY。有可选输入group_idx时，shape仅支持2维，无可选输入group_idx时，shape仅支持3维。
        group_idx（Tensor): 可选参数，每个分组结束位置，公式中的groupIdxOptional，shape仅支持1维。
          
    返回:
        out（aclTensor*，计算输出）: bias的梯度，公式中的out。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入
grad_y = torch.rand(40, 10, device=device, dtype=dtype)
num_groups = 4
# 1. 从 [1, grad_y.shape[0]) 中随机选择 num_groups - 1 个数
indices = torch.randint(1, grad_y.shape[0], (num_groups - 1,), dtype=torch.int32, device=grad_y.device)

# 2. 排序（确保递增）
indices = torch.sort(indices).values

# 3. 添加结束点
group_idx = torch.cat([indices, torch.tensor([grad_y.shape[0]], dtype=torch.int32, device=grad_y.device)])

# 使用grouped_bias_add_grad_custom执行计算
result = kernel_gen_ops.grouped_bias_add_grad(grad_y, group_idx)
```


## 约束与限制

- group_idx最大支持2048个数。
- 有可选输入group_idx时，需要保证Tensor数据是递增排列，且最后一个数值需要等于gradY第0维的大小。
- 有可选输入group_idx时，需要保证Tensor数值不超过INT32最大值，并且是非负数。