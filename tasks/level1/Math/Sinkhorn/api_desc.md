# aclnnSqrt

## 功能描述
计算Sinkhorn距离，可以用于MoE模型中的专家路由。
### 算子功能
`aclnnSqrt` 实现了对输入数据逐元素开平方，返回开平方结果的功能。

### 计算公式

$$
p=Sinkhorn(cost, tol)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.sqrt()` 函数形式提供：

```python
def sqrt(x):
    """
    实现自定义开平方操作。
    
    参数:
        cost（aclTensor*，计算输入）：表示成本张量，公式中的`cost`，Device侧的aclTensor。数据类型支持BFLOAT16、FLOAT16、FLOAT。[数据格式](common/数据格式.md)支持ND，输入为二维矩阵且列数不超过4096。支持[非连续的Tensor](common/非连续的Tensor.md)。
        tol (aclScalar*, 入参) ：表示计算Sinkhorn的误差，数据类型支持FLOAT。如果传入空指针，则tol取0.0001。
          
    返回:
        p（aclTensor*，计算输出）：表示最优传输张量，公式中的`p`，Device侧的aclTensor。数据类型支持BFLOAT16、FLOAT16、FLOAT。[数据格式](common/数据格式.md)支持ND。shape维度为2。不支持[非连续的Tensor](common/非连续的Tensor.md)。数据类型和shape与入参`cost`的数据类型和shape一致。
    
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入成本张量
cost = torch.rand(4, 8, device=device, dtype=dtype)
tolf = 0.0001

# 使用sqrt_custom执行计算
result = kernel_gen_ops.sqrt(cost, tol)
```


## 约束与限制

- 无