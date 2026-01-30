# Angle

## 功能描述

### 算子功能
计算输入张量逐元素的角度（单位：弧度）。

### 计算公式
假设输入张量为 $x$，输出张量为 $y$，则：
$$
y=angle(x)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.angle()` 函数形式提供：

```python
def angle(x):
    """
    实现自定义 Angle 操作。
    
    参数:
        x (Tensor): Device侧的aclTensor，数据类型支持FLOAT16, FLOAT, COMPLEX64, BOOL, UINT8, INT8, INT16, INT32, INT64，数据格式支持ND。
        
    返回:
        Tensor:Device侧的aclTensor，数据类型支持FLOAT16, FLOAT，数据格式支持ND。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)

# 使用 angle 执行计算
result = kernel_gen_ops.angle_v2(x)
```
## 约束与限制

- 张量数据格式支持ND。


