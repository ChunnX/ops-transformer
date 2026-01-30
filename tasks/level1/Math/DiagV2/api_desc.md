# DiagV2

## 功能描述

### 算子功能
根据输入的二维张量，提取由diagonal指定的对角线元素。
  - 如果diagonal = 0，选择主对角线；
  - 如果diagonal > 0，选择主对角线上方的第diagonal条对角线；
  - 如果diagonal < 0，选择主对角线下方的第-diagonal条对角线。

### 计算公式

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.diag()` 函数形式提供：

```python
def diag(x):
    """
    实现自定义 Diag 操作。
    
    参数:
        x (Tensor): Device侧的aclTensor，数据类型支持INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BFLOAT16、DOUBLE、BOOL，数据格式支持ND。
        diagonal(可选参数): 表示选择对角线的位置，默认为0。数据类型支持INT。
    返回:
        Tensor:Device侧的aclTensor，数据类型支持INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BFLOAT16、DOUBLE、BOOL，数据格式支持ND。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.tensor([[1.0, 2.0, 3.0], 
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],], dtype=torch.float)

# 使用 angle 执行计算
result = kernel_gen_ops.diag_v2(x)
```
## 约束与限制

- 张量数据格式支持ND。


