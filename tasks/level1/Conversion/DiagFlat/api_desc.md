# aclnnDiagFlat

## 功能描述

将输入张量先按元素展平为一维，再构造方阵：展平后的元素落在指定对角线 `diagonal` 上，其余位置为 0。与 `torch.diagflat` 语义一致。

## Python 接口

源码中绑定函数名为模板 `custom_pybind_api`，编译时替换为与算子一致的 `diag_flat`。Python 侧调用：

```python
def diag_flat(x: Tensor, diagonal: int) -> Tensor:
    """
    参数:
        x: 输入，ND，最大 8 维；支持 float16/float32/bfloat16/float64、整型、bool、complex64 等（与 CANN 平台支持列表一致）。
        diagonal: 对角线索引；0 为主对角线，正为上三角方向，负为下三角方向。
    返回:
        二维张量，形状为 (n + |diagonal|, n + |diagonal|)，n = x.numel()。
    """
```

## 约束与限制

- 输入、输出数据类型须在被支持列表内且一致。
- 无反向。

## 用例

```python
import torch
import kernel_gen_ops

x = torch.randn(2, 3, device="npu", dtype=torch.float16)
y = kernel_gen_ops.diag_flat(x, 0)
```
