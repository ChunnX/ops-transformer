# aclnnTransformBiasRescaleQkv

## 功能描述

### 算子功能
将 QKV 融合张量（含 batch、token、3*dim）与 bias 相加，并按头拆分、缩放后输出 Q、K、V 三个张量。用于注意力等场景的 QKV 变换。

### 计算公式
- 输入 qkv 形状为 [batch, token, 3*dim]，qkv_bias 形状为 [3*dim]。
- 输出 q、k、v 形状均为 [batch, num_heads, token, dim_per_head]，其中 dim_per_head = dim / num_heads。
- 对 Q 部分会乘以缩放因子 1/sqrt(dim_per_head)。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.transform_bias_rescale_qkv()` 函数形式提供：

```python
def transform_bias_rescale_qkv(qkv: Tensor, qkv_bias: Tensor, num_heads: int):
    """
    实现 QKV 加 bias、按头拆分与 rescale。

    参数:
        qkv (Tensor): 输入 QKV 融合张量，形状 [batch, token, 3*dim]，支持 float16/float32/bfloat16。
        qkv_bias (Tensor): bias，形状 [3*dim]，与 qkv 同 dtype。
        num_heads (int): 头数。

    返回:
        (Tensor, Tensor, Tensor): (q, k, v)，每个形状 [batch, num_heads, token, dim_per_head]。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

batch, token, dim, num_heads = 2, 128, 256, 8
qkv = torch.randn(batch, token, 3 * dim, dtype=torch.float16, device='npu:0')
qkv_bias = torch.randn(3 * dim, dtype=torch.float16, device='npu:0')
q, k, v = kernel_gen_ops.transform_bias_rescale_qkv(qkv, qkv_bias, num_heads)
```

## 约束与限制

- 张量数据格式支持 ND。
- dim 必须能被 num_heads 整除；3*dim 为 qkv 最后一维。
- qkv_bias 形状必须为 [3*dim]。
