# aclnnNsaCompressGrad

## 功能描述

### 算子功能

`aclnnNsaCompress` 的反向：根据 `outputGrad` 与正向输入 `input`、`weight` 及序列长度信息，计算 `inputGrad` 与 `weightGrad`。

## 接口定义

### Python 接口

通过 PyBind11 封装，Python 侧为 `kernel_gen_ops.nsa_compress_grad`：

```python
def nsa_compress_grad(
    output_grad: Tensor,
    input: Tensor,
    weight: Tensor,
    act_seq_len_cumsum: Tensor,
    compress_block_size: int,
    compress_stride: int,
    act_seq_len_type: int,
) -> tuple[Tensor, Tensor]:
    """
    参数:
        output_grad: [M, N, D]，与正向输出同形状。
        input / weight: 与正向一致。
        act_seq_len_cumsum: INT64 一维 cumsum。
        compress_block_size / compress_stride / act_seq_len_type: 与正向一致；type 当前仅支持 0。
    返回:
        (input_grad, weight_grad)，形状分别与 input、weight 一致。
    """
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**；不包含运行环境或排障说明。
- `layoutOptional` 在绑定中固定为 TND 语义。
- `actSeqLenOptional` 不可为空；`actSeqLenType` 仅支持 0。
