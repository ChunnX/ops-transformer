# aclnnNsaCompress

## 功能描述

### 算子功能

在 NSA 压缩路径上按 `actSeqLen` 分段，对 TND 布局的 `input` 与 `weight` 做滑窗压缩，输出压缩后的 token 特征。

### 计算公式

对每个 batch 段内、步长为 `compressStride` 的滑窗（窗长 `compressBlockSize`），将窗内 token 与权重做按头广播相乘后在窗长维度上规约（实现细节以算子为准）。

## 接口定义

### Python 接口

通过 PyBind11 封装，Python 侧为 `kernel_gen_ops.nsa_compress`：

```python
def nsa_compress(
    input: Tensor,
    weight: Tensor,
    act_seq_len_cumsum: Tensor,
    compress_block_size: int,
    compress_stride: int,
    act_seq_len_type: int,
) -> Tensor:
    """
    参数:
        input: [T, N, D]，FLOAT16/BFLOAT16。
        weight: [compressBlockSize, N]，与 input 同 dtype。
        act_seq_len_cumsum: INT64 一维，前缀和形式的各 batch 结束位置。
        compress_block_size / compress_stride: 滑窗长度与步长（须满足 Host 侧约束，如 16 对齐）。
        act_seq_len_type: 当前仅支持 0（cumsum）。
    返回:
        压缩结果 [M, N, D]，M 由序列长度与滑窗参数决定。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

input_t = torch.randn(64, 2, 16, dtype=torch.float16, device="npu")
weight = torch.randn(32, 2, dtype=torch.float16, device="npu")
act = torch.tensor([64], dtype=torch.int64, device="npu")
out = kernel_gen_ops.nsa_compress(input_t, weight, act, 32, 16, 0)
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**；不包含运行环境或排障说明。
- `layoutOptional` 在绑定中固定为 TND 语义。
- `actSeqLenOptional` 不可为空；`actSeqLenType` 仅支持 0。
- 具体 shape、对齐与数据类型约束以 Host 侧 `CheckParams` 与产品文档为准。
