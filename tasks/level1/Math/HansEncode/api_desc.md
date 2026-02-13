# aclnnHansEncode

## 功能描述
数据压缩（HansEncode）。
### 算子功能
`aclnnHansEncode` 实现了对输入的inputTensor的指数位所在字节实现PDF统计，返回统计的pdf结果，以及decode时需要的数据信息。

### 计算公式

无

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.hans_encode()` 函数形式提供：

```python
def hans_encode(grad_y, grouped_idx):
    """
    实现数据压缩操作。
    
    参数:
        input_tensor (Tensor): shape支持多维，要求数据数量是32768的倍数，dtype=float16/bfloat16/f32
        pdf_ref (Tensor): shape [1, 256]，dtype=int32
        statistic (bool): 是否进行 PDF 统计，默认为True
        reshuff (bool): 是否对各核编码后的结果进行内存重整，默认False
          
    返回:
        pdf_out（Tensor）: [1, 256], int32 指数位统计结果
        mantissa_out（Tensor）: 表示输出的尾数部分, dtype与input一致
        fixed_out（Tensor）: 表示压缩的第一段输出, dtype与input一致
        var_out（Tensor）: 表示压缩超过fixedOut后的输出, dtype与input一致
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入
dtype = 'float32'
device = 'npu'
input_tensor = torch.randn(1, 65536, dtype=dtype, device=device)
pdf_ref = torch.zeros(1, 256, dtype=torch.int32, device=device)
statistic = True 
reshuff = False 

# 使用hans_encode_custom执行计算
pdf_out, mantissa_out, fixed_out, var_out = kernel_gen_ops.hans_encode(input_tensor, pdf_ref, statistic, reshuff)
```


## 约束与限制

- input_tensor数据量要求是32768的倍数。