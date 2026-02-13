# aclnnHansDecode

## 功能描述
数据压缩（HansEncode）的反向计算。
### 算子功能
`aclnnHansDecode` 实现了对压缩后的数据进行还原，返回还原后的数据。

### 计算公式

无

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.hans_decode()` 函数形式提供：

```python
def hans_decode(grad_y, grouped_idx):
    """
    实现数据解码操作。
    
    参数:
        mantissa_out（Tensor）:  [1, mantissa_size] 表示输出的尾数部分, dtype = float32/float16/bfloat16
        fixed_out（Tensor）: [1, fixed_size] 表示压缩的第一段输出, dtype与mantissa一致
        var_out（Tensor）: [1, var_size] 与fixed_size一致，表示压缩超过fixedOut后的输出，dtype与mantissa一致
        pdf_out（Tensor）: [1, 256], int32 指数位统计结果
        reshuff (bool): 是否对各核编码后的结果进行内存重整，默认False
    返回:
        recover (Tensor): 还原后数据
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

# 使用hans_decode_custom执行计算
recover = kernel_gen_ops.hans_decode(
            mantissa_out, fixed_out, var_out, pdf_out, reshuff
        )
```


## 约束与限制

- mantissaSize应与inputSize（即recover）数据大小有如下关系：mantissaSize == (dtypeBytes - 1) * inputSize // dtypeBytes。dtypeBytes为数据类型字节大小（如，float32为4）。
- fixedSize应与inputSize有如下关系： fixedSize == inputSize // dtypeBytes。
