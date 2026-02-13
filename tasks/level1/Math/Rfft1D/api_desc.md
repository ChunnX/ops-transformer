# aclnnRfft1D

## 功能描述

### 算子功能
对输入实信号做一维实数 FFT（RFFT），输出为复数频谱的实部与虚部（或等价表示）。支持 Cooley-Tukey 与 Bluestein 算法，norm 支持 backward/forward/ortho。

### 参数
- **x**: 输入实值张量 (float32)
- **n**: 可选，FFT 长度
- **norm**: 1=backward, 2=forward, 3=ortho

## 接口定义

### Python 接口

```python
def rfft1_d(input: Tensor, n: Optional[int], norm: int):
    """
    参数:
        input (Tensor): 输入实信号，float32。
        n (int, optional): FFT 长度，默认最后一维长度。
        norm (int): 1=backward, 2=forward, 3=ortho。
    返回:
        Tensor: RFFT 结果。
    """
```

## 约束与限制

- 输入 dtype 为 float32。
- n 需大于 0；norm 取 1/2/3。
