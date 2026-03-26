# aclnnCircularPad2dBackward / aclnnCircularPad3dBackward（CircularPadGrad）

## 功能描述

CircularPad 的反向：输入为正向输出的梯度 `grad_output`（与 `circular_pad` 输出同形），输出为对原输入 `self` 的梯度 `grad_input`（与正向 `self` 同形）。语义等价于对 `F.pad(..., mode="circular")` 做 autograd 反向。

## Python 封装（Bench）

编译阶段将 `custom_pybind_api` 替换为 **`kernel_gen_ops.circular_pad_grad`**，`aclnnCustomOp` 替换为 **`aclnnCircularPadGrad`**。

```python
def circular_pad_grad(grad_output: Tensor, padding: Tensor) -> Tensor:
    """
    参数:
        grad_output: 与正向输出同 dtype/device；维数 ≥3（4 元 pad）或 ≥4（6 元 pad）。
        padding: 一维 int64，须为 **CPU** contiguous；顺序与正向 CircularPad 一致：
            - 4 元：[left, right, top, bottom]（W、H）
            - 6 元：[left, right, top, bottom, front, back]（W、H、D）
    返回:
        grad_input，与正向 self 同形、同 dtype。
    """
```

## 约束与限制

- **dtype**：float16、bfloat16、float32（与 OpDef 一致）。
- **padding**：各分量须 **小于** 对应正向 `self` 空间维长度（与 aclnn 文档一致）；`grad_output` 空间维须等于 `self` 对应维加上各向 padding 之和。
- **最后一维尺寸**：文档约束 float16/bf16 下 W 维度 &lt; 16362，float32 下 &lt; 24544（与 CircularPad 系列一致）。
