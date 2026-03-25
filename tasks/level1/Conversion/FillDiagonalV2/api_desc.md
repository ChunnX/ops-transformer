# aclnnInplaceFillDiagonal（任务 FillDiagonalV2）

## 功能描述

就地将张量主对角线（及 `wrap=True` 时的折叠对角线）填充为标量 `fillValue`。与 `torch.Tensor.fill_diagonal_(fill, wrap=...)` 一致。参考：`ops-math-dev/conversion/fill_diagonal_v2/docs/aclnnInplaceFillDiagonal.md`。

## Python 绑定

模板中函数名为 `custom_pybind_api`，`batch_compile` 后对应 `kernel_gen_ops.fill_diagonal_v2`。`EXEC_NPU_CMD` 第一参数须为 `aclnnCustomOp`（替换为 `aclnnFillDiagonalV2`）。**fillValue** 与 **xRef 须同 dtype**（aclnn EZ1001）；AICore 从 **GM** 读 fill，**pybind** 须将 fill 规范为与 **x 同 device、同 dtype、shape `[1]`**（勿将 fill 留在 CPU，否则易出现 vector core / 507035）。

```python
def fill_diagonal_v2(x: Tensor, fill_value: Tensor, wrap: bool) -> Tensor:
    """
    x: 输入输出（通常在 NPU），≥2 维；维数>2 时各维长度须相等。
    fill_value: 单元素、shape [1]；须与 **x 同 dtype、同 device**（由 prepare_inputs / ModelNew / pybind `.to` 保证）。
    wrap: Python bool；prepare_inputs / ModelNew 中应显式 bool(...) 再传入。
    """
```

## 约束

- `selfRef` 维数 ≥ 2；维度 > 2 时各维相等。
- `fillValue` 须可转为 FLOAT 做校验，且转换到 `selfRef` dtype 时不溢出。

## 示例

```python
import torch
import kernel_gen_ops

x = torch.randn(64, 64, device="npu", dtype=torch.float16)
f = torch.tensor([3.0], dtype=torch.float16, device="npu")
y = kernel_gen_ops.fill_diagonal_v2(x.clone(), f, False)
```
