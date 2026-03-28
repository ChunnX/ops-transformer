# aclnnTransposeV2

## 功能描述

### 算子功能

按 `perm` 对输入 `x` 做维度置换（ND），仅支持 tiling 已实现的置换：`(0,2,1)`、`(1,0,2)`（3D）与 `(0,2,1,3)`（4D）。输出 `y` 与 `x` dtype 一致。

### 计算公式

`y[i0,...,i_{n-1}] = x[...,]`，其中输出第 `k` 维对应输入第 `perm[k]` 维（与 `torch.permute` 一致）。

## 接口定义

### Python 接口

该操作通过 PyBind11 封装 C++ 实现，编译替换后在 Python 中以 `kernel_gen_ops.transpose_v2()` 提供（模板中为 `custom_pybind_api` / `aclnnCustomOp` → `aclnnTransposeV2`）。`perm` 建议在 **CPU** 上为 `int64` 且 **contiguous**（tiling 在 Host 读 `perm` 数据指针）。

```python
def transpose_v2(x: Tensor, perm: Tensor) -> Tensor:
    """
    参数:
        x: NPU 上 ND 张量，float16 / bfloat16 / float32。
        perm: 一维 int64，长度等于 x.dim()，取值须为上述支持的置换之一。

    返回:
        y: 与 x 同 dtype，shape 为 perm 描述下的置换结果。
    """
```

## 使用案例

```
import torch
import torch_npu
import kernel_gen_ops

device = "npu:0"
x = torch.randn(2, 3, 4, device=device, dtype=torch.float32)
perm = torch.tensor([0, 2, 1], dtype=torch.int64, device="cpu")

y = kernel_gen_ops.transpose_v2(x, perm)
```

## 约束与限制

- `perm` 仅支持 `[0,2,1]`、`[1,0,2]`（3D）与 `[0,2,1,3]`（4D）；其它排列 tiling 会失败。
- FP16/BF16 与 FP32 走不同 tiling key；半精度可参考 `prepare_inputs.custom_check_precision` 放宽阈值。
- `perm` 若需在 Host 侧解引用，须 **CPU contiguous**，与 `tips.md` 中 Host 读参约定一致。
