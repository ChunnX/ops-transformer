# aclnnStackGroupPoints

## 功能描述

### 算子功能

按 batch 将点云 **features** 与 **indices** 对齐后，根据 `indices` 中记录的采样下标，从各 batch 的特征中取出对应通道，堆叠为输出 **`y`**，形状为 `(M, C, nsample)`。`features_batch_cnt`、`indices_batch_cnt` 分别描述各 batch 在 `features` 行与 `indices` 行上的数量划分。

### 计算公式

对输出位置 `(m, c, s)`，先由 `indices_batch_cnt` 确定行 `m` 所属 batch，再由 `features_batch_cnt` 得到该 batch 在 `features` 中的起点；用 `indices[m, s]` 作为 batch 内点下标，读取 `features` 中对应 `(点, c)` 的值（与 kernel 中 `ComputeAndCopyOut` 一致）。

## 接口定义

### Python 接口

编译后由 `batch_compile.py` 将 `custom_pybind_api` 替换为 `stack_group_points`。

```python
def stack_group_points(
    features: Tensor,
    features_batch_cnt: Tensor,
    indices: Tensor,
    indices_batch_cnt: Tensor,
) -> Tensor:
    """
    参数:
        features: (N, C)，float16/float32。
        features_batch_cnt: (B+1,) int32，前 B 个为各 batch 点数且和为 N，末元素占位（与 CANN golden 一致）。
        indices: (M, nsample) int32，各行采样下标为 batch 内点索引。
        indices_batch_cnt: (B,) int32，各 batch 在 indices 中占有的行数，长度和为 M。

    返回:
        Tensor: (M, C, nsample)，dtype 与 features 一致。
    """
```

## 使用案例

```python
import torch
import torch_npu
import kernel_gen_ops

device = "npu:0"
features = torch.randn(30, 8, dtype=torch.float32, device=device)
fbc = torch.tensor([10, 20, 0], dtype=torch.int32, device=device)
indices = torch.zeros(12, 4, dtype=torch.int32, device=device)
ibc = torch.tensor([4, 8], dtype=torch.int32, device=device)

y = kernel_gen_ops.stack_group_points(features, fbc, indices, ibc)
```

## 约束与限制

- `features` 为 2D，`indices` 为 2D；两个 batch_cnt 均为 1D int32。
- `sum(features_batch_cnt[:B]) == N`（末位不计入），`sum(indices_batch_cnt) == M`。
- Tiling 区分 **float16**（tiling key 0）与 **float32**（tiling key 1），用例需覆盖两种 dtype。
- 自定义 OPP 由框架设置 `ASCEND_CUSTOM_OPP_PATH`。
