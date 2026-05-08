# GroupedMatMulAllReduce

## 功能描述

多组矩阵乘与 AllReduce 通信融合：对动态输入列表中的每一组执行 MatMul，并在给定 HCCL 组上完成规约。`WORLD_SIZE=1` 时通信为恒等。

## Python 验证接口

`validation` 中通过 PyBind 导出 `kernel_gen_ops.grouped_mat_mul_all_reduce(...)`，参数与 `aclnnGroupedMatMulAllReduce` 两段式 API 的语义一致（pybind 内部分别调用 GetWorkspaceSize 与 launch）。

主要参数：`x_list`、`weight_list`、`bias_list`（与组数对齐的张量列表）、可选 `group_list`、`split_item`、`group`、`reduce_op`、`comm_turn`，以及作为输出缓冲的 `y_list`。

## 约束

- `x` / `weight` / `y`：FLOAT16、BFLOAT16；`bias`：FLOAT 或 FLOAT16（与 OpDef 一致）。
- 单组内 `x` 末维与 `weight` 第 0 维为契约维 K；`weight` 为 2 维 `(K, N)`。
- MC2：`group` 为通信域名字符串；`reduceOp` 验证路径常用 `sum`。

## 示例

```python
import torch
import kernel_gen_ops

# 见 validation/prepare_inputs.py 与 test_cases.csv
```
