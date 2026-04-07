# aclnnDynamicQuant

## 功能描述

### 算子功能

对最后一维做 **per-token（pertoken）对称** 动态量化：按每个「行」（除最后一维外下标相同的一组元素）计算 `max(abs(x * smooth))`（无 `smooth` 时为 `max(abs(x))`），将量化系数写回 `scale`，并将对应片段量化到 **int8**。

### 计算公式

对每个前缀下标 `p`（对应 `x` 的前 `ndim-1` 维），记 `v = x[p, :]`，若提供 `smooth` 则先逐元素 `v = v * smooth`（`smooth` 为一维，长度等于 `x` 最后一维，与 `x` 广播相乘）。则：

1. `m = max_i |v_i|`（若 `m` 过小则实现侧有 epsilon 保护）
2. 输出 `scale[p] = m / 127`
3. `y[p, :] = rint(v * (127 / m))` 并 clip 到 `[-128, 127]`，dtype 为 int8  

**CPU golden**：见 `validation/module.py` 中 `golden_dynamic_quant_int8`，与 `op_kernel/dynamic_quant.h` 对称路径（`Compute`）一致；`rint` 对齐内核 CAST_RINT 主路径。

## 接口定义

### Python 接口

Bench 编译后由 PyBind 导出 **`kernel_gen_ops.dynamic_quant`**（与任务目录 `DynamicQuant` 的 `kernel_name` 一致），底层为 **`aclnnDynamicQuant`**。

```python
def dynamic_quant(
    x: torch.Tensor,
    smooth_scales: Optional[torch.Tensor],
) -> List[torch.Tensor]:
    """
    参数:
        x: NPU 侧 ND 张量，秩 >= 2、< 9；dtype 与芯片能力一致（如 310P 多为 float16）。
        smooth_scales: 可选，一维，长度等于 x 最后一维，dtype 与 x 一致；为 None 时不乘 smooth。

    返回:
        [y, scale]：y 与 x 同 shape、int8；scale 为 float32，shape 为 x.shape[:-1]。
    """
```

## 使用案例

```python
import torch
import torch_npu
import kernel_gen_ops

device = "npu:0"
x = torch.randn(8, 32, device=device, dtype=torch.float16)
smooth = torch.rand(32, device=device, dtype=torch.float16) * 0.5 + 0.25
y, scale = kernel_gen_ops.dynamic_quant(x, smooth)
y2, scale2 = kernel_gen_ops.dynamic_quant(x, None)
```

## 约束与限制

- 本文档仅描述算子功能、接口与数据约束；不包含 conda、CANN、shell、`run_multi_test` 或运行环境说明。
- L2 **`aclnnDynamicQuant`** 固定为对称、pertoken；**不**通过本接口传入 `group_index`、`quantMode` 等（完整能力见 `aclnnDynamicQuantV2` / `aclnnDynamicQuantV3`）。
- **bfloat16** 与 **910B / 部分架构** 能力相关；仅 310P 类环境时请在 `test_cases.csv` 中避免 bf16 用例或按环境裁剪。
- `test_cases.csv` 覆盖可选 smooth、不同末维对齐/规模及 fp16/bf16，以触发 310P / 非 310P 等多套 tiling 分支。
