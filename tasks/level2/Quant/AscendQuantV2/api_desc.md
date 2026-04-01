# aclnnAscendQuant

## 功能描述

### 算子功能

对浮点张量 `x` 按 `scale`（及可选的 `offset`）做缩放与偏移后，按指定舍入模式量化到整型（本 validation 仅覆盖 **int8** 输出）。可选 `sqrt_mode` 时，在实现上与再乘一次 `scale` 等价（与 AICore 中 `Mul` 两次路径一致）。

### 计算公式

记广播后的 `scale` 为与 `x` 同形状（或可广播到 `x`），则中间结果为：

1. `t = x * scale`
2. 若 `sqrt_mode` 为真：`t = t * scale`
3. 若提供 `offset`：`t = t + offset`
4. 按 `round_mode` 舍入：`round` 与 **numpy `np.round(..., 0)`** 一致（与 ST golden 相同），其余为 `floor` / `ceil` / `trunc`；再 **clip** 到 `[-128, 127]` 并转为 int8。

**CPU golden**：`validation/module.py` 中 `Model` 与 **`ops-nn-dev/quant/ascend_quant/tests/st/aclnnAscendQuant/executor_aclnnAscendQuant.py`** 对齐（见 **`.codex/TODO.md`** 中 AscendQuantV2 → ops-nn-dev）；一维 `scale`/`offset` 会按 **`axis=-1`** 扩成与 `x` 同秩再参与运算。

## 接口定义

### Python 接口

该操作通过 PyBind11 封装 C++ 实现；Bench 编译后 Python 中函数名为 `kernel_gen_ops.ascend_quant_v2`（与目录 `AscendQuantV2` 对应的 `kernel_name` 一致）。底层调用 **libcust_opapi** 中的 **`aclnnAscendQuant`**（与 `op_api/aclnn_ascend_quant.cpp` 一致）。

```python
def ascend_quant_v2(
    x: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor],
    sqrt_mode: bool,
    round_mode: str,
    dst_type: int,
) -> List[torch.Tensor]:
    """
    参数:
        x: NPU 侧 ND 浮点张量，dtype 支持 float16 / float32 / bfloat16（与芯片能力一致）。
        scale: NPU 侧张量；除最后一维外其余维须为 1，且最后一维与 x 最后一维相同或为 1；dtype 与 x 须满足 aclnn 兼容性（如 fp16 配 fp16 scale）。
        offset: 可选，形状与 dtype 约束与 scale 类似；为 None 时不参与加法。
        sqrt_mode: 是否启用 sqrt 路径（与内核两次乘 scale 行为一致）。
        round_mode: 字符串，取值为 round / floor / ceil / trunc 之一。
        dst_type: 输出逻辑类型，validation 使用 ge::DT_INT8 的整型枚举值 **2**。

    返回:
        单元素列表，元素为与 x 同 shape 的 int8 张量。
    """
```

## 使用案例

```python
import torch
import torch_npu
import kernel_gen_ops

device = "npu:0"
x = torch.randn(8, 32, device=device, dtype=torch.float16)
scale = torch.rand(1, 32, device=device, dtype=torch.float16) * 0.1 + 0.05
offset = torch.zeros(1, 32, device=device, dtype=torch.float16)
out = kernel_gen_ops.ascend_quant_v2(x, scale, offset, False, "round", 2)
y = out[0]  # int8
```

## 约束与限制

- 输出为 **int8** 时，`dst_type` 与输出 tensor 的 dtype 需与 **ge::DT_INT8** 一致；本任务 validation 固定 `dst_type=2`（以当前 CANN/GE 枚举为准，若环境不同请按头文件核对）。
- `scale` / `offset` 须满足 aclnn 侧 `CheckDimOne` 与广播规则。
- **FRACTAL_NZ** 等格式走独立 tiling 分支，本 `test_cases.csv` 未构造 NZ 输入，仅覆盖 ND + **per_channel / per_tensor** 主路径。
- L2 封装中 `axis` 固定为 **-1**，与 **per-head** 专用 tiling 分支无对齐用例；若需覆盖 `KEY_PER_HEAD`，需在 aclnn 层暴露 `axis` 后再扩展 CSV。
- `bfloat16` 与 **910B / regbase** 等能力相关，在仅 310P 的环境上部分用例可能被 aclnn 拒绝。
