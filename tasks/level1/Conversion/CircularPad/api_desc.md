# aclnnCircularPad（CircularPad）

## 功能描述

### 算子功能

对输入张量 `x` 在 **最后两维（2D 情形）** 或 **最后三维（3D 情形）** 上做 **circular（循环）填充**：越界下标按模长绕回有效区间，等价于 PyTorch `torch.nn.functional.pad(x, padding, mode="circular")` 对应当前 `padding` 长度时的语义。

### 计算公式

- **2D（`padding` 长度为 4）**：设 `x` 形状为 `(..., H, W)`，输出在末尾两维上的空间尺寸为  
  `H' = H + top + bottom`，`W' = W + left + right`。原 `(h, w)` 处元素在填充后映射到扩展网格上仍按 circular 规则从原张量取值。
- **3D（`padding` 长度为 6）**：设 `x` 形状为 `(..., D, H, W)`，输出空间尺寸为  
  `D' = D + front + back`，`H' = H + top + bottom`，`W' = W + left + right`，同样在三维块上按循环方式取源数据。

`padding` 各分量与空间维的对应关系见下文 **约束与限制**（与 aclnn 文档 **左右上下[前后]** 顺序一致）。

## 接口定义

### Python 接口

该操作通过 PyBind11 封装 C++ 实现；Bench 中 `batch_compile` 将符号映射为 **`kernel_gen_ops.circular_pad`**（源码中由 `custom_pybind_api` 替换而来）。

```python
def circular_pad(x: Tensor, padding: Tensor) -> Tensor:
    """
    对 x 做 circular pad，调用底层 aclnnCircularPad。

    参数:
        x (Tensor): 输入张量，Device 侧（如 NPU），数据格式 ND；支持 dtype：
            int8、float16、bfloat16、float32、int32（与算子注册一致）。
        padding (Tensor): 一维 int64 张量，长度须为 4 或 6；**须为 CPU 张量**
            （Host 内存），以便构图/tiling 阶段在 Host 读取 padding 数值。
            语义顺序与 aclnn 一致：
            - 长度为 4：[left, right, top, bottom]，对应最后一维 W 与倒数第二维 H。
            - 长度为 6：[left, right, top, bottom, front, back]，对应 W、H、倒数第三维 D。

    返回:
        Tensor: 与 x 相同 dtype、device；空间维已按 padding 扩大后的 ND 张量。
    """
```

说明：在任务 `module.py` 中若 `padding` 为 list/tuple，会先转为 **`torch.int64` 且 `device="cpu"`** 再调用本接口。

## 使用案例

```python
import torch
import kernel_gen_ops

# 2D：4D 输入 [N,C,H,W]，padding 顺序 left, right, top, bottom
x = torch.randn(1, 1, 4, 4, device="npu", dtype=torch.float32)
pad = torch.tensor([1, 1, 1, 1], dtype=torch.int64, device="cpu")
y = kernel_gen_ops.circular_pad(x, pad)

# 3D：5D 输入 [N,C,D,H,W]，6 个 padding
x3 = torch.randn(1, 1, 4, 8, 16, device="npu", dtype=torch.float16)
pad3 = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.int64, device="cpu")
y3 = kernel_gen_ops.circular_pad(x3, pad3)
```

## 约束与限制

- **`padding` 长度**：仅支持 **4**（最后两维填充）或 **6**（最后三维填充）；`x` 的维数须分别满足 **≥3** 或 **≥4**。
- **`padding` 存储**：须为 **CPU** 上的 **int64**、**一维 contiguous**（与当前 validation/pybind 实现一致）；**勿**将 `padding` 放在与 `x` 相同的 NPU 上，否则 Host tiling 读 `GetData` 可能异常。
- **与 aclnnCircularPad2d / 3d 文档一致的数值约束**（违反时行为未定义或可能报错）：
  - **4 元**：`left`、`right` 均需 **小于** `x` **最后一维**长度；`top`、`bottom` 均需 **小于** `x` **倒数第二维**长度。
  - **6 元**：前两个 **< 最后一维（W）**；中间两个 **< 倒数第二维（H）**；后两个 **< 倒数第三维（D）**。
- **输出形状**：`out` 后两维（或后三维）尺寸等于 `self` 对应维尺寸加上相应方向 padding 之和（与 aclnn 文档描述一致）。
- **数据类型**：`x` 与 `y` dtype 一致；支持类型见算子 `OpDef`（int8、fp16、bf16、fp32、int32）；`padding` 为 int64。
