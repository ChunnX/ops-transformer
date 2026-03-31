# aclnnIou（IouV2）

## 功能描述

对预测框 `bboxes` 与真值框 `gtboxes` 计算逐对重叠度，支持 **IOU**（交并比）与 **IOF**（交与前景面积比，分母为 gt 框面积）。边长计算时加入 `eps` 防止数值问题；交集边长为 `relu(min(x2)+eps - max(x1))` 等形式，与 AscendC 实现一致。

## 接口定义

### Python（validation）

通过 PyBind 暴露为 `kernel_gen_ops.iou_v2`（构建时由 `custom_pybind_api` 替换为 `iou_v2`）：

```python
def iou_v2(
    bboxes: torch.Tensor,
    gtboxes: torch.Tensor,
    mode: str,
    eps: float,
    aligned: bool,
) -> torch.Tensor:
    """
    mode: "iou" 或 "iof"。
    aligned=True: bboxes / gtboxes 均为 [4, N]，输出 overlap [N, 1]（逐元素一对）。
    aligned=False: bboxes [N_b, 4]，gtboxes [N_g, 4]，输出 overlap [N_g, N_b]。
    """
```

## 约束与限制

- 数据类型：FLOAT32 / FLOAT16 / BFLOAT16（以芯片与 `op_def` 配置为准）。
- `aligned=True` 时两输入第二维（框个数）必须相等。
- 非对齐模式下输入为 `[N, 4]`，最后一维为 `x1, y1, x2, y2`。

## 使用案例

```python
import torch
import kernel_gen_ops

b = torch.rand(4, 16, device="npu", dtype=torch.float16)
g = torch.rand(4, 16, device="npu", dtype=torch.float16)
out = kernel_gen_ops.iou_v2(b, g, "iou", 1.0, True)
```
