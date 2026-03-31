import torch
import torch.nn as nn


def _overlap_reference(
    bboxes_nm4: torch.Tensor,
    gt_nm4: torch.Tensor,
    eps: float,
    mode: str,
) -> torch.Tensor:
    """
    与 IouV2 kernel 一致的 CPU 参考：面积在边长上加 eps 再相乘；
    交集边长为 relu(min(x2)+eps - max(x1)) 等形式。
    bboxes_nm4: (N_a, 4), gt_nm4: (N_b, 4)，xyxy。
    返回 (N_b, N_a)，与 GE inferShape 中 overlap[gt, bbox] 一致。
    """
    a = bboxes_nm4.unsqueeze(0).float()  # 1, N_a, 4  — 列对应 bbox
    b = gt_nm4.unsqueeze(1).float()  # N_b, 1, 4 — 行对应 gt

    x1a, y1a, x2a, y2a = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x1b, y1b, x2b, y2b = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    wa = (x2a - x1a) + eps
    ha = (y2a - y1a) + eps
    wb = (x2b - x1b) + eps
    hb = (y2b - y1b) + eps
    area_a = wa * ha
    area_b = wb * hb

    ix1 = torch.maximum(x1a, x1b)
    iy1 = torch.maximum(y1a, y1b)
    ix2 = torch.minimum(x2a, x2b)
    iy2 = torch.minimum(y2a, y2b)
    iw = torch.relu(ix2 + eps - ix1)
    ih = torch.relu(iy2 + eps - iy1)
    inter = iw * ih

    if mode == "iof":
        denom = area_b.clamp_min(1e-45)
    else:
        denom = (area_a + area_b - inter).clamp_min(1e-45)
    return (inter / denom).to(bboxes_nm4.dtype)


class Model(nn.Module):
    """CPU 金标准：与 op_kernel 中 IOU/IOF 公式对齐。"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        bboxes: torch.Tensor,
        gtboxes: torch.Tensor,
        mode: str,
        eps: float,
        aligned: bool,
    ) -> torch.Tensor:
        mode = mode.lower() if isinstance(mode, str) else "iou"
        if aligned:
            # [4, N] -> (N, 4)
            boxes_a = bboxes.t().contiguous()
            boxes_b = gtboxes.t().contiguous()
            o = _overlap_reference(boxes_a, boxes_b, eps, mode)
            return torch.diagonal(o).to(bboxes.dtype).unsqueeze(1)
        return _overlap_reference(bboxes, gtboxes, eps, mode)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        bboxes: torch.Tensor,
        gtboxes: torch.Tensor,
        mode: str,
        eps: float,
        aligned: bool,
    ) -> torch.Tensor:
        import kernel_gen_ops

        return kernel_gen_ops.iou_v2(bboxes, gtboxes, str(mode), float(eps), bool(aligned))
