import numpy as np
import torch


def parse_bool_param(param):
    if isinstance(param, str) and param.lower() in ["true", "t", "1"]:
        return True
    if param == 1:
        return True
    return False


def _rand_boxes_xyxy(n: int, rng: np.random.Generator, lo=0.0, hi=10.0):
    """生成 n 个有效 xyxy 框（x2>x1, y2>y1）。"""
    boxes = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        x1, y1 = rng.uniform(lo, hi - 1, 2)
        x2 = rng.uniform(x1 + 0.1, hi)
        y2 = rng.uniform(y1 + 0.1, hi)
        boxes[i] = [x1, y1, x2, y2]
    return boxes


def _to_layout(boxes_nm4: torch.Tensor, aligned: bool) -> torch.Tensor:
    """[N,4] -> aligned 时 [4,N]，否则仍为 [N,4]。"""
    if aligned:
        return boxes_nm4.t().contiguous()
    return boxes_nm4


def custom_check_precision(row, ref_output, test_output):
    """按 dtype 放宽阈值，与 AscendC 半精度舍入一致。"""
    dtype_str = str(row.get("dtype", "float16")).lower()
    if dtype_str in ("float16", "half"):
        max_abs, max_rel = 2e-2, 2e-2
    elif dtype_str in ("bfloat16", "bf16", "bfloat"):
        max_abs, max_rel = 5e-2, 5e-2
    else:
        max_abs, max_rel = 1e-3, 1e-3

    from framework.utils import check_precision

    return check_precision(ref_output, test_output, max_abs, max_rel)


def get_inputs(param, device=None):
    rng = np.random.default_rng(7 + int(param.get("case_id", 0)))

    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype_str = str(param.get("dtype", "float16")).lower()
    dtype = dtype_map.get(dtype_str, torch.float16)

    aligned = parse_bool_param(param.get("aligned", False))
    mode = str(param.get("mode", "iou")).lower()
    if mode not in ("iou", "iof"):
        mode = "iou"
    eps = float(param.get("eps", 1.0))

    n = int(param.get("n", 32))
    n_gt = int(param.get("n_gt", 8))
    n_bbox = int(param.get("n_bbox", 12))

    if aligned:
        boxes_a = torch.tensor(_rand_boxes_xyxy(n, rng), dtype=dtype)
        boxes_b = torch.tensor(_rand_boxes_xyxy(n, rng), dtype=dtype)
        bboxes = _to_layout(boxes_a, True)
        gtboxes = _to_layout(boxes_b, True)
    else:
        boxes_a = torch.tensor(_rand_boxes_xyxy(n_bbox, rng), dtype=dtype)
        boxes_b = torch.tensor(_rand_boxes_xyxy(n_gt, rng), dtype=dtype)
        bboxes = boxes_a
        gtboxes = boxes_b

    if device:
        bboxes = bboxes.to(device)
        gtboxes = gtboxes.to(device)

    return bboxes, gtboxes, mode, eps, aligned


def get_init_inputs(param, device=None):
    return []
