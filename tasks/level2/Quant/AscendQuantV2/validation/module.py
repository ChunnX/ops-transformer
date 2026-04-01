"""
CPU golden 与 ops-nn-dev ST 对齐：
`ops-nn-dev/quant/ascend_quant/tests/st/aclnnAscendQuant/executor_aclnnAscendQuant.py`
（.codex/TODO.md：AscendQuantV2 → quant → ops-nn-dev；L2 接口名为 aclnnAscendQuant，axis 固定 -1）。
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

# 与 executor_aclnnAscendQuant.py 中 aclnnAscendQuant 路径一致（未暴露 axis 时默认 -1）
_GOLDEN_AXIS = -1


def _golden_aclnn_ascend_quant_int8(
    x: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor],
    sqrt_mode: bool,
    round_mode: str,
    dst_type: int,
) -> torch.Tensor:
    """
    复刻 executor 中 __call__ 的计算顺序与舍入（float32 → numpy → np.round 等），仅实现 dst_type==2 (int8)。
    """
    x_t = x.to(torch.float32)
    scale_t = scale.to(torch.float32)
    offset_t = offset.to(torch.float32) if offset is not None else None

    if len(scale_t.shape) == 1:
        scale_new_shape = [1] * len(x_t.shape)
        scale_new_shape[_GOLDEN_AXIS] = scale_t.shape[0]
        scale_t = torch.reshape(scale_t, scale_new_shape)
        if offset_t is not None:
            offset_t = torch.reshape(offset_t, scale_new_shape)

    x_np = x_t.detach().cpu().numpy()
    scale_np = scale_t.detach().cpu().numpy()
    offset_np = offset_t.detach().cpu().numpy() if offset_t is not None else None

    if sqrt_mode:
        scale_sqrt = x_np * scale_np
        scale_rst = scale_sqrt * scale_np
    else:
        scale_rst = x_np * scale_np
    if offset_np is not None:
        add_offset = scale_rst + offset_np
    else:
        add_offset = scale_rst

    rm = str(round_mode).strip()
    if rm == "round":
        round_data = np.round(add_offset, 0)
    elif rm == "floor":
        round_data = np.floor(add_offset)
    elif rm == "ceil":
        round_data = np.ceil(add_offset)
    elif rm == "trunc":
        round_data = np.trunc(add_offset)
    else:
        raise ValueError(f"unsupported round_mode: {round_mode}")

    if dst_type == 2:
        round_data = np.clip(round_data, -128, 127)
        return torch.from_numpy(round_data.astype(np.int8)).to(x.device)
    raise ValueError(f"golden 仅支持 dst_type==2 (int8)，收到 {dst_type}")


class Model(nn.Module):
    def __init__(self, sqrt_mode: bool, round_mode: str, dst_type: int):
        super().__init__()
        self.sqrt_mode = bool(sqrt_mode)
        self.round_mode = str(round_mode)
        self.dst_type = int(dst_type)

    def forward(
        self, x: torch.Tensor, scale: torch.Tensor, offset: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        assert self.dst_type == 2, "validation 仅校验 int8 输出 (ge::DT_INT8 == 2)"
        y = _golden_aclnn_ascend_quant_int8(
            x, scale, offset, self.sqrt_mode, self.round_mode, self.dst_type
        )
        return [y]


class ModelNew(nn.Module):
    def __init__(self, sqrt_mode: bool, round_mode: str, dst_type: int):
        super().__init__()
        self.sqrt_mode = bool(sqrt_mode)
        self.round_mode = str(round_mode)
        self.dst_type = int(dst_type)

    def forward(
        self, x: torch.Tensor, scale: torch.Tensor, offset: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        import kernel_gen_ops

        return kernel_gen_ops.ascend_quant_v2(
            x, scale, offset, self.sqrt_mode, self.round_mode, self.dst_type
        )
