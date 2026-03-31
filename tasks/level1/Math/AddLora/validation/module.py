import torch
import torch.nn as nn

import kernel_gen_ops


def _reference_add_lora(y, x, weight_b, weight_a, indices, layer_idx, scale, y_offset, y_slice_size):
    # weight_a[w, L, :, :] is [R, H1], weight_b[w, L, :, :] is [H2, R] — 与 aclnn 文档一致
    out = y.clone()
    bsz = x.shape[0]
    for b in range(bsz):
        w = int(indices[b].item())
        wa = weight_a[w, layer_idx]
        wb = weight_b[w, layer_idx]
        z1 = (wa @ x[b].unsqueeze(-1)).squeeze(-1)
        z2 = (wb @ z1.unsqueeze(-1)).squeeze(-1) * scale
        sl = slice(y_offset, y_offset + y_slice_size)
        out[b, sl] = out[b, sl] + z2
    return out


class Model(nn.Module):
    def __init__(self, layer_idx: int, scale: float, y_offset: int, y_slice_size: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.scale = float(scale)
        self.y_offset = int(y_offset)
        self.y_slice_size = int(y_slice_size)

    def forward(self, y, x, weight_b, indices, weight_a):
        return _reference_add_lora(
            y,
            x,
            weight_b,
            weight_a,
            indices,
            self.layer_idx,
            self.scale,
            self.y_offset,
            self.y_slice_size,
        )


class ModelNew(nn.Module):
    def __init__(self, layer_idx: int, scale: float, y_offset: int, y_slice_size: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.scale = float(scale)
        self.y_offset = int(y_offset)
        self.y_slice_size = int(y_slice_size)

    def forward(self, y, x, weight_b, indices, weight_a):
        return kernel_gen_ops.add_lora(
            y,
            x,
            weight_b,
            indices,
            weight_a,
            self.layer_idx,
            self.scale,
            self.y_offset,
            self.y_slice_size,
        )
