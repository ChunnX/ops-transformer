from typing import List, Tuple
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch golden model for RoPE with sin/cos cache."""

    def __init__(self, num_q_heads: int, num_kv_heads: int, head_size: int,
                 is_neox_style: bool = True, mrope_section: list = None,
                 cache_mode: int = 0):
        super(Model, self).__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        self.mrope_section = mrope_section if mrope_section is not None else [0, 0, 0]
        self.cache_mode = cache_mode

    def forward(self, positions: torch.Tensor, query_in: torch.Tensor,
                key_in: torch.Tensor, cos_sin_cache: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_out = self._apply_rope(positions, query_in, cos_sin_cache, self.num_q_heads)
        key_out = self._apply_rope(positions, key_in, cos_sin_cache, self.num_kv_heads)
        return query_out, key_out

    def _apply_rope(self, positions: torch.Tensor, x: torch.Tensor,
                    cos_sin_cache: torch.Tensor, num_heads: int) -> torch.Tensor:
        num_tokens = x.shape[0]
        head_size = self.head_size
        rotary_dim = cos_sin_cache.shape[-1]
        orig_dtype = x.dtype

        x_heads = x.reshape(num_tokens, num_heads, head_size).float()
        x_rot = x_heads[:, :, :rotary_dim]
        x_pass = x_heads[:, :, rotary_dim:]

        cos_all = cos_sin_cache[:, :rotary_dim // 2].float()
        sin_all = cos_sin_cache[:, rotary_dim // 2:].float()

        if len(positions.shape) == 1:
            pos = positions.long()
        else:
            pos = positions[0].long()

        cos_vals = cos_all[pos]  # [num_tokens, rotary_dim//2]
        sin_vals = sin_all[pos]  # [num_tokens, rotary_dim//2]

        cos_vals = cos_vals.unsqueeze(1)  # [num_tokens, 1, rd//2]
        sin_vals = sin_vals.unsqueeze(1)  # [num_tokens, 1, rd//2]

        if self.is_neox_style:
            x1 = x_rot[:, :, :rotary_dim // 2]
            x2 = x_rot[:, :, rotary_dim // 2:]
            out_rot = torch.cat(
                [x1 * cos_vals - x2 * sin_vals,
                 x2 * cos_vals + x1 * sin_vals], dim=-1)
        else:
            x1 = x_rot[:, :, 0::2]
            x2 = x_rot[:, :, 1::2]
            rotated = torch.cat(
                [x1 * cos_vals - x2 * sin_vals,
                 x2 * cos_vals + x1 * sin_vals], dim=-1)
            out_rot = torch.stack(
                [rotated[:, :, :rotary_dim // 2],
                 rotated[:, :, rotary_dim // 2:]], dim=-1)
            out_rot = out_rot.reshape(num_tokens, num_heads, rotary_dim)

        out_heads = torch.cat([out_rot.to(orig_dtype), x_pass.to(orig_dtype)], dim=-1)
        return out_heads.reshape(num_tokens, num_heads * head_size)


class ModelNew(nn.Module):
    """Custom kernel version for RoPE with sin/cos cache."""

    def __init__(self, num_q_heads: int, num_kv_heads: int, head_size: int,
                 is_neox_style: bool = True, mrope_section: list = None,
                 cache_mode: int = 0):
        super(ModelNew, self).__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        self.mrope_section = mrope_section if mrope_section is not None else [0, 0, 0]
        self.cache_mode = cache_mode

    def forward(self, positions: torch.Tensor, query_in: torch.Tensor,
                key_in: torch.Tensor, cos_sin_cache: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_stride = self.num_q_heads * self.head_size
        k_stride = self.num_kv_heads * self.head_size
        import torch_npu
        mrope_section_list = self.mrope_section
        return kernel_gen_ops.rope_with_sin_cos_cache(
            positions, query_in, key_in, cos_sin_cache,
            self.num_q_heads, self.num_kv_heads, self.head_size,
            mrope_section_list, q_stride, k_stride,
            self.is_neox_style, self.cache_mode)
