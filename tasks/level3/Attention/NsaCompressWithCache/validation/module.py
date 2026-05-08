"""
CPU 金标准与 ops-transformer `executor_aclnnNsaCompressWithCache.py` 中 `__call__` 路径一致（act_seq_len 为各 batch 实际长度）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _get_last_kv(
    input_kv: torch.Tensor,
    act_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    batch_idx: int,
    compress_block_size: int,
    page_block_size: int,
) -> torch.Tensor:
    """input_kv: [blocks_num, page_bs, N, D] on float。"""
    act_seq_len = int(act_seq_lens[batch_idx].item())
    block_num = (act_seq_len - 1) // page_block_size
    tile_act_seq_len = act_seq_len - block_num * page_block_size
    if tile_act_seq_len < compress_block_size:
        cur_block_idx = int(block_table[batch_idx, block_num].item())
        pre_block_idx = int(block_table[batch_idx, block_num - 1].item())
        input_kv_cur = input_kv[cur_block_idx, :tile_act_seq_len]
        input_kv_pre = input_kv[
            pre_block_idx, page_block_size - compress_block_size + tile_act_seq_len : page_block_size
        ]
        return torch.cat((input_kv_pre, input_kv_cur), dim=0)
    block_idx = int(block_table[batch_idx, block_num].item())
    return input_kv[block_idx, tile_act_seq_len - compress_block_size : tile_act_seq_len]


def _ref_compress_with_cache(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    slot_mapping: torch.Tensor,
    act_seq_len: torch.Tensor,
    block_table: torch.Tensor,
    compress_block_size: int,
    compress_stride: int,
    output_cache: torch.Tensor,
) -> torch.Tensor:
    out = output_cache.clone()
    bsz = int(slot_mapping.shape[0])
    ipt_f = input_tensor.float()
    w_f = weight.float()
    for batch_idx in range(bsz):
        seq_len = int(act_seq_len[batch_idx].item())
        if seq_len >= compress_block_size and (seq_len - compress_block_size) % compress_stride == 0:
            last_kv = _get_last_kv(ipt_f, act_seq_len, block_table, batch_idx, compress_block_size, int(input_tensor.shape[1]))
            weight_last_kv = last_kv * w_f.unsqueeze(-1)
            compress_last_kv = weight_last_kv.sum(dim=0)
            idx = int(slot_mapping[batch_idx].item())
            out[idx].copy_(compress_last_kv.to(out.dtype))
    return out


class Model(nn.Module):
    def __init__(
        self,
        compress_block_size: int,
        compress_stride: int,
        act_seq_len_type: int,
        page_block_size: int,
        layout: str,
    ):
        super().__init__()
        self.compress_block_size = compress_block_size
        self.compress_stride = compress_stride
        self.act_seq_len_type = act_seq_len_type
        self.page_block_size = page_block_size
        self.layout = layout

    def forward(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        slot_mapping: torch.Tensor,
        act_seq_len: torch.Tensor,
        block_table: torch.Tensor,
        output_cache: torch.Tensor,
    ) -> torch.Tensor:
        return _ref_compress_with_cache(
            input_tensor,
            weight,
            slot_mapping,
            act_seq_len,
            block_table,
            self.compress_block_size,
            self.compress_stride,
            output_cache,
        )


class ModelNew(nn.Module):
    def __init__(
        self,
        compress_block_size: int,
        compress_stride: int,
        act_seq_len_type: int,
        page_block_size: int,
        layout: str,
    ):
        super().__init__()
        self.compress_block_size = compress_block_size
        self.compress_stride = compress_stride
        self.act_seq_len_type = act_seq_len_type
        self.page_block_size = page_block_size
        self.layout = layout

    def forward(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        slot_mapping: torch.Tensor,
        act_seq_len: torch.Tensor,
        block_table: torch.Tensor,
        output_cache: torch.Tensor,
    ) -> torch.Tensor:
        import kernel_gen_ops

        return kernel_gen_ops.nsa_compress_with_cache(
            input_tensor,
            weight,
            slot_mapping,
            act_seq_len,
            block_table,
            self.layout,
            int(self.compress_block_size),
            int(self.compress_stride),
            int(self.act_seq_len_type),
            int(self.page_block_size),
            output_cache,
        )
