"""
CPU 参考：对无 mask 的 TND+GQA 场景给出标准缩放点积注意力（仅用于 attention_out / softmax 统计的对照）。
与 NSA「compress 块 + 重要性 topk」融合实现并非数学等价；精度通过 prepare_inputs.custom_check_precision 放宽阈值。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def _tnd_gqa_attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_value: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """query [N2,T,G,Dh], key [T2,N2,Dh], value [T2,N2,D2]。"""
    n2, t1, g, dh = query.shape
    t2, n2_k, dh_k = key.shape
    t2v, n2_v, d2 = value.shape
    assert n2 == n2_k == n2_v and t2 == t2v and dh == dh_k

    q32 = query.float()
    k32 = key.float()
    v32 = value.float()

    out = torch.zeros(n2, t1, g, d2, dtype=torch.float32)
    smax = torch.zeros(t1, n2 * g, 8, dtype=torch.float32)
    ssum = torch.zeros(t1, n2 * g, 8, dtype=torch.float32)

    scale = float(scale_value)
    for n in range(n2):
        k_vec = k32[:, n, :]
        v_vec = v32[:, n, :]
        for ti in range(t1):
            for gi in range(g):
                qv = q32[n, ti, gi, :]
                scores = (qv @ k_vec.T) * scale
                scores = scores - scores.max()
                w = torch.softmax(scores, dim=-1)
                acc = w @ v_vec
                out[n, ti, gi, :] = acc
                hi = n * g + gi
                smax[ti, hi, 0] = scores.max()
                ssum[ti, hi, 0] = w.sum()

    return smax, ssum, out


class Model(nn.Module):
    def __init__(
        self,
        scale_value: float,
        head_num: int,
        sparse_mode: int,
        compress_block_size: int,
        compress_stride: int,
        select_block_size: int,
        select_block_count: int,
    ):
        super().__init__()
        self.scale_value = scale_value
        self.head_num = head_num
        self.sparse_mode = sparse_mode
        self.compress_block_size = compress_block_size
        self.compress_stride = compress_stride
        self.select_block_size = select_block_size
        self.select_block_count = select_block_count

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
        attention_out: torch.Tensor,
        topk_indices_out: torch.Tensor,
        actual_seq_qlen: torch.Tensor,
        actual_cmp_seq_kvlen: torch.Tensor,
        actual_sel_seq_kvlen: torch.Tensor,
    ) -> List[torch.Tensor]:
        del softmax_max, softmax_sum, attention_out, actual_seq_qlen, actual_cmp_seq_kvlen, actual_sel_seq_kvlen
        smax, ssum, out = _tnd_gqa_attention_ref(query, key, value, self.scale_value)
        topk = torch.zeros(topk_indices_out.shape, dtype=torch.int32, device=query.device)
        return [
            smax.to(dtype=torch.float32),
            ssum.to(dtype=torch.float32),
            out.to(dtype=query.dtype),
            topk,
        ]


class ModelNew(nn.Module):
    def __init__(
        self,
        scale_value: float,
        head_num: int,
        sparse_mode: int,
        compress_block_size: int,
        compress_stride: int,
        select_block_size: int,
        select_block_count: int,
    ):
        super().__init__()
        self.scale_value = scale_value
        self.head_num = head_num
        self.sparse_mode = sparse_mode
        self.compress_block_size = compress_block_size
        self.compress_stride = compress_stride
        self.select_block_size = select_block_size
        self.select_block_count = select_block_count

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
        attention_out: torch.Tensor,
        topk_indices_out: torch.Tensor,
        actual_seq_qlen: torch.Tensor,
        actual_cmp_seq_kvlen: torch.Tensor,
        actual_sel_seq_kvlen: torch.Tensor,
    ) -> List[torch.Tensor]:
        import kernel_gen_ops

        return kernel_gen_ops.nsa_compress_attention(
            query,
            key,
            value,
            softmax_max,
            softmax_sum,
            attention_out,
            topk_indices_out,
            actual_seq_qlen,
            actual_cmp_seq_kvlen,
            actual_sel_seq_kvlen,
            float(self.scale_value),
            int(self.head_num),
            "TND",
            int(self.sparse_mode),
            int(self.compress_block_size),
            int(self.compress_stride),
            int(self.select_block_size),
            int(self.select_block_count),
        )
