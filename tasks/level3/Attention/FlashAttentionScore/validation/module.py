"""CPU 参考：BNSD 布局下的标准缩放点积注意力（四路输出与 aclnn 对齐）。"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def _ref_flash_attention_bnsd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_value: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """query/key/value: [B,N,S,D]；返回 softmax_max, softmax_sum, softmax_out, attention_out。"""
    q32 = query.float()
    k32 = key.float()
    v32 = value.float()
    scores = torch.matmul(q32, k32.transpose(-1, -2)) * scale_value
    smax = scores.max(dim=-1, keepdim=True)[0]
    x_sub = scores - smax
    exp_v = torch.exp(x_sub)
    sum_v = exp_v.sum(dim=-1, keepdim=True)
    prob = exp_v / sum_v
    out = torch.matmul(prob, v32)

    b, n, s1, _ = prob.shape
    smax_b = smax.expand(b, n, s1, 8)
    ssum_b = sum_v.expand(b, n, s1, 8)
    return smax_b, ssum_b, prob.to(query.dtype), out.to(query.dtype)


class Model(nn.Module):
    def __init__(
        self,
        scale_value: float,
        head_num: int,
        keep_prob: float,
        pre_tockens: int,
        next_tockens: int,
        inner_precise: int,
        sparse_mode: int,
    ):
        super().__init__()
        self.scale_value = scale_value
        self.head_num = head_num
        self.keep_prob = keep_prob
        self.pre_tockens = pre_tockens
        self.next_tockens = next_tockens
        self.inner_precise = inner_precise
        self.sparse_mode = sparse_mode

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
        softmax_out: torch.Tensor,
        attention_out: torch.Tensor,
    ) -> List[torch.Tensor]:
        del softmax_max, softmax_sum, softmax_out, attention_out
        smax, ssum, sm, out = _ref_flash_attention_bnsd(query, key, value, self.scale_value)
        return [smax, ssum, sm, out]


class ModelNew(nn.Module):
    def __init__(
        self,
        scale_value: float,
        head_num: int,
        keep_prob: float,
        pre_tockens: int,
        next_tockens: int,
        inner_precise: int,
        sparse_mode: int,
    ):
        super().__init__()
        self.scale_value = scale_value
        self.head_num = head_num
        self.keep_prob = keep_prob
        self.pre_tockens = pre_tockens
        self.next_tockens = next_tockens
        self.inner_precise = inner_precise
        self.sparse_mode = sparse_mode

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
        softmax_out: torch.Tensor,
        attention_out: torch.Tensor,
    ) -> List[torch.Tensor]:
        import kernel_gen_ops

        return kernel_gen_ops.custom_pybind_api(
            query,
            key,
            value,
            softmax_max,
            softmax_sum,
            softmax_out,
            attention_out,
            self.scale_value,
            self.head_num,
            self.keep_prob,
            self.pre_tockens,
            self.next_tockens,
            self.inner_precise,
            self.sparse_mode,
        )
