"""
CPU 金标准与 ops-transformer `attention/mla_prolog_v2/tests/pytest/prologv2_no_quant_pa_bsnd.py`
中 GeneralizedPrologV2（BF16、PA_BSND、无 dequant/quant 可选输入）路径一致。
"""

from __future__ import annotations

import copy
import math
from typing import List, Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _mla_prolog_ref_pa_bsnd_bf16(
    token_x: torch.Tensor,
    w_dq: torch.Tensor,
    w_uq_qr: torch.Tensor,
    w_uk: torch.Tensor,
    w_dkv_kr: torch.Tensor,
    gamma_cq: torch.Tensor,
    gamma_ckv: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    index_table: torch.Tensor,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    cq_epsilon: float,
    ckv_epsilon: float,
    cache_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """与 pytest 中 GeneralizedPrologV2.forward 对齐（BF16 / PA_BSND）；第 5 路为 V2 的 dequant_scale_q_nope（非全量化形状 (1,)）。"""
    B = int(token_x.shape[0])
    S1 = int(token_x.shape[1])
    He = int(token_x.shape[2])
    Hcq = int(w_dq.shape[1])
    Hckv = int(gamma_ckv.shape[0])
    N1 = int(w_uk.shape[0])
    D = int(w_uk.shape[1])
    Dr = int(w_uq_qr.shape[1]) // N1 - D
    N2 = int(kv_cache.shape[2])
    BlockSize = int(kv_cache.shape[1])
    T = B * S1

    token_x = token_x.reshape(T, He).float()
    w_dq = w_dq.float()
    w_uq_qr = w_uq_qr.float()
    w_uk = w_uk.float()
    w_dkv_kr = w_dkv_kr.float()
    gamma_cq = gamma_cq.float()
    gamma_ckv = gamma_ckv.float()
    sin_bsd = sin.float()
    cos_bsd = cos.float()
    index_table = index_table.reshape(T).long()

    matmul1_res = torch.matmul(token_x, w_dq)
    matmul1_res = matmul1_res.to(torch.bfloat16).float()
    ep1 = float(cq_epsilon)
    norm1_res = matmul1_res / torch.sqrt(torch.mean(matmul1_res**2, dim=-1, keepdim=True) + ep1)
    norm1_res = norm1_res * gamma_cq

    matmul2_res = torch.matmul(norm1_res, w_uq_qr).float()
    matmul2_res = matmul2_res.to(torch.bfloat16).float()
    matmul2_res = matmul2_res.reshape(T, N1, D + Dr)
    splitd1_res1 = matmul2_res[:, :, :D]
    splitd1_res2 = matmul2_res[:, :, D:]

    splitd1_res1 = splitd1_res1.transpose(0, 1)
    out1 = torch.zeros((N1, T, Hckv), dtype=torch.float32)
    splitd1_res1 = splitd1_res1.to(torch.bfloat16).float()
    for n1_index in range(N1):
        out1[n1_index, :, :] = torch.matmul(splitd1_res1[n1_index, :, :], w_uk[n1_index, :, :])
    out1 = out1.transpose(0, 1).reshape(B, S1, N1, Hckv)

    expanded_cos = cos_bsd.repeat(1, N1, 1).reshape(T, N1, Dr)
    expanded_sin = sin_bsd.repeat(1, N1, 1).reshape(T, N1, Dr)
    q = splitd1_res2.reshape(T, N1, Dr // 2, 2).transpose(3, 2).reshape(T, N1, Dr)
    out2 = (q * expanded_cos) + (rotate_half(q) * expanded_sin)
    out2 = out2.reshape(B, S1, N1, Dr)

    matmul4_res = torch.matmul(token_x, w_dkv_kr).float()
    matmul4_res = matmul4_res.to(torch.bfloat16).float()
    splitd2_res1 = matmul4_res[:, :Hckv]
    splitd2_res2 = matmul4_res[:, Hckv:]

    ep2 = float(ckv_epsilon)
    norm2_res = splitd2_res1 / torch.sqrt(torch.mean(splitd2_res1**2, dim=-1, keepdim=True) + ep2)
    norm2_res = norm2_res * gamma_ckv

    kv_work = copy.deepcopy(kv_cache)
    kr_work = copy.deepcopy(kr_cache)
    out3_shape = kv_work.shape
    out4_shape = kr_work.shape
    if str(cache_mode) == "PA_BSND":
        B_num = out3_shape[0]
        kv_flat = kv_work.reshape(B_num * BlockSize, N2, Hckv)
        norm_bf = norm2_res.to(torch.bfloat16)
        for i in range(T):
            for j in range(N2):
                kv_flat[index_table[i], j, :] = norm_bf[i, :]
        out3 = kv_flat.reshape(out3_shape)

        kr_flat = kr_work.reshape(B_num * BlockSize, N2, Dr)
        cos_flat = cos_bsd.reshape(B * S1, Dr)
        sin_flat = sin_bsd.reshape(B * S1, Dr)
        k_rot = splitd2_res2.reshape(T, 1, Dr // 2, 2).transpose(3, 2).reshape(T, Dr)
        rot2 = (k_rot * cos_flat) + (rotate_half(k_rot) * sin_flat)
        rot_bf = rot2.to(torch.bfloat16)
        for i in range(T):
            for j in range(N2):
                kr_flat[index_table[i], j, :] = rot_bf[i, :]
        out4 = kr_flat.reshape(out4_shape)
    else:
        raise ValueError("validation 仅实现 PA_BSND cache_mode")

    dq_nope = torch.zeros((1,), dtype=torch.float32, device=token_x.device)
    return out1.to(torch.bfloat16), out2.to(torch.bfloat16), out3, out4, dq_nope


class Model(nn.Module):
    def __init__(self, rmsnorm_epsilon_cq: float, rmsnorm_epsilon_ckv: float, cache_mode: str):
        super().__init__()
        self.rmsnorm_epsilon_cq = rmsnorm_epsilon_cq
        self.rmsnorm_epsilon_ckv = rmsnorm_epsilon_ckv
        self.cache_mode = cache_mode

    def forward(
        self,
        token_x: torch.Tensor,
        weight_dq: torch.Tensor,
        weight_uq_qr: torch.Tensor,
        weight_uk: torch.Tensor,
        weight_dkv_kr: torch.Tensor,
        rmsnorm_gamma_cq: torch.Tensor,
        rmsnorm_gamma_ckv: torch.Tensor,
        rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        cache_index: torch.Tensor,
        kv_cache: torch.Tensor,
        kr_cache: torch.Tensor,
        query_out: torch.Tensor,
        query_rope_out: torch.Tensor,
        dequant_scale_q_nope_out: torch.Tensor,
    ) -> List[torch.Tensor]:
        del query_out, query_rope_out, dequant_scale_q_nope_out
        return list(
            _mla_prolog_ref_pa_bsnd_bf16(
                token_x,
                weight_dq,
                weight_uq_qr,
                weight_uk,
                weight_dkv_kr,
                rmsnorm_gamma_cq,
                rmsnorm_gamma_ckv,
                rope_sin,
                rope_cos,
                cache_index,
                kv_cache,
                kr_cache,
                self.rmsnorm_epsilon_cq,
                self.rmsnorm_epsilon_ckv,
                self.cache_mode,
            )
        )


class ModelNew(nn.Module):
    def __init__(self, rmsnorm_epsilon_cq: float, rmsnorm_epsilon_ckv: float, cache_mode: str):
        super().__init__()
        self.rmsnorm_epsilon_cq = rmsnorm_epsilon_cq
        self.rmsnorm_epsilon_ckv = rmsnorm_epsilon_ckv
        self.cache_mode = cache_mode

    def forward(
        self,
        token_x: torch.Tensor,
        weight_dq: torch.Tensor,
        weight_uq_qr: torch.Tensor,
        weight_uk: torch.Tensor,
        weight_dkv_kr: torch.Tensor,
        rmsnorm_gamma_cq: torch.Tensor,
        rmsnorm_gamma_ckv: torch.Tensor,
        rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        cache_index: torch.Tensor,
        kv_cache: torch.Tensor,
        kr_cache: torch.Tensor,
        query_out: torch.Tensor,
        query_rope_out: torch.Tensor,
        dequant_scale_q_nope_out: torch.Tensor,
    ) -> List[torch.Tensor]:
        import torch_npu

        dev = token_x.device

        def _to_fractal_nz(src: torch.Tensor) -> torch.Tensor:
            x = src.contiguous().to(dev)
            out = torch_npu.empty_with_format(
                tuple(x.shape), dtype=x.dtype, device=dev, acl_format=29
            )
            out.copy_(x)
            return out

        w_dq_nz = _to_fractal_nz(weight_dq)
        w_uq_nz = _to_fractal_nz(weight_uq_qr)
        w_dkv_nz = _to_fractal_nz(weight_dkv_kr)

        import kernel_gen_ops

        return list(
            kernel_gen_ops.mla_prolog_v2(
                token_x,
                w_dq_nz,
                w_uq_nz,
                weight_uk,
                w_dkv_nz,
                rmsnorm_gamma_cq,
                rmsnorm_gamma_ckv,
                rope_sin,
                rope_cos,
                cache_index,
                kv_cache,
                kr_cache,
                float(self.rmsnorm_epsilon_cq),
                float(self.rmsnorm_epsilon_ckv),
                self.cache_mode,
                query_out,
                query_rope_out,
                dequant_scale_q_nope_out,
            )
        )
