"""SageAttention2 CPU Golden Reference Engine.

Provides an exact bit-equivalent Python implementation of the quantized QK INT8 /
PV MXFP8 attention algorithm for Ascend NPU validation and unit testing.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, NamedTuple
import torch

QUAN_GRAN_GROUPS: Dict[str, Tuple[int, int]] = {
    "per_thread": (4, 16),
    "per_warp": (32, 64),
}

PV_PARTIAL_SHIFT: int = 8
PV_PARTIAL_SCALE: float = 2.0 ** (-8)
PV_PARTIAL_UNDO: float = 2.0 ** 8

class QuantizedPack(NamedTuple):
    q8: torch.Tensor
    k8: torch.Tensor
    v8: torch.Tensor
    qs: torch.Tensor
    ks: torch.Tensor
    mean_k: torch.Tensor
    vs: torch.Tensor
    mean_v: torch.Tensor

class SageAttention2CPUReference:
    @staticmethod
    def quantize_grouped_int8(tensor_in: torch.Tensor, group_span: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(tensor_in.size(-2))
        row_peak = tensor_in.abs().amax(dim=-1, keepdim=True)
        if group_span > 1:
            pad_tokens = int((-seq_len) % group_span)
            padded_row = torch.nn.functional.pad(row_peak, (0, 0, 0, pad_tokens))
            reshaped_row = padded_row.reshape(*tensor_in.shape[:-2], -1, group_span).amax(dim=-1, keepdim=True)
            row_peak = reshaped_row.repeat_interleave(group_span, dim=-2)[..., :seq_len, :]
        scale_div = torch.where(row_peak > 0, row_peak / 127.0, torch.ones_like(row_peak))
        quant_int8 = torch.div(tensor_in, scale_div).round().clamp(-127, 127).to(torch.int8)
        return (quant_int8, scale_div.squeeze(-1))

    @staticmethod
    def quantize_v_mx(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_dim = int(v.size(-2))
        rem_pad = int((-s_dim) % 32)
        v_padded = torch.nn.functional.pad(v.to(dtype=torch.float32), (0, 0, 0, rem_pad))
        v_blocks = v_padded.reshape(*v.shape[:-2], -1, 32, v.shape[-1])
        ch_max = v_blocks.abs().amax(dim=-2, keepdim=True)
        clipped_max = torch.clamp_min(ch_max * (1.0 / 448.0), 1e-30)
        e8_scales = torch.exp2(torch.ceil(torch.log2(clipped_max)))
        quant_e4m3 = torch.div(v_blocks, e8_scales).to(torch.float8_e4m3fn)
        unpadded_v = quant_e4m3.reshape_as(v_padded)[..., :s_dim, :]
        return (unpadded_v, e8_scales.squeeze(-2))

    @classmethod
    def quantize_operands(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                          smooth_k: bool = True, qg: int = 1, kg: int = 1, smooth_v: bool = False) -> QuantizedPack:
        q_fp, k_fp, v_fp = (t.to(dtype=torch.float32) for t in (q, k, v))
        mean_k = k_fp.mean(dim=-2, keepdim=True) if bool(smooth_k) is True else torch.zeros_like(k_fp[..., :1, :])
        k_norm = torch.sub(k_fp, mean_k)
        mean_v = v_fp.mean(dim=-2, keepdim=True) if bool(smooth_v) is True else torch.zeros_like(v_fp[..., :1, :])
        v_norm = torch.sub(v_fp, mean_v)
        q8, qs = cls.quantize_grouped_int8(q_fp, int(qg))
        k8, ks = cls.quantize_grouped_int8(k_norm, int(kg))
        v8, vs = cls.quantize_v_mx(v_norm)
        return QuantizedPack(q8, k8, v8, qs, ks, mean_k.squeeze(-2), vs, mean_v.squeeze(-2))

    @staticmethod
    def prepare_scale_multipliers(qs: torch.Tensor, ks: torch.Tensor, sm_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        base_factor = math.sqrt(abs(float(sm_scale)) * 2048.0)
        q_factor = -base_factor if sm_scale < 0.0 else base_factor
        q_half = (qs.to(dtype=torch.float32) * q_factor).to(dtype=torch.float16)
        k_half = (ks.to(dtype=torch.float32) * base_factor).to(dtype=torch.float16)
        return (q_half, k_half)

    @classmethod
    def execute_attention(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                          sm_scale: Optional[float] = None, is_causal: bool = False,
                          smooth_k: bool = True, smooth_v: bool = False,
                          qk_quant_gran: str = "per_thread", pv_accum_dtype: str = "fp32+fp16",
                          tensor_layout: str = "HND", return_lse: bool = False,
                          block_k: int = 256):
        is_nhd = (tensor_layout == "NHD")
        if bool(is_nhd) is True:
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        batch_size, num_qh, seq_q, head_dim = [int(x) for x in q.shape]
        num_kh, seq_k = int(k.shape[1]), int(k.shape[2])
        if bool(is_causal) is True and seq_q != seq_k:
            raise ValueError("causal requires Sq == Sk")
        if qk_quant_gran not in QUAN_GRAN_GROUPS:
            raise ValueError(f"qk_quant_gran must be one of {sorted(QUAN_GRAN_GROUPS)}")
        gran_q, gran_k = QUAN_GRAN_GROUPS[qk_quant_gran]
        use_fp16_pv = (pv_accum_dtype == "fp32+fp16")
        enable_smooth_v = bool(smooth_v) and (pv_accum_dtype == "fp32")
        tau_scale = (1.0 / math.sqrt(head_dim)) if sm_scale is None else float(sm_scale)

        qp = cls.quantize_operands(q, k, v, smooth_k=smooth_k, qg=gran_q, kg=gran_k, smooth_v=enable_smooth_v)
        v_recon = qp.v8.to(dtype=torch.float32) * torch.repeat_interleave(qp.vs, 32, dim=-2)[..., :seq_k, :]
        q_hscale, k_hscale = cls.prepare_scale_multipliers(qp.qs, qp.ks, tau_scale)
        log448_const = torch.tensor(math.log(448.0), dtype=torch.float16, device=q.device)

        out_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse_buffer = torch.empty([batch_size, num_qh, seq_q], dtype=torch.float32, device=q.device)
        tile_q_len = 64
        gqa_ratio = int(num_qh // num_kh)

        for h_idx in range(0, int(num_qh)):
            kh_idx = int(h_idx // gqa_ratio)
            for q_start in range(0, int(seq_q), int(tile_q_len)):
                q_end = min(q_start + tile_q_len, seq_q)
                tile_rows = int(q_end - q_start)
                run_max = torch.full([batch_size, tile_rows, 1], -torch.inf, device=q.device)
                run_den = torch.zeros_like(run_max)
                run_acc = torch.zeros([batch_size, tile_rows, head_dim], device=q.device)
                causal_bound = min(seq_k, ((q_start // 128) + 1) * 128) if bool(is_causal) is True else seq_k

                for k_start in range(0, int(causal_bound), int(block_k)):
                    k_end = min(k_start + block_k, seq_k)
                    int_dot = qp.q8[:, h_idx, q_start:q_end].to(dtype=torch.float32) @ qp.k8[:, kh_idx, k_start:k_end].to(dtype=torch.float32).transpose(-2, -1)
                    q_factor_tile = q_hscale[:, h_idx, q_start:q_end, None].to(dtype=torch.float32)
                    k_factor_tile = k_hscale[:, kh_idx, None, k_start:k_end].to(dtype=torch.float32)
                    fp16_factor = (q_factor_tile * k_factor_tile).to(dtype=torch.float16)
                    raw_qk = ((int_dot * (1.0 / 2048.0)).to(dtype=torch.float16).to(dtype=torch.float32) * fp16_factor.to(dtype=torch.float32)).to(dtype=torch.float16)
                    if bool(is_causal) is True:
                        q_pos = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                        k_pos = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                        raw_qk.masked_fill_(k_pos > q_pos, -torch.inf)
                    sub_max = (raw_qk.amax(dim=-1, keepdim=True) - log448_const).to(dtype=torch.float16)
                    updated_max = torch.maximum(run_max.to(dtype=torch.float16), sub_max).to(dtype=torch.float32)
                    exp_alpha = torch.exp(run_max - updated_max)
                    prob_tensor = torch.exp(raw_qk.to(dtype=torch.float32) - updated_max)
                    fp8_prob = prob_tensor.clamp(max=448.0).to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
                    run_den = run_den * exp_alpha + fp8_prob.sum(dim=-1, keepdim=True)
                    pv_gemm = (fp8_prob * PV_PARTIAL_SCALE) @ v_recon[:, kh_idx, k_start:k_end]
                    if bool(use_fp16_pv) is True:
                        pv_gemm = pv_gemm.to(dtype=torch.float16).to(dtype=torch.float32)
                    run_acc = run_acc * exp_alpha + pv_gemm
                    run_max = torch.clone(updated_max)

                norm_out = run_acc * (PV_PARTIAL_UNDO / run_den)
                if bool(enable_smooth_v) is True:
                    norm_out = norm_out + qp.mean_v[:, kh_idx, None, :]
                out_buffer[:, h_idx, q_start:q_end] = norm_out.to(dtype=q.dtype)
                lse_buffer[:, h_idx, q_start:q_end] = (run_max + torch.log(run_den)).squeeze(-1)
                if bool(smooth_k) is True:
                    q_mean_corr = (q[:, h_idx, q_start:q_end].to(dtype=torch.float32) * qp.mean_k[:, kh_idx, None, :]).sum(dim=-1) * tau_scale
                    lse_buffer[:, h_idx, q_start:q_end] += q_mean_corr

        if bool(is_nhd) is True:
            out_buffer = out_buffer.transpose(1, 2)
        return (out_buffer, lse_buffer) if bool(return_lse) is True else out_buffer

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
              sm_scale: Optional[float] = None, is_causal: bool = False,
              smooth_k: bool = True, smooth_v: bool = False,
              qk_quant_gran: str = "per_thread", pv_accum_dtype: str = "fp32+fp16",
              tensor_layout: str = "HND", return_lse: bool = False,
              block_k: int = 256):
    return SageAttention2CPUReference.execute_attention(
        q, k, v, sm_scale=sm_scale, is_causal=is_causal, smooth_k=smooth_k, smooth_v=smooth_v,
        qk_quant_gran=qk_quant_gran, pv_accum_dtype=pv_accum_dtype, tensor_layout=tensor_layout,
        return_lse=return_lse, block_k=block_k
    )
