import logging
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from ml_dtypes import bfloat16
from dataclasses import dataclass

random.seed(21)
np.random.seed(21)
torch.manual_seed(21)
torch.set_printoptions(precision=6, sci_mode=False)
torch.set_printoptions(threshold=float('inf'))

DEBUG = False
if not DEBUG:
    print = lambda *args, **kwargs: None


# =====================================================================
# MXFP4 (E2M1) 量化/反量化/打包工具
# =====================================================================
QUANT_MODE_OCP = 2
QUANT_MODE_CX = 3
CX_DEFAULT_DST_TYPE_MAX = 6.0
E8M0_BIAS = 127
FP4_E2M1_EMAX = 2

FP4_E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

_FP4_POS_LEVELS_FP32 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_FP4_POS_MIDPOINTS_FP32 = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def _resolve_cx_params(dst_type_max):
    cx_val = CX_DEFAULT_DST_TYPE_MAX if (dst_type_max is None or dst_type_max == 0) else float(dst_type_max)
    log2_cx = math.log2(cx_val)
    ceil_log2_cx = math.ceil(log2_cx)
    return log2_cx, ceil_log2_cx


def generate_fp4_e2m1_encoding_table():
    encodings, values = [], []
    for s in [0, 1]:
        for e in range(4):
            for m in range(2):
                enc = (s << 3) | (e << 1) | m
                val = (-1) ** s * (2.0 ** (e - 1)) * (1.0 + m / 2.0) if e != 0 \
                    else (-1) ** s * (2.0 ** 0) * (m / 2.0)
                encodings.append(enc)
                values.append(val)
    fp32_to_encoding = {}
    for enc, val in zip(encodings, values):
        if val not in fp32_to_encoding:
            fp32_to_encoding[val] = enc
    return fp32_to_encoding


FP32_TO_FP4 = generate_fp4_e2m1_encoding_table()


def quantize_fp4(tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        expanded = tensor.unsqueeze(-1)
        idx = torch.argmin(torch.abs(expanded - FP4_E2M1_VALUES.to(tensor.device)), dim=-1)
        return FP4_E2M1_VALUES.to(tensor.device)[idx]


def compute_e8m0_scale_block(block: torch.Tensor, mode: str = "baseline",
                             quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0) -> torch.Tensor:
    max_abs = torch.max(torch.abs(block))
    if max_abs.item() < 1e-30:
        return torch.tensor(1.0, device=block.device)
    log_max = torch.log2(max_abs)
    if quant_mode == QUANT_MODE_CX:
        exp = torch.ceil(log_max - log2_cx)
    else:
        exp = torch.floor(log_max) - FP4_E2M1_EMAX if mode == "baseline" \
            else torch.ceil(log_max) - FP4_E2M1_EMAX
    exp = torch.clamp(exp, -127.0, 127.0)
    return 2.0 ** exp


def mx_quantize(tensor: torch.Tensor, block_size: int = 32, mode: str = "baseline",
                quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0):
    original_shape = tensor.shape
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    last = tensor.shape[-1]
    pad = 0
    if last % block_size != 0:
        pad = block_size - (last % block_size)
        tensor = torch.nn.functional.pad(tensor, (0, pad), value=0.0)
    reshaped = tensor.reshape(-1, block_size)
    n_blk = reshaped.shape[0]
    scales = torch.empty(n_blk, device=tensor.device)
    fp4 = torch.empty_like(reshaped)
    for i in range(n_blk):
        sc = compute_e8m0_scale_block(reshaped[i], mode=mode,
                                      quant_mode=quant_mode, log2_cx=log2_cx)
        scales[i] = sc
        fp4[i] = quantize_fp4(reshaped[i] / sc)
    fp4 = fp4.view(tensor.shape)
    if pad > 0:
        fp4 = fp4[..., :-pad]
    return fp4, scales


def mx_dequantize(fp4_vals: torch.Tensor, scales: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    original_shape = fp4_vals.shape
    last = original_shape[-1] if fp4_vals.dim() > 0 else 1
    pad = 0
    if fp4_vals.dim() > 0 and last % block_size != 0:
        pad = block_size - (last % block_size)
        fp4_vals = torch.nn.functional.pad(fp4_vals, (0, pad), value=0.0)
    reshaped = fp4_vals.reshape(-1, block_size)
    dq = reshaped * scales.unsqueeze(1).to(reshaped.device)
    dq = dq.view(fp4_vals.shape)
    if pad > 0:
        dq = dq[..., :last]
    return dq


def mxfp4_qdq(t: torch.Tensor, axis: int, block_size: int = 32, mode: str = "baseline",
              quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0) -> torch.Tensor:
    t_last = t.movedim(axis, -1).contiguous()
    fp4_sim, scales = mx_quantize(t_last, block_size=block_size, mode=mode,
                                  quant_mode=quant_mode, log2_cx=log2_cx)
    dq = mx_dequantize(fp4_sim, scales, block_size=block_size)
    return dq.movedim(-1, axis).contiguous()


def convert_fp32_simulated_to_fp4_encoding(fp32_sim: torch.Tensor) -> torch.Tensor:
    flat = fp32_sim.flatten().cpu()
    enc = torch.zeros_like(flat, dtype=torch.uint8)
    for fp32_val, encoding in FP32_TO_FP4.items():
        enc[flat == fp32_val] = encoding
    return enc.to(fp32_sim.device).reshape(fp32_sim.shape)


def pack_fp4_to_uint8(fp4_enc: torch.Tensor) -> torch.Tensor:
    shape = fp4_enc.shape
    if shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {shape[-1]}")
    reshaped = fp4_enc.reshape(-1, 2)
    packed = (reshaped[:, 1] << 4) | (reshaped[:, 0] & 0x0F)
    return packed.reshape(*shape[:-1], shape[-1] // 2)


def mxfp4_quantize_pack_last(tensor: torch.Tensor, quant_axis: int,
                             block_size: int = 32, mode: str = "baseline",
                             quant_mode: int = QUANT_MODE_OCP, log2_cx: float = 0.0):
    nd = tensor.dim()
    qax = quant_axis % nd
    assert tensor.shape[-1] % 2 == 0, f"last dim must be even, got {tensor.shape[-1]}"
    t_q_last = tensor.movedim(qax, -1).contiguous()
    fp4_sim, scales_flat = mx_quantize(t_q_last, block_size=block_size, mode=mode,
                                       quant_mode=quant_mode, log2_cx=log2_cx)
    codes = convert_fp32_simulated_to_fp4_encoding(fp4_sim).movedim(-1, qax).contiguous()
    packed = pack_fp4_to_uint8(codes)
    qdim = tensor.shape[qax]
    n_blk = (qdim + block_size - 1) // block_size
    rest = list(t_q_last.shape[:-1])
    scales_real = scales_flat.reshape(*rest, n_blk).movedim(-1, qax).contiguous()
    exp_int = torch.log2(scales_real).round().long()
    e8m0 = (exp_int + E8M0_BIAS).clamp(0, 254).to(torch.uint8)
    return packed, e8m0


def _e8m0_qk_to_v3(e8m0_uint8):
    *lead, nd = e8m0_uint8.shape
    paired = e8m0_uint8.reshape(*lead, nd // 2, 2)
    return paired.contiguous().view(torch.float8_e8m0fnu)


def _e8m0_v_to_v3(e8m0_uint8):
    B, N2, nb, D = e8m0_uint8.shape
    if nb % 2 != 0:
        pad_block = torch.full((B, N2, 1, D), E8M0_BIAS, dtype=torch.uint8)
        e8m0_uint8 = torch.cat([e8m0_uint8, pad_block], dim=2)
        nb += 1
    return e8m0_uint8.reshape(B, N2, nb // 2, 2, D).permute(0, 1, 2, 4, 3).contiguous().view(torch.float8_e8m0fnu)


# =====================================================================
# blockwise_snap_local P量化
# =====================================================================
def _pack_fp4_e2m1_along_last(idx_tensor):
    *batch, n = idx_tensor.shape
    if n % 2 == 1:
        pad = torch.zeros(*batch, 1, dtype=idx_tensor.dtype, device=idx_tensor.device)
        idx_tensor = torch.cat([idx_tensor, pad], dim=-1)
        n += 1
    pairs = idx_tensor.reshape(*batch, n // 2, 2)
    low = pairs[..., 0].to(torch.uint8) & 0xF
    high = pairs[..., 1].to(torch.uint8) & 0xF
    return (high << 4) | low


def _blockwise_snap_local_quantize_p_eff(S: torch.Tensor, m_running, seqk_dim: int,
                                          mx_block_size: int = 32, mx_mode: str = "baseline",
                                          is_cx: bool = False,
                                          neg_log2_cx: float = 0.0,
                                          p_quant_exp_shift: float = 2.0):
    NEG_INF = float('-inf')
    device = S.device
    src_dtype = S.dtype
    ndim = S.dim()
    skp = seqk_dim if seqk_dim >= 0 else ndim + seqk_dim
    seqk_len = S.shape[skp]
    LN2 = math.log(2.0)

    pad = (mx_block_size - seqk_len % mx_block_size) % mx_block_size
    if pad > 0:
        pshape = list(S.shape)
        pshape[skp] = pad
        S_padded = torch.cat([S, torch.full(pshape, NEG_INF, dtype=src_dtype, device=device)], dim=skp)
    else:
        S_padded = S
    S_moved = S_padded.movedim(skp, -1)
    other = S_moved.shape[:-1]
    n_blk = S_moved.shape[-1] // mx_block_size
    S_r = S_moved.reshape(*other, n_blk, mx_block_size)

    m_block_raw = S_r.max(dim=-1).values
    m_block_raw_safe = torch.where(torch.isinf(m_block_raw) & (m_block_raw < 0),
                                   torch.zeros_like(m_block_raw), m_block_raw)
    if is_cx:
        INV_LN2_fp16 = torch.tensor(1.0 / LN2, dtype=torch.float16, device=device)
        LN2_fp16 = torch.tensor(LN2, dtype=torch.float16, device=device)
        neg_log2_cx_fp16 = torch.tensor(neg_log2_cx, dtype=torch.float16, device=device)
        k_raw = m_block_raw_safe * INV_LN2_fp16 + neg_log2_cx_fp16
        K_block = torch.ceil(k_raw)
        m_block_snap = K_block * LN2_fp16
    else:
        INV_LN2_fp16 = torch.tensor(1.0 / LN2, dtype=torch.float16, device=device)
        LN2_fp16 = torch.tensor(LN2, dtype=torch.float16, device=device)
        K_block = torch.floor(m_block_raw_safe * INV_LN2_fp16)
        m_block_snap = (K_block - 2) * LN2_fp16

    P_local = torch.exp(S_r - m_block_snap.unsqueeze(-1))
    P_local_bf = P_local.to(torch.bfloat16).to(P_local.dtype)

    levels = _FP4_POS_LEVELS_FP32.to(device=device, dtype=src_dtype)
    midpoints = _FP4_POS_MIDPOINTS_FP32.to(device=device, dtype=src_dtype)
    idx = torch.bucketize(P_local_bf.contiguous(), midpoints, right=True)
    P_q = levels[idx]

    K_ij = K_block.max(dim=-1).values
    K_ij_fp32 = K_ij.float()
    m_new = K_ij_fp32 if m_running is None else torch.maximum(m_running, K_ij_fp32)
    K_new_s = m_new.to(src_dtype)

    if is_cx:
        K_diff = (K_block - K_new_s.unsqueeze(-1))
        corr = torch.exp2(K_diff - torch.tensor(p_quant_exp_shift, dtype=torch.float16, device=device))
    else:
        K_diff = K_block - K_new_s.unsqueeze(-1)
        corr = torch.exp2(K_diff - 2.0)

    corr_broad = corr.repeat_interleave(mx_block_size, dim=-1)[..., :seqk_len].movedim(-1, skp)
    P_q_moved = P_q.reshape(*other, n_blk * mx_block_size)[..., :seqk_len].movedim(-1, skp)
    P_eff = P_q_moved * corr_broad
    return P_eff, m_new


# =====================================================================
# s_dtype 解析
# =====================================================================
_S_DTYPE_MAP = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
_OUT_DTYPE_MAP = {'bfloat16': torch.bfloat16, 'float16': torch.float16}


def _resolve_s_dtype(s_dtype):
    if s_dtype not in _S_DTYPE_MAP:
        raise ValueError(f"s_dtype must be one of {list(_S_DTYPE_MAP)}, got '{s_dtype}'")
    return _S_DTYPE_MAP[s_dtype]


def _resolve_out_dtype(out_dtype):
    if out_dtype not in _OUT_DTYPE_MAP:
        raise ValueError(f"out_dtype must be one of {list(_OUT_DTYPE_MAP)}, got '{out_dtype}'")
    return _OUT_DTYPE_MAP[out_dtype]


# =====================================================================
# block sparse pattern 生成
# =====================================================================
def gen_block_sparse_pattern(q_seqlen_list, kv_seqlen_list, block_shape, batch, num_heads,
                             sparsity_ratio, is_varied_sparsity):
    q_s_aligned = max(q_seqlen_list)
    kv_s_aligned = max(kv_seqlen_list)
    s_block_x = block_shape[0]
    s_block_y = block_shape[1]
    q_s_block_num_aligned = (q_s_aligned + s_block_x - 1) // s_block_x
    kv_s_block_num_aligned = (kv_s_aligned + s_block_y - 1) // s_block_y

    block_sparse_mask = np.zeros((batch, num_heads, q_s_block_num_aligned, kv_s_block_num_aligned), dtype=np.int8)
    block_sparse_idx = np.ones((batch, num_heads, q_s_block_num_aligned, kv_s_block_num_aligned), dtype=np.int32)
    block_sparse_idx = -block_sparse_idx
    block_sparse_count = np.zeros((batch, num_heads, q_s_block_num_aligned), dtype=np.int32)

    for b in range(batch):
        q_s_block_num_aval = (q_seqlen_list[b] + s_block_x - 1) // s_block_x
        kv_s_block_num_aval = (kv_seqlen_list[b] + s_block_y - 1) // s_block_y
        max_rsvd_kv_s_block_num = max(1, int(kv_s_block_num_aval * sparsity_ratio))
        for n1 in range(num_heads):
            for q_blk_id in range(q_s_block_num_aval):
                cur_row_sparse_mask = np.zeros(kv_s_block_num_aval, dtype=np.int8)
                rsvd_kv_s_block_num = max_rsvd_kv_s_block_num
                if is_varied_sparsity:
                    random.seed(b * num_heads * q_s_block_num_aval + n1 * q_s_block_num_aval + q_blk_id)
                    rsvd_kv_s_block_num = random.randint(1, max_rsvd_kv_s_block_num)
                cur_row_sparse_idx = sorted(np.random.choice(kv_s_block_num_aval, rsvd_kv_s_block_num, replace=False))
                for idx in cur_row_sparse_idx:
                    cur_row_sparse_mask[idx] = 1
                block_sparse_mask[b, n1, q_blk_id, 0:kv_s_block_num_aval] = cur_row_sparse_mask
                block_sparse_idx[b, n1, q_blk_id, 0:rsvd_kv_s_block_num] = cur_row_sparse_idx
                block_sparse_count[b, n1, q_blk_id] = rsvd_kv_s_block_num
    return block_sparse_mask, block_sparse_idx, block_sparse_count


# =====================================================================
# 数据生成辅助
# =====================================================================
def randn_x(shape, x=1.0, dtype=torch.float32, device=None):
    data_range = [-x, x]
    return data_range[0] + (data_range[1] - data_range[0]) * torch.rand(shape, dtype=dtype, device=device)


# =====================================================================
# CPU golden — BNSD block-sparse MXFP4 分块 flash attention
# =====================================================================
def cpu_block_sparse_attention_bnsd_mxfp4_with_lse(query, key, value, block_sparse_mask,
                                                    block_shape, scale_value, num_kv_heads,
                                                    p_quant='snap_local', s_dtype='fp16',
                                                    out_dtype='bfloat16',
                                                    quant_mode=QUANT_MODE_OCP, dst_type_max=0.0):
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    B, N, S, D = q.shape
    N2 = k.shape[1]
    S2 = k.shape[2]
    group_size = N // N2
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (S + block_x - 1) // block_x
    ceil_kv = (S2 + block_y - 1) // block_y
    s_torch_dtype = _resolve_s_dtype(s_dtype)

    is_cx = (quant_mode == QUANT_MODE_CX)
    neg_log2_cx = 0.0
    log2_cx = 0.0
    if is_cx:
        log2_cx, ceil_log2_cx = _resolve_cx_params(dst_type_max)
        neg_log2_cx = -log2_cx
        p_quant_exp_shift = float(ceil_log2_cx)
    else:
        p_quant_exp_shift = 2.0

    q_q = mxfp4_qdq(q, axis=-1, quant_mode=quant_mode, log2_cx=log2_cx)
    k_q = mxfp4_qdq(k, axis=-1, quant_mode=quant_mode, log2_cx=log2_cx)
    v_q = mxfp4_qdq(v, axis=2, quant_mode=quant_mode, log2_cx=log2_cx)

    out = torch.zeros((B, N, S, D), dtype=torch.float32)
    lse = torch.full((B, N, S, 1), float('-inf'), dtype=torch.float32)
    scale_s = torch.as_tensor(scale_value, dtype=s_torch_dtype)

    for b in range(B):
        for n in range(N):
            n2 = n // group_size
            for qb in range(ceil_q):
                q_lo = qb * block_x
                q_hi = min(q_lo + block_x, S)
                if q_lo >= S:
                    break
                sel_j = [j for j in range(ceil_kv) if block_sparse_mask[b, n, qb, j] != 0]
                if not sel_j:
                    continue
                chunks = [sel_j[i:i + 2] for i in range(0, len(sel_j), 2)]
                Q_blk = q_q[b, n, q_lo:q_hi]
                nq = q_hi - q_lo
                m_run = None
                l_run = torch.zeros(nq, dtype=torch.float32)
                acc = torch.zeros((nq, D), dtype=torch.float32)

                for chunk in chunks:
                    positions = []
                    for j in chunk:
                        st = j * block_y
                        ed = min(st + block_y, S2)
                        positions.extend(range(st, ed))
                    if not positions:
                        continue
                    pos = torch.as_tensor(positions, dtype=torch.long)
                    K_blk = k_q[b, n2][pos]
                    V_blk = v_q[b, n2][pos]

                    S_fp32 = (K_blk @ Q_blk.t()).to(torch.float32)
                    S_s = (S_fp32 * scale_s).to(s_torch_dtype)

                    if p_quant == 'snap_local':
                        P_eff_s, m_new = _blockwise_snap_local_quantize_p_eff(
                            S_s, m_running=m_run, seqk_dim=0,
                            is_cx=is_cx, neg_log2_cx=neg_log2_cx, p_quant_exp_shift=p_quant_exp_shift)
                        P_eff_s = torch.nan_to_num(P_eff_s, nan=0.0)
                        P_fp32 = P_eff_s.float()
                        alpha = (torch.zeros(nq, dtype=torch.float32) if m_run is None
                                 else torch.exp2(m_run.float() - m_new.float()))
                    else:
                        m_s = S_s.max(dim=0).values
                        m_safe_s = torch.where(torch.isinf(m_s) & (m_s < 0), torch.zeros_like(m_s), m_s)
                        P_ij = torch.exp(S_s - m_safe_s.unsqueeze(0))
                        P_fp32 = torch.nan_to_num(P_ij, nan=0.0).float()
                        m_new = m_safe_s.float()
                        alpha = (torch.zeros(nq, dtype=torch.float32) if m_run is None
                                 else torch.exp(m_run - m_new))

                    reduce_sum = P_fp32.sum(dim=0)
                    l_run = alpha * l_run + reduce_sum
                    acc = alpha.unsqueeze(-1) * acc + (P_fp32.t() @ V_blk)
                    m_run = m_new

                l_safe = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
                acc = acc / l_safe.unsqueeze(-1)
                out[b, n, q_lo:q_hi] = acc
    return out.to(_resolve_out_dtype(out_dtype)), lse.to(torch.float32)


# =====================================================================
# 精度比较
# =====================================================================
def check_result(expect, result, data_type, pct_thd=0.005):
    real_data = result.cpu().numpy().flatten()
    data_compe = expect.cpu().numpy().flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        return 100.0, "Pass"
    if real_data.size != data_compe.size:
        return 0.0, "Failed"
    if data_type in ('bfloat16', 'float16'):
        rtol, atol, max_diff_hd = 0.0078125, 0.0001, 10.0
    else:
        rtol, atol, max_diff_hd = 0.005, 0.000025, 10.0
    diff_result = np.isclose(real_data.astype(np.float32), data_compe.astype(np.float32),
                             rtol=rtol, atol=atol, equal_nan=True)
    err_idx = np.where(diff_result != np.array((True,)))[0]
    split_count = real_data.size
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0
    diff_abs = np.abs(data_compe - real_data)
    b = np.add(np.maximum(np.abs(real_data), np.abs(data_compe)), 10e-10)
    err_diff = (diff_abs / b)[err_idx]
    pct_thd_val = (1 - pct_thd) * 100.0
    result_str = "Pass" if fulfill_percent >= pct_thd_val else "Failed"
    if len(err_diff) > 0 and np.max(err_diff) >= max_diff_hd:
        result_str = "Failed"
    return fulfill_percent, result_str


# =====================================================================
# ATK golden 计算类
# =====================================================================
class TestBlockSparseAttentionMxfp4GenBmGt():
    @dataclass
    class AuxAttrs:
        num_heads: int
        kv_heads: int
        embedding_size: int
        inner_prec: int
        scale: float
        q_layout: str
        kv_layout: str
        attention_out_dtype: str
        quant_mode: int
        dst_type_max: float

    @dataclass
    class AttentionInputs:
        query: any
        key: any
        value: any
        block_sparse_mask: any
        block_sparse_idx: any
        block_sparse_count: any
        block_sparse_shape: any
        q_seqlen_list: any
        kv_seqlen_list: any
        aux_attrs: any

    def compute_output(self, attention_inputs):
        """Benchmark 分支（is_benchmark=False）：MXFP4 qdq + snap_local P量化 分块flash"""
        query_fp32 = attention_inputs.query
        key_fp32 = attention_inputs.key
        value_fp32 = attention_inputs.value
        block_sparse_mask_np = attention_inputs.block_sparse_mask
        block_sparse_idx = attention_inputs.block_sparse_idx
        block_sparse_count = attention_inputs.block_sparse_count
        block_sparse_shape = attention_inputs.block_sparse_shape
        q_seqlen_list = attention_inputs.q_seqlen_list
        kv_seqlen_list = attention_inputs.kv_seqlen_list

        batch = len(q_seqlen_list)
        num_heads = attention_inputs.aux_attrs.num_heads
        kv_heads = attention_inputs.aux_attrs.kv_heads
        embedding_size = attention_inputs.aux_attrs.embedding_size
        q_layout = attention_inputs.aux_attrs.q_layout
        kv_layout = attention_inputs.aux_attrs.kv_layout
        scale = attention_inputs.aux_attrs.scale
        inner_prec = attention_inputs.aux_attrs.inner_prec
        out_dtype_str = attention_inputs.aux_attrs.attention_out_dtype
        quant_mode = attention_inputs.aux_attrs.quant_mode
        dst_type_max = attention_inputs.aux_attrs.dst_type_max
        group_size = int(num_heads / kv_heads)

        block_shape_x = block_sparse_shape[0]
        block_shape_y = block_sparse_shape[1]

        q_tensor = torch.from_numpy(query_fp32).float()
        k_tensor = torch.from_numpy(key_fp32).float()
        v_tensor = torch.from_numpy(value_fp32).float()

        if q_layout == "TND":
            q_parts, k_parts, v_parts = [], [], []
            q_offset, kv_offset = 0, 0
            for b in range(batch):
                sq, skv = q_seqlen_list[b], kv_seqlen_list[b]
                q_parts.append(q_tensor[q_offset:q_offset + sq].unsqueeze(0).permute(0, 2, 1, 3))
                k_parts.append(k_tensor[kv_offset:kv_offset + skv].unsqueeze(0).permute(0, 2, 1, 3))
                v_parts.append(v_tensor[kv_offset:kv_offset + skv].unsqueeze(0).permute(0, 2, 1, 3))
                q_offset += sq
                kv_offset += skv
        elif q_layout == "BNSD":
            q_parts = [q_tensor[b:b + 1, :, :q_seqlen_list[b]] for b in range(batch)]
            k_parts = [k_tensor[b:b + 1, :, :kv_seqlen_list[b]] for b in range(batch)]
            v_parts = [v_tensor[b:b + 1, :, :kv_seqlen_list[b]] for b in range(batch)]
        elif q_layout == "BSND":
            q_parts = [q_tensor[b:b + 1, :q_seqlen_list[b]].permute(0, 2, 1, 3) for b in range(batch)]
            k_parts = [k_tensor[b:b + 1, :kv_seqlen_list[b]].permute(0, 2, 1, 3) for b in range(batch)]
            v_parts = [v_tensor[b:b + 1, :kv_seqlen_list[b]].permute(0, 2, 1, 3) for b in range(batch)]

        mask_tensor = torch.from_numpy(block_sparse_mask_np)

        cpu_chunks = []
        for bi in range(batch):
            cq_b = (q_seqlen_list[bi] + block_shape_x - 1) // block_shape_x
            ck_b = (kv_seqlen_list[bi] + block_shape_y - 1) // block_shape_y
            mask_b = mask_tensor[bi:bi + 1, :, :cq_b, :ck_b]
            out_b, _ = cpu_block_sparse_attention_bnsd_mxfp4_with_lse(
                q_parts[bi], k_parts[bi], v_parts[bi], mask_b,
                [block_shape_x, block_shape_y], scale, kv_heads,
                p_quant='snap_local', s_dtype='fp16', out_dtype=out_dtype_str,
                quant_mode=quant_mode, dst_type_max=dst_type_max)
            cpu_chunks.append(out_b)

        return self._assemble_result(cpu_chunks, q_layout, batch, num_heads, embedding_size, q_seqlen_list)

    def compute_output_benchmark(self, attention_inputs):
        """Benchmark 真值分支（is_benchmark=True）：用量化前的原始 fp32 QKV 做标准 BSA"""
        query_fp32 = attention_inputs.query
        key_fp32 = attention_inputs.key
        value_fp32 = attention_inputs.value
        block_sparse_mask_np = attention_inputs.block_sparse_mask
        block_sparse_idx = attention_inputs.block_sparse_idx
        block_sparse_count = attention_inputs.block_sparse_count
        block_sparse_shape = attention_inputs.block_sparse_shape
        q_seqlen_list = attention_inputs.q_seqlen_list
        kv_seqlen_list = attention_inputs.kv_seqlen_list

        batch = len(q_seqlen_list)
        num_heads = attention_inputs.aux_attrs.num_heads
        kv_heads = attention_inputs.aux_attrs.kv_heads
        embedding_size = attention_inputs.aux_attrs.embedding_size
        q_layout = attention_inputs.aux_attrs.q_layout
        kv_layout = attention_inputs.aux_attrs.kv_layout
        scale = attention_inputs.aux_attrs.scale
        group_size = int(num_heads / kv_heads)

        block_shape_x = block_sparse_shape[0]
        block_shape_y = block_sparse_shape[1]

        # 直接用原始 fp32 数据（不做 mxfp4 qdq）
        q_tensor = torch.from_numpy(query_fp32).float()
        k_tensor = torch.from_numpy(key_fp32).float()
        v_tensor = torch.from_numpy(value_fp32).float()

        if q_layout == "TND":
            q_parts, k_parts, v_parts = [], [], []
            q_offset, kv_offset = 0, 0
            for b in range(batch):
                sq, skv = q_seqlen_list[b], kv_seqlen_list[b]
                q_parts.append(q_tensor[q_offset:q_offset + sq].unsqueeze(0).permute(0, 2, 1, 3))
                k_parts.append(k_tensor[kv_offset:kv_offset + skv].unsqueeze(0).permute(0, 2, 1, 3))
                v_parts.append(v_tensor[kv_offset:kv_offset + skv].unsqueeze(0).permute(0, 2, 1, 3))
                q_offset += sq
                kv_offset += skv
        elif q_layout == "BNSD":
            q_parts = [q_tensor[b:b + 1, :, :q_seqlen_list[b]] for b in range(batch)]
            k_parts = [k_tensor[b:b + 1, :, :kv_seqlen_list[b]] for b in range(batch)]
            v_parts = [v_tensor[b:b + 1, :, :kv_seqlen_list[b]] for b in range(batch)]
        elif q_layout == "BSND":
            q_parts = [q_tensor[b:b + 1, :q_seqlen_list[b]].permute(0, 2, 1, 3) for b in range(batch)]
            k_parts = [k_tensor[b:b + 1, :kv_seqlen_list[b]].permute(0, 2, 1, 3) for b in range(batch)]
            v_parts = [v_tensor[b:b + 1, :kv_seqlen_list[b]].permute(0, 2, 1, 3) for b in range(batch)]

        mask_tensor = torch.from_numpy(block_sparse_mask_np)

        cpu_chunks = []
        for bi in range(batch):
            cq_b = (q_seqlen_list[bi] + block_shape_x - 1) // block_shape_x
            ck_b = (kv_seqlen_list[bi] + block_shape_y - 1) // block_shape_y
            mask_b = mask_tensor[bi:bi + 1, :, :cq_b, :ck_b]

            q_b = q_parts[bi]   # [1, N, S1, D] fp32
            k_b = k_parts[bi]   # [1, N2, S2, D] fp32
            v_b = v_parts[bi]   # [1, N2, S2, D] fp32
            B_b, N_b, S1_b, D_b = q_b.shape
            N2_b = k_b.shape[1]
            S2_b = k_b.shape[2]

            out_b = torch.zeros((1, N_b, S1_b, D_b), dtype=torch.float32)

            for n in range(N_b):
                n2 = n // group_size
                for qb_idx in range(cq_b):
                    q_lo = qb_idx * block_shape_x
                    q_hi = min(q_lo + block_shape_x, S1_b)
                    if q_lo >= S1_b:
                        break
                    sel_j = [j for j in range(ck_b) if mask_b[0, n, qb_idx, j] != 0]
                    if not sel_j:
                        continue
                    positions = []
                    for j in sel_j:
                        st = j * block_shape_y
                        ed = min(st + block_shape_y, S2_b)
                        positions.extend(range(st, ed))
                    if not positions:
                        continue
                    pos = torch.as_tensor(positions, dtype=torch.long)
                    Q_blk = q_b[0, n, q_lo:q_hi]       # [nq, D]
                    K_blk = k_b[0, n2][pos]             # [npos, D]
                    V_blk = v_b[0, n2][pos]             # [npos, D]

                    # fp32 标准 attention
                    s = (Q_blk.float() @ K_blk.float().t()) * scale
                    row_max = s.max(dim=-1, keepdim=True).values
                    p = torch.exp(s - row_max)
                    p = p / p.sum(dim=-1, keepdim=True)
                    o = p.float() @ V_blk.float()
                    out_b[0, n, q_lo:q_hi] = o

            cpu_chunks.append(out_b)

        return self._assemble_result(cpu_chunks, q_layout, batch, num_heads, embedding_size, q_seqlen_list)

    def _assemble_result(self, cpu_chunks, q_layout, batch, num_heads, embedding_size, q_seqlen_list):
        """将 per-batch BNSD 结果拼回原始 layout"""
        if q_layout == "TND":
            result_parts = []
            for bi in range(batch):
                result_parts.append(cpu_chunks[bi].permute(0, 2, 1, 3).flatten(0, 1))
            result = torch.cat(result_parts, dim=0)
        elif q_layout == "BNSD":
            max_s = max(q_seqlen_list)
            result = torch.zeros(batch, num_heads, max_s, embedding_size,
                                 dtype=cpu_chunks[0].dtype)
            for bi in range(batch):
                result[bi, :, :q_seqlen_list[bi]] = cpu_chunks[bi][0]
        elif q_layout == "BSND":
            max_s = max(q_seqlen_list)
            result = torch.zeros(batch, max_s, num_heads, embedding_size,
                                 dtype=cpu_chunks[0].dtype)
            for bi in range(batch):
                result[bi, :q_seqlen_list[bi]] = cpu_chunks[bi][0].permute(1, 0, 2)

        return result.to(torch.float32)


def tensor_to_numpy_array(t):
    if t is None:
        return None
    if t.dtype is torch.bfloat16:
        return t.to(torch.float32).numpy().astype(bfloat16)
    else:
        return t.numpy()


def numpy_array_to_tensor(arr):
    if arr is None:
        return None
    if isinstance(arr, list):
        arr = np.array(arr)
    if arr.dtype.name == 'bfloat16':
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(arr)


# =====================================================================
# ATK 接口注册: 直接调用 block_sparse_attention_v3
# =====================================================================
try:
    from cann_ops_transformer.ops import block_sparse_attention_v3
    _HAS_BSA_V3 = True
except Exception:
    block_sparse_attention_v3 = None
    _HAS_BSA_V3 = False


@register("aclnn_blocksparseattention_v3")
class BlockSparseAttentionV3Api(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(BlockSparseAttentionV3Api, self).__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        logging.info(f"=======================================init_by_input_data=======================================")
        self.q_input_layout = input_data.kwargs["qInputLayout"]
        self.kv_input_layout = input_data.kwargs["kvInputLayout"]
        self.kv_heads = input_data.kwargs["numKeyValueHeads"]
        self.scale_value = input_data.kwargs["scaleValue"]
        self.inner_precise = input_data.kwargs["innerPrecise"]
        self.softmax_lse_flag = input_data.kwargs["softmaxLseFlag"]
        self.quant_mode = input_data.kwargs.get("quantMode", QUANT_MODE_OCP)
        self.dst_type_max = input_data.kwargs.get("dstTypeMax", 0.0)
        self.block_shape = input_data.kwargs["blockShape"]
        self.q_seqlen_list = input_data.kwargs["actualSeqLengths"]
        self.kv_seqlen_list = input_data.kwargs["actualSeqLengthsKv"]

        attention_out = input_data.kwargs["attentionOut"]
        if attention_out.dtype == torch.bfloat16:
            self.attention_out_dtype_str = "bfloat16"
            self.attention_out_torch_dtype = torch.bfloat16
        else:
            self.attention_out_dtype_str = "float16"
            self.attention_out_torch_dtype = torch.float16

        if self.q_input_layout == "TND":
            self.batch = len(self.q_seqlen_list)
            query_packed = input_data.kwargs["query"]
            self.num_heads = query_packed.shape[1]
            self.headDim = query_packed.shape[2] * 2
        elif self.q_input_layout == "BNSD":
            query_packed = input_data.kwargs["query"]
            self.batch = query_packed.shape[0]
            self.num_heads = query_packed.shape[1]
            self.headDim = query_packed.shape[3] * 2
        elif self.q_input_layout == "BSND":
            query_packed = input_data.kwargs["query"]
            self.batch = query_packed.shape[0]
            self.num_heads = query_packed.shape[2]
            self.headDim = query_packed.shape[3] * 2

        random.seed(21)
        np.random.seed(21)
        torch.manual_seed(21)

        sparsity_ratio = random.uniform(0.0, 1.0)
        is_varied_sparsity = 0

        self.log2_cx = 0.0
        if self.quant_mode == QUANT_MODE_CX:
            self.log2_cx, _ = _resolve_cx_params(self.dst_type_max)

        q_s_list = self.q_seqlen_list
        kv_s_list = self.kv_seqlen_list

        if self.q_input_layout == "TND":
            T1 = sum(q_s_list)
            T2 = sum(kv_s_list)
            self.query_fp32 = randn_x((T1, self.num_heads, self.headDim), 1.0).numpy()
            self.key_fp32 = randn_x((T2, self.kv_heads, self.headDim), 1.0).numpy()
            self.value_fp32 = randn_x((T2, self.kv_heads, self.headDim), 1.0).numpy()
        elif self.q_input_layout == "BNSD":
            max_s1 = max(q_s_list)
            max_s2 = max(kv_s_list)
            self.query_fp32 = randn_x((self.batch, self.num_heads, max_s1, self.headDim), 1.0).numpy()
            self.key_fp32 = randn_x((self.batch, self.kv_heads, max_s2, self.headDim), 1.0).numpy()
            self.value_fp32 = randn_x((self.batch, self.kv_heads, max_s2, self.headDim), 1.0).numpy()
        elif self.q_input_layout == "BSND":
            max_s1 = max(q_s_list)
            max_s2 = max(kv_s_list)
            self.query_fp32 = randn_x((self.batch, max_s1, self.num_heads, self.headDim), 1.0).numpy()
            self.key_fp32 = randn_x((self.batch, max_s2, self.kv_heads, self.headDim), 1.0).numpy()
            self.value_fp32 = randn_x((self.batch, max_s2, self.kv_heads, self.headDim), 1.0).numpy()

        self.block_sparse_mask_np, self.block_sparse_idx, self.block_sparse_count = gen_block_sparse_pattern(
            self.q_seqlen_list, self.kv_seqlen_list,
            self.block_shape,
            self.batch, self.num_heads,
            sparsity_ratio, is_varied_sparsity)

        self.block_sparse_mask_tensor = numpy_array_to_tensor(self.block_sparse_mask_np)

        # 打包 QKV 为 MXFP4 格式
        q_fp32_tensor = torch.from_numpy(self.query_fp32).float()
        k_fp32_tensor = torch.from_numpy(self.key_fp32).float()
        v_fp32_tensor = torch.from_numpy(self.value_fp32).float()

        if self.q_input_layout == "TND":
            q_p_ch, k_p_ch, v_p_ch = [], [], []
            q_s_ch, k_s_ch, v_s_ch = [], [], []
            q_offset, kv_offset = 0, 0
            for bi in range(self.batch):
                sq, skv = q_s_list[bi], kv_s_list[bi]
                q_b = q_fp32_tensor[q_offset:q_offset + sq].unsqueeze(0).permute(0, 2, 1, 3)
                k_b = k_fp32_tensor[kv_offset:kv_offset + skv].unsqueeze(0).permute(0, 2, 1, 3)
                v_b = v_fp32_tensor[kv_offset:kv_offset + skv].unsqueeze(0).permute(0, 2, 1, 3)
                q_p, q_e = mxfp4_quantize_pack_last(q_b, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
                k_p, k_e = mxfp4_quantize_pack_last(k_b, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
                v_p, v_e = mxfp4_quantize_pack_last(v_b, quant_axis=-2, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
                q_p_ch.append(q_p.permute(0, 2, 1, 3).flatten(0, 1))
                k_p_ch.append(k_p.permute(0, 2, 1, 3).flatten(0, 1))
                v_p_ch.append(v_p.permute(0, 2, 1, 3).flatten(0, 1))
                q_s_ch.append(_e8m0_qk_to_v3(q_e).permute(0, 2, 1, 3, 4).flatten(0, 1))
                k_s_ch.append(_e8m0_qk_to_v3(k_e).permute(0, 2, 1, 3, 4).flatten(0, 1))
                v_s_ch.append(_e8m0_v_to_v3(v_e).permute(0, 2, 1, 3, 4).flatten(0, 1))
                q_offset += sq
                kv_offset += skv
            self.q_packed = torch.cat(q_p_ch, dim=0).contiguous()
            self.k_packed = torch.cat(k_p_ch, dim=0).contiguous()
            self.v_packed = torch.cat(v_p_ch, dim=0).contiguous()
            self.q_scale = torch.cat(q_s_ch, dim=0).contiguous()
            self.k_scale = torch.cat(k_s_ch, dim=0).contiguous()
            self.v_scale = torch.cat(v_s_ch, dim=0).contiguous()
        else:
            q_bnsd = q_fp32_tensor if self.q_input_layout == "BNSD" else q_fp32_tensor.permute(0, 2, 1, 3)
            k_bnsd = k_fp32_tensor if self.q_input_layout == "BNSD" else k_fp32_tensor.permute(0, 2, 1, 3)
            v_bnsd = v_fp32_tensor if self.q_input_layout == "BNSD" else v_fp32_tensor.permute(0, 2, 1, 3)

            q_packed_t, q_e8m0 = mxfp4_quantize_pack_last(q_bnsd, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
            k_packed_t, k_e8m0 = mxfp4_quantize_pack_last(k_bnsd, quant_axis=-1, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
            v_packed_t, v_e8m0 = mxfp4_quantize_pack_last(v_bnsd, quant_axis=-2, quant_mode=self.quant_mode, log2_cx=self.log2_cx)
            self.q_scale = _e8m0_qk_to_v3(q_e8m0)
            self.k_scale = _e8m0_qk_to_v3(k_e8m0)
            self.v_scale = _e8m0_v_to_v3(v_e8m0)

            if self.q_input_layout == "BSND":
                self.q_packed = q_packed_t.permute(0, 2, 1, 3).contiguous()
                self.k_packed = k_packed_t.permute(0, 2, 1, 3).contiguous()
                self.v_packed = v_packed_t.permute(0, 2, 1, 3).contiguous()
                self.q_scale = self.q_scale.permute(0, 2, 1, 3, 4).contiguous()
                self.k_scale = self.k_scale.permute(0, 2, 1, 3, 4).contiguous()
                self.v_scale = self.v_scale.permute(0, 2, 1, 3, 4).contiguous()
            else:
                self.q_packed = q_packed_t
                self.k_packed = k_packed_t
                self.v_packed = v_packed_t

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            testObj = TestBlockSparseAttentionMxfp4GenBmGt()
            aux_attrs = testObj.AuxAttrs(
                self.num_heads, self.kv_heads, self.headDim,
                self.inner_precise, self.scale_value,
                self.q_input_layout, self.kv_input_layout,
                self.attention_out_dtype_str,
                self.quant_mode, self.dst_type_max)

            attention_inputs = testObj.AttentionInputs(
                self.query_fp32, self.key_fp32, self.value_fp32,
                self.block_sparse_mask_np, self.block_sparse_idx, self.block_sparse_count,
                self.block_shape, self.q_seqlen_list, self.kv_seqlen_list,
                aux_attrs)

            if self.task_result.is_benchmark_task:
                result = testObj.compute_output_benchmark(attention_inputs)
            else:
                result = testObj.compute_output(attention_inputs)
            return result.to(torch.float32)

        # elif self.device == "pyaclnn":
        elif self.device in ("npu", "pyaclnn"):
            # NPU 调用 block_sparse_attention_v3
            import torch_npu
            with torch.no_grad():
                npu_out, _ = block_sparse_attention_v3(
                    self.q_packed.contiguous().npu(),
                    self.k_packed.contiguous().npu(),
                    self.v_packed.contiguous().npu(),
                    self.block_sparse_mask_tensor.contiguous().npu(),
                    list(self.block_shape),
                    self.q_seqlen_list, self.kv_seqlen_list,
                    num_heads=self.num_heads, num_key_value_heads=self.kv_heads,
                    q_input_layout=self.q_input_layout, kv_input_layout=self.kv_input_layout,
                    mask_type=0, scale_value=self.scale_value,
                    inner_precise=self.inner_precise, block_size=0,
                    pre_tokens=2147483647, next_tokens=2147483647,
                    softmax_lse_flag=False,
                    quant_mode=self.quant_mode, dst_type_max=self.dst_type_max,
                    q_dequant_scale=self.q_scale.npu(),
                    k_dequant_scale=self.k_scale.npu(),
                    v_dequant_scale=self.v_scale.npu(),
                    attention_out_dtype=self.attention_out_torch_dtype,
                )
            torch.npu.synchronize()
            return npu_out.cpu().to(torch.float32)
