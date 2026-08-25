#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
"""npu_block_sparse_attention mxfp8 (e4m3fn + e8m0) 正向单测 + CPU golden.

0817 几何: kvBaseTile=128, QK 半面 64x128, 基块 128x128, G=8, Dpv=12。
Q/K/V/P = e4m3fn, scale = e8m0, MX 组 32, 仅 OCP (quant_mode=4)。
attentionOut 形状不翻倍(与 mxfp4 packed uint8 不同)。
NPU I/O layout: BNSD / BSND / uniform TND(golden 仍 BNSD, 仅 permute+flatten)。
TND varlen(s1_list/s2_list) / act_s pad / gap_sparse / ocp_scale_stride_trap 已接线。

合入前回归（OCP 1–3，约 27 例，不含 kvs4096）: 见常量 ``MXFP8_OCP_REG_K``，或::

    ./run.sh --reg


golden: Q/K/V mxfp8 qdq(fp32); 分块 flash + online softmax;
P 走 kernel 同构 snap_local(m_snap=(K-2)*ln2) 再 round 到 e4m3。
判据: qfa check_result (np.isclose rtol=1/128 atol=1e-4, 满足率=100% 且 maxRE<10)。
"""
import math
import os
import unittest
import numpy as np
import torch
import torch.nn.functional as F

_FORCE_CPU = os.environ.get("BSA_CPU_ONLY", "0") == "1"

try:
    if _FORCE_CPU:
        raise ImportError("BSA_CPU_ONLY=1")
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

try:
    if torch_npu is None:
        raise ImportError("torch_npu unavailable")
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion
except Exception:
    class TestCase(unittest.TestCase):
        pass

    def run_tests():
        unittest.main()

    class _NoopDecorator:
        def __init__(self, *a, **k):
            pass

        def __call__(self, fn):
            return fn

    SkipIfNotGteCANNVersion = _NoopDecorator

try:
    from cann_ops_transformer.ops import block_sparse_attention_v3
    _HAS_BSA_V3 = True
    _BSA_V3_IMPORT_ERROR = None
except Exception as _e:
    block_sparse_attention_v3 = None
    _HAS_BSA_V3 = False
    _BSA_V3_IMPORT_ERROR = _e

from compare_utils import check_result_printf_res

RAND_SEED = 21
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)

_CLI_OVERRIDES = {}
PRINTF_CHECK_RES = False
_E8M0_QK_FLIP_PAIR = False
E8M0_BIAS = 127
QUANT_MODE_MXFP8_OCP = 4
FP8_E4M3_EMAX = 8
FP8_E4M3_MAX = 448.0
MX_GROUP = 32

# 合入前必跑（OCP 1–3 回归）。pytest -k 须整段当一个参数、不要换行。
# 覆盖: TND varlen / act_s pad / scale-stride-trap / gap_sparse / BSND·TND D=64。
# 排除: kvs4096/4097 长 golden。新增回归例请带这些子串之一，或 (bsnd_mha|tnd_mha|tnd_gqa)+d64。
MXFP8_OCP_REG_K = (
    "tnd_varlen or actq or scale_stride_trap or gap_sparse or "
    "(d64 and (bsnd_mha or tnd_mha or tnd_gqa) and not kvs4096 and not kvs4097)"
)


def rand_range(shape, data_range=(-10, 10), dtype=torch.float32, device=None):
    return data_range[0] + (data_range[1] - data_range[0]) * torch.rand(
        shape, dtype=dtype, device=device)


def randn_x(shape, x=1.0, dtype=torch.float32, device=None):
    return rand_range(shape, (-x, x), dtype=dtype, device=device)


def make_ocp_cross_head_scale_trap(shape_bnsd, dtype=torch.float32):
    """强制相邻头 e8m0 全局不等, 专打 TND/BSND Q/K-scale 行步长漏乘 N。

    MXFP8 OCP: scale_exp = floor(log2(max_abs)) - 8, 且 max/scale 须 < 448 以免 e4m3 触顶。
      偶头: |x|∈(2.0, 3.4) → binade [2,4) → exp=-7 (e8m0=120), q<=3.4*128=435
      奇头: |x|∈(4.0, 6.8) → binade [4,8) → exp=-6 (e8m0=121), q<=6.8*64=435
    不可照搬 mxfp4 的 (0.60,0.90)/(1.20,1.80): 那是 emax=2 的相邻 binade,
    在 emax=8 下落到 2^-9/2^-8, softmax 后 O≈0, atol=1e-4 把 gold=0 vs npu~3e-4 判失败。
    普通 U[-1,1] 下各头 scale 近乎全等, 读错也看不见, 故需本陷阱。"""
    B, N, S, D = shape_bnsd
    assert N >= 2, f"scale-stride trap 需要 N>=2, 当前 N={N}"
    out = torch.empty(B, N, S, D, dtype=dtype)
    for n in range(N):
        lo, hi = (2.0, 3.4) if (n % 2) == 0 else (4.0, 6.8)
        chunk = lo + (hi - lo) * torch.rand(B, S, D, dtype=dtype)
        sign = torch.where(torch.rand(B, S, D) > 0.5, 1.0, -1.0).to(dtype)
        out[:, n, :, :] = chunk * sign
    return out
    B, N, S, D = shape_bnsd
    assert N >= 2, f"scale-stride trap 需要 N>=2, 当前 N={N}"
    out = torch.empty(B, N, S, D, dtype=dtype)
    for n in range(N):
        lo, hi = (0.60, 0.90) if (n % 2) == 0 else (1.20, 1.80)
        chunk = lo + (hi - lo) * torch.rand(B, S, D, dtype=dtype)
        sign = torch.where(torch.rand(B, S, D) > 0.5, 1.0, -1.0).to(dtype)
        out[:, n, :, :] = chunk * sign
    return out


def _make_qkv(b, n1, n2, s1, s2, qk_d, v_d, data_x=10.0, data_kind="default"):
    """统一造 Q/K/V (BNSD)。data_kind='ocp_scale_stride_trap' 时走跨头 e8m0 陷阱。"""
    if data_kind == "ocp_scale_stride_trap":
        query = make_ocp_cross_head_scale_trap((b, n1, s1, qk_d)).to(torch.float32)
        key = make_ocp_cross_head_scale_trap((b, n2, s2, qk_d)).to(torch.float32)
        value = make_ocp_cross_head_scale_trap((b, n2, s2, v_d)).to(torch.float32)
    else:
        query = randn_x((b, n1, s1, qk_d), data_x).to(torch.float32)
        key = randn_x((b, n2, s2, qk_d), data_x).to(torch.float32)
        value = randn_x((b, n2, s2, v_d), data_x).to(torch.float32)
    return query, key, value


def _resolve_s_dtype(s_dtype):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s_dtype]


def _resolve_out_dtype(out_dtype):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[out_dtype]


def mxfp8_quantize_native(tensor, quant_axis, group_size=MX_GROUP):
    """沿 quant_axis 做 MXFP8 OCP 量化。
    返回 (e4m3 张量同原 shape, e8m0 uint8 沿量化轴 ceil(dim/32), fp32 qdq)。"""
    nd = tensor.dim()
    qax = quant_axis % nd
    t = tensor.movedim(qax, -1).contiguous().float()
    *lead, d = t.shape
    ng = math.ceil(d / group_size)
    pad = ng * group_size - d
    x = F.pad(t, (0, pad)) if pad else t
    g = x.reshape(*lead, ng, group_size)
    all_zero = torch.all(g == 0, dim=-1)
    max_vals = torch.max(torch.abs(g), dim=-1)[0].clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - FP8_E4M3_EMAX
    two = torch.tensor(2.0, dtype=torch.float32, device=t.device)
    scale = torch.where(all_zero, torch.ones_like(shared_exp), torch.pow(two, shared_exp))
    q = (g / scale.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    q_out = q.reshape(*lead, ng * group_size)[..., :d].movedim(-1, qax).contiguous()
    dq = (q.float() * scale.unsqueeze(-1)).reshape(*lead, ng * group_size)[..., :d]
    dq = dq.movedim(-1, qax).contiguous()
    e8m0 = (shared_exp.round().long() + E8M0_BIAS).clamp(0, 254).to(torch.uint8)
    e8m0 = e8m0.movedim(-1, qax).contiguous()
    return q_out, e8m0, dq


def mxfp8_qdq(t, axis, group_size=MX_GROUP):
    _, _, dq = mxfp8_quantize_native(t, quant_axis=axis, group_size=group_size)
    return dq


def _to_e8m0_uint8(exp_int):
    biased = exp_int.to(torch.int64) + E8M0_BIAS
    return torch.clamp(biased, min=0, max=254).to(torch.uint8)


def _e8m0_qk_to_v3(e8m0_uint8):
    *lead, nd = e8m0_uint8.shape
    paired = e8m0_uint8.reshape(*lead, nd // 2, 2)
    if _E8M0_QK_FLIP_PAIR:
        paired = paired.flip(-1)
    return paired.contiguous().view(torch.float8_e8m0fnu)


def _e8m0_v_to_v3(e8m0_uint8):
    B, N2, nb, D = e8m0_uint8.shape
    if nb % 2 != 0:
        pad_block = torch.full((B, N2, 1, D), E8M0_BIAS, dtype=torch.uint8)
        e8m0_uint8 = torch.cat([e8m0_uint8, pad_block], dim=2)
        nb += 1
    return e8m0_uint8.reshape(B, N2, nb // 2, 2, D).permute(0, 1, 2, 4, 3).contiguous().view(
        torch.float8_e8m0fnu)


def _to_npu_layout(q, k, v, q_scale, k_scale, v_scale, cpu_out, input_layout):
    """Golden 保持 BNSD; 仅 NPU I/O 重排到 BSND / uniform TND(同 mxfp4 QFA 哲学)。
    Q/K/V: [B,N,S,D] --permute(0,2,1,3)--> BSND; TND 再 flatten(B,S)->T。
    e8m0 5D: Q/K [B,N,S,D//64,2], V [B,N2,nb//2,D,2] 同维 permute; mask 不重排。"""
    if input_layout == "BNSD":
        return q, k, v, q_scale, k_scale, v_scale, cpu_out
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    q_scale = q_scale.permute(0, 2, 1, 3, 4)
    k_scale = k_scale.permute(0, 2, 1, 3, 4)
    v_scale = v_scale.permute(0, 2, 1, 3, 4)
    cpu_out = cpu_out.permute(0, 2, 1, 3)
    if input_layout == "TND":
        q = q.flatten(0, 1)
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)
        q_scale = q_scale.flatten(0, 1)
        k_scale = k_scale.flatten(0, 1)
        v_scale = v_scale.flatten(0, 1)
        cpu_out = cpu_out.flatten(0, 1)
    return (q.contiguous(), k.contiguous(), v.contiguous(),
            q_scale.contiguous(), k_scale.contiguous(), v_scale.contiguous(),
            cpu_out.contiguous())


def _blockwise_snap_local_quantize_p_eff(S, m_running, seqk_dim=0, mx_block_size=32):
    """与 mxfp4 OCP 同构: m_snap=(K_block-2)*ln2, corr=2^(K_diff-2)。
    P_local 经 bf16 round-trip 后 round 到 e4m3（对齐 kernel Exp 后 fp16→bf16→fp16，再 float→e4m3 RINT）。"""
    neg_inf = float("-inf")
    device = S.device
    src_dtype = S.dtype
    ndim = S.dim()
    skp = seqk_dim if seqk_dim >= 0 else ndim + seqk_dim
    seqk_len = S.shape[skp]
    ln2 = math.log(2.0)
    pad = (mx_block_size - seqk_len % mx_block_size) % mx_block_size
    if pad > 0:
        pshape = list(S.shape)
        pshape[skp] = pad
        s_padded = torch.cat([S, torch.full(pshape, neg_inf, dtype=src_dtype, device=device)], dim=skp)
    else:
        s_padded = S
    s_moved = s_padded.movedim(skp, -1)
    other = s_moved.shape[:-1]
    n_blk = s_moved.shape[-1] // mx_block_size
    s_r = s_moved.reshape(*other, n_blk, mx_block_size)
    m_block_raw = s_r.max(dim=-1).values
    m_block_raw_safe = torch.where(
        torch.isinf(m_block_raw) & (m_block_raw < 0), torch.zeros_like(m_block_raw), m_block_raw)
    inv_ln2 = torch.tensor(1.0 / ln2, dtype=torch.float16, device=device)
    ln2_fp16 = torch.tensor(ln2, dtype=torch.float16, device=device)
    k_block = torch.floor(m_block_raw_safe * inv_ln2)
    m_block_snap = (k_block - 2) * ln2_fp16
    p_local = torch.exp(s_r - m_block_snap.unsqueeze(-1))
    p_local_bf = p_local.to(torch.bfloat16).to(p_local.dtype)
    p_q = p_local_bf.float().to(torch.float8_e4m3fn).float().to(src_dtype)
    k_ij = k_block.max(dim=-1).values
    k_ij_fp32 = k_ij.float()
    m_new = k_ij_fp32 if m_running is None else torch.maximum(m_running, k_ij_fp32)
    k_new_s = m_new.to(src_dtype)
    k_diff = k_block - k_new_s.unsqueeze(-1)
    corr = torch.exp2(k_diff - 2.0)
    corr_broad = corr.repeat_interleave(mx_block_size, dim=-1)[..., :seqk_len].movedim(-1, skp)
    p_q_moved = p_q.reshape(*other, n_blk * mx_block_size)[..., :seqk_len].movedim(-1, skp)
    return p_q_moved * corr_broad, m_new


def _build_block_sparse_mask(b, n1, ceil_q, ceil_kv, kind="dense"):
    if kind == "dense":
        return torch.ones(b, n1, ceil_q, ceil_kv, dtype=torch.int8)
    if kind == "sparse":
        m = torch.zeros(b, n1, ceil_q, ceil_kv, dtype=torch.int8)
        for i in range(ceil_q):
            n_attend = torch.randint(1, ceil_kv + 1, (1,)).item()
            idx = torch.randperm(ceil_kv)[:n_attend]
            m[:, :, i, idx] = 1
        return m
    if kind == "gap_sparse":
        m = torch.zeros(b, n1, ceil_q, ceil_kv, dtype=torch.int8)
        m[:, :, :, 0] = 1
        m[:, :, :, ceil_kv - 1] = 1
        return m
    raise ValueError(f"unsupported mask kind: {kind}")


def _build_block_sparse_mask_varlen(b, n1, act_q, act_kv, block_shape, kind="dense"):
    """变长 TND 的 block_sparse_mask [B,n1,ceil_q_max,ceil_kv_max] int8。
    batch b 实际 token 数 act_q[b]/act_kv[b]; 超出 ceil_q_b/ceil_kv_b 的块置 0。
    dense/sparse 仅在 [0..ceil_q_b, 0..ceil_kv_b] 内有效。"""
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    max_s1 = max(act_q)
    max_s2 = max(act_kv)
    ceil_q = (max_s1 + block_x - 1) // block_x
    ceil_kv = (max_s2 + block_y - 1) // block_y
    m = torch.zeros(b, n1, ceil_q, ceil_kv, dtype=torch.int8)
    for bi in range(b):
        cq_b = (act_q[bi] + block_x - 1) // block_x
        ck_b = (act_kv[bi] + block_y - 1) // block_y
        if kind == "dense":
            m[bi, :, :cq_b, :ck_b] = 1
        elif kind == "sparse":
            for i in range(cq_b):
                n_attend = torch.randint(1, ck_b + 1, (1,)).item()
                idx = torch.randperm(ck_b)[:n_attend]
                m[bi, :, i, idx] = 1
        else:
            raise ValueError(f"unsupported mask kind: {kind}")
    return m


def _quantize_qkv_to_v3(query, key, value):
    q_fp8, q_e8m0, _ = mxfp8_quantize_native(query, quant_axis=-1)
    k_fp8, k_e8m0, _ = mxfp8_quantize_native(key, quant_axis=-1)
    v_fp8, v_e8m0, _ = mxfp8_quantize_native(value, quant_axis=-2)
    return (q_fp8, k_fp8, v_fp8,
            _e8m0_qk_to_v3(q_e8m0), _e8m0_qk_to_v3(k_e8m0), _e8m0_v_to_v3(v_e8m0))


def cpu_block_sparse_attention_bnsd_mxfp8(query, key, value, block_sparse_mask,
                                          block_shape, scale_value, num_kv_heads,
                                          p_quant="snap_local", s_dtype="fp16",
                                          out_dtype="bfloat16"):
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    bsz, n_q, s1, d = q.shape
    n2 = k.shape[1]
    s2 = k.shape[2]
    group_size = n_q // n2
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (s1 + block_x - 1) // block_x
    ceil_kv = (s2 + block_y - 1) // block_y
    s_torch_dtype = _resolve_s_dtype(s_dtype)
    q_q = mxfp8_qdq(q, axis=-1)
    k_q = mxfp8_qdq(k, axis=-1)
    v_q = mxfp8_qdq(v, axis=2)
    out = torch.zeros((bsz, n_q, s1, d), dtype=torch.float32)
    scale_s = torch.as_tensor(scale_value, dtype=s_torch_dtype)
    for bi in range(bsz):
        for ni in range(n_q):
            n2i = ni // group_size
            for qb in range(ceil_q):
                q_lo = qb * block_x
                q_hi = min(q_lo + block_x, s1)
                sel_j = [j for j in range(ceil_kv) if block_sparse_mask[bi, ni, qb, j] != 0]
                if not sel_j:
                    continue
                chunks = [sel_j[i:i + 2] for i in range(0, len(sel_j), 2)]
                q_blk = q_q[bi, ni, q_lo:q_hi]
                nq = q_hi - q_lo
                m_run = None
                l_run = torch.zeros(nq, dtype=torch.float32)
                acc = torch.zeros((nq, d), dtype=torch.float32)
                for chunk in chunks:
                    positions = []
                    for j in chunk:
                        st = j * block_y
                        ed = min(st + block_y, s2)
                        positions.extend(range(st, ed))
                    if not positions:
                        continue
                    pos = torch.as_tensor(positions, dtype=torch.long)
                    k_blk = k_q[bi, n2i][pos]
                    v_blk = v_q[bi, n2i][pos]
                    s_fp32 = (k_blk @ q_blk.t()).to(torch.float32)
                    s_s = s_fp32.to(s_torch_dtype) * scale_s
                    if p_quant == "snap_local":
                        p_eff_s, m_new = _blockwise_snap_local_quantize_p_eff(
                            s_s, m_running=m_run, seqk_dim=0)
                        p_eff_s = torch.nan_to_num(p_eff_s, nan=0.0)
                        p_fp32 = p_eff_s.float()
                        alpha = (torch.zeros(nq, dtype=torch.float32) if m_run is None
                                 else torch.exp2(m_run.float() - m_new.float()))
                    else:
                        m_s = s_s.max(dim=0).values
                        m_safe_s = torch.where(torch.isinf(m_s) & (m_s < 0), torch.zeros_like(m_s), m_s)
                        p_ij = torch.exp(s_s - m_safe_s.unsqueeze(0))
                        p_fp32 = torch.nan_to_num(p_ij, nan=0.0).float()
                        m_new = m_safe_s.float()
                        alpha = (torch.zeros(nq, dtype=torch.float32) if m_run is None
                                 else torch.exp(m_run - m_new))
                    reduce_sum = p_fp32.sum(dim=0)
                    l_run = alpha * l_run + reduce_sum
                    acc = alpha.unsqueeze(-1) * acc + (p_fp32.t() @ v_blk)
                    m_run = m_new
                l_safe = torch.where(l_run > 0, l_run, torch.ones_like(l_run))
                out[bi, ni, q_lo:q_hi] = acc / l_safe.unsqueeze(-1)
    return out.to(_resolve_out_dtype(out_dtype))


def check_result(expect, result, data_type, pct_thd=0.0):
    real_data = result.cpu().numpy().flatten()
    data_compe = expect.cpu().numpy().flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        return 100.0, "Pass"
    if real_data.size != data_compe.size:
        return 0.0, "Failed"
    rtol, atol, max_diff_hd = 0.0078125, 0.0001, 10.0
    diff_result = np.isclose(real_data.astype(np.float32), data_compe.astype(np.float32),
                             rtol=rtol, atol=atol, equal_nan=True)
    err_idx = np.where(diff_result != np.array((True,)))[0]
    split_count = real_data.size
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0
    diff_abs = np.abs(data_compe - real_data)
    b = np.add(np.maximum(np.abs(real_data), np.abs(data_compe)), 10e-10)
    err_diff = (diff_abs / b)[err_idx]
    pct_thd = (1 - pct_thd) * 100.0
    result_str = "Pass" if fulfill_percent >= pct_thd else "Failed"
    if len(err_diff) > 0 and np.max(err_diff) >= max_diff_hd:
        result_str = "Failed"
    return fulfill_percent, result_str


def compare(golden, real, src_dtype):
    if golden is None or real is None:
        return 0.0, "Failed"
    golden_fp32 = golden.to(torch.float32)
    real_fp32 = real.cpu().to(torch.float32)
    if os.environ.get("BSA_MXFP8_HOST_DIAG", "0") == "1":
        g = golden_fp32.reshape(-1, golden_fp32.shape[-2], golden_fp32.shape[-1])
        r = real_fp32.reshape(-1, real_fp32.shape[-2], real_fp32.shape[-1])
        gs = g[0]
        rs = r[0]
        rtol, atol = 1.0 / 128.0, 1e-4
        close = torch.isclose(rs, gs, rtol=rtol, atol=atol)
        fail_rows = (~close).any(dim=-1).nonzero(as_tuple=False).flatten()
        print(f"[HOST-ERR] fail_rows={fail_rows[:16].tolist()} nfail_rows={int((~close).any(dim=-1).sum())}")
        for s in fail_rows[:8].tolist():
            nfail = int((~close[s]).sum())
            diff = (rs[s] - gs[s]).abs()
            rel = diff / gs[s].abs().clamp(min=1e-12)
            w = int(rel.argmax())
            fail_d = (~close[s]).nonzero(as_tuple=False).flatten()[:12].tolist()
            print(f"[HOST-ERR] s={s} nfailD={nfail} failD[:12]={fail_d} "
                  f"npu_d0={float(rs[s,0]):.6f} gold_d0={float(gs[s,0]):.6f} "
                  f"worstD={w} npu={float(rs[s,w]):.6f} gold={float(gs[s,w]):.6f} rel={float(rel[w]):.3f} "
                  f"rowsum_npu={float(rs[s].sum()):.4f} rowsum_gold={float(gs[s].sum()):.4f}")
    if os.environ.get("BSA_MXFP8_TRAP_DIAG", "0") == "1":
        rtol, atol = 1.0 / 128.0, 1e-4
        close = torch.isclose(real_fp32, golden_fp32, rtol=rtol, atol=atol, equal_nan=True)
        nfail = int((~close).sum())
        print(f"[TRAP-DIAG] shape_g={tuple(golden_fp32.shape)} shape_r={tuple(real_fp32.shape)} "
              f"nfail={nfail}/{close.numel()}")
        g = golden_fp32
        r = real_fp32
        ratio = (r / g.clamp(min=1e-12)).abs()
        rel = (r - g).abs() / g.abs().clamp(min=1e-12)
        fail = (~close)
        if g.dim() == 4:
            # BNSD [B,N,S,D] or BSND [B,S,N,D]
            print(f"[TRAP-DIAG] fail_by_dim1={fail.sum(dim=(0,2,3)).tolist()}")
            d_fail = fail.sum(dim=(0, 1, 2))
            print(f"[TRAP-DIAG] fail_by_D groups32="
                  f"{[int(d_fail[i:i+32].sum()) for i in range(0, d_fail.numel(), 32)]}")
            print(f"[TRAP-DIAG] fail_by_D={d_fail.tolist()}")
            # 若错读邻头 scale, npu/gold 应聚在 2 或 0.5
            fr = ratio[fail]
            if fr.numel() > 0:
                print(f"[TRAP-DIAG] fail |npu/gold| p50={float(fr.median()):.4f} "
                      f"mean={float(fr.mean()):.4f} "
                      f"near2={(fr - 2).abs().lt(0.15).float().mean().item()*100:.1f}% "
                      f"near05={(fr - 0.5).abs().lt(0.08).float().mean().item()*100:.1f}%")
            w = int(rel.argmax())
            idx = np.unravel_index(w, g.shape)
            print(f"[TRAP-DIAG] worst idx={idx} npu={float(r.reshape(-1)[w]):.6f} "
                  f"gold={float(g.reshape(-1)[w]):.6f} rel={float(rel.reshape(-1)[w]):.3f} "
                  f"ratio={float(ratio.reshape(-1)[w]):.4f}")
        elif g.dim() == 3:
            # TND [T,N,D]
            print(f"[TRAP-DIAG] fail_by_N={fail.sum(dim=(0,2)).tolist()}")
            d_fail = fail.sum(dim=(0, 1))
            print(f"[TRAP-DIAG] fail_by_D groups32="
                  f"{[int(d_fail[i:i+32].sum()) for i in range(0, d_fail.numel(), 32)]}")
            print(f"[TRAP-DIAG] fail_by_D={d_fail.tolist()}")
            fr = ratio[fail]
            if fr.numel() > 0:
                print(f"[TRAP-DIAG] fail |npu/gold| p50={float(fr.median()):.4f} "
                      f"mean={float(fr.mean()):.4f} "
                      f"near2={(fr - 2).abs().lt(0.15).float().mean().item()*100:.1f}% "
                      f"near05={(fr - 0.5).abs().lt(0.08).float().mean().item()*100:.1f}%")
            w = int(rel.argmax())
            idx = np.unravel_index(w, g.shape)
            print(f"[TRAP-DIAG] worst idx={idx} npu={float(r.reshape(-1)[w]):.6f} "
                  f"gold={float(g.reshape(-1)[w]):.6f} rel={float(rel.reshape(-1)[w]):.3f} "
                  f"ratio={float(ratio.reshape(-1)[w]):.4f}")
    if PRINTF_CHECK_RES:
        return check_result_printf_res(golden_fp32, real_fp32, src_dtype)
    return check_result(golden_fp32, real_fp32, src_dtype)


def _call_bsa_v3(q, k, v, mask, block_shape, act_q, act_kv, n1, n2,
                 input_layout, scale, inner_precise, quant_mode,
                 q_scale, k_scale, v_scale, out_dtype, device_id):
    torch_npu.npu.set_device(device_id)
    print("npu calc start")
    npu_out, _ = block_sparse_attention_v3(
        q.contiguous().npu(), k.contiguous().npu(), v.contiguous().npu(),
        mask.contiguous().npu(), list(block_shape), act_q, act_kv,
        num_heads=n1, num_key_value_heads=n2,
        q_input_layout=input_layout, kv_input_layout=input_layout,
        mask_type=0, scale_value=scale, inner_precise=inner_precise, block_size=0,
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=False, quant_mode=quant_mode, dst_type_max=0.0,
        q_dequant_scale=q_scale.npu(), k_dequant_scale=k_scale.npu(),
        v_dequant_scale=v_scale.npu(),
        attention_out_dtype=_resolve_out_dtype(out_dtype),
    )
    torch_npu.npu.synchronize()
    return npu_out


def run_bsa_mxfp8_eager(b, n1, n2, s1=None, s2=None, g=1, qk_d=128, v_d=None,
                        input_layout="BNSD", out_dtype="bfloat16",
                        block_shape=(128, 128), mask_kind="dense",
                        inner_precise=4, quant_mode=QUANT_MODE_MXFP8_OCP,
                        p_quant="snap_local", s_dtype="fp16",
                        dump=False, device_id=0, data_x=10.0,
                        s1_list=None, s2_list=None, act_s1=None, act_s2=None,
                        data_kind="default"):
    """fp32 造数 -> mxfp8 量化 -> CPU golden -> NPU -> check_result.
    s1/s2(uniform) 或 s1_list/s2_list(TND varlen) 二选一。
    act_s1/act_s2(BNSD/BSND padded): actualSeq < 张量 s1/s2, 只比有效区。
    data_kind: 'default' | 'ocp_scale_stride_trap'(跨头 e8m0 不等, 打 BSND/TND scale 行步长)。"""
    g = n1 // n2
    assert input_layout in ("BNSD", "BSND", "TND"), f"支持 BNSD/BSND/TND, 当前 {input_layout}"
    assert n1 == n2 * g
    assert block_shape[0] in (64, 128)
    assert block_shape[1] % 128 == 0, f"0817 blockShapeY 须为 128 倍数, 当前 {block_shape[1]}"
    assert qk_d in (64, 128)
    assert quant_mode == QUANT_MODE_MXFP8_OCP
    assert out_dtype in ("bfloat16", "float16"), f"out_dtype 仅 bfloat16/float16, 当前 {out_dtype}"
    assert mask_kind in ("dense", "sparse", "gap_sparse"), f"mask_kind={mask_kind}"
    assert data_kind in ("default", "ocp_scale_stride_trap"), f"data_kind={data_kind}"
    if data_kind == "ocp_scale_stride_trap":
        assert n1 >= 2 and n2 >= 2, (
            f"ocp_scale_stride_trap 需要 n1>=2 且 n2>=2, 当前 n1={n1} n2={n2}")
        assert input_layout in ("BNSD", "BSND", "TND"), (
            f"ocp_scale_stride_trap 仅对 BNSD/BSND/TND, 当前 {input_layout}")
    if v_d is None:
        v_d = qk_d
    scale = 1.0 / math.sqrt(qk_d)
    src_dtype = out_dtype
    print("MXFP8 OCP QUANT")
    if data_kind == "ocp_scale_stride_trap":
        print(f"[data_kind] ocp_scale_stride_trap: 跨头强制 OCP e8m0 偶头=120/奇头=121 "
              f"(layout={input_layout} n1={n1} n2={n2})")

    # ===== TND varlen: per-batch 造数/golden/打包 + concat T =====
    if input_layout == "TND" and s1_list is not None:
        assert s2_list is not None, "TND varlen 需同时给 s1_list/s2_list"
        block_x, block_y = int(block_shape[0]), int(block_shape[1])
        act_q = [int(x) for x in s1_list]
        act_kv = [int(x) for x in s2_list]
        assert len(act_q) == b and len(act_kv) == b, "s1_list/s2_list 长度须 == b"
        mask = _build_block_sparse_mask_varlen(b, n1, act_q, act_kv, block_shape, mask_kind)
        q_list, k_list, v_list = [], [], []
        for bi in range(b):
            q_b, k_b, v_b = _make_qkv(1, n1, n2, act_q[bi], act_kv[bi], qk_d, v_d,
                                      data_x=data_x, data_kind=data_kind)
            q_list.append(q_b)
            k_list.append(k_b)
            v_list.append(v_b)
        print("cpu calc start")
        cpu_chunks = []
        for bi in range(b):
            cq_b = (act_q[bi] + block_x - 1) // block_x
            ck_b = (act_kv[bi] + block_y - 1) // block_y
            mask_b = mask[bi:bi + 1, :, :cq_b, :ck_b]
            out_b = cpu_block_sparse_attention_bnsd_mxfp8(
                q_list[bi], k_list[bi], v_list[bi], mask_b, block_shape, scale, n2,
                p_quant=p_quant, s_dtype=s_dtype, out_dtype=out_dtype)
            cpu_chunks.append(out_b.permute(0, 2, 1, 3).flatten(0, 1))
        cpu_out = torch.cat(cpu_chunks, dim=0).contiguous()
        print("golden calc end")
        if _FORCE_CPU or not _HAS_BSA_V3:
            print(f"cpu_out shape={tuple(cpu_out.shape)}")
            return -1.0, "SKIP"
        q_ch, k_ch, v_ch, qs_ch, ks_ch, vs_ch = [], [], [], [], [], []
        for bi in range(b):
            q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale = _quantize_qkv_to_v3(
                q_list[bi], k_list[bi], v_list[bi])
            q_ch.append(q_fp8.permute(0, 2, 1, 3).flatten(0, 1))
            k_ch.append(k_fp8.permute(0, 2, 1, 3).flatten(0, 1))
            v_ch.append(v_fp8.permute(0, 2, 1, 3).flatten(0, 1))
            qs_ch.append(q_scale.permute(0, 2, 1, 3, 4).flatten(0, 1))
            ks_ch.append(k_scale.permute(0, 2, 1, 3, 4).flatten(0, 1))
            vs_ch.append(v_scale.permute(0, 2, 1, 3, 4).flatten(0, 1))
        npu_out = _call_bsa_v3(
            torch.cat(q_ch, 0).contiguous(), torch.cat(k_ch, 0).contiguous(),
            torch.cat(v_ch, 0).contiguous(), mask, block_shape, act_q, act_kv,
            n1, n2, "TND", scale, inner_precise, quant_mode,
            torch.cat(qs_ch, 0).contiguous(), torch.cat(ks_ch, 0).contiguous(),
            torch.cat(vs_ch, 0).contiguous(), out_dtype, device_id)
        return compare(cpu_out, npu_out, src_dtype)

    # ===== padded BNSD/BSND: actualSeq < 张量 s1/s2, 只比有效区 =====
    if input_layout in ("BNSD", "BSND") and act_s1 is not None:
        assert s1 is not None and s2 is not None and act_s2 is not None
        act_q = [int(x) for x in ([act_s1] * b if isinstance(act_s1, int) else act_s1)]
        act_kv = [int(x) for x in ([act_s2] * b if isinstance(act_s2, int) else act_s2)]
        assert len(act_q) == b and len(act_kv) == b, "act_s1/act_s2 长度须 == b"
        assert all(a <= s1 for a in act_q) and all(a <= s2 for a in act_kv), (
            "act_s1/act_s2 每 batch 须 <= s1/s2(padded 张量形状)")
        block_x, block_y = int(block_shape[0]), int(block_shape[1])
        mask = _build_block_sparse_mask_varlen(b, n1, act_q, act_kv, block_shape, mask_kind)
        ceil_q_pad = (s1 + block_x - 1) // block_x
        ceil_kv_pad = (s2 + block_y - 1) // block_y
        mask = F.pad(mask, (0, ceil_kv_pad - mask.shape[3], 0, ceil_q_pad - mask.shape[2]), value=0)
        q_list, k_list, v_list = [], [], []
        for bi in range(b):
            q_b, k_b, v_b = _make_qkv(1, n1, n2, act_q[bi], act_kv[bi], qk_d, v_d,
                                      data_x=data_x, data_kind="default")
            q_list.append(q_b)
            k_list.append(k_b)
            v_list.append(v_b)
        print("cpu calc start")
        cpu_chunks = []
        for bi in range(b):
            cq_b = (act_q[bi] + block_x - 1) // block_x
            ck_b = (act_kv[bi] + block_y - 1) // block_y
            mask_b = mask[bi:bi + 1, :, :cq_b, :ck_b]
            out_b = cpu_block_sparse_attention_bnsd_mxfp8(
                q_list[bi], k_list[bi], v_list[bi], mask_b, block_shape, scale, n2,
                p_quant=p_quant, s_dtype=s_dtype, out_dtype=out_dtype)
            if input_layout == "BSND":
                out_b = out_b.permute(0, 2, 1, 3).contiguous()
            cpu_chunks.append(out_b)
        print("golden calc end")
        if _FORCE_CPU or not _HAS_BSA_V3:
            print(f"cpu_out(valid) per-batch {[tuple(c.shape) for c in cpu_chunks]}")
            return -1.0, "SKIP"
        q_pad = torch.cat([F.pad(q_list[bi], (0, 0, 0, s1 - q_list[bi].shape[2]))
                           for bi in range(b)], 0).contiguous()
        k_pad = torch.cat([F.pad(k_list[bi], (0, 0, 0, s2 - k_list[bi].shape[2]))
                           for bi in range(b)], 0).contiguous()
        v_pad = torch.cat([F.pad(v_list[bi], (0, 0, 0, s2 - v_list[bi].shape[2]))
                           for bi in range(b)], 0).contiguous()
        q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale = _quantize_qkv_to_v3(q_pad, k_pad, v_pad)
        dummy_out = torch.zeros(b, n1, s1, qk_d)
        q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, _ = _to_npu_layout(
            q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, dummy_out, input_layout)
        npu_out = _call_bsa_v3(
            q_fp8, k_fp8, v_fp8, mask, block_shape, act_q, act_kv,
            n1, n2, input_layout, scale, inner_precise, quant_mode,
            q_scale, k_scale, v_scale, out_dtype, device_id)
        g_valid, n_valid = [], []
        for bi in range(b):
            g_valid.append(cpu_chunks[bi].reshape(-1))
            if input_layout == "BSND":
                n_valid.append(npu_out[bi, :act_q[bi], :, :].reshape(-1).cpu())
            else:
                n_valid.append(npu_out[bi, :, :act_q[bi], :].reshape(-1).cpu())
        return compare(torch.cat(g_valid), torch.cat(n_valid), src_dtype)

    # ===== uniform BNSD / BSND / TND =====
    assert s1 is not None and s2 is not None, "uniform 路径需 s1/s2(变长 TND 用 s1_list/s2_list)"
    query, key, value = _make_qkv(b, n1, n2, s1, s2, qk_d, v_d,
                                  data_x=data_x, data_kind=data_kind)
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (s1 + block_x - 1) // block_x
    ceil_kv = (s2 + block_y - 1) // block_y
    mask = _build_block_sparse_mask(b, n1, ceil_q, ceil_kv, mask_kind)
    print("cpu calc start")
    cpu_out = cpu_block_sparse_attention_bnsd_mxfp8(
        query, key, value, mask, block_shape, scale, n2,
        p_quant=p_quant, s_dtype=s_dtype, out_dtype=out_dtype)
    print("golden calc end")
    if _FORCE_CPU or not _HAS_BSA_V3:
        print(f"cpu_out shape={tuple(cpu_out.shape)}")
        return -1.0, "SKIP"
    q_fp8, q_e8m0, q_dq = mxfp8_quantize_native(query, quant_axis=-1)
    k_fp8, k_e8m0, k_dq = mxfp8_quantize_native(key, quant_axis=-1)
    v_fp8, v_e8m0, v_dq = mxfp8_quantize_native(value, quant_axis=-2)
    if os.environ.get("BSA_MXFP8_TRAP_DIAG", "0") == "1":
        for name, e in (("Q", q_e8m0), ("K", k_e8m0)):
            for ni in range(e.shape[1]):
                uniq = torch.unique(e[0, ni]).tolist()
                print(f"[TRAP-E8M0] {name} head{ni} unique={uniq[:8]}{'...' if len(uniq)>8 else ''}")
    s_host = None
    if os.environ.get("BSA_MXFP8_HOST_DIAG", "0") == "1" and input_layout == "BNSD":
        s_host = (k_dq[0, 0] @ q_dq[0, 0].t()).to(torch.float16).float()
        print("[HOST-S] kvs0 qs0/57/58/63/64/114/127:",
              [float(s_host[0, i]) for i in (0, 57, 58, 63, 64, 114, 127)])
        print("[HOST-S] qs57 k0-7:", [float(s_host[i, 57]) for i in range(min(8, s2))])
        print("[HOST-S] qs57 max/argmax", float(s_host[:s2, 57].max()), int(s_host[:s2, 57].argmax()),
              "qs114 max/argmax", float(s_host[:s2, 114].max()), int(s_host[:s2, 114].argmax()))
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert q_fp8.shape[-1] == qk_d, "mxfp8 输出 last dim 不翻倍"
    q_scale = _e8m0_qk_to_v3(q_e8m0)
    k_scale = _e8m0_qk_to_v3(k_e8m0)
    v_scale = _e8m0_v_to_v3(v_e8m0)
    q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, cpu_out = _to_npu_layout(
        q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, cpu_out, input_layout)
    npu_out = _call_bsa_v3(
        q_fp8, k_fp8, v_fp8, mask, block_shape, [s1] * b, [s2] * b,
        n1, n2, input_layout, scale, inner_precise, quant_mode,
        q_scale, k_scale, v_scale, out_dtype, device_id)
    if os.environ.get("BSA_MXFP8_HOST_DIAG", "0") == "1" and s_host is not None and input_layout == "BNSD":
        scale_s = torch.as_tensor(scale, dtype=torch.float16)
        v_row = v_dq[0, 0, :s2].float()
        npu_f = npu_out.cpu().to(torch.float32)[0, 0]
        gold_f = cpu_out.to(torch.float32)[0, 0]
        for qs in (57, 114):
            s_col = (s_host[:s2, qs:qs + 1].to(torch.float16) * scale_s)
            p_eff, _ = _blockwise_snap_local_quantize_p_eff(s_col, None, seqk_dim=0)
            p = p_eff[:, 0].float()
            o_snap = p @ v_row
            kmax = int(s_host[:s2, qs].argmax())
            o_hot = v_row[kmax]
            npu_row = npu_f[qs]
            gold_row = gold_f[qs]
            p_sum = float(p.sum())
            topv, topi = torch.topk(p, k=min(4, s2))
            def _cos(a, b):
                return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
            print(f"[HOST-P] qs={qs} p_sum={p_sum:.6f} argmaxK={kmax} topk_idx={topi.tolist()} topk_p={[float(x) for x in topv]}")
            print(f"[HOST-P] qs={qs} cos npu/gold={_cos(npu_row, gold_row):.6f} npu/snap={_cos(npu_row, o_snap):.6f} "
                  f"npu/onehot={_cos(npu_row, o_hot):.6f} gold/snap={_cos(gold_row, o_snap):.6f}")
            print(f"[HOST-P] qs={qs} maxabs npu-gold={float((npu_row-gold_row).abs().max()):.6f} "
                  f"npu-snap={float((npu_row-o_snap).abs().max()):.6f} "
                  f"npu-onehot={float((npu_row-o_hot).abs().max()):.6f} "
                  f"npu_d0={float(npu_row[0]):.6f} snap_d0={float(o_snap[0]):.6f} gold_d0={float(gold_row[0]):.6f} hot_d0={float(o_hot[0]):.6f}")
    return compare(cpu_out, npu_out, src_dtype)


class TestBSAMxfp8Golden(TestCase):
    def _run(self, **cfg):
        cfg_merged = {**cfg, **_CLI_OVERRIDES}
        print()
        pct, res = run_bsa_mxfp8_eager(**cfg_merged)
        cpu_mode = _FORCE_CPU or not _HAS_BSA_V3
        if res == "SKIP":
            if cpu_mode:
                self.skipTest("CPU-only 模式: 无 NPU 算子可比对, SKIP")
            self.skipTest(f"block_sparse_attention_v3 不可用: {_BSA_V3_IMPORT_ERROR}")
        assert res == "Pass", (
            f"BSA mxfp8 case 失败: {cfg_merged}\n  fulfill={pct}% result={res}\n"
            f"  要求 fulfill=100% (rtol=1/128 atol=1e-4, maxRE<10)")
        print(f"[PASS][NPU vs golden] {cfg_merged.get('mask_kind', 'dense')} "
              f"layout={cfg_merged.get('input_layout', 'BNSD')} "
              f"b={cfg_merged['b']} n1={cfg_merged['n1']} n2={cfg_merged['n2']} "
              f"s1={cfg_merged.get('s1') if cfg_merged.get('s1') is not None else cfg_merged.get('s1_list')} "
              f"s2={cfg_merged.get('s2') if cfg_merged.get('s2') is not None else cfg_merged.get('s2_list')} "
              f"qk_d={cfg_merged.get('qk_d', 128)} "
              f"out={cfg_merged.get('out_dtype', 'bfloat16')} block={cfg_merged['block_shape']} "
              f"fulfill={pct:.4f}%")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_qs128_kvs128(self, device="npu"):
        """0817: qs=128 kvs=128 单基块。"""
        self._run(b=1, n1=1, n2=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_qs128_kvs64(self, device="npu"):
        """0817: kvs=64 半面(一个 128 块内不满)。"""
        self._run(b=1, n1=1, n2=1, s1=128, s2=64, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_qs64_kvs128(self, device="npu"):
        """0817: qs=64。"""
        self._run(b=1, n1=1, n2=1, s1=64, s2=128, qk_d=128,
                  block_shape=[64, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_qs128_kvs160_tail(self, device="npu"):
        """0817: s2=160 = 128+32 尾块。"""
        self._run(b=1, n1=1, n2=1, s1=128, s2=160, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_qs128_kvs256(self, device="npu"):
        """0817: 两基块(2x128) + drain。"""
        self._run(b=1, n1=1, n2=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_qs128_kvs192(self, device="npu"):
        """0817: GQA + kvs=192(128+64)。"""
        self._run(b=1, n1=4, n2=1, s1=128, s2=192, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_sparse_qs128_kvs256(self, device="npu"):
        """0817: sparse gather。"""
        self._run(b=1, n1=2, n2=2, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind="sparse")

    # --- 跨 TILE_GROUP(G=8, s2>1024)：CopyPScale L1 按 loop%12 映射 ---
    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_qs64_kvs1152(self, device="npu"):
        """qs64 CopyPScale 半行路径 + 组 2 仅 1 tile。"""
        self._run(b=1, n1=1, n2=1, s1=64, s2=1152, qk_d=128,
                  block_shape=[64, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_qs64_kvs2048(self, device="npu"):
        """qs64 + 两整组；L1 stride-2 在槽 8 起写需回绕。"""
        self._run(b=1, n1=1, n2=1, s1=64, s2=2048, qk_d=128,
                  block_shape=[64, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_qs128_kvs1056_tail(self, device="npu"):
        """组 2 首块即 kvs32 尾块 (1024+32)。"""
        self._run(b=1, n1=1, n2=1, s1=128, s2=1056, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_qs128_kvs1280(self, device="npu"):
        """组 2 两整 tile (10 基块)。"""
        self._run(b=1, n1=1, n2=1, s1=128, s2=1280, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_qs128_kvs1152(self, device="npu"):
        """GQA 跨组；多 head 任务可能落在奇数 loop (kvsFirst=1)。"""
        self._run(b=1, n1=4, n2=1, s1=128, s2=1152, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_sparse_qs128_kvs1152(self, device="npu"):
        """sparse gather 跨组。"""
        self._run(b=1, n1=2, n2=2, s1=128, s2=1152, qk_d=128,
                  block_shape=[128, 128], mask_kind="sparse")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_qs128_kvs3072(self, device="npu"):
        """3 组 / 24 基块：groupMax UB 环(12)与 L1 P-scale 环同时回绕。"""
        self._run(b=1, n1=1, n2=1, s1=128, s2=3072, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b2_qs128_kvs1152(self, device="npu"):
        """batch=2 多 task，覆盖非 0 起始 loop。"""
        self._run(b=2, n1=1, n2=1, s1=128, s2=1152, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bsnd_mha_dense_qs128_kvs128(self, device="npu"):
        """layout smoke: BSND n1=2, golden 仍 BNSD, 仅 NPU I/O permute。"""
        self._run(b=1, n1=2, n2=2, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense", input_layout="BSND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_mha_dense_qs128_kvs128(self, device="npu"):
        """layout smoke: uniform TND n1=2。"""
        self._run(b=1, n1=2, n2=2, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense", input_layout="TND")

    # ==================== 从 mxfp4 BNSD OCP 移植 ====================
    # runner 已支持 uniform BSND/TND、TND varlen、act_s pad、gap_sparse、scale-stride-trap。
    # 仍未移植: CX / dst_type_max / blockShapeY=256。

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs128_d128(self, device="npu"):
        """[from mxfp4] ① BNSD dense 单块: 基本精度(block_shape=[128,128], S=128 -> 1x1)。"""
        self._run(b=1, n1=1, n2=1, g=1, s1=128, s2=128,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b2_qn4_kvn4_g1_qs256_kvs4096_d128(self, device="npu"):
        """[from mxfp4] MHA dense s2=4096 长 KV, b=2 验证 batch + 长 KV dense。"""
        self._run(b=2, n1=4, n2=4, g=1, s1=256, s2=4096,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs1_kvs31_d128(self, device="npu"):
        """[from mxfp4] MHA sparse 边界: s1=1 decode + s2=31(32*1-1), Q pad + K 单块不满。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=31,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_sparse_b1_qn8_kvn8_g1_qs127_kvs159_d128(self, device="npu"):
        """[from mxfp4] MHA sparse 边界: s1=127 + s2=159(32*5-1) 双尾块, K 尾 31 行非对齐。"""
        self._run(b=1, n1=8, n2=8, g=1, s1=127, s2=159,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b1_qn8_kvn2_g4_qs256_kvs287_d128(self, device="npu"):
        """[from mxfp4] GQA dense g4, s2=287(32*9-1) 尾块不满, 验证 K 尾块处理。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=287,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b2_qn4_kvn1_g4_qs128_kvs256_d128(self, device="npu"):
        """[from mxfp4] GQA dense g4 b=2, 小 KV 验证 batch + 单 kv head dense。"""
        self._run(b=2, n1=4, n2=1, g=4, s1=128, s2=256,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b4_qn8_kvn2_g4_qs384_kvs768_d128(self, device="npu"):
        """[from mxfp4] GQA dense g4 b=4, s1=384/s2=768 非 128 对齐多 q-block dense。"""
        self._run(b=4, n1=8, n2=2, g=4, s1=384, s2=768,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b8_qn6_kvn2_g3_qs256_kvs512_d128(self, device="npu"):
        """[from mxfp4] GQA dense g3 b=8, 大 batch + g3 头分组(非 2/4 幂) dense。"""
        self._run(b=8, n1=6, n2=2, g=3, s1=256, s2=512,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b1_qn16_kvn4_g4_qs257_kvs8448_d128(self, device="npu"):
        """[from mxfp4] GQA dense g4 唯一 n1=16, s2=8448 超长 KV, 验证大 head + 超长 dense。"""
        self._run(b=1, n1=16, n2=4, g=4, s1=257, s2=8448,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b1_qn8_kvn4_g2_qs64_kvs479_d128(self, device="npu"):
        """[from mxfp4] GQA dense g2, s2=479(32*15-1) 较大尾块, 验证 g2 + K 尾 31 行。"""
        self._run(b=1, n1=8, n2=4, g=2, s1=64, s2=479,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b1_qn8_kvn2_g4_qs512_kvs1024_d128(self, device="npu"):
        """[from mxfp4] GQA dense g4, s1=512/s2=1024 多 block dense。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=512, s2=1024,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b1_qn8_kvn1_g8_qs256_kvs511_d128(self, device="npu"):
        """[from mxfp4] GQA dense g8 单 kv head, s2=511 尾块(512-1), 验证 g8 + K 尾 1 行。"""
        self._run(b=1, n1=8, n2=1, g=8, s1=256, s2=511,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b1_qn8_kvn2_g4_qs1024_kvs4096_d128(self, device="npu"):
        """[from mxfp4] GQA dense g4 + s1=1024 + s2=4096 全大维度 dense。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=1024, s2=4096,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b1_qn8_kvn2_g4_qs63_kvs95_d128(self, device="npu"):
        """[from mxfp4] GQA sparse 双边界: s1=63(64-1) + s2=95(32*3-1), Q/K 双尾块 sparse。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=63, s2=95,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b1_qn6_kvn2_g3_qs127_kvs255_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g3, s1=127/s2=255 边界非对齐 sparse 尾块。"""
        self._run(b=1, n1=6, n2=2, g=3, s1=127, s2=255,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b2_qn4_kvn1_g4_qs128_kvs256_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g4 b=2, 小 KV 验证 batch + 单 kv head sparse gather。"""
        self._run(b=2, n1=4, n2=1, g=4, s1=128, s2=256,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b3_qn8_kvn2_g4_qs512_kvs1024_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g4 b=3, s1=512/s2=1024 多 block 大序列 sparse。"""
        self._run(b=3, n1=8, n2=2, g=4, s1=512, s2=1024,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b1_qn8_kvn2_g4_qs1_kvs8448_d128(self, device="npu"):
        """[from mxfp4] GQA sparse s1=1 decode + s2=8448 超长 KV, Q pad + 跨 tile rescale。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=1, s2=8448,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b4_qn8_kvn2_g4_qs384_kvs768_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g4 b=4, s1=384/s2=768 非 128 对齐 sparse 多 block。"""
        self._run(b=4, n1=8, n2=2, g=4, s1=384, s2=768,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b8_qn6_kvn2_g3_qs256_kvs512_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g3 b=8, 大 batch + g3 头分组 sparse gather。"""
        self._run(b=8, n1=6, n2=2, g=3, s1=256, s2=512,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b1_qn8_kvn2_g4_qs512_kvs8448_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g4, s2=8448 超长 KV sparse, 验证 n1=8 + 超长 sparse gather。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=512, s2=8448,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b1_qn8_kvn4_g2_qs64_kvs223_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g2, s2=223(32*7-1) 较大尾块, 验证 g2 + K 尾 31 行 sparse。"""
        self._run(b=1, n1=8, n2=4, g=2, s1=64, s2=223,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b1_qn8_kvn1_g8_qs256_kvs799_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g8 单 kv head, s2=799(32*25-1) 较大尾块 sparse。"""
        self._run(b=1, n1=8, n2=1, g=8, s1=256, s2=799,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b5_qn4_kvn1_g4_qs128_kvs256_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g4 b=5, 单 kv head 验证 b=5 + g4 sparse。"""
        self._run(b=5, n1=4, n2=1, g=4, s1=128, s2=256,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b1_qn8_kvn2_g4_qs1024_kvs4096_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g4 + s1=1024 + s2=4096 全大维度 sparse。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=1024, s2=4096,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b1_qn4_kvn2_g2_qs64_kvs128_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g2 极小序列, s1=64 单 block sparse 验证 g2。"""
        self._run(b=1, n1=4, n2=2, g=2, s1=64, s2=128,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b1_qn8_kvn4_g2_qs500_kvs4096_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g2, s2=4096 验证 g2 头分组 + 长 KV sparse。"""
        self._run(b=1, n1=8, n2=4, g=2, s1=500, s2=4096,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b1_qn6_kvn3_g2_qs768_kvs1024_d128(self, device="npu"):
        """[from mxfp4] GQA sparse g2, s1=768(6 个 q-block) + s2=1024(8 个 kv tile) 多 block sparse。"""
        self._run(b=1, n1=6, n2=3, g=2, s1=768, s2=1024,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_sparse_b2_qn1_kvn1_g1_qs25600_kvs33024_d128(self, device="npu"):
        """[from mxfp4] ① BNSD 超大序列 sparse: s1=25600/s2=33024, b=2 验证超长 sparse gather + online rescale。"""
        self._run(b=2, n1=1, n2=1, g=1, s1=25600, s2=33024,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b56_qn8_kvn8_g1_qs512_kvs512_d128(self, device="npu"):
        """[from mxfp4] ① BNSD 大 batch dense: b=56 qn8/kvn8, 验证大 batch 泛化(唯一 b>=56 用例)。"""
        self._run(b=56, n1=8, n2=8, g=1, s1=512, s2=512,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs128_d64(self, device="npu"):
        """[from mxfp4] D64 BNSD dense 最小基线: n=1 单头单块(与 ① 的 d128 基线镜像),"""
        self._run(b=1, n1=1, n2=1, g=1, s1=128, s2=128,
                  qk_d=64, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs1_kvs31_d64(self, device="npu"):
        """[from mxfp4] D64 BNSD sparse decode 极小边界: s1=1 + s2=31(单 kv 块不满, V-scale 单 32-block),"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=31,
                  qk_d=64, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b2_qn8_kvn2_g4_qs256_kvs512_d64(self, device="npu"):
        """[from mxfp4] D64 BNSD GQA dense 常规: b=2 + qn8/kvn2/g4, 头映射与 D=64 行宽叠加。"""
        self._run(b=2, n1=8, n2=2, g=4, s1=256, s2=512,
                  qk_d=64, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_sparse_b1_qn8_kvn2_g4_qs127_kvs255_d64(self, device="npu"):
        """[from mxfp4] D64 BNSD GQA sparse 非对齐边界: s1=127/s2=255 双尾块不满 + 稀疏 gather。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=127, s2=255,
                  qk_d=64, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b1_qn2_kvn2_g1_qs256_kvs4096_d64(self, device="npu"):
        """[from mxfp4] D64 BNSD dense 长 KV: s2=4096(32 kv 块), 跨 chunk rescale × D=64 窄行搬运。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=4096,
                  qk_d=64, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b1_qn2_kvn2_g1_qs256_kvs4097_d64(self, device="npu"):
        """D64 BNSD dense 长 KV 尾块: s2=4097 = 32×128 + 1，跨组 rescale + D=64 pad。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=4097,
                  qk_d=64, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b8_qn4_kvn4_g1_qs128_kvs256_d64(self, device="npu"):
        """[from mxfp4] D64 BNSD dense 大 batch: b=8, batch 基址(N*S*D 缩半)与 task 切分叠加。"""
        self._run(b=8, n1=4, n2=4, g=1, s1=128, s2=256,
                  qk_d=64, block_shape=[128, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b1_qn2_kvn2_g1_qs127_kvs256_d128_blk64x128(self, device="npu"):
        """[from mxfp4] 补 BNSD block=[64,128]: 此前 block 变体仅 BSND——补 BNSD 下小 Q-block(64)"""
        self._run(b=1, n1=2, n2=2, g=1, s1=127, s2=256,
                  qk_d=128, block_shape=[64, 128], mask_kind='dense')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs256_kvs4096_d64(self, device="npu"):
        """[from mxfp4] D64 BNSD sparse 长 KV: s2=4096 稀疏跳块 × D=64 窄行搬运(d64 长 KV 此前仅 dense)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=4096,
                  qk_d=64, block_shape=[128, 128], mask_kind='sparse')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128_fp16out(self, device="npu"):
        """[from mxfp4] fp16 out: BNSD MHA dense 基线, out_dtype='float16'(NPU attention_out_dtype=float16)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense', out_dtype='float16')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs1024_d128_fp16out(self, device="npu"):
        """[from mxfp4] fp16 out: BNSD MHA sparse(8 kv tile 稀疏 gather), out_dtype='float16'。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse', out_dtype='float16')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b1_qn8_kvn2_g4_qs256_kvs1024_d128_fp16out(self, device="npu"):
        """[from mxfp4] fp16 out: BNSD GQA g4 dense 多 block, out_dtype='float16'(GQA 头映射 × fp16 cast)。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=1024,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense', out_dtype='float16')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs4097_d128_fp16out(self, device="npu"):
        """[from mxfp4] fp16 out + 长 KV: BNSD MHA dense s2=4097(32 kv 块 + 尾块 1 行), 跨 chunk rescale × fp16 输出。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=4097,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense', out_dtype='float16')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs8192_d128_fp16out(self, device="npu"):
        """[from mxfp4] fp16 out + 长 KV: BNSD MHA dense s2=8192(64 kv 块), online rescale 累积极限 × fp16 输出。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=8192,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense', out_dtype='float16')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_gqa_dense_b1_qn8_kvn2_g4_qs256_kvs8448_d128_fp16out(self, device="npu"):
        """[from mxfp4] fp16 out + 长 KV: BNSD GQA g4 dense s2=8448(66 kv 块)超长, GQA stride × 长 KV rescale × fp16。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=8448,
                  qk_d=128, block_shape=[128, 128], mask_kind='dense', out_dtype='float16')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_sparse_b1_qn2_kvn2_g1_qs256_kvs8192_d128_fp16out(self, device="npu"):
        """[from mxfp4] fp16 out + 长 KV sparse: BNSD MHA sparse s2=8192(64 kv 块随机 gather), 稀疏跳块 × 长 KV rescale × fp16。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=8192,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse', out_dtype='float16')

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_sparse_b2_qn1_kvn1_g1_qs25600_kvs33024_d128_fp16out(self, device="npu"):
        """[from mxfp4] ① BNSD 超大序列 sparse: s1=25600/s2=33024, b=2 验证超长 sparse gather + online rescale。"""
        self._run(b=2, n1=1, n2=1, g=1, s1=25600, s2=33024,
                  qk_d=128, block_shape=[128, 128], mask_kind='sparse', out_dtype='float16')

    # ==================== 对标 MXFP4 OCP pytest 1-3（代表性子集）====================
    # 1: TND varlen + act_s pad
    # 2: BSND/TND × D=64 / sparse / 非对齐
    # 3: gap_sparse + ocp_scale_stride_trap
    # 合入前必跑: MXFP8_OCP_REG_K / ./run.sh --reg
    # 不拷 mxfp4 全量 ~80 varlen / CX / blockY=256。

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_256_s2_256_128_d128(self, device="npu"):
        """[1] TND varlen dense: per-batch concat golden + actualSeq。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128, block_shape=[128, 128],
                  mask_kind="dense", input_layout="TND",
                  s1_list=[128, 256], s2_list=[256, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_varlen_gqa_dense_b2_qn8_kvn2_g4_s1_128_256_s2_512_256_d128(self, device="npu"):
        """[1] TND varlen GQA dense。"""
        self._run(b=2, n1=8, n2=2, g=4, qk_d=128, block_shape=[128, 128],
                  mask_kind="dense", input_layout="TND",
                  s1_list=[128, 256], s2_list=[512, 256])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_varlen_mha_sparse_b2_qn2_kvn2_g1_s1_128_256_s2_256_128_d128(self, device="npu"):
        """[1] TND varlen sparse。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128, block_shape=[128, 128],
                  mask_kind="sparse", input_layout="TND",
                  s1_list=[128, 256], s2_list=[256, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_127_129_s2_256_256_d128(self, device="npu"):
        """[1] TND varlen 非对齐 qs。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128, block_shape=[128, 128],
                  mask_kind="dense", input_layout="TND",
                  s1_list=[127, 129], s2_list=[256, 256])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_varlen_mha_dense_b2_qn2_kvn2_g1_s1_128_256_s2_256_128_d64(self, device="npu"):
        """[1] TND varlen dense D=64。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=64, block_shape=[128, 128],
                  mask_kind="dense", input_layout="TND",
                  s1_list=[128, 256], s2_list=[256, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b1_qn2_kvn2_g1_actq200_padq256_actkv200_padkv256_d128(self, device="npu"):
        """[1] BNSD act_s pad: 只比 [0:act_q)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense",
                  act_s1=200, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bsnd_mha_dense_b1_qn2_kvn2_g1_actq200_padq256_actkv200_padkv256_d128(self, device="npu"):
        """[1] BSND act_s pad。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense", input_layout="BSND",
                  act_s1=200, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b1_qn2_kvn2_g1_actq1_padq128_actkv128_padkv128_d128(self, device="npu"):
        """[1] BNSD act_s decode-ish: act_s1=1 pad=128。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense",
                  act_s1=1, act_s2=128)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b2_qn2_kvn2_g1_actq200_128_padq256_actkv200_128_padkv256_d128(self, device="npu"):
        """[1] BNSD act_s per-batch 不等长。"""
        self._run(b=2, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense",
                  act_s1=[200, 128], act_s2=[200, 128])

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_sparse_b1_qn2_kvn2_g1_actq200_padq256_actkv200_padkv256_d128(self, device="npu"):
        """[1] BNSD act_s sparse。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind="sparse",
                  act_s1=200, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_dense_b1_qn2_kvn2_g1_actq200_padq256_actkv200_padkv256_d64(self, device="npu"):
        """[1] BNSD act_s pad D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=256, s2=256, qk_d=64,
                  block_shape=[128, 128], mask_kind="dense",
                  act_s1=200, act_s2=200)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128(self, device="npu"):
        """[2] BSND dense 两基块(超出 qs128 冒烟)。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense", input_layout="BSND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bsnd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs1024_d128(self, device="npu"):
        """[2] BSND sparse。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind="sparse", input_layout="BSND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bsnd_mha_dense_b1_qn2_kvn2_g1_qs127_kvs256_d128(self, device="npu"):
        """[2] BSND 非对齐 qs。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=127, s2=256, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense", input_layout="BSND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d64(self, device="npu"):
        """[2] BSND D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=64,
                  block_shape=[128, 128], mask_kind="dense", input_layout="BSND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bsnd_mha_dense_b1_qn2_kvn2_g1_qs129_kvs400_d64(self, device="npu"):
        """[2] BSND D=64 非对齐。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=129, s2=400, qk_d=64,
                  block_shape=[128, 128], mask_kind="dense", input_layout="BSND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bsnd_mha_sparse_b1_qn2_kvn2_g1_qs1_kvs1024_d64(self, device="npu"):
        """[2] BSND D=64 sparse decode。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=1, s2=1024, qk_d=64,
                  block_shape=[128, 128], mask_kind="sparse", input_layout="BSND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs400_d128(self, device="npu"):
        """[2] TND uniform 非 128 对齐 KV。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=400, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense", input_layout="TND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs1024_d128(self, device="npu"):
        """[2] TND uniform sparse。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=128,
                  block_shape=[128, 128], mask_kind="sparse", input_layout="TND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs128_d64(self, device="npu"):
        """[2] TND uniform D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=64,
                  block_shape=[128, 128], mask_kind="dense", input_layout="TND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_mha_sparse_b1_qn2_kvn2_g1_qs128_kvs1024_d64(self, device="npu"):
        """[2] TND uniform D=64 sparse。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=1024, qk_d=64,
                  block_shape=[128, 128], mask_kind="sparse", input_layout="TND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_gqa_dense_b1_qn8_kvn2_g4_qs256_kvs512_d64(self, device="npu"):
        """[2] TND GQA D=64。"""
        self._run(b=1, n1=8, n2=2, g=4, s1=256, s2=512, qk_d=64,
                  block_shape=[128, 128], mask_kind="dense", input_layout="TND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bsnd_mha_dense_b1_qn2_kvn2_g1_qs128_kvs256_d128_blk64x128(self, device="npu"):
        """[2] BSND block=[64,128]。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=256, qk_d=128,
                  block_shape=[64, 128], mask_kind="dense", input_layout="BSND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bnsd_mha_gap_sparse_b1_qn2_kvn2_g1_qs128_kvs512_d128(self, device="npu"):
        """[3] BNSD gap_sparse: 仅 attend kv-block 0+末。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind="gap_sparse")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bsnd_mha_gap_sparse_b1_qn2_kvn2_g1_qs128_kvs512_d128(self, device="npu"):
        """[3] BSND gap_sparse。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind="gap_sparse", input_layout="BSND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_tnd_mha_gap_sparse_b1_qn2_kvn2_g1_qs128_kvs512_d128(self, device="npu"):
        """[3] TND gap_sparse。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=128,
                  block_shape=[128, 128], mask_kind="gap_sparse", input_layout="TND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_bsnd_mha_gap_sparse_b1_qn2_kvn2_g1_qs128_kvs512_d64(self, device="npu"):
        """[3] BSND gap_sparse D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=512, qk_d=64,
                  block_shape=[128, 128], mask_kind="gap_sparse", input_layout="BSND")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_ocp_bnsd_mha_dense_scale_stride_trap_qs128_kvs128_d128(self, device="npu"):
        """[3-diag] BNSD trap 对照: 无 N 行步长, 区分造数 vs layout。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense", input_layout="BNSD",
                  data_kind="ocp_scale_stride_trap")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_ocp_tnd_mha_dense_scale_stride_trap_qs128_kvs128_d128(self, device="npu"):
        """[3] TND scale-stride-trap: 跨头 e8m0 118/119。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense", input_layout="TND",
                  data_kind="ocp_scale_stride_trap")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_ocp_bsnd_mha_dense_scale_stride_trap_qs128_kvs128_d128(self, device="npu"):
        """[3] BSND scale-stride-trap。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense", input_layout="BSND",
                  data_kind="ocp_scale_stride_trap")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_ocp_tnd_mha_dense_scale_stride_trap_qs127_kvs255_d128(self, device="npu"):
        """[3] TND trap 非对齐。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=127, s2=255, qk_d=128,
                  block_shape=[128, 128], mask_kind="dense", input_layout="TND",
                  data_kind="ocp_scale_stride_trap")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_ocp_tnd_mha_dense_scale_stride_trap_qs128_kvs128_d64(self, device="npu"):
        """[3] TND trap D=64。"""
        self._run(b=1, n1=2, n2=2, g=1, s1=128, s2=128, qk_d=64,
                  block_shape=[128, 128], mask_kind="dense", input_layout="TND",
                  data_kind="ocp_scale_stride_trap")

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_bsa_mxfp8_ocp_tnd_varlen_mha_dense_scale_stride_trap_s1_128_256_s2_256_128_d128(self, device="npu"):
        """[3] TND varlen + trap: concat 与跨头 scale 叠加。"""
        self._run(b=2, n1=2, n2=2, g=1, qk_d=128, block_shape=[128, 128],
                  mask_kind="dense", input_layout="TND",
                  s1_list=[128, 256], s2_list=[256, 128],
                  data_kind="ocp_scale_stride_trap")

if __name__ == "__main__":
    run_tests()
