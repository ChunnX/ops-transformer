#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
"""npu_block_sparse_attention mxfp8 (e4m3fn + e8m0) 正向单测 + CPU golden.

0817 几何: kvBaseTile=128, QK 半面 64x128, 基块 128x128, G=8, Dpv=12。
Q/K/V/P = e4m3fn, scale = e8m0, MX 组 32, 仅 OCP (quant_mode=4)。
attentionOut 形状不翻倍(与 mxfp4 packed uint8 不同)。

golden: Q/K/V mxfp8 qdq(fp32); 分块 flash + online softmax;
P 走 kernel 同构 snap_local(m_snap=(K-2)*ln2) 再 round 到 e4m3。
判据: qfa check_result (np.isclose rtol=1/128 atol=1e-4, 满足率>=99.5% 且 maxRE<10)。
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
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion
except Exception:
    torch_npu = None

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


def rand_range(shape, data_range=(-10, 10), dtype=torch.float32, device=None):
    return data_range[0] + (data_range[1] - data_range[0]) * torch.rand(
        shape, dtype=dtype, device=device)


def randn_x(shape, x=1.0, dtype=torch.float32, device=None):
    return rand_range(shape, (-x, x), dtype=dtype, device=device)


def _make_qkv(b, n1, n2, s1, s2, qk_d, v_d, data_x=10.0):
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


def _blockwise_snap_local_quantize_p_eff(S, m_running, seqk_dim=0, mx_block_size=32):
    """与 mxfp4 OCP 同构: m_snap=(K_block-2)*ln2, corr=2^(K_diff-2)。
    P_local 经 bf16 round-trip 后 round 到 e4m3(对齐 kernel Cast fp16->bf16->e4m3)。"""
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
    raise ValueError(f"unsupported mask kind: {kind}")


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


def check_result(expect, result, data_type, pct_thd=0.005):
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
    if PRINTF_CHECK_RES:
        return check_result_printf_res(golden_fp32, real_fp32, src_dtype)
    return check_result(golden_fp32, real_fp32, src_dtype)


def run_bsa_mxfp8_eager(b, n1, n2, s1, s2, g=1, qk_d=128, v_d=None,
                        input_layout="BNSD", out_dtype="bfloat16",
                        block_shape=(128, 128), mask_kind="dense",
                        inner_precise=4, quant_mode=QUANT_MODE_MXFP8_OCP,
                        p_quant="snap_local", s_dtype="fp16",
                        dump=False, device_id=0, data_x=10.0):
    g = n1 // n2
    assert input_layout == "BNSD", f"首批仅 BNSD, 当前 {input_layout}"
    assert n1 == n2 * g
    assert block_shape[0] in (64, 128)
    assert block_shape[1] % 128 == 0, f"0817 blockShapeY 须为 128 倍数, 当前 {block_shape[1]}"
    assert qk_d in (64, 128)
    assert quant_mode == QUANT_MODE_MXFP8_OCP
    if v_d is None:
        v_d = qk_d
    scale = 1.0 / math.sqrt(qk_d)
    src_dtype = out_dtype
    query, key, value = _make_qkv(b, n1, n2, s1, s2, qk_d, v_d, data_x=data_x)
    block_x, block_y = int(block_shape[0]), int(block_shape[1])
    ceil_q = (s1 + block_x - 1) // block_x
    ceil_kv = (s2 + block_y - 1) // block_y
    mask = _build_block_sparse_mask(b, n1, ceil_q, ceil_kv, mask_kind)
    print("MXFP8 OCP QUANT")
    print("cpu calc start")
    cpu_out = cpu_block_sparse_attention_bnsd_mxfp8(
        query, key, value, mask, block_shape, scale, n2,
        p_quant=p_quant, s_dtype=s_dtype, out_dtype=out_dtype)
    print("golden calc end")
    if _FORCE_CPU or not _HAS_BSA_V3:
        print(f"cpu_out shape={tuple(cpu_out.shape)}")
        return -1.0, "SKIP"
    q_fp8, q_e8m0, _ = mxfp8_quantize_native(query, quant_axis=-1)
    k_fp8, k_e8m0, _ = mxfp8_quantize_native(key, quant_axis=-1)
    v_fp8, v_e8m0, _ = mxfp8_quantize_native(value, quant_axis=-2)
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert q_fp8.shape[-1] == qk_d, "mxfp8 输出 last dim 不翻倍"
    q_scale = _e8m0_qk_to_v3(q_e8m0)
    k_scale = _e8m0_qk_to_v3(k_e8m0)
    v_scale = _e8m0_v_to_v3(v_e8m0)
    torch_npu.npu.set_device(device_id)
    print("npu calc start")
    npu_out, _ = block_sparse_attention_v3(
        q_fp8.contiguous().npu(), k_fp8.contiguous().npu(),
        v_fp8.contiguous().npu(), mask.contiguous().npu(), list(block_shape),
        [s1] * b, [s2] * b,
        num_heads=n1, num_key_value_heads=n2,
        q_input_layout=input_layout, kv_input_layout=input_layout,
        mask_type=0, scale_value=scale, inner_precise=inner_precise, block_size=0,
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=False, quant_mode=quant_mode, dst_type_max=0.0,
        q_dequant_scale=q_scale.npu(), k_dequant_scale=k_scale.npu(), v_dequant_scale=v_scale.npu(),
        attention_out_dtype=_resolve_out_dtype(out_dtype),
    )
    torch_npu.npu.synchronize()
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
            f"  若同挂可先 --p_quant none 冒烟, 再查 e8m0 配对/P e4m3 snap")
        print(f"[PASS][NPU vs golden] {cfg_merged.get('mask_kind', 'dense')} "
              f"b={cfg_merged['b']} n1={cfg_merged['n1']} n2={cfg_merged['n2']} "
              f"s1={cfg_merged['s1']} s2={cfg_merged['s2']} block={cfg_merged['block_shape']} "
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


if __name__ == "__main__":
    run_tests()
